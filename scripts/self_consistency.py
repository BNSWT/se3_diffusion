import os
import sys
import time
import numpy as np
import hydra
import torch
import subprocess
import logging
import traceback
import argparse
from functools import partial
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional
from contextlib import nullcontext
import multiprocessing as mp
try:
    from ProteinMPNN import protein_mpnn_pyrosetta
except:
    logging.info("Pyrosetta is not installed in current environment, rosetta optimization is not available")
from analysis import utils as au
from analysis import metrics
from data import utils as du
from experiments import utils as eu
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import esm

def run_mpnn(
    mpnn_model,
    mpnn_conf,
    tmp_dir: str,
    reference_pdb_path: str,
    motif_mask: Optional[np.ndarray]=None):
    # Run PorteinMPNN
    prefix = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    if "pyrosetta" in sys.modules and mpnn_conf.pyrosetta:
        # get a dict contains mpnn_sequence,mpnn_score,mpnn_pose and their traj
        mpnn_results = protein_mpnn_pyrosetta.mpnn_design(mpnn_model,reference_pdb_path,mpnn_conf)
    else:
        output_path = os.path.join(tmp_dir, prefix+"_parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{mpnn_conf.pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={reference_pdb_path}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{mpnn_conf.pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            tmp_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(mpnn_conf.num_seqs),
            '--sampling_temp',
            str(mpnn_conf.temperature),
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        if mpnn_conf.gpu_id is not None:
            pmpnn_args.append('--device')
            pmpnn_args.append(str(mpnn_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                logging.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            tmp_dir,
            "seqs",
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        mpnn_results = []
        for i, (header, string) in enumerate(fasta_seqs.items()):
            # ignore original sequence
            if i==0:continue
            score = float(header.split("score=")[1].split(",")[0])
            mpnn_results.append({"index":i+1,"sequence" : string,"score":score,"header":header})
    return mpnn_results

class Sampler:

    def __init__(
            self,
            conf: DictConfig,
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)
        self._folding_model.share_memory()

    def run(self,pdb_list,output_dir):

        timestap = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss") if self._infer_conf.timestap else ""
        tmp_dir = os.path.join(output_dir,timestap, 'backup')
        final_results = []
        process_list = []
        if "pyrosetta" in sys.modules:
            self.mpnn_model = protein_mpnn_pyrosetta.model_init(device=self.device,config=self._infer_conf.mpnn)
        with (mp.Pool(processes=mp.cpu_count() if mp.cpu_count() < self._infer_conf.mpnn.cpus else self._infer_conf.mpnn.cpus) if not self._infer_conf.single_process else nullcontext() ) as pool:
            os.makedirs(tmp_dir,exist_ok=True)
            for pdb_path in pdb_list:
                self._log.info(f'run self consistency of file  {pdb_path}')     
                    # Run ProteinMPNN
                if self._infer_conf.single_process:
                    mpnn_result = run_mpnn(self.mpnn_model,self._infer_conf.mpnn,tmp_dir,pdb_path)
                    esmfold_results = self.run_esmfold(mpnn_result,decoy_pdb_dir=output_dir,tmp_dir=tmp_dir,reference_pdb_path= pdb_path)
                    self._log.info(esmfold_results)
                    final_results.append(esmfold_results)
                else:
                    def run_esm_fold_callback(result, output_dir: str, reference_pdb_path: str, motif_mask: np.ndarray=None):
                        # this is so important!! without this block subprocees will die silently without any output
                        try:
                            result = self.run_esmfold(result,decoy_pdb_dir = output_dir,tmp_dir=tmp_dir,reference_pdb_path= reference_pdb_path)
                            self._log.info(result)
                            final_results.append(result)
                        except:
                            raise Exception(traceback.format_exc())
                        def error_handler(error):
                            print(error)
                            sys.stdout.flush()                            
                        run_esm_fold_callback_partial = partial(run_esm_fold_callback,output_dir=output_dir,reference_pdb_path=pdb_path)
                        esmfold_result = pool.apply_async(run_mpnn,args=(self.mpnn_model,self._infer_conf.mpnn,tmp_dir,pdb_path),callback=run_esm_fold_callback_partial,error_callback=error_handler)
                        process_list.append(esmfold_result)
            if not self._infer_conf.single_process:
                for process in process_list:
                    process.wait()
            csv_path = os.path.join(output_dir,"self_consistency.csv")
            final_results = pd.concat(final_results).reset_index(drop=True)
            if os.path.exists(csv_path):
                try:
                    existed_result = pd.read_csv(csv_path)
                    final_results = pd.concat([existed_result,final_results])
                except:
                    csv_path = os.path.join(output_dir,"self_consistency.csv")
                    self._log.warning(f"{csv_path} has exists,and merge operation failed,override to original csv ")
            final_results.to_csv(os.path.join(output_dir,"self_consistency.csv"))
            fasta_path = os.path.join(output_dir, 'mpnn.fasta')
            with open(fasta_path,'w') as f:
                for i in range(len(final_results["header"])):
                    f.write(f'>{final_results["header"][i]}\n {final_results["sequence"][i]}\n')
            # clean tmp file
            shutil.rmtree(tmp_dir)
            # clean tmp file

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)

        return output
    
    def run_esmfold(
        self,
        mpnn_results,
        decoy_pdb_dir: str,
        tmp_dir : str,
        reference_pdb_path: str,
        motif_mask: Optional[np.ndarray]=None,
        ):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """
        # if isinstance(mpnn_result,AsyncResult):
        #     mpnn_result = mpnn_result.get()
        self._log.info(f"mpnn stage completed , {len(mpnn_results)} seqs of {reference_pdb_path} send to ESMFOLD")
        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        final_results = {
            'mpnn_score' : [],
            'tm_score': [],
            'sample_path': [],
            'mpnn_sample_path': [],
            'esm_sample_path': [],
            'header': [],
            'sequence': [],
            'rmsd': [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            final_results['motif_rmsd'] = []  
        if self._infer_conf.mpnn.dump and "pose" in mpnn_results[0]:
            final_results['mpnn_rmsd'] = []
            final_results['mpnn_tm_score'] = []
        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        mpnn_dir = os.path.join(decoy_pdb_dir, 'mpnn')
        os.makedirs(esmf_dir, exist_ok=True)
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        prefix = os.path.splitext(os.path.basename(reference_pdb_path))[0]
        mpnn_sample_path = None
        for i, mpnn_result in enumerate(mpnn_results):
            if self._infer_conf.mpnn.dump and "pose" in mpnn_result:
                os.makedirs(mpnn_dir, exist_ok=True)
                mpnn_sample_name = f'{prefix}_mpnn_{i}'
                mpnn_sample_path = os.path.join(mpnn_dir, f'{mpnn_sample_name}.pdb')
                open(mpnn_sample_path,'w').write(mpnn_result["pose"])
            
            string = mpnn_result["sequence"]
            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'{prefix}_esmf_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            print(f"esmfold success : {reference_pdb_path}'s {i+1} sequence")
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with mpnn protein(relaxed)
            if self._infer_conf.mpnn.dump and "pose" in mpnn_result:
                mpnn_sample_path = os.path.join(mpnn_dir, f'{mpnn_sample_name}.pdb')
                mpnn_feats = du.parse_pdb_feats('mpnn', mpnn_sample_path)
                mpnn_rmsd = metrics.calc_aligned_rmsd(
                    mpnn_feats['bb_positions'], esmf_feats['bb_positions'])
                _, mpnn_tm_score = metrics.calc_tm_score(
                    mpnn_feats['bb_positions'], esmf_feats['bb_positions'],
                    sample_seq, sample_seq)
                final_results['mpnn_rmsd'].append(mpnn_rmsd)
                final_results['mpnn_tm_score'].append(mpnn_tm_score)

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['bb_positions'], esmf_feats['bb_positions'])
            if motif_mask is not None:
                sample_motif = sample_feats['bb_positions'][motif_mask]
                of_motif = esmf_feats['bb_positions'][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(
                    sample_motif, of_motif)
                final_results['motif_rmsd'].append(motif_rmsd)
            final_results['header'].append(mpnn_sample_name if 'header' not in mpnn_result else mpnn_result["header"])
            final_results['mpnn_score'].append(mpnn_result['score'])
            final_results['rmsd'].append(rmsd)
            final_results['tm_score'].append(tm_score)
            final_results["sample_path"].append(reference_pdb_path)
            final_results['mpnn_sample_path'].append(mpnn_sample_path)
            final_results['esm_sample_path'].append(esmf_sample_path)
            final_results['sequence'].append(string)


        csv_path = os.path.join(tmp_dir, f'{prefix}.csv')
        fasta_path = os.path.join(tmp_dir, f'{prefix}.fasta')
        with open(fasta_path,'w') as f:
            for i in range(len(final_results["header"])):
                f.write(f'>{final_results["header"][i]}\n {final_results["sequence"][i]}\n')
        final_results = pd.DataFrame(final_results)
        print("esmfold return")
        final_results.to_csv(csv_path)
        return final_results

@hydra.main(version_base=None, config_path=f'{os.path.dirname(__file__)}/../config', config_name="inference")
def run(conf: DictConfig) -> None:
    logging.getLogger(__name__).setLevel(logging.INFO)
    conf.inference.output_dir = os.path.realpath(conf.inference.output_dir)
    cache_path = os.path.realpath(f'{os.path.dirname(__file__)}/../')
    conf = eu.replace_path_with_cwd(conf,path=cache_path)
    print(conf.inference)
    pdb_list = []
    if 'pdb' not in conf and 'pdb_list' not in conf:
        raise ValueError('Must specify either pdb or pdb_list with +pdb= or +pdb_list=')
    elif 'pdb' in conf:
        pdb_list.append(conf.pdb)
    else:
        pdb_list = open(conf.pdb_list).read().strip().splitlines()
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run(pdb_list,conf.inference.output_dir)
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')
if __name__ == '__main__':
    run()