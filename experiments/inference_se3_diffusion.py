"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""

import os
import sys
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import traceback
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
from data import residue_constants
from analysis import plotting
from typing import Dict
from experiments import train_se3_diffusion
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
import esm

logging.basicConfig(level=logging.WARNING)

# TODO add waiting time analysis between cpu and gpu to keep balance
# TODO accomodate mpnn model in main process, it's useful when best_selection is on(large batch size)

def process_chain(design_pdb_feats):
    chain_feats = {
        'aatype': torch.tensor(design_pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(design_pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(design_pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    seq_idx = design_pdb_feats['residue_index'] - np.min(design_pdb_feats['residue_index']) + 1
    chain_feats['seq_idx'] = seq_idx
    chain_feats['res_mask'] = design_pdb_feats['bb_mask']
    chain_feats['residue_index'] = design_pdb_feats['residue_index']
    return chain_feats


def create_pad_feats(pad_amt):
    return {        
        'res_mask': torch.ones(pad_amt),
        'fixed_mask': torch.zeros(pad_amt),
        'rigids_impute': torch.zeros((pad_amt, 4, 4)),
        'torsion_impute': torch.zeros((pad_amt, 7, 2)),
    }

if __name__ == '__main__':
    mp.set_start_method("spawn")
    manager = mp.Manager()
    mpnn_result_dict = manager.dict()
    logging.disable(logging.DEBUG)

def run_mpnn(
    mpnn_model,
    mpnn_conf,
    tmp_dir: str,
    reference_pdb_path: str,
    motif_mask: Optional[np.ndarray]=None):
    # Run PorteinMPNN
    prefix = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    # if "pyrosetta" in sys.modules:
    if False:
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
            conf_overrides: Dict=None
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
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples

        self._rng = np.random.default_rng(self._infer_conf.seed)

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
        self._weights_path = self._infer_conf.weights_path
        output_dir =self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.abspath(os.path.join(output_dir, dt_string))
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        self._pmpnn_dir = self._infer_conf.pmpnn_dir

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)
        self._folding_model.share_memory()

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(
            self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion.Experiment(
            conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {
            k.replace('module.', ''):v for k,v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def init_data(
            self,
            *,
            rigids_impute,
            torsion_impute,
            fixed_mask,
            res_mask,
        ):
        num_res = res_mask.shape[0]
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, num_res+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx * res_mask,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_impute,
            'sc_ca_t': torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)
        return init_feats

    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        all_sample_lengths = range(
            self._sample_conf.min_length,
            self._sample_conf.max_length+1,
            self._sample_conf.length_step
        )
        timestap = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss") if self._infer_conf.timestap else ""
        sample_dir = os.path.join(self._output_dir,timestap, 'diffusion')
        traj_dir = os.path.join(self._output_dir,timestap, 'trajactory')
        movie_dir = os.path.join(self._output_dir,timestap, 'movie')
        tmp_dir = os.path.join(self._output_dir,timestap, 'backup')
        final_results = []
        process_list = []
        if "pyrosetta" in sys.modules:
            self.mpnn_model = protein_mpnn_pyrosetta.model_init(device=self.device,config=self._infer_conf.mpnn)
        with (mp.Pool(processes=mp.cpu_count() if mp.cpu_count() < self._infer_conf.mpnn.cpus else self._infer_conf.mpnn.cpus) if not self._infer_conf.single_process else nullcontext() ) as pool:
            os.makedirs(tmp_dir,exist_ok=True)
            if self._infer_conf.option.save_trajactory or self._infer_conf.option.plot.switch_on:
                os.makedirs(traj_dir,exist_ok=True)
            if self._infer_conf.option.plot.switch_on:
                os.makedirs(movie_dir,exist_ok=True)
            os.makedirs(sample_dir,exist_ok=True)
            for sample_length in all_sample_lengths:
                length_prefix = str(sample_length)
                self._log.info(f'Sampling length {sample_length}')
                for sample_i in range(self._sample_conf.samples_per_length):
                    sample_prefix = f'{length_prefix}_{str(sample_i)}_'
                    if os.path.exists(os.path.join(sample_dir,sample_prefix+'sample.pdb')):
                        continue
                    sample_output = self.sample(sample_length)
                    traj_paths = self.save_traj(
                        sample_output['prot_traj'],
                        sample_output['prot_0_traj'],
                        np.ones(sample_length),
                        output_dir=sample_dir,
                        save_traj = self._infer_conf.option.save_trajactory or self._infer_conf.option.plot.switch_on,
                        prefix = sample_prefix
                    )
                    if self._infer_conf.option.plot.switch_on:
                        # traj is flipped auto in model
                        if not self._infer_conf.option.plot.flip:
                            flip = lambda x: np.flip(np.stack(x), (0,))
                            sample_output['rigid_traj'] = flip(sample_output['rigid_traj'])
                            sample_output['rigid_0_traj'] = flip(sample_output['rigid_0_traj'])
                        plotting.write_traj(sample_output['rigid_traj'],os.path.join(movie_dir,sample_prefix+"rigid_movie.gif"))
                        plotting.write_traj(sample_output['rigid_0_traj'],os.path.join(movie_dir,sample_prefix+"rigid_0_movie.gif"))
                        
                    if self._infer_conf.option.self_consistency:
                        # Run ProteinMPNN
                        pdb_path = traj_paths['sample_path']

                        if self._infer_conf.single_process:
                            mpnn_result = run_mpnn(self.mpnn_model,self._infer_conf.mpnn,tmp_dir,pdb_path)
                            esmfold_results = self.run_esmfold(mpnn_result,decoy_pdb_dir=self._output_dir,tmp_dir=tmp_dir,reference_pdb_path= pdb_path)
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
                            run_esm_fold_callback_partial = partial(run_esm_fold_callback,output_dir=self._output_dir,reference_pdb_path=pdb_path)
                            # run_mpnn(self.mpnn_model,self._infer_conf.mpnn,sc_output_dir,pdb_path)
                            esmfold_result = pool.apply_async(run_mpnn,args=(self.mpnn_model,self._infer_conf.mpnn,tmp_dir,pdb_path),callback=run_esm_fold_callback_partial,error_callback=error_handler)
                            process_list.append(esmfold_result)
            if not self._infer_conf.single_process:
                for process in process_list:
                    process.wait()
            csv_path = os.path.join(self._output_dir,"self_consistency.csv")
            final_results = pd.concat(final_results).reset_index(drop=True)
            if os.path.exists(csv_path):
                try:
                    existed_result = pd.read_csv(csv_path)
                    final_results = pd.concat([existed_result,final_results])
                except:
                    csv_path = os.path.join(self._output_dir,"self_consistency.csv")
                    self._log.warning(f"{csv_path} has exists,and merge operation failed,override to original csv ")
            final_results.to_csv(os.path.join(self._output_dir,"self_consistency.csv"))
            fasta_path = os.path.join(self._output_dir, 'mpnn.fasta')
            with open(fasta_path,'w') as f:
                for i in range(len(final_results["header"])):
                    f.write(f'>{final_results["header"][i]}\n {final_results["sequence"][i]}\n')
            # clean tmp file
            shutil.rmtree(tmp_dir)
            # clean tmp file

    def save_traj(
            self,
            bb_prot_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str,
            save_traj : bool,
            prefix :str = "",
        ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = os.path.join(output_dir, prefix+'sample')
        prot_traj_path = os.path.join(output_dir, prefix+'bb_traj')
        x0_traj_path = os.path.join(output_dir, prefix+'x0_traj')

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = au.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors
        )
        if save_traj:
            prot_traj_path = au.write_prot_to_pdb(
                bb_prot_traj,
                prot_traj_path,
                b_factors=b_factors
            )
            x0_traj_path = au.write_prot_to_pdb(
                x0_traj,
                x0_traj_path,
                b_factors=b_factors
            )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            'x0_traj_path': x0_traj_path,
        }
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
    def sample(self, sample_length: int):
        """Sample based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs. See train_se3_diffusion.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, sample_length+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
            'sc_ca_t': np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t, 
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:
    logging.getLogger(__name__).setLevel(logging.INFO)
    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')
if __name__ == '__main__':
    run()
