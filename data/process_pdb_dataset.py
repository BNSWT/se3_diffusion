"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""
import tree
import logging
import argparse
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import time
import io

import mdtraj as md
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser
from tqdm import tqdm

from data import errors, mmcif_parsing, parsers
from data import utils as du
from data import residue_constants
from data.protein import Protein,to_pdb
from data.utils import quaternary_category

from Bio.Seq import Seq
from Bio import pairwise2


logging.basicConfig(level=logging.INFO,
    # define this things every time set basic config
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ])
base_dir = os.path.dirname(os.path.abspath(__file__))+'/'
# Define the parser
parser = argparse.ArgumentParser(
    description='mmCIF processing script.')
parser.add_argument(
    '--mmcif_dir',
    help='Path to directory with mmcif files.',
    type=str,
    default=base_dir+'./mmCIF/')
parser.add_argument(
    '--max_file_size',
    help='Max file size.',
    type=int,
    default=3000000)  # Only process files up to 3MB large.
parser.add_argument(
    '--min_file_size',
    help='Min file size.',
    type=int,
    default=1000)  # Files must be at least 1KB.
parser.add_argument(
    '--max_resolution',
    help='Max resolution of files.',
    type=float,
    default=5.0)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=20)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default=base_dir+'./processed_pdb')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')
parser.add_argument(
    '--mode',
    help='Whether to overwrite processed pickel.',
    choices=["overwrite","update"])
# some option for sequence and structure clusters
parser.add_argument(
    '--dump_chain_pdb',
    help='whther dump each protein chain in pdb format',
    action='store_true')
parser.add_argument(
    '--dump_chain_fasta',
    help='whther dump each protein chain in pdb format',
    action='store_true')
def score_sequence(seq_i,seq_j):
    alignments = pairwise2.align.globalxx(seq_i, seq_j)
    
    # Get the alignment score
    alignment_score = alignments[0].score
    
    # Calculate the similarity as a percentage
    similarity = (alignment_score / len(seq_i)) * 100
    return similarity

def assembly_filter(struct_chains,assembels):
    """ this function aims to extract all chains needed for all unique assembly in this cif
        assembels (Mapping[assemble_id,str]):
            details: Who defined the assembly way : author_defined_assembly, software_defined_assembly ..etc
            oligomeric_details: the definition of assembly way : monomeric, dimeric ...etc
            oligomeric_count: How many chains in this assemble
            oper_expression : chains list in this assemble
            asym_id_list : List[str] chain ids of this assemble
        
        Return:
            chain_sets : all chains id needed for all unique assembles
            assembely_sets : all unique assembles id
        
    """
    chain_sets = set()
    chain_sequence = {}
    assembely_sets = {}
    for chain_id, chain in struct_chains.items():
        chain_sequence[chain_id] = "".join([ residue_constants.restype_3to1.get(res.resname, 'X') for res in chain ]).strip("X")
    
    for assemble_id,assembely_property in assembels.items():

        ######### currently only not support oper_expression !!! #############
        if assembely_property['oper_expression'] != '1':
            continue

        assemble_sequences = [chain_sequence[chain_id] for chain_id in assembely_property['asym_id_list'] if chain_id in struct_chains and chain_sequence[chain_id] != ""]
        
        if assembely_property['oligomeric_details'] not in assembely_sets:
            assembely_sets[assembely_property['oligomeric_details']] = {assemble_id:assemble_sequences}
        else:
            # check if this is a new assmbles compare to previous asemmble in same oligomeric type
            # by all-to-all sequence alignment
            one_to_all_match = {assemble_id:True for assemble_id in list(assembely_sets[assembely_property['oligomeric_details']].keys())}
            for pre_assemble_id,pre_assemble_sequences in assembely_sets[assembely_property['oligomeric_details']].items():
                for assemble_sequence in assemble_sequences:
                    if not any([min(score_sequence(assemble_sequence,pre_assemble_sequence),score_sequence(pre_assemble_sequence,assemble_sequence))>95 for pre_assemble_sequence in pre_assemble_sequences]):
                        one_to_all_match[pre_assemble_id] = False
                        break
                if one_to_all_match[pre_assemble_id] == True:
                    # this assemble has matched of pre assemble, drop it
                    break
            if not any([match for ammseble_id,match in one_to_all_match.items()]):
                # no match, add this assemble to unique assemble
                assembely_sets[assembely_property['oligomeric_details']][assemble_id] = assemble_sequences
    for assembely_set in assembely_sets:
        for assemble_id in assembely_sets[assembely_set]:
            for chain_id in assembels[assemble_id]['asym_id_list']:
                chain_sets.add(chain_id)
    assembely_sets = set([assemble_id  for oligomeric_details,assembles in assembely_sets.items() for assemble_id in assembles])
    return chain_sets,assembely_sets


def _retrieve_mmcif_files(
        mmcif_dir: str, max_file_size: int, min_file_size: int, debug: bool):
    """Set up all the mmcif files to read."""
    print('Gathering mmCIF paths')
    total_num_files = 0
    all_mmcif_paths = []
    for subdir in tqdm(os.listdir(mmcif_dir)):
        mmcif_file_dir = os.path.join(mmcif_dir, subdir)
        if not os.path.isdir(mmcif_file_dir):
            continue
        for mmcif_file in os.listdir(mmcif_file_dir):
            if not mmcif_file.endswith('.cif'):
                continue
            mmcif_path = os.path.join(mmcif_file_dir, mmcif_file)
            total_num_files += 1
            # if min_file_size <= os.path.getsize(mmcif_path) <= max_file_size:
            all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 10:
            all_mmcif_paths = all_mmcif_paths[:10]
            break
    print(
        f'Processing {len(all_mmcif_paths)} files out of {total_num_files}')
    return all_mmcif_paths


def process_mmcif(
        mmcif_path: str, max_resolution: int, write_dir: str , verbose : bool):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    # metadata as assemble as a unit
    metadatas = []
    assemble_metadatas = []
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '')
    mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
    if not os.path.isdir(mmcif_subdir):
        os.mkdir(mmcif_subdir)
    processed_mmcif_path = os.path.join(mmcif_subdir, f'{mmcif_name}.pkl')
    processed_mmcif_path = os.path.abspath(processed_mmcif_path)
    try:
        with open(mmcif_path, 'r') as f:
            parsed_mmcif = mmcif_parsing.parse(
                file_id=mmcif_name, mmcif_string=f.read())
    except:
        raise errors.FileExistsError(
            f'Error file do not exist {mmcif_path}'
        )
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object
    raw_mmcif = parsed_mmcif.raw_string

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}')
    # TODO flatern medadatas
    metadata = {
        'pdb_name' : mmcif_name,
        'processed_path' : processed_mmcif_path,
        'raw_path' : mmcif_path,
        'resolution': mmcif_resolution,
        'structure_method' : mmcif_header['structure_method'],
        'release_date' : mmcif_header["release_date"]
    }

    # Extract all chains
    # fix bug, chain id of upper letter and lower case can exist in the same file
    struct_chains = {
        chain.id: chain
        for chain in parsed_mmcif.structure.get_chains()}

    # We only dump this chains, by the description of cif file)
        # 
    keep_chains = None

    ##### assemble filter for all unique assembles and their chains #####
    if '_pdbx_struct_assembly.oligomeric_count' in raw_mmcif:
        try:
            assembly_property = mmcif_parsing.get_assembly_mapping(raw_mmcif)
            keep_chains,keep_assembles = assembly_filter(struct_chains,assembly_property)
            # only keep protein chains
            keep_chains = [chain_id for chain_id in keep_chains if chain_id in struct_chains]
            for assemble_id in keep_assembles:
                protein_chains = [chain_id for chain_id in  assembly_property[assemble_id]['asym_id_list'] if chain_id in keep_chains]
                assemble_metadatas.append({
                    'assemble_id' : assemble_id,
                    'details' : assembly_property[assemble_id]['details'],
                    'oligomeric_details': assembly_property[assemble_id]['oligomeric_details'],
                    'oligomeric_count': assembly_property[assemble_id]['oligomeric_count'],
                    'oper_expression': assembly_property[assemble_id]['oper_expression'],
                    'asym_id_list': protein_chains,
                })
            # Return when there is no proper assemble by reasons below:
            # 1. those assembles need oper_expression not support now
            if not keep_assembles:
                return None
        except Exception as e:
            raise errors.DataError(f"Error occured during parsing assembles of {mmcif_path} \n")
    else:
        []

    # Extract features
    chains_dict = {}
    chains_sequence = {}
    for chain_id, chain in struct_chains.items():
        try:
            # direct use char chain id
            chain_prot = parsers.process_chain(chain, chain_id)
            chain_dict = dataclasses.asdict(chain_prot)

            ##################### chain filter ###############
            if keep_chains and chain_id not in keep_chains:
                continue
            ##################### chain filter ###############

            # cut out unknown residues of both side
            modeled_idx = np.where(chain_dict["aatype"] != 20)[0]
            min_idx = np.min(modeled_idx)
            max_idx = np.max(modeled_idx)
            chain_dict = tree.map_structure(
                lambda x: x[min_idx:(max_idx+1)], chain_dict)
            chain_length = chain_dict['aatype'].shape[0]
            chain_dict["residue_index"] = np.arange(0,chain_length)
            chains_sequence[chain_id] = "".join([ residue_constants.restype_3to1.get(res.resname, 'X') for res in chain ])[min_idx:(max_idx+1)]
            chains_dict[chain_id] = chain_dict
        except Exception as e:
            logging.warning(errors.DataError(f"Error occured during process {mmcif_path} chain {chain_id}"))


    if not chains_dict:
        raise errors.DataError(f"No protein chains found in {mmcif_path}")
    dump_chains = set()
    for assemble_data in assemble_metadatas:
        try:
            assemble_data["asym_id_list"] = [chain_id for chain_id in assemble_data["asym_id_list"] if chain_id in chains_dict]
            
            if not assemble_data["asym_id_list"]:
                logging.info(errors.DataError(f"no protein chains exists in assemble {assemble_data['assemble_id']}"))
                continue
            
            assemble_chains_dict = [chains_dict[chain_id] for chain_id in assemble_data["asym_id_list"]]
            dump_chains.extend(assemble_data["asym_id_list"])
            assemble_feat_dict = du.concat_np_features(assemble_chains_dict, False)
            # trasform to pdb
            pdb_path = mmcif_path.replace('.cif', '.pdb')
            protein_keys = ["atom_positions", "aatype", "atom_mask", "residue_index", "chain_index", "b_factors"]
            protein_dict = {k:v for k,v in assemble_feat_dict.items() if k in protein_keys}
            chain_protein = Protein(**protein_dict)
            pdb_string = to_pdb(chain_protein)
            open(pdb_path,"w").write(pdb_string)

            # MDtraj
            traj = md.load(pdb_path)
            pdb_ss = md.compute_dssp(traj, simplified=True)
            pdb_ss = pdb_ss[0]
            # DG calculation
            pdb_dg = md.compute_rg(traj)
            os.remove(pdb_path)
            metadatas.append({
                # file level metadata
                **metadata,
                # assemble level metadata
                **assemble_data,
                # assemble level geometric data
                'modeled_seq_len' : assemble_feat_dict['aatype'].shape[0],
                'coil_percent' : np.sum(pdb_ss == 'C') / len(pdb_ss),
                'helix_percent' : np.sum(pdb_ss == 'H') / len(pdb_ss),
                'strand_percent' : np.sum(pdb_ss == 'E') / len(pdb_ss),
                'radius_gyration' : pdb_dg[0],

            })
        except Exception:
            logging.warning(errors.DataError(f"Error occured during process {mmcif_path} assemble {assemble_data['assemble_id']}\n"))
    
    # dump pdb and fasta for cluster
    for chain_id in dump_chains:
        if args.dump_chain_fasta and chain_id in chains_sequence :
            fasta_path = os.path.join(mmcif_subdir, f'{mmcif_name}_{chain_id}.fasta')
            if not os.path.exists(fasta_path):
                open(fasta_path,"w").write(f">{mmcif_name}_{chain_id}\n{chains_sequence[chain_id]}")
        if args.dump_chain_pdb and chain_id in chains_dict :
            pdb_path = os.path.join(mmcif_subdir, f'{mmcif_name}_{chain_id}.pdb')
            chain_protein = Protein(**chains_dict[chain_id])
            pdb_string = to_pdb(chain_protein)
            if not os.path.exists(pdb_path):
                open(pdb_path,"w").write(pdb_string)
    
    if not metadatas:
        raise errors.DataError(f"No assemble found in {mmcif_path}")
    if verbose:
        logging.info(metadatas)
    # Write features to pickles.
    if not os.path.exists(processed_mmcif_path) or args.mode=='overwrite':
        du.write_pkl(processed_mmcif_path, chains_dict)

    # Return metadata
    return metadatas


def process_serially(
        all_mmcif_paths, max_resolution, write_dir):
    all_metadata = []
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            metadata = process_mmcif(
                mmcif_path,
                max_resolution,
                write_dir,
                verbose = True)
            elapsed_time = time.time() - start_time
            logging.info(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except Exception as e:
            logging.exception(f'Failed {mmcif_path}: {e}')
    return all_metadata


def process_fn(
        mmcif_path,
        verbose=None,
        max_resolution=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_mmcif(
            mmcif_path= mmcif_path,
            max_resolution= max_resolution,
            write_dir= write_dir,
            verbose = verbose)
        elapsed_time = time.time() - start_time
        logging.info(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return metadata
    except Exception as e:
        # exception will print full traceback
        logging.warning(f'Failed {mmcif_path}: {e}')
        return None


def main(args):
    # Get all mmcif files to read.
    all_mmcif_paths = _retrieve_mmcif_files(
        args.mmcif_dir, args.max_file_size, args.min_file_size, args.debug)
    total_num_paths = len(all_mmcif_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_mmcif_paths= all_mmcif_paths,
            max_resolution= args.max_resolution,
            write_dir= write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            max_resolution=args.max_resolution,
            write_dir=write_dir)
        # Uses max number of available cores.
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_mmcif_paths)
            
    all_metadata = [x for x in all_metadata if x is not None]
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')

    # flatten each file-level metadata to assemble-level metadata
    flatten_list = lambda lst: [x for sublst in lst for x in (flatten_list(sublst) if isinstance(sublst, list) else [sublst])]
    flattened_metadata = flatten_list(all_metadata)

    metadata_df = pd.DataFrame(flattened_metadata)
    if not os.path.exists(metadata_path) or args.mode=='overwrite':
        metadata_df.to_csv(metadata_path, index=False)
    elif args.mode=='update':
        if os.path.exists(metadata_path):
            old_metadata_df = pd.read_csv(metadata_path)
            metadata_df = pd.concat([old_metadata_df,metadata_df],ignore_index=True)


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)