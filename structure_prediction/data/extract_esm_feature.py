"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""

import argparse
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import time
import pickle
from copy import deepcopy
import mdtraj as md
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBIO, MMCIFParser
from tqdm import tqdm
import torch.multiprocessing
# amazing! add this line work, without this line, OSError: [Errno 24] Too many open files just at processor start
# and i dont know why
torch.multiprocessing.set_sharing_strategy('file_system')
from data import errors, mmcif_parsing, parsers
from data import utils as du
from data import residue_constants
# Define the parser
parser = argparse.ArgumentParser(
    description='mmCIF processing script.')
parser.add_argument(
    '--mmcif_dir',
    help='Path to directory with mmcif files.',
    type=str)
parser.add_argument(
    '--cluster_file',
    help='Path to cluster file like pdb40 or some other, help to accelerate extract feature, drop protein not in this file.',
    type=str,
    default=None)
parser.add_argument(
    '--dir_format',
    help='''where each file is writen to, 
    code[1:3] : wirte feature to dir of middle code(2,3) of 4 letter pdb code(follow standard framediff dataprocess),
    code_chain : write feature to dir of code_chain, follow openfold dataset(OpenProteinSet)
    code : write feature to dir of code, follow AFDB
    ''',
    type=str,
    default="code_chain",choices=["code[1:3]","code","code_chain"])
parser.add_argument(
    '--max_file_size',
    help='Max file size.',
    type=int,
    default=3000000000)  # Only process files up to 3MB large.
parser.add_argument(
    '--min_file_size',
    help='Min file size.',
    type=int,
    default=1)  # Files must be at least 1KB.
parser.add_argument(
    '--max_resolution',
    help='Max resolution of files.',
    type=float,
    default=5.0)
parser.add_argument(
    '--max_len',
    help='Max length of protein.',
    type=int,
    default=512)
parser.add_argument(
    '--min_len',
    help='Min length of protein.',
    type=int,
    default=60)
parser.add_argument(
    '--num_processes',
    help='Number of processes.if gpu is used, this is the number of gpus',
    type=int,
    default=30)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='./data/processed_pdb')
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true',
    default=False)

args = parser.parse_args()

if args.cluster_file:
    valid_files = set([ name  for line in open(args.cluster_file,"r").readlines() for name in line.strip().split()])

def _retrieve_mmcif_files(
        mmcif_dir: str, max_file_size: int, min_file_size: int, debug: bool):
    """Set up all the mmcif files to read."""
    print('Gathering mmCIF paths')
    total_num_files = 0
    all_mmcif_paths = []
    mmcif_dir = mmcif_dir
    for subdir in tqdm(os.listdir(mmcif_dir)):
        mmcif_file_dir = os.path.join(mmcif_dir, subdir)
        if not os.path.isdir(mmcif_file_dir):
            continue
        for mmcif_file in os.listdir(mmcif_file_dir):
            mmcif_path = os.path.join(mmcif_file_dir, mmcif_file)
            total_num_files += 1
            if min_file_size <= os.path.getsize(mmcif_path) <= max_file_size:
                all_mmcif_paths.append(mmcif_path)
        if debug and total_num_files >= 100:
            # Don't process all files for debugging
            break
    print(
        f'Processing {len(all_mmcif_paths)} files of {total_num_files}')
    return all_mmcif_paths


def process_mmcif(
        mmcif_path: str, max_resolution: int, write_dir: str,model : torch.nn.Module,batch_converter):
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    mmcif_name = os.path.basename(mmcif_path).replace('.cif', '')
    if '_' in mmcif_name or mmcif_name.endswith(("gz","zip","pdb")):
        raise ValueError(f"{mmcif_name} is not supported name type" )
    # framdiff format
    with open(mmcif_path, 'r') as f:
        parsed_mmcif = mmcif_parsing.parse(
            file_id=mmcif_name, mmcif_string=f.read())
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(
            f'Encountered errors {parsed_mmcif.errors}'
        )
    parsed_mmcif = parsed_mmcif.mmcif_object

    # Parse mmcif header
    mmcif_header = parsed_mmcif.header
    mmcif_resolution = mmcif_header['resolution']
    if mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(
            f'Too high resolution {mmcif_resolution}')
    if mmcif_resolution == 0.0:
        raise errors.ResolutionError(
            f'Invalid resolution {mmcif_resolution}')

    # Extract all chains
    struct_chains = {
        chain.get_id(): chain
        for chain in parsed_mmcif.structure.get_chains()}

    # Extract features
    esm_data = {}
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        # chain_id_int = du.chain_str_to_int(chain_id)
        if args.cluster_file and any([name in valid_files for name in [mmcif_name,f'{mmcif_name}_{chain_id}']]):
            pass
        else :
            continue
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        if np.sum(chain_dict['aatype'] != 20) == 0:
            print('No modeled residues')
            continue
        if chain_dict['aatype'].shape[0] < args.min_len:
            print(f'name : {mmcif_name} chain : {chain_id} length : {chain_dict["aatype"].shape[0]} Too short ')
            continue
        if chain_dict['aatype'].shape[0] > args.max_len:
            print(f'name : {mmcif_name} chain : {chain_id} length : {chain_dict["aatype"].shape[0]} Too long ')
            continue
        
        modeled_idx = np.where(chain_dict['aatype'] != 20)[0]
        sequence = "".join([residue_constants.restypes_with_x[aatype] for aatype in chain_dict['aatype']])
        data = []
        data.append(("", sequence))
        batch_tokens = batch_converter(data)[2].to(next(model.parameters()).device)
        
        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[model.num_layers],return_contacts=True)
        token_representations = results["representations"][model.num_layers].cpu()[0]
        contact_map = results["contacts"].cpu()[0]
        esm_data[chain_id] = {"representation":token_representations,"contact_map":contact_map}
        if args.dir_format == "code_chain":
            mmcif_subdir = os.path.join(write_dir, f'{mmcif_name}_{chain_id}')
            os.makedirs(mmcif_subdir,exist_ok=True)
            processed_mmcif_path = os.path.join(mmcif_subdir, f'esm2.pkl')
            processed_mmcif_path = os.path.abspath(processed_mmcif_path)
            pickle.dump(esm_data[chain_id], open(processed_mmcif_path, 'wb'))

    # create folder with different format
    if args.dir_format == "code[1:3]":
        mmcif_subdir = os.path.join(write_dir, mmcif_name[1:3].lower())
        if not os.path.isdir(mmcif_subdir):
            os.mkdir(mmcif_subdir)
    elif args.dir_format == "code":
        mmcif_subdir = os.path.join(write_dir, mmcif_name)
        if not os.path.isdir(mmcif_subdir):
            os.mkdir(mmcif_subdir)
            
    #wirte file with different dir_format
    if args.dir_format == "code":
        processed_mmcif_path = os.path.join(mmcif_subdir, f'esm2.pkl')
        processed_mmcif_path = os.path.abspath(processed_mmcif_path)
        pickle.dump(esm_data, open(processed_mmcif_path, 'wb'))
    elif args.dir_format == "code[1:3]":
        processed_mmcif_path = os.path.join(mmcif_subdir, f'{mmcif_name}_esm2.pkl')
        processed_mmcif_path = os.path.abspath(processed_mmcif_path)
        pickle.dump(esm_data, open(processed_mmcif_path, 'wb'))
    elif args.dir_format == "code_chain":
        pass
    else : 
        raise ValueError(f"{args.dir_format} not implemented")

    return True


def process_serially(
        all_mmcif_paths, max_resolution,  write_dir,model,batch_converter):
    succeed_num = 0
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            process_mmcif(
                mmcif_path,
                max_resolution,
                write_dir,model,batch_converter)
            elapsed_time = time.time() - start_time
            succeed_num+=1
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        except errors.DataError as e:
            print(f'Failed {mmcif_path}: {e}')
    return succeed_num


def process_fn(
        mmcif_path,
        model,
        batch_converter,
        verbose=None,
        max_resolution=None,
        write_dir=None,):
    try:
        start_time = time.time()
        process_mmcif(
            mmcif_path,
            max_resolution,
            write_dir,model,batch_converter)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {mmcif_path} in {elapsed_time:2.2f}s')
        return True
    except errors.DataError as e:
        if verbose:
            print(f'Failed {mmcif_path}: {e}')

def process_fn_list(
    mmcif_paths,model,device,
    **kwargs):
    model.to(device).eval()
    for mmcif_path in mmcif_paths:
        process_fn(mmcif_path,model=model, **kwargs)
def main(args):
    device_count = torch.cuda.device_count()
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    if device_count == 0:
        device_count = args.num_processes
        print("Use CPU, num_processes  ", device_count)
        idx_to_device = {idx:'cpu' for idx in range(device_count)}
    else:
        # 显示 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"Found {device_count} available GPU(s) with CUDA version {cuda_version}")
        devices = [f"cuda:{i}" for i in range(device_count)]
        if args.num_processes>device_count:
            print("num_process is too much, set to gpu count : ",device_count)
            args.num_processes = device_count
        idx_to_device = {idx:devices[idx] for idx in range(device_count)}
    # if device_count == 0:
    #     device_count = args.num_processes
    #     print("Use CPU, num_processes  ", device_count)
    #     device_models = {index:deepcopy(model) for index in range(device_count)}
    #     idx_to_device = {idx:'cpu' for idx in range(device_count)}
    # else:
    #     # 显示 CUDA 版本
    #     cuda_version = torch.version.cuda
    #     print(f"Found {device_count} available GPU(s) with CUDA version {cuda_version}")
    #     devices = [f"cuda:{i}" for i in range(device_count)]
    #     for device in devices:
    #         device_model = deepcopy(model)
    #         device_model.to(device).eval()
    #         device_models[device] = device_model
    #     if args.num_processes>device_count:
    #         print("num_process is too much, set to gpu count : ",device_count)
    #         args.num_processes = device_count
    #     idx_to_device = {idx:devices[idx] for idx in range(device_count)}
    batch_converter = alphabet.get_batch_converter()
    # 检测系统上的 GPU 数量
    # Get all mmcif files to read.
    all_mmcif_paths = _retrieve_mmcif_files(
        args.mmcif_dir, args.max_file_size, args.min_file_size, args.debug)
    total_num_paths = len(all_mmcif_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if torch.cuda.is_available():
        mp.set_start_method('spawn',force=True)
    print(f'Files will be written to {write_dir}')
    split_data_for_each_model = {i:all_mmcif_paths[i:len(all_mmcif_paths):args.num_processes] for i in range(device_count)}

    # Process each mmcif file
    if args.num_processes == 1:
        model = model.to(idx_to_device[0]).eval()
        succeed_nums = process_serially(
            all_mmcif_paths,
            args.max_resolution,
            write_dir,
            model,batch_converter)
    else:
        _process_fn = fn.partial(
            process_fn_list,
            verbose=args.verbose,
            max_resolution=args.max_resolution,
            write_dir=write_dir,
            batch_converter=deepcopy(batch_converter))
        # If GPU is available, use all gpu, else uses max number of available cores of cpu.
        with mp.Pool(processes=args.num_processes) as pool:
        # 使用进程池在每个 GPU 上启动训练任务
            results = []
            for idx in range(args.num_processes):
                kwargs = {"mmcif_paths":split_data_for_each_model[idx],"model":model,"device":idx_to_device[idx]}
                result = pool.apply_async(_process_fn, kwds=kwargs)
                results.append(result)
            succeed_nums = [result.get() for result in results]
    succeeded = len(succeed_nums)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # use GPU if available
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)