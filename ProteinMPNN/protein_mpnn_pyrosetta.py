import argparse
import os.path
import  os, sys
import logging
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os.path
import torch.multiprocessing as mp
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
from ProteinMPNN.protein_mpnn_utils import  _scores, _S_to_seq, tied_featurize
from ProteinMPNN.protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN
from ProteinMPNN.pyrosetta_utils import thread_mpnn_seq,parse_pose,pose_to_string

init( "-beta_nov16 -in:file:silent_struct_type binary -output_pose_energies_table false" +
    " -holes:dalphaball /home/caolongxingLab/caolongxing/bin/DAlphaBall.gcc" +
    " -use_terminal_residues true -mute basic.io.database core.scoring protocols all" +
    " -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8" +
    " -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8" )

def parse_args( argv ):
    argv_tmp = sys.argv
    sys.argv = argv
    description = 'do protein sequence design using the MPNN model ...'
    parser = argparse.ArgumentParser( description = description )
    parser.add_argument('-pdbs', type=str, nargs='*', help='name of the input pdb file')
    parser.add_argument('-pdb_list', type=str, help='a list file of all pdb files')
    parser.add_argument("-omit_AAs", type=list, default='CX', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    parser.add_argument( "-temperature", type=float, default=0.0001, help='An a3m file containing the MSA of your target' )
    parser.add_argument( "-relax_cycles", type=int, default="1", help="The number of MPNN->FastRelax cycles to perform (default 1)" )
    parser.add_argument("-output_path", type=str, default='', help="the output path")
    parser.add_argument('-prefix', type=str, default='', help='the prefix of the output file name')
    args = parser.parse_args()
    sys.argv = argv_tmp

    return args

# args = parse_args( sys.argv )




xml = "/storage/caolongxingLab/wangchentong/work/se3_diffusion/ProteinMPNN/helper_scripts/xml_scripts/design.xml"
objs = protocols.rosetta_scripts.XmlObjects.create_from_file( xml )
# Load the movers we will need
pack_monomer = objs.get_mover( 'pack_monomer' )
relax_monomer = objs.get_mover( 'relax_monomer' )
fades_monomer = objs.get_mover( 'fastdes_monomer' )
sfxn = core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")


def model_init(device , config):
    
    file_path = os.path.realpath(__file__)
    k = file_path.rfind("/")
    if config.ca_only:
        model_folder_path = file_path[:k] + '/ca_model_weights/'
    else:
        model_folder_path = file_path[:k] + '/vanilla_model_weights/'
    checkpoint_path = model_folder_path + f'{config.model_name}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_dim = 128
    num_layers = 3 
    model = ProteinMPNN(ca_only=config.ca_only, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=config.backbone_noise, k_neighbors=checkpoint['num_edges']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# TODO add binder design mask for mpnn
def mpnn_design( model, pose,config,chains = None,pool : mp.Pool = None ):
    device = next(model.parameters()).device
    sample_nums = config.best_selection.sample_nums if config.best_selection.switch_on else 1
    if isinstance(pose,pyrosetta.rosetta.core.pose.Pose):
        pose = core.pose.deep_copy(pose)
    if os.path.exists(pose) and pose.endswith('.pdb'):
        pose = pyrosetta.pose_from_file(pose)
    mpnn_result = []
    def mpnn_cycle(pose):
        # global settings
        omit_AAs_list = 'CX'
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))
        pdb_dict_list = parse_pose(pose,chains, ca_only=config.ca_only)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)
        batch_seqs = []
        batch_scores = []
        with torch.no_grad():
            batch_clones = [copy.deepcopy(dataset_valid[0]) for i in range(1)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, None, None, None, None, None,ca_only=config.ca_only)
            pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float() #1.0 for true, 0.0 for false
            for i in range(sample_nums):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=config.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=False, bias_by_res=bias_by_res_all)
            
                S_sample = sample_dict["S"] 
            
                log_probs =  model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])

                mask_for_loss = mask*chain_M*chain_M_pos
                scores = _scores(S_sample, log_probs, mask_for_loss)
                scores = scores.cpu().data.numpy()
                masked_chain_length_list = masked_chain_length_list_list[0]
                masked_list = masked_list_list[0]
                seq = _S_to_seq(S_sample[0], chain_M[0])
                score = scores[0]
                batch_seqs.append(seq)
                batch_scores.append(score)
        index = np.argmin(scores,axis=0)
        return batch_seqs[index],batch_scores[index]
    if config.fades :
        fades_monomer.apply(pose)
    for index in range(1,config.num_seqs+1):
        score_traj = []
        sequnce_traj = []
        mpnn_pose = core.pose.deep_copy(pose)
        pose_traj = [pose_to_string(mpnn_pose)]

        for i in range(config.cycle):
            mpnn_pose = core.pose.deep_copy(mpnn_pose)
            new_seq,mpnn_score = mpnn_cycle(mpnn_pose)
            if config.cycle>1 or config.dump:
                mpnn_pose = thread_mpnn_seq(mpnn_pose, new_seq)
                relax_monomer.apply(mpnn_pose)
                # Pyrosetta Pose cant not serialization without build from source code
                pose_traj.append(pose_to_string(mpnn_pose))
            score_traj.append(mpnn_score)
            sequnce_traj.append(new_seq)
        mpnn_dict = {"index":index,"sequence":sequnce_traj[-1],"score" : score_traj[-1],"pose":pose_traj[-1],"sequnce_traj":sequnce_traj,"score_traj":score_traj,"pose_traj":pose_traj}
        mpnn_result.append(mpnn_dict)

    return mpnn_result