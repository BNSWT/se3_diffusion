"""PDB dataset loader."""
import math
from typing import TypeVar, Optional, Iterator
import sys
import os
import functools
import torch

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
from omegaconf import DictConfig
from torch.utils import data
# get toplevel data module
sys.path.insert(0,os.path.dirname(__file__)+'/../../')
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils


# TODO load mmcif data cooperate with framediff
class PDBDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._diffuser = diffuser
        self._is_training = is_training
        self._data_conf = DictConfig(data_conf)
        if self._is_training:
            self.data_conf.update(self.data_conf.train)
        else:
            self.data_conf.update(self.data_conf.eval)
        self._init_metadata()
        logging.info(f'Loaded {len(self.csv)} {"training" if is_training else "eval"} samples')

    @property
    def is_training(self):
        return self._is_training

    @property
    def data_conf(self):
        return self._data_conf
    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv
        # if filter_conf.allowed_oligomer is not None and len(filter_conf.allowed_oligomer) > 0:
        #     pdb_csv = pdb_csv[pdb_csv.oligomeric_detail.isin(
        #         filter_conf.allowed_oligomer)]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        # if filter_conf.max_helix_percent is not None:
        #     pdb_csv = pdb_csv[
        #         pdb_csv.helix_percent < filter_conf.max_helix_percent]
        # if filter_conf.max_loop_percent is not None:
        #     pdb_csv = pdb_csv[
        #         pdb_csv.coil_percent < filter_conf.max_loop_percent]
        # if filter_conf.min_beta_percent is not None:
        #     pdb_csv = pdb_csv[
        #         pdb_csv.strand_percent > filter_conf.min_beta_percent]
        # if filter_conf.rog_quantile is not None \
        #     and filter_conf.rog_quantile > 0.0:
        #     prot_rog_low_pass = _rog_quantile_curve(
        #         pdb_csv, 
        #         filter_conf.rog_quantile,
        #         np.arange(filter_conf.max_len))
        #     row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
        #         lambda x: prot_rog_low_pass[x-1])
        #     pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[:filter_conf.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
        self.csv = pdb_csv
    @functools.lru_cache(maxsize=32)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        """standard openfold data transform
        - means origin has this transform 
        + means we new add it
        =>normal transform
        not add now because we do not need msa
            data_transforms.cast_to_64bit_ints,
            data_transforms.correct_msa_restypes,
            data_transforms.squeeze_features,
            data_transforms.randomly_replace_msa_with_unknown(0.0),
            data_transforms.make_seq_mask,
            data_transforms.make_msa_mask,
            data_transforms.make_hhblits_profile,
        - data_transforms.make_atom14_masks
        """

        # Run through OpenFold data transforms.
        chain_feats = {
            # protein feats
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double(),

            # template feats
            "template_all_atom_positions": torch.Tensor(processed_feats["template_all_atom_positions"]),
            "template_all_atom_mask": torch.Tensor(processed_feats["template_all_atom_mask"]),
            "template_aatype": torch.Tensor(processed_feats["template_aatype"]),
            "template_sum_probs": torch.Tensor(processed_feats["template_sum_probs"]),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        #TODO tamplate feature introduce
        # template transform
        data_transforms.fix_templates_aatype(chain_feats),
        data_transforms.make_template_mask(chain_feats),
        data_transforms.make_pseudo_beta("template_")(chain_feats)
        data_transforms.atom37_to_torsion_angles("template_")(chain_feats),

        #supervise transfrom
        data_transforms.atom37_to_torsion_angles("")(chain_feats),
        data_transforms.make_pseudo_beta("")(chain_feats),
        data_transforms.get_backbone_frames(chain_feats),
        data_transforms.get_chi_angles(chain_feats),

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        if any([not str(idx).isdigit() for idx in chain_idx]):
            chain_idx = np.array([du.chain_str_to_int(idx) for idx in chain_idx ])
        res_idx = processed_feats['residue_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # To speed up processing, only take necessary features
        final_feats = {
            # backbone feature
            'aatype': chain_feats['aatype'],
            'seq_idx': new_res_idx,
            'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            "residx_atom37_to_atom14": chain_feats['residx_atom37_to_atom14'],
            'residue_index': processed_feats['residue_index'],
            'res_mask': processed_feats['bb_mask'],
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            
            # fape loss feture
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            'rigidgroups_gt_exists' : chain_feats['rigidgroups_gt_exists'],
            "rigidgroups_gt_frames" : chain_feats['rigidgroups_gt_frames'],
            'rigidgroups_alt_gt_frames' : chain_feats['rigidgroups_alt_gt_frames'],

            # this feature is used both for calculating O postion in all_atom.compute_backbone function and supervised chi loss
            'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
            "chi_mask": chain_feats["chi_mask"],
            'torsion_angles_sin_cos' : chain_feats['torsion_angles_sin_cos'],
            'torsion_angles_mask' : chain_feats['torsion_angles_mask'],

            #fape loss feature,we can directly calculate by transform rigids
            # "backbone_rigid_mask": chain_feats["backbone_rigid_mask"],
            # "backbone_rigid_tensor": chain_feats["backbone_rigid_tensor"],

            # rename feature follow openfold
            'all_atom_positions': chain_feats['all_atom_positions'],
            "atom37_atom_exists": chain_feats["atom37_atom_exists"],


            # side chain feature
            # used in compute_renamed_ground_truth for ambigous prediction
            "atom14_gt_exists": chain_feats["atom14_gt_exists"],
            "atom14_gt_positions": chain_feats["atom14_gt_positions"],
            "atom14_alt_gt_exists": chain_feats["atom14_alt_gt_exists"],
            "atom14_alt_gt_positions": chain_feats["atom14_alt_gt_positions"],
            "atom14_atom_exists": chain_feats["atom14_atom_exists"],
            "atom14_atom_is_ambiguous": chain_feats["atom14_atom_is_ambiguous"],

            #TODO tamplate feature introduce
            # template feature
            "template_mask": chain_feats["template_mask"],
            "template_aatype": chain_feats["template_aatype"],
            "template_pseudo_beta": chain_feats["template_pseudo_beta"],
            "template_pseudo_beta_mask": chain_feats["template_pseudo_beta_mask"],
            "template_all_atom_mask": chain_feats["template_all_atom_mask"],
            "template_all_atom_positions": chain_feats["template_all_atom_positions"],
            "template_torsion_angles_mask": chain_feats["template_torsion_angles_mask"],
            "template_torsion_angles_sin_cos": chain_feats["template_torsion_angles_sin_cos"],
            "template_alt_torsion_angles_sin_cos": chain_feats["template_alt_torsion_angles_sin_cos"],
            "template_sum_probs": chain_feats["template_sum_probs"],

        }
        return final_feats


    def __len__(self):
        return len(self.csv)

    # TODO care this three feature  i delete 'fixed_mask','rigids_0','sc_ca_t'
    def __getitem__(self, idx):

        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        elif 'name' in  csv_row:
            pdb_name = csv_row['name']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])[:, 0]
        gt_all_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['rigids_all_0'] = gt_all_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse. Prepare prev_step features for self_condition training
        if self.is_training:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            dt = (1.0-self._data_conf.min_t)/self._data_conf.num_t
            dt = [min(t-self._data_conf.min_t,dt*i) for i in self._data_conf.delta_t_range][rng.integers(len(self._data_conf.delta_t_range))]
            diff_feats_t_prev = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            diff_feats_t_prev['t'] = t
            rigids_t = self._diffuser.reverse(
                rigid_t=rigid_utils.Rigid.from_tensor_7(diff_feats_t_prev['rigids_t']),
                rot_score=du.move_to_np(diff_feats_t_prev["rot_score"]),
                trans_score=du.move_to_np(diff_feats_t_prev["trans_score"]),
                diffuse_mask=None,
                t=t,
                dt=dt,
                center=False,
                noise_scale=0.0,
            )
            rot_score = self._diffuser.calc_rot_score(
                rigids_t.get_rots()[None,...],
                gt_bb_rigid.get_rots()[None,...],
                t = torch.Tensor([t-dt])
            )[0]
            trans_score = self._diffuser.calc_trans_score(
                rigids_t.get_trans()[None,...],
                gt_bb_rigid.get_trans()[None,...],
                t = torch.Tensor([t-dt]),
            )[0]
            rot_score_scaling,trans_score_scaling = self._diffuser.score_scaling(t-dt)
            diff_feats_t = {
                'rigids_t': rigids_t.to_tensor_7(),
                'trans_score': trans_score,
                'rot_score': rot_score,
                'trans_score_scaling': trans_score_scaling,
                'rot_score_scaling': rot_score_scaling,
                'fixed_mask': fixed_mask,
                **{"self_condition_"+k:v for k,v in diff_feats_t_prev.items()}
            }
            t = t-dt
        else:
            t = 1.0
            diff_feats_t = self._diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t
        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name
