"""PDB dataset loader."""
import math
from typing import TypeVar, Optional, Iterator
import ast
import os

import torch
import torch.distributed as dist

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn

from torch.utils import data
from data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y

class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser
        self._rng = np.random.default_rng()

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self):
        """
        Initialize metadata

        2023.5.31 UPDATED
        Property:
        # file level metadata
            pdb_name - pdb name
            processed_path - pkl path of processed feature
            raw_path - raw protein path
            resolution - resolution
            structure_method - x-ray diffraction ...etc
            release_date - release date of this cif, boost speed of template featurize in structure_prediction/data/templates.py
            
        # assemble level metadata
            assemble_id - (pdb_name,assemble_id) is the unique key of each row
            details - author_defined_assembly, software_defined_assembly ...etc, mitght considered to be a filter?
            oligomeric_details - monomeric, dimeric ...etc
            oligomeric_count - chains exists in this assemble
            oper_expression - currently only "1", other oper expression is filtered by data/process_pdb_dataset.py 
            asym_id_list - chain_ids exists in this assemble
            modeled_seq_len - total complex residue nums

        # assemble level geometric feature
        - all features below calculate soely based on this assemble strcture
        - which means the feature can be different by caculate each chain individually
        - eg, beta strand can form on chain-chain interface and clasp in single chain state
        - useful feature for dssp conditioned tranning
            coil_percent - coil_percent
            helix_percent - helix_percent
            strand_percent - strand_percent
            radius_gyration - radius_gyration
        
        """

        # Process CSV with different filtering criterions.


        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path,converters={'asym_id_list': lambda x : ast.literal_eval(x) })
        self.raw_csv = pdb_csv
        if filter_conf.allowed_oligomer is not None and len(filter_conf.allowed_oligomer) > 0:
            filter_conf.allowed_oligomer= [oligomer.strip() for oligomer in filter_conf.allowed_oligomer]
            filter_conf.allowed_oligomer.extend([f'{oligomer}ic' for oligomer in filter_conf.allowed_oligomer])
            self._log.info(f'Filtering oligomers: {filter_conf.allowed_oligomer}')
            pdb_csv = pdb_csv[pdb_csv.oligomeric_details.str.lower().isin(filter_conf.allowed_oligomer)]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile is not None \
            and filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, 
                filter_conf.rog_quantile,
                np.arange(filter_conf.max_len))
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x-1])
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[:filter_conf.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
        self._create_split(pdb_csv)

        self.cluster_csv = None
        if self.data_conf.cluster and self.is_training:
            if os.path.exists(self.data_conf.cluster):
                if self.data_conf.cluster.endswith(".tsv"):
                    cluster_tsv = pd.read_csv(self.data_conf.cluster,delimiter='\t',names=['repID','memID'],index_col='memID')
                    
                    self._log.info(f'load cluster file {self.data_conf.cluster}')
                    self._log.info(f'cluster info : {len(cluster_tsv.repID.unique())} clusters, {len(cluster_tsv)} chains')

                    cluster_dict = cluster_tsv['repID'].to_dict()
                    pdb_csv['rep_id_list'] = pdb_csv.apply(lambda x : tuple(sorted([cluster_dict.get(f'{x["pdb_name"]}_{asym_id}') for asym_id in x['asym_id_list'] if f'{x["pdb_name"]}_{asym_id}' in cluster_dict ])),axis=1)
                    self.csv = pdb_csv[pdb_csv['rep_id_list']!=()]
                    self.cluster_csv = pd.DataFrame(self.csv.groupby('rep_id_list',group_keys=False).apply(lambda group_data:group_data.index.tolist()),columns=['mem_id_list']).reset_index()
                else:
                    self._log.warning(f'Cluster file {self.data_conf.cluster} is not tsv format, cluster not used')
            else:
                self._log.warning(f'Cluster file {self.data_conf.cluster} not exists, cluster not used')
        if self.is_training :
            if self.cluster_csv is None:
                self._log.info(f'Training: {len(self.csv)} examples')
            else:
                self._log.info(f'Training: {len(self.cluster_csv)} clusters, {len(self.csv)} examples')

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
    # cache make the same sample in same batch 
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path , asym_id_list):

        chains_dict = du.read_pkl(processed_file_path)

        assemble_feats = du.concat_np_features([chains_dict[chain_id] for chain_id in asym_id_list], False)
        
        processed_feats = du.parse_chain_feats(assemble_feats)

        # Run through OpenFold data transforms.
        assemble_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        data_transforms.atom37_to_frames(assemble_feats)
        data_transforms.make_atom14_masks(assemble_feats)
        data_transforms.make_atom14_positions(assemble_feats)
        data_transforms.atom37_to_torsion_angles()(assemble_feats)
        data_transforms.atom37_to_torsion_angles("")(assemble_feats),
        data_transforms.make_pseudo_beta("")(assemble_feats),
        data_transforms.get_backbone_frames(assemble_feats),
        data_transforms.get_chi_angles(assemble_feats),

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
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
            'aatype': assemble_feats['aatype'],
            'seq_idx': new_res_idx,
            'chain_idx': new_chain_idx,
            'residx_atom37_to_atom14': assemble_feats['residx_atom37_to_atom14'],
            'residx_atom14_to_atom37': assemble_feats['residx_atom14_to_atom37'],
            'residue_index': processed_feats['residue_index'],
            'res_mask': processed_feats['bb_mask'],
            'atom37_pos': assemble_feats['all_atom_positions'],
            'atom37_mask': assemble_feats['all_atom_mask'],
            "atom37_atom_exists": assemble_feats["atom37_atom_exists"],
            'atom14_pos': assemble_feats['atom14_gt_positions'],
            'rigidgroups_0': assemble_feats['rigidgroups_gt_frames'],
            'torsion_angles_sin_cos': assemble_feats['torsion_angles_sin_cos'],
            # fape loss feture
            "backbone_rigid_mask": assemble_feats["backbone_rigid_mask"],
            "backbone_rigid_tensor": assemble_feats["backbone_rigid_tensor"],
            'rigidgroups_0': assemble_feats['rigidgroups_gt_frames'],
            'rigidgroups_gt_exists' : assemble_feats['rigidgroups_gt_exists'],
            "rigidgroups_gt_frames" : assemble_feats['rigidgroups_gt_frames'],
            'rigidgroups_alt_gt_frames' : assemble_feats['rigidgroups_alt_gt_frames'],
            # side chain feature
            # used in compute_renamed_ground_truth for ambigous prediction
            "atom14_gt_exists": assemble_feats["atom14_gt_exists"],
            "atom14_gt_positions": assemble_feats["atom14_gt_positions"],
            "atom14_alt_gt_exists": assemble_feats["atom14_alt_gt_exists"],
            "atom14_alt_gt_positions": assemble_feats["atom14_alt_gt_positions"],
            "atom14_atom_exists": assemble_feats["atom14_atom_exists"],
            "atom14_atom_is_ambiguous": assemble_feats["atom14_atom_is_ambiguous"],
        }
        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv) if self.cluster_csv is None else len(self.cluster_csv)

    def __getitem__(self, idx):
   
        # Use a fixed seed for evaluation.
        if not self.is_training:
            rng = np.random.default_rng(idx)
        else:
            rng = self._rng

        # Sample data example.
        if self.cluster_csv is None:
            csv_row = self.csv.iloc[idx]
        else:
            mem_ids = self.cluster_csv.iloc[idx]['mem_id_list']
            mem_id = rng.choice(mem_ids)
            csv_row = self.csv.loc[mem_id]

        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row['processed_path']

        assemble_feats = self._process_csv_row(processed_file_path,tuple(csv_row['asym_id_list']))

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            assemble_feats['rigidgroups_0'])[:, 0]
        diffused_mask = np.ones_like(assemble_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        assemble_feats['fixed_mask'] = fixed_mask
        assemble_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        diff_feats_t = {}
        # Sample t and diffuse.
        if self.is_training:
            # prev step feature for self-condition 
            t = np.random.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t_prev = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            diff_feats_t_prev['t'] = t
            # training step feature
            dt = (1.0-self._data_conf.min_t)/self._data_conf.num_t
            dt = [min(t-self._data_conf.min_t,dt*i) for i in self._data_conf.delta_t_range][rng.integers(len(self._data_conf.delta_t_range))]
            rigids_t = self._diffuser.reverse(
                rigid_t=rigid_utils.Rigid.from_tensor_7(diff_feats_t_prev['rigids_t']),
                rot_score=du.move_to_np(diff_feats_t_prev["rot_score"]),
                trans_score=du.move_to_np(diff_feats_t_prev["trans_score"]),
                diffuse_mask=None,
                t=t,
                dt=dt,
                center=False,
                noise_scale=1.0,
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
            diff_feats_t['t'] = t-dt
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
            diff_feats_t['t'] = t
        assemble_feats.update(diff_feats_t)

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), assemble_feats)
        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class LengthSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
        ):
        self._data_conf = data_conf
        self._dataset = dataset

    def __iter__(self):
        return iter(range(len(self._dataset)))

    def __len__(self):
        return len(self._dataset)

class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
        ):
        self._data_conf = data_conf
        self._dataset = dataset
        self._dataset_indices = list(range(len(self._dataset)))
        self._batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        random.shuffle(self._dataset_indices)
        repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
        return iter(repeated_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self._dataset_indices) * self._batch_size


# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, 
                *,
                data_conf,
                dataset,
                batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,order: bool = False,
                seed: int = 0, drop_last: bool = False,copy_num : int = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._dataset_indices = list(range(len(self._dataset)))
        self._copy_num = copy_num if copy_num else batch_size
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._dataset) if copy_num is None else copy_num * len(self._dataset)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.order = order
        self.seed = seed if seed is not None else 0

    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._dataset)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._copy_num)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # sort indices, speed up for length-sorted index
        if self.order:
            indices = sorted(indices)

        # 
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch