import hydra
from omegaconf import DictConfig
import os
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import wandb
import hydra
import logging
import copy
import random
import pandas as pd
from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from openfold.utils import rigid_utils as ru
from openfold.utils.loss import (
    compute_renamed_ground_truth,
    compute_fape,
    supervised_chi_loss,
    lddt_ca
)
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from analysis import metrics
from data import pdb_data_loader
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import score_network
from experiments import utils as eu

from structure_prediction.data.pdb_data_loader import PDBDataset

class Experiment:
    
    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])
        # 
        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_wandb = self._exp_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp
        self._conf.experiment.seed = du.seed_everything(self._conf.experiment.seed)
        self._generator = np.random.default_rng(self._conf.experiment.seed)

        if self._use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.setLevel("ERROR")
                self._use_wandb = False
                self._exp_conf.ckpt_dir = None
        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if conf.experiment.warm_start:
            ckpt_dir = conf.experiment.warm_start
            self._log.info(f'Warm starting from: {ckpt_dir}')
            ckpt_files = [
                x for x in os.listdir(ckpt_dir)
                if 'pkl' in x or '.pth' in x
            ]
            if len(ckpt_files) != 1:
                raise ValueError(f'Ambiguous ckpt in {ckpt_dir}')
            ckpt_name = ckpt_files[0]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            ckpt_model = ckpt_pkl['model']

            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)
            conf.experiment.warm_start = ckpt_dir

            # For compatibility with older checkpoints.
            if 'optimizer' in ckpt_pkl:
                ckpt_opt = ckpt_pkl['optimizer']
            if 'epoch' in ckpt_pkl:
                self.trained_epochs = ckpt_pkl['epoch']
            if 'step' in ckpt_pkl:
                self.trained_steps = ckpt_pkl['step']

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = score_network.ScoreNetwork(self._model_conf, self._diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate)
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)

        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                )
            self._exp_conf.ckpt_dir = ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            OmegaConf.save(conf, os.path.join(ckpt_dir, 'config.yaml'))
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')
        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                )
            self._exp_conf.eval_dir = eval_dir
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir, exist_ok=True)
            OmegaConf.save(conf, os.path.join(eval_dir, 'config.yaml'))
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._log.info(f'Evaluation will not be saved.')
        self._aux_data_history = deque(maxlen=100)

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        
        # TODO valid dataset is not implemnted

        conf_train = DictConfig(self._data_conf)
        conf_train.update(self._data_conf.train)
        conf_eval = DictConfig(self._data_conf)
        conf_eval.update(self._data_conf.eval)
        # Datasets
        train_dataset = PDBDataset(
            data_conf=conf_train,
            diffuser=self._diffuser,
            is_training=True
        )
        train_sampler = pdb_data_loader.DistributedTrainSampler(
            data_conf=conf_train,
            dataset=train_dataset,
            batch_size=self._exp_conf.batch_size,
            copy_num= self._exp_conf.copy_num,
            num_replicas=self.ddp_info["world_size"] if self._use_ddp else 1,
            rank= self.ddp_info["rank"] if self._use_ddp else 0,
            seed = self._conf.experiment.seed,
            shuffle=True,
            # only do limited shuffle for multi gpu training
            order=True
        )

        train_loader = du.create_data_loader(
            train_dataset,
            data_conf=conf_train,
            sampler=train_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=self._exp_conf.num_loader_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        valid_dataset = None
        valid_loader = None
        valid_sampler = None
        if self.ddp_info['rank'] == 0 and self._exp_conf.eval_dir is not None and self._data_conf.eval.csv_path is not None:
            valid_dataset = PDBDataset(
                data_conf=conf_eval,
                diffuser=self._diffuser,
                is_training=False
            )
            valid_loader = du.create_data_loader(
                valid_dataset,
                data_conf=conf_eval,
                sampler=None,
                np_collate=False,
                length_batch=False,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

        return train_loader, valid_loader, train_sampler, valid_sampler

    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        os.makedirs(os.path.join(self._exp_conf.wandb_dir,self._exp_conf.name), exist_ok=True)
        wandb.init(
            project='se3-diffusion-strucutre-prediction',
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=os.path.join(self._exp_conf.wandb_dir,self._exp_conf.name),
        )
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')
        
    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_wandb and replica_id == 0:
            self.init_wandb()

        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            #muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [
                f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]
                ]
                #DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=True)
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                #DP mode
                else:
                    if len(self._available_gpus)>self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        self._model.train()

        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info('Done')
        return logs
    
    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(
            self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        
        for train_feats in train_loader:
            ckpt_metrics = None
            eval_time = None
            step_time = time.time()
            train_feats = tree.map_structure(
                lambda x: x.to(device), train_feats)
            
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1
            step_time = time.time() - step_time
            # print(f"batch_size :{train_feats['aatype'].shape[0]}, res length :{train_feats['aatype'].shape[1]}, template size {train_feats['template_aatype'].shape[1]}, {step_time:.2f}")
            example_per_sec = self._exp_conf.batch_size / step_time
            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(
                    f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if (self.trained_steps % self._exp_conf.ckpt_freq) == 0 or (self._exp_conf.early_ckpt and self.trained_steps == 100):
                if self._exp_conf.ckpt_dir is not None :
                    ckpt_path = os.path.join(
                        self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                    du.write_checkpoint(
                        ckpt_path,
                        self.model.state_dict(),
                        self._conf,
                        self._optimizer.state_dict(),
                        self.trained_epochs,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True
                    )

                # Run evaluation
                if valid_loader:
                    self._log.info(f'Running evaluation on step {self.trained_steps}')
                    start_time = time.time()
                    eval_dir = os.path.join(
                        self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                    os.makedirs(eval_dir, exist_ok=True)
                    ckpt_metrics = self.eval_fn(
                        eval_dir, valid_loader, device,
                        noise_scale=self._exp_conf.noise_scale
                    )
                    eval_time = time.time() - start_time
                    self._log.info(f'Finished evaluation in {eval_time:.2f}s')

            # Remote log to Wandb.
            if self._use_wandb:
                wandb_logs = {
                    'loss': loss,
                    'rotation_loss': aux_data['rot_loss'],
                    'translation_loss': aux_data['trans_loss'],
                    'bb_atom_loss': aux_data['bb_atom_loss'],
                    'dist_mat_loss': aux_data['batch_dist_mat_loss'],
                    'batch_size': aux_data['examples_per_step'],
                    'res_length': aux_data['res_length'],
                    'examples_per_sec': example_per_sec,
                    'num_epochs': self.trained_epochs,
                }
                # Stratified losses
                for k,v in aux_data.items() :
                    if "_loss" in k and "batch_" in k and (f'{k[6:]}_t_filter' not in self._exp_conf or train_feats['t'][0]<self._exp_conf[f'{k[6:]}_t_filter']):
                        wandb_logs.update(eu.t_stratified_loss(
                            du.move_to_np(train_feats['t']),
                            du.move_to_np(aux_data[k]),
                            loss_name=k[6:],
                        ))

                if ckpt_metrics is not None:
                    wandb_logs['eval_time'] = eval_time
                    for metric_name in metrics.ALL_METRICS:
                        wandb_logs[metric_name] = ckpt_metrics[metric_name].mean()
                    eval_table = wandb.Table(
                        columns=ckpt_metrics.columns.to_list()+['structure'])
                    for _, row in ckpt_metrics.iterrows():
                        pdb_path = row['sample_path']
                        row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                        eval_table.add_data(*row_metrics)
                    wandb_logs['sample_metrics'] = eval_table

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps"
                    )
                raise Exception(f'NaN encountered')

        if return_logs:
            return global_logs

    def eval_fn(self, eval_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats['res_mask'].bool())
            fixed_mask = du.move_to_np(valid_feats['fixed_mask'].bool())
            aatype = du.move_to_np(valid_feats['aatype'])
            gt_prot = du.move_to_np(valid_feats['atom37_pos'])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(
                lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats, min_t=min_t, num_t=num_t, noise_scale=noise_scale)
            final_prot = infer_out['prot_traj'][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_diffused_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_diffused = np.sum(unpad_diffused_mask) / num_res

                # Extract argmax predicted aatype
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    os.path.join(
                        eval_dir,
                        f'len_{num_res}_sample_{i}_diffused_{percent_diffused:.2f}.pdb'
                    ),
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        diffuse_mask=unpad_diffused_mask,
                    )
                except ValueError as e:
                    self._log.warning(
                        f'Failed evaluation of length {num_res} sample {i}: {e}')
                    continue
                sample_metrics['step'] = self.trained_steps
                sample_metrics['num_res'] = num_res
                sample_metrics['fixed_residues'] = np.sum(unpad_fixed_mask)
                sample_metrics['diffused_percentage'] = percent_diffused
                sample_metrics['sample_path'] = saved_path
                sample_metrics['gt_pdb'] = pdb_names[i]
                sample_metrics['lddt'] = float(lddt_ca(
                    infer_out['final_atom_positions'][i],
                    infer_out['all_atom_positions'][i],
                    infer_out['all_atom_mask'][i],
                    per_residue=False))
                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, 'metrics.csv')
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics
    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """

        self_condition = None
        model_out = None

        if self._model_conf.embed.embed_self_conditioning and self._generator.random() > 0.5:
            prev_batch = {}
            prev_batch = ({k:v for k,v in batch.items() if "self_condition_" not in k})
            prev_batch.update({k[len("self_condition_"):]:v for k,v in batch.items() if "self_condition_" in k})
            with torch.no_grad():
                self_condition = self._model(prev_batch)
                if self._generator.random() < 0.5:
                    self_condition = {k:v for k,v in self_condition.items() if  k not in ['edge_embed','node_embed']}
        if self._conf.model.profile:
        # with torch.autograd.profiler.profile(use_cuda=True,profile_memory=True,with_stack=True,use_cpu=False,use_kineto=True) as prof:
            with torch.autograd.profiler.profile(use_cuda=True,profile_memory=True,with_stack=False) as prof:
                model_out = self._model(batch,self_condition=self_condition,)
        else:
            model_out = self._model(batch,self_condition=self_condition)
        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch['rot_score']
        gt_trans_score = batch['trans_score']
        rot_score_scaling = batch['rot_score_scaling']
        trans_score_scaling = batch['trans_score_scaling']
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
        pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

        # Translation score loss
        trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
        trans_score_loss = torch.sum(
            trans_score_mse / trans_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch['rigids_0'][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out['rigids'][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = (
            trans_score_loss * (batch['t'] > self._exp_conf.trans_x0_threshold)
            + trans_x0_loss * (batch['t'] <= self._exp_conf.trans_x0_threshold)
        )
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._diff_conf.diffuse_trans)

        # Rotation loss
        rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
        rot_loss = torch.sum(
            rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight
        rot_loss *= int(self._diff_conf.diffuse_rot)

        # Backbone atom loss
        pred_atom37 = model_out['atom37'][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(torch.float32))
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
            gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3)
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask  = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight
       
        # Fape Loss 
        # rename_dict = compute_renamed_ground_truth(batch,model_out['positions'])
        # alt_naming_is_better = rename_dict['alt_naming_is_better']
        # renamed_atom14_gt_positions = rename_dict['renamed_atom14_gt_positions']
        # renamed_atom14_gt_exists = rename_dict['renamed_atom14_gt_exists']
        # renamed_gt_frames = (
        #     1.0 - alt_naming_is_better[..., None, None, None]
        # ) * batch["rigidgroups_gt_frames"] + alt_naming_is_better[
        #     ..., None, None, None
        # ] * batch["rigidgroups_alt_gt_frames"]
        # sidechain_frames = model_out['sidechain_frames']
        # batch_dims = sidechain_frames.shape[:-4]
        # sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
        # sidechain_frames = ru.Rigid.from_tensor_4x4(sidechain_frames)
        # renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
        # renamed_gt_frames = ru.Rigid.from_tensor_4x4(renamed_gt_frames)
        # rigidgroups_gt_exists = batch["rigidgroups_gt_exists"].reshape(*batch_dims, -1)
        # sidechain_atom_pos = model_out["positions"]
        # sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
        # renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        #     *batch_dims, -1, 3
        # )
        # renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)
        # fape_loss = compute_fape(
        #     pred_frames= sidechain_frames, 
        #     target_frames= renamed_gt_frames,
        #     frames_mask= rigidgroups_gt_exists,
        #     pred_positions =sidechain_atom_pos,
        #     target_positions = renamed_atom14_gt_positions,
        #     positions_mask = renamed_atom14_gt_exists,
        #     length_scale=1/self._exp_conf.coordinate_scaling,
        #     l1_clamp_distance=10)
        
        # Backbone Fape Loss
        fape_loss = compute_fape(
            pred_frames=ru.Rigid.from_tensor_7(model_out['rigids']),
            target_frames=ru.Rigid.from_tensor_7(batch['rigids_t']),
            frames_mask=loss_mask,
            pred_positions=pred_trans_x0,
            target_positions=gt_trans_x0,
            positions_mask=loss_mask,
            length_scale=1/self._exp_conf.coordinate_scaling,
            l1_clamp_distance=10,
        )

        fape_loss *= self._exp_conf.fape_loss_weight
        fape_loss *= batch['t'] < self._exp_conf.fape_loss_t_filter

        # chi_loss = supervised_chi_loss(
        #     # [B,1,N,7,2] add structure block dimision 
        #     angles_sin_cos= model_out["angles"][None,...],
        #     unnormalized_angles_sin_cos=model_out["unnormalized_angles"][None,...],
        #     aatype=batch["aatype"],
        #     seq_mask=batch["res_mask"].float(),
        #     chi_mask=batch["chi_mask"],
        #     chi_angles_sin_cos=batch["chi_angles_sin_cos"],
        #     chi_weight=self._exp_conf.chi_weight,
        #     angle_norm_weight=self._exp_conf.angle_norm_weight
        #     )
        # chi_loss *= batch['t'] < self._exp_conf.sidechain_loss_t_filter
        lddt = lddt_ca(
            model_out['final_atom_positions'],
            batch['all_atom_positions'],
            batch['atom37_atom_exists'],
            per_residue=False)
        final_loss = (
            rot_loss
            + trans_loss
            + bb_atom_loss
            + dist_mat_loss
            + fape_loss
        )

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            'batch_train_loss': final_loss,
            'batch_rot_loss': rot_loss.detach(),
            'batch_trans_loss': trans_loss.detach(),
            'batch_bb_atom_loss': bb_atom_loss.detach(),
            'batch_dist_mat_loss': dist_mat_loss.detach(),
            "batch_fape_loss": fape_loss.detach(),
            "batch_lddt_loss" : lddt.detach(),
            # "batch_chi_loss": chi_loss.detach(),
            'total_loss': normalize_loss(final_loss).detach(),
            'rot_loss': normalize_loss(rot_loss).detach(),
            'trans_loss': normalize_loss(trans_loss).detach(),
            'bb_atom_loss': normalize_loss(bb_atom_loss).detach(),
            'dist_mat_loss': normalize_loss(dist_mat_loss).detach(),
            "fape_loss": normalize_loss(fape_loss).detach(),
            # "chi_loss": normalize_loss(chi_loss).detach(),
            'examples_per_step': torch.tensor(batch_size),
            'res_length': torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        # self._aux_data_history.append({
        #     'aux_data': aux_data,
        #     'model_out': model_out,
        #     'batch': batch
        # })

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            self_condition=True,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_rigids_0 = []
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            model_out = None
            for t in reverse_steps:
                if t > min_t:
                    if not (self._model_conf.embed.embed_self_conditioning and self_condition):
                        model_out=None
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self._model(sample_feats,self_condition=model_out)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
                else:
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
                    all_rigids_0.append(du.move_to_np(rigid_pred))
                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi']
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred),
                        psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(
                    rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_rigids_0 = flip(all_rigids_0)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
            "final_atom_positions": model_out["final_atom_positions"],
            'all_atom_positions': sample_feats['all_atom_positions'],
            "all_atom_mask": model_out['final_atom_mask'],
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_rigids_0
            ret['prot_0_traj'] = all_bb_0_pred
        return ret

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats
    
@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(conf: DictConfig) -> None:

    os.environ["WANDB_START_METHOD"] = "thread"
    exp = Experiment(conf=conf)
    exp.start_training()
    
if __name__ == '__main__':
    run()



'''
    model input:
        'aatype': torch.Size([2, 256]), 
        'seq_idx': torch.Size([2, 256]), 
        'chain_idx': torch.Size([2, 256]), 
        'residx_atom14_to_atom37': torch.Size([2, 256, 14]), 
        'residx_atom37_to_atom14': torch.Size([2, 256, 37]), 
        'residue_index': torch.Size([2, 256]), 
        'res_mask': torch.Size([2, 256]), 
        'atom37_pos': torch.Size([2, 256, 37, 3]), 
        'atom37_mask': torch.Size([2, 256, 37]), 
        'atom14_pos': torch.Size([2, 256, 14, 3]), 
        'rigidgroups_0': torch.Size([2, 256, 8, 4, 4]), 
        'rigidgroups_gt_exists': torch.Size([2, 256, 8]), 
        'rigidgroups_gt_frames': torch.Size([2, 256, 8, 4, 4]), 
        'rigidgroups_alt_gt_frames': torch.Size([2, 256, 8, 4, 4]), 
        'torsion_angles_sin_cos': torch.Size([2, 256, 7, 2]), 
        'torsion_angles_mask': torch.Size([2, 256, 4]), 
        'all_atom_positions': torch.Size([2, 256, 37, 3]), 
        'atom37_atom_exists': torch.Size([2, 256, 37]), 
        'atom14_gt_exists': torch.Size([2, 256, 14]), 
        'atom14_gt_positions': torch.Size([2, 256, 14, 3]), 
        'atom14_alt_gt_exists': torch.Size([2, 256, 14]), 
        'atom14_alt_gt_positions': torch.Size([2, 256, 14, 3]), 
        'atom14_atom_exists': torch.Size([2, 256, 14]), 
        'atom14_atom_is_ambiguous': torch.Size([2, 256, 14]), 
        'fixed_mask': torch.Size([2, 256]), 
        'sc_ca_t': torch.Size([2, 256, 3]), 
        'trans_score': torch.Size([2, 256, 3]), 
        'rot_score': torch.Size([2, 256, 3]), 
        'template_mask': torch.Size([2, 4]), 
        'template_aatype': torch.Size([2, 4, 256]), 
        'template_pseudo_beta': torch.Size([2, 4, 256, 3]), 
        'template_pseudo_beta_mask': torch.Size([2, 4, 256]), 
        'template_all_atom_mask': torch.Size([2, 4, 256, 37]), 
        'template_all_atom_positions': torch.Size([2, 4, 256, 37, 3]), 
        'template_torsion_angles_mask': torch.Size([2, 4, 256, 7]), 
        'template_torsion_angles_sin_cos': torch.Size([2, 4, 256, 7, 2]), 
        'template_alt_torsion_angles_sin_cos': torch.Size([2, 4, 256, 7, 2]), 
        'template_sum_probs': torch.Size([2, 4, 1]), 
        't': torch.Size([2]), 
        'rot_score_scaling': torch.Size([2]), 
        'trans_score_scaling': torch.Size([2]), 
        'rigids_0': torch.Size([2, 256, 7]), 
        'rigids_t': torch.Size([2, 256, 7]), 
        'rigids_all_0': torch.Size([2, 256, 8, 7])

    model output:
        'rigids': torch.Size([2, 256, 7]), 
        'psi': torch.Size([2, 256, 2]), 
        'rot_score': torch.Size([2, 256, 3]), 
        'trans_score': torch.Size([2, 256, 3]), 
        'atom14': torch.Size([2, 256, 14, 3]), 
        'atom37': torch.Size([2, 256, 37, 3]), 
        'node_embed': torch.Size([2, 256, 256]), 
        'edge_embed': torch.Size([2, 256, 256, 128]), 
        'all_rigids': torch.Size([2, 256, 8, 7]), 
        'sidechain_frames': torch.Size([2, 256, 8, 4, 4]), 
        'unnormalized_angles': torch.Size([2, 256, 7, 2]), 
        'angles': torch.Size([2, 256, 7, 2]), 
        'positions': torch.Size([2, 256, 14, 3]), 
        'final_atom_positions': torch.Size([2, 256, 37, 3]), 
        'final_atom_mask': torch.Size([2, 256, 37])
    '''