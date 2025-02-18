"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
from data import utils as du
from data import all_atom
from model import ipa_pytorch
import functools as fn
from openfold.model.primitives import Attention
from openfold.model.template import (
    LightTemplatePairStackBlock,
    TemplatePointwiseAttention,
)
from openfold.model.pair_transition import PairTransition
from openfold.model.embedders import (
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
)
from openfold.model.msa import (
    MSAAttention
)
from openfold.utils.tensor_utils import (
    permute_final_dims,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    atom_order,
    resname_to_idx,
    STANDARD_ATOM_MASK
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
    atom14_to_atom37,
    pseudo_beta_fn,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import (
    dict_multimap,
    tensor_tree_map,
)
Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # output dimension
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size

        # Time step embedding
        node_in = self._embed_conf.feature.t + 1 + 21
        edge_in = (self._embed_conf.feature.t + 1 + 21) * 2

        # Sequence index embedding
        if self._embed_conf.feature.index:
            self.index_embedder = fn.partial(
                get_index_embedding,
                embed_size=self._embed_conf.feature.index
            )
            node_in += self._embed_conf.feature.index
            edge_in += self._embed_conf.feature.index
        # relative embedding
        if self._embed_conf.feature.rel_pos:
            self.rel_pos_embedder = PositinalEmbedder(max_relative_idx=self._embed_conf.feature.rel_pos)
            edge_in += self.rel_pos_embedder.no_bins

        # self-condition embedding
        if self._embed_conf.self_condition.version == 'baseline':
            edge_in += self._embed_conf.no_bins
        elif self._embed_conf.self_condition.version == 'template':
            self.template_embedder = TemplateEmbedder(self._embed_conf.template)
        elif not self._embed_conf.self_condition.version:
            pass
        else:
            raise ValueError(f"self_condition embedder : {self._embed_conf.self_condition.version} is not implemented")
        
        self.node_embedder = nn.Sequential(
            nn.Linear(node_in, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.feature.t
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            batch,
            t,
            fixed_mask,
            self_condition=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_condition: Mapping [str,Tensor] contains 'rigids' of prev step

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        seq_idx = batch['seq_idx']
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat(
        [   
            prot_t_embed,
            fixed_mask,
            torch.nn.functional.one_hot((batch["aatype"] if self._embed_conf.feature.aatype else torch.ones_like(batch["aatype"])*resname_to_idx["UNK"]),21).to(torch.float32)
        ],
        dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        if self._embed_conf.feature.index:
            node_feats.append(self.index_embedder(seq_idx))
            rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
            rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
            pair_feats.append(self.index_embedder(rel_seq_offset))
        
        # Relative position embedding
        if self._embed_conf.feature.rel_pos:
            pair_feats.append(self.rel_pos_embedder(batch).reshape([num_batch, num_res**2, -1]))

        # slef_condition feature process
        if self_condition:
            batch_size,res_num = batch["aatype"].shape[:2]
            standard_atom_mask =  self_condition["final_atom_positions"].new_tensor(STANDARD_ATOM_MASK)
            if not self._embed_conf.self_condition.aatype:
                self_condition['aatype'] = batch["aatype"]
            elif self._embed_conf.self_condition.aatype=='mask':
                self_condition["aatype"] = torch.ones([batch_size,res_num],dtype=torch.long,device=batch["aatype"].device)*resname_to_idx['GLY']
            else:
                raise ValueError(f"self_condition aatype : {self._embed_conf.self_condition.aatype} is not implemented")
            
            if not self._embed_conf.self_condition.all_atom_mask:
                self_condition["final_atom_mask"] = standard_atom_mask[self_condition["aatype"]]
            elif self._embed_conf.self_condition.all_atom_mask=='backbone':
                self_condition["final_atom_mask"] = self_condition["final_atom_mask"]*standard_atom_mask[resname_to_idx['GLY']][None,None,:]
            else:
                raise ValueError(f"self_condition all_atom_mask : {self._embed_conf.self_condition.all_atom_mask} is not implemented")

            self_condition["final_atom_positions"] = self_condition["final_atom_positions"]*self_condition["final_atom_mask"][...,None]

        # Self-conditioning distogram.
        if self._embed_conf.self_condition == 'baseline':
            sc_dgram = du.calc_distogram(
                torch.zeros_like(batch["rigids_t"][...,:4]) if not self_condition else self_condition["rigids"],
                self._embed_conf.feature.distogram.min_bin,
                self._embed_conf.feature.distogram.max_bin,
                self._embed_conf.feature.distogram.no_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])

        if self._embed_conf.self_condition.version == 'template':
            #### template embed and self-condition embed ####
            seq_mask = batch["res_mask"].float()
            pair_mask = seq_mask[...,:,None] * seq_mask[...,None,:]
            template_batch = {k:v for k,v in batch.items() if 'template_' in k} if "template_mask" in batch and batch["template_mask"].any() else None
            template_node_embed,template_edge_embed = self.template_embedder(
                node_embed=node_embed,
                edge_embed=edge_embed,
                pair_mask=pair_mask,
                template_batch=template_batch,
                self_condition=self_condition,
                )
            node_embed = node_embed + template_node_embed
            edge_embed = edge_embed + template_edge_embed

        return node_embed, edge_embed

class PositinalEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """
    def __init__(
        self,
        max_relative_idx: int,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Pair embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(PositinalEmbedder, self).__init__()

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.no_bins = 2 * max_relative_idx + 2 + 2

    def forward(self, batch):
        def one_hot(x, v_bins):
            reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
            diffs = x[..., None] - reshaped_bins
            am = torch.argmin(torch.abs(diffs), dim=-1)
            return nn.functional.one_hot(am, num_classes=len(v_bins)).float()
        pos = batch["residue_index"]
        chain_index = batch["chain_idx"]
        chain_index_same = (chain_index[..., None] == chain_index[..., None, :])
        asym_id = batch["asym_id"] if "asym_id" in batch else batch["chain_idx"]
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        # intra chain relative positional encoding, same asym_id share same relative position
        clipped_offset = torch.clamp(
            offset + self.max_relative_idx, 0, 2 * self.max_relative_idx
        )
        clipped_offset = torch.where(
            asym_id_same, 
            clipped_offset,
            (2 * self.max_relative_idx + 1) * 
            torch.ones_like(clipped_offset)
        )
        rel_feats = []
        
        boundaries = torch.arange(
            start=0, end=2 * self.max_relative_idx + 2, device=clipped_offset.device
        )
        rel_pos = one_hot(
            clipped_offset, boundaries,
        )
        rel_feats.append(rel_pos)

        # inter chain positional encoding, only 0 or 1
        chain_offset = torch.where(
            chain_index_same,  
            torch.ones_like(clipped_offset),
            torch.zeros_like(clipped_offset)
        )
        chain_boundaries = torch.arange(start=0,end=2,device=clipped_offset.device)
        chain_rel_pos = one_hot(
            chain_offset, chain_boundaries,
        )
        rel_feats.append(chain_rel_pos)

        rel_feats = torch.concat(rel_feats,dim=-1).float()

        return rel_feats

class TemplateColumnWiseAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, inf=1e9):
        super(TemplateColumnWiseAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.mha = Attention(
            self.c_in,
            self.c_in,
            self.c_in,
            self.c_hidden,
            self.no_heads,
            gating=True,
        )
    def forward(self, t, s,template_mask):
        bias = self.inf * (template_mask[..., None, None, None, :] - 1)

        # [*, N_res, 1, C_s]
        s = s.unsqueeze(-2)
        # [*, N_temp, N_res,  C_s] => [*, N_res, N_temp, C_s]
        t = permute_final_dims(t, ( 1, 0, 2))

        # [*, N_res, 1, C_s]
        biases = [bias]
        s = self.mha(q_x=s, kv_x=t, biases=biases)

        # [*, N_res, C_s]
        s = s.squeeze(-2)

        return s
class TemplateCrossEmbedder(nn.Module):
    """
    A naive fixup for misssing msa block
    Cross information of z,template_pair by point wise attention
    Cross information of s,template_angle by column wise attention
    input:
        t_s
        t_z
        s
        z

    """
            
    def __init__(
        self,
        config,
    ):
        super(TemplateCrossEmbedder, self).__init__()
        self.template_pointwise_att = TemplatePointwiseAttention(**config.template_pointwise_attention)
        self.template_columnwise_attention = TemplateColumnWiseAttention(**config.template_column_wise_attention)

    def forward(self,t_s,t_z,s,z,template_mask):

        s = self.template_columnwise_attention(t_s,s,template_mask)
        z = self.template_pointwise_att(t_z,z,template_mask)
        return s,z

class TemplateEmbedder(nn.Module):
    def __init__(self, template_config):

        super(TemplateEmbedder, self).__init__()  
        self.config = template_config
        self.self_condition_s = nn.Linear(template_config.c_s,template_config.c_s)
        self.self_condition_z = nn.Linear(template_config.c_z,template_config.c_t)
        self.template_angle_embedder = TemplateAngleEmbedder(**template_config.template_angle_embedder)
        self.template_pair_embedder = TemplatePairEmbedder(**template_config.template_pair_embedder)
        self.template_pair_stack = LightTemplatePairStackBlock(**template_config.template_pair_stack)
        self.template_cross_embedder = TemplateCrossEmbedder(template_config.template_cross_embedder)

    def forward(self,node_embed,edge_embed,pair_mask,template_batch=None,self_condition=None,):
        
        embeded_templates = []

        if template_batch:
            embeded_templates.append(self.template_embed(template_batch, pair_mask))
        if self_condition:
            embeded_templates.append(self.self_condition_embed(self_condition, pair_mask))
        if len(embeded_templates)>0:
            template_embed = dict_multimap(
                fn.partial(torch.cat, dim=1),embeded_templates
            )
            template_embed["template_pair_embedding"]  = torch.utils.checkpoint.checkpoint(fn.partial(self.template_pair_stack,mask=pair_mask[:,None]),template_embed["template_pair_embedding"],use_reentrant=False)
            t_s,t_z = self.template_cross_embedder(
                t_s = template_embed["template_angle_embedding"],
                t_z = template_embed["template_pair_embedding"],
                s= node_embed,
                z= edge_embed,
                template_mask = template_embed["template_mask"])
            return t_s,t_z
        else:
            return torch.zeros_like(node_embed),torch.zeros_like(edge_embed)
        
    def template_embed(self, batch, pair_mask): 

        # Discripency : AF2 and Openfold emded template amgle feature into msa feature, which do not exist in our model.
        # To avoid lost information of template angle, we add a light weight axial attention like msa transformer to embed template torsion into sequence embedding. 
        # Embed the templates one at a time (with a poor man's vmap)
        template_embeds = []
        n_batch,n_templ,n_res = batch["template_aatype"].shape[0:3]

        for i in range(n_templ):
            if batch["template_mask"][:,i].sum() == 0:
                template_embeds.append({
                    "angle" : torch.zeros([n_batch,1,n_res,self.config.c_s],device=pair_mask.device).detach(),
                    "pair" : torch.zeros([n_batch,1,n_res,n_res,self.config.c_t],device=pair_mask.device).detach(),
                })
                continue
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, 1, idx),
                batch,
            )

            single_template_embeds = {}

            template_angle_feat = build_template_angle_feat(
                single_template_feats,
            )

            # [*, S_t, N, C_m]
            a = self.template_angle_embedder(template_angle_feat)

            single_template_embeds["angle"] = a

            # [*, S_t, N, N, C_t]
            t = build_template_pair_feat(
                single_template_feats,
                inf=self.config.inf,
                eps=self.config.eps,
                **self.config.distogram,
            ).to(dtype=torch.float)

            t = self.template_pair_embedder(t)

            single_template_embeds.update({"pair": t})

            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            fn.partial(torch.cat, dim=1),
            template_embeds,
        )
        # template_embeds["pair"] = template_embeds["pair"] + self.template_pair_stack(template_embeds["pair"],pair_mask[:,None])

        # [*, S_t, N, N, C_z]
        # template_embeds["pair"] = self.template_pair_stack(
        #     template_embeds["pair"], 
        #     pair_mask.unsqueeze(-3).to(dtype=torch.float), 
        #     chunk_size = None
        # )

        ret = {
            "template_pair_embedding": template_embeds['pair'],
            "template_angle_embedding": template_embeds["angle"],
            "template_mask" : batch["template_mask"]
        }

        return ret
    def self_condition_embed(self, out,pair_mask): 
        
        batch_size,res_num = out["final_atom_positions"].shape[:2]

        self_condition_embeds = {}
        
        torsion_angles, torsion_mask = all_atom.prot_to_torsion_angles(
            out['aatype'],out["final_atom_positions"],out["final_atom_mask"])
        pseudo_beta , pseudo_beta_mask = pseudo_beta_fn(out['aatype'],out["final_atom_positions"],out["final_atom_mask"])
        
        condition_feats = {
            "template_aatype" : out['aatype'],
            "template_all_atom_positions" : out["final_atom_positions"],
            "template_all_atom_mask" : out["final_atom_mask"],
            "template_pseudo_beta" : pseudo_beta,
            "template_pseudo_beta_mask" : pseudo_beta_mask,
            "template_torsion_angles_sin_cos" : torsion_angles,
            "template_alt_torsion_angles_sin_cos" : torsion_angles,
            "template_torsion_angles_mask" : torsion_mask,
        }

        if  "node_embed" in out and "edge_embed" in out:
            condition_feats.update({
                "node_embed" : out["node_embed"],
                "edge_embed" : out["edge_embed"]
            })

        condition_feats = tensor_tree_map(lambda x : x[:,None,...],condition_feats)

        template_angle_feat = build_template_angle_feat(
            condition_feats,
        )
        # [*, S_t, N, C_m]
        a = self.template_angle_embedder(template_angle_feat)

        self_condition_embeds["angle"] = a

        # [*, S_t, N, N, C_t]
        t = build_template_pair_feat(
            condition_feats,
            inf=self.config.inf,
            eps=self.config.eps,
            **self.config.distogram,
        )
        self_condition_embeds["pair"] = self.template_pair_embedder(t)
        
        # [*, S_t, N, N, C_z]
        # self_condition_embeds["pair"] = self.template_pair_stack(
        #     self_condition_embeds["pair"], 
        #     pair_mask.to(dtype=torch.float), 
        #     chunk_size = None
        # )
        
        # [B,1]
        template_mask = torch.ones([batch_size,1]).to(device=pair_mask.device,dtype=torch.float)


        if "node_embed" in out and "edge_embed" in condition_feats:
            self_condition_embeds["angle"] = self_condition_embeds["angle"] + self.self_condition_s(condition_feats["node_embed"])
            self_condition_embeds["pair"] = self_condition_embeds["pair"] + self.self_condition_z(condition_feats["edge_embed"])
        
        ret = {
            "template_pair_embedding": self_condition_embeds["pair"],
            "template_angle_embedding": self_condition_embeds["angle"],
            "template_mask" : template_mask
        }

        return ret

class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats,self_condition=None):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            batch=input_feats,
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_condition=self_condition,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])
        rigids_pred = model_out['final_rigids']
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)

        pred_out = {
            #rigids
            'rigids' : rigids_pred.to_tensor_7(),
            # angle feature
            'psi': psi_pred,
            # score
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
            # atom14 for backbone
            'atom14' : bb_representations[-1].to(rigids_pred.device),
            # atom37 for backbone
            'atom37' : bb_representations[0].to(rigids_pred.device)
        }
        if self._model_conf.sidechain:
            # sidechain prediction
            all_frames_to_global = self.torsion_angles_to_frames(
                model_out['final_rigids'],
                model_out['angles'],
                input_feats["aatype"],
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                input_feats["aatype"],
            )
            final_atom_positions = atom14_to_atom37(
                pred_xyz, input_feats
            )
            pred_out_sidechain = {
                # rigids [B,N,8,7]
                'all_rigids' : all_frames_to_global.to_tensor_7(),

                # sidechain fape loss
                'sidechain_frames' : all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": model_out['unnormalized_angles'],
                "angles": model_out['angles'],
                # atom 14
                'positions' : pred_xyz,

                # atom 37 for atom loss and fape loss
                "final_atom_positions" : final_atom_positions,
                "final_atom_mask": input_feats["atom37_atom_exists"],
            }
            pred_out.update(pred_out_sidechain)
        return pred_out


    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )