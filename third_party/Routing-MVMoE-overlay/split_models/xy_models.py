from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce

from models.MOELayer import MoE, SparseDispatcher
from models.MTLModel import MTL_Decoder as DirectMTLDecoder
from models.MTLModel import MTL_Encoder as DirectMTLEncoder
from models.MOEModel import MTL_Decoder as DirectMOEDecoder
from models.MOEModel import MTL_Encoder as DirectMOEEncoder
from models.MOEModel_Light import MTL_Decoder as DirectLightDecoder
from models.MOEModel_Light import MTL_Encoder as DirectLightEncoder


def _distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    """Autograd-aware global mean used by batch-dependent MoE decisions."""
    if not _distributed_active():
        return value
    return differentiable_all_reduce(value, op=dist.ReduceOp.SUM) / dist.get_world_size()


class DistributedConsistentMoE(MoE):
    """MoE whose input-choice balancing loss uses the global DDP batch.

    Expert dispatch remains local to each rank. Only the batch statistics used
    by the auxiliary loss are reduced, which makes its value and gradient
    match a single process operating on the concatenated global batch.
    """

    def forward(self, x, loss_coef=1e-3, prob_emb=None):
        if (
            not _distributed_active()
            or self.routing_method != "input_choice"
            or self.routing_level not in ("node", "instance")
        ):
            return super().forward(x, loss_coef=loss_coef, prob_emb=prob_emb)

        output_shape = list(x.size()[:-1]) + [self.output_size]
        if self.routing_level == "instance":
            if x.dim() != 3:
                raise ValueError("instance-level MoE input must be three-dimensional")
        else:
            x = x.reshape(-1, self.input_size) if x.dim() != 2 else x

        gates, load = self.noisy_top_k_gating(x, self.training)
        # The coefficient of variation is scale-invariant. Averaging the
        # per-rank sums therefore gives the same loss as concatenating tokens.
        importance = _distributed_mean(gates.sum(0))
        load = _distributed_mean(load)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        dispatcher = SparseDispatcher(
            self.num_experts, gates, routing_level=self.routing_level
        )
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [
            self.experts[index](expert_inputs[index])
            for index in range(self.num_experts)
        ]
        output = dispatcher.combine(expert_outputs)
        return output.reshape(output_shape), loss


def _enable_distributed_consistent_moe(module: nn.Module) -> None:
    """Upgrade upstream MoE instances without modifying the upstream checkout."""
    for child in module.modules():
        if isinstance(child, MoE) and not isinstance(child, DistributedConsistentMoE):
            child.__class__ = DistributedConsistentMoE


def _reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch, count, _ = qkv.shape
    return qkv.reshape(batch, count, head_num, -1).transpose(1, 2)


def _multi_head_attention(q, k, v, ninf_mask):
    key_dim = q.size(-1)
    score = torch.matmul(q, k.transpose(2, 3)) / (key_dim ** 0.5)
    score = score + ninf_mask[:, None, :, :]
    weights = F.softmax(score, dim=3)
    out = torch.matmul(weights, v).transpose(1, 2)
    return out.reshape(out.size(0), out.size(1), -1)


def _make_static_encoder(kind: str, model_params) -> nn.Module:
    """Build the unchanged original encoder with five customer features."""
    if kind == "mtl":
        encoder = DirectMTLEncoder(**model_params)
    elif kind == "moe":
        encoder = DirectMOEEncoder(**model_params)
    elif kind == "moe_light":
        encoder = DirectLightEncoder(**model_params)
    else:
        raise ValueError(kind)
    return encoder


class SplitDecoder(nn.Module):
    """Original MVMoE decoder context with a customer-only action space."""

    def __init__(self, kind: str, **model_params):
        super().__init__()
        self.model_params = model_params
        if kind == "mtl":
            direct = DirectMTLDecoder(**model_params)
        elif kind == "moe":
            direct = DirectMOEDecoder(**model_params)
        else:
            raise ValueError(kind)
        self.Wq_last = direct.Wq_last
        self.Wk = direct.Wk
        self.Wv = direct.Wv
        self.multi_head_combine = direct.multi_head_combine
        self.k = self.v = self.single_head_key = None

    def set_kv(self, encoded_nodes):
        heads = self.model_params["head_num"]
        self.k = _reshape_by_heads(self.Wk(encoded_nodes), heads)
        self.v = _reshape_by_heads(self.Wv(encoded_nodes), heads)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last, attr, ninf_mask):
        heads = self.model_params["head_num"]
        decoder_input = torch.cat((encoded_last, attr), dim=2)
        q = _reshape_by_heads(self.Wq_last(decoder_input), heads)
        out_concat = _multi_head_attention(q, self.k, self.v, ninf_mask)
        aux_loss = out_concat.new_zeros(())
        if isinstance(self.multi_head_combine, MoE):
            attention_out, aux_loss = self.multi_head_combine(out_concat)
        else:
            attention_out = self.multi_head_combine(out_concat)
        score = torch.matmul(attention_out, self.single_head_key)
        score = self.model_params["logit_clipping"] * torch.tanh(
            score / self.model_params["sqrt_embedding_dim"]
        )
        return F.softmax(score + ninf_mask, dim=2), aux_loss


class SplitLightDecoder(nn.Module):
    """MVMoE-L decoder with the original four dynamic attributes."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        heads = model_params["head_num"]
        direct = DirectLightDecoder(**model_params)
        self.Wq_last = direct.Wq_last
        self.Wk, self.Wv = direct.Wk, direct.Wv
        self.hierarchical_gating = direct.hierarchical_gating
        self.multi_head_combine_dense = direct.multi_head_combine_dense
        if self.hierarchical_gating:
            self.dense_or_moe = direct.dense_or_moe
            self.multi_head_combine_moe = direct.multi_head_combine_moe
        self.k = self.v = self.single_head_key = None
        self.branch_probs = None

    def set_kv(self, encoded_nodes):
        heads = self.model_params["head_num"]
        self.k = _reshape_by_heads(self.Wk(encoded_nodes), heads)
        self.v = _reshape_by_heads(self.Wv(encoded_nodes), heads)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.branch_probs = None

    def forward(self, encoded_last, attr, ninf_mask, temperature=1.0, step=1):
        heads = self.model_params["head_num"]
        decoder_input = torch.cat((encoded_last, attr), dim=2)
        q = _reshape_by_heads(self.Wq_last(decoder_input), heads)
        out_concat = _multi_head_attention(q, self.k, self.v, ninf_mask)
        aux_loss = out_concat.new_zeros(())

        if self.hierarchical_gating:
            # Original direct decoding initializes this choice at selected_count=2
            # (depot, then first customer). Giant tours omit depot, hence step=1.
            if step == 1 or self.branch_probs is None:
                pooled = out_concat.mean(dim=0).mean(dim=0).unsqueeze(0)
                pooled = _distributed_mean(pooled)
                self.branch_probs = F.softmax(self.dense_or_moe(pooled) / temperature, dim=-1)
            if _distributed_active():
                if dist.get_rank() == 0:
                    branch = self.branch_probs.multinomial(1).squeeze(0)
                else:
                    branch = torch.zeros((), device=out_concat.device, dtype=torch.long)
                dist.broadcast(branch, src=0)
            else:
                branch = self.branch_probs.multinomial(1).squeeze(0)
            if branch.item() == 1:
                attention_out, aux_loss = self.multi_head_combine_moe(out_concat)
            else:
                attention_out = self.multi_head_combine_dense(out_concat)
            attention_out = attention_out * self.branch_probs.squeeze(0)[branch]
        else:
            attention_out = self.multi_head_combine_dense(out_concat)

        score = torch.matmul(attention_out, self.single_head_key)
        score = self.model_params["logit_clipping"] * torch.tanh(
            score / self.model_params["sqrt_embedding_dim"]
        )
        return F.softmax(score + ninf_mask, dim=2), aux_loss


def _get_encoding(encoded_nodes, indices):
    embedding_dim = encoded_nodes.size(2)
    gather = indices[:, :, None].expand(-1, -1, embedding_dim)
    return encoded_nodes.gather(1, gather)


class _FeatureAlignedSplitModel(nn.Module):
    """Original static and dynamic inputs with customer-only decoding."""

    kind = "mtl"

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = model_params["eval_type"]
        self.encoder = _make_static_encoder(self.kind, model_params)
        self.decoder = (
            SplitLightDecoder(**model_params)
            if self.kind == "moe_light"
            else SplitDecoder(self.kind, **model_params)
        )
        _enable_distributed_consistent_moe(self)
        self.encoded_nodes = None
        self.aux_loss = torch.tensor(0.0)
        self.temperature = 1.0

    def set_eval_type(self, eval_type: str):
        if eval_type not in ("argmax", "softmax"):
            raise ValueError("eval_type must be argmax or softmax")
        self.eval_type = eval_type

    def set_temperature(self, temperature: float):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def pre_forward(self, reset_state):
        # Match the original fixed input order exactly: (x, y, q, a, b).
        node_features = torch.cat(
            (
                reset_state.node_xy,
                reset_state.node_demand[:, :, None],
                reset_state.node_tw_start[:, :, None],
                reset_state.node_tw_end[:, :, None],
            ),
            dim=2,
        )
        if node_features.size(2) != 5:
            raise RuntimeError("customer encoder input must contain five features")
        encoded = self.encoder(reset_state.depot_xy, node_features)
        if isinstance(encoded, tuple):
            self.encoded_nodes, encoder_aux = encoded
            self.aux_loss = encoder_aux
        else:
            self.encoded_nodes = encoded
            self.aux_loss = self.encoded_nodes.new_zeros(())
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, selected: Optional[torch.Tensor] = None):
        batch, pomo = state.BATCH_IDX.shape
        device = self.encoded_nodes.device
        if state.selected_count == 0:
            selected = state.START_NODE if selected is None else selected
            prob = torch.ones(batch, pomo, device=device, dtype=self.encoded_nodes.dtype)
            return selected, prob

        encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
        attr = torch.cat(
            (
                state.load[:, :, None],
                state.current_time[:, :, None],
                state.length[:, :, None],
                state.open[:, :, None],
            ),
            dim=2,
        )
        if self.kind == "moe_light":
            probs, decoder_aux = self.decoder(
                encoded_last, attr, state.ninf_mask,
                temperature=self.temperature, step=state.selected_count,
            )
        else:
            probs, decoder_aux = self.decoder(encoded_last, attr, state.ninf_mask)
        self.aux_loss = self.aux_loss + decoder_aux

        if selected is None:
            if self.training or self.eval_type == "softmax":
                selected = probs.reshape(batch * pomo, -1).multinomial(1).reshape(batch, pomo)
            else:
                selected = probs.argmax(dim=2)
        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
        if not torch.isfinite(prob).all() or (prob <= 0).any():
            raise RuntimeError("policy selected a masked or non-finite action")
        return selected, prob


class POMOMTLSplit(_FeatureAlignedSplitModel):
    """POMO-MTL-Split with original inputs and final exact Split."""

    kind = "mtl"


class MVMoE4ESplit(_FeatureAlignedSplitModel):
    """MVMoE/4E-Split with original inputs and final exact Split."""

    kind = "moe"


class MVMoE4ELSplit(_FeatureAlignedSplitModel):
    """MVMoE/4E-L-Split with original inputs and final exact Split."""

    kind = "moe_light"


def get_split_model(name: str) -> Type[_FeatureAlignedSplitModel]:
    models = {
        "MTL_SPLIT": POMOMTLSplit,
        "POMO_MTL_SPLIT": POMOMTLSplit,
        "MOE_SPLIT": MVMoE4ESplit,
        "MVMOE_4E_SPLIT": MVMoE4ESplit,
        "MOE_LIGHT_SPLIT": MVMoE4ELSplit,
        "MVMOE_4E_L_SPLIT": MVMoE4ELSplit,
    }
    key = name.upper()
    if key not in models:
        raise ValueError(f"unknown Split model {name!r}; choose one of {sorted(models)}")
    return models[key]
