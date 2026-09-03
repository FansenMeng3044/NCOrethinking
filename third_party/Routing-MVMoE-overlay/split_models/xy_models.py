from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MOELayer import MoE
from models.MTLModel import MTL_Decoder as DirectMTLDecoder
from models.MTLModel import MTL_Encoder as DirectMTLEncoder
from models.MOEModel import MTL_Decoder as DirectMOEDecoder
from models.MOEModel import MTL_Encoder as DirectMOEEncoder
from models.MOEModel_Light import MTL_Decoder as DirectLightDecoder
from models.MOEModel_Light import MTL_Encoder as DirectLightEncoder


BL_CONTEXT_DIM = 5
BL_CANDIDATE_DIM = 4


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


def _xy_raw_embedding(model_params) -> nn.Module:
    embedding_dim = model_params["embedding_dim"]
    if model_params["num_experts"] > 1 and "Raw" in model_params["expert_loc"]:
        return MoE(
            input_size=2,
            output_size=embedding_dim,
            num_experts=model_params["num_experts"],
            k=model_params["topk"],
            T=1.0,
            noisy_gating=True,
            routing_level=model_params["routing_level"],
            routing_method=model_params["routing_method"],
            moe_model="Linear",
        )
    return nn.Linear(2, embedding_dim)


def _make_xy_encoder(kind: str, model_params) -> nn.Module:
    if kind == "mtl":
        encoder = DirectMTLEncoder(**model_params)
        encoder.embedding_node = nn.Linear(2, model_params["embedding_dim"])
    elif kind == "moe":
        encoder = DirectMOEEncoder(**model_params)
        encoder.embedding_node = _xy_raw_embedding(model_params)
    elif kind == "moe_light":
        encoder = DirectLightEncoder(**model_params)
        encoder.embedding_node = _xy_raw_embedding(model_params)
    else:
        raise ValueError(kind)
    return encoder


class XYOnlyDecoder(nn.Module):
    """Coordinate encoder query augmented only by dynamic B/L state."""

    def __init__(self, kind: str, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_bl = nn.Linear(BL_CONTEXT_DIM, head_num * qkv_dim, bias=False)
        self.bl_candidate_score = nn.Linear(BL_CANDIDATE_DIM, 1, bias=False)

        if kind == "mtl":
            direct = DirectMTLDecoder(**model_params)
        elif kind == "moe":
            direct = DirectMOEDecoder(**model_params)
        else:
            raise ValueError(kind)
        self.Wk = direct.Wk
        self.Wv = direct.Wv
        self.multi_head_combine = direct.multi_head_combine
        self.k = self.v = self.single_head_key = None

    def set_kv(self, encoded_nodes):
        heads = self.model_params["head_num"]
        self.k = _reshape_by_heads(self.Wk(encoded_nodes), heads)
        self.v = _reshape_by_heads(self.Wv(encoded_nodes), heads)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(
        self, encoded_first, encoded_last, bl_context, bl_candidate, ninf_mask
    ):
        heads = self.model_params["head_num"]
        q = _reshape_by_heads(self.Wq_first(encoded_first), heads)
        q = q + _reshape_by_heads(self.Wq_last(encoded_last), heads)
        q = q + _reshape_by_heads(self.Wq_bl(bl_context), heads)
        out_concat = _multi_head_attention(q, self.k, self.v, ninf_mask)
        aux_loss = out_concat.new_zeros(())
        if isinstance(self.multi_head_combine, MoE):
            attention_out, aux_loss = self.multi_head_combine(out_concat)
        else:
            attention_out = self.multi_head_combine(out_concat)
        score = torch.matmul(attention_out, self.single_head_key)
        score = score + self.bl_candidate_score(bl_candidate).squeeze(3)
        score = self.model_params["logit_clipping"] * torch.tanh(
            score / self.model_params["sqrt_embedding_dim"]
        )
        return F.softmax(score + ninf_mask, dim=2), aux_loss


class XYOnlyLightDecoder(nn.Module):
    """MVMoE-L decoder with coordinate encoding and dynamic B/L state."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        heads, qkv_dim = model_params["head_num"], model_params["qkv_dim"]
        self.Wq_first = nn.Linear(embedding_dim, heads * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, heads * qkv_dim, bias=False)
        self.Wq_bl = nn.Linear(BL_CONTEXT_DIM, heads * qkv_dim, bias=False)
        self.bl_candidate_score = nn.Linear(BL_CANDIDATE_DIM, 1, bias=False)
        direct = DirectLightDecoder(**model_params)
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

    def forward(
        self, encoded_first, encoded_last, bl_context, bl_candidate, ninf_mask,
        temperature=1.0, step=1,
    ):
        heads = self.model_params["head_num"]
        q = _reshape_by_heads(self.Wq_first(encoded_first), heads)
        q = q + _reshape_by_heads(self.Wq_last(encoded_last), heads)
        q = q + _reshape_by_heads(self.Wq_bl(bl_context), heads)
        out_concat = _multi_head_attention(q, self.k, self.v, ninf_mask)
        aux_loss = out_concat.new_zeros(())

        if self.hierarchical_gating:
            # Original direct decoding initializes this choice at selected_count=2
            # (depot, then first customer). Giant tours omit depot, hence step=1.
            if step == 1 or self.branch_probs is None:
                pooled = out_concat.mean(dim=0).mean(dim=0).unsqueeze(0)
                self.branch_probs = F.softmax(self.dense_or_moe(pooled) / temperature, dim=-1)
            branch = self.branch_probs.multinomial(1).squeeze(0)
            if branch.item() == 1:
                attention_out, aux_loss = self.multi_head_combine_moe(out_concat)
            else:
                attention_out = self.multi_head_combine_dense(out_concat)
            attention_out = attention_out * self.branch_probs.squeeze(0)[branch]
        else:
            attention_out = self.multi_head_combine_dense(out_concat)

        score = torch.matmul(attention_out, self.single_head_key)
        score = score + self.bl_candidate_score(bl_candidate).squeeze(3)
        score = self.model_params["logit_clipping"] * torch.tanh(
            score / self.model_params["sqrt_embedding_dim"]
        )
        return F.softmax(score + ninf_mask, dim=2), aux_loss


def _get_encoding(encoded_nodes, indices):
    embedding_dim = encoded_nodes.size(2)
    gather = indices[:, :, None].expand(-1, -1, embedding_dim)
    return encoded_nodes.gather(1, gather)


class _XYOnlySplitModel(nn.Module):
    """XY-only static encoding with B/L-aware autoregressive decoding."""

    kind = "mtl"

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = model_params["eval_type"]
        self.encoder = _make_xy_encoder(self.kind, model_params)
        self.decoder = (
            XYOnlyLightDecoder(**model_params)
            if self.kind == "moe_light"
            else XYOnlyDecoder(self.kind, **model_params)
        )
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
        # This is the enforcement boundary: these are the only two inputs read.
        depot_xy, node_xy = reset_state.depot_xy, reset_state.node_xy
        encoded = self.encoder(depot_xy, node_xy)
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

        encoded_first = _get_encoding(self.encoded_nodes, state.START_NODE)
        encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
        if self.kind == "moe_light":
            probs, decoder_aux = self.decoder(
                encoded_first, encoded_last, state.bl_context,
                state.bl_candidate, state.ninf_mask,
                temperature=self.temperature, step=state.selected_count,
            )
        else:
            probs, decoder_aux = self.decoder(
                encoded_first, encoded_last, state.bl_context,
                state.bl_candidate, state.ninf_mask
            )
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


class POMOMTLSplit(_XYOnlySplitModel):
    """POMO-MTL-Split: XY encoding, B/L decoding, and C/TW Split."""

    kind = "mtl"


class MVMoE4ESplit(_XYOnlySplitModel):
    """MVMoE/4E-Split: XY encoding, B/L decoding, and C/TW Split."""

    kind = "moe"


class MVMoE4ELSplit(_XYOnlySplitModel):
    """MVMoE/4E-L-Split: XY encoding, B/L decoding, and C/TW Split."""

    kind = "moe_light"


def get_split_model(name: str) -> Type[_XYOnlySplitModel]:
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
