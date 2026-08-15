import torch
import torch.nn as nn
import torch.nn.functional as F

from POMO.CVRPModel import (
    CVRP_Encoder,
    CVRP_Decoder,
    _get_encoding,
    multi_head_attention,
    reshape_by_heads,
)


class VRPTWModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = VRPTWEncoder(**model_params)
        self.decoder = VRPTWDecoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        node_features = torch.cat(
            (
                reset_state.node_xy,
                reset_state.node_demand[:, :, None],
                reset_state.node_service_time[:, :, None],
                reset_state.node_tw_start[:, :, None],
                reset_state.node_tw_end[:, :, None],
            ),
            dim=2,
        )
        self.encoded_nodes = self.encoder(reset_state.depot_xy, node_features)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size, pomo_size = state.BATCH_IDX.shape
        device = state.BATCH_IDX.device
        if state.selected_count == 0:
            return (
                torch.zeros(batch_size, pomo_size, dtype=torch.long, device=device),
                torch.ones(batch_size, pomo_size, device=device),
            )
        if state.selected_count == 1:
            return state.START_NODE, torch.ones(batch_size, pomo_size, device=device)

        encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
        probs = self.decoder(
            encoded_last, state.load, state.current_time, state.ninf_mask
        )
        if self.training or self.model_params["eval_type"] == "softmax":
            while True:
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                    selected = selected.squeeze(1).reshape(batch_size, pomo_size)
                probability = probs[state.BATCH_IDX, state.POMO_IDX, selected]
                if (probability > 0).all():
                    return selected, probability
        selected = probs.argmax(dim=2)
        return selected, None


class VRPTWEncoder(CVRP_Encoder):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        self.embedding_node = nn.Linear(6, model_params["embedding_dim"])


class VRPTWDecoder(CVRP_Decoder):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        self.Wq_last = nn.Linear(
            model_params["embedding_dim"] + 2,
            model_params["head_num"] * model_params["qkv_dim"],
            bias=False,
        )

    def forward(self, encoded_last_node, load, current_time, ninf_mask):
        head_num = self.model_params["head_num"]
        context = torch.cat(
            (encoded_last_node, load[:, :, None], current_time[:, :, None]), dim=2
        )
        q_last = reshape_by_heads(self.Wq_last(context), head_num=head_num)
        attention = multi_head_attention(
            q_last, self.k, self.v, rank3_ninf_mask=ninf_mask
        )
        mh_out = self.multi_head_combine(attention)
        score = torch.matmul(mh_out, self.single_head_key)
        score = score / self.model_params["sqrt_embedding_dim"]
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        return F.softmax(score + ninf_mask, dim=2)
