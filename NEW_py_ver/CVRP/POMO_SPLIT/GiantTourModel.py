import torch
import torch.nn as nn
import torch.nn.functional as F


class GiantTourModel(nn.Module):
    """POMO policy that outputs only a permutation of the customers.

    The depot is encoded as context but is permanently masked from the action
    space by the environment. Static customer attributes and the dynamic decoder
    context match the corresponding Direct model. Capacity and time-window
    feasibility masks are omitted; route partitioning is handled by terminal
    Split decoding.
    """

    def __init__(self, **model_params):
        super().__init__()
        node_feature_dim = model_params.get("node_feature_dim")
        if node_feature_dim not in (3, 6):
            raise ValueError(
                "node_feature_dim must be 3 for CVRP or 6 for CVRPTW"
            )
        self.model_params = model_params
        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        node_features = [reset_state.node_xy, reset_state.node_demand[:, :, None]]
        if self.model_params["node_feature_dim"] == 6:
            required = ("node_service_time", "node_tw_start", "node_tw_end")
            missing = [name for name in required if not hasattr(reset_state, name)]
            if missing:
                raise ValueError(
                    "CVRPTW customer features are missing from reset state: {}".format(
                        ", ".join(missing)
                    )
                )
            node_features.extend(
                getattr(reset_state, name)[:, :, None] for name in required
            )
        customer_features = torch.cat(node_features, dim=2)
        self.encoded_nodes = self.encoder(reset_state.depot_xy, customer_features)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size, pomo_size = state.BATCH_IDX.shape

        if state.current_node is None:
            # Node 0 is the depot. The POMO starts are customers 1..pomo_size.
            selected = torch.arange(
                1, pomo_size + 1, device=state.BATCH_IDX.device, dtype=torch.long
            )[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(batch_size, pomo_size, device=state.BATCH_IDX.device)
            return selected, prob

        encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
        if self.model_params["node_feature_dim"] == 6:
            probs = self.decoder(
                encoded_last, state.load, state.ninf_mask,
                current_time=state.current_time,
            )
        else:
            probs = self.decoder(encoded_last, state.load, state.ninf_mask)

        if self.training or self.model_params["eval_type"] == "softmax":
            while True:
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                    selected = selected.squeeze(1).reshape(batch_size, pomo_size)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
                if (prob > 0).all():
                    break
        else:
            selected = probs.argmax(dim=2)
            prob = None
        return selected, prob


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        node_feature_dim = model_params["node_feature_dim"]
        # The depot has coordinates only; customer features match the Direct
        # encoder for the selected problem class.
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(node_feature_dim, embedding_dim)
        self.layers = nn.ModuleList(
            EncoderLayer(**model_params) for _ in range(model_params["encoder_layer_num"])
        )

    def forward(self, depot_xy, node_features):
        out = torch.cat(
            (self.embedding_depot(depot_xy), self.embedding_node(node_features)), dim=1
        )
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.norm1 = AddAndInstanceNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, model_params["ff_hidden_dim"])
        self.norm2 = AddAndInstanceNorm(embedding_dim)

    def forward(self, inputs):
        q = _reshape_by_heads(self.Wq(inputs), self.head_num)
        k = _reshape_by_heads(self.Wk(inputs), self.head_num)
        v = _reshape_by_heads(self.Wv(inputs), self.head_num)
        attention_out = self.combine(_multi_head_attention(q, k, v))
        out1 = self.norm1(inputs, attention_out)
        return self.norm2(out1, self.ff(out1))


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.head_num = head_num
        dynamic_dim = 2 if model_params["node_feature_dim"] == 6 else 1
        self.Wq_last = nn.Linear(
            embedding_dim + dynamic_dim, head_num * qkv_dim, bias=False
        )
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        self.k = _reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        self.v = _reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last, load, ninf_mask, current_time=None):
        context_parts = [encoded_last, load[:, :, None]]
        if self.model_params["node_feature_dim"] == 6:
            if current_time is None:
                raise ValueError("CVRPTW decoder requires current_time")
            context_parts.append(current_time[:, :, None])
        context = torch.cat(context_parts, dim=2)
        q_last = _reshape_by_heads(self.Wq_last(context), self.head_num)
        out = _multi_head_attention(
            q_last, self.k, self.v, rank3_ninf_mask=ninf_mask
        )
        mh_out = self.combine(out)
        score = torch.matmul(mh_out, self.single_head_key)
        score = score / self.model_params["sqrt_embedding_dim"]
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        return F.softmax(score + ninf_mask, dim=2)


def _get_encoding(encoded_nodes, node_indices):
    embedding_dim = encoded_nodes.size(2)
    gather_index = node_indices[:, :, None].expand(-1, -1, embedding_dim)
    return encoded_nodes.gather(1, gather_index)


def _reshape_by_heads(qkv, head_num):
    batch_size, node_count, _ = qkv.shape
    return qkv.reshape(batch_size, node_count, head_num, -1).transpose(1, 2)


def _multi_head_attention(q, k, v, rank3_ninf_mask=None):
    key_dim = q.size(3)
    score = torch.matmul(q, k.transpose(2, 3)) / (key_dim ** 0.5)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :]
    weights = F.softmax(score, dim=3)
    out = torch.matmul(weights, v).transpose(1, 2)
    return out.reshape(out.size(0), out.size(1), -1)


class AddAndInstanceNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, first, second):
        return self.norm((first + second).transpose(1, 2)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, inputs):
        return self.W2(F.relu(self.W1(inputs)))
