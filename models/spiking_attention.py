from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate
from einops import rearrange
from einops.layers.torch import Rearrange

# backend = 'cupy'  # 'torch'
# surr = surrogate.ATan()  # surrogate.Sigmoid()


class LTLIFNode(neuron.ParametricLIFNode):
    """
    Learnable Voltage Threshold LIF Node
    """

    def __init__(self, v_threshold=1.0, v_threshold_min=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
        self.v_threshold_min = v_threshold_min
        self.spike_rate = None

    def forward(self, x: torch.Tensor):
        self.v_threshold.data.clamp_(min=self.v_threshold_min)
        spikes_out = super().forward(x)
        self.spike_rate = (spikes_out > 0).float().mean()  # retains gradient, I hope...
        return spikes_out

    def get_spike_params(self):
        thresh = self.v_threshold.item()
        tau = (1.0 / self.w.sigmoid()).item()
        return np.array([thresh, tau])

    def get_explicit_neuron_count(self):
        # for the architectural context, a shape is implicitly defined
        # excluding batch dimension, product of all other dimensions
        # is exactly the number of neurons present in the module
        if hasattr(self.v, "shape"):
            shape_tensor = torch.tensor(self.v.shape)
            return torch.prod(shape_tensor[1:]).item()
        else:
            return 1


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        if surrogate_func == "atan":
            surr = surrogate.ATan()
        elif surrogate_func == "sigmoid":
            surr = surrogate.Sigmoid()
        else:
            raise NotImplementedError("Only supports 'atan' and 'sigmoid' surrogate")

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_linear = layer.Linear(
            in_features, hidden_features, step_mode="m"
        )  # nn.Linear(in_features, hidden_features)
        self.fc1_bn = layer.BatchNorm1d(
            hidden_features, step_mode="m"
        )  # nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )

        self.fc2_linear = layer.Linear(
            hidden_features, out_features, step_mode="m"
        )  # nn.Linear(hidden_features, out_features)
        self.fc2_bn = layer.BatchNorm1d(
            out_features, step_mode="m"
        )  # nn.BatchNorm1d(out_features)
        self.fc2_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )

        self.c_hidden = hidden_features
        self.c_output = out_features

        self.dropout = layer.Dropout(drop, step_mode="m")
        self.dropout = layer.Dropout(drop, step_mode="m")

    def forward(self, x):
        t, b, c, f = x.shape
        # x_ = x.flatten(0, 1)
        x = rearrange(x, "t b c f -> (t b) c f")
        x = self.fc1_linear(x)  # self.fc1_linear(x_)
        x = rearrange(x, "t b c -> t b c 1")
        x = self.fc1_bn(
            x
        )  # self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = rearrange(x, "t b c 1 -> t b c")
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)  # self.fc2_linear(x.flatten(0,1))
        x = rearrange(x, "t b c -> t b c 1")
        x = self.fc2_bn(
            x
        )  # self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = rearrange(x, "t b c 1 -> t b c")
        x = self.fc2_lif(x)
        x = rearrange(x, "(t b) c f -> t b c f", b=b, t=t)
        return self.dropout(x)

    def get_spike_params(self):
        p1 = self.fc1_lif.get_spike_params()
        p2 = self.fc2_lif.get_spike_params()
        return np.stack([p1, p2], axis=0)


class SSA(nn.Module):
    def __init__(
        self,
        dim,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        if surrogate_func == "atan":
            surr = surrogate.ATan()
        elif surrogate_func == "sigmoid":
            surr = surrogate.Sigmoid()
        else:
            raise NotImplementedError("Only supports 'atan' and 'sigmoid' surrogate")

        if qk_out_dim is None:
            qk_out_dim = dim
        if v_out_dim is None:
            v_out_dim = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        # self.dim = dim
        self.head_dim = head_dim
        self.scale = qk_scale if qk_scale is not None else (head_dim**-0.5)

        self.q_linear = layer.Linear(
            dim, qk_out_dim, step_mode="m", bias=qkv_bias
        )  # nn.Linear(dim, dim)
        self.q_bn = layer.BatchNorm1d(qk_out_dim, step_mode="m")  # nn.BatchNorm1d(dim)
        self.q_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )

        self.k_linear = layer.Linear(
            dim, qk_out_dim, step_mode="m", bias=qkv_bias
        )  # nn.Linear(dim, dim)
        self.k_bn = layer.BatchNorm1d(qk_out_dim, step_mode="m")  # nn.BatchNorm1d(dim)
        self.k_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )

        self.v_linear = layer.Linear(
            dim, v_out_dim, step_mode="m", bias=qkv_bias
        )  # nn.Linear(dim, dim)
        self.v_bn = layer.BatchNorm1d(v_out_dim, step_mode="m")  # nn.BatchNorm1d(dim)
        self.v_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )

        self.attn_lif = LTLIFNode(
            init_tau=2.0, surrogate_function=surr, step_mode="m", backend=backend
        )
        self.attn_dropout = layer.Dropout(attn_drop, step_mode="m")

        self.proj_linear = layer.Linear(
            v_out_dim, v_out_dim, step_mode="m"
        )  # nn.Linear(dim, dim)
        self.proj_bn = layer.BatchNorm1d(
            v_out_dim, step_mode="m"
        )  # nn.BatchNorm1d(dim)
        self.proj_lif = LTLIFNode(
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
        )

        assert (
            qk_out_dim % head_dim == 0
        ), f"dim {dim} should be divided by num_heads {head_dim}."
        self.num_heads = qk_out_dim // self.head_dim

    def transform_for_scores(self, x: torch.Tensor, head_dim: int):
        # (..., seq_len, dim) -> (..., n_heads, seq_len, head_dim)
        *dims, seq, hid = x.size()
        x = x.view(*dims, seq, self.num_heads, head_dim)
        return x.transpose(-3, -2)

    def forward(self, x):
        t, b, c, f = x.shape

        # assert c % self.dim == 0  # Need to check that

        # x = rearrange(x, "t b (p d) -> (t b) p d", d=self.dim)  # Separate the values into patches of size self.dim
        x = rearrange(x, "t b c f -> (t b) c f")

        # q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
        q_linear_out = self.q_linear(x)  # t b p d_q
        q_linear_out = rearrange(q_linear_out, "t b c -> t b c 1")
        q_linear_out = self.q_bn(q_linear_out)  # t b p d_q
        q_linear_out = rearrange(q_linear_out, "t b c 1 -> t b c")
        q_linear_out = self.q_lif(q_linear_out)  # t b p d_q
        # q = q_linear_out.reshape(TB, N, self.num_heads, C // self.num_heads)  # q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        q = self.transform_for_scores(q_linear_out, self.head_dim)
        # q = rearrange(q_linear_out, "(t) c (h f) -> (t) h c f", f=self.head_dim)  # Separate the values into p patches

        # k_linear_out = rearrange(x, "t b c f -> (t b) c f")
        k_linear_out = self.k_linear(x)
        k_linear_out = rearrange(k_linear_out, "t b c -> t b c 1")
        k_linear_out = self.k_bn(
            k_linear_out
        )  # self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = rearrange(k_linear_out, "t b c 1 -> t b c")
        k_linear_out = self.k_lif(k_linear_out)
        # k = k_linear_out.reshape(TB, N, self.num_heads, C // self.num_heads)  # k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # k = rearrange(k_linear_out, "(t) c (h f) -> (t) h c f", f=self.head_dim)  # Separate the values into p patches
        k = self.transform_for_scores(k_linear_out, self.head_dim)

        # v_linear_out = rearrange(x, "t b c f -> (t b) c f")
        v_linear_out = self.v_linear(x)
        v_linear_out = rearrange(v_linear_out, "t b c -> t b c 1")
        v_linear_out = self.v_bn(
            v_linear_out
        )  # self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = rearrange(v_linear_out, "t b c 1 -> t b c")
        v_linear_out = self.v_lif(v_linear_out)
        # v = v_linear_out.reshape(TB, N, self.num_heads, C // self.num_heads)  # v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # v = rearrange(v_linear_out, "(t) c (h f) -> (t) h c f", f=self.head_dim)  # Separate the values into p patches
        v = self.transform_for_scores(v_linear_out, self.head_dim)

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = self.attn_dropout(attn)

        weighted = attn @ v

        # x = rearrange(x, "(t) h c d -> (t) c (h d)")

        *dims, n_heads, seq, hid = weighted.size()
        weighted = weighted.transpose(-3, -2)
        weighted = weighted.reshape(*dims, seq, n_heads * hid)

        # x = x.reshape(TB, N, C).contiguous()
        weighted = self.attn_lif(weighted)
        # x = x.flatten(0, 1)
        weighted = self.proj_linear(weighted)
        weighted = rearrange(weighted, "t b c -> t b c 1")
        weighted = self.proj_bn(weighted)
        weighted = rearrange(weighted, "t b c 1 -> t b c")
        weighted = self.proj_lif(weighted)
        # weighted = self.proj_lif(self.proj_bn(self.proj_linear(weighted)))  # self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        x = rearrange(x, "(t b) c f -> t b c f", t=t, c=c)
        return x

    def get_spike_params(self):
        pass


class SSABlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        head_dim: int = 64,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        lin_drop=0.0,
        attn_drop=0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        # assert input_dim % feature_dim == 0
        # assert feature_dim % head_dim == 0
        assert output_dim is None or output_dim == input_dim

        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = SSA(
            input_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=output_dim,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            surrogate_func=surrogate_func,
            backend=backend,
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.mlp = MLP(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            drop=lin_drop,
            surrogate_func=surrogate_func,
            backend=backend,
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


if __name__ == "__main__":
    ssa = SSABlock(
        input_dim=1152,
        feature_dim=128,
        head_dim=64,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        lin_drop=0.0,
        attn_drop=0.0,
        surrogate_func="atan",
        backend="torch",
    ).to(torch.device("cpu"))

    randmat = torch.randn(100, 4, 1152).to(torch.device("cpu"))

    out = ssa(randmat)
