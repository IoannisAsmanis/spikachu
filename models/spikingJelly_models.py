import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate
import hydra
from omegaconf import DictConfig, OmegaConf
from models.spiking_attention import SSABlock, LTLIFNode
from einops.layers.torch import Rearrange
from einops import rearrange


def pick_surrogate(surrName):
    if surrName == "atan":
        surr = surrogate.ATan()
    elif surrName == "sigmoid":
        surr = surrogate.Sigmoid()
    else:
        raise NotImplementedError("Only supports 'atan' and 'sigmoid' surrogate")
    return surr


class ConnectionBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        hidden_size: int,
        n_layers: int = 2,
        skip_connect: bool = False,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        """
        :param tau: Initial value of neuron's membrane potential time constant (it is learnable)
        :param hidden_size: Number of neurons in hidden layer(s)
        :param n_layers: Number of layers
        :param skip_connect: If true, skip connects input to output layer
        Note: The input will be added to the output after propagation though all the layers.
        """
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        process = [
            LTLIFNode(
                init_tau=tau,
                surrogate_function=surr,
                step_mode="m",
                backend=backend,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=False,
            ),
            # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surr, step_mode='m', backend=backend, v_threshold=1.),
            layer.Linear(hidden_size, hidden_size, step_mode="m"),
            Rearrange("t b c -> t b c 1"),
            layer.BatchNorm1d(hidden_size, step_mode="m"),
            Rearrange("t b c 1 -> t b c"),
            layer.Dropout(p=p_drop, step_mode="m"),
        ] * (n_layers - 1)

        out = [
            LTLIFNode(
                init_tau=tau,
                surrogate_function=surr,
                step_mode="m",
                backend=backend,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=False,
            ),
            # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surr, step_mode='m', backend=backend, v_threshold=1.),
            layer.Linear(hidden_size, hidden_size, step_mode="m"),
            Rearrange("t b c -> t b c 1"),
            layer.BatchNorm1d(hidden_size, step_mode="m"),
            Rearrange("t b c 1 -> t b c"),
        ]

        self.skip_connection = skip_connect
        self.net = nn.Sequential(*(process + out))

    def forward(self, x):
        """
        :param x: Time x Batch x Channel
        :return: Time x Batch x Channel
        """
        return self.net(x) + x if self.skip_connection else self.net(x)

    def get_spike_params(self):
        net_params = []
        for module in self.net:
            if isinstance(module, LTLIFNode):
                net_params.append(module.get_spike_params())
        # num LIF x 2
        return np.stack(net_params, axis=0)


class FeedForwardSNN(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 2,
        skip_step: int = None,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        """
        :param tau: Initial value of neuron's membrane potential time constant (it is learnable)
        :param input_size: Number of neurons in the input layer
        :param hidden_size: Number of neurons in the hidden layer(s)
        :param output_size: Number of neurons in the output layer
        :param n_layers: Number of hidden layers
        :param skip_step: Specifies the number of layers between skip connections in the hidden layers
        """
        super().__init__()

        if surrogate_func == "atan":
            surr = surrogate.ATan()
        elif surrogate_func == "sigmoid":
            surr = surrogate.Sigmoid()
        else:
            raise NotImplementedError("Only supports 'atan' and 'sigmoid' surrogate")

        if n_layers == 0:
            self.net = nn.Sequential()

        elif n_layers == 1:
            assert (
                skip_step is None and hidden_size is None
            ), "n_layers = 1 -> Skip_step and hidden size must be set to None."
            self.net = nn.Sequential(
                *[
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surr, step_mode='m', backend=backend, v_threshold=1.),
                    layer.Linear(input_size, output_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(output_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
            )

        else:
            if (
                input_size != hidden_size
                or input_size != output_size
                or hidden_size != output_size
            ):
                assert (
                    n_layers >= 2
                ), "n_layers >= 2 since input_size != hidden_size != output_size."
                n_layers -= 2
                input_block = [
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surr, step_mode='m', backend=backend, v_threshold=1.),
                    layer.Linear(input_size, hidden_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(hidden_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
                output_block = [
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    # neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surr, step_mode='m', backend=backend, v_threshold=1.),
                    layer.Linear(hidden_size, output_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(output_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
            else:
                input_block = output_block = []

            if skip_step is None:  # No skip connections
                hidden_block = [
                    ConnectionBlock(
                        tau,
                        v_threshold,
                        v_reset,
                        hidden_size,
                        n_layers=n_layers,
                        skip_connect=False,
                        p_drop=p_drop,
                        backend=backend,
                    )
                ]
            else:
                assert (
                    n_layers % skip_step == 0
                ), "Error in FeedForwardSNN: n_layers must be divisible by skip_step"

                hidden_block = [
                    ConnectionBlock(
                        tau,
                        v_threshold,
                        v_reset,
                        hidden_size,
                        n_layers=skip_step,
                        skip_connect=True,
                        p_drop=p_drop,
                        backend=backend,
                    )
                    for _ in range(n_layers // skip_step)
                ]

            self.net = nn.Sequential(*(input_block + hidden_block + output_block))

    def forward(self, x):
        """
        :param x: Time x Batch x Channel
        :return:  Time x Batch x Channel
        """
        return self.net(x)

    def get_spike_params(self):
        all_params = []
        for module in self.net:
            if isinstance(module, LTLIFNode):
                ltlif_params = module.get_spike_params()
                all_params.append(ltlif_params[None, ...])
            elif isinstance(module, ConnectionBlock):
                cb_params = module.get_spike_params()
                all_params.append(cb_params)

        # num LIFs x 2
        all_params = np.concatenate(all_params, axis=0)
        return all_params


class MultiScaleSNN(nn.Module):
    def __init__(
        self,
        list_of_taus: list[float],
        v_threshold: float,
        v_reset: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 2,
        skip_step: int = 2,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        """
        :param list_of_taus: Initial value of neuron's membrane potential time constant (it is learnable) for each branch.
        :param input_size: Number of neurons in the input layer (same for each branch)
        :param hidden_size: Number of neurons in the hidden layer(s) (same for each branch)
        :param output_size: Number of neurons in the output layer that combines info across branches
        :param n_layers: Number of hidden layers
        :param skip_step: Specifies the number of layers between skip connection in the hidden layers
        """
        super().__init__()

        self.models = nn.ModuleList(
            FeedForwardSNN(
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                n_layers=n_layers,
                skip_step=skip_step,
                p_drop=p_drop,
                surrogate_func=surrogate_func,
                backend=backend,
            )
            for tau in list_of_taus
        )

        self.num_scales = len(list_of_taus)

    def forward(self, x):
        """
        :param x: Time x Batch x Channel
        :return: Time x Batch x Channel
        """
        # if x.dim() == 3:
        assert x.dim() == 3
        features = [torch.jit.fork(model, x) for model in self.models]
        outputs = torch.stack([torch.jit.wait(f) for f in features], dim=-2)

        # elif x.dim() == 4:
        #     features = [torch.jit.fork(model, x[:, :, scale, :]) for model, scale in zip(self.models, range(self.num_scales))]

        return outputs

    def get_spike_params(self):
        # return n_taus x n_LIF modules per branch x n_params
        all_params = []
        for model in self.models:
            all_params.append(model.get_spike_params())
        return np.stack(all_params, axis=0)


class SpikingNeuralNet(nn.Module):
    def __init__(
        self,
        cfg,
        # list_of_taus_ms: list[float],
        # input_size_ms: int,
        # hidden_size_ms: int,
        # n_layers_ms: int,
        # skip_step_ms: int,
        # tau_ffn: float,
        # hidden_size_ffn: int,
        # n_layers_ffn: int,
        # skip_step_ffn: int,
        # list_of_taus_integrator: list[float],
        # hidden_size_integrator: int,
        # n_layers_integrator: int,
        # skip_step_integrator: int,
        # output_tau: float,
        # output_size: int
    ):
        """
        :param cfg:
            cfg.model.setup.multi_scale_snn.list_of_taus_ms: List of taus for multi-scale snn
            cfg.model.setup.multi_scale_snn.input_size: Input size for multi-scale snn
            cfg.model.setup.multi_scale_snn.hidden_size: Hidden size for multi-scale snn
            cfg.model.setup.multi_scale_snn.output_size: Output size for multi-scale snn
            cfg.model.setup.multi_scale_snn.n_layers: Number of hidden layers for each branch of the multi-scale snn
            cfg.model.setup.multi_scale_snn.skip_step: Skip step between layers for each branch of the multi-scale snn

            cfg.model.setup.feed_forward.tau: Initial value of membrane potential for all neurons of the FeedForward SNN
            fg.model.setup.feed_forward.input_size: Input size for FeedForward SNN (same as output size of Multi-scale snn)
            cfg.model.setup.feed_forward.hidden_size: Hidden size for FeedForward SNN
            cfg.model.setup.feed_forward.output_size: Output size for FeedForward SNN
            cfg.model.setup.feed_forward.n_layers: Number of hidden layers for each branch of the FeedForward SNN
            cfg.model.setup.feed_forward.skip_step: Skip step between layers of the FeedForward SNN

            cfg.model.setup.integrator.list_of_taus: List of taus for integrator block (another multi-scale snn) querr
            cfg.model.setup.integrator.input_size: Input size for integrator block (same as output of FeedForward SNN)
            cfg.model.setup.integrator.hidden_size: Number of neurons for each hidden layer of each branch of the integrator.
            cfg.model.setup.integrator.output_size: Number of neurons for the output layer across branches of th integrator.
            cfg.model.setup.integrator.n_layers: Number of hidden layers for each branch of the integrator.
            cfg.model.setup.integrator.skip_step: Skip step between layers for each branch of the integrator.
        """
        super().__init__()

        surr = pick_surrogate(cfg.model.setup.surrogate)

        if "multi_scale_snn" in cfg.model.setup:
            self.multi_scale = MultiScaleSNN(
                list_of_taus=cfg.model.setup.multi_scale_snn.list_of_taus_ms,
                v_threshold=cfg.model.setup.multi_scale_snn.v_threshold,
                v_reset=cfg.model.setup.multi_scale_snn.v_reset,
                input_size=cfg.model.setup.multi_scale_snn.input_size,
                hidden_size=cfg.model.setup.multi_scale_snn.hidden_size,
                output_size=cfg.model.setup.multi_scale_snn.output_size,
                n_layers=cfg.model.setup.multi_scale_snn.n_layers,
                skip_step=cfg.model.setup.multi_scale_snn.skip_step,
                p_drop=cfg.model.setup.multi_scale_snn.p_drop,
                surrogate_func=cfg.model.setup.surrogate,
                backend=cfg.model.setup.backend,  # change for trainable thresholds
            )
        else:
            self.multi_scale = nn.Sequential()

        if "SSA" in cfg.model.setup:
            self.stacked_SSAs = nn.Sequential(
                *[
                    SSABlock(
                        input_dim=cfg.model.setup.SSA.input_dim,  # dim,
                        qk_out_dim=cfg.model.setup.SSA.qk_out_dim,
                        v_out_dim=cfg.model.setup.SSA.v_out_dim,
                        output_dim=cfg.model.setup.SSA.output_dim,
                        head_dim=cfg.model.setup.SSA.head_dim,
                        mlp_ratio=cfg.model.setup.SSA.mlp_ratio,
                        qkv_bias=cfg.model.setup.SSA.qkv_bias,
                        qk_scale=cfg.model.setup.SSA.qk_scale,
                        lin_drop=cfg.model.setup.SSA.lin_drop,
                        attn_drop=cfg.model.setup.SSA.attn_drop,
                        surrogate_func=cfg.model.setup.surrogate,
                        backend=cfg.model.setup.backend,
                    )
                    for _ in range(cfg.model.setup.SSA.n_blocks)
                ]
            )
        else:
            self.stacked_SSAs = nn.Sequential()

        if "feed_forward" in cfg.model.setup:
            self.feed_forward = FeedForwardSNN(
                tau=cfg.model.setup.feed_forward.tau,
                v_threshold=cfg.model.setup.feed_forward.v_threshold,
                v_reset=cfg.model.setup.feed_forward.v_reset,
                input_size=cfg.model.setup.feed_forward.input_size,
                hidden_size=cfg.model.setup.feed_forward.hidden_size,
                output_size=cfg.model.setup.feed_forward.output_size,
                n_layers=cfg.model.setup.feed_forward.n_layers,
                skip_step=cfg.model.setup.feed_forward.skip_step,
                p_drop=cfg.model.setup.feed_forward.p_drop,
                surrogate_func=cfg.model.setup.surrogate,
                backend=cfg.model.setup.backend,
            )
        else:
            self.feed_forward = nn.Sequential()

        if "integrator" in cfg.model.setup:
            self.integrator = MultiScaleSNN(
                list_of_taus=cfg.model.setup.integrator.list_of_taus,
                v_threshold=cfg.model.setup.integrator.v_threshold,
                v_reset=cfg.model.setup.integrator.v_reset,
                input_size=cfg.model.setup.integrator.input_size,
                hidden_size=cfg.model.setup.integrator.hidden_size,
                output_size=cfg.model.setup.integrator.output_size,
                n_layers=cfg.model.setup.integrator.n_layers,
                skip_step=cfg.model.setup.integrator.skip_step,
                p_drop=cfg.model.setup.integrator.p_drop,
                surrogate_func=cfg.model.setup.surrogate,
                backend=cfg.model.setup.backend,
            )
        else:
            self.integrator = nn.Sequential()

        self.output_read = neuron.ParametricLIFNode(
            init_tau=cfg.model.setup.output_tau,
            surrogate_function=surr,
            step_mode="m",
            v_reset=None,
            v_threshold=0.0,
            backend=cfg.model.setup.backend,
            store_v_seq=True,
            decay_input=False,
        )

        self.output_proj = layer.Linear(
            cfg.model.setup.pre_output_size, cfg.model.setup.output_size, step_mode="m"
        )

    def get_average_spiking_rate(self):
        all_srs = 0.0
        n_srs = 0
        for name, module in self.named_modules():
            if isinstance(module, LTLIFNode):
                spiking_rate = module.spike_rate  # contains grad hopefully
                all_srs += spiking_rate
                n_srs += 1
        return all_srs / n_srs  # mean of means logic, not perfect but whatever...

    def forward(self, x):
        """
        :param x: Time x Batch x Channel
        :return: Time x Batch x Velocities
        """

        assert x.dim() == 3
        x = self.multi_scale(x)

        # x = rearrange(x, 't b f -> t b 1 f')
        # x = self.conv_snn(x)

        assert x.dim() == 4
        x = self.stacked_SSAs(x)

        # assert x.dim() == 4
        # x = self.pooling(x)

        assert x.dim() == 4
        x = rearrange(x, "t b c f -> t b (c f)")
        x = self.feed_forward(x)

        assert x.dim() == 3
        x = self.integrator(x)
        x = rearrange(x, "t b c f -> t b (c f)")

        self.output_read(x)
        return self.output_proj(self.output_read.v_seq)

    def get_spike_params(self):
        ffn_sp = self.feed_forward.get_spike_params()  # (n_layers_ff, 2,)
        ms_sp = self.multi_scale.get_spike_params()  # (n_taus_ms, n_layers, 2)
        int_sp = self.integrator.get_spike_params()  # (n_taus_int, n_layers_int, 2)
        return ffn_sp, ms_sp, int_sp


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="configs_spikingjelly_snn.yaml",
)
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SpikingNeuralNet(
        cfg=cfg
        # list_of_taus_ms = [2.0, 4.0],
        # input_size_ms = 32,
        # hidden_size_ms = 64,
        # n_layers_ms = 2,
        # skip_step_ms = 2,
        # tau_ffn = 2.0,
        # hidden_size_ffn = 64,
        # n_layers_ffn = 2,
        # skip_step_ffn = 2,
        # list_of_taus_integrator = [2.0, 4.0],
        # hidden_size_integrator = 32,
        # n_layers_integrator = 2,
        # skip_step_integrator = 2,
        # output_tau = 2.0,
        # output_size = 2
    ).to(device)

    randx = torch.randn([10, 4, 128]).to(device)

    out = net(randx)

    print(out.size())
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")


if __name__ == "__main__":
    main()
