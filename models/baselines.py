import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import hydra


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.25, nonlinearity="ReLU"):
        super().__init__()
        self.nonlinearity = getattr(nn, nonlinearity, None)
        if self.nonlinearity is None:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            self.nonlinearity(),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.model(x)


class ConfigurableMLP(nn.Module):
    def __init__(self, dims, p_drop=0.25, nonlinearity="ReLU"):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(
                LinearBlock(dims[i], dims[i + 1], p_drop, nonlinearity=nonlinearity)
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Wiener(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.LazyLinear(output_size),
        )

    def forward(self, x):
        return self.model(x)


class RecurrentBlock(nn.Module):
    def __init__(
        self,
        rec_type,
        input_dim,
        hidden_dim,
        n_rec_layers,
        output_dim,
        p_drop=0.25,
        bidirectional=False,
    ):
        super().__init__()

        self.rec_type = rec_type

        rec_mod = getattr(nn, rec_type.upper(), None)
        if rec_mod is None:
            raise ValueError(f"Unsupported recurrent type: {rec_type}")

        self.rec_core = rec_mod(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_rec_layers,
            batch_first=True,
            dropout=p_drop,
            bidirectional=bidirectional,
        )

        self.readout = nn.Sequential(
            nn.Linear(
                hidden_dim * (2 if bidirectional else 1),
                output_dim,
            ),
            nn.Dropout(p_drop),
        )

    def forward(
        self, x, past_hidden_state=None, return_hidden_state=False, last_time_only=True
    ):
        out, hn = self.rec_core(x, past_hidden_state)
        if last_time_only:
            out = out[:, -1, :]
        out = self.readout(out)
        if return_hidden_state:
            return out, hn
        else:
            return out

    @property
    def device(self):
        return next(self.parameters()).device

    def init_hidden_state(self, batch_size):
        if self.rec_type == "LSTM":
            # LSTM hidden state is a tuple of (h_n, c_n)
            return (
                torch.zeros(
                    self.rec_core.num_layers,
                    batch_size,
                    self.rec_core.hidden_size,
                ).to(self.device),
                torch.zeros(
                    self.rec_core.num_layers,
                    batch_size,
                    self.rec_core.hidden_size,
                ).to(self.device),
            )
        else:
            # apparently always batch second
            return torch.zeros(
                self.rec_core.num_layers, batch_size, self.rec_core.hidden_size
            ).to(self.device)


class ConfigurableRecurrentModel(nn.Module):
    def __init__(self, n_blocks, block_cfg):
        super().__init__()

        self.n_blocks = n_blocks
        self.block_cfg = block_cfg

        blocks = []
        interm_block_cfg = block_cfg.copy()
        interm_block_cfg["output_dim"] = block_cfg["input_dim"]
        for i in range(n_blocks - 1):
            blocks.append(RecurrentBlock(**interm_block_cfg))
        blocks.append(RecurrentBlock(**block_cfg))  # last block with final output dim
        self.model = nn.ModuleList(blocks)

        self.rec_type = block_cfg["rec_type"]

    def forward(self, x, past_hidden_state=None):
        new_hidden = []

        # loop over blocks in the architecture
        for i in range(self.n_blocks):

            # recover hidden state
            if past_hidden_state is not None and isinstance(past_hidden_state, tuple):
                # LSTM hidden state is a tuple of (h_n, c_n)
                phs = (past_hidden_state[0][i], past_hidden_state[1][i])
            elif past_hidden_state is not None:
                phs = past_hidden_state[i]
            else:
                phs = None

            # forward pass for each block
            x, nh = self.model[i](
                x, phs, return_hidden_state=True, last_time_only=False
            )

            # store hidden state
            new_hidden.append(nh)
        return x, new_hidden

    def init_hidden_state(self, batch_size):
        if self.rec_type == "LSTM":
            # LSTM hidden state is a tuple of (h_n, c_n)
            all_hidden_tuples = [
                self.model[i].init_hidden_state(batch_size)
                for i in range(self.n_blocks)
            ]
            h_n = torch.stack([h[0] for h in all_hidden_tuples], dim=0)
            c_n = torch.stack([h[1] for h in all_hidden_tuples], dim=0)
            return (h_n, c_n)
        else:
            hidden_state_list = [
                self.model[i].init_hidden_state(batch_size)
                for i in range(self.n_blocks)
            ]
            return torch.stack(hidden_state_list, dim=0)


# def main():
#     # sample configurations with approximately 3M trainable params

#     mlp = ConfigurableMLP(dims=[100, 256, 1024, 1024, 1024, 512, 256, 2], p_drop=0.1)
#     print(f"MLP trainable parameters: {count_trainable_parameters(mlp)}")

#     n_blocks = 6
#     block_cfg = {
#         "rec_type": "RNN",
#         "input_dim": 100,
#         "hidden_dim": 242,
#         "n_rec_layers": 5,
#         "output_dim": 2,
#         "p_drop": 0.1,
#     }
#     rnn = ConfigurableRecurrentModel(n_blocks, block_cfg)

#     n_blocks = 4
#     block_cfg = {
#         "rec_type": "GRU",
#         "input_dim": 100,
#         "hidden_dim": 184,
#         "n_rec_layers": 4,
#         "output_dim": 2,
#         "p_drop": 0.1,
#     }
#     gru = ConfigurableRecurrentModel(n_blocks, block_cfg)

#     n_blocks = 4
#     block_cfg = {
#         "rec_type": "LSTM",
#         "input_dim": 100,
#         "hidden_dim": 162,
#         "n_rec_layers": 4,
#         "output_dim": 2,
#         "p_drop": 0.1,
#     }
#     lstm = ConfigurableRecurrentModel(n_blocks, block_cfg)

#     print(f"RNN trainable parameters: {count_trainable_parameters(rnn)}")
#     print(f"GRU trainable parameters: {count_trainable_parameters(gru)}")
#     print(f"LSTM trainable parameters: {count_trainable_parameters(lstm)}")

#     dummy_data = torch.randn(32, 10, 100)  # batch_size=32, seq_length=10, input_dim=100

#     with torch.no_grad():
#         lin_out = mlp(dummy_data)
#         print(f"MLP output shape: {lin_out.shape}")

#         recs = {"RNN": rnn, "GRU": gru, "LSTM": lstm}
#         for name, model in recs.items():
#             hidden_state = model.init_hidden_state(dummy_data.size(0))
#             output, new_hidden = model(dummy_data, hidden_state)
#             nhs = new_hidden[0]

#             lstm_flag = False
#             if isinstance(nhs, tuple):
#                 nhs = nhs[0]  # lstm
#                 lstm_flag = True
#             print(
#                 f"{name} output shape: {output.shape}, new hidden state shape: "
#                 f"{'' if not lstm_flag else '2 x '}{nhs.shape} x {len(new_hidden)}"
#             )


@hydra.main(
    config_path="../configs", config_name="configs_baselines.yaml", version_base=None
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    model_select = cfg.model.type + "_cfg"
    init_cfg = cfg.model.setup[model_select]
    model = hydra.utils.instantiate(init_cfg)
    print(f"Model type: {type(model)}")
    print(f"Model trainable parameters: {count_trainable_parameters(model)}")

    # Dummy data for testing
    dummy_data = torch.randn(32, 10, 100)
    dummy_labels = torch.randn(32, 10, 2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    optimizer.zero_grad()

    outputs = model(
        dummy_data,
    )
    if isinstance(outputs, tuple):
        outputs = outputs[
            0
        ]  # disregard hidden state for loss calculation in recurrent models

    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    print(f"Dummy data shape: {dummy_data.shape}")
    print(f"Dummy labels shape: {dummy_labels.shape}")
    print(f"Model output shape: {outputs.shape}")
    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    main()
