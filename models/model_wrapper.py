from tokenize import group
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
import collections
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
import time
from einops import rearrange
from poyo.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    compute_loss_or_metric,
)
from poyo.data import pad, chain, track_mask, track_batch
from poyo.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
)
from models.spikingJelly_models import SpikingNeuralNet
from models import baselines
from models.baselines import ConfigurableMLP, ConfigurableRecurrentModel, Wiener
from poyo.taxonomy import Task, OutputType
from poyo.taxonomy.task import REACHING
import sys
from .homogenizer import PoyoHomogenizer, CausalHomogenizer
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model_wrapper(nn.Module):
    def __init__(
        self,
        *,
        num_neurons,
        dim=128,
        num_latents=1000,
        emb_init_scale=0.02,
        cfg=None,
        init_embeddings=True,
    ):
        super().__init__()

        self.cfg = cfg
        self.delta_t_raw_forward = None  # to be initialized after forward pass

        if cfg is None:
            sys.exit(
                "Please specify the cfg based on which the SNN will be initialized."
            )
            sys.exit(
                "Please specify the cfg based on which the SNN will be initialized."
            )

        # cfg.model.setup.input_size = num_neurons

        # Initialize the Channel Homogenizer (Cross-Attention)
        if "PoyoHomogenizer" in cfg.model.setup:

            self.homogenizer = PoyoHomogenizer(
                dim=cfg.model.setup.input_size,
                context_dim=cfg.model.setup.PoyoHomogenizer.context_dim,
                dim_head=cfg.model.setup.PoyoHomogenizer.dim_head,
                # depth=0,
                cross_heads=cfg.model.setup.PoyoHomogenizer.cross_heads,
                # self_heads=8,
                ffn_dropout=cfg.model.setup.PoyoHomogenizer.ffn_dropout,
                lin_dropout=cfg.model.setup.PoyoHomogenizer.lin_dropout,
                atn_dropout=cfg.model.setup.PoyoHomogenizer.atn_dropout,
                use_memory_efficient_attn=cfg.model.setup.PoyoHomogenizer.use_memory_efficient_attn,
            )

            self.preprocess = self._process_with_poyo_homogenizer

        elif "CausalHomogenizer" in cfg.model.setup:

            self.homogenizer = CausalHomogenizer(
                num_latents=cfg.model.setup.CausalHomogenizer.num_latents,
                latent_dim=cfg.model.setup.CausalHomogenizer.latent_dim,
                input_dim=cfg.model.setup.CausalHomogenizer.input_dim,
                qk_out_dim=cfg.model.setup.CausalHomogenizer.qk_out_dim,
                v_out_dim=cfg.model.setup.CausalHomogenizer.v_out_dim,
                num_cross_attn_heads=cfg.model.setup.CausalHomogenizer.num_cross_attn_heads,
                cross_attn_widening_factor=cfg.model.setup.CausalHomogenizer.cross_attn_widening_factor,
                use_query_residual=cfg.model.setup.CausalHomogenizer.use_query_residual,
                cross_attn_lin_dropout=cfg.model.setup.CausalHomogenizer.cross_attn_lin_dropout,
                cross_attention_dropout=cfg.model.setup.CausalHomogenizer.cross_attention_dropout,
                num_virtual_channels=cfg.model.setup.CausalHomogenizer.num_virtual_channels,
            )

            self.preprocess = self._process_with_causal_homogenizer

            # self.num_neurons = int(num_neurons.iloc[0])

        else:
            try:
                if len(num_neurons) != 1:
                    sys.exit(
                        "Can only work with single-sessions when homogenization is not employed"
                    )
            except:
                pass
            self.num_neurons = int(num_neurons)

            cfg.model.setup.input_size = self.num_neurons
            if cfg.model.type.lower() == "mlp":
                cfg.model.setup.input_size *= cfg.model.setup.memory + 1

            self.preprocess = self._process_without_homogenizer

        if cfg.model.type == "SpikingJellySNN":
            self.model = SpikingNeuralNet(cfg)
            self._forward = self._forward_spiking_jelly
            self._apply_memory = self._apply_memory_snn_poyo
            self.get_spike_params = self.model.get_spike_params

        elif cfg.model.type.upper() in ["MLP", "GRU", "RNN", "LSTM"]:
            ## replace with new comparable baselines
            # self.model = MLP(
            #     output_size=cfg.model.setup.output_size,
            #     p_drop=cfg.model.setup.p_drop,
            # )
            model_select = cfg.model.type + "_cfg"
            init_cfg = cfg.model.setup[model_select]
            self.model = hydra.utils.instantiate(init_cfg)

            self._forward = (
                self._forward_mlp
                if cfg.model.type.upper() == "MLP"
                else self._forward_gru
            )
            self._apply_memory = (
                self._apply_memory_mlp_wiener
                if cfg.model.type.upper() == "MLP"
                else lambda x: x  # self._apply_memory_gru
            )

        # elif cfg.model.type.upper() in []:
        #     self.model = GRU(
        #         input_dim=cfg.model.setup.input_size,
        #         hidden_dim=cfg.model.setup.hidden_dim,  # 64,
        #         layer_dim=cfg.model.setup.layer_dim,  # 3,
        #         output_dim=cfg.model.setup.output_size,
        #         p_drop=cfg.model.setup.p_drop,
        #     )
        #     self._forward = self._forward_gru
        #     self._apply_memory = self._apply_memory_gru

        elif cfg.model.type == "Wiener":
            self.model = Wiener(cfg.model.setup.output_size)
            self._forward = self._forward_wiener
            self._apply_memory = self._apply_memory_mlp_wiener

        else:
            sys.exit(
                "Please specify model as 'MultiScaleSNN', 'MLP', 'GRU', or Wiener."
            )

        self.memory = (
            cfg.model.setup.memory if "memory" in cfg.model.setup.keys() else None
        )
        self.seq_length = None
        self.model_name = cfg.model.type

        if "CausalHomogenizer" in cfg.model.setup:
            self.latent_emb = Embedding(
                cfg.model.setup.CausalHomogenizer.num_latents,
                cfg.model.setup.CausalHomogenizer.latent_dim,
                init_scale=emb_init_scale,
            )
            self.unit_emb = InfiniteVocabEmbedding(
                cfg.model.setup.CausalHomogenizer.input_dim, init_scale=emb_init_scale
            )
            self.session_emb = InfiniteVocabEmbedding(
                cfg.model.setup.CausalHomogenizer.input_dim, init_scale=emb_init_scale
            )
            self.spike_type_emb = Embedding(
                4,
                cfg.model.setup.CausalHomogenizer.input_dim,
                init_scale=emb_init_scale,
            )
        else:
            self.latent_emb = Embedding(
                num_latents, cfg.model.setup.input_size, init_scale=emb_init_scale
            )
            self.unit_emb = InfiniteVocabEmbedding(
                cfg.model.setup.input_size, init_scale=emb_init_scale
            )
            self.session_emb = InfiniteVocabEmbedding(
                cfg.model.setup.input_size, init_scale=emb_init_scale
            )
            self.spike_type_emb = Embedding(
                4, cfg.model.setup.input_size, init_scale=emb_init_scale
            )

        # self.dim = cfg.model.setup.input_size

        self.unit_tokenizer_var = 1
        self.unit_tokenizer_map = collections.defaultdict(int)
        self.session_tokenizer_var = 1
        self.session_tokenizer_map = collections.defaultdict(int)

        if init_embeddings:
            self.init_embeddings()

    def init_embeddings(self):
        self.unit_emb.initialize_vocab([])
        self.session_emb.initialize_vocab([])

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # Add extra stuff needed for model to function properly
        state["unit_tokenizer_map"] = dict(self.unit_tokenizer_map)
        state["unit_tokenizer_var"] = self.unit_tokenizer_var
        state["session_tokenizer_map"] = dict(self.session_tokenizer_map)
        state["session_tokenizer_var"] = self.session_tokenizer_var
        return state

    def load_state_dict(self, state_dict, strict=True):

        if "unit_tokenizer_map" in state_dict:
            self.unit_tokenizer_map = collections.defaultdict(
                int, state_dict["unit_tokenizer_map"]
            )
            del state_dict["unit_tokenizer_map"]
        if "unit_tokenizer_var" in state_dict:
            self.unit_tokenizer_var = state_dict["unit_tokenizer_var"]
            del state_dict["unit_tokenizer_var"]
        if "session_tokenizer_map" in state_dict:
            self.session_tokenizer_map = collections.defaultdict(
                int, state_dict["session_tokenizer_map"]
            )
            del state_dict["session_tokenizer_map"]
        if "session_tokenizer_var" in state_dict:
            self.session_tokenizer_var = state_dict["session_tokenizer_var"]
            del state_dict["session_tokenizer_var"]

        super().load_state_dict(state_dict, strict=strict)

    def create_spike_tensor(
        self,
        spike_unit_index,
        spike_timestamps,
        input_mask,
        spike_type,
        bin_size=0.01,
        memory=None,
        max_time=1.0,
    ):
        """
        Write me pls:
        Returns the binned spikes tensor. Batch x Timepoints x Num_Neurons
        """
        # Ensure input tensors are on the same device
        device = spike_unit_index.device

        # Filter valid spike events using the input mask
        valid_mask = input_mask.bool() & (
            (spike_type == 0) | (spike_type == 3)
        )  # Only consider spike_type 0 or 3

        batch_size, l = spike_unit_index.shape

        # Flatten the batch and sequence dimensions
        spike_unit_index = spike_unit_index[valid_mask]  # Shape: (N,)
        spike_timestamps = spike_timestamps[valid_mask]  # Shape: (N,)
        batch_indices = (
            torch.arange(batch_size, device=device, dtype=torch.int)
            .unsqueeze(1)
            .expand(batch_size, l)[valid_mask]
        )  # Shape: (N,)

        # Compute time bins by discretizing timestamps
        num_bins = int(max_time / bin_size)
        bin_indices = (
            spike_timestamps / bin_size
        ).int()  # Shape: (batch_size, num_spikes)
        assert torch.all(0 <= bin_indices) and torch.all(bin_indices <= num_bins - 1)

        # Create the binned_spikes tensor
        binned_spikes = torch.zeros(
            (batch_size, num_bins, self.num_neurons), dtype=torch.int, device=device
        )

        # Add the spikes to the binned spikes

        ## GIANNIS: is this some weird Monkey-T bug??
        spike_unit_index -= spike_unit_index.min()
        binned_spikes.index_put_(
            (batch_indices, bin_indices, spike_unit_index),
            torch.ones_like(batch_indices, dtype=torch.int32, device=device),
            accumulate=True,
        )

        # Apply memory to the tensor if desired
        binned_spikes = self._apply_memory(binned_spikes)

        return binned_spikes.float()

    def identify_binned_spiked_units(
        self,
        spike_unit_index,
        spike_timestamps,
        input_mask,
        spike_type,
        bin_size=0.01,
        memory=None,
        max_time=1.0,
    ):
        """
        Write me plis:
        Returns a tensor of shape batch_size x timepoints x [variable]: Each tensor batch_size x timepoints contains the unit_ids of all the units that fired during that batch x timepoint. Zeros @ end are padding.
        TODO:  eliminate sorting!!
        """

        device = spike_unit_index.device
        batch_size, num_spikes = spike_unit_index.shape
        time_bins = int(max_time // bin_size) + 1  # Compute number of bins

        # Get rid of initial start and end tokens
        valid_mask = input_mask.bool() & (
            (spike_type == 0) | (spike_type == 3)
        )  # Only consider spike_type 0 or 3

        spike_unit_index = spike_unit_index[valid_mask]  # Shape: (N,)
        spike_timestamps = spike_timestamps[valid_mask]  # Shape: (N,)
        batch_indices = (
            torch.arange(batch_size, device=device, dtype=torch.int)
            .unsqueeze(1)
            .expand(batch_size, num_spikes)[valid_mask]
        )  # Shape: (N,)
        spike_indices = (
            torch.arange(num_spikes, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)[valid_mask]
        )  # Shape: (N,)

        # Compute bin indices
        bin_indices = (
            spike_timestamps // bin_size
        ).int()  # Shape: (batch_size, num_spikes)

        # Create a (batch_size, time_bins, num_spikes) tensor filled with zeros
        binned_units = torch.zeros(
            (batch_size, time_bins, num_spikes), device=device, dtype=torch.int
        )

        # Create index masks for scattering unit indices
        # batch_indices = torch.arange(batch_size, device=device, dtype=torch.int).view(-1, 1).expand_as(spike_timestamps)  # (batch_size, num_spikes)

        # Use scatter_ to place unit indices into bins
        # binned_units[batch_indices, bin_indices, torch.arange(num_spikes)] = spike_unit_index.int()
        binned_units[batch_indices, bin_indices, spike_indices] = spike_unit_index.int()

        # Sort along the last dimension and remove excess zeros
        binned_units, _ = torch.sort(binned_units, dim=-1, descending=True)

        # TODO: eliminate sorting!!

        # Find the max nonzero elements per (batch, bin) to determine final padding size
        max_units_per_bin = (binned_units > 0).sum(dim=-1).max().item()

        # Apply memory
        return self._apply_memory(
            binned_units[:, :, :max_units_per_bin]
        )  # binned_units[:, :, :max_units_per_bin]

    def snn_parameter_analysis(self):
        if "snn" in self.model_name.lower():
            if (
                "CausalHomogenizer" in self.cfg.model.setup
                or "PoyoHomogenizer" in self.cfg.model.setup
            ):
                preproc_params = sum(
                    p.numel() for p in self.homogenizer.parameters() if p.requires_grad
                )
            else:
                preproc_params = 0
            snn_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = preproc_params + snn_params
            print(f"Total parameters: {total_params}")
            print(f"Preprocessing parameters: {preproc_params}")
            print(f"SNN parameters: {snn_params}")

            print(
                f"Preprocessing parameters as a percentage of total parameters: {preproc_params/total_params*100:.2f}%"
            )
            print(
                f"SNN parameters as a percentage of total parameters: {snn_params/total_params*100:.2f}%"
            )

    def unit_tokenizer(self, unit_ids):
        initialize_vocab = []
        for unit_id in unit_ids:
            if not self.unit_tokenizer_map[unit_id]:
                self.unit_tokenizer_map[unit_id] = self.unit_tokenizer_var
                self.unit_tokenizer_var += 1
                initialize_vocab.append(unit_id)
        self.unit_emb.extend_vocab(initialize_vocab)
        return np.array(self.unit_emb.tokenizer(list(unit_ids)))

    def session_tokenizer(self, session_id):
        if not self.session_tokenizer_map[session_id]:
            self.session_tokenizer_map[session_id] = self.session_tokenizer_var
            self.session_tokenizer_var += 1
            self.session_emb.extend_vocab([session_id])
        return self.session_emb.tokenizer(session_id)

    def moving_average(self, input_tensor, window_size):
        """
        Computes the moving average of a 3D tensor along axis 1 (time axis) using convolution.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_features).
            window_size (int): The window size for the moving average.

        Returns:
            torch.Tensor: The smoothed tensor after applying the moving average over axis 1.
        """
        # Transpose the tensor to move axis 1 (seq_len) to the last position: (batch_size, num_features, seq_len)
        input_tensor = input_tensor.transpose(
            1, 2
        )  # Shape: (batch_size, num_features, seq_len)

        # Get the number of features (channels)
        num_features = input_tensor.shape[1]

        # Create the moving average kernel with size (num_features, 1, window_size)
        kernel = (
            torch.ones(num_features, 1, window_size, device=input_tensor.device)
            / window_size
        )

        # Apply 1D convolution (moving average) along the time axis (now last axis)
        smoothed_tensor = F.conv1d(
            input_tensor, kernel, padding="same", groups=num_features
        )

        # Transpose back to the original shape (batch_size, seq_len, num_features)
        smoothed_tensor = smoothed_tensor.transpose(1, 2)

        return smoothed_tensor

    def _process_with_poyo_homogenizer(
        self,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask,  # (B, N_in)
        # input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        # latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):

        inputs = self.unit_emb(spike_unit_index) + self.spike_type_emb(spike_type)
        latents = self.latent_emb(latent_index)
        # latents = self.latent_emb(torch.arange(self.latent_emb.num_embeddings).repeat(spike_unit_index.shape[0], 1))
        output_queries = self.session_emb(session_index)

        latents = self.homogenizer(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=spike_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=input_mask,
            # input_seqlen=input_seqlen,
            # latent_seqlen=latent_seqlen,
            # output_query_seqlen=output_seqlen,
        )

        return latents

    def _process_with_causal_homogenizer(
        self,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask,  # (B, N_in)
        # input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        # latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):

        inputs = self.identify_binned_spiked_units(
            spike_unit_index,
            spike_timestamps,
            input_mask,
            spike_type,
            bin_size=0.01,
            memory=None,
            max_time=self.cfg.sampler.max_time,
        )

        b, t, c = inputs.shape

        input_mask = inputs == 0
        input_mask = rearrange(input_mask, "b t c -> (b t) c")
        # input_mask = input_mask.unsqueeze(1)

        inputs = self.unit_emb(inputs)

        latent_index = rearrange(
            torch.arange(self.latent_emb.num_embeddings), "n -> () () n"
        )
        latent_index = torch.arange(
            self.latent_emb.num_embeddings, device=inputs.device
        ).repeat(b, t, 1)

        latents = self.latent_emb(latent_index)
        # latents = self.latent_emb(torch.arange(self.latent_emb.num_embeddings).repeat(spike_unit_index.shape[0], 1))
        output_queries = self.session_emb(session_index)

        inputs = rearrange(inputs, "b t c d -> (b t) c d")
        latents = rearrange(latents, "b t c d -> (b t) c d")

        latents = self.homogenizer(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=spike_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=input_mask,
            # input_seqlen=input_seqlen,
            # latent_seqlen=latent_seqlen,
            # output_query_seqlen=output_seqlen,
        )

        latents = rearrange(latents, pattern="(b t) c -> b t c", b=b, t=t)

        return latents

    def _process_without_homogenizer(
        self,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask,  # (B, N_in)
        # input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        # latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        inputs = self.create_spike_tensor(
            spike_unit_index=spike_unit_index,
            spike_timestamps=spike_timestamps,
            input_mask=input_mask,
            spike_type=spike_type,
            max_time=self.cfg.sampler.max_time,
        )
        return inputs

    def _forward_spiking_jelly(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        output_pred = self.model(inputs)
        output_pred = output_pred.permute(1, 0, 2)
        return output_pred

    def _forward_mlp(self, inputs):
        return self.model(inputs)

    def _forward_wiener(self, inputs):
        return self.model(inputs)

    def _forward_gru(self, inputs):
        output_pred, _ = self.model(inputs)
        # output_pred = rearrange(output_pred, "(b l) t -> b l t", l=self.seq_length)
        return output_pred

    def _apply_memory_snn_poyo(self, spike_tensor: torch.Tensor):
        return spike_tensor

    def _apply_memory_mlp_wiener(self, spike_tensor: torch.Tensor):
        sequence = [
            torch.roll(spike_tensor, shifts=i, dims=1) for i in range(self.memory + 1)
        ]
        # return torch.cat(sequence, dim=-1)
        memory_tensor = torch.stack(sequence, dim=2)  # (B, T, mem+1, N_features)
        memory_tensor = rearrange(memory_tensor, "b t m n -> b t (m n)")
        return memory_tensor

    def _apply_memory_gru(self, spike_tensor: torch.Tensor):
        sequence = [
            torch.roll(spike_tensor, shifts=i, dims=1) for i in range(self.memory + 1)
        ]
        spike_tensor = torch.stack(sequence[::-1], dim=2)
        spike_tensor = rearrange(spike_tensor, "b l t n -> (b l) t n")
        return spike_tensor

    def forward_packed(self, kwarg_dict):
        return self.forward(**kwarg_dict)

    def _benchmarked_function_ms(self, function, function_args, device):
        # Start timing
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        # Function call
        results = function(*function_args)

        # End timing
        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
        else:
            end_time = time.perf_counter()
            elapsed_time_ms = (end_time - start_time) * 1000

        # Return results and elapsed time
        return results, elapsed_time_ms

    def forward(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        inputs = self.preprocess(
            # input sequence
            spike_unit_index,  # (B, N_in)
            spike_timestamps,  # (B, N_in)
            spike_type,  # (B, N_in)
            input_mask,  # (B, N_in)
            # input_seqlen=None,
            # latent sequence
            latent_index,  # (B, N_latent)
            latent_timestamps,  # (B, N_latent)
            # latent_seqlen=None,
            # output sequence
            session_index,  # (B,)
            output_timestamps,  # (B, N_out)
            output_seqlen,
            output_batch_index,
            output_mask,
            output_values,
            output_weights,
        )

        # Generate the binned spike data
        # inputs = self.preprocess(
        #     spike_unit_index=spike_unit_index,
        #     spike_timestamps=spike_timestamps,
        #     input_mask=input_mask,
        #     spike_type=spike_type,
        #     session_index=session_index,
        # )

        if (
            hasattr(self.cfg.model, "enable_benchmarking")
            and self.cfg.model.enable_benchmarking
        ):
            output_pred, fwd_ms = self._benchmarked_function_ms(
                self._forward, (inputs,), spike_unit_index.device
            )
            # TODO: how do we track performance over training?
            self.delta_t_raw_forward = fwd_ms / 1000  # convert to seconds
        else:
            output_pred = self._forward(inputs)

        # Smoothing layer
        output_pred = self.moving_average(output_pred, 20)

        assert output_mask is not None

        # bugfix
        output_mask = output_mask[:, : output_pred.shape[1]]

        masked_preds = output_pred[output_mask]
        lead_dim_cut = min(masked_preds.shape[0], output_values.shape[0])

        loss = compute_loss_or_metric(
            "mse",
            OutputType.CONTINUOUS,
            masked_preds[:lead_dim_cut],
            output_values[:lead_dim_cut].float(),
            output_weights[:lead_dim_cut],
        )

        if "snn" in self.cfg.model.type.lower():
            exponent = 2 if self.cfg.model.setup.sparsity_loss == "l2" else 1
            raw_sparsity_loss = self.model.get_average_spiking_rate() ** exponent
            loss += raw_sparsity_loss * self.cfg.model.setup.sparsity_loss_weight
            print(f"Raw Sparsity Loss: {raw_sparsity_loss.item()}")
            print(f"Loss: {loss.item()}")
            print(f"Avg spike activity: {self.model.get_average_spiking_rate()}")

        output = masked_preds[:lead_dim_cut]

        return (output, loss, None)


class POYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        latent_step,
        num_latents_per_step,
        using_memory_efficient_attn: bool = True,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.using_memory_efficient_attn = using_memory_efficient_attn
        self.eval = eval

    def __call__(self, data):
        # context window
        start, end = 0, 0.6  # data.domain, data.end

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        spike_token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(spike_unit_index)]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        latent_index = np.arange((end - start) / self.latent_step).astype(
            int
        )  ############################### This is not from its mother
        ### prepare outputs
        session_index = self.session_tokenizer(data.session)

        output_timestamps = data.cursor.timestamps
        output_values = data.cursor.vel
        output_subtask_index = data.cursor.subtask_index

        # compute weights

        if not data.config:  # l
            data.config["reach_decoder"] = {}
        weight = data.config["reach_decoder"].get("weight", 1.0)
        subtask_weights = data.config["reach_decoder"].get("subtask_weights", {})
        # num_subtasks = Task.REACHING.max_value()
        num_subtasks = REACHING.max_value() + 1  # l
        subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
        for subtask, subtask_weight in subtask_weights.items():
            subtask_weight_map[Task.from_string(subtask).value] = subtask_weight
        # subtask_weight_map[0] = 5.0 #l
        # subtask_weight_map[1] = 5.0 #l
        subtask_weight_map[2] = 5.0  # l
        # subtask_weight_map[3] = 5.0 #l
        # subtask_weight_map[4] = 5.0 #l
        # subtask_weight_map[5] = 5.0 #l
        subtask_weight_map *= weight
        output_weights = subtask_weight_map[output_subtask_index]

        # TODO: When in evaluation mode, return the R2 score during the reaching only - not the whole task.
        # if self.eval:
        #     output_timestamps = output_timestamps[(output_subtask_index>=2)&(output_subtask_index<=2)]
        #     output_values = output_values[(output_subtask_index>=2)&(output_subtask_index<=2)]
        #     output_weights = output_weights[(output_subtask_index>=2)&(output_subtask_index<=2)]
        #     output_subtask_index = output_subtask_index[(output_subtask_index>=2)&(output_subtask_index<=2)]

        if not self.using_memory_efficient_attn:
            # Padding
            batch = {
                # input sequence
                "spike_unit_index": pad(spike_unit_index),
                "spike_timestamps": pad(spike_timestamps),
                "spike_type": pad(spike_token_type_index),
                "input_mask": track_mask(spike_unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # output sequence
                "session_index": pad(np.repeat(session_index, len(output_timestamps))),
                "output_timestamps": pad(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
                "output_mask": track_mask(output_timestamps),
            }
        else:
            # Chaining
            batch = {
                # input sequence
                "spike_unit_index": chain(spike_unit_index),
                "spike_timestamps": chain(spike_timestamps),
                "spike_type": chain(spike_token_type_index),
                "input_seqlen": len(spike_unit_index),
                # latent sequence
                "latent_index": chain(latent_index),
                "latent_timestamps": chain(latent_timestamps),
                "latent_seqlen": len(latent_index),
                # output sequence
                "session_index": chain(
                    np.repeat(session_index, len(output_timestamps))
                ),
                "output_timestamps": chain(output_timestamps),
                "output_seqlen": len(output_timestamps),
                "output_batch_index": track_batch(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(output_subtask_index)

        return batch
