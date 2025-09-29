import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from einops import rearrange
from poyo.nn import RotaryEmbedding, RotaryCrossAttention
from models.attention import CrossAttention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PoyoHomogenizer(nn.Module):
    def __init__(
        self,
        *,
        dim=128,
        context_dim=None,
        dim_head=64,
        # depth=2,
        cross_heads=1,
        # self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        use_memory_efficient_attn=False,
    ):
        super().__init__()

        if use_memory_efficient_attn:
            ValueError("Cannot use memory efficient attention ")

        self.rotary_emb = RotaryEmbedding(dim_head)

        # self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            use_memory_efficient_attn=False,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # # Processing transfomers (qkv-latent)
        # self.proc_layers = nn.ModuleList([])
        # for i in range(depth):
        #     self.proc_layers.append(
        #         nn.ModuleList(
        #             [
        #                 RotarySelfAttention(
        #                     dim=dim,
        #                     heads=self_heads,
        #                     dropout=atn_dropout,
        #                     dim_head=dim_head,
        #                     rotate_value=True,
        #                     use_memory_efficient_attn=use_memory_efficient_attn,
        #                 ),
        #                 nn.Sequential(
        #                     nn.LayerNorm(dim),
        #                     FeedForward(dim=dim, dropout=ffn_dropout),
        #                 ),
        #             ]
        #         )
        #     )

        # self.dec_atn = RotaryCrossAttention(
        #     dim=dim,
        #     heads=cross_heads,
        #     dropout=atn_dropout,
        #     dim_head=dim_head,
        #     rotate_value=False,
        #     use_memory_efficient_attn=use_memory_efficient_attn,
        # )
        # self.dec_ffn = nn.Sequential(
        #     nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        # )

        self.dim = dim
        self.using_memory_efficient_attn = (
            False  # self.enc_atn.using_memory_efficient_attn
        )

    def forward(
        self,
        *,  # (   padded   ) or (   chained   )
        inputs,  # (B, N_in, dim) or (N_all_in, dim)
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries,  # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps,  # (B, N_in) or (N_all_in,)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        input_mask=None,  # (B, N_in) or None
        input_seqlen=None,  # None or (B,)
        latent_seqlen=None,  # None or (B,)
        output_query_seqlen=None,  # None or (B,)
    ):

        # Make sure the arguments make sense
        assert inputs.dim() == 3
        assert latents.dim() == 3
        assert output_queries.dim() == 3
        assert input_timestamps.dim() == 2
        assert latent_timestamps.dim() == 2
        assert output_query_timestamps.dim() == 2
        assert input_mask.dim() == 2

        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        # output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            context_mask=input_mask,  # used if default attention
            query_seqlen=latent_seqlen,  # used if memory efficient attention
            context_seqlen=input_seqlen,  # used if memory efficient attention
        )

        latents = latents + self.enc_ffn(latents)

        return latents


class CausalHomogenizer(nn.Module):
    def __init__(
        self,
        # *,
        # dim=128,
        # context_dim=None,
        # dim_head=64,
        # # depth=2,
        # cross_heads=1,
        # # self_heads=8,
        # ffn_dropout=0.2,
        # lin_dropout=0.4,
        # atn_dropout=0.0,
        # use_memory_efficient_attn=False,
        # cross_attn_widening_factor=4,
        num_latents: int,
        latent_dim: int,
        input_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        num_cross_attn_heads: int = 1,
        cross_attn_widening_factor: int = 1,
        use_query_residual: bool = True,
        cross_attn_lin_dropout: float = 0.0,
        cross_attention_dropout: float = 0.0,
        # num_neurons: list = [],
        # n_self_attn_blocks: int = 1,
        # self_attn_heads: int = 1,
        # self_attn_lin_dropout: float = 0.0,
        # self_attention_dropout: float = 0.0,
        num_virtual_channels: int = 128,
    ):

        super().__init__()

        self.cross_attn = CrossAttention(
            kv_dim=input_dim,
            q_dim=latent_dim,
            widening_factor=cross_attn_widening_factor,
            num_heads=num_cross_attn_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual,
            dropout=cross_attn_lin_dropout,
            attention_dropout=cross_attention_dropout,
        )

        # self_attn_layers = [
        #     SelfAttention(
        #         hidden_dim=latent_dim,
        #         qk_out_dim=None,
        #         v_out_dim=None,
        #         widening_factor=4,
        #         num_heads=self_attn_heads,
        #         dropout=self_attn_lin_dropout,
        #         attention_dropout=self_attention_dropout,
        #     )
        #     for _ in range(n_self_attn_blocks)
        # ]

        # self.self_attn = nn.Sequential(*self_attn_layers)

        # self.projection = nn.Linear(dim, 1)
        self.projection = nn.Linear(latent_dim * num_latents, num_virtual_channels)

        # self.num_neurons = num_neurons

        # self.embed = nn.Embedding(int(num_neurons.sum() + 1), latent_dim, padding_idx=0)

    def forward(
        self,
        *,  # (   padded   ) or (   chained   )
        inputs,  # (B, N_in, dim) or (N_all_in, dim)
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries,  # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps,  # (B, N_in) or (N_all_in,)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        input_mask=None,  # (B, N_in) or None
        input_seqlen=None,  # None or (B,)
        latent_seqlen=None,  # None or (B,)
        output_query_seqlen=None,  # None or (B,))
    ):

        # b_, t_, e_ = x.size()
        # x = rearrange(x, "b t e -> (b t) e")

        # Generate a mask for all the 0s in x [the ~mask indexes all 1s] - those are the padded channels
        # padding_mask = inputs == 0

        # Do the cross-attention with learnable latents
        latents = self.cross_attn(
            inputs_kv=inputs,
            inputs_q=latents,
            attention_mask=input_mask,
        )

        # # Stack a couple of self-attention layers on the latents to enhance fitting ability
        # latents = self.self_attn(latents)

        # # Project the latents from their embeddings to the final size of their
        latents = rearrange(latents, "(b) e f-> b (e f)")
        latents = self.projection(latents)

        return latents
