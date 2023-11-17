from typing import Mapping, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch import Tensor
import cv2

from mingpt.multigame_dt_utils import (
    accuracy,
    autoregressive_generate,
    cross_entropy,
    decode_return,
    encode_return,
    encode_reward,
    sample_from_logits,
    variance_scaling_,
)

from mingpt.visualize_attention import visualize_attn_np, attention_patches_mean, attention_layers_mean, visualize_attn_heatmap, visualize_attn_tensor, visualize_attn_tensor_detach
from mingpt.visualize_attention2 import visualize_predict

from mingpt.block_dt import Block

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")



#------------------------------------------------
class Transformer(nn.Module):
    r"""A transformer stack."""
    _render_layer_id = None
    _render_attn_head_id = None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
    ):
        logger.info(f"Transformer:: embed_dim:{embed_dim} num_heads(d_model // 64)={num_heads} , num_layers:{num_layers} ")
        super().__init__()
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

        init_scale = 2.0 / self._num_layers
        self.layers = nn.ModuleList([])
        for i in range(self._num_layers):
            logger.info(f"Creating Block {i}, num_heads:{num_heads}") 
            block = Block(embed_dim, num_heads, init_scale, dropout_rate)
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        r"""Connects the transformer.

        Args:
        h: Inputs, [B, T, D].
        mask: Padding mask, [B, T].
        custom_causal_mask: Customized causal mask, [T, T].
        prefix_length: Number of prefix tokens that can all attend to each other.

        Returns:
        Array of shape [B, T, D].
        """
        if mask is not None:
            # Make sure we're not passing any information about masked h.
            h = h * mask[:, :, None]
            mask = mask[:, None, None, :]

        # logger.info(f"forward() - h.shape:{h.shape}")   # h.shape:torch.Size([1, 156, 1280])

        self._render_layer_id = 8          # 0-9
        # self._render_attn_head_id = 0     # 0-15 (4x4). See L211
        self._render_attn_head_id = 19     # 0-19 (4x4). See L252
        for i, block in enumerate(self.layers):   # Block.forward(2) called here
            h = block(
                x=h,
                layer_id=i,
                render_layer_id=self._render_layer_id,
                render_attn_head_id=self._render_attn_head_id,
                mask=mask,
                custom_causal_mask=custom_causal_mask,
                prefix_length=prefix_length,
            )
        h = self.norm_f(h)
        return h
