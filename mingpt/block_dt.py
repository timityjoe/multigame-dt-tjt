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

from mingpt.attention_dt import Attention, MLP

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")


#------------------------------------------------
class CausalSelfAttention(Attention):
    r"""Self attention with a causal mask applied."""

    def forward(
        self,
        x: Tensor,
        layer_id, 
        render_layer_id,
        render_attn_head_id,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Expect queries of shape [B, T, D].")

        # logger.info(f"CausalSelfAttention x.shape:{x.shape} mask:{mask}")  # x.shape:torch.Size([1, 156, 1280])
        # logger.info(f"forward() - layer_id:{layer_id}, render_layer_id:{render_layer_id}")

        seq_len = x.shape[1]
        # If custom_causal_mask is None, the default causality assumption is
        # sequential (a lower triangular causal mask).
        causal_mask = custom_causal_mask
        if causal_mask is None:
            device = x.device
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        causal_mask = causal_mask[None, None, :, :]

        # Visualize; np_causal_mask = 4x4 = 16 patches
        # np_causal_mask = causal_mask[0].data.cpu().numpy()
        # np_causal_mask = np.moveaxis(np_causal_mask, 0, -1)
        # logger.info(f"np_causal_mask.shape:{np_causal_mask.shape}")  #causal_mask.shape:torch.Size([1, 1, 156, 156])
        # visualize_attn_np(np_causal_mask, "np_causal_mask")

        # Similar to T5, tokens up to prefix_length can all attend to each other.
        causal_mask[:, :, :, :prefix_length] = 1
        mask = mask * causal_mask if mask is not None else causal_mask
        # logger.info(f"mask.shape:{mask.shape}")  # mask.shape:torch.Size([2, 1, 156, 156])

        # Visualize; np_mask = 4x4 = 16 patches
        # np_mask = mask[0].data.cpu().numpy()
        # np_mask = np.moveaxis(np_mask, 0, -1)
        # logger.info(f"np_mask.shape:{np_mask.shape}")
        # visualize_attn_np(np_mask, "np_mask")

        return super().forward(x, 
                            layer_id=layer_id, 
                            render_layer_id=render_layer_id,
                            render_attn_head_id=render_attn_head_id,
                            mask=mask)       # Attention.forward(4) called here

#------------------------------------------------
class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, init_scale: float, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)

        # logger.info(f" num_heads:{num_heads}") 

        self.attn = CausalSelfAttention(embed_dim, num_heads=num_heads, w_init_scale=init_scale)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, init_scale)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, layer_id, render_layer_id, render_attn_head_id, **kwargs):
        # logger.info(f"forward() - layer_id:{layer_id}, render_layer_id:{render_layer_id}")
        # logger.info(f"forward() - x.shape:{x.shape}")       # x.shape:torch.Size([1, 156, 1280])

        x = x + self.dropout_1(self.attn(self.ln_1(x), 
                                         layer_id=layer_id, 
                                         render_layer_id=render_layer_id,
                                         render_attn_head_id = render_attn_head_id,
                                         **kwargs))         # CausalSelfAttention.forward(3) called here
        x = x + self.dropout_2(self.mlp(self.ln_2(x)))
        return x
