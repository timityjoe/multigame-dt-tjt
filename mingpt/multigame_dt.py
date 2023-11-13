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

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

#------------------------------------------------
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#------------------------------------------------
class MLP(nn.Module):
    r"""A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self,
        in_dim: int,
        init_scale: float,
        widening_factor: int = 4,
    ):
        # logger.info(f"in_dim:{in_dim}, init_scale:{init_scale}, widening_factor:{widening_factor}")
        super().__init__()
        self._init_scale = init_scale
        self._widening_factor = widening_factor

        self.fc1 = nn.Linear(in_dim, self._widening_factor * in_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(self._widening_factor * in_dim, in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.fc1.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc1.bias)
        variance_scaling_(self.fc2.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x len = 2
        # logger.info(f"x1 type:{type(x)} len:{len(x)}")
        x = self.fc1(x)
        # logger.info(f"x2 type:{type(x)} len:{len(x)}")
        x = self.act(x)
        # logger.info(f"x3 type:{type(x)} len:{len(x)}")
        x = self.fc2(x)
        # logger.info(f"x4 type:{type(x)} len:{len(x)}")
        return x

#------------------------------------------------
class Attention(nn.Module):
    _np_attn_mean = None
    _np_attn_heatmap = None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        w_init_scale: Optional[float] = None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        logger.info(f"Attention:: dim:{dim} , num_heads:{num_heads}, w_init_scale:{w_init_scale}")
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.w_init_scale = w_init_scale

        # Transformer Query Key Value - qkv
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.qkv.weight, scale=self.w_init_scale)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        variance_scaling_(self.proj.weight, scale=self.w_init_scale)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, 
                layer_id, 
                render_layer_id,
                render_attn_head_id,
                mask: Optional[Tensor] = None) -> Tensor:
        # logger.info(f"forward() - layer_id:{layer_id}, render_layer_id:{render_layer_id}")

        B, T, C = x.shape
        # logger.info(f"B:{B} T:{T} C:{C}") # B:1 T:156 C:1280

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # logger.info(f"qkv.shape:{qkv.shape}") # qkv.shape:torch.Size([3, 2, 20, 156, 64])

        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # q.shape:torch.Size([2, 20, 156, 64]) k.shape:torch.Size([2, 20, 156, 64]) v.shape:torch.Size([2, 20, 156, 64])
        # logger.info(f"q.shape:{q.shape} k.shape:{q.shape} v.shape:{q.shape}") 

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max  # max_neg_value
            attn = attn.masked_fill(~mask.to(dtype=torch.bool), mask_value)

        attn = attn.softmax(dim=-1)

        # logger.info(f"len(attn):{len(attn)}, type:{type(attn)}") # attn len = 2
        # logger.info(f"attn.shape:{attn.shape}") # attn.shape:torch.Size([2, 20, 156, 156])

        # self._np_attn_mean = attention_patches_mean(attn)
        # logger.info(f"  _np_attn_mean.shape:{self._np_attn_mean.shape}") # _np_attn_mean.shape:(156, 156, 3)
        # visualize_attn_np(self._np_attn_mean, "_np_attn_mean")

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        # logger.info(f"  (1)x.shape:{x.shape}") # x.shape:torch.Size([2, 156, 1280])
        x = self.proj(x)
        # logger.info(f"  (2)x.shape:{x.shape}") # x.shape:torch.Size([2, 156, 1280])

        #------------------------------------------------------
        # Mod by Tim: 
        # See https://github.com/mashaan14/VisionTransformer-MNIST/blob/main/VisionTransformer_MNIST.ipynb
        # kT = k.transpose(-2, -1)
        # attention_matrix = q @ kT
        # # logger.info(f" attention matrix:{attention_matrix.shape}") # attention matrix torch.Size([1, 20, 156, 156])
        # # Note, Heads = 20, 36+3 tokens = 39, 39*num_steps(4) = 156

        # attention_matrix_mean = torch.mean(attention_matrix, dim=0)
        # # logger.info(f"  attention_matrix_mean.shape:{attention_matrix_mean.shape}") # attention_matrix_mean.shape:torch.Size([20, 156, 156])

        # # if (layer_id == render_layer_id):
        # #     copy = attention_matrix_mean
        # #     copy = torch.mean(copy, dim=0)
        # #     logger.info(f"  attention_matrix_mean.shape:{copy.shape}")
        # #     copy = copy.cuda().detach().cpu().clone().numpy()
        # #     copy = cv2.applyColorMap(copy.astype(np.uint8), cv2.COLORMAP_INFERNO)
        # #     visualize_attn_np(copy, f"attention_matrix_mean layer_id:{layer_id}")

        # # To account for residual connections, we add an identity matrix to the
        # # attention matrix and re-normalize the weights.
        # residual_att = torch.eye(attention_matrix_mean.size(1)).to('cuda:0')
        # aug_att_mat = attention_matrix_mean + residual_att
        # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # # Recursively multiply the weight matrices
        # joint_attentions = torch.zeros(aug_att_mat.size()).to('cuda:0')
        # joint_attentions[0] = aug_att_mat[0]
        # for n in range(1, aug_att_mat.size(0)):
        #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        # # logger.info(f"  joint_attentions.shape:{joint_attentions.shape}") # joint_attentions.shape:torch.Size([20, 156, 156])

        # attn_heatmap = joint_attentions[0, 1:].reshape((156, 155))
        # # logger.info(f"  attn_heatmap.shape:{attn_heatmap.shape}")   # attn_heatmap.shape:torch.Size([156, 155])

        # attn_heatmap_resized = torch.nn.functional.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0), [210, 160], mode='bilinear').view(210, 160, 1)
        

        # attn_heatmap_resized = attn_heatmap_resized.cuda().detach().cpu().clone().numpy()
        # # logger.info(f"attn_heatmap_resized:{attn_heatmap_resized.shape}") # attn_heatmap_resized:torch.Size([210, 160, 1])

        # if (layer_id == render_layer_id):
        #     visualize_attn_np(attn_heatmap_resized, f"attn_heatmap_resized layer_id:{layer_id}")

        #------------------------------------------------------
        # Note:
        # Image observation is broken down into 6x6 = 36 image patches/tokens
        # Total = 36+3 = 39 tokens
        # Heads = 20, 36+3 tokens = 39, 39*num_steps(4) = 156
        x1 = (q @ k.transpose(-2, -1)) * self.scale
        # logger.info(f"  x1.shape:{x1.shape}")   # x1.shape:torch.Size([1, 20, 156, 156])

        #------------------------------------------------------
        # if (layer_id == render_layer_id):
        #     for (i, iter) in enumerate(x1[0]):
        #         copy = iter 
        #         if (i == 19):
        #             # logger.info(f"copy.shape:{copy.shape}")     # copy.shape:torch.Size([156, 156])
        #             # (tps)tokens_per_step = 39
        #             # (T)num_steps = 4

        #             # Reorder [T*tps, T*tps, C] into [T*T, tps, tps, C]
        #             # copy = torch.reshape(copy, [16, 39, 39])
        #             # # View the last (15). Adjust in L406
        #             # copy = copy[render_attn_head_id]    
        #             # copy = np.swapaxes(copy, 0, 1)

        #             # copy = torch.nn.functional.interpolate(copy.unsqueeze(0).unsqueeze(0), [210, 160], mode='bilinear').view(210, 160, 1)
        #             copy = torch.nn.functional.interpolate(copy.unsqueeze(0).unsqueeze(0), [210, 160], mode="nearest").view(210, 160, 1)

        #             np_copy = copy.cuda().detach().cpu().clone().numpy()
        #             np_copy = np_copy / 10.
        #             np_copy = cv2.applyColorMap(np_copy.astype(np.uint8), cv2.COLORMAP_INFERNO )
        #             # logger.info(f"  np_copy.shape:{np_copy.shape}")     # np_copy.shape:(39, 39, 3)
        #             self._np_attn_heatmap = np_copy
        #             # visualize_attn_np(self._np_attn_heatmap, f"np_{i}")

        #------------------------------------------------------
        # keep only the output patch attention
        copy2 = x1
        nh = 20
        # copy2 = copy2[0, :, 0, 1:]
        copy2 = copy2[0, :, 0, 0:]
        # copy2 = copy2[0, :, 0:, 0]
        # logger.info(f"    (1)copy2.shape:{copy2.shape}")   # copy2.shape:torch.Size([20, 155])

        # input of size 3120
        # w_featmap = 39
        # h_featmap = 4
        w_featmap = 12
        h_featmap = 13
        copy2 = copy2.reshape(nh, w_featmap, h_featmap)
        # logger.info(f"    (2)copy2.shape:{copy2.shape}")   # 

        # copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0).unsqueeze(0), [210, 160], mode="nearest").view(210, 160, 1)
        # copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0), scale_factor=16, mode="nearest")[0]
        copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0), [210, 160], mode="nearest")[0]
        # logger.info(f"    (3)copy2.shape:{copy2.shape}")    # copy2.shape:torch.Size([20, 210, 160])
        copy2 = copy2[render_attn_head_id] 
        # logger.info(f"    (3)copy2.shape:{copy2.shape}") 

        np_copy = copy2.cuda().detach().cpu().clone().numpy()
        # np_copy = np_copy / 10.
        np_copy = cv2.applyColorMap(np_copy.astype(np.uint8), cv2.COLORMAP_INFERNO )
        # logger.info(f"  np_copy.shape:{np_copy.shape}")     # np_copy.shape:(39, 39, 3)
        self._np_attn_heatmap = np_copy
        # visualize_attn_np(self._np_attn_heatmap, f"np_{i}")

        #------------------------------------------------------
        x1 = x1.softmax(dim=-1)
        np_x1 = x1.detach().cpu().numpy()
        # x1 = x1         # x1.shape:torch.Size([2, 20, 156, 156])
        # x1 = x1[0]      # x1.shape:torch.Size([20, 156, 156])
        # np_x1 = np_x1[0][0]      # x1.shape:torch.Size([156, 156])
        # logger.info(f"  (3)x1.shape:{x1.shape}") # 

        # if (layer_id == render_layer_id):
        #     for (i, iter) in enumerate(np_x1[0]):
        #         np_x1 = iter
        #         # logger.info(f"  i:{i}")
        #         # logger.info(f"  np_x1:{np_x1}")
        #         # logger.info(f"  np_x1.shape:{np_x1.shape}")
        #         np_x1 = np_x1 * 255.
        #         np_x1 = cv2.applyColorMap(np_x1.astype(np.uint8), cv2.COLORMAP_INFERNO )
                
        #         if (i == 19):
        #             # logger.info(f"np_x1.shape:{np_x1.shape}") # np_x1.shape:(156, 156, 3)
        #             # (tps)tokens_per_step = 39
        #             # (T)num_steps = 4
        #             # Reorder [T*tps, T*tps, C] into [T*T, tps, tps, C]
        #             np_x1 = np.reshape(np_x1, [16, 39, 39, 3])

        #             # View the mean (of the 4x4 = 16)
        #             # np_x1 = np.mean(np_x1, axis=0)
        #             # logger.info(f"  np_x1.shape:{np_x1.shape}")

        #             # View the last (15)
        #             np_x1 = np_x1[0]
        #             visualize_attn_np(np_x1, f"np_{i}")
        # # logger.info(f"Viz all 20 heads")
        #------------------------------------------------------

        # x len = 2
        # logger.info(f"len(x):{len(x)}, type:{type(x)}")


        return x

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

        self._render_layer_id = 5          # 0-9
        self._render_attn_head_id = 0     # 0-15 (4x4). See L211
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

#------------------------------------------------
class MultiGameDecisionTransformer(nn.Module):
    _np_attn_container = None
    _np_attn_mean = None
    _np_rgb_img = None

    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        num_actions: int,
        num_rewards: int,
        return_range: Tuple[int],
        d_model: int,
        num_layers: int,
        dropout_rate: float,
        predict_reward: bool,
        single_return_token: bool,
        conv_dim: int,
    ):
        super().__init__()

        logger.info(f"MultiGameDecisionTransformer:: 0) Start") 
        logger.info(f"      img_size:{img_size} patch_size:{patch_size} num_actions:{num_actions} num_rewards:{num_rewards} return_range:{return_range}")
        logger.info(f"      d_model:{d_model} num_layers:{num_layers} dropout_rate:{dropout_rate} conv_dim:{conv_dim}")

        # Expected by the transformer model.
        if d_model % 64 != 0:
            raise ValueError(f"Model size {d_model} must be divisible by 64")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.num_returns = return_range[1] - return_range[0]
        self.return_range = return_range
        self.d_model = d_model
        self.predict_reward = predict_reward
        self.conv_dim = conv_dim
        self.single_return_token = single_return_token
        self.spatial_tokens = True

        transformer_num_heads = self.d_model // 64

        logger.info(f"MultiGameDecisionTransformer:: 1) Create transformer - transformer_num_heads:{transformer_num_heads}") 
        self.transformer = Transformer(
            embed_dim=self.d_model,
            num_heads=self.d_model // 64,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        patch_height, patch_width = self.patch_size[0], self.patch_size[1]
        # If img_size=(84, 84), patch_size=(14, 14), then P = 84 / 14 = 6.

        logger.info(f"MultiGameDecisionTransformer:: 2) Image embedding") 
        self.image_emb = nn.Conv2d(
            in_channels=1,
            out_channels=self.d_model,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            padding="valid",
        )  # image_emb is now [BT x D x P x P].
        logger.info(f"    type(self.image_emb):{type(self.image_emb)}")

        patch_grid = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        num_patches = patch_grid[0] * patch_grid[1]

        logger.info(f"MultiGameDecisionTransformer:: 3) num_patches:{num_patches}, patch_grid[0]:{patch_grid[0]}, patch_grid[1]:{patch_grid[1]}") 
        # MultiGameDecisionTransformer:: 3) num_patches:36, patch_grid[0]:6, patch_grid[1]:6

        self.image_pos_enc = nn.Parameter(torch.randn(1, 1, num_patches, self.d_model))

        self.ret_emb = nn.Embedding(self.num_returns, self.d_model)
        self.act_emb = nn.Embedding(self.num_actions, self.d_model)
        if self.predict_reward:
            self.rew_emb = nn.Embedding(self.num_rewards, self.d_model)

        num_steps = 4
        num_obs_tokens = num_patches if self.spatial_tokens else 1
        if self.predict_reward:
            tokens_per_step = num_obs_tokens + 3
        else:
            tokens_per_step = num_obs_tokens + 2
        self.positional_embedding = nn.Parameter(torch.randn(tokens_per_step * num_steps, self.d_model))

        self.ret_linear = nn.Linear(self.d_model, self.num_returns)
        self.act_linear = nn.Linear(self.d_model, self.num_actions)
        logger.info(f"ret_linear num_returns:{self.num_returns} ")  # 120
        logger.info(f"act_linear num_actions:{self.num_actions} ")  # 18

        if self.predict_reward:
            self.rew_linear = nn.Linear(self.d_model, self.num_rewards)

        # Mod by Tim: Init the array
        # self._np_attn_container = np.empty((156, 156, 3, transformer_num_heads)) 
        self._np_attn_container = np.empty((transformer_num_heads, 156, 156, 3))    
        self._np_attn_mean = np.empty((156, 156, 3)) 

    #------------------------------------------
    # Mod by Tim: Retrieve attention maps
    # There are 10 layers, each with 20 attention heads (represented by a singular mean)
    # So perform another mean across the 10 layers
    def get_attention_map(self):
        layers = self.transformer.layers
        # logger.info(f"get_attention_map:: len(layers):{len(layers)}, type:{type(layers)}")

        for i, layer in enumerate(layers):
            if layer.attn._np_attn_mean is None:
                # logger.info(f"  Layer:{i} Not initialized; type(_np_attn_mean):{type(layer.attn._np_attn_mean)}, returning...")
                continue

            # logger.info(f"  layer.attn._np_attn_mean type:{type(layer.attn._np_attn_mean)}")
            # logger.info(f"  layer.attn._np_attn_mean len:{len(layer.attn._np_attn_mean)}")
            # logger.info(f"  layer.attn._np_attn_mean shape:{layer.attn._np_attn_mean.shape}")
            # logger.info(f"  ")
             # attn = CausalSelfAttention, inherited from attn
            # visualize_attn(layer.attn._np_attn_mean, i)
            # np.put(self._np_attn_container, i, layer.attn._np_attn_mean)
            self._np_attn_container[i] = layer.attn._np_attn_mean

            # logger.info(f"  model.attn.__np_attn_mean_container type:{type(self._np_attn_mean_container)}")
            # logger.info(f"  model.attn.__np_attn_mean_container len:{len(self._np_attn_mean_container)}")
            # logger.info(f"  model.attn._np_attn_container shape:{self._np_attn_container.shape}")
            # if i is len(layers):
            #     logger.info(f"  i:{i} np_attn_mean_container:{self._np_attn_mean_container}")
    
    def get_last_selfattention(self, x):
        # x = self.prepare_tokens(x)
        # for i, blk in enumerate(self.blocks):
        for i, blk in enumerate(self.transformer.layers):
            if i < len(self.transformer.layers) - 1:
                # x = blk(x)
                pass
            else:
                # return attention of the last block
                # return blk(x, return_attention=True)
                value =  self.transformer.layers[i].attn
                # value = blk._attn
                logger.info(f"get_last_selfattention() - type:{type(value)}")
                logger.info(f"get_last_selfattention() - shape:{value.shape}")
                return value

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)     

    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        print(f"prepare_tokens - x:{x.shape}, B:{B}, nc:{nc}, w:{w}, h:{h}")

        # x = self.patch_embed(x)  # patch linear embedding
        # x = self.image_emb(x)  # patch linear embedding
        # print(f"    (1)x:{x.shape}")

        # add the [CLS] token to the embed patch tokens
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(f"    cls_tokens.shape:{cls_tokens.shape}")
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(f"    (2)x:{x.shape}")

        # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h)
        # print(f"    (3)x:{x.shape}")

        # value = self.pos_drop(x)
        # print(f"    self.pos_drop(x).shape:{value.shape}")
        return x
            

    def update_game_image(self, img):
        # logger.info(f"update_game_image() - img.shape:{img.shape}")  # img.shape:(210, 160, 3)
        self._np_rgb_img = img
        self._np_rgb_img = self._np_rgb_img * 255.
        self._np_rgb_img = cv2.applyColorMap(self._np_rgb_img.astype(np.uint8), cv2.COLORMAP_INFERNO )
        img = self._np_rgb_img
        # logger.info(f"type:{type(self._np_rgb_img)}")

        # With attention map
        # Adjusted in Transformer.forward(), L396
        render_layer_id = self.transformer._render_layer_id
        # render_attn_head_id = self.transformer._render_attn_head_id
        np_heatmap = self.transformer.layers[render_layer_id].attn._np_attn_heatmap
        img = cv2.addWeighted(self._np_rgb_img, 0.5, np_heatmap, 0.5, 0)

        cv2.imshow(f"img", img)
        cv2.waitKey(1) 
        # cv2.destroyAllWindows()


    #------------------------------------------

    def reset_parameters(self):
        nn.init.trunc_normal_(self.image_emb.weight, std=0.02)
        nn.init.zeros_(self.image_emb.bias)
        nn.init.normal_(self.image_pos_enc, std=0.02)

        nn.init.trunc_normal_(self.ret_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.act_emb.weight, std=0.02)
        if self.predict_reward:
            nn.init.trunc_normal_(self.rew_emb.weight, std=0.02)

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        variance_scaling_(self.ret_linear.weight)
        nn.init.zeros_(self.ret_linear.bias)
        variance_scaling_(self.act_linear.weight)
        nn.init.zeros_(self.act_linear.bias)
        if self.predict_reward:
            variance_scaling_(self.rew_linear.weight)
            nn.init.zeros_(self.rew_linear.bias)

    def _image_embedding(self, image: Tensor):
        r"""Embed [B x T x C x W x H] images to tokens [B x T x output_dim] tokens.

        Args:
            image: [B x T x C x W x H] image to embed.

        Returns:
            Image embedding of shape [B x T x output_dim] or [B x T x _ x output_dim].
        """
        assert len(image.shape) == 5
        # logger.info(f"_image_embedding() - image:{image.shape}")  # image:torch.Size([1, 4, 1, 84, 84])

        image_dims = image.shape[-3:]
        batch_dims = image.shape[:2]
        # logger.info(f"image_dims:{image_dims} batch_dims:{batch_dims}") 
        # image_dims:torch.Size([1, 84, 84]) 
        # batch_dims:torch.Size([1, 4])

        # Mod by Tim: Visualize the incoming images patches (84 x 84). These seem to be the 
        # B = Batch of num_envs
        # T = 4 per env
        # C - Color channel (1)
        # np_image = image.cpu().numpy()
        # np_image = np.mean(np_image, axis=0)
        # np_image = np.mean(np_image, axis=0)
        # visualize_attn_tensor(torch.from_numpy(np_image))

        # Reshape to [BT x C x H x W].
        # logger.info(f"(1)image.shape:{image.shape}")   # image.shape:torch.Size([1, 4, 1, 84, 84])
        image = torch.reshape(image, (-1,) + image_dims) 
        # logger.info(f"(2)image.shape:{image.shape}")   # image.shape:torch.Size([4, 1, 84, 84])

        # np_image = image.cpu().numpy()
        # np_image = np.mean(np_image, axis=0)
        # visualize_attn_tensor(torch.from_numpy(np_image))

        # Perform any-image specific processing.
        image = image.to(dtype=torch.float32) / 255.0
        # logger.info(f"(3)image.shape:{image.shape}")    # image.shape:torch.Size([4, 1, 84, 84])

        image_emb = self.image_emb(image)  # [BT x D x P x P] 
        # logger.info(f"(4)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([4, 1280, 6, 6])

        # haiku.Conv2D is channel-last, so permute before reshape below for consistency
        # [BT x P x P x D] 
        image_emb = image_emb.permute(0, 2, 3, 1)  
        # See L411, nn.Conv2d()
        # logger.info(f"(5)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([4, 6, 6, 1280])
        

        # Reshape to [B x T x P*P x D].
        image_emb = torch.reshape(image_emb, batch_dims + (-1, self.d_model))
        # logger.info(f"(6)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([1, 4, 36, 1280])

        image_emb = image_emb + self.image_pos_enc  
        # logger.info(f"(7)image_emb.shape:{image_emb.shape}")    # image_emb:torch.Size([1, 4, 36, 1280])

        return image_emb

    def _embed_inputs(self, obs: Tensor, ret: Tensor, act: Tensor, rew: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Embed only prefix_frames first observations.
        # obs are [B x T x C x H x W].
        # logger.info(f"_embed_inputs()")
        # logger.info(f"  (1)obs.shape:{obs.shape}")  # [B x T x C x W x H] - obs.shape:torch.Size([1, 4, 1, 84, 84])
        obs_emb = self._image_embedding(obs)
        # logger.info(f"  (2)obs_emb.shape:{obs_emb.shape}")  # [B x T x P*P x D] - obs_emb.shape:torch.Size([1, 4, 36, 1280])

        # Embed returns and actions
        # Encode returns.
        ret = encode_return(ret, self.return_range)
        rew = encode_reward(rew)
        ret_emb = self.ret_emb(ret)
        act_emb = self.act_emb(act)
        if self.predict_reward:
            rew_emb = self.rew_emb(rew)
        else:
            rew_emb = None
        return obs_emb, ret_emb, act_emb, rew_emb

    # Mod by Tim:
    # Note in subsequent forward passes, "token_emb" becomes "x"
    def forward(self, inputs: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        r"""Process sequence."""
        # logger.info(f"forward() - inputs:{inputs}")

        num_batch = inputs["actions"].shape[0]
        num_steps = inputs["actions"].shape[1] 
        # input_actions = inputs["actions"]
        # logger.info(f"forward() - input_actions:{input_actions.shape}")               # input_actions:torch.Size([1, 4])
        # input_observations = inputs["observations"]
        # logger.info(f"forward() - input_observations:{input_observations.shape}")     # input_observations:torch.Size([1, 4, 1, 84, 84])
        # input_RTG = inputs["returns-to-go"]
        # logger.info(f"  input_RTG:{input_RTG.shape}")                                 # input_RTG:torch.Size([1, 4])
        # input_rewards = inputs["rewards"]
        # logger.info(f"  input_rewards:{input_rewards.shape}")                         # input_rewards:torch.Size([1, 4])


        # Embed inputs.
        obs_emb, ret_emb, act_emb, rew_emb = self._embed_inputs(
            inputs["observations"],
            inputs["returns-to-go"],
            inputs["actions"],
            inputs["rewards"],
        )
        device = obs_emb.device

        # logger.info(f"forward() ")
        # logger.info(f"  obs_emb:{obs_emb.shape}")   # [B x T x P*P x D] - obs_emb:torch.Size([1, 4, 36, 1280])
        # logger.info(f"  ret_emb:{ret_emb.shape}")   # ([1, 4, 1280])
        # logger.info(f"  act_emb:{act_emb.shape}")   # ([1, 4, 1280])
        # logger.info(f"  rew_emb:{rew_emb.shape}")   # ([1, 4, 1280])

        if self.spatial_tokens:
            # obs is [B x T x W x D]
            num_obs_tokens = obs_emb.shape[2]
            # logger.info(f"  num_obs_tokens:{num_obs_tokens}")   # 36

            obs_emb = torch.reshape(obs_emb, obs_emb.shape[:2] + (-1,))
            # logger.info(f"  obs_emb:{obs_emb.shape}")   # obs_emb:torch.Size([1, 4, 46080])
            # obs is [B x T x W*D]
        else:
            num_obs_tokens = 1

        # logger.info(f"forward() self.spatial_tokens:{self.spatial_tokens} num_obs_tokens:{num_obs_tokens}")
        # forward() self.spatial_tokens:True num_obs_tokens:36
        

        # Collect sequence.
        # Embeddings are [B x T x D].
        if self.predict_reward:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb, rew_emb], dim=-1)
            # logger.info(f"  token_emb:{token_emb.shape}")   # token_emb:torch.Size([1, 4, 49920]), bcos 46080+1280+1280+1280 = 49920

            tokens_per_step = num_obs_tokens + 3
            # tokens_per_step is 36+3=39
            # sequence is [obs ret act rew ... obs ret act rew]
        else:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb], dim=-1)
            tokens_per_step = num_obs_tokens + 2
            # sequence is [obs ret act ... obs ret act]

        # logger.info(f"  (1)token_emb:{token_emb.shape}, tokens_per_step:{tokens_per_step}")   
        # token_emb:torch.Size([1, 4, 49920]), tokens_per_step:36+3=39

        token_emb = torch.reshape(token_emb, [num_batch, tokens_per_step * num_steps, self.d_model])
        # logger.info(f"  (2)token_emb:{token_emb.shape}")   # token_emb:torch.Size([1, 156, 1280])
        # Note:Tokens_per_step(39)*4 = 156
        # So (4*49920)/156 = 1280

        # Create position embeddings.
        token_emb = token_emb + self.positional_embedding
        # logger.info(f"  (3)token_emb:{token_emb.shape}") # token_emb:torch.Size([1, 156, 1280])

        # Run the transformer over the inputs.
        # Token dropout.
        batch_size = token_emb.shape[0]
        obs_mask = np.ones([batch_size, num_steps, num_obs_tokens], dtype=bool)
        # logger.info(f"  obs_mask:{obs_mask.shape}")     # obs_mask:(1, 4, 36)

        ret_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        act_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        rew_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        if self.single_return_token:
            # Mask out all return tokens expect the first one.
            ret_mask[:, 1:] = 0
        if self.predict_reward:
            mask = [obs_mask, ret_mask, act_mask, rew_mask]
            # logger.info(f"  predict_reward - yes")
        else:
            mask = [obs_mask, ret_mask, act_mask]
            # logger.info(f"  predict_reward - no")

        mask = np.concatenate(mask, axis=-1)    
        # logger.info(f"  (1)mask:{mask.shape}") # mask:(1, 4, 39)

        mask = np.reshape(mask, [batch_size, tokens_per_step * num_steps])
        # mask:(1, 156), batch_size:1, tokens_per_step:39, num_steps:4
        # logger.info(f"  (2)mask:{mask.shape}, batch_size:{batch_size}, tokens_per_step:{tokens_per_step}, num_steps:{num_steps}") # 

        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        # logger.info(f"  mask:{mask.shape}")     # mask:torch.Size([1, 156])

        custom_causal_mask = None
        if self.spatial_tokens:
            # Temporal transformer by default assumes sequential causal relation.
            # This makes the transformer causal mask a lower triangular matrix.
            #     P1 P2 R  a  P1 P2 ... (Ps: image patches)
            # P1  1  0* 0  0  0  0
            # P2  1  1  0  0  0  0
            # R   1  1  1  0  0  0
            # a   1  1  1  1  0  0
            # P1  1  1  1  1  1  0*
            # P2  1  1  1  1  1  1
            # ... (0*s should be replaced with 1s in the ideal case)
            # But, when we have multiple tokens for an image (e.g. patch tokens, conv
            # feature map tokens, etc) as inputs to transformer, this assumption does
            # not hold, because there is no sequential dependencies between tokens.
            # Therefore, the ideal causal mask should not mask out tokens that belong
            # to the same images from each others.
            seq_len = token_emb.shape[1]
            sequential_causal_mask = np.tril(np.ones((seq_len, seq_len)))
            num_timesteps = seq_len // tokens_per_step
            num_non_obs_tokens = tokens_per_step - num_obs_tokens
            # logger.info(f"  seq_len:{seq_len}, num_timesteps:{num_timesteps}, num_non_obs_tokens:{num_non_obs_tokens}")
            # seq_len:156, num_timesteps:4, num_non_obs_tokens:3

            diag = [
                np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros((num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timesteps * 2)
            ]
            block_diag = scipy.linalg.block_diag(*diag)
            # logger.info(f"  block_diag:{block_diag.shape} ")     # block_diag:(156, 156) 

            custom_causal_mask = np.logical_or(sequential_causal_mask, block_diag)
            # logger.info(f"  (1)custom_causal_mask.shape:{custom_causal_mask.shape} ")    # custom_causal_mask.shape:(156, 156)

            custom_causal_mask = torch.tensor(custom_causal_mask, dtype=torch.bool, device=device)
            # logger.info(f"  (2)custom_causal_mask.shape:{custom_causal_mask.shape} ")   # custom_causal_mask.shape:torch.Size([156, 156]) 
 
        # logger.info(f"  token_emb.shape:{token_emb.shape} ")                    # token_emb.shape:torch.Size([1, 156, 1280])
        # logger.info(f"  mask.shape:{mask.shape} ")                              # mask.shape:torch.Size([1, 156])
        # logger.info(f"  custom_causal_mask.shape:{custom_causal_mask.shape} ")  # custom_causal_mask.shape:torch.Size([156, 156])

        output_emb = self.transformer(token_emb, mask, custom_causal_mask)      # Transformer.forward(1) called here

        # logger.info(f"  output_emb:{output_emb.shape} ")    # output_emb:torch.Size([1, 156, 1280]) 

        # Output_embeddings are [B x 3T x D].
        # Next token predictions (tokens one before their actual place).
        ret_pred = output_emb[:, (num_obs_tokens - 1) :: tokens_per_step, :]
        # logger.info(f"  (1)ret_pred.shape:{ret_pred.shape} ")      # ret_pred.shape:torch.Size([1, 4, 1280]) 
        act_pred = output_emb[:, (num_obs_tokens - 0) :: tokens_per_step, :]
        # logger.info(f"  (1)act_pred.shape:{act_pred.shape} ")      # act_pred.shape:torch.Size([1, 4, 1280])
        embeds = torch.cat([ret_pred, act_pred], dim=-1)
        # logger.info(f"  embeds.shape:{embeds.shape} ")      # embeds.shape:torch.Size([1, 4, 2560]) Note: 2560/1280 = 2

        # Project to appropriate dimensionality.
        # See L441; nn.Linear()
        ret_pred = self.ret_linear(ret_pred)
        # logger.info(f"  (2)ret_pred.shape:{ret_pred.shape} ")   
        # ret_pred.shape:torch.Size([1, 4, 120]) See run_atari.py L114 self.num_returns; RETURN_RANGE = [-20, 100]
        act_pred = self.act_linear(act_pred)
        # logger.info(f"  (2)act_pred.shape:{act_pred.shape} ") 
        # act_pred.shape:torch.Size([1, 4, 18])  See run_atari.py L111 self.num_actions; NUM_ACTIONS=18  

        # Return logits as well as pre-logits embedding.
        result_dict = {
            "embeds": embeds,
            "action_logits": act_pred,
            "return_logits": ret_pred,
        }
        if self.predict_reward:
            rew_pred = output_emb[:, (num_obs_tokens + 1) :: tokens_per_step, :]
            rew_pred = self.rew_linear(rew_pred)
            result_dict["reward_logits"] = rew_pred
        # Return evaluation metrics.
        result_dict["loss"] = self.sequence_loss(inputs, result_dict)
        result_dict["accuracy"] = self.sequence_accuracy(inputs, result_dict)
        return result_dict

    def _objective_pairs(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Get logit-target pairs for the model objective terms."""
        act_target = inputs["actions"]
        ret_target = encode_return(inputs["returns-to-go"], self.return_range)
        act_logits = model_outputs["action_logits"]
        ret_logits = model_outputs["return_logits"]
        # logger.info(f"  act_logits:{act_logits.shape} ret_logits:{ret_logits.shape}")

        if self.single_return_token:
            ret_target = ret_target[:, :1]
            ret_logits = ret_logits[:, :1, :]
        obj_pairs = [(act_logits, act_target), (ret_logits, ret_target)]
        if self.predict_reward:
            rew_target = encode_reward(inputs["rewards"])
            rew_logits = model_outputs["reward_logits"]
            obj_pairs.append((rew_logits, rew_target))
        return obj_pairs

    def sequence_loss(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the loss on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def sequence_accuracy(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the accuracy on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [accuracy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def optimal_action(
        self,
        inputs: Mapping[str, Tensor],
        return_range: Tuple[int] = (-100, 100),
        single_return_token: bool = False,
        opt_weight: Optional[float] = 0.0,
        num_samples: Optional[int] = 128,
        action_temperature: Optional[float] = 1.0,
        return_temperature: Optional[float] = 1.0,
        action_top_percentile: Optional[float] = None,
        return_top_percentile: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        deterministic: bool = False,
        torch_device: torch.device = None
    ):
        r"""Calculate optimal action for the given sequence model."""
        logits_fn = self.forward
        # logger.info(f"logits_fn:{logits_fn} ")


        obs, act, rew = inputs["observations"], inputs["actions"], inputs["rewards"]
        # logger.info(f"obs.shape:{obs.shape} act:{act} rew:{rew}")
        # obs.shape:torch.Size([2, 4, 1, 84, 84]) act:tensor([[1, 1, 1, 0],
        # [0, 0, 4, 0]], device='cuda:0') rew:tensor([[0., 0., 0., 0.],
        # [0., 0., 0., 0.]], device='cuda:0', dtype=torch.float64)
        

        assert len(obs.shape) == 5
        assert len(act.shape) == 2
        inputs = {
            "observations": obs,
            "actions": act,
            "rewards": rew,
            "returns-to-go": torch.zeros_like(act),
        }
        sequence_length = obs.shape[1]
        # Use samples from the last timestep.
        timestep = -1
        # A biased sampling function that prefers sampling larger returns.
        def ret_sample_fn(rng, logits):
            assert len(logits.shape) == 2
            # Add optimality bias.
            if opt_weight > 0.0:
                # Calculate log of P(optimality=1|return) := exp(return) / Z.
                logits_opt = torch.linspace(0.0, 1.0, logits.shape[1])
                logits_opt = torch.repeat_interleave(logits_opt[None, :], logits.shape[0], dim=0)
                # Sample from log[P(optimality=1|return)*P(return)].
                logits = logits + opt_weight * logits_opt
            logits = torch.repeat_interleave(logits[None, ...], num_samples, dim=0)
            ret_sample = sample_from_logits(
                logits,
                generator=rng,
                deterministic=deterministic,
                temperature=return_temperature,
                top_percentile=return_top_percentile,
            )

            # logger.info(f"logits:{logits} ")
            # logger.info(f"len(logits):{len(logits)} type:{type(logits)} ")  #len logits 128
            # logger.info(f"logits.shape:{logits.shape} ") # logits.shape:torch.Size([128, 2, 120])

            # Pick the highest return sample.
            ret_sample, _ = torch.max(ret_sample, dim=0)
            # Convert return tokens into return values.
            ret_sample = decode_return(ret_sample, return_range)
            return ret_sample

        # Set returns-to-go with an (optimistic) autoregressive sample.
        if single_return_token:
            # Since only first return is used by the model, only sample that (faster).
            ret_logits = logits_fn(inputs)["return_logits"][:, 0, :]        # MultiGameDecisionTransformer.forward(0) called here
            # logger.info(f"  ret_logits.shape:{ret_logits.shape}")   # ret_logits.shape:torch.Size([1, 120])

            ret_sample = ret_sample_fn(rng, ret_logits)
            inputs["returns-to-go"][:, 0] = ret_sample
        else:
            # Auto-regressively regenerate all return tokens in a sequence.
            ret_logits_fn = lambda input: logits_fn(input)["return_logits"]
            ret_sample = autoregressive_generate(
                ret_logits_fn,
                inputs,
                "returns-to-go",
                sequence_length,
                generator=rng,
                deterministic=deterministic,
                sample_fn=ret_sample_fn,
            )
            inputs["returns-to-go"] = ret_sample

        # --- Extract attention map(s)
        # logger.info(f"  X) Get Attention Map(s)")
        # self.get_attention_map()
        # self._np_attn_mean = attention_layers_mean(self._np_attn_container)

        # --- Visualize Attention
        # if self._np_rgb_img is not None:
        #     visualize_predict(self, self._np_rgb_img, self.img_size, self.patch_size, torch_device)


        # Generate a sample from action logits.
        act_logits = logits_fn(inputs)["action_logits"][:, timestep, :]
        act_sample = sample_from_logits(
            act_logits,
            generator=rng,
            deterministic=deterministic,
            temperature=action_temperature,
            top_percentile=action_top_percentile,
        )
        # logger.info(f"len(act_sample):{len(act_sample)}, type:{type(act_sample)}")

        return act_sample
