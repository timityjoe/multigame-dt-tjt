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
# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim,
#                               kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

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
        # logger.info(f"  (1)attn.shape:{attn.shape}")    # attn.shape:torch.Size([1, 20, 156, 156])

        touchpoint = attn

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max  # max_neg_value
            attn = attn.masked_fill(~mask.to(dtype=torch.bool), mask_value)

        attn = attn.softmax(dim=-1)
        # logger.info(f"  (2)attn.shape:{attn.shape}")    # attn.shape:torch.Size([1, 20, 156, 156])


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
        # See L460
        # https://sites.google.com/view/multi-game-transformers
        copy2 = touchpoint
        # logger.info(f"  x1.shape:{x1.shape}")   # x1.shape:torch.Size([1, 20, 156, 156])

        # keep only the output patch attention
        # copy2 = copy2[0, :, 0, :]
        copy2 = copy2[0, :, :, 0]
        # logger.info(f"    (1)copy2.shape:{copy2.shape}")   # copy2.shape:torch.Size([20, 156])

        # Change from [20, 156] to [156]
        # copy2 = torch.mean(copy2, dim=0)
        # Or, select by layer
        copy2 = copy2[render_attn_head_id]
        # [156] to [4, 39]
        copy2 = copy2.reshape(4, 39)

        # [4, 39] to [39]
        copy2 = torch.mean(copy2, dim=0)
        # Or, select latest layer
        # copy2 = copy2[3]

        # [39] to [36]
        copy2 = copy2[0:36]
        # logger.info(f"    (2)copy2.shape:{copy2.shape}")


        # input of size 3120
        w_featmap = 6
        h_featmap = 6
        copy2 = copy2.reshape(w_featmap, h_featmap)
        # logger.info(f"    (3)copy2.shape:{copy2.shape}")   # 

        # copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0).unsqueeze(0), [210, 160], mode="nearest").view(210, 160, 1)
        # copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0), scale_factor=16, mode="nearest")[0]
        # copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0), [210, 160], mode="nearest")[0]

        copy2 = torch.nn.functional.interpolate(copy2.unsqueeze(0).unsqueeze(0), [210, 160], mode="nearest").view(210, 160, 1)
        # logger.info(f"    (4)copy2.shape:{copy2.shape}")    # copy2.shape:torch.Size([20, 210, 160])
        # copy2 = copy2[render_attn_head_id] 
        # logger.info(f"    (3)copy2.shape:{copy2.shape}") 

        np_copy = copy2.cuda().detach().cpu().clone().numpy()

        np_copy = np_copy / 5.
        # logger.info(f"  np_copy:{np_copy}")
        # Subtract 0.5 from tensor
        # np_copy = np.subtract(np_copy, 0.9)

        np_copy = cv2.applyColorMap(np_copy.astype(np.uint8), cv2.COLORMAP_INFERNO )
        # logger.info(f"  np_copy.shape:{np_copy.shape}")     # np_copy.shape:(39, 39, 3)

        if (layer_id == render_layer_id):
            self._np_attn_heatmap = np_copy
            # visualize_attn_np(self._np_attn_heatmap, f"np_{i}")

        #------------------------------------------------------
        # x1 = x1.softmax(dim=-1)
        # np_x1 = x1.detach().cpu().numpy()
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
