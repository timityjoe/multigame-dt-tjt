# From 
# https://medium.com/@aryanjadon/visualizing-attention-in-vision-transformer-c871908d86de
# Explanation of hierarchy
# 1 MGDT
# img_size:(84, 84) patch_size:(14, 14) num_actions:18 num_rewards:4 return_range:[-20, 100]
# d_model:1280 num_layers:10 dropout_rate:0.1 conv_dim:256
#
# 1 MGDT has 1 transformer, and each transformer has:
#       - 10 layers/block, each layers/block has:
#               - 1 CausalSelfAttention, each block has:
#                       - 20 attention heads 
#                       -

# From
# https://github.com/aryan-jadon/Medium-Articles-Notebooks
# https://ai.plainenglish.io/visualizing-attention-in-vision-transformer-c871908d86de


# import os
# import torch
import numpy as np
# import math
# import torch
import torch.nn as nn

# import ipywidgets as widgets
# import io
# from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import warnings
warnings.filterwarnings("ignore")


def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

def visualize_predict(model, img, img_size, patch_size, device):
    print(f"visualize_predict()")
    img_pre = transform(img, img_size)
    attention = visualize_attention(model, img_pre, patch_size, device)
    print(f"    attention.shape:{attention.shape}")
    # plot_attention(img, attention)


def visualize_attention(model, img, patch_size, device):
    print(f"visualize_attention() patch_size:{patch_size}, device:{device}")
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    print(f"    img.shape:{img.shape}")

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    print(f"    w_featmap:{w_featmap} h_featmap:{h_featmap}")

    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]  # number of head
    print(f"    1)attentions.shape:{attentions.shape}, nh:{nh}")

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(f"    2)attentions.shape:{attentions.shape}")

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    print(f"    3)attentions.shape:{attentions.shape}")

    return attentions


def plot_attention(img, attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()
