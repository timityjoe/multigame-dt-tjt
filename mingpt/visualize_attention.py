# From 
# https://medium.com/@aryanjadon/visualizing-attention-in-vision-transformer-c871908d86de

# import os
# import torch
import numpy as np
# import math
# from functools import partial
import torch
import torch.nn as nn

# import ipywidgets as widgets
# import io
# from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import cv2

import warnings
warnings.filterwarnings("ignore")

import time
from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

# Main call
def visualize_predict(model, img, img_size, patch_size, device):
    img_pre = transform(img, img_size)
    attention = visualize_attention(model, img_pre, patch_size, device)
    plot_attention(img, attention)

#---------------------------------------------------------------
def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

def convertToCV(tensor):
    tensor = torch.squeeze(tensor)
    tensor = tensor.cpu().float().detach()
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.permute(1, 2, 0)
    tensor = ((tensor +1)/2.0) * 255.0
    tensor = tensor.numpy()
    return tensor

# def plot_attention(img, attention):
def plot_attention(attention):
    # n_heads = attention.shape[0]
    n_heads = attention.shape[1]
    logger.info(f"plot_attention:: attention.shape:{attention.shape} n_heads:{n_heads} type:{type(attention)}")

    # plt.figure(figsize=(10, 10))
    # text = ["Original Image", "Head Mean"]
    # for i, fig in enumerate([img, np.mean(attention, 0)]):
    #     plt.subplot(1, 2, i+1)
    #     plt.imshow(fig, cmap='inferno')
    #     plt.title(text[i])
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # for i in range(n_heads):
    #     plt.subplot(n_heads//3, 3, i+1)
    #     plt.imshow(attention[i], cmap='inferno')
    #     plt.title(f"Head n: {i+1}")
    # plt.tight_layout()
    # plt.show()

    for i in range(n_heads):
        # numpy_image = attention[i].argmax(1).cpu().numpy()
        # cv2_image = np.transpose(numpy_image, (1, 2, 0))
        # cv2.imshow(f"attn_head_{i}", numpy_image)
        attn = attention[i]

        # logger.info(f"plot_attention:: attention[{i}].shape:{attention[i].shape} type:{type(attention[i])}")
        logger.info(f"plot_attention:: attn.shape:{attn.shape}")

        numpy_image = convertToCV(attn[1])
        logger.info(f"plot_attention:: numpy_image.shape:{numpy_image.shape} type:{type(numpy_image)}")

        cv2.imshow(f"attn_head_{i}", numpy_image)
        cv2.waitKey(2000) 
    # time.sleep(20) # in seconds
    logger.info("Closing plot_attention()...")