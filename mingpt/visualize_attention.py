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

#---------------------------------------------------------------
def convertToCV(tensor):
    tensor = torch.squeeze(tensor)
    tensor = tensor.cpu().float().detach()
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.permute(1, 2, 0)
    tensor = ((tensor +1)/2.0) * 255.0
    tensor = tensor.numpy()
    return tensor


def min_max(x, mins, maxs, axis=None):
    result = (x - mins)/(maxs - mins)
    return result


# def plot_attention(img, attention):
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

def plot_attention(attention):
    # n_heads = attention.shape[0]
    n_heads = attention.shape[1]
    logger.info(f"plot_attention:: attention.shape:{attention.shape} n_heads:{n_heads} type:{type(attention)}")

    for i in range(0, n_heads):
        logger.info(f"  attention[0]{i}.shape:{attention[0][i].shape} type:{type(attention[0][i])}")
        attn = attention[0][i]

        # logger.info(f"plot_attention:: attention[{i}].shape:{attention[i].shape} type:{type(attention[i])}")
        logger.info(f"  attn.shape:{attn.shape}")

        # img = attn[0]
        # logger.info(f"  img.shape:{img.shape} type:{type(img)}")
        numpy_image = convertToCV(attn)

        # Min max the image
        max_len = numpy_image.max(axis=None, keepdims=True)
        min_len = numpy_image.min(axis=None, keepdims=True)
        numpy_image = min_max(numpy_image, min_len, max_len)

        # Appy color map
        numpy_image = numpy_image * 255.
        numpy_image = cv2.applyColorMap(numpy_image.astype(np.uint8), cv2.COLORMAP_INFERNO )

        # numpy_image = img.argmax(1).cpu().numpy()
        logger.info(f"  numpy_image.shape:{numpy_image.shape} type:{type(numpy_image)}")

        cv2.imshow(f"attn_head_{i}", numpy_image)
        cv2.waitKey(500) 
    # time.sleep(20) # in seconds
    logger.info("Closing plot_attention()...")



def plot_attention2(attention):
    # n_heads = attention.shape[0]
    n_heads = attention.shape[1]
    # logger.info(f"plot_attention:: attention.shape:{attention.shape} n_heads:{n_heads} type:{type(attention)}") # attention.shape:torch.Size([2, 20, 156, 156])
    attention_array = attention[0]
    # logger.info(f"  B41: attention_array.shape:{attention_array.shape} ") 
    # numpy_image_array = np.mean(numpy_image_array, axis=(0,1))
    attention_array = torch.mean(attention_array, axis=0)
    # logger.info(f"  AFT1: numpy_image_array.shape:{numpy_image_array.shape} numpy_image_array:{numpy_image_array}") 
    # logger.info(f"  AFT1: attention_array.shape:{attention_array.shape} ")


    # View individual (of the 20) patches
    for i in range(0, n_heads):
        # logger.info(f"  attention[0]{i}.shape:{attention[0][i].shape} type:{type(attention[0][i])}")
        attn = attention[0][i]

        # logger.info(f"plot_attention:: attention[{i}].shape:{attention[i].shape} type:{type(attention[i])}")
        # logger.info(f"  attn.shape:{attn.shape}")

        # img = attn[0]
        # logger.info(f"  img.shape:{img.shape} type:{type(img)}")
        numpy_image = convertToCV(attn)
        # numpy_image = attn.argmax(0).cpu().numpy()

        # Min max the image
        max_len = numpy_image.max(axis=None, keepdims=True)
        min_len = numpy_image.min(axis=None, keepdims=True)
        numpy_image = min_max(numpy_image, min_len, max_len)
        # logger.info(f"  numpy_image.shape:{numpy_image.shape}")

        # Appy color map
        numpy_image = numpy_image * 255.
        numpy_image = cv2.applyColorMap(numpy_image.astype(np.uint8), cv2.COLORMAP_INFERNO )
        # logger.info(f"  numpy_image.shape:{numpy_image.shape}")

        # Add to the array
        # np.append(numpy_image_array, numpy_image)
        # numpy_image_array = np.dstack((numpy_image_array, numpy_image))
        # numpy_image_array = np.stack((numpy_image_array, numpy_image),axis=2)
        # logger.info(f"  numpy_image.shape      :{numpy_image.shape} ")
        # logger.info(f"  numpy_image_array.shape:{numpy_image_array.shape} ")

        # cv2.imshow(f"attn_head_{i}", numpy_image)
        # cv2.imshow(f"attn_head", numpy_image)
        # cv2.waitKey(500) 

    # View singular mean (of the 20) patches
    numpy_image_array = convertToCV(attention_array)
    # logger.info(f"  B42: numpy_image_array.shape:{numpy_image_array.shape} ")
    max_len = numpy_image_array.max(axis=None, keepdims=True)
    min_len = numpy_image_array.min(axis=None, keepdims=True)
    numpy_image_array = min_max(numpy_image_array, min_len, max_len)
    numpy_image = numpy_image * 255.
    numpy_image = cv2.applyColorMap(numpy_image.astype(np.uint8), cv2.COLORMAP_INFERNO )
    # logger.info(f"  AFT2: numpy_image_array.shape:{numpy_image_array.shape} ")

    # Plot the mean (single patch)
    # cv2.imshow(f"numpy_image_array", numpy_image_array)
    # cv2.waitKey(2000) 
    # cv2.destroyAllWindows()

    # logger.info("Closing plot_attention()...")

    # Returns the mean; numpy_image_array.shape:(156, 156, 1)
    return numpy_image