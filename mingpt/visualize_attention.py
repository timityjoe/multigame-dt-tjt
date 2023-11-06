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


import numpy as np
import torch
import torch.nn as nn
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

def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img

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


def visualize_attn(attention, index):
    # n_heads = attention.shape[1]
    # logger.info(f"plot_attention:: attention.shape:{attention.shape} n_heads:{n_heads} type:{type(attention)}")
    cv2.imshow(f"attn layer{index}", attention)
    cv2.waitKey(500) 
    cv2.destroyAllWindows()

#---------------------------------------------------------------

def attention_patches_mean(attention):
    # n_heads = attention.shape[0]
    n_heads = attention.shape[1]
    # logger.info(f"plot_attention:: attention.shape:{attention.shape} n_heads:{n_heads} type:{type(attention)}") # attention.shape:torch.Size([2, 20, 156, 156])
    attention_array = attention[0]
    # logger.info(f"  B41: attention_array.shape:{attention_array.shape} ") 
    # numpy_image_array = np.mean(numpy_image_array, axis=(0,1))
    attention_array = torch.mean(attention_array, axis=0) # ([156, 156])
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
    np_image = convertToCV(attention_array)
    # logger.info(f"  B42: numpy_image_array.shape:{numpy_image_array.shape} ")
    max_len = np_image.max(axis=None, keepdims=True)
    min_len = np_image.min(axis=None, keepdims=True)
    np_image = min_max(np_image, min_len, max_len)
    np_image = np_image * 255.
    np_image = cv2.applyColorMap(np_image.astype(np.uint8), cv2.COLORMAP_INFERNO )
    # logger.info(f"  AFT2: numpy_image_array.shape:{numpy_image_array.shape} ")

    # # Check:
    # max_len_aft = np_image.max(axis=None, keepdims=True)
    # min_len_aft = np_image.min(axis=None, keepdims=True)
    # logger.info(f"  max_len:{max_len} max_len_aft:{max_len_aft} min_len:{min_len} min_len_aft:{min_len_aft}")
    # logger.info(f"  np_image.shape:{np_image.shape} type:{type(np_image)}")

    # Plot the mean (single patch)
    # cv2.imshow(f"np_image", np_image)
    # cv2.waitKey(2000) 
    # cv2.destroyAllWindows()

    # logger.info("Closing plot_attention()...")

    # Returns the mean; numpy_image_array.shape:(156, 156, 1)
    return np_image


#---------------------------------------------------------------
def attention_layers_mean(_np_attn_container):
    n_heads = _np_attn_container.shape[0]
    # logger.info(f"B4 ::attention.shape:{_np_attn_container.shape}, n_heads:{n_heads} ") # attention.shape:(156, 156, 3, 20)

    # View individual (of the 20) patches
    for i in range(0, n_heads):
        attn = _np_attn_container[i]     
        max_len = attn.max(axis=None, keepdims=True)
        min_len = attn.min(axis=None, keepdims=True)
        logger.info(f"  max_len:{max_len} min_len:{min_len}")
        logger.info(f"  attn.shape:{attn.shape} ")        
        cv2.imshow(f"attn_{i}", attn)
        cv2.waitKey(500) 


    np_image = np.mean(_np_attn_container, axis=0)
    # logger.info(f"AFT::np_image.shape:{np_image.shape} ")

    # tensor_image = torch.from_numpy(_np_attn_container)
    # logger.info(f"B4 ::tensor_image.shape:{tensor_image.shape}, type:{type(tensor_image)} ")
    # tensor_image = torch.mean(tensor_image, axis=3)
    # logger.info(f"AFT::tensor_image.shape:{tensor_image.shape} ")
    # np_image = tensor_image.numpy()

    # View singular mean (of the 10) patches
    # logger.info(f"  B42: numpy_image_array.shape:{numpy_image_array.shape} ")
    max_len = np_image.max(axis=None, keepdims=True)
    min_len = np_image.min(axis=None, keepdims=True)
    np_image = min_max(np_image, min_len, max_len)
    np_image = np_image * 255.
    np_image = cv2.applyColorMap(np_image.astype(np.uint8), cv2.COLORMAP_INFERNO )
    # logger.info(f"  max_len:{max_len} min_len:{min_len}")
    # logger.info(f"  np_image.shape:{np_image.shape}")

    # Plot the mean (single patch)
    # cv2.imshow(f"_np_attn_container", np_image)
    # cv2.waitKey(500) 
    # cv2.destroyAllWindows()

    # Returns the mean; numpy_image_array.shape:(156, 156, 1)
    return np_image


#---------------------------------------------------------------
def visualize_attn_heatmap(model, img, patch_size, device):
    print(f"visualize_attention()")
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

    np_attentions = attentions.cpu().numpy()
    visualize_attn_np(np_attentions, "np_attentions")

    return attentions

def visualize_attn_np(np_attention, string):
    # n_heads = attention.shape[1]
    np_attention = np_attention * 255.
    np_attention = cv2.applyColorMap(np_attention.astype(np.uint8), cv2.COLORMAP_INFERNO )
    logger.info(f"visualize_attn_np:{np_attention.shape} type:{type(np_attention)}")
    cv2.imshow(f"{string}", np_attention)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()