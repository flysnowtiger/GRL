#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @time:2019/6/5下午3:12
# @Author: Yu Ci

import torch
import torch.nn.functional as F

import numpy as np
import cv2


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)  # torch.Size([1, 1, 256, 128])
    cam = 255 * cam.squeeze()  # torch.Size([256, 128])
    cam = cam.detach().cpu()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)  # <class 'tuple'>: (256, 128, 3)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))  # torch.Size([3, 256, 128])
    heatmap = heatmap.float() / 255  # torch.Size([3, 256, 128])
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])  # torch.Size([3, 256, 128])

    result = heatmap + img.cpu()
    # result = heatmap
    result = result.div(result.max())

    return result, img.cpu()


def visualize2(cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = cam.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    cam = cam.detach().cpu()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap
    result = result.div(result.max())

    return result