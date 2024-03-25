# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:23:26 2024

@author: Debra Hogue

Based on HolisticAttention.py from RankNet by Lv et al.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import scipy.stats as st

"""
===================================================================================================
    
    Generate a 2D Gaussian kernel with the specified length and standard deviation.

    Args:
        - kernlen (int): Length of the kernel.
        - nsig (float): Standard deviation of the Gaussian distribution.

    Returns:
        - kernel (ndarray): 2D Gaussian kernel normalized to sum to 1.

    Example usage:
        kernel = gkern(kernlen=16, nsig=3)
    
    This function calculates a 2D Gaussian kernel using the cumulative distribution function of
    a standard normal distribution. The resulting kernel is normalized to ensure its elements sum
    to 1, making it suitable for image processing tasks such as blurring or filtering.
    
===================================================================================================
"""
def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


"""
===================================================================================================
    
    Perform min-max normalization on the input tensor along specified dimensions.

    Args:
        - in_ (Tensor): Input tensor to be normalized.

    Returns:
        - Tensor: Min-max normalized tensor.

    Example usage:
        normalized_tensor = min_max_norm(input_tensor)

    This function calculates the min-max normalization of a given input tensor along the specified
    dimensions. It scales the tensor values to a range of [0, 1] based on the minimum and maximum
    values along the specified dimensions. Small epsilon (1e-8) is added to the denominator to avoid
    division by zero.
    
===================================================================================================
"""
def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


"""
===================================================================================================
    Holistic Attention (HA) Module
    
    This module implements a holistic attention mechanism using a Gaussian kernel and min-max
    normalization to weight the input feature maps.
    
    The HA module applies a holistic attention mechanism by convolving the input attention map with
    a Gaussian kernel and then performing min-max normalization on the resulting soft attention
    map. The weighted soft attention is then used to modulate the input feature maps, enhancing
    relevant information based on the attention weights.
    
    Attributes:
        - gaussian_kernel (Parameter): Parameter containing the Gaussian kernel weights.

    Methods:
        - __init__(self): Constructor method to initialize the HA module with a Gaussian kernel.
        - forward(self, attention, x): Forward pass through the HA module to apply attention
          weighting on the input feature maps.

    Example usage:
        ha_module = HA()
        weighted_features = ha_module(attention_map, input_features)
    
===================================================================================================
"""
class HA(nn.Module):
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x