# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:53:16 2024

@author: Debra Hogue

Modified B2_ResNet class from RankNet - Lv et al.
"""

import torch.nn as nn
import math


"""
===================================================================================================
    
    Creates a 3x3 convolutional layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution operation. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 convolutional layer with the specified parameters.
    
===================================================================================================
"""
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


"""
===================================================================================================
    Basic Block Class
    
    This class defines a basic residual block for ResNet architectures. It includes two convolutional
    layers with batch normalization and ReLU activation.

    Attributes:
        - expansion (int): Expansion factor for the basic block.

    Methods:
        - __init__(self, inplanes, planes, stride=1, downsample=None): Constructor method for 
          initializing the basic block.
        - forward(self, x): Forward pass through the basic block.
    
===================================================================================================
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


"""
===================================================================================================
    Bottleneck Class
    
    This class defines the bottleneck block used in the ResNet architecture. It consists of three 
    convolutional layers: 1x1, 3x3, and 1x1, with batch normalization and ReLU activation after each 
    convolutional layer. The skip connection is added before the final ReLU activation.

    Parameters:
        - inplanes (int): Number of input channels.
        - planes (int): Number of output channels.
        - stride (int, optional): Stride for the convolutional layers (default is 1).
        - downsample (nn.Module, optional): Downsample layer for the skip connection (default is None).

    Attributes:
        - expansion (int): Expansion factor for the bottleneck block.

    Methods:
        - forward(x): Forward pass through the bottleneck block.
    
===================================================================================================
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


"""
===================================================================================================
    B2 ResNet Class
    
    This class defines a modified ResNet50 architecture with two branches. It includes layers for 
    feature extraction and downsampling.

    Attributes:
        - expansion (int): Expansion factor for the bottleneck block.

    Methods:
        - __init__(): Constructor method for initializing the B2_ResNet architecture.
        - _make_layer(block, planes, blocks, stride=1): Helper method to create a layer in the 
          ResNet architecture.
        - forward(x): Forward pass through the B2_ResNet architecture.
        - get_feature_maps(): Method to retrieve the saved intermediate feature maps.
    
===================================================================================================
"""
class B2_ResNet(nn.Module):
    expansion = 4
    
    # ResNet50 with two branches
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64
        super(B2_ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.inplanes = 512
        self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        self.feature_maps = {}  # Dictionary to store feature maps
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        self.feature_maps['layer1'] = x.clone().detach()  # Save feature map after layer 1
        x = self.layer2(x)
        self.feature_maps['layer2'] = x.clone().detach()  # Save feature map after layer 2
        x1 = self.layer3_1(x)
        self.feature_maps['layer3_1'] = x1.clone().detach()  # Save feature map after layer 3_1
        x1 = self.layer4_1(x1)
        self.feature_maps['layer4_1'] = x1.clone().detach()  # Save feature map after layer 4_1

        x2 = self.layer3_2(x)
        self.feature_maps['layer3_2'] = x2.clone().detach()  # Save feature map after layer 3_2
        x2 = self.layer4_2(x2)
        self.feature_maps['layer4_2'] = x2.clone().detach()  # Save feature map after layer 4_2

        return x1, x2

    def get_feature_maps(self):
        return self.feature_maps