# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:06:05 2024

@author: Debra Hogue

Based on the backbone.py from RankNet y Lv et al.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax


"""
===================================================================================================
    Generator Class
    
    This class implements a generator for a CODS, utilizing a Saliency Feature Encoder
    (Saliency_feat_encoder) to generate predictions.

    Attributes:
        - sal_encoder (Saliency_feat_encoder): Instance of the Saliency Feature Encoder for feature
          extraction and prediction generation.

    Methods:
        - __init__(self, channel): Constructor method for initializing the generator with a channel
          parameter.
        - forward(self, x): Forward pass through the generator to generate fixation predictions,
          cod_pred1, and cod_pred2.
    
===================================================================================================
"""
class Generator(nn.Module):
    def __init__(self, channel=64):
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel)

    def forward(self, x):
        self.sal_feat = self.sal_encoder(x)
        return self.sal_feat


"""
===================================================================================================
    Channel Attention Module (CAM) Class
    Paper: Dual Attention Network for Scene Segmentation
    
    This class implements a Channel Attention Module (CAM) based on the Dual Attention Network for
    Scene Segmentation.

    Attributes:
        - gamma (Parameter): Learnable parameter for adjusting the attention value.
        - softmax (Softmax): Softmax function for computing attention weights.
    
    Methods:
        - __init__(self): Constructor method to initialize the CAM_Module.
        - forward(self, x): Forward pass through the CAM_Module to compute attention values for input
          feature maps x.
        - get_feature_maps(self): Method to retrieve the feature maps stored during the forward pass
          for visualization or analysis purposes.
    
    Example usage:
        cam = CAM_Module()
        attention_output = cam(input_features)
        feature_maps = cam.get_feature_maps()
    
===================================================================================================
"""
class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        
        self.feature_maps = {}  # Dictionary to store feature maps
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        
        # Store feature maps in the dictionary
        self.feature_maps['input_feature'] = x
        self.feature_maps['attention_output'] = out
        
        return out

    def get_feature_maps(self):
        return self.feature_maps
    

"""
===================================================================================================
    Position Attention Module (PAM) Class
    Paper: Dual Attention Network for Scene Segmentation
    
    This class implements the Position Attention Module (PAM) as described in the paper "Dual
    Attention Network for Scene Segmentation". It computes attention values and enhances input
    feature maps based on query, key, and value convolutions.

    Attributes:
        - in_dim (int): Dimension of the input feature maps.
        - query_conv (nn.Conv2d): Convolution layer for queries in the attention mechanism.
        - key_conv (nn.Conv2d): Convolution layer for keys in the attention mechanism.
        - value_conv (nn.Conv2d): Convolution layer for values in the attention mechanism.
        - gamma (Parameter): Learnable parameter for scaling the attention output.
        - softmax (Softmax): Softmax layer for computing attention scores.

    Methods:
        - __init__(self, in_dim): Constructor method to initialize the PAM module with the input
          dimension.
        - forward(self, x): Forward pass through the PAM module to compute attention values and
          enhance input feature maps.
        - get_feature_maps(self): Method to retrieve the stored feature maps during the forward pass.
    
===================================================================================================
"""
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
        self.feature_maps = {}  # Dictionary to store feature maps

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        
        # Store feature maps in the dictionary
        self.feature_maps['input_feature'] = x
        self.feature_maps['attention_output'] = out
            
        return out

    def get_feature_maps(self):
        return self.feature_maps


"""
===================================================================================================
    Classifier Module Class
    
    This class implements the Classifier Module, which consists of a series of dilated convolutional
    layers for semantic segmentation tasks. Each convolutional layer has different dilation rates and
    padding sizes to capture multi-scale contextual information.

    Attributes:
        - dilation_series (list): List of dilation rates for the dilated convolutional layers.
        - padding_series (list): List of padding sizes for the dilated convolutional layers.
        - NoLabels (int): Number of output labels for classification.
        - input_channel (int): Number of input channels to the classifier module.
        - conv2d_list (nn.ModuleList): Module list containing the dilated convolutional layers.

    Methods:
        - __init__(self, dilation_series, padding_series, NoLabels, input_channel): Constructor
          method to initialize the Classifier Module with the specified parameters.
        - forward(self, x): Forward pass through the Classifier Module to compute classification
          scores based on the input feature maps.
    
===================================================================================================
"""
class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


"""
===================================================================================================
    Channel Attention (CA) Layer Class
    
    This class implements the Channel Attention (CA) Layer, which performs channel-wise attention
    mechanism by learning channel weights based on the global average pooling of input feature maps.
    It is used to enhance the representation of important channels and suppress less relevant ones.

    Attributes:
        - channel (int): Number of input channels to the CA Layer.
        - reduction (int): Reduction factor for channel dimensionality reduction.
        - avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer for global average pooling.
        - conv_du (nn.Sequential): Sequential module containing convolutional layers for channel
          dimensionality reduction and sigmoid activation for channel weight computation.
        - feature_maps (dict): Dictionary to store intermediate feature maps during forward pass.

    Methods:
        - __init__(self, channel, reduction=16): Constructor method to initialize the CA Layer with
          the specified number of input channels and reduction factor.
        - forward(self, x): Forward pass through the CA Layer to compute channel attention weights
          and apply channel-wise attention to the input feature maps.
        - get_feature_maps(self): Method to retrieve the stored feature maps for visualization or
          analysis purposes.
    
===================================================================================================
"""
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        
        self.feature_maps = {}  # Dictionary to store feature maps

    def forward(self, x):
        y = self.avg_pool(x)
        
        # Store feature maps in the dictionary
        self.feature_maps['input_feature'] = x
        self.feature_maps['global_avg_pooling'] = y
        self.feature_maps['channel_weights'] = y
        
        y = self.conv_du(y)
        
        return x * y

    def get_feature_maps(self):
        return self.feature_maps
    

"""
===================================================================================================
    Residual Channel Attention Block (RCAB) Class
    Paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    
    This class implements the Residual Channel Attention Block (RCAB) as proposed in the paper
    "Image Super-Resolution Using Very Deep Residual Channel Attention Networks". It consists of
    convolutional layers, optional batch normalization, activation function, and a Channel Attention
    (CA) Layer to enhance feature representation and preserve important channel information.
    
    Input: B*C*H*W
    Output: B*C*H*W
    
    Attributes:
        - n_feat (int): Number of input and output feature channels.
        - kernel_size (int): Size of the convolutional kernel.
        - reduction (int): Reduction factor for channel dimensionality reduction in the CA Layer.
        - bias (bool): Whether to include bias in convolutional layers.
        - bn (bool): Whether to use batch normalization.
        - act (nn.Module): Activation function applied after the first convolutional layer.
        - res_scale (float): Residual scaling factor for the output.

    Methods:
        - __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False,
          act=nn.ReLU(True), res_scale=1): Constructor method to initialize the RCAB with
          specified parameters.
        - default_conv(self, in_channels, out_channels, kernel_size, bias=True): Helper method
          to create a default convolutional layer.
        - forward(self, x): Forward pass through the RCAB to compute the residual output after
          applying channel attention and optional activation function.
        - get_feature_maps(self): Method to retrieve the stored feature maps for visualization or
          analysis purposes.
    
===================================================================================================
"""
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        
        self.feature_maps = {}  # Dictionary to store feature maps

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
                
        # Store feature maps in the dictionary
        self.feature_maps['input_feature'] = x
        self.feature_maps['output_feature'] = res
        self.feature_maps['residual_connection'] = x + res
        
        res += x
        
        return res

    def get_feature_maps(self):
        return self.feature_maps
    

"""
===================================================================================================
    Saliency Feature Encoder Class
    
    This class implements a feature encoder module for saliency prediction, based on a ResNet-based
    architecture.

    Attributes:
        - channel (int): Number of input channels. Default is 32.

    Methods:
        - __init__(self, channel=32): Constructor method to initialize the Saliency_feat_encoder with
          the specified number of input channels. Default channel size is 32.
        - forward(self, x): Forward pass through the Saliency_feat_encoder to compute the saliency
          prediction based on input images x.
        - initialize_weights(self): Method to initialize the weights of the encoder using pretrained
          ResNet-50 weights.
        - get_feature_maps(self): Method to retrieve the feature maps stored during the forward pass
          for visualization or analysis purposes.
    
    Example usage:
        encoder = Saliency_feat_encoder(channel=64)
        features = encoder.get_feature_maps()
    
===================================================================================================
"""
class Saliency_feat_encoder(nn.Module):
    def __init__(self, channel=32):
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        self.conv1 = nn.Conv2d(256, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1024, channel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2048, channel, kernel_size=3, padding=1)

        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cam_attention = CAM_Module()
        self.pam_attention = PAM_Module(channel)
        self.racb_layer = RCAB(channel * 4)
        
        if self.training:
            self.initialize_weights()
            
        self.feature_maps = {}  # Dictionary to store feature maps

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8
        
        # Store feature maps in the dictionary
        self.feature_maps['x1'] = x1
        self.feature_maps['x2'] = x2
        self.feature_maps['x3'] = x3
        self.feature_maps['x4'] = x4

        conv5_feat = self.layer5(x4)
        
        # put position attention at bottleneck of network
        conv5_feat = self.pam_attention(conv5_feat)
        
        # Store feature maps in the dictionary
        self.feature_maps['conv5_feat'] = conv5_feat
        
        # put channel attention at bottleneck of network
        # conv5_feat = self.cam_attention(conv5_feat)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.upsample2(self.conv3(x3))
        conv4_feat = self.upsample2(self.conv4(x4))
        conv5_feat = self.upsample2(conv5_feat)
        
        # Store feature maps in the dictionary
        self.feature_maps['conv2_feat'] = conv2_feat
        self.feature_maps['conv3_feat'] = conv3_feat
        self.feature_maps['conv4_feat'] = conv4_feat
        self.feature_maps['conv5_feat_upsampled'] = conv5_feat

        cat_feat = self.relu(torch.cat((conv2_feat, conv3_feat, conv4_feat, conv5_feat), 1))

        # residual channel attention
        cat_feat = self.racb_layer(cat_feat)
        cat_feat = self.layer6(cat_feat)
        
        # Store feature maps in the dictionary
        self.feature_maps['cat_feat'] = cat_feat

        return self.upsample(cat_feat)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

    def get_feature_maps(self):
        return self.feature_maps