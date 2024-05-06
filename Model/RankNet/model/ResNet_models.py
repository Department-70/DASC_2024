# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:38:21 2024

@author: Debra Hogue

Modified ResNet_models.py from RankNet by Lv et al.g
"""

import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
# from torch.autograd import Variable
# from torch.distributions import Normal, Independent, kl
# import numpy as np


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
    def __init__(self, channel):
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel)


    def forward(self, x):
        fix_pred, cod_pred1, cod_pred2 = self.sal_encoder(x)
        fix_pred = F.upsample(fix_pred, size=(x.shape[2], x.shape[3]), mode='bilinear',
                                        align_corners=True)
        cod_pred1 = F.upsample(cod_pred1, size=(x.shape[2], x.shape[3]), mode='bilinear',
                              align_corners=True)
        cod_pred2 = F.upsample(cod_pred2, size=(x.shape[2], x.shape[3]), mode='bilinear',
                              align_corners=True)
        feature_maps = self.sal_encoder.get_feature_maps()
        return fix_pred, cod_pred1, cod_pred2, feature_maps


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
    Basic Convolution 2D Class
        
    This class implements a basic 2D convolutional layer followed by batch normalization.
    It is commonly used as a building block in various neural network architectures.

    Attributes:
        - in_planes (int): Number of input channels.
        - out_planes (int): Number of output channels.
        - kernel_size (int or tuple): Size of the convolutional kernel.
        - stride (int or tuple, optional): Stride for the convolution operation. Default is 1.
        - padding (int or tuple, optional): Padding added to the input during convolution. Default is 0.
        - dilation (int or tuple, optional): Dilation factor for the convolution kernel. Default is 1.

    Methods:
        - __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1): Constructor
          method to initialize the BasicConv2d with specified parameters.
        - forward(self, x): Forward pass through the BasicConv2d layer to compute the output after
          convolution and batch normalization.
    
===================================================================================================
"""
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


"""
===================================================================================================
    Triple Convolution Class
    
    This class implements a sequence of three convolutional layers with 1x1, 3x3, and 3x3 kernel sizes,
    respectively, followed by batch normalization in each layer.

    Attributes:
        - in_channel (int): Number of input channels.
        - out_channel (int): Number of output channels.

    Methods:
        - __init__(self, in_channel, out_channel): Constructor method to initialize the Triple_Conv
          with specified input and output channel sizes.
        - forward(self, x): Forward pass through the Triple_Conv layer to compute the output after
          the sequence of three convolutional operations and batch normalization.
    
===================================================================================================
"""
class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


"""
===================================================================================================
    Saliency Feature Decoder Class
    
    This class implements a feature decoder module for saliency prediction, based on a ResNet-based
    encoder-decoder architecture.

    Attributes:
        - channel (int): Number of input channels.

    Methods:
        - __init__(self, channel): Constructor method to initialize the Saliency_feat_decoder with
          the specified number of input channels.
        - _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
          Helper method to create prediction layers with specified configurations.
        - forward(self, x1, x2, x3, x4): Forward pass through the Saliency_feat_decoder to compute
          the saliency prediction based on input feature maps x1, x2, x3, and x4.
        - get_feature_maps(self): Method to retrieve the feature maps stored during the forward pass
          for visualization or analysis purposes.
    
===================================================================================================
"""
class Saliency_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*4)
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2*channel)
        self.conv432 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 3*channel)
        self.conv4321 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 4*channel)

        self.cls_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        self.feature_maps = {}  # Dictionary to store feature maps
        

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x1,x2,x3,x4):

        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)
        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat),1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)
        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)

        sal_pred = self.cls_layer(conv4321)

        # Store feature maps in the dictionary
        self.feature_maps['conv1_feat'] = conv1_feat
        self.feature_maps['conv2_feat'] = conv2_feat
        self.feature_maps['conv3_feat'] = conv3_feat
        self.feature_maps['conv4_feat'] = conv4_feat
        self.feature_maps['conv43'] = conv43
        self.feature_maps['conv432'] = conv432
        self.feature_maps['conv4321'] = conv4321

        return sal_pred
    
    def get_feature_maps(self):
        return self.feature_maps
    

"""
===================================================================================================
    Fixation Feature Decoder Class
    
    This class implements a feature decoder module for fixation prediction, based on a ResNet-based
    encoder-decoder architecture.

    Attributes:
        - channel (int): Number of input channels.

    Methods:
        - __init__(self, channel): Constructor method to initialize the Fix_feat_decoder with
          the specified number of input channels.
        - _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
          Helper method to create prediction layers with specified configurations.
        - forward(self, x1, x2, x3, x4): Forward pass through the Fix_feat_decoder to compute
          the fixation prediction based on input feature maps x1, x2, x3, and x4.
        - get_feature_maps(self): Method to retrieve the feature maps stored during the forward pass
          for visualization or analysis purposes.
    
===================================================================================================
"""
class Fix_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Fix_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb4 = RCAB(channel * 4)

        self.cls_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        self.feature_maps = {}  # Dictionary to store feature maps
        

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def forward(self, x1,x2,x3,x4):

        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv3_feat = self.conv3(x3)
        conv4_feat = self.conv4(x4)

        conv4321 = torch.cat((conv1_feat, self.upsample2(conv2_feat),self.upsample4(conv3_feat), self.upsample8(conv4_feat)),1)
        conv4321 = self.racb4(conv4321)

        sal_pred = self.cls_layer(conv4321)
        
        # Store feature maps in the dictionary
        self.feature_maps['conv1_feat'] = conv1_feat
        self.feature_maps['conv2_feat'] = conv2_feat
        self.feature_maps['conv3_feat'] = conv3_feat
        self.feature_maps['conv4_feat'] = conv4_feat
        self.feature_maps['conv4321'] = conv4321

        return sal_pred

    def get_feature_maps(self):
        return self.feature_maps


"""
===================================================================================================
    Saliency Feature Encoder Class
    
    This class implements a feature encoder module for saliency prediction, based on a ResNet-based
    architecture.

    Attributes:
        - channel (int): Number of input channels.

    Methods:
        - __init__(self, channel): Constructor method to initialize the Saliency_feat_encoder with
          the specified number of input channels.
        - forward(self, x): Forward pass through the Saliency_feat_encoder to compute the saliency
          prediction based on input images x.
        - initialize_weights(self): Method to initialize the weights of the encoder using pretrained
          ResNet-50 weights.
        - get_feature_maps(self): Method to retrieve the feature maps stored during the forward pass
          for visualization or analysis purposes.
    
===================================================================================================
"""
class Saliency_feat_encoder(nn.Module):
    def __init__(self, channel):
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.cod_dec = Fix_feat_decoder(channel)
        self.sal_dec = Saliency_feat_decoder(channel)

        self.HA = HA()

        if self.training:
            self.initialize_weights()
        
        self.feature_maps = {}  # Dictionary to store feature maps

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        fix_pred = self.cod_dec(x1,x2,x3,x4)
        init_pred = self.sal_dec(x1,x2,x3,x4)

        x2_2 = self.HA(1-self.upsample05(fix_pred).sigmoid(), x2)
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 8 x 8
        ref_pred = self.sal_dec(x1,x2_2,x3_2,x4_2)
        
        # Store feature maps in the dictionary
        self.feature_maps['x1'] = x1
        self.feature_maps['x2'] = x2
        self.feature_maps['x3'] = x3
        self.feature_maps['x4'] = x4
        self.feature_maps['x2_2'] = x2_2
        self.feature_maps['x3_2'] = x3_2
        self.feature_maps['x4_2'] = x4_2

        return self.upsample4(fix_pred),self.upsample4(init_pred),self.upsample4(ref_pred)

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