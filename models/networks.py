# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.nn import functional as F
import numpy as np
from models.layers_natten import *
# from .elan_block import conv, deconv, ELAB
# from .layers import RSTB



class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Normalize(nn.Module):
    def forward(self, x, power):
        N = x.shape[0]
        pwr = torch.mean(x**2, (1, 2, 3), True)
        return np.sqrt(power) * x / torch.sqrt(pwr)

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##################################################################################################################################

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def define_SE(depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
              norm_layer = nn.LayerNorm, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = Source_Encoder(depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_SD(depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
              norm_layer = nn.LayerNorm, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = Source_Decoder(depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_CA(in_channel, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = CA(in_planes=in_channel)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_BCM(band, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    net = BCM(band)
    return init_net(net, init_type, init_gain, gpu_ids)

class Source_Encoder(nn.Module):

    def __init__(self, depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_0 = conv(3, 64, kernel_size=5, stride=2)
        self.layers_1 = ResViTBlock(dim=64,
                                depth=depths[0],
                                num_heads=num_heads[0],
                                kernel_size=kernel_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                                norm_layer=norm_layer,
        )

        self.layers_2 = conv(64, 128, kernel_size=3, stride=2)
        self.layers_3 = ResViTBlock(dim=128,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
        )
        
        self.layers_4 = conv(128, 192, kernel_size=3, stride=2)
        self.layers_5 = ResViTBlock(dim=192,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        
        self.layers_6 = conv(192, 256, kernel_size=3, stride=2)
        self.layers_7 = ResViTBlock(dim=256,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )
        
    def forward(self, x):
        x = self.layers_0(x)
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = self.layers_5(x)
        x = self.layers_6(x)
        x = self.layers_7(x)
        return x

class Source_Decoder(nn.Module):
    def __init__(self,depths, num_heads, kernel_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer):
        super().__init__()

        last = []
        first = []

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_0 = ResViTBlock(dim=256,
                        depth=depths[0],
                        num_heads=num_heads[0],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
        )
        self.layers_1 = deconv(256, 192, kernel_size=3, stride=2)
        self.layers_2 = ResViTBlock(192,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
        )
        
        self.layers_3 = deconv(192, 128, kernel_size=3, stride=2)
        self.layers_4 = ResViTBlock(128,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        
        self.layers_5 = deconv(128, 64, kernel_size=3, stride=2)
        self.layers_6 = ResViTBlock(64,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )
        
        self.layers_7 = deconv(64, 3, kernel_size=3, stride=2)
        
        last += [nn.Sigmoid()]
        first += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                      nn.BatchNorm2d(256), nn.ReLU(True),
                      nn.Sigmoid()]

        self.first = nn.Sequential(*first)

        self.last = nn.Sequential(*last)
        
        
        self.mod1 = AFF(256)
        self.mod2 = AFF(192)
        self.mod3 = AFF(128)
        self.mod4 = AFF(64)

    def forward(self, x, SNR):
        x = self.first(x) * x + x
        x = self.layers_0(x)  #256
        x = self.mod1(x, SNR) #256
    
        x = self.layers_1(x)
        x = self.layers_2(x)  #192
        x = self.mod2(x, SNR) #192
        
        x = self.layers_3(x)
        x = self.layers_4(x)  #128
        x = self.mod3(x, SNR) #128

        x = self.layers_5(x)
        x = self.layers_6(x)  #64
        x = self.mod4(x, SNR) #64
 
        x = self.layers_7(x)
        x = 2 * self.last(x) - 1
        return x

class CA(nn.Module):#
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少, 因此需要一个比例系数ratio进行缩放
        """
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class AQL(nn.Module):
    def __init__(self, in_channel, band=64):
        super(AQL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 192, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv4 = nn.Conv2d(192, in_channel, kernel_size=1, stride=1, padding=0)
        self.last = nn.Conv2d(in_channel, band, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.sigmoid(self.conv4(x))
        x = identity - identity * x
        x = self.last(x)
        return x  

class BCM(nn.Module):
    def __init__(self, in_channel):
        super(BCM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 384, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(384, 384, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv3 = nn.Conv2d(384, 384, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv4 = nn.Conv2d(384, in_channel, kernel_size=1, stride=1, padding=0)
        self.first = nn.ConvTranspose2d(in_channel, 256, kernel_size=3, stride=1, output_padding=0, padding=1)
    
    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = identity + identity * x
        x = self.first(x)
        return x 

class modulation(nn.Module):
    def __init__(self, C_channel):

        super(modulation, self).__init__()

        activation = nn.ReLU(True)

        # Policy network
        model_multi = [nn.Linear(C_channel + 1, C_channel), activation,
                       nn.Linear(C_channel, C_channel), nn.Sigmoid()]

        model_add = [nn.Linear(C_channel + 1, C_channel), activation,
                     nn.Linear(C_channel, C_channel)]

        self.model_multi = nn.Sequential(*model_multi)
        self.model_add = nn.Sequential(*model_add)

    def forward(self, z, SNR):

        # Policy/gate network
        N, C, W, H = z.shape

        z_mean = torch.mean(z, (-2, -1))
        z_cat = torch.cat((z_mean, SNR), -1)

        factor = self.model_multi(z_cat).view(N, C, 1, 1)
        addition = self.model_add(z_cat).view(N, C, 1, 1)

        return z * factor + addition

class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        # xo = x * wei + 2 * residual * (1 - wei)
        xo = x * wei
        return xo