# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .base_model import BaseModel
from . import networks
import math
# from .elan_block import conv


class DynaAWGNModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_L2']
        self.visual_names = ['real_A', 'fake', 'real_B']

        # self.model_names = ['SE', 'CE', 'G', 'P']
        self.model_names = ['SE', 'CA', 'SD', 'BCM_' + str(opt.band)]

        # define networks
        self.netSE = networks.define_SE(depths = [2, 4, 2, 2],
                                        num_heads = [8, 16, 8, 8],
                                        kernel_size = 7,
                                        mlp_ratio = 2.,
                                        qkv_bias = True,
                                        qk_scale = None,
                                        drop_rate = 0.,
                                        attn_drop_rate = 0.,
                                        drop_path_rate = 0.1,
                                        norm_layer = nn.LayerNorm,
                                        init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.netSD = networks.define_SD(depths = [2, 2, 4, 2],
                                        num_heads = [8, 8, 16, 8],
                                        kernel_size = 7,
                                        mlp_ratio = 2.,
                                        qkv_bias = True,
                                        qk_scale = None,
                                        drop_rate = 0.,
                                        attn_drop_rate = 0.,
                                        drop_path_rate = 0.1,
                                        norm_layer = nn.LayerNorm,
                                        init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        
        band = opt.band
        self.netCA = networks.define_CA(in_channel=256,init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        setattr(self, f"netBCM_{opt.band}", networks.define_BCM(band=band,
                                                            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids))  

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if opt.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            netBCM_name = f"netBCM_{opt.band}"

            netBCM_params = getattr(self, netBCM_name).parameters()
            params = list(self.netSE.parameters()) + list(self.netSD.parameters()) + list(self.netCA.parameters()) + list(netBCM_params)

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr_joint)
            self.optimizers.append(self.optimizer_G)

        self.opt = opt

    def name(self):
        return 'DynaAWGN_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def forward(self):
        # Generate SNR
        if self.opt.isTrain:
            self.snr = torch.rand(self.real_A.shape[0], 1, 1, 1).to(self.device) * (self.opt.SNR_MAX-self.opt.SNR_MIN) + self.opt.SNR_MIN
        else:
            self.snr = torch.ones(self.real_A.shape[0], 1, 1, 1).to(self.device) * self.opt.SNR

        # Generate latent vector
        latent = self.netSE(self.real_A)
       
        latent = self.netCA(latent)

        # Normalize each channel
        latent_sum = torch.sqrt((latent**2).mean((-2, -1), keepdim=True))
        latent = latent / latent_sum

        # Pass through the AWGN channel
        with torch.no_grad():
            sigma = torch.pow(10.0, -self.snr / 20.0)
            noise = sigma.view(self.real_A.shape[0], 1, 1, 1) * torch.randn_like(latent)

        latent = latent + noise
       
        cpp_choose = f"netBCM_{self.opt.band}"
        if hasattr(self, cpp_choose):
            latent = getattr(self, cpp_choose)(latent[:,:self.opt.band,:,:])

        self.fake = self.netSD(latent, self.snr)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B)
        self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
