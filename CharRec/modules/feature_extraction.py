# This file is adapted from CLOVA AI's deep-text-recognition-benchmark:
# https://github.com/clovaai/deep-text-recognition-benchmark
#
# Original copyright:
# Copyright (c) 2019-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0.
# Modifications were made for the proposed container code recognition framework.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# DeformableGRCNN ---------------------------------------------------------------------------------------------------------------------
class DeformableGRCNN_l_FeatureExtractor(nn.Module):
    def __init__(self):
        super(DeformableGRCNN_l_FeatureExtractor, self).__init__()
        self.output_channel = [128, 256, 512, 1280]
        self.ConvNet = nn.Sequential(nn.Conv2d(1, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     GRCL(self.output_channel[0], self.output_channel[0], num_iteration=4, kernel_size=3, pad=1),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     GRCL(self.output_channel[0], self.output_channel[1], num_iteration=3, kernel_size=3, pad=1),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[2], 2, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True),
                                    
                                     DeformableBlock(self.output_channel[2], kernel_size=3, stride=1, padding=1),

                                     nn.Conv2d(self.output_channel[2], self.output_channel[3], 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)



# DeformableCNN ---------------------------------------------------------------------------------------------------------------------
class DeformableCNN_l_FeatureExtractor(nn.Module):
    def __init__(self):
        super(DeformableCNN_l_FeatureExtractor, self).__init__()
        self.output_channel = [128, 256, 512, 1280]
        self.ConvNet = nn.Sequential(nn.Conv2d(1, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[2], 2, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True),
                                  
                                     DeformableBlock(self.output_channel[2], kernel_size=3, stride=1, padding=1),

                                     nn.Conv2d(self.output_channel[2], self.output_channel[3], 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)


class DeformableCNN_s_FeatureExtractor(nn.Module):
    def __init__(self):
        super(DeformableCNN_s_FeatureExtractor, self).__init__()
        self.output_channel = [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(nn.Conv2d(1, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[1], 2, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True),

                                     DeformableBlock(self.output_channel[1], kernel_size=3, stride=1, padding=1),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[3], 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)



# CNN ---------------------------------------------------------------------------------------------------------------------
class CNN_s_FeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN_s_FeatureExtractor, self).__init__()
        self.output_channel = [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(nn.Conv2d(1, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                                     nn.MaxPool2d(2, (2, 1), (0, 1)),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[1], kernel_size=(2, 1), stride=1, padding=(0, 1), bias=False),
                                     nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True),
                                     
                                     nn.Sequential(nn.Conv2d(self.output_channel[1], self.output_channel[1], kernel_size=(1, 5), stride=1, padding=(0, 8), dilation=(1, 4), padding_mode='replicate', bias=False),
                                                   nn.BatchNorm2d(self.output_channel[1]), nn.SiLU(True),
                                                   nn.Conv2d(self.output_channel[1], self.output_channel[1], kernel_size=3, stride=1, padding=1, groups=2, padding_mode='replicate', bias=False),
                                                   nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True),),

                                     nn.Conv2d(self.output_channel[1], self.output_channel[3], 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))

    def forward(self, input):
         return self.ConvNet(input)



# Utils ---------------------------------------------------------------------------------------------------------------------
class GRCL(nn.Module):
    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()

        self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.BN_x_init = nn.BatchNorm2d(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = F.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))
        return x



class GRCL_unit(nn.Module):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()

        self.BN_gfu = nn.BatchNorm2d(output_channel)
        self.BN_grx = nn.BatchNorm2d(output_channel)
        self.BN_fu = nn.BatchNorm2d(output_channel)
        self.BN_rx = nn.BatchNorm2d(output_channel)
        self.BN_Gx = nn.BatchNorm2d(output_channel)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = F.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = F.relu(x_first_term + x_second_term)
        return x



class DeformableBlock(nn.Module):
  def __init__(self, dim, kernel_size=3, stride=1, padding=1, groups=1):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.offset_conv = nn.Conv2d(dim,
                                 2 * kernel_size * kernel_size,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=True)

    self.deform_conv = DeformConv2d(dim,
                                    dim,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=groups,
                                    bias=False)

    self.bn = nn.BatchNorm2d(dim)
    self.act = nn.SiLU(inplace=True)

  def forward(self, x):
      offset = self.offset_conv(x)        
      out = self.deform_conv(x, offset)    
      out = self.bn(out)
      out = self.act(out)
      return out
