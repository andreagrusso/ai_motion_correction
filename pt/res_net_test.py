# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:28:14 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import torch
import torch.nn as nn


def conv3x3x3(in_ch, out_ch, stride=1):
    return nn.Conv3d(in_ch,
                     out_ch,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_ch, out_ch, stride=1):
    return nn.Conv3d(in_ch,
                     out_ch,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):#, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_ch, out_ch, stride)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        # self.bn2 = nn.BatchNorm3d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    
    def __init__(self)