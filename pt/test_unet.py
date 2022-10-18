# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:02:16 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch, torchvision
import torch.nn as nn
from external_layers import SpatialTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchinfo import summary


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
    
class Encoder(nn.Module):
    def __init__(self, chs=(2,8, 16, 32, 64,128,256,512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool3d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            #print('Enc ouptut:',x.shape)
            x = self.pool(x)
        return ftrs
    
    
class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32, 16, 8)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i+1], 3,2, output_padding=1, padding=1) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            #print('Upsampling shape',x.shape)
            #enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, encoder_features[i]], dim=1)
            x        = self.dec_blocks[i](x)
            #print(x.shape)
        return x
    

    
class UNet(nn.Module):
    def __init__(self, enc_chs=(2,8, 16, 32, 64,128,256,512), dec_chs=(512, 256, 128, 64, 32, 16, 8), 
                 num_class=2, retain_dim=False, out_sz=(128,128,128)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv3d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out


class Unet_Stn(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
        self.unet = UNet()
        
        self.localization = nn.Sequential(
                nn.Conv3d(2, 4, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(4, 8, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True),
                nn.Conv3d(128, 256, kernel_size=3, padding=1),
                nn.MaxPool3d(2),
                nn.LeakyReLU(True))
                   
        self.regression = nn.Sequential(nn.Linear(256,12),
                                          nn.LeakyReLU(True))
          
          # Initialize the weights/bias with identity transformation
        self.regression[0].weight.data.zero_()
        self.regression[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 
                                                         0, 1, 0, 0,
                                                         0, 0, 1, 0], dtype=torch.float))
        
        
    def stn(self, x):
        
        unet_output = self.unet(x)
        
        xs = self.localization(unet_output)
        xs = xs.view(xs.shape[0],-1)
        theta = self.regression(xs)
        theta = theta.view(-1, 3, 4)
        
        movable = torch.unsqueeze(x[:,1,:,:,:],0)
        #movable = torch.ex
    
        grid = F.affine_grid(theta, movable.shape,align_corners=True)
        
        #x = self.special_trans(grid)
        x = F.grid_sample(movable, grid,
                          align_corners=True,
                          mode = 'bilinear')
    
        return x, theta
        
    def forward(self, fixed, movable):
        
        #concatenatation over the last dimension
        concat_data = torch.cat((fixed,movable),dim=1)#.to(device)      
        
        padded_output, theta = self.stn(concat_data)
        #print(padded_output.shape)
        
        return padded_output, theta
    

unet = Unet_Stn()
summary(unet, [(1,1,128,128,128),(1,1,128,128,128)])