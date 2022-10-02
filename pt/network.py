# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from external_layers import SpatialTransformer
from torch.utils.tensorboard import SummaryWriter

#%%


class AffineNet(nn.Module):
    
    #this function will define the skeleton of the network
    def __init__(self):
      super(AffineNet, self).__init__()
      
      
      #encoder part
      self.encoder = nn.Sequential(
          nn.Conv3d(in_channels=2, out_channels=8, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=8, out_channels=16, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=16, out_channels=32, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=32, out_channels=64, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=64, out_channels=128, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=128, out_channels=256, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Conv3d(in_channels=256, out_channels=512, 
                    kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU()
          #nn.Flatten()
          )

      #dense layers to estimate affines      
      self.rotation = nn.Sequential(nn.Linear(in_features=512,out_features=9),
                                    nn.Tanh(),
                                    nn.Dropout3d(0.3))
      self.translation = nn.Sequential(nn.Linear(512,3),
                                       nn.Dropout3d(0.3))
      


      self.linear_before_dec = nn.Linear(12,512)
      
     #decoder part

      self.decoder = nn.Sequential(
          
        nn.ConvTranspose3d(in_channels=512, out_channels=256, padding=1, 
                           kernel_size=3, stride=2, output_padding=1),#output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=256, out_channels=128, padding=1,
                           kernel_size=3, stride=2, output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=128, out_channels=64, padding=1,
                           kernel_size=3, stride=2, output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=64, out_channels=32, padding=1,
                           kernel_size=3, stride=2,output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=32, out_channels=16, padding=1,
                           kernel_size=3, stride=2, output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=16, out_channels=8, padding=1,
                           kernel_size=3, stride=2,output_padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels=8, out_channels=2, padding=1,
                           kernel_size=3, stride=2,output_padding=1),
        nn.LeakyReLU()
        )     
      self.last_dec_layer = nn.Sequential(nn.ConvTranspose3d(in_channels=2, out_channels=1, 
                                 kernel_size=1, stride=1, 
                                 output_padding=0, padding=0),
              nn.LeakyReLU())
         
      #TO CHECK 
      self.spatial_trans = SpatialTransformer(size=(128,128,128))         



    def forward(self,fixed,movable):
        
        #concatenatation over the last dimension
        concat_data = torch.cat((fixed,movable),dim=1)
        
        #encoder
        #encoded = self.encoder(concat_data)
        #activations for skip-connections
        acts = []
        encoded = concat_data
        for layer in self.encoder:
            encoded = layer(encoded)
            #print(encoded.shape)
            acts.append(encoded)
        
        #affine params
        #encoded = encoded.permute(0,4,3,2,1)
        rot_params = self.rotation(torch.transpose(encoded,1,-1))
        trans_params = self.translation(torch.transpose(encoded,1,-1))
        affine_params = torch.cat((trans_params,rot_params),dim=-1)
        #decoder
        decoded = self.linear_before_dec(affine_params)
        
        decoded = torch.transpose(decoded,-1,1)
        
        #skip-connections
        # for a, layer in zip(acts[::-1], self.decoder):
        #     input_decoder = torch.cat((a, decoded),dim=-1)
        #     decoded = layer(input_decoder)
        for layer in self.decoder:
            decoded = layer(decoded)
            #print(decoded.shape)
            
        #last decoder layer
        decoded = self.last_dec_layer(decoded)
            
        #spatial transorformation
        padded_output = self.spatial_trans(movable,decoded)
        
        return padded_output, rot_params, trans_params
        
        
        # fixed_array = np.squeeze(fixed.detach().numpy())
        # movable_array = np.squeeze(movable.detach().numpy())

        # f,ax = plt.subplots(1,3)
        # ax[0].imshow(fixed_array[:,:,64])
        # ax[1].imshow(movable_array[:,:,64])    
        # ax[2].imshow(f_map[:,:,64])
        
     