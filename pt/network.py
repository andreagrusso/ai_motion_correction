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


#3d convolution for the first part of the U-Net (using leaky relu)
class DownConv3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(DownConv3D, self).__init__()
        
        self.conv_3d_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 
                      kernel, stride, padding=1),
            nn.LeakyReLU()
            )
        
    def forward(self, tensord_data):
        
        return self.conv_3d_block(tensor_data)
        

#3d convolution for the second part of the U-Net (using leaky relu)
class UpConv3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(UpConv3D, self).__init__()
        
        self.conv_3d_block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 
                      kernel, stride, 
                      output_padding=1, padding=1),
            nn.LeakyReLU()
            )
        
    def forward(self, tensord_data):
        
        return self.conv_3d_block(tensor_data)



class linear_layer(nn.Module):
    
    def __init__(self, in_features, out_features, activation_fn, drop):
        super(linear_layer, self).__init__()
        
        if activation_fn == 'tanh':
            self.linear = nn.Sequential(nn.Linear(in_features,out_features),
                                          nn.Tanh(),
                                          nn.Dropout3d(drop))
        else:
            self.linear = nn.Sequential(nn.Linear(in_features,out_features),
                                          nn.Dropout3d(drop)) 
    
    def forward(self,x):
        
        return self.linear(x)
        
        
class AffineNet(nn.Module):
    
    #this function will define the skeleton of the network
    def __init__(self):
      super(AffineNet, self).__init__()
      
      #1st conv
      self.conv_1 = DownConv3D(2,8,3,2)
      #2nd conv
      self.conv_2 = DownConv3D(8,16,3,2)
      #3rd conv
      self.conv_3 = DownConv3D(16,32,3,2)
      #4th conv
      self.conv_4 = DownConv3D(32,64,3,2)
      #5th conv
      self.conv_5 = DownConv3D(64,128,3,2)
      #6th conv
      self.conv_6 = DownConv3D(128,256,3,2)
      #7th conv
      self.conv_7 = DownConv3D(256,512,3,2)
      
      #dense layer for the rotation params
      self.rot_params = linear_layer(512, 9, 'tanh', 0.3)
      
      #dense layer for the rotation params
      self.trans_params = linear_layer(512, 9, 'linear', 0.3)
      
      #from 12 to 512 again
      self.linear_before_dec = nn.Linear(12,512)

      
      #deconvolution part
      self.upoconv_1 = UpConv3D(512,256,3,2)
      #2nd upsampling
      self.upoconv_2 = UpConv3D(256*2, 128*2, 3, 2) #skip conn with conv6
      #3rd upsampling
      self.upoconv_3 = UpConv3D(128*2, 64*2, 3, 2) #skip conn with conv5
      #4th upsampling
      self.upoconv_4 = UpConv3D(64*2, 32*2, 3, 2) #skip conn with conv4
      #5h upsampling
      self.upoconv_5 = UpConv3D(32*2, 16*2, 3, 2) #skip conn with conv3
      #6th upsampling
      self.upoconv_6 = UpConv3D(16*2, 8*2, 3, 2) #skip conn with conv2
      #7th upsampling
      self.upoconv_7 = UpConv3D(8*2, 2*2, 3, 2) #skip conn with conv1
      #last downsampling
      self.last_dec_layer = nn.Sequential(nn.ConvTranspose3d(in_channels=2*2, out_channels=1, 
                            kernel_size=2, stride=2, dilation=1, 
                            output_padding=0, padding=0),
                                          nn.LeakyReLU())  
      
      #TO CHECK 
      self.spatial_trans = SpatialTransformer(size=(128,128,128))  
 
        
    def forward(self,fixed, movable):
        
        #concatenatation over the last dimension
        concat_data = torch.cat((fixed,movable),dim=1)        
        #first layer
        res_conv_1 = self.conv_1(concat_data)
        #2nd layer
        res_conv_2 = self.conv_2(res_conv_1)
        #3rd layer
        res_conv_3 = self.conv_3(res_conv_2)
        #4th layer
        res_conv_4 = self.conv_4(res_conv_3)
        #5th layer
        res_conv_5 = self.conv_5(res_conv_4)
        #6th
        res_conv_6 = self.conv_6(res_conv_5)
        #7th 
        res_conv_7 = self.conv_7(res_conv_6)

        rot_params = self.rot_params(torch.transpose(res_conv_7,1,-1))
        trans_params = self.trans_params(torch.transpose(res_conv_7,1,-1))
        affine_params = torch.cat((trans_params,rot_params),dim=-1)
        
        encoded = self.linear_before_dec(affine_params)
        encoded = torch.transpose(encoded,-1,1)

        #decoder
        res_deconv_1 = self.upoconv_1(encoded)
        #2nd decoder layer with skip
        res_deconv_2 = self.upoconv_2(torch.cat((res_conv_6,res_deconv_1),dim=-1))        
        #3rd decoder layer with skip
        res_deconv_3 = self.upoconv_3(torch.cat((res_conv_5,res_deconv_2),dim=-1))       
        #4th decoder layer with skip
        res_deconv_4 = self.upoconv_4(torch.cat((res_conv_4,res_deconv_3),dim=-1))        
        #5th decoder layer with skip
        res_deconv_5 = self.upoconv_5(torch.cat((res_conv_3,res_deconv_4),dim=-1)) 
        #6th decoder layer with skip
        res_deconv_6 = self.upoconv_6(torch.cat((res_conv_2,res_deconv_5),dim=-1))
        #6th decoder layer with skip
        res_deconv_7 = self.upoconv_7(torch.cat((res_conv_1,res_deconv_6),dim=-1))

        #last decoder layer
        decoded = self.last_dec_layer(res_deconv_7)
            
        #spatial transorformation
        padded_output = self.spatial_trans(movable,decoded)
        
        return padded_output, rot_params, trans_params
                 
        

        
      
      
      
    #   nn.Sequential(
    #       nn.Conv3d(in_channels=2, out_channels=8, 
    #                 kernel_size=3, stride=2, padding=1),
    #       F.leaky_relu()
    #       #nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=8, out_channels=16, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=16, out_channels=32, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=32, out_channels=64, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=64, out_channels=128, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=128, out_channels=256, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU(),
    #       nn.Conv3d(in_channels=256, out_channels=512, 
    #                 kernel_size=3, stride=2, padding=1),
    #       nn.LeakyReLU()
    #       #nn.Flatten()
    #       )

    #   #dense layers to estimate affines      
    #   self.rotation = nn.Sequential(nn.Linear(in_features=512,out_features=9),
    #                                 nn.Tanh(),
    #                                 nn.Dropout3d(0.3))
    #   self.translation = nn.Sequential(nn.Linear(512,3),
    #                                    nn.Dropout3d(0.3))
      


    #   self.linear_before_dec = nn.Linear(12,512)
      
    #  #decoder part

    #   self.decoder = nn.Sequential(
          
    #     nn.ConvTranspose3d(in_channels=512, out_channels=256, padding=1, 
    #                        kernel_size=3, stride=2, output_padding=1),#output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=256, out_channels=128, padding=1,
    #                        kernel_size=3, stride=2, output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=128, out_channels=64, padding=1,
    #                        kernel_size=3, stride=2, output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=64, out_channels=32, padding=1,
    #                        kernel_size=3, stride=2,output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=32, out_channels=16, padding=1,
    #                        kernel_size=3, stride=2, output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=16, out_channels=8, padding=1,
    #                        kernel_size=3, stride=2,output_padding=1),
    #     nn.LeakyReLU(),
    #     nn.ConvTranspose3d(in_channels=8, out_channels=2, padding=1,
    #                        kernel_size=3, stride=2,output_padding=1),
    #     nn.LeakyReLU()
    #     )     
    #   self.last_dec_layer = nn.Sequential(nn.ConvTranspose3d(in_channels=2, out_channels=1, 
    #                              kernel_size=1, stride=1, 
    #                              output_padding=0, padding=0),
    #           nn.LeakyReLU())
         
    #   #TO CHECK 
    #   self.spatial_trans = SpatialTransformer(size=(128,128,128))         



    # def forward(self,fixed,movable):
        
    #     #concatenatation over the last dimension
    #     concat_data = torch.cat((fixed,movable),dim=1)
        
    #     #encoder
    #     #encoded = self.encoder(concat_data)
    #     #activations for skip-connections
    #     acts = []
    #     encoded = concat_data
    #     for idx, layer in enumerate(self.encoder):
    #         encoded = layer(encoded)
    #         #save only the ac
    #         # if idx in range(1,len(self.encoder),2):
    #         #     acts.append(encoded)
        
    #     #affine params
    #     #encoded = encoded.permute(0,4,3,2,1)
    #     rot_params = self.rotation(torch.transpose(encoded,1,-1))
    #     trans_params = self.translation(torch.transpose(encoded,1,-1))
    #     affine_params = torch.cat((trans_params,rot_params),dim=-1)
    #     #decoder
    #     decoded = self.linear_before_dec(affine_params)
        
    #     decoded = torch.transpose(decoded,-1,1)
        
    #     #skip-connections
    #     # for a, layer in zip(acts[::-1], self.decoder):
    #     #     input_decoder = torch.cat((a, decoded),dim=-1)
    #     #     decoded = layer(input_decoder)
    #     for layer in self.decoder:
    #         decoded = layer(decoded)
    #         #print(decoded.shape)
            
    #     #last decoder layer
    #     decoded = self.last_dec_layer(decoded)
            
    #     #spatial transorformation
    #     padded_output = self.spatial_trans(movable,decoded)
        
    #     return padded_output, rot_params, trans_params
        
        
        # fixed_array = np.squeeze(fixed.detach().numpy())
        # movable_array = np.squeeze(movable.detach().numpy())

        # f,ax = plt.subplots(1,3)
        # ax[0].imshow(fixed_array[:,:,64])
        # ax[1].imshow(movable_array[:,:,64])    
        # ax[2].imshow(f_map[:,:,64])
        
     