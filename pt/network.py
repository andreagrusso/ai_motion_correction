# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

#from torch.utils.tensorboard import SummaryWriter

#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AffineNet(nn.Module):
    
    #this function will define the skeleton of the network
    def __init__(self):
      super(AffineNet, self).__init__()
      
      self.theta = torch.tensor([1, 0, 0, 0, 
                                0, 1, 0, 0,
                                0, 0, 1, 0], dtype=torch.float)
      
      self.localization = nn.Sequential(
              nn.Conv3d(2, 4, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True))
      

        #regression on 12 parameters
      self.regression = nn.Sequential(nn.Linear(512*1*1*1,12),
                                      nn.Dropout(p=0.3))
        
          # Initialize the weights/bias with identity transformation
      #self.regression[0].weight.data.zero_()
      self.regression[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 
                                                           0, 1, 0, 0,
                                                           0, 0, 1, 0], dtype=torch.float))
        
        
        #### regression separetely on rotation and translation
        #dense layer for the rotation params
      self.rot_params = nn.Sequential(nn.Linear(512*1*1*1, 9),nn.Tanh())# tanh activation?
      #self.rot_params[0].weight.data.zero_()
      self.rot_params[0].bias.data.copy_(torch.tensor([1, 0, 0, 
                                                         0, 1, 0,
                                                         0, 0, 1], dtype=torch.float))
        
        #dense layer for the translation params
      self.trans_params = nn.Linear(512*1*1*1, 3)           
      #self.trans_params.weight.data.zero_()
      self.trans_params.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))



    
    #transformation network
    def stn(self, x):
        
        #learn the features
        xs = self.localization(x)
        xs = xs.view(xs.shape[0],-1)
        
        #rotation and translation parameters
        rot_params = self.rot_params(xs).view(-1,3,3)
        trans_params = self.trans_params(xs).view(-1,3,1)
        #concat translation and rotation
        theta = torch.cat((rot_params,trans_params),dim=-1)


        
        
        if x.shape[0] > 1: #batch_size higher than 1
        
            #exapnd on channel dim to keep batch size
            movable = torch.unsqueeze(x[:,1,:,:,:],1) 
            
            grid = torch.empty((x.shape[0],
                                x.shape[2],
                                x.shape[3],
                                x.shape[4],
                                3)).to(device) #affine gird creates 3 channels (3 axis)
        
            for i in range(x.shape[0]):
                
                single_theta = torch.unsqueeze(theta[i,:,:],0)#select one item at time
                movable_size = (1, 1, x.shape[2], x.shape[3], x.shape[4])
                
                #get the single interpolation grid
                grid[i,:,:,:,:] = F.affine_grid(single_theta, 
                                                movable_size,
                                                align_corners=False)
                
        else:         
        
            #exapnd on batch dim to keep correct shape
            movable = torch.unsqueeze(x[:,1,:,:,:],0)
            
            #find the interpolation grid
            grid = F.affine_grid(theta, 
                                 movable.shape,
                                 align_corners=False)
            
            
            
        #apply the grid
        x = F.grid_sample(movable, grid,
                          align_corners=False,
                          mode = 'bilinear')
    
        return x, theta
        
    def forward(self, fixed, movable):
        
        #concatenatation over the last dimension
        concat_data = torch.cat((fixed,movable),dim=1)#.to(device)
        #apply transformation network
        padded_output, theta = self.stn(concat_data)
        
        return padded_output, theta


#%% UNet + STN

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.LeakyReLU  = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.LeakyReLU(self.conv2(self.LeakyReLU(self.conv1(x))))
    
    
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
        
        # #dense layer for the rotation params
        # self.rot_params = linear_layer(256, 9, 'tanh', 0.3)
        
        # #dense layer for the rotation params
        # self.trans_params = linear_layer(256, 3, 'linear', 0.3)
        
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
                   
        self.regression = nn.Sequential(nn.Linear(256,12))
          
          # Initialize the weights/bias with identity transformation
        #self.regression[0].weight.data.zero_()
        self.regression[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 
                                                         0, 1, 0, 0,
                                                         0, 0, 1, 0], dtype=torch.float))
        
        
        #### regression separetely on rotation and translation
        #dense layer for the rotation params
        self.rot_params = nn.Sequential(nn.Linear(512*1*1*1, 9)) #nn.Tanh activation?
      #self.rot_params[0].weight.data.zero_()
        self.rot_params[0].bias.data.copy_(torch.tensor([1, 0, 0, 
                                                         0, 1, 0,
                                                         0, 0, 1], dtype=torch.float))
        
        #dense layer for the translation params
        self.trans_params = nn.Linear(512*1*1*1, 3)           
      #self.trans_params.weight.data.zero_()
        self.trans_params.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))        
        
        
    def stn(self, x):
        
        unet_output = self.unet(x)
        
        xs = self.localization(unet_output)
        xs = xs.view(xs.shape[0],-1)
        theta = self.regression(xs)
        theta = theta.view(-1, 3, 4)
        # rot_params = self.rot_params(xs)
        # trans_params = self.trans_params(xs)
        
        # rot_params = rot_params.view(-1,3,3)
        # trans_params = trans_params.view(-1,1,3)
        
        # theta = torch.zeros((1,3,4))
        # theta[:,:,:-1] = rot_params
        # theta[:,:,-1] = trans_params
        # theta = theta.to(device)
        
        #movable = x[:,1,:,:,:]
        #movable = torch.ex
    
        theta = theta.view(-1, 3, 4).to(device)
        x = x.to(device)
        
        if x.shape[0] > 1: #batch_size higher than 1
        
            #exapnd on channel dim to keep batch size
            movable = torch.unsqueeze(x[:,1,:,:,:],1) 
            
            grid = torch.empty((x.shape[0],
                                x.shape[2],
                                x.shape[3],
                                x.shape[4],
                                3)).to(device) #affine gird creates 3 channels (3 axis)
        
            for i in range(x.shape[0]):
                
                single_theta = torch.unsqueeze(theta[i,:,:],0)#select one item at time
                movable_size = (1, 1, x.shape[2], x.shape[3], x.shape[4])
                
                #get the single interpolation grid
                grid[i,:,:,:,:] = F.affine_grid(single_theta, 
                                                movable_size,
                                                align_corners=False)
                
        else:         
        
            #exapnd on batch dim to keep correct shape
            movable = torch.unsqueeze(x[:,1,:,:,:],0)
            
            #find the interpolation grid
            grid = F.affine_grid(theta, 
                                 movable.shape,
                                 align_corners=False)
        
           
        x = F.grid_sample(movable, grid,
                          align_corners=False,
                          mode = 'bilinear')
    
        return x, theta
        
    def forward(self, fixed, movable):
        
        #concatenatation over the last dimension
        concat_data = torch.cat((fixed,movable),dim=1)#.to(device)      
        
        padded_output, theta = self.stn(concat_data)
        
        return padded_output, theta



#%% Recurrent STN


class ReSTN(nn.Module):
    
    #this function will define the skeleton of the network
    def __init__(self):
      super(ReSTN, self).__init__()
      
      self.identity_transformation = torch.tensor([1, 0, 0, 0, 
                                                       0, 1, 0, 0,
                                                       0, 0, 1, 0], 
                                                  dtype=torch.float)
      
      
      self.localization = nn.Sequential(
              nn.Conv3d(2, 4, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True),
              nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(True))
                 
      # self.regression = nn.Sequential(nn.Linear(256*1*1*1,128),
      #                                  nn.LeakyReLU(True),
      #                                  nn.Linear(128,64),
      #                                  nn.LeakyReLU(True),
      #                                  nn.Linear(64,12))

      
      
      self.rotation = nn.Sequential(nn.Linear(256*1*1*1,128),
                                       nn.LeakyReLU(True),
                                       nn.Linear(128,64),
                                       nn.LeakyReLU(True),
                                       nn.Linear(64,9))      
      self.rotation[-1].bias.data.copy_(torch.tensor([1, 0, 0, 
                                                      0, 1, 0,
                                                      0, 0, 1], dtype=torch.float))
      
      
      self.translation = nn.Sequential(nn.Linear(256*1*1*1,128),
                                        nn.LeakyReLU(True),
                                        nn.Linear(128,64),
                                        nn.LeakyReLU(True),
                                        nn.Linear(64,32),
                                        nn.LeakyReLU(True),
                                        nn.Linear(32,3))
      self.translation[-1].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
      

        
        # Initialize the weights/bias with identity transformation
        
      # self.regression[-1].weight.data.zero_()
      # self.regression[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 
      #                                                     0, 1, 0, 0,
      #                                                     0, 0, 1, 0], dtype=torch.float))
      

    def stn_transformation(self,x, theta):
        
        theta = theta.view(-1, 3, 4).to(device)
        x = x.to(device)
        
        if x.shape[0] > 1: #batch_size higher than 1
        
            #exapnd on channel dim to keep batch size
            movable = torch.unsqueeze(x[:,1,:,:,:],1) 
            
            grid = torch.empty((x.shape[0],
                                x.shape[2],
                                x.shape[3],
                                x.shape[4],
                                3)).to(device) #affine gird creates 3 channels (3 axis)
        
            for i in range(x.shape[0]):
                
                single_theta = torch.unsqueeze(theta[i,:,:],0)#select one item at time
                movable_size = (1, 1, x.shape[2], x.shape[3], x.shape[4])
                
                #get the single interpolation grid
                grid[i,:,:,:,:] = F.affine_grid(single_theta, 
                                                movable_size,
                                                align_corners=False)
                
        else:         
        
            #exapnd on batch dim to keep correct shape
            movable = torch.unsqueeze(x[:,1,:,:,:],0)
            
            #find the interpolation grid
            grid = F.affine_grid(theta, 
                                 movable.shape,
                                 align_corners=False)
            
        x = F.grid_sample(x, grid,
                          align_corners=False,
                          mode = 'bilinear')
        
        return x

    def combine(self,curr_theta, delta_theta):
        
        #curr_theta.view
        tmp_zeros_vec = torch.zeros((curr_theta.shape[0],1,4)).to(device)
        a = torch.hstack([curr_theta,tmp_zeros_vec])
        a[:,-1,-1] = torch.ones((curr_theta.shape[0]))
        b = torch.hstack([delta_theta, tmp_zeros_vec])
        b[:,-1,-1] = torch.ones((curr_theta.shape[0]))
        #b = b + torch.eye(4, device = device)        
        return torch.matmul(a, b)
 


    def network(self, fixed, warped, movable, old_theta):
       
        data = torch.cat((fixed, warped),dim=1).to(device)
        #localization 
        features = self.localization(data)
        features = features.view(features.shape[0],1,-1)
        

        
        #get the parameters
        delta_rot = self.rotation(features).view(-1,3,3)
        delta_tra = self.translation(features).view(-1,3,1)
        #stack them together
        delta_theta = torch.cat((delta_rot,delta_tra),dim=-1)#
       
        #delta_theta = self.regression(features)
        #combine the delta params with the current one
        theta = self.combine(old_theta, 
                             delta_theta.view(-1,3,4))

       
        #apply new transformation
        warped_data = self.stn_transformation(movable,
                                              theta[:,:-1,:])
       
        return theta[:,:-1,:], warped_data
       
        
    def forward(self, fixed, movable):
        
        all_warped = []
        all_theta = []
        
        curr_theta = torch.tile((self.identity_transformation).view(-1,3,4), 
                                (movable.shape[0],1,1)).to(device)
        warped_data = movable
        
        all_warped.append(warped_data)
        all_theta.append(curr_theta)           
        
        
        
        for i in range(4):
            #print(i)
            
            
            curr_theta, warped_data = self.network(fixed, 
                                              warped_data,
                                              movable,
                                              curr_theta)
            
            all_warped.append(warped_data)
            all_theta.append(curr_theta)           

            
        return warped_data, curr_theta
            











        
        
        # import numpy as np
        # fixed_deconv = np.squeeze(res_deconv_7[0,0,:,:,:].cpu().detach().numpy())
        # mov_deconv = np.squeeze(res_deconv_7[0,1,:,:,:].cpu().detach().numpy())
        # pad = np.squeeze(padded_output.cpu().detach().numpy())
        # dec = np.squeeze(decoded.cpu().detach().numpy())
        
        # fixed_input = np.squeeze(fixed.cpu().detach().numpy())
        # mov_input = np.squeeze(movable.cpu().detach().numpy())
       
        # f,ax = plt.subplots(1,4)
        # ax[0].imshow(fix[:,:,64]-m[:,:,64])#fixed_deconv[:,:,64])
        # ax[1].imshow(u[0,:,:,64]-fix[:,:,64])#mov_deconv[:,:,64])
        # ax[2].imshow(u[1,:,:,64]-m[:,:,64])
        # ax[3].imshow(u[1,:,:,64]-u[0,:,:,64])
        # ax[1,0].imshow(w2[0,:,:,64])
        # ax[1,1].imshow(w2[1,:,:,64])
    
        # f, ax = plt.subplots(1,3)
        # ax[0].imshow(out[0,:,:,64])
        # ax[1].imshow(out[1,:,:,64])
        # ax[2].imshow(out[0,:,:,64]-out[1,:,:,64])
   