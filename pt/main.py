# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:09:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch
import torch.nn as nn
from torch.optim import Adam
import nibabel as nb
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
from network import AffineNet
from losses import NCC, regularizer_rot_matrix


#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'

# raw_input_1 = nb.load(os.path.join(datadir,'sub-03_0-141.nii.gz')).get_fdata()[:,:,:,0]
# raw_input_2 = nb.load(os.path.join(datadir,'sub-03_0-141.nii.gz')).get_fdata()[:,:,:,0]

# scaler = StandardScaler()
# scaled_input_1 = scaler.fit_transform(raw_input_1.flatten().reshape(-1,1)).reshape(raw_input_1.shape)
# scaled_input_2 = scaler.fit_transform(raw_input_2.flatten().reshape(-1,1)).reshape(raw_input_2.shape)

# scaled_input_1 = np.pad(scaled_input_1,pad_width=((32,32),(32,32),(44,44)),constant_values=0)
# scaled_input_2 = np.pad(scaled_input_2,pad_width=((32,32),(32,32),(44,44)),constant_values=0)

# input_1 = torch.from_numpy(scaled_input_1).float()
# input_2 = torch.from_numpy(scaled_input_2).float()



# input_1 = input_1[None, None, :]
# input_2 = input_2[None, None, :]

num_epochs = 20

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = NCC()#nn.MSELoss()
loss_matrix = regularizer_rot_matrix()
model = AffineNet()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("The model will be running on", device, "device")
device='cpu'
# Convert model parameters and buffers to CPU or Cuda
model.to(device)
input_1.to(device)
input_2.to(device)

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    running_acc = 0.0

        
    # get the inputs
    #images = Variable(images.to(device))
   # labels = Variable(labels.to(device))

    # zero the parameter gradients
    optimizer.zero_grad()
    # predict classes using images from the training set
    outputs = model(input_1,input_2)
    # compute the loss based on model output and real labels
    loss_image = torch.nn.MSELoss()(outputs[0], input_1)#loss_fn.loss(outputs[0], input_1)#
    loss_rot_params = loss_matrix.loss(outputs[1])
    # backpropagate the loss
    print('Loss image:', loss_image)
    print('loss matrix:', loss_rot_params)
    loss = loss_image + loss_rot_params
    loss.backward()
    # adjust parameters based on the calculated gradients
    optimizer.step()
    
  
    
f, ax = plt.subplots(1,2)
ax[0].imshow(np.squeeze(input_2.detach().numpy())[:,:,64]-np.squeeze(input_1.detach().numpy())[:,:,64], cmap='Greys_r')
ax[1].imshow(np.squeeze(outputs[0].detach().numpy())[:,:,64]-np.squeeze(input_1.detach().numpy())[:,:,64], cmap='Greys_r') 

print(outputs[1])
print(outputs[2])   