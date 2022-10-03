# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:09:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch, os, pickle
from torch.optim import Adam
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
 
from network import AffineNet
from losses import NCC, regularizer_rot_matrix
from generators import Training_dataset, Validation_dataset


#%% Data import
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'

training_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_training_set.pkl'), 'rb'))
validation_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_validation_set.pkl'), 'rb'))
testing_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_testing_set.pkl'),'rb'))

print('Training size:',len(training_files))
print('Validation size:',len(validation_files))
print('Testing size:',len(testing_files))

#%% Import model

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = NCC()#nn.MSELoss()
loss_matrix = regularizer_rot_matrix()
model = AffineNet()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("The model will be running on", device, "device")
#device='cpu'

#%% Optimizer

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

#%%

max_epochs = 3
params = {'batch_size': 4,
          'shuffle': True}


training_set = Training_dataset(training_files[:100], (128,128,128))
validation_set = Validation_dataset(validation_files[:100], (128,128,128))

training_generator = torch.utils.data.DataLoader(training_set, **params)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)


for epoch in range(max_epochs):

    print('##################################################')
    print('############# TRAINING ###########################')
    #training    
    for fixed, movable in training_generator:   
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(fixed,movable)
        # compute the loss based on model output and real labels
        loss_image = loss_fn.loss(outputs[0], fixed['data'])#torch.nn.MSELoss()(outputs[0], fixed['data'])##
        loss_rot_params = loss_matrix.loss(outputs[1])
        # backpropagate the loss
        print('Loss image:', loss_image)
        print('loss matrix:', loss_rot_params)
        loss = loss_image + loss_rot_params
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()
        

    print('###################################################')
    print('############# VALIDATION ##########################')
    #validation
    for fixed_val, movable_val in validation_generator:   
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(fixed_val,movable_val)
        # compute the loss based on model output and real labels
        loss_image = torch.nn.MSELoss()(outputs[0], fixed_val['data'])#loss_fn.loss(outputs[0], input_1)#
        loss_rot_params = loss_matrix.loss(outputs[1])
        print('Loss image:', loss_image)
        print('loss matrix:', loss_rot_params)
  
    
  
    
f, ax = plt.subplots(2,params['batch_size'])

orig_diff = np.squeeze(movable_val['data']-fixed_val['data'].detach().numpy())
aligned_diff = np.squeeze(outputs[0].detach().numpy()-fixed_val['data'].detach().numpy())

for i,batch in enumerate(range(params['batch_size'])):
    
    ax[0,i].imshow(orig_diff[i,:,:,64], cmap='Greys_r')
    ax[1,i].imshow(aligned_diff[i,:,:,64], cmap='Greys_r')
    

    
#ax[0].imshow(np.squeeze(movable_val['data'].detach().numpy())[2,:,:,64]-np.squeeze(fixed_val['data'].detach().numpy())[2,:,:,64], cmap='Greys_r')
#ax[1].imshow(np.squeeze(outputs[0].detach().numpy())[2,:,:,64]-np.squeeze(fixed_val['data'].detach().numpy())[2,:,:,64], cmap='Greys_r') 

# print(outputs[1])
# print(outputs[2])   