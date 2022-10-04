# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:09:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch, os, pickle
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
 
from network import AffineNet
from losses import NCC, regularizer_rot_matrix
from generators import Training_dataset, Validation_dataset
from util_functions import output_processing


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
# loss_fn = NCC()#nn.MSELoss()
loss_matrix = regularizer_rot_matrix()
model = AffineNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")
model.to(device)

#%% Optimizer

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

#%%

max_epochs = 2
batch_size = 1

training_set = Training_dataset(training_files[:20], (128,128,128))
validation_set = Validation_dataset(validation_files[:20], (128,128,128))

training_generator = torch.utils.data.DataLoader(training_set, batch_size = batch_size, shuffle=True)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = batch_size, shuffle=True)


for epoch in range(max_epochs):

    print('##################################################')
    print('############# TRAINING ###########################')
    #training    
    for fixed, movable, orig_dim in training_generator:   
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(fixed['data'].to(device),movable['data'].to(device))
        # compute the loss based on model output and real labels
        loss_image = NCC(outputs[0], fixed['data'])#torch.nn.MSELoss()(outputs[0], fixed['data'])##
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
    for fixed_val, movable_val,orig_dim in validation_generator:   
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(fixed_val['data'].to(device),movable_val['data'].to(device))
        # compute the loss based on model output and real labels
        loss_image = NCC(outputs[0], fixed_val['data'])#torch.nn.MSELoss()(outputs[0], fixed_val['data'])
        print('Loss image:', loss_image)
        #print('loss matrix:', loss_rot_params)
  
    
  
    
f, ax = plt.subplots(2,batch_size)

orig_diff = np.squeeze((movable_val['data']-fixed_val['data']).detach().numpy())
aligned_diff = np.squeeze((outputs[0].cpu()-fixed_val['data']).detach().numpy())

# for i,batch in enumerate(range(batch_size)):
    
ax[0].imshow(orig_diff[:,:,64], cmap='Greys_r')
ax[1].imshow(aligned_diff[:,:,64], cmap='Greys_r')
    

#%% Testing data

testing_set = Validation_dataset(testing_files, (128,128,128))
testing_generator = torch.utils.data.DataLoader(testing_set, batch_size = 1, shuffle=False)

motion_params = np.empty((len(testing_files), 6))
aligned_data = []
all_dice_post = []
all_dice_pre = []
i=0

for fixed_test, movable_test, orig_dim in testing_generator:
    
    outputs = model(fixed_test['data'].to(device), 
                    movable_test['data'].to(device))
    
    crop_vol, curr_motion, dice_post, dice_pre = output_processing(fixed_test['data'],
                                                         movable_test['data'],
                                                         outputs, 
                                                         orig_dim)
    aligned_data.append(crop_vol)
    motion_params[i,:] = curr_motion
    all_dice_post.append(dice_post)
    all_dice_pre.append(dice_pre)
    
    i +=1
    
    
f, ax = plt.subplots(1,1)

ax.plot(all_dice_post)
ax.plot(all_dice_pre)
plt.legend(['post','pre']) 

f, ax = plt.subplots(1,1)

ax.plot(motion_params)
plt.legend(['X trans','Y trans','Z trans',
            'X rot','Y rot','Z rot'])      


# input_1 = nb.load(os.path.join(datadir,'sub-03_0-141.nii.gz')).get_fdata()[:,:,:,0]
# input_2 = nb.load(os.path.join(datadir,'sub-03_0-141.nii.gz')).get_fdata()[:,:,:,0]
# input_1 = np.pad(input_1,pad_width=((32,32),(32,32),(44,44)),constant_values=0)
# input_2 = np.pad(input_2,pad_width=((32,32),(32,32),(44,44)),constant_values=0)

# input_1 = torch.from_numpy(input_1).float()
# input_2 = torch.from_numpy(input_2).float()



# input_1 = input_1[None, None, :]
# input_2 = input_2[None, None, :]


# test_outputs = model.predict(input_1, input_2)
