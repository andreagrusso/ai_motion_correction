# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:09:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch, os, pickle, time, glob 
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import WeightedRandomSampler

 
from network import AffineNet,Unet_Stn, AffineNetCoord
from losses import NCC, regularizer_rot_matrix
from generators import Create_train_dataset, Create_test_dataset
from samplers import mySampler
from util_functions import output_processing
from online_create_pairs import create_pairs, create_pairs_for_testing

# from dicom2nifti.convert_dicom import dicom_array_to_nifti
# import pydicom
#%% Data import
datadir = '/mnt/c/Users/NeuroIm/Documents/data/ai_motion_correction'#/home/ubuntu22/Desktop/ai_mc/'
outdir = '/mnt/c/Users/NeuroIm/Documents/data/ai_motion_correction/preliminary_nn_results/test'#'/home/ubuntu22/Desktop/ai_mc/preliminary_nn_results'


# datadir = '/home/ubuntu22/Desktop/ai_mc/'
# outdir = '/home/ubuntu22/Desktop/ai_mc/preliminary_nn_results/affine_bs1_Multidp03_ep20_aug'
#%% Import model

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
# loss_fn = NCC()#nn.MSELoss()
loss_matrix = regularizer_rot_matrix()
model = AffineNet()#ReSTN()# #Unet_Stn()#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")
model.to(device)

#%% Optimizer

optimizer = Adam(model.parameters(), lr=0.0001)
lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 4, 
                                              gamma=0.5, 
                                              last_epoch=- 1, 
                                              verbose=False)

#%% Set some variables

max_epochs = 5
batch = 4

training_image_loss = dict()
training_matrix_loss = dict()

validation_image_loss = dict()
validation_matrix_loss = dict()

#to save the best model and show parameters
min_valid_loss = np.inf


training_loss = []
validation_loss = []

train_iter = 0
val_iter = 0

nr_of_pairs = -1
#%% Loop for training

for epoch in range(max_epochs):
    
    #create training pairs on the fly
    # training_files, validation_files = create_pairs(os.path.join(datadir,'dcm'))
    training_files, validation_files, sub_weights = create_pairs(os.path.join(datadir,'dcm'),nr_of_pairs)
    sub_weights = torch.Tensor(sub_weights)

    print('Training size', len(training_files))
    print('validation size', len(validation_files))

    print('New set of pairs!')

    
    training_set = Create_train_dataset(training_files,(128,128,128))
    
    my_sampler = WeightedRandomSampler(sub_weights.type('torch.DoubleTensor'), 
                                       len(sub_weights))
    
    training_generator = torch.utils.data.DataLoader(training_set, 
                                                            batch_size = batch, 
                                                            sampler=my_sampler)
    

    
    
    validation_set = Create_test_dataset(validation_files, (128,128,128))
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                        batch_size = batch, 
                                                        shuffle=True)
    

    
    ####################TRAINING####################################
    start = time.time()

        
    #training 
    running_loss = 0.0
    for fixed, movable, orig_dim in training_generator:
        
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(fixed['data'].type(torch.FloatTensor).to(device),
                        movable['data'].type(torch.FloatTensor).to(device))

        # compute the loss based on model output and real labels
        loss_image = NCC(outputs[0], fixed['data']) #torch.nn.MSELoss()(outputs[0], fixed['data'])##
        loss_rot_params = loss_matrix.loss(outputs[1])
        loss = loss_image  + loss_rot_params
        
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()
        
        #storing losses
        training_loss += [loss.item(), train_iter, epoch]
        train_iter +=1
        running_loss += loss.item()

  
    print('###########################################################')
    print('################## TRAINING SCORES ########################')
    print('Epoch ', epoch, ' ', 'average loss: ', running_loss/len(training_files))

        


    ###############VALIDATION#############################
    running_loss = 0.0
    for fixed_val, movable_val, orig_dim in validation_generator:  
        

        

        # predict classes using images from the training set
        outputs = model(fixed_val['data'].type(torch.FloatTensor).to(device),
                        movable_val['data'].type(torch.FloatTensor).to(device))
        
        # compute the loss based on model output and real labels
        val_loss_image = NCC(outputs[0], fixed_val['data'])#torch.nn.MSELoss()(outputs[0], fixed_val['data'])
        val_loss_rot_params = loss_matrix.loss(outputs[1])
        val_loss = val_loss_image  + val_loss_rot_params
  
        #storing losses
        validation_loss += [val_loss.item(), val_iter, epoch]
        val_iter +=1
        running_loss += val_loss.item()

        

    print('###########################################################')
    print('################## VALIDATION SCORES ######################')
    print('Epoch ', epoch, ' ', 'average loss: ', running_loss/len(validation_files))
    mean_valid_loss = running_loss/len(validation_files)
    

    if min_valid_loss > mean_valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{mean_valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = mean_valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 
                   os.path.join(outdir,'AffineNet_saved_model.pth'))
        print('Model saved!')
            
    
    print('Update learning rate')
    lr_schedule.step()
    print('Elapsed time (s):',time.time()-start)
    
    


#%% Save performance of the both training and validation
training_loss = np.array(training_loss).reshape((int(len(training_loss)/3),3))
validation_loss = np.array(validation_loss).reshape((int(len(validation_loss)/3),3))

training_df = pd.DataFrame(data = training_loss, columns=['Loss', 'Iter', 'Epoch'])
validation_df = pd.DataFrame(data = validation_loss, columns=['Loss', 'Iter', 'Epoch'])

training_df.to_csv(os.path.join(outdir,'AffineNet_training_loss.csv'))
validation_df.to_csv(os.path.join(outdir,'AffineNet_validation_loss.csv'))


#%% testing 


# model = AffineNet()
# model.load_state_dict(torch.load(os.path.join(outdir,'AffineNet_data_saved_model.pth')))
# model.eval()
# model = model.to(device)
# sub = 'sub00_run1'
# testing_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_'+sub+'_testing_set.pkl'),'rb'))
# testing_set = Create_test_dataset(testing_files, (128,128,128))
# testing_generator = torch.utils.data.DataLoader(testing_set, shuffle=False)

# motion_params = np.empty((len(testing_set), 6))
# aligned_data = []
# mse = []
# i=0
# timing = []

# for fixed_test, movable_test, orig_dim, world_affine in testing_generator: #just testing
#     start = time.time()
#     outputs = model(fixed_test['data'].type(torch.FloatTensor).to(device),
#                     movable_test['data'].type(torch.FloatTensor).to(device))
    
#     crop_vol, curr_motion, mse_post, mse_pre = output_processing(fixed_test['data'],
#                                                          movable_test['data'],
#                                                          outputs, 
#                                                          orig_dim,
#                                                          world_affine)
#     timing.append(time.time()-start)
#     aligned_data.append(crop_vol)
#     motion_params[i,:] = curr_motion
#     mse += [mse_pre] + [mse_post]
    
#     i +=1


# #%% Just interested in motion
# plt.plot(motion_params)
# plt.legend(['Trans X', 'Trans Y', 'Trans Z',
#             'Rot X', 'Rot Y', 'Rot Z'])
