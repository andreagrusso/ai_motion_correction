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
 
from network import AffineNet,Unet_Stn, ReSTN
from losses import NCC, regularizer_rot_matrix
from generators import Create_dataset
from util_functions import output_processing
from online_create_pairs import create_pairs

# from dicom2nifti.convert_dicom import dicom_array_to_nifti
# import pydicom
#%% Data import
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'
outdir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/preliminary_nn_results'

#%% Import model

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
# loss_fn = NCC()#nn.MSELoss()
loss_matrix = regularizer_rot_matrix()
model = AffineNet()#ReSTN()#Unet_Stn() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")
model.to(device)

#%% Optimizer

optimizer = Adam(model.parameters(), lr=0.0001)
lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 2, 
                                              gamma=0.5, 
                                              last_epoch=- 1, 
                                              verbose=False)

#%% Set some variables

max_epochs = 10
batch = 1

training_image_loss = dict()
training_matrix_loss = dict()

validation_image_loss = dict()
validation_matrix_loss = dict()

#to save the best model and show parameters
min_valid_loss = np.inf



#%% Loop for training

for epoch in range(max_epochs):
    
    nii2remove = []
    #create training pairs on the fly
    training_files, validation_files = create_pairs(os.path.join(datadir,'dcm'))
    print('New set of pairs!')
    print(len(training_files))
    print(len(validation_files))
    
    training_set = Create_dataset(training_files, 
                                                 (128,128,128))    
    training_generator = torch.utils.data.DataLoader(training_set, 
                                                            batch_size = batch, 
                                                            shuffle=True)
    
    validation_set = Create_dataset(validation_files, (128,128,128))
    validation_generator = torch.utils.data.DataLoader(validation_set,
                                                       batch_size = batch, 
                                                       shuffle=True)
    
    
    #create list in dict to save losses
    epoch_key = 'Epoch'+str(epoch)
    training_image_loss[epoch_key]=[]
    training_matrix_loss[epoch_key]=[]
    validation_image_loss[epoch_key]=[]
    validation_matrix_loss[epoch_key]=[]
    
    
    #dicom_array_to_nifti(pydicom.dcmread(training_files[0][1]),'test.dcm',reorient_nifti=True)
    ####################TRAINING####################################
    start = time.time()
    # if epoch == 0:
        
    #training    
    for fixed, movable, orig_dim in training_generator:
        
        #nii2remove += nii1
        #nii2remove += nii2
        
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
        training_image_loss[epoch_key].append(loss_image.cpu().detach().numpy())
        training_matrix_loss[epoch_key].append(loss_rot_params.cpu().detach().numpy())
  
    print('###########################################################')
    print('################## TRAINING SCORES ########################')
    print('Mean Loss Image',epoch_key,':',np.average(training_image_loss[epoch_key]))
    print('Mean Loss Matrix',epoch_key,':',np.average(training_matrix_loss[epoch_key]))
        


    ###############VALIDATION#############################
    for fixed_val, movable_val, orig_dim in validation_generator:  
        
        #nii2remove += nii1
        #nii2remove += nii2
        

        # predict classes using images from the training set
        outputs = model(fixed_val['data'].type(torch.FloatTensor).to(device),
                        movable_val['data'].type(torch.FloatTensor).to(device))
        
        # compute the loss based on model output and real labels
        val_loss_image = NCC(outputs[0], fixed_val['data'])#torch.nn.MSELoss()(outputs[0], fixed_val['data'])
        val_loss_rot_params = loss_matrix.loss(outputs[1])
        val_loss = val_loss_image  + val_loss_rot_params
  
        #storing losses
        validation_image_loss[epoch_key].append(val_loss_image.cpu().detach().numpy())
        validation_matrix_loss[epoch_key].append(val_loss_rot_params.cpu().detach().numpy())
        

    print('###########################################################')
    print('################## VALIDATION SCORES ######################')
    print('Mean Loss Image',epoch_key,':',np.average(validation_image_loss[epoch_key]))
    print('Mean Loss Matrix',epoch_key,':',np.average(validation_matrix_loss[epoch_key]))
    
    mean_valid_loss = np.average(validation_image_loss[epoch_key])

    if min_valid_loss > mean_valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{mean_valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = mean_valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 
                   os.path.join(outdir,'saved_model.pth'))
        print('Model saved!')
            
    
   
    print('Elapsed time (s):',time.time()-start)
    
    
    #[os.remove(tmp) for tmp in nii2remove]
      
        
#%% Some visualization

orig_diff = np.squeeze((movable_val['data']-fixed_val['data']).detach().numpy())
aligned_diff = np.squeeze((outputs[0].cpu()-fixed_val['data']).detach().numpy())


if batch>1:
    f, ax = plt.subplots(2,batch)
    
    for i,b in enumerate(range(batch)):
            
        ax[0,i].imshow(orig_diff[i,:,:,64], cmap='Greys_r')
        ax[1,i].imshow(aligned_diff[i,:,:,64], cmap='Greys_r')
else:
    
    f, ax = plt.subplots(2,1)
    ax[0].imshow(orig_diff[:,:,64], cmap='Greys_r')
    ax[1].imshow(aligned_diff[:,:,64], cmap='Greys_r')

#%% Testing data

### LOAD BEST MODEL######
model2 = AffineNet()
model2.load_state_dict(torch.load(os.path.join(outdir,'saved_model.pth')))
model2.eval()
model2 = model2.to(device)

testing_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_sub00_run1_testing_set.pkl'),'rb'))
testing_set = Create_dataset(testing_files, (128,128,128))
testing_generator = torch.utils.data.DataLoader(testing_set, shuffle=False)

motion_params = np.empty((len(testing_set), 6))
aligned_data = []
all_dice_post = []
all_dice_pre = []
i=0

for fixed_test, movable_test, orig_dim in testing_generator: #just testing
    
    outputs = model2(fixed_test['data'].type(torch.FloatTensor).to(device),
                    movable_test['data'].type(torch.FloatTensor).to(device))
    
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
plt.title('AI motion')    

#load spm aligned parameters
spm_params_dir = os.path.join(datadir,'dcm','test','sub10','run7')
file = glob.glob(os.path.join(spm_params_dir,'rp*.txt'))[0]
spm_motion = np.loadtxt(os.path.join(file))

f, ax = plt.subplots(1,1)

ax.plot(spm_motion)
plt.legend(['X trans','Y trans','Z trans',
            'X rot','Y rot','Z rot'])  
plt.title('Orig motion')

#%% Save performance of the both training and validation


training_image_loss_df = pd.DataFrame.from_dict(training_image_loss).astype('float32')
training_matrix_loss_df = pd.DataFrame.from_dict(training_matrix_loss).astype('float32')

validation_image_loss_df = pd.DataFrame.from_dict(validation_image_loss).astype('float32')
validation_matrix_loss_df = pd.DataFrame.from_dict(validation_matrix_loss).astype('float32')


