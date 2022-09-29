# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:55:15 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, pickle, pydicom
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from utils.ManageData4NN import ManageData4NN

#%%
#datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/colab_tests'
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/dcm'

#testing data
testing_files = pickle.load(open(os.path.join(datadir,'dcm_testing_set.pkl'),'rb'))


modeltype = 'NCC_bs8_30ep'
modelname = 'DCM_30ep_8bs_VoxMorph_Aug_NCC'
modeldir = os.path.join(datadir,modeltype,modelname)

model = load_model(os.path.join(datadir,'test_voxmorph'), compile=False)

model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[5].output)


testing_data_mgr = ManageData4NN()
for idx,testing_pair in enumerate(testing_files[:1]):
    
    #the target is in the second position
    trg_dcm_file = pydicom.dcmread(testing_pair[1])
    testing_data_mgr.generate_target(trg_dcm_file)
    targ_input = testing_data_mgr.get_target()
    
    #movable for NN
    mov_dcm_file = pydicom.dcmread(testing_pair[0])
    mov_input = testing_data_mgr.generate_movable(mov_dcm_file)
    

    output = model.predict([targ_input,mov_input])
    # fig, ax = plt.subplots(2,4)

    # output = np.squeeze(output)

    # ax[0,0].imshow(output[:,:,32,0], cmap='Greys_r')
    # ax[0,1].imshow(output[:,:,32,1], cmap='Greys_r')
    # ax[0,2].imshow(output[:,:,32,2], cmap='Greys_r')
    # ax[0,3].imshow(output[:,:,32,3], cmap='Greys_r')
    # ax[1,0].imshow(output[:,:,32,4], cmap='Greys_r')
    # ax[1,1].imshow(output[:,:,32,5], cmap='Greys_r')
    # ax[1,2].imshow(output[:,:,32,6], cmap='Greys_r')
    # ax[1,3].imshow(output[:,:,32,7], cmap='Greys_r')
    
    
#%%Visualize




        