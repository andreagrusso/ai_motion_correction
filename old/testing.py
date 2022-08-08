# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:23:14 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

import os, pickle, pydicom, time

# from utils import affine_flow, DataGenerator, \
#     Dense3DSpatialTransformer, similarity_loss, \
#         loss_for_matrix, loss_for_vector

from utils_dcm_2inputs import affine_flow, DataGenerator, \
    Dense3DSpatialTransformer, similarity_loss, \
        loss_for_matrix, loss_for_vector
        
from ManageData4NN import ManageData4NN

#%%

datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/train_dcm'

training_files = pickle.load(open(os.path.join(datadir,'dcm_training_set.pkl'), 'rb'))
validation_files = pickle.load(open(os.path.join(datadir,'dcm_validation_set.pkl'), 'rb'))
testing_files = pickle.load(open(os.path.join(datadir,'dcm_testing_set.pkl'),'rb'))
#%%

json_file = open(os.path.join(datadir,'test_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json,
                                               custom_objects={'Dense3DSpatialTransformer':Dense3DSpatialTransformer, 
                         'affine':affine_flow})
# load weights into new model
loaded_model.load_weights(os.path.join(datadir,"test_model.h5"))
print("Loaded model from disk")

#%%
testing_data_mgr = ManageData4NN()

for testing_pair in testing_files:
    
    start = time.time()
    #the target is in the second position
    testing_data_mgr.generate_target(pydicom.dcmread(testing_pair[1]))
    targ_input = testing_data_mgr.get_target()
    
    #movable
    mov_input = testing_data_mgr.generate_movable(pydicom.dcmread(testing_pair[0]))


    #launch the model prediction
    
    output,W,b = affine_mdl.predict([targ_input,mov_input])

    interp_outut, motion = testing_data_mgr.process_otput(output,W,b)   
    print('Elapsed time:',time.time()-start)