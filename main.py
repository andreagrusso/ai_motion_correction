# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:36:25 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
from tensorflow.keras.utils import plot_model

import os, pickle, pydicom, time
import numpy as np
import matplotlib.pyplot as plt


from affine_network import AffineNeuralNetworkModel
from utils_dcm_2inputs import  DataGenerator, similarity_loss, \
        loss_for_matrix, loss_for_vector
        
from ManageData4NN import ManageData4NN

#%% Import data


datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/train_dcm'

training_files = pickle.load(open(os.path.join(datadir,'dcm_training_set.pkl'), 'rb'))
validation_files = pickle.load(open(os.path.join(datadir,'dcm_validation_set.pkl'), 'rb'))
testing_files = pickle.load(open(os.path.join(datadir,'dcm_testing_set.pkl'),'rb'))

    
#%% Neural networl

affine_mdl = AffineNeuralNetworkModel()

print(affine_mdl.summary())
plot_model(affine_mdl, show_shapes=True, show_layer_names=True, dpi=72)

#%%
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True)

tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,name="Adam")


affine_mdl.compile(loss={'padded_output':similarity_loss,
                         'rot_matrix':loss_for_matrix,
                         'trans_matrix':loss_for_vector},
                         optimizer='Adam',run_eagerly=True)

#affine_mdl.compile(loss=[similarity_loss, loss_for_matrix, loss_for_vector],
#                         optimizer='Adam',run_eagerly=True)



#from tensorflow.keras.callbacks import LearningRateScheduler
# Define callbacks.
#checkpoint_cb = keras.callbacks.ModelCheckpoint(
#    "3d_image_classification.h5", save_best_only=True
#)
#early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 1


#load the dict file containing the path of movable and target
#it has been prepared in advance
    

#initialize the training generator
training_generator = DataGenerator(training_files, 
                                   batch_size=1, 
                                   dim=(128, 128,128),
                                   n_channels=1, 
                                   shuffle=True)

#initialize the training generator
validation_generator = DataGenerator(validation_files, 
                                   batch_size=2, 
                                   dim=(128, 128,128),
                                   n_channels=1, 
                                   shuffle=True)

#fit the model
affine_mdl.fit(
    x = training_generator,
    validation_data = validation_generator, 
    epochs=epochs,
    #shuffle=True,
    verbose=2)

#%%
affine_mdl.save(os.path.join(datadir,'test_save'))


#%%

# affine_mdl.save(os.path.join(datadir,'test_model'))
# affine_mdl.save_weights(os.path.join(datadir,'test_model.h5'))
# model_json = affine_mdl.to_json()
# with open(os.path.join(datadir,"test_model.json"), "w") as json_file:
#     json_file.write(model_json)
    
#%%
#testing the model

#instatiate the class
testing_data_mgr = ManageData4NN()
all_motion = []

for testing_pair in testing_files:
    
    start = time.time()
    #the target is in the second position
    testing_data_mgr.generate_target(pydicom.dcmread(testing_pair[1]))
    targ_input = testing_data_mgr.get_target()
    
    #movable for NN
    mov_input = testing_data_mgr.generate_movable(pydicom.dcmread(testing_pair[0]))
    

    


    #launch the model prediction
    
    output,W,b = affine_mdl.predict([targ_input,mov_input])

    interp_outut, motion = testing_data_mgr.process_otput(testing_data_mgr.get_orig_movable(),
                                                          W,
                                                          b)
    all_motion.append(motion)
    print('Elapsed time:',time.time()-start)

