# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:36:25 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
#tensorflow
import tensorflow as tf
from tensorflow.keras.utils import plot_model

#python
import os, pickle, pydicom, time, math
import matplotlib.pyplot as plt
import numpy as np

#custom functions
from network import VoxMorphAffine, AffineNeuralNetworkModel
from losses import NCC, loss_for_matrix, similarity_loss,regularizer_rot_matrix
from generators import  DataGenerator_train, DataGenerator_val
from ManageData4NN import ManageData4NN 


#%% Import data


datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/dcm'

training_files = pickle.load(open(os.path.join(datadir,'dcm_training_set.pkl'), 'rb'))
validation_files = pickle.load(open(os.path.join(datadir,'dcm_validation_set.pkl'), 'rb'))
testing_files = pickle.load(open(os.path.join(datadir,'dcm_testing_set.pkl'),'rb'))

print('Training size:',len(training_files))
print('Validation size:',len(validation_files))
print('Testing size:',len(testing_files))
    
#%% Neural networl

#affine_mdl = AffineNeuralNetworkModel()
affine_mdl = VoxMorphAffine()


print(affine_mdl.summary())
plot_model(affine_mdl, show_shapes=True, show_layer_names=True, dpi=72)

#%%
#initial_learning_rate = 0.0001
# Train the model, doing validation at the end of each epoch
epochs = 1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=10000,
#     decay_rate=0.5,
#     staircase=True)


def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 1
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

adam = tf.keras.optimizers.Adam(
    learning_rate=step_decay(epochs),name="Adam")

loss_class = NCC()
#MI_loss = MutualInformation()

# affine_mdl.compile(loss={'padded_output':similarity_loss,
#                           'rot_matrix':loss_for_matrix,
#                           'trans_matrix':loss_for_vector},
#                           optimizer=adam,run_eagerly=True)
affine_mdl.compile(loss={'padded_output':similarity_loss,
                          'affine_matrix':regularizer_rot_matrix},
                          optimizer=adam,run_eagerly=True)

# affine_mdl.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),#optimizer='Adam'
#                           run_eagerly=True,
#                           loss={'padded_output':tf.keras.losses.MeanSquaredError(),#loss_class.loss,#MI_loss_fun.loss,#similarity_loss,
#                           'affine_matrix':voxmorph_loss_for_matrix})

#affine_mdl.compile(loss=[similarity_loss, loss_for_matrix, loss_for_vector],
#                         optimizer='Adam',run_eagerly=True)



#from tensorflow.keras.callbacks import LearningRateScheduler
# Define callbacks.
#checkpoint_cb = keras.callbacks.ModelCheckpoint(
#    "3d_image_classification.h5", save_best_only=True
#)

early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=0.1, 
                                                  patience=3, 
                                                  verbose=2,
                                                  mode='auto')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(datadir,'checkpoint_test'),
    monitor='val_loss',
    mode='min',
    save_best_only=True)




#load the dict file containing the path of movable and target
#it has been prepared in advance
    

#initialize the training generator
training_generator = DataGenerator_train(training_files, 
                                    batch_size=1, 
                                    dim=(128, 128,128),
                                    n_channels=1, 
                                    shuffle=True)

#initialize the training generator
validation_generator = DataGenerator_val(validation_files, 
                                    batch_size=1, 
                                    dim=(128, 128,128),
                                    n_channels=1, 
                                    shuffle=True)

#fit the model
affine_mdl.fit(
    x = training_generator,
    validation_data = validation_generator, 
    epochs=epochs,
    callbacks=[model_checkpoint_callback],
    verbose=2)

#%%
affine_mdl.save(os.path.join(datadir,'test_voxmorph'))

#%%
affine_mdl = tf.keras.models.load_model(os.path.join(datadir,'test_voxmorph'), compile=False)

#%%
#testing the model
all_motion = []
#instatiate the class
testing_data_mgr = ManageData4NN()

dcm_outdir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/dcm/aligned_test'

for idx,testing_pair in enumerate(testing_files):
    print(idx)
    
    start = time.time()
    #the target is in the second position
    trg_dcm_file = pydicom.dcmread(testing_pair[1])
    testing_data_mgr.generate_target(trg_dcm_file)
    targ_input = testing_data_mgr.get_target()
    
    #movable for NN
    mov_dcm_file = pydicom.dcmread(testing_pair[0])
    testing_data_mgr.generate_movable(mov_dcm_file)
    

    


    #launch the model prediction 
    output,W,b = affine_mdl.predict([targ_input,testing_data_mgr.movable])

    interp_output, motion = testing_data_mgr.process_otput(testing_data_mgr.get_orig_movable(),
                                                          W,
                                                          b)
    #voxelmorph_based version
    #output,affine_matrix = affine_mdl.predict([targ_input,testing_data_mgr.movable])

    # interp_output, motion = testing_data_mgr.process_output_voxmorph(affine_matrix, #the orig movable is a property of the class
    #                                                       mov_dcm_file, 
    #                                                       dcm_outdir, 
    #                                                       idx)
    
    all_motion.append(motion)
    print('Elapsed time:',time.time()-start)
    

#%%
plt.plot(all_motion)
plt.legend(['Trans X', 'Trans Y', 'Trans Z', 'Rot X', 'Rot Y', 'Rot Z'])

orig_target = testing_data_mgr.mosaic_to_mat(pydicom.dcmread(testing_pair[1]))

fig,ax = plt.subplots(2,2)

ax[0,0].imshow(interp_output[:,:,32], cmap='Greys_r', interpolation=None)
ax[0,0].set_title('Aligned movable')
ax[0,1].imshow(orig_target[:,:,32], cmap='Greys_r', interpolation=None)
ax[0,1].set_title('Target')

ax[1,0].imshow(testing_data_mgr.get_orig_movable()[:,:,32], cmap='Greys_r', interpolation=None)
ax[1,0].set_title('Orig movable')
ax[1,1].imshow(orig_target[:,:,32], cmap='Greys', interpolation=None)
ax[1,1].set_title('Target')

fig2, ax2 = plt.subplots(1,2)
ax2[0].imshow(interp_output[:,:,32]-orig_target[:,:,32], cmap='Greys_r')
ax2[0].set_title('Aligned - Target')
ax2[1].imshow(testing_data_mgr.get_orig_movable()[:,:,32]-orig_target[:,:,32], cmap='Greys_r')
ax2[1].set_title('Orig - Target')

print('Diff aligned-target:', abs(np.sum(interp_output[:,:,32]-orig_target[:,:,32])))
print('Diff movable-target:', abs(np.sum(testing_data_mgr.get_orig_movable()[:,:,32]-orig_target[:,:,32])))

#ax[2].imshow(output[:,:,32], cmap='grey')
#ax[2].imshow(orig_target[:,:,32], alpha=0.5, cmap='jet')

#%%
from scipy.stats import zscore
fig,ax = plt.subplots(1,3)

zscore_orig_target = zscore(testing_data_mgr.orig_target.reshape(-1,1)).reshape((testing_data_mgr.orig_target.shape))
zscore_orig_movable = zscore(testing_data_mgr.orig_movable.reshape(-1,1)).reshape((testing_data_mgr.orig_movable.shape))

ax[0].imshow(testing_data_mgr.movable[0,:,:,64,0]-testing_data_mgr.target[0,:,:,64,0], cmap='Greys_r')
ax[1].imshow(testing_data_mgr.orig_movable[:,:,32]-testing_data_mgr.orig_target[:,:,32], cmap='Greys_r')
ax[2].imshow(zscore_orig_movable[:,:,32]-zscore_orig_target[:,:,32], cmap='Greys_r')
