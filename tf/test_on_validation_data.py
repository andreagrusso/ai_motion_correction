# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:35:50 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import os, pydicom, glob, pickle
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from ManageData4NN import ManageData4NN


#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/dcm'


validation_files = pickle.load(open(os.path.join(datadir,'dcm_training_set.pkl'), 'rb'))

#sort according the movable files
validation_files.sort(key = lambda validation_files: validation_files[0])

list_pairs = validation_files[:166]


datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'

trg_dcm_files = sorted(glob.glob(os.path.join(datadir,'dcm',
                                                'test_nifti',
                                                'test_on_training_and_val',
                                                'DCM',
                                                '*.dcm')))

trg_dcm_file = trg_dcm_files[0]

#model = tf.keras.models.load_model('C:/Users/NeuroIm/Documents/data/ai_motion_correction/colab_tests/CC_bs1_30ep/DCM_30ep_1bs_VoxMorphAffine_Aug_CC')
model = load_model(os.path.join(datadir,
                                'colab_tests',
                                'CC_bs1_30ep_RotMatTanh_dp03',
                                'DCM_30ep_1bs_VoxMorphAffine_tanh_dropout_Aug_CC'), 
                   compile=False)


spm_motion = np.loadtxt(os.path.join(datadir,'dcm',
                                                'test_nifti',
                                                'test_on_training_and_val',
                                                'rp_test_on_training_and_val_fMRI_IMAGERY_20170504131722_3.txt'))



dcm_outdir = os.path.join(datadir,'tmp')
data_mgr = ManageData4NN()


all_motion = np.zeros_like(spm_motion)


for idx,pair in enumerate(list_pairs):
    
    trg_dcm_file = pair[1]
    mov_dcm_file = pair[0]
    tmp_name = os.path.basename(mov_dcm_file).split('.')[0]
    real_idx = int(tmp_name.split('-')[-1])
    print(real_idx)
    
    data_mgr.generate_target(pydicom.dcmread(trg_dcm_file))
    targ_input = data_mgr.get_target()

    #movable for NN
    mov_dcm = pydicom.dcmread(mov_dcm_file)
    data_mgr.generate_movable(mov_dcm)





    #launch the model prediction 
    output,affine_matrix = model.predict([targ_input,data_mgr.movable])
    


    interp_output, motion = data_mgr.process_output_voxmorph(affine_matrix, #the orig movable is a property of the class
                                                          mov_dcm, 
                                                          dcm_outdir, 
                                                          idx)

    all_motion[real_idx,:] = motion
    
    
    
#%%load SPM motion


    
# ai_motion = pd.read_csv(os.path.join(datadir,'dcm',
#                                                 'test_nifti',
#                                                 'test_on_training_and_val',
#                                                 'val_sub_motion.csv'), header=None).values
ai_motion = np.array(all_motion)
    
fig, ax = plt.subplots(2,3)

ax[0,0].plot(ai_motion[:,0])
ax[0,0].plot(spm_motion[:,0])
ax[0,0].set_title('X trans')
ax[0,0].legend(['AI','SPM'],loc="upper right")


ax[0,1].plot(ai_motion[:,1])
ax[0,1].plot(spm_motion[:,1])
ax[0,1].set_title('Y trans')
ax[0,1].legend(['AI','SPM'],loc="upper right")


ax[0,2].plot(ai_motion[:,2])
ax[0,2].plot(spm_motion[:,2])
ax[0,2].set_title('Z trans')
ax[0,2].legend(['AI','SPM'],loc="upper right")


ax[1,0].plot(ai_motion[:,3])
ax[1,0].plot(spm_motion[:,3])
ax[1,0].set_title('X rot')
ax[1,0].legend(['AI','SPM'],loc="upper right")


ax[1,1].plot(ai_motion[:,4])
ax[1,1].plot(spm_motion[:,4])
ax[1,1].set_title('Y rot')
ax[1,1].legend(['AI','SPM'],loc="upper right")


ax[1,2].plot(ai_motion[:,5])
ax[1,2].plot(spm_motion[:,5])
ax[1,2].set_title('Z rot')
ax[1,2].legend(['AI','SPM'],loc="upper right")


