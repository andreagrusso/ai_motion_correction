# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob
import nibabel as nb
from nilearn.image import new_img_like
import json
import pickle 

#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/train'
colabdir = '/content/drive/MyDrive/Colab Notebooks/ai_motion/data/train'

full_timeseries_dir = os.path.join(datadir,'full_time_series')
output_mov_dir = os.path.join(datadir,'movables')
output_targ_dir = os.path.join(datadir,'targets')

orig_nifti_files = glob.glob(os.path.join(full_timeseries_dir,'*nii.gz'))


training_files = []
training_files_colab = []

for file in orig_nifti_files:
    
    sub_name = os.path.basename(file)[:-7]
    nifti = nb.load(file)
    nifti_data = nifti.get_fdata()
    
    #generate the target
    target_data = nifti_data[:,:,:,0]
    target_data_file = new_img_like(nifti, target_data, copy_header=True)
    target_file_path = os.path.join(output_targ_dir,sub_name +'_00'+'.nii.gz')
    target_data_file.to_filename(target_file_path)
    
    #generate the movables up to 200 volumes per subject 
    for j in range(1,nifti_data.shape[-1]):
              
        movable = new_img_like(nifti, nifti_data[:,:,:,j], copy_header=True)
        movable_file_path = os.path.join(output_mov_dir,sub_name +'_0'+str(j)+'.nii.gz')
        movable.to_filename(movable_file_path)
        
        #add to the dictionary
        training_files.append([movable_file_path,target_file_path])
        training_files_colab.append([movable_file_path.replace(datadir,colabdir),
                               target_file_path.replace(datadir,colabdir)])
       
        
        
pickle.dump(training_files,open(os.path.join(datadir,'training_file.pkl'),'wb'))
pickle.dump(training_files_colab,open(os.path.join(datadir,'colab_training_file.pkl'),'wb'))
        

        


