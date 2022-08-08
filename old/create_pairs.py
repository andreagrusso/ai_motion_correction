# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, time, glob
import scipy.ndimage as nd
import nibabel as nb
import numpy as np
from nilearn.image import new_img_like


#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/sub05_TBV'
outdir = os.path.join(datadir,'pairs')

nifti_files = glob.glob(os.path.join(datadir,'*nii.gz'))


for file in nifti_files:
    
    sub_name = os.path.basename(file)[:9]
    nifti = nb.load(file)
    nifti_data = nifti.get_fdata()
    fixed_data = nifti_data[:,:,:,0]
    fixed_data = np.expand_dims(fixed_data,axis=-1) #4-th dim
    
    
    for j in range(1,nifti_data.shape[-1]):
        
        tmp_vol = nifti_data[:,:,:,j]
        tmp_vol = np.expand_dims(tmp_vol,axis=-1) #4-th dim
        
        new_vol = np.concatenate((fixed_data, tmp_vol), axis=-1)
        
        pair = new_img_like(nifti, new_vol, copy_header=True)
        pair.to_filename(os.path.join(outdir,sub_name +'_0-'+str(j)+'.nii.gz'))
        


