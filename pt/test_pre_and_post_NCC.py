# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:53:11 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import torch, os
from losses import NCC
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

#%% Data
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/dcm/test_nifti'
orig_file = 'DCM_fMRI_RESPONSE_P02_20170613140929_4.nii'
aligned_file =  'rDCM_fMRI_RESPONSE_P02_20170613140929_4.nii'

orig_data = nib.load(os.path.join(datadir,orig_file)).get_fdata()
aligned_data = nib.load(os.path.join(datadir,aligned_file)).get_fdata()

print(np.sum(orig_data-aligned_data))

#%% Loss function



orig_fixed_vol = orig_data[:,:,:,0]
tensor_orig_fixed = torch.unsqueeze(torch.Tensor(orig_fixed_vol),0)
aligned_fixed_vol = aligned_data[:,:,:,0]
tensor_aligned_fixed = torch.unsqueeze(torch.Tensor(aligned_fixed_vol),0)

all_orig_ncc = []
all_aligned_ncc = []

for i in range(1,orig_data.shape[-1]):
    
    orig_mov_tensor = torch.unsqueeze(torch.Tensor(orig_data[:,:,:,i]),0)
    aligned_mov_tensor = torch.unsqueeze(torch.Tensor(aligned_data[:,:,:,i]),0)
    
    orig_ncc = NCC(tensor_orig_fixed.cuda(), orig_mov_tensor.cuda())
    aligned_ncc = NCC(tensor_aligned_fixed.cuda(), aligned_mov_tensor.cuda())
    
    all_orig_ncc.append(orig_ncc.cpu().detach().numpy())
    all_aligned_ncc.append(aligned_ncc.cpu().detach().numpy())
    
    

f, ax = plt.subplots(1,1)
ax.plot(all_aligned_ncc)
ax.plot(all_orig_ncc)
plt.legend(['Aligned','Orig'])
    
    

    
    
    
