# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:56:37 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, pickle, pydicom, time
import matplotlib.pyplot as plt
import numpy as np
from evalutation_fun import evaluate_alignment

#%%

datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'
trg_file = os.path.join(datadir,'dcm','test_nifti','DCM_fMRI_RESPONSE_P02_20170613140929_4.nii')
ann_mov_file = os.path.join(datadir,'colab_tests','CC_bs1_30ep_RotMatTanh_dp03','nifti.nii.gz')
std_mov_file = os.path.join(datadir,'dcm','test_nifti','rDCM_fMRI_RESPONSE_P02_20170613140929_4.nii')
motion_params = os.path.join(datadir,'dcm','test_nifti','rp_DCM_fMRI_RESPONSE_P02_20170613140929_4.txt')

metric = 'NCC'

output_metric = evaluate_alignment(ann_mov_file, std_mov_file, trg_file, metric)

fig,ax = plt.subplots()

ax.plot(output_metric.T)
plt.legend(['AI','STD'])
plt.tight_layout()

motion = np.loadtxt(motion_params)
fig2, ax2 = plt.subplots()
ax2.plot(motion)
ax2.legend(labels=['Trans X','Trans Y','Trans Z',
                  'Rot X','Rot Y','Rot Z'],
           loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()