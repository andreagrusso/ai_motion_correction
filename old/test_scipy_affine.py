# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, time
import scipy.ndimage as nd
import nibabel as nb
import numpy as np
from scipy.io import loadmat

#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/ants_test_scipy'

fixed_file = os.path.join(datadir,'fixed.nii.gz')
movable_file = os.path.join(datadir,'movable.nii.gz')
mat_file = os.path.join(datadir,'aligned0GenericAffine.mat')

fixed_data = nb.load(fixed_file)
movable_data = nb.load(movable_file)
mat = loadmat(mat_file)
aff = np.eye(4)
aff[:3, :3] = mat['AffineTransform_double_3_3'][:9].reshape((3,3))
aff[:3, 3] =  mat['AffineTransform_double_3_3'][9:].squeeze()

start = time.time()
aligned = nd.affine_transform(movable_data.get_fdata(), np.linalg.inv(aff), order=3)
print(time.time()-start)



#%%
from nilearn.image import new_img_like

aligned_nii = new_img_like(fixed_data, aligned, copy_header=True)
aligned_nii.to_filename(os.path.join(datadir,'scipy_aligned_invmat.nii.gz'))
#%%
#print(fixed_data.header)