# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:54:27 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import os, torch, shutil
import torchio as tio
import nibabel as nb
import ants
from nilearn.image import new_img_like 
from scipy.io import loadmat
from scipy.ndimage import affine_transform
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from Norm2RAS import convert_mat


  

def output_processing(movable,outputs,orig_dim):
    
   
    matrix = np.squeeze(outputs[1].cpu().detach().numpy())
    data_output = torch.squeeze(outputs[0],0)
    
    #################### CROPPING #############################################
    #crop the movable first to get the proper world affine
    orig_dim = [val.detach().numpy()[0] for val in orig_dim]
    
    #include=data beacuse it is a dict and not a TorchIO obj
    cropping = tio.transforms.CropOrPad(tuple(orig_dim))
    
    #get the affine. This is the NII affine without padding (original nifti) in
    # RAs+ space
    world_affine = np.squeeze(movable['affine'].cpu().detach().numpy())#numpy array float
    
    #crop the output vol
    crop_vol = cropping(data_output)
    crop_vol = np.squeeze(crop_vol.cpu().detach().numpy())

   
    ################### AFFINE CONVERSION #####################################

    matrix = convert_mat(matrix,world_affine)
    
    ################### MOTION PARAMETERS #####################################
    motion_params = ants_motion(matrix)   
    
    
    return crop_vol, matrix, motion_params

    
#%% Align volumes with RAS+ matrix and estimate MSE 

def align_and_mse(fixed, movable, matrix, orig_dim):
    '''This funxtion apply the affine matrix to a volume with the sicpy function 
    and then it estimates the MSE with the fixed'''
    
    loss = torch.nn.MSELoss()

    if len(fixed.shape) == 3:
        #align and transoform to tensor
    
        
        if matrix.size != 1:
            #real matrix (push/backward mapping) 
            matrix = np.vstack((matrix,np.array([0,0,0,1])))
            movable = affine_transform(movable, np.linalg.inv(matrix))

         
        fixed = np.expand_dims(fixed, 0)
        movable = np.expand_dims(movable, 0)    
        fixed = tio.ScalarImage(tensor = fixed)
        movable = tio.ScalarImage(tensor = movable) 
        
        transform = tio.Compose([
            tio.transforms.ToCanonical(),
            tio.transforms.RescaleIntensity(out_min_max=(0, 1)), #MinMaxscaling
            tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.50)), #masking
            tio.transforms.CropOrPad((128,128,128)) #padding
        ])
        
        
        fixed = transform(fixed)
        movable = transform(movable)
    
        
        #estimathe MSe loss
        rmse = loss(fixed['data'], movable['data'])
    
    
    else:  
        #the input received is the output of the network
        rmse = loss(fixed, movable)
        
        
    return rmse.item()
        
 
    
    

#%% 

    
def compare_affine_params(affine1, affine2):
    
    return(abs(abs(affine1)-abs(affine2)))


def params2mat(params):
    
    ''' Function to transform ants matrix to RAS+ space
    Inspired by https://github.com/netstim/leaddbs/blob/master/helpers/ea_antsmat2mat.m'''
    
    mat = np.eye(4)
    c = np.eye(4)
    
    rot = params['AffineTransform_float_3_3'][:9].reshape(3,3)
    c[:-1,-1] = params['fixed'].reshape(1,-1)
    
    
    #translation
    mat[:-1,-1] = params['AffineTransform_float_3_3'][9:].reshape(1,-1)
    mat[:-1,:-1] = rot
    #get a 3x4 matrix
    
    mat2 = c @ np.linalg.inv(mat) @ np.linalg.inv(c)

    # mat2 = mat2*np.array([[1,  1, -1, -1],
    #  [1,1,-1,-1],
    #  [-1,-1, 1,  1],
    #  [1, 1,1, 1]])
    lps2ras = np.diag([-1,-1,1,1])
    mat2 = lps2ras @ np.linalg.inv(mat2) @ np.linalg.inv(lps2ras) 
    
    return mat2[:-1,:]

def ants_motion(params):
    
    dx, dy, dz = params[:,-1]
    
    rot_y = np.arcsin(params[0,2])
    
    cos_rot_y = np.cos(rot_y)
    rot_x = np.arctan2(params[1,2] / cos_rot_y, params[2,2] / cos_rot_y)
    rot_z = np.arctan2(params[0,1] / cos_rot_y, params[0,0] / cos_rot_y)
    
    return np.array([dx, dy, dz, 
                     np.degrees(rot_x), np.degrees(rot_y), np.degrees(rot_z)]) 



def ants_moco(datafile, outdir):
    """
    Perform the motion correction with antspy.
    Credit to Alessandra Pizzuti
    Adpated from 
    https://github.com/27-apizzuti/Atomics/blob/main/MotionCorrection/my_ants_rigid_motion_correction.py
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    outdir : TYPE
        DESCRIPTION.

    Returns
    -------
    
    arrays of affine matrices

    """
    
    tmp_dir = os.path.join(outdir,'tmp')
    print('Create temporary directory')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    #get the name of the fiel with no extension (assuming no dots in the name)
    basename = os.path.basename(datafile).split('.')[0]
    
    #losd nifti 
    nii = nb.load(datafile)
    data = nii.get_fdata()
    #get the fixed (first vol)
    fixed = data[..., 0]
    
    
    #save as nifti the reference volume
    img = nb.Nifti1Image(fixed, header=nii.header, affine=nii.affine)
    nb.save(img, os.path.join(tmp_dir, '{}_ref_vol.nii.gz'.format(basename)))
    print('...Save {} in {}'.format('{}_ref_vol.nii.gz'.format(basename), os.path.join(tmp_dir)))
    fixed = ants.image_read(os.path.join(tmp_dir, '{}_ref_vol.nii.gz'.format(basename)))
    
    
    
    #prepare output data
    aligned_data = np.zeros_like(data)
    #backward matrices
    bwd_matrices = np.zeros((3,4,data.shape[-1]))
    #forward matrices
    fwd_matrices = np.zeros((3,4,data.shape[-1]))   
    
    bw_motion = np.zeros((data.shape[-1],6))
    fw_motion = np.zeros((data.shape[-1],6))

    for idx_vol in range(data.shape[-1]):
        print('Volume {}'.format(idx_vol))
        myvol = data[..., idx_vol]

        # Save individual volumes: this step is needed since ANTS registration input must be both ANTs object.
        img = nb.Nifti1Image(myvol, header=nii.header, affine=nii.affine)
        nb.save(img, os.path.join(tmp_dir, '{}_vol_{}.nii.gz'.format(basename, idx_vol)))
        print('...Save {} in {}'.format('{}_vol_{}.nii.gz'.format(basename, idx_vol), os.path.join(tmp_dir)))
        moving = ants.image_read(os.path.join(tmp_dir, '{}_vol_{}.nii.gz'.format(basename, idx_vol)))
    
        print('...Find trasformation matrix for {}, vol {}'.format(basename, idx_vol))
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform = 'Rigid')
        print('...Save transformation matrix')
        os.system(f"cp {mytx['fwdtransforms'][0]} {outdir}/{basename}_vol_{idx_vol}_affine.mat")
        fwd_params = loadmat(os.path.join(outdir,'{}_vol_{}_affine.mat'.format(basename, idx_vol)))
        fwd_matrices[...,idx_vol] = params2mat(fwd_params)


        os.system(f"cp {mytx['invtransforms'][0]} {outdir}/{basename}_vol_{idx_vol}_affine.mat")
        bwd_params = loadmat(os.path.join(outdir,'{}_vol_{}_affine.mat'.format(basename, idx_vol)))
        bwd_matrices[...,idx_vol] = params2mat(bwd_params)
        os.system(f"rm {outdir}/{basename}_vol_{idx_vol}_affine.mat")        
        
        
        bw_motion[idx_vol,...] = ants_motion(params2mat(bwd_params))
        fw_motion[idx_vol, ...] = ants_motion(params2mat(fwd_params))

    
        # // Apply transformation
        mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, 
                                              transformlist=mytx['fwdtransforms'], 
                                              interpolator='bSpline')
        ants.image_write(mywarpedimage, os.path.join(tmp_dir, '{}_vol_{}_warped.nii.gz'.format(basename, idx_vol)))
        # Step needed to read the warped image
        nii2 = nb.load(os.path.join(tmp_dir, '{}_vol_{}_warped.nii.gz'.format(basename, idx_vol)))
        mywarp = nii2.get_fdata()
        aligned_data[..., idx_vol] = mywarp
        
    new_nii = new_img_like(nii,aligned_data)
    new_nii.to_filename(os.path.join(outdir,'{}_ants_warped.nii.gz'.format(basename)))
    print('ANTs aligned nifti saved!')
    print('... Removing temporary directory')
    shutil.rmtree(tmp_dir)
    
    
        
    return bwd_matrices, fwd_matrices, fw_motion, bw_motion, aligned_data
    
    
    
