# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:54:27 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import os, torch, math, shutil
import torchio as tio
from losses import Dice

from dicom2nifti.convert_dicom import dicom_array_to_nifti
import pydicom


def mosaic_to_mat(dcm_file):
    """ Snippets taken from https://dicom2nifti.readthedocs.io/ and arranged for
    our needs.
    """


    
    
    dicom_header = []
    dicom_header.append(pydicom.read_file(dcm_file,
                                 defer_size=None,
                                 stop_before_pixels=False,
                                 force=False))
    
    
      
    outfile = dcm_file[:-3]+'nii'
        
    
    nii = dicom_array_to_nifti(dicom_header,outfile,reorient_nifti=True)
    
    mat = nii['NII'].get_fdata()
    if os.path.isfile(nii['NII_FILE']):
        os.remove(nii['NII_FILE'])
    
    return mat
    
    



# def mosaic_to_mat(mosaic_dcm):
    
    
    
#     acq_matrix = np.array(mosaic_dcm.AcquisitionMatrix)
#     acq_matrix = acq_matrix[acq_matrix!=0]
#     vox_col, vox_row = mosaic_dcm.Columns, mosaic_dcm.Rows
#     data_2d = mosaic_dcm.pixel_array
    
#     if '0x0019, 0x100a' in mosaic_dcm.keys():
#         nr_slices = mosaic_dcm[0x0019, 0x100a].value
#     else:
#         #print('DCM without number of total slices')
#         nr_slices = int(vox_col/acq_matrix[1])*int(vox_row/acq_matrix[0])
    
#     # data_matrix = np.zeros((acq_matrix[0],acq_matrix[1], nr_slices))
#     data_matrix = np.zeros((nr_slices, acq_matrix[1], acq_matrix[0]))

    
#     col_idx = np.arange(0,vox_col+1,acq_matrix[1])
#     row_idx = np.arange(0,vox_row+1,acq_matrix[0])
    
#     i=0 #index to substract from the total number of slice
#     for r, row_id in enumerate(row_idx[:-1]):
        
#         if i==nr_slices-1:
#             break
        
#         #loop over columns
#         for c,col_id in enumerate(col_idx[:-1]):
            
#             c_slice = data_2d[row_id:row_idx[r+1],col_id:col_idx[c+1]]
#             data_matrix[i,:,:] = np.fliplr(c_slice)
#             i += 1
            
#             if i==nr_slices-1:
#                 break          
    
#     data_matrix = np.rot90(np.transpose(data_matrix,[2,1,0]))
#     return data_matrix



def mat_to_mosaic(mosaic_dcm, data_matrix, outdir, idx_dcm, name):
    """
    

    Parameters
    ----------
    mosaic_dcm : dicom file from pydicom
        This is the original dicom file that will be used as a canvas 
        to write the new aligned data.
    data_matrix : numpy 3d data
        3D numpy matrix of the aligned data.
    outdir : string
        Directory where output will be saved.
    idx_dcm : integer
        Number of the dicom aligned compared to the length of the full
        time series.
    name : str
        Name for the new dcm.

    Returns
    -------
    None.

    """
            

    acq_matrix = np.array(mosaic_dcm.AcquisitionMatrix)
    acq_matrix = acq_matrix[acq_matrix!=0]
    vox_col, vox_row = mosaic_dcm.Columns, mosaic_dcm.Rows
    
    nr_slices = data_matrix.shape[2]
    
    old_pixel_array = mosaic_dcm.pixel_array
    new_pixel_array = np.zeros_like(old_pixel_array)
    
    col_idx = np.arange(0,vox_col+1,acq_matrix[1])
    row_idx = np.arange(0,vox_row+1,acq_matrix[0])
    
    i=0 #index to substract from the total number of slice
    for r, row_id in enumerate(row_idx[:-1]):
        
        if i==nr_slices-1:
            break
        
        #loop over columns
        for c,col_id in enumerate(col_idx[:-1]):
            
            new_pixel_array[row_id:row_idx[r+1],col_id:col_idx[c+1]] = data_matrix[:,:,i]
            i += 1
            
            if i==nr_slices-1:
                break
            
    
    #swap the old data with the new
    #mosaic_dcm.pixel_array = new_pixel_array
    mosaic_dcm.PixelData = new_pixel_array.tobytes()

    mosaic_dcm.save_as(os.path.join(outdir,name+str(idx_dcm)+'.dcm'))
        
    
    
    

def output_processing(fixed,movable,outputs,orig_dim):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    dice_fn = Dice()
    mse_fn = torch.nn.MSELoss()
    
    data_tensor = outputs[0]
    matrix = np.squeeze(outputs[1].cpu().detach().numpy())
    
    
    matrix = ThetaToM(matrix, 128, 128, 128)
    matrix = matrix[:-1,:]

    #rot_params = outputs[1].cpu().detach().numpy()
    #trans_params = outputs[2].cpu().detach().numpy()
    
    orig_dim = [val.detach().numpy()[0] for val in orig_dim]
    
    motion_params = np.empty((1,6))
    
    
    #return volume to original dimension
    tensor = tio.ScalarImage(tensor=torch.squeeze(data_tensor,0))
    padding = tio.transforms.CropOrPad(tuple(orig_dim))
    crop_vol = padding(tensor)
    crop_vol = np.squeeze(crop_vol['data'].cpu().detach().numpy())
    
    #vec = np.zeros((1,4))
    #vec[-1] = 1
    #matrix = np.linalg.inv(np.vstack((matrix,vec)))
    #matrix = matrix[:-1,:]
    #motion parameters
    #motion_params[0,:3] = trans_params.reshape(1,-1)
    motion_params[0,:3] = matrix[:,-1].reshape(1,-1)

    
    #rotation params (hopefully in radians)
    #source Rigid Body Registration John Ashburner & Karl J. Friston
    #rot_mat = rot_params.reshape(3,3)+np.eye(3)
    rot_mat = matrix[:,:-1]
    q5 = np.arcsin(rot_mat[0,2]) #q5
    motion_params[0,4] = np.rad2deg(q5)
    
    q4 = math.atan2(rot_mat[1,2]/math.cos(q5),
                    rot_mat[2,2]/math.cos(q5)) #q4
    motion_params[0,3] = np.rad2deg(q4)
    
    q6 = math.atan2(rot_mat[0,1]/math.cos(q5),
                    rot_mat[0,0]/math.cos(q5)) #q6
    motion_params[0,5] = np.rad2deg(q6)
    
    
    #estimate the dice coefficient with the target
    fixed = fixed.to(device)
    #print(fixed.device)
    data_tensor = data_tensor.to(device)
    #print(data_tensor.device)
    movable = movable.to(device)
    #print(movable.device)
    
    dice_post = mse_fn(fixed,data_tensor)#dice_fn.loss(fixed,data_tensor)
    dice_post = dice_post.cpu().detach().numpy()
    
    #dice index with the original data
    dice_pre = mse_fn(fixed,movable)#dice_fn.loss(fixed,movable)
    dice_pre = dice_pre.cpu().detach().numpy()

    return crop_vol, motion_params, dice_post, dice_pre

    
    
#%% Coordinates transformation test

def get_N(W, H, D):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((4, 4), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[0, 2] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[1, 2] = 0
    N[2, 2] = 2.0 / D
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[2, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H, D):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H, D)
    return np.linalg.inv(N)



def ThetaToM(theta, w, h, d, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    
    theta_aug = np.concatenate([theta, np.zeros((1, 4))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h, d)
    N_inv = get_N_inv(w, h, d)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv
    return M

#%% Training loops

