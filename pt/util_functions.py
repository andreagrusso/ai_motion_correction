# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:54:27 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import os, torch, math
import torchio as tio
from losses import Dice


# def get_padding(orig_input_dims):
#   desidered_input_dims = [128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
#   axis_diff = (np.array(desidered_input_dims)-np.array(2*orig_input_dims))/2
#   pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
#             for i in range(3)]
#   return pads


# def _3Dpadding(data, pads):
#     """Zeropad the 3D data
#     :param data: 3D numpy array to be padded
#     :return: padded 3D numpy array
#     """
#     return np.pad(data,pad_width=tuple(pads),constant_values=0)



def mosaic_to_mat(mosaic_dcm):
    
    acq_matrix = np.array(mosaic_dcm.AcquisitionMatrix)
    acq_matrix = acq_matrix[acq_matrix!=0]
    vox_col, vox_row = mosaic_dcm.Columns, mosaic_dcm.Rows
    data_2d = mosaic_dcm.pixel_array
    
    if '0x0019, 0x100a' in mosaic_dcm.keys():
        nr_slices = mosaic_dcm[0x0019, 0x100a].value
    else:
        #print('DCM without number of total slices')
        nr_slices = int(vox_col/acq_matrix[1])*int(vox_row/acq_matrix[0])
    
    data_matrix = np.zeros((acq_matrix[0],acq_matrix[1], nr_slices))
    
    col_idx = np.arange(0,vox_col+1,acq_matrix[1])
    row_idx = np.arange(0,vox_row+1,acq_matrix[0])
    
    i=0 #index to substract from the total number of slice
    for r, row_id in enumerate(row_idx[:-1]):
        
        if i==nr_slices-1:
            break
        
        #loop over columns
        for c,col_id in enumerate(col_idx[:-1]):
            
            data_matrix[:,:,i] = data_2d[row_id:row_idx[r+1],col_id:col_idx[c+1]]
            i += 1
            
            if i==nr_slices-1:
                break          

    return data_matrix



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
    
    dice_fn = Dice()
    
    data_tensor = outputs[0]
    rot_params = outputs[1].cpu().detach().numpy()
    trans_params = outputs[2].cpu().detach().numpy()
    
    orig_dim = [val.detach().numpy()[0] for val in orig_dim]
    
    motion_params = np.empty((1,6))
    
    
    #return volume to original dimension
    tensor = tio.ScalarImage(tensor=torch.squeeze(data_tensor,0))
    padding = tio.transforms.CropOrPad(tuple(orig_dim))
    crop_vol = padding(tensor)
    crop_vol = np.squeeze(crop_vol['data'].cpu().detach().numpy())
    
    
    #motion parameters
    motion_params[0,:3] = trans_params.reshape(1,-1)
    
    #rotation params (hopefully in radians)
    #source Rigid Body Registration John Ashburner & Karl J. Friston
    rot_mat = rot_params.reshape(3,3)+np.eye(3)
    q5 = np.arcsin(rot_mat[0,2]) #q5
    motion_params[0,4] = np.rad2deg(q5)
    
    q4 = math.atan2(rot_mat[1,2]/math.cos(q5),
                    rot_mat[2,2]/math.cos(q5)) #q4
    motion_params[0,3] = np.rad2deg(q4)
    
    q6 = math.atan2(rot_mat[0,1]/math.cos(q5),
                    rot_mat[0,0]/math.cos(q5)) #q6
    motion_params[0,5] = np.rad2deg(q6)
    
    
    #estimate the dice coefficient with the target
    dice_post = dice_fn.loss(fixed,data_tensor)
    dice_post = dice_post.cpu().detach().numpy()
    
    #dice index with the original data
    dice_pre = dice_fn.loss(fixed,movable)
    dice_pre = dice_pre.cpu().detach().numpy()

    return crop_vol, motion_params, dice_post, dice_pre

    
    
    