# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:27:15 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import pydicom
import os

#%%
# def _mosaic_to_block(mosaic_dcm):
#     """
#     Convert a mosaic slice to a block of data by reading the headers, splitting the mosaic and appending
#     """
#     # get the mosaic type
#     mosaic_type = _get_mosaic_type(mosaic)
#     number_x, number_y, size_x, size_y, size_z = _get_mosaic_block_dimensions(mosaic)

#     # recreate 2d slice
#     data_2d = mosaic.pixel_array
#     # create 3d block
#     data_3d = np.zeros((size_z, size_y, size_x), dtype=data_2d.dtype)
#     # fill 3d block by taking the correct portions of the slice
#     z_index = 0
#     for y_index in range(0, number_y):
#         if z_index >= size_z:
#             break
#         for x_index in range(0, number_x):
#             if mosaic_type == MosaicType.ASCENDING:
#                 data_3d[z_index, :, :] = data_2d[size_y * y_index:size_y * (y_index + 1),
#                                          size_x * x_index:size_x * (x_index + 1)]
#             else:
#                 data_3d[size_z - (z_index + 1), :, :] = data_2d[size_y * y_index:size_y * (y_index + 1),
#                                                          size_x * x_index:size_x * (x_index + 1)]
#             z_index += 1
#             if z_index >= size_z:
#                 break
#     # reorient the block of data
#     data_3d = np.transpose(data_3d, (2, 1, 0))
    


#%% 
def mosaic_to_mat(mosaic_dcm):
    
    acq_matrix = np.array(mosaic_dcm.AcquisitionMatrix)
    acq_matrix = acq_matrix[acq_matrix!=0]
    vox_col, vox_row = mosaic_dcm.Columns, mosaic_dcm.Rows
    data_2d = mosaic_dcm.pixel_array
    nr_slices = mosaic_dcm[0x0019, 0x100a].value
    
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



# def bg_removal(matrix):
    
#     xm, ym = np.ogrid[0:512:10, 0:512:10]
#     markers = np.zeros_like(a).astype(int16)
#     markers[xm, ym]= np.arange(xm.size*ym.size).reshape((xm.size,ym.size))
#     res2 = ndimage.watershed_ift(a.astype(uint8), markers)
#     res2[xm, ym] = res2[xm-1, ym-1] # remove the isolate seeds
#     imshow(res2, cmap=cm.jet)
#%%
test_dcm_dir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/P01/imagery/DCM'
test_dcm_file = 'micluh_04052017_MREG_P1_MM-0003-0001-00001.dcm'

mosaic_dcm = pydicom.dcmread(os.path.join(test_dcm_dir,test_dcm_file))
#orientation = dcm.ImageOrientationPatient

matrix = mosaic_to_mat(mosaic_dcm)