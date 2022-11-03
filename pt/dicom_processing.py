#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:00:11 2022

@author: ubuntu22
"""

from dicom2nifti.convert_dicom import dicom_array_to_nifti
import pydicom, os
import numpy as np

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
        
    nii = dicom_array_to_nifti(dicom_header,outfile)

    
    return nii['NII_FILE']
    
    


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
        
    
    
  