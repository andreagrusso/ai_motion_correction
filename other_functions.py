# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:47:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import tensorflow as tf


def get_padding(orig_input_dims):
  desidered_input_dims = [128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
  axis_diff = (np.array(desidered_input_dims)-np.array(2*orig_input_dims))/2
  pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
            for i in range(3)]
  return pads



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


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])


def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3




