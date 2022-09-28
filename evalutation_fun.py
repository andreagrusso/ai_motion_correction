# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:45:45 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os
import numpy as np
import nibabel as nb
import tensorflow as tf
from losses import MutualInformation, similarity_loss, NCC, Dice

#%%

def mat2tensor(data):
    """
    
    Parameters
    ----------
    data : 3D numpy array
        3D array that has to be converted in tensor.

    Returns
    -------
    tf_res : tensor
        Tensor with n_channel = 1.

    """
    
    #convert to tensor
    tf_res = tf.convert_to_tensor(data, dtype=tf.float32)
    #add channel (last dim)
    tf_res = tf.expand_dims(tf_res, axis=-1)
    
    return tf_res

    
def evaluate_alignment(ann_mov_file, std_mov_file, trg_file, metric):
    """
    

    Parameters
    ----------
    ann_mov_file : str
        File path of the nifti aligned with Artificial Neural Network.
    std_mov_file : str
        File path of the nifti aligned with a standard procedure.
    target_file : str
        File path of the nifti aligned with SPM (other).
    metric : str
        It could be MI=Mutual Information or CC = cross-correlation.

    Returns
    -------
    ts_metric : numpy array
        It is a numpy array containing the metric for each volume of the 
        time series. Two time series as we evaluate both the AI aligned and the
        nifti aligned with a more standard procedure
    """
    

    ai_data = nb.load(ann_mov_file).get_fdata()
    std_data = nb.load(std_mov_file).get_fdata()
    trg_data = nb.load(trg_file).get_fdata()
    

    #check size
    if ai_data.shape[-1] != trg_data.shape[-1]:
        
        print('Target and test data does not have the same size!')
        return
    
    #array to store the metrics relative to the two time series
    ts_metric = np.empty((2,ai_data.shape[-1]))
    
    #loop over the timepoints
    for i in range(ai_data.shape[-1]):
        
        tf_ai_data = mat2tensor(np.squeeze(ai_data[:,:,:,i]))
        tf_std_data = mat2tensor(np.squeeze(std_data[:,:,:,i]))
        tf_trg_data = mat2tensor(np.squeeze(trg_data[:,:,:,i]))

        
        if metric == 'MI':
            #use the mutual information
            MI_loss = MutualInformation()
            ts_metric[0,i] = tf.reduce_mean(1-MI_loss.volumes(tf_trg_data, tf_ai_data))
            ts_metric[1,i] = tf.reduce_mean(1-MI_loss.volumes(tf_trg_data, tf_std_data))

        if metric == 'CC':
            #use the cross correlation
            ts_metric[0,i] = similarity_loss(tf_trg_data, tf_ai_data)
            ts_metric[1,i] = similarity_loss(tf_trg_data, tf_std_data)
        
        if metric == 'NCC':
            
            NCC_loss = NCC()
            #use the normalized cross correlation
            ts_metric[0,i] = tf.reduce_mean(NCC_loss.loss(tf_trg_data, tf_ai_data))
            ts_metric[1,i] = tf.reduce_mean(NCC_loss.loss(tf_trg_data, tf_std_data))
            
        if metric == 'DICE':
            
            bin_trg_data = tf.where(tf_trg_data>0,1,0)
            bin_ai_data = tf.where(tf_ai_data>0,1,0)
            bin_std_data = tf.where(tf_std_data>0,1,0)
            
            bin_trg_data = tf.cast(bin_trg_data, tf.float32)
            bin_ai_data = tf.cast(bin_ai_data, tf.float32)
            bin_std_data = tf.cast(bin_std_data, tf.float32)
            
            Dice_loss = Dice()
            #use the normalized cross correlation
            ts_metric[0,i] = tf.reduce_mean(Dice_loss.loss(bin_trg_data, bin_ai_data))
            ts_metric[1,i] = tf.reduce_mean(Dice_loss.loss(bin_trg_data, bin_std_data))

    return ts_metric         
        
        