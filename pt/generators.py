# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 09:12:41 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import torch 
import torchio as tio

import numpy as np
import pydicom

from util_functions import mosaic_to_mat
import numpy as np

#%%

class Training_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        
        

    def __len__(self):
        """Denotes the number training data
        :return: number of training data
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        """ Generates one sample of data """


        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        pair = self.pairs[ID]
        trg_mat = mosaic_to_mat(pydicom.dcmread(pair[1]))
        mov_mat = mosaic_to_mat(pydicom.dcmread(pair[0]))
        orig_dim = mov_mat.shape

        
        trg_mat = np.expand_dims(trg_mat, axis=0)
        mov_mat = np.expand_dims(mov_mat, axis=0)
        
        #try to use torchio for preprocessing
        mov_tensor = tio.ScalarImage(tensor=mov_mat)
        trg_tensor = tio.ScalarImage(tensor=trg_mat)
        
        #Min Max scaling
        scaler = tio.transforms.RescaleIntensity(out_min_max=(0, 1))
        scaled_mov = scaler(mov_tensor)
        scaled_trg = scaler(trg_tensor)
        
        #masking
        #the threshld is the median. It looks high but it works
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.5))
        m_scaled_mov = mask(scaled_mov)
        m_scaled_trg = mask(scaled_trg)
        
        #padding
        padding = tio.transforms.CropOrPad((128,128,128))
        zeropad_mov = padding(m_scaled_mov)
        zeropad_trg = padding(m_scaled_trg)
        
        if np.sum(mov_mat-trg_mat)!=0:
        
            #ranfdom affine -5:5 degree and mm
            affine_aug = tio.transforms.RandomAffine(scales=0,
                                               degrees=[np.random.uniform(5),
                                                        np.random.uniform(5),
                                                        np.random.uniform(5)],
                                               translation=[np.random.uniform(5),
                                                        np.random.uniform(5),
                                                        np.random.uniform(5)],
                                               isotropic=True)
            
            aug_zeropad_mov = affine_aug(zeropad_mov)
            
                   
    
            return zeropad_trg, aug_zeropad_mov, orig_dim
        
        else:
            return zeropad_trg, zeropad_mov, orig_dim
        

class Validation_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        
        

    def __len__(self):
        """Denotes the number training data
        :return: number of training data
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        """ Generates one sample of data """


        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        pair = self.pairs[ID]
        trg_mat = mosaic_to_mat(pydicom.dcmread(pair[1]))
        mov_mat = mosaic_to_mat(pydicom.dcmread(pair[0]))
        orig_dim = mov_mat.shape

        
        trg_mat = np.expand_dims(trg_mat, axis=0)
        mov_mat = np.expand_dims(mov_mat, axis=0)
        
        #try to use torchio for preprocessing
        mov_tensor = tio.ScalarImage(tensor=mov_mat)
        trg_tensor = tio.ScalarImage(tensor=trg_mat)
        
        #Min Max scaling
        scaler = tio.transforms.RescaleIntensity(out_min_max=(0, 1))
        scaled_mov = scaler(mov_tensor)
        scaled_trg = scaler(trg_tensor)
        
        #masking
        #the threshld is the median. It looks high but it works
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.5))
        m_scaled_mov = mask(scaled_mov)
        m_scaled_trg = mask(scaled_trg)
        
        #padding
        padding = tio.transforms.CropOrPad((128,128,128))
        zeropad_mov = padding(m_scaled_mov)
        zeropad_trg = padding(m_scaled_trg)
        
        
               

        return zeropad_trg,zeropad_mov, orig_dim

  
# # import matplotlib.pyplot as plt   
# # f,ax = plt.subplots(1,2)
# # ax[0].imshow((movable_mat-target_mat)[:,:,32], cmap ='Greys_r')
# # ax[1].imshow((scaled_mov_data-scaled_targ_data)[:,:,32], cmap = 'Greys_r')
# # ax[1,0].imshow((m_scaled_mov_data-m_scaled_targ_data)[:,:,32], cmap='Greys_r')
# # ax[1,1].imshow((z_scaled_mov_data-z_scaled_targ_data)[:,:,32], cmap='Greys_r')