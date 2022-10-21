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

from sklearn.preprocessing import MinMaxScaler


#%%


class Create_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

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
      

        trg_mat, world_affine = mosaic_to_mat(pair[1])#.astype('float32')
        mov_mat, world_affine = mosaic_to_mat(pair[0])#.astype('float32')
        #the two world affine should be the same (given that the two images have same dimensions)
        
        orig_dim = mov_mat.shape
        
        #get padding
        desidered_input_dims = [128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
        axis_diff = (np.array(desidered_input_dims)-np.array(2*orig_dim))/2
        pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
                  for i in range(3)]
        #zeropadding in numpy to preserve center of the 128**3 bounding box
        trg_mat = np.pad(trg_mat,pad_width=pads)
        mov_mat = np.pad(mov_mat,pad_width=pads)
        
        
        trg_mat = np.expand_dims(trg_mat, axis=0)
        mov_mat = np.expand_dims(mov_mat, axis=0)
        
        
        
        #try to use torchio for preprocessing
        mov_tensor = tio.ScalarImage(tensor=mov_mat)
        trg_tensor = tio.ScalarImage(tensor=trg_mat)
        
        #Min Max scaling (non zero voxels)
        scaler = tio.transforms.RescaleIntensity(out_min_max=(0, 1),
                                                 masking_method=lambda x: x > 0)
        scaled_mov = scaler(mov_tensor)
        scaled_trg = scaler(trg_tensor)
        
        #masking
        #the threshld is the median. It looks high but it works
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.90))
        m_scaled_mov = mask(scaled_mov)
        m_scaled_trg = mask(scaled_trg)
        
        #padding
        # padding = tio.transforms.CropOrPad((128,128,128))
        # zeropad_mov = padding(m_scaled_mov)
        # zeropad_trg = padding(m_scaled_trg)
            
        return m_scaled_trg, m_scaled_mov, orig_dim, world_affine






class Create_test_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

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
        
        
        trg_mat, world_affine = mosaic_to_mat(pair[1])#.astype('float32')
        mov_mat, world_affine = mosaic_to_mat(pair[0])#.astype('float32')
        orig_dim = mov_mat.shape
        
        #get padding
        desidered_input_dims = [128.0, 128.0, 128.0, 128.0, 128.0, 128.0]
        axis_diff = (np.array(desidered_input_dims)-np.array(2*orig_dim))/2
        pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
                  for i in range(3)]
        #zeropadding in numpy to preserve center of the 128**3 bounding box
        trg_mat = np.pad(trg_mat,pad_width=pads)
        mov_mat = np.pad(mov_mat,pad_width=pads)
        
        
        trg_mat = np.expand_dims(trg_mat, axis=0)
        mov_mat = np.expand_dims(mov_mat, axis=0)
        
        #try to use torchio for preprocessing
        mov_tensor = tio.ScalarImage(tensor=mov_mat)
        trg_tensor = tio.ScalarImage(tensor=trg_mat)
        
        #Min Max scaling (non zero voxels)
        scaler = tio.transforms.RescaleIntensity(out_min_max=(0, 1),
                                                 masking_method=lambda x: x > 0)
        scaled_mov = scaler(mov_tensor)
        scaled_trg = scaler(trg_tensor)
        
        
                
        # #ranfdom affine only for movable
        affine_aug = tio.transforms.RandomAffine(scales=0,
                                            degrees=[0,0,0],#sampling between -2,2
                                            translation=[1,3,5],#sampling between -2,2
                                            image_interpolation='bspline',
                                            center='origin')
            
        scaled_mov = affine_aug(scaled_mov)
        
        
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.90))
        m_scaled_mov = mask(scaled_mov)
        m_scaled_trg = mask(scaled_trg)
        


        
        return  m_scaled_trg, m_scaled_mov, orig_dim
        



# # import matplotlib.pyplot as plt   
# # f,ax = plt.subplots(1,2)
# # ax[0].imshow((movable_mat-target_mat)[:,:,32], cmap ='Greys_r')
# # ax[1].imshow((scaled_mov_data-scaled_targ_data)[:,:,32], cmap = 'Greys_r')
# # ax[1,0].imshow((m_scaled_mov_data-m_scaled_targ_data)[:,:,32], cmap='Greys_r')
# # ax[1,1].imshow((z_scaled_mov_data-z_scaled_targ_data)[:,:,32], cmap='Greys_r')