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
      

        trg_mat = mosaic_to_mat(pair[1]).astype('float32')
        mov_mat = mosaic_to_mat(pair[0]).astype('float32')
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
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.6))
        m_scaled_mov = mask(scaled_mov)
        m_scaled_trg = mask(scaled_trg)
        
        #padding
        padding = tio.transforms.CropOrPad((128,128,128))
        zeropad_mov = padding(m_scaled_mov)
        zeropad_trg = padding(m_scaled_trg)
            
        return zeropad_trg, zeropad_mov, orig_dim






# class Training_dataset(torch.utils.data.Dataset):
    
#     """Generates data for Pytorch
#     Sequence based data generator. Suitable for building data generator for training and prediction.
#     """
    
#     def __init__(self, pairs_list,dims):
        
#         self.pairs = pairs_list #each pair contains movable and corresponding target
#         self.list_IDs = range(len(pairs_list)) #indices of the list
#         self.dims = dims #list of the desidered dims for zero padding
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

#     def __len__(self):
#         """Denotes the number training data
#         :return: number of training data
#         """
#         return len(self.list_IDs)

#     def __getitem__(self, index):
        
#         """ Generates one sample of data """


#         # Select sample
#         ID = self.list_IDs[index]


#         # Load data and get label
#         pair = self.pairs[ID]
#         trg_mat = mosaic_to_mat(pydicom.dcmread(pair[1]))
#         mov_mat = mosaic_to_mat(pydicom.dcmread(pair[0]))
        
#         # trg_mat = np.swapaxes(trg_mat, 0, 2)
#         # mov_mat = np.swapaxes(mov_mat, 0, 2)

#         orig_dim = mov_mat.shape
        
        
#         #scaling in sklearn
#         scaler = MinMaxScaler()
        
#         scaled_trg_mat = scaler.fit_transform(trg_mat.flatten().reshape(-1,1))
#         scaled_trg_mat = scaled_trg_mat.reshape(trg_mat.shape)
        
#         scaled_mov_mat = scaler.fit_transform(mov_mat.flatten().reshape(-1,1))
#         scaled_mov_mat = scaled_mov_mat.reshape(mov_mat.shape)
        
        
#         #masking
#         mask = lambda x: x >= np.quantile(x,0.6)
#         masked_mov_mat = mask(scaled_mov_mat)*scaled_mov_mat
#         masked_trg_mat = mask(scaled_trg_mat)*scaled_trg_mat
        
        
#         #zeropadding
#         axis_diff = (np.array(2*[128,128,128])-np.array(2*orig_dim))/2
#         pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
#                   for i in range(3)]
        
        
#         zeropad_mov_mat = np.pad(masked_mov_mat, pad_width=pads)
#         zeropad_trg_mat = np.pad(masked_trg_mat, pad_width=pads)
        
        
#         zeropad_mov_mat = np.expand_dims(zeropad_mov_mat, axis=0)
#         zeropad_trg_mat = np.expand_dims(zeropad_trg_mat, axis=0)
              
        
#         #try to use torchio for preprocessing
#         mov_tensor = tio.ScalarImage(tensor=zeropad_mov_mat)
#         trg_tensor = tio.ScalarImage(tensor=zeropad_trg_mat)
        
        
#         # #bring everything in RAS space
#         # orientation = tio.transforms.ToCanonical()
#         # mov_tensor = orientation(mov_tensor)
#         # trg_tensor = orientation(trg_tensor)
              
#         if np.sum(mov_mat-trg_mat)!=0:
                
#             # #ranfdom affine -5:5 degree and mm
#             affine_aug = tio.transforms.RandomAffine(scales=0,
#                                                 degrees=[2,2,2],#sampling between -2,2
#                                                 translation=[2,2,2],#sampling between -2,2
#                                                 image_interpolation='bspline',
#                                                 center='origin')
                
#             mov_tensor = affine_aug(mov_tensor)
        


#         # #remove negative outside brain caused by interp of rotation
#         only_pos_mask = tio.transforms.Mask(masking_method=lambda x: x >=1e-6)
#         mov_tensor = only_pos_mask(mov_tensor)
#         trg_tensor = only_pos_mask(trg_tensor)

        
 


        
#         return trg_tensor, mov_tensor, orig_dim
        

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
        trg_mat = mosaic_to_mat(pair[1])
        mov_mat = mosaic_to_mat(pair[0])
        
        # trg_mat = np.swapaxes(trg_mat, 0, 2)
        # mov_mat = np.swapaxes(mov_mat, 0, 2)
        orig_dim = mov_mat.shape
        
        mov_tensor = tio.ScalarImage(tensor=mov_mat)
        trg_tensor = tio.ScalarImage(tensor=trg_mat)
        
        scaler = tio.transforms.RescaleIntensity(out_min_max=(0,1))
        trg_tensor = scaler(trg_mat)
        mov_tensor = scaler(mov_mat)
        
        mask = tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.6))
        trg_tensor = mask(trg_tensor)
        mov_tensor = mask(mov_tensor)
        
        padding = tio.transforms.CropOrPad((128,128,128))
        trg_tensor = padding(trg_tensor)
        mov_tensor = padding(mov_tensor)
        
        
        return trg_tensor, mov_tensor, orig_dim
  


# # import matplotlib.pyplot as plt   
# # f,ax = plt.subplots(1,2)
# # ax[0].imshow((movable_mat-target_mat)[:,:,32], cmap ='Greys_r')
# # ax[1].imshow((scaled_mov_data-scaled_targ_data)[:,:,32], cmap = 'Greys_r')
# # ax[1,0].imshow((m_scaled_mov_data-m_scaled_targ_data)[:,:,32], cmap='Greys_r')
# # ax[1,1].imshow((z_scaled_mov_data-z_scaled_targ_data)[:,:,32], cmap='Greys_r')