# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:47:15 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
from tensorflow.keras.utils import Sequence

import numpy as np
import pydicom
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

from other_functions import mosaic_to_mat


#%%
class DataGenerator_train(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,training_file, batch_size=8, dim=(128, 128,128),
                 n_channels=1, shuffle=True):
        """Initialization
        :param dict_file: Dictionary containing the path of the movable as keys
                            and the path of the target as value
        :param orig_dim: dimension of the original image
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.training_file = training_file
        self.indexes = training_file.index
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.training_file) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        #keys to access the movable and the target data
        list_IDs_temp  = [self.indexes[k] for k in indexes]

        # Generate data
        # the key is the path to the movable
        # the element of the key is the target
        # X is a concatenation of the two data after zero badding
        
        # call the function to generate the data
        # in our case X is composed of nifti pairs
        X = self._generate_data(list_IDs_temp)

        return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        
        self.indexes = np.arange(len(self.training_file))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_data(self, list_IDs_temp):
        """Generates data containing batch_size 3D images
        :param list_keys_temp: list of keys to access the dictionary
        :return: tupla of batches of 3D images (train + target) and orig dims (for later cropping)
        """
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels)) #training
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels)) #training
        Y = np.empty((self.batch_size, *self.dim, self.n_channels)) #target


        # Generate data
        for i, key in enumerate(list_IDs_temp):
            
            tmp_movable = pydicom.dcmread(self.training_file[key][0])
            tmp_target = pydicom.dcmread(self.training_file[key][1])
            
            #get the matrix
            movable_mat = mosaic_to_mat(tmp_movable)
            target_mat = mosaic_to_mat(tmp_target)
            
            #augment motion only when the movable is not the target itself
            if np.sum(movable_mat-target_mat) !=0:
                #augmetation only for the movable
                movable_mat = self._augment_motion(movable_mat)
                
            #scaling the data
            scaler = MinMaxScaler()
            scaled_mov_data = scaler.fit_transform(movable_mat.flatten().reshape(-1,1)).reshape(movable_mat.shape)
            scaled_targ_data = scaler.fit_transform(target_mat.flatten().reshape(-1,1)).reshape(target_mat.shape)
            # z_scaled_mov_data = zscore(movable_mat.flatten().reshape(-1,1)).reshape(movable_mat.shape)
            # z_scaled_targ_data = zscore(target_mat.flatten().reshape(-1,1)).reshape(target_mat.shape)
            #scaled_mov_data = movable_mat
            #scaled_targ_data = target_mat
            
            # #mask the data #median as threshold
            scaled_mov_data[scaled_mov_data<np.percentile(scaled_mov_data.flatten(),50)]=0
            scaled_targ_data[scaled_targ_data<np.percentile(scaled_targ_data.flatten(),50)]=0

            
           
            # we can assume that target and movable have the same dimensions
            pads = self._get_padding(movable_mat.shape)
            
            
            zeropad_movable = self._3Dpadding(scaled_mov_data,pads)
            zeropad_target = self._3Dpadding(scaled_targ_data,pads)
            
            X1[i,] = np.expand_dims(zeropad_target,axis=-1)
            X2[i,] = np.expand_dims(zeropad_movable,axis=-1)
            Y[i,] = np.expand_dims(zeropad_target,axis=-1)
            # print(X.shape)
            # print(Y.shape)


        return [X1,X2],Y
   

    def _augment_motion(self,data):
        """
        Function to increase randomly rotations and translation. Could be useful
        to increase the knowledge of the model to motion

        Parameters
        ----------
        data : NUmpy array
            Matrix to which the augmented motion needed to be applied.

        Returns
        -------
        aug_data: Numpy array
            Data with augmented motions.

        """
        
        
        affine_mat = np.eye(4)
        disp_vector = np.array([np.random.uniform(low=-5, high=5),
                                np.random.uniform(low=-5, high=5),
                                np.random.uniform(low=-5, high=5)])
        #print('Disp vector:',disp_vector)
        rot_angles = [np.random.uniform(low=-5, high=5),
                      np.random.uniform(low=-5, high=5),
                      np.random.uniform(low=-5, high=5)]
        #rotation are centered in the image (intrisic rotation)
        rot_mat = Rotation.from_euler('XYZ', rot_angles, degrees=True).as_matrix()
        affine_mat[:-1,:-1] = rot_mat
        #no scaling
        # affine_mat[0,0] = 1
        # affine_mat[1,1] = 1
        # affine_mat[2,2] = 1
        affine_mat[:-1,-1] = disp_vector
        
        new_data = affine_transform(data, affine_mat)
        
        return new_data
    
    
    def _get_padding(self,orig_input_dims):
        """Estimate the needed padding to have 128x128x128 data
        :orig_input_dims: list of data dimensions
        :return: amount of padding neeeded
        """ 
        axis_diff = (np.array(2*self.dim)-np.array(2*orig_input_dims))/2
        pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
                  for i in range(3)]
        return pads
    
    
    
    def _3Dpadding(self,data, pads):
        """Zeropad the 3D data
        :param data: 3D numpy array to be padded
        :return: padded 3D numpy array
        """
        return np.pad(data,pad_width=tuple(pads),constant_values=0)
        
    
    def _concatenate_data(self,target, movable):
        
        """

        Parameters
        ----------
        target : numpy array
            3D numpy array containing the target (zeropadded).
        movable : numpy array
           3D numpy array containing the target (zeropadded).

        Returns
        -------
        input_pair : numpy array
            4D numpy array containing target and movable (zeropadded).

        """
        
        target = np.expand_dims(target,axis=-1)
        movable = np.expand_dims(movable,axis=-1)
       
        input_pair = np.concatenate((target,movable),
                                    axis=-1)    
         
        return input_pair

#%% The two class differentiate only in the fact that data augmentation is
#not performed ion the validation data set. There is a better method to save
#lines but it is OK for now...
    
class DataGenerator_val(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,training_file, batch_size=8, dim=(128, 128,128),
                 n_channels=1, shuffle=True):
        """Initialization
        :param dict_file: Dictionary containing the path of the movable as keys
                            and the path of the target as value
        :param orig_dim: dimension of the original image
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.training_file = training_file
        self.indexes = training_file.index
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.training_file) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # self.indexes = np.arange(len(self.dict_file))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        #print('Self indexes:',self.indexes)

        # Find list of IDs
        # list of the dictionary keys
        #self.keys_list = list(self.dict_file.keys())
        #keys to access the movable and the target data
        list_IDs_temp  = [self.indexes[k] for k in indexes]

        # Generate data
        # the key is the path to the movable
        # the element of the key is the target
        # X is a concatenation of the two data after zero badding
        
        # call the function to generate the data
        # in our case X is composed of nifti pairs
        X = self._generate_data(list_IDs_temp)

        # if self.to_fit:
        #     y = self._generate_y(list_IDs_temp)
        #     return X, y
        # else:
        return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        
        self.indexes = np.arange(len(self.training_file))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_data(self, list_IDs_temp):
        """Generates data containing batch_size 3D images
        :param list_keys_temp: list of keys to access the dictionary
        :return: tupla of batches of 3D images (train + target) and orig dims (for later cropping)
        """
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels)) #training
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels)) #training
        Y = np.empty((self.batch_size, *self.dim, self.n_channels)) #target


        # Generate data
        for i, key in enumerate(list_IDs_temp):
            
            tmp_movable = pydicom.dcmread(self.training_file[key][0])
            tmp_target = pydicom.dcmread(self.training_file[key][1])
            
            #get the matrix
            movable_mat = mosaic_to_mat(tmp_movable)
            target_mat = mosaic_to_mat(tmp_target)
            

            #scaling the data
            scaler = MinMaxScaler()
            scaled_mov_data = scaler.fit_transform(movable_mat.flatten().reshape(-1,1)).reshape(movable_mat.shape)
            scaled_targ_data = scaler.fit_transform(target_mat.flatten().reshape(-1,1)).reshape(target_mat.shape)
            #scaled_mov_data = zscore(movable_mat.flatten().reshape(-1,1)).reshape(movable_mat.shape)
            #scaled_targ_data = zscore(target_mat.flatten().reshape(-1,1)).reshape(target_mat.shape)
            
            # #mask the data #median as threshold
            scaled_mov_data[scaled_mov_data<np.percentile(scaled_mov_data.flatten(),50)]=0
            scaled_targ_data[scaled_targ_data<np.percentile(scaled_targ_data.flatten(),50)]=0

            
           
            # we can assume that target and movable have the same dimensions
            pads = self._get_padding(movable_mat.shape)
            
            
            zeropad_movable = self._3Dpadding(scaled_mov_data,pads)
            zeropad_target = self._3Dpadding(scaled_targ_data,pads)
            
            X1[i,] = np.expand_dims(zeropad_target,axis=-1)
            X2[i,] = np.expand_dims(zeropad_movable,axis=-1)
            Y[i,] = np.expand_dims(zeropad_target,axis=-1)
            # print(X.shape)
            # print(Y.shape)


        return [X1,X2],Y
   
    
    
    def _get_padding(self,orig_input_dims):
        """Estimate the needed padding to have 128x128x128 data
        :orig_input_dims: list of data dimensions
        :return: amount of padding neeeded
        """ 
        axis_diff = (np.array(2*self.dim)-np.array(2*orig_input_dims))/2
        pads=[tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
                  for i in range(3)]
        return pads
    
    
    
    def _3Dpadding(self,data, pads):
        """Zeropad the 3D data
        :param data: 3D numpy array to be padded
        :return: padded 3D numpy array
        """
        return np.pad(data,pad_width=tuple(pads),constant_values=0)
        
    
    def _concatenate_data(self,target, movable):
        
        """

        Parameters
        ----------
        target : numpy array
            3D numpy array containing the target (zeropadded).
        movable : numpy array
           3D numpy array containing the target (zeropadded).

        Returns
        -------
        input_pair : numpy array
            4D numpy array containing target and movable (zeropadded).

        """
        
        target = np.expand_dims(target,axis=-1)
        movable = np.expand_dims(movable,axis=-1)
       
        input_pair = np.concatenate((target,movable),
                                    axis=-1)    
         
        return input_pair
    
    
    
# import matplotlib.pyplot as plt   
# f,ax = plt.subplots(1,2)
# ax[0].imshow((movable_mat-target_mat)[:,:,32], cmap ='Greys_r')
# ax[1].imshow((scaled_mov_data-scaled_targ_data)[:,:,32], cmap = 'Greys_r')
# ax[1,0].imshow((m_scaled_mov_data-m_scaled_targ_data)[:,:,32], cmap='Greys_r')
# ax[1,1].imshow((z_scaled_mov_data-z_scaled_targ_data)[:,:,32], cmap='Greys_r')