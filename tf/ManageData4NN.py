# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:26:56 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
#%%



class ManageData4NN():
    
    """ A class to manage the input and the output of the NN during
    real-time (and non real-time) motion correction. 
    """
    def __init__(self):
        
        self.NN_input_dim = [128,128,128] #fixed dimensions
        self.orig_dim = [] #this will be filled after the first 
        #it would be good to store the target as it is the same for the whole procedure
        self.target = np.empty((self.NN_input_dim))
        #padding needed to zeropadding and cropping
        self.pads = []
        #movable data in its original format (no padding/scaling/masking)
        self.orig_movable = []
        #target data in its original format (no padding/scaling/masking)
        self.orig_target = []
        #processed movable input (scaled and masked)
        self.movable = []
            
            

    def mosaic_to_mat(self, mosaic_dcm):
        
        """ Function to transform mosaic dicoms into 3D numpy array
        """
        
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
        self.orig_dim = [acq_matrix[0],acq_matrix[1], nr_slices]
        
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


    def mat_to_mosaic(self, mosaic_dcm, data_matrix, outdir, idx_dcm):
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

        mosaic_dcm.save_as(os.path.join(outdir,'test_'+str(idx_dcm)+'.dcm'))
        


    def masking(self,data):
        """
        

        Parameters
        ----------
        data : Numpy array
            3D matrix obtained from the transformation of the mosaic that
            need to be skull-stripped.

        Returns
        -------
        Masked data.

        """
        
        # scaler = MinMaxScaler()
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data.flatten().reshape(-1,1)).reshape(data.shape)
        #scaled_data = zscore(data.flatten().reshape(-1,1)).reshape(data.shape)
        #masking
        scaled_data[scaled_data<np.percentile(scaled_data.flatten(),20)]=0 
        

        return scaled_data

    def _3D_get_padding(self):
        """ Function to get the needed amount of zeropadding 
        """
        if not self.NN_input_dim:
            print('Bad class instance: no dimensions of NN inputs!')
            exit()

        if not self.orig_dim:
            print('Bad class instance: no dimensions of dicom input data! \n')
            print('Please call first generate_target method!')
            exit()
            
        axis_diff = (np.array(2*self.NN_input_dim)-np.array(2*self.orig_dim))/2
        
        pads = [tuple([int(np.ceil(axis_diff[i])),int(np.floor(axis_diff[i+3]))])
                  for i in range(3)]
        
        return pads 


    def _3Dpadding(self,data):
        """Zeropad the 3D data
        :param data: 3D numpy array to be padded
        :return: padded 3D numpy array
        """
        
        tmp_data = np.pad(data,pad_width=tuple(self.pads),constant_values=0)
        #add 4th dimension (i.e. the channel of the TF tensor)
        tmp_data = np.expand_dims(tmp_data,axis=0)
        
        return np.expand_dims(tmp_data, axis=-1)
        

    def _3Dcropping(self,data):
        """Crop the 3D data to have the original dimension
        :param data: 3D numpy array to be cropped
        :return: cropped 3D numpy array
        """
        
        x_tup,y_tup,z_tup = self.pads
        orig_x,orig_y,orig_z = self.orig_dim
    

        tmp_data = data[x_tup[0]:orig_x-x_tup[1],
                              y_tup[0]:orig_y-y_tup[1],
                              z_tup[0]:orig_z-z_tup[1]]
        
        return tmp_data
    
        
    def generate_target(self,mosaic_data):
        """ The aim of the function is to handle the target data
        """
        #transform mosaic data to 3D matrix
        data_matrix = self.mosaic_to_mat(mosaic_data)
        
        #store unprocessed target data (no masking, no scaling)
        self.orig_target = data_matrix
        
        #get the matrix dimensions of the data (for future cropping)
        self.orig_dim = data_matrix.shape
        
        #masking (TO IMPLEMENT)
        data_matrix = self.masking(data_matrix)
        
        #estimate the needed amount of padding. This would be the same for padding and cropping
        self.pads = self._3D_get_padding()
        
        #pad the input to have 1x128x28x128x1 dim
        self.target = self._3Dpadding(data_matrix)
        
    
    def get_target(self):
        """
        Return the padded target data
        """
        if len(self.target.shape)==3:
            print('The target si still empty! Please call first generate_target')
            exit()
            
        return self.target
    
    
    def get_orig_movable(self):
        """
        Return the padded target data
        """
        if len(self.orig_movable)==0:
            print('The movable si still empty! Please call first generate_movanble')
            exit()
            
        return self.orig_movable
    
    
    def generate_movable(self,mosaic_data):
        """
        

        Parameters
        ----------
        mosaic_data : Mosaic dicom data
            .

        Returns
        -------
        Numpy array
            3D array of the movable.

        """
        #transform mosaic data to 3D matrix
        data_matrix = self.mosaic_to_mat(mosaic_data)
        
        #store the movable data in original format
        self.orig_movable = data_matrix
                
        #masking (TO IMPLEMENT)
        data_matrix = self.masking(data_matrix)
        
        
        #estimate the needed amount of padding. This would be the same for padding and cropping
        self.pads = self._3D_get_padding()
        
        #pad the input to have 128x28x128x1 dim
        self.movable = self._3Dpadding(data_matrix)
    
    


    def process_otput(self,data,W,b):
        """
        Interpolate the zeropadded movable with the affine and crop it to the
        original dimensions        

        Parameters
        ----------
        data : Numpy array
            Output of the NN (image).
        W : Numpy array
            Output of the NN (rotation matrix).
        b: Numpy array
            Output of the NN (displacement vectors)

        Returns
        -------
        interp_data : numpy array
            Data interpolated with scipy.
        motion_params : numpy array
            Six motion parameters for that volume

        """
        affine = np.zeros((4,4))
        affine[:-1,:-1] = W.flatten().reshape(3,3)
        affine[:-1,-1] = b
        affine = affine + np.eye(4)
        #print(b)
        
        #could be that with scipy we don't need to use padded data
        interp_data = affine_transform(data,affine)
        
        motion_params = np.zeros((6))
        
        #translation params

        motion_params[:3] = b.reshape(1,-1)#np.reshape(b, [1,3]).reshape(-1,1)
        #rotation params (hopefully in radians)
        #source Rigid Body Registration John Ashburner & Karl J. Friston
        q5 = np.arcsin(affine[0,2]) #q5
        motion_params[4] = np.rad2deg(q5)
        
        q4 = np.arctan2(affine[1,2]/np.cos(q5),
                        affine[2,2]/np.cos(q5)) #q4
        motion_params[3] = np.rad2deg(q4)
        
        q6 = np.arctan2(affine[0,1]/np.cos(q5),
                        affine[0,0]/np.cos(q5)) #q6
        motion_params[5] = np.rad2deg(q6)
        
        #motion_params[3:] = np.rad2deg(motion_params[3:])
       
        return interp_data,motion_params
        
    

    def process_output_voxmorph(self,
                                affine_matrix, 
                                mosaic_dcm, 
                                outdir, 
                                idx_dcm):
        
        """
        Interpolate the zeropadded movable with the affine and crop it to the
        original dimensions        
    
        Parameters
        ----------
        data : Numpy array
            Output of the NN (image).
        affine_matrix : Numpy array 3x4
            Output of the NN (affine matrix).
    
    
        Returns
        -------
        interp_data : numpy array
            Data interpolated with scipy.
        motion_params : numpy array
            Six motion parameters for that volume
    
        """
        affine = np.eye(4)
        affine_matrix = np.squeeze(affine_matrix)
        #transform from degree to radians
        affine[:-1,:] = affine_matrix


        #could be that with scipy we don't need to use padded data
        interp_data = affine_transform(self.orig_movable,affine)
        
        self.mat_to_mosaic(mosaic_dcm, interp_data, outdir, idx_dcm)
        
        
        motion_params = np.zeros((6))
        
        #translation params
        #affine[:-1,:-1] = (affine_matrix[:,:-1]*np.pi)/180 #switch to radians
    
        motion_params[:3] = affine[:-1,-1]
        #rotation params (hopefully in radians)
        #source Rigid Body Registration John Ashburner & Karl J. Friston
        motion_params[4] = np.arcsin(affine[0,2]) #q5
        
        motion_params[3] = np.arctan2(affine[1,2]/np.cos(motion_params[4]),
                                      affine[2,2]/np.cos(motion_params[4])) #q4
        
        motion_params[5] = np.arctan2(affine[0,1]/np.cos(motion_params[4]),
                                      affine[0,0]/np.cos(motion_params[4])) #q6
        
        motion_params[3:] = np.rad2deg(motion_params[3:])
        
        return interp_data,motion_params        
            
    