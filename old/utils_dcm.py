# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:36:43 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
from tensorflow.keras.utils import Sequence

import numpy as np
#from nilearn.masking import compute_epi_mask
#import matplotlib.pyplot as plt
#import nibabel as nb
import pydicom
from sklearn.preprocessing import MinMaxScaler


#%%


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


#funtion to estimate the novel grid of points
#def affine_flow(W, b, img):
def affine_flow(inputs):
    
    """
    Function to transform the W rotation parameters and b displacement
    parameters in an affine flow to apply to the movable image
    """
    W, b = inputs
    #print(W.shape,b.shape)
    #dims = [el.astype('float') for el in dims]
    len1, len2, len3 = [128.0, 128.0, 128.0]
    #W, b, input_pair = inputs
    #len1, len2, len3 = input_pair.shape.as_list()[1:4]
    W = tf.reshape(W, [-1, 3, 3]) 
    #b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
    #b = tf.reshape(b, [-1, 3])
    #print(W.shape,b.shape)
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])

    return (xr * wx + yr * wy) + (zr * wz + b)



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


#loss to ensure specific characteristic to the estimated matrix
def loss_for_matrix(target,W):
    
    """
    Function that combines the two losses applied to the rotation matrix
    One loss is on the orhtogonality to no scaling
    The other loss is on the determinant to ensure no flipping
    The two losses are summed

    Parameters
    ----------
    target : keras layer
        keras layer containing the target.
        Keras always pass the target and the prediction to a loss function, 
        although here is not useful
    W : keras layer
        Keras layer indicating the rotation parameters.

    Returns
    -------
    loss: float
        Sum of the two losses

    """
    
    #determinant should be close to 1
      #target that is an image is not used
      #print(A)
    flow_multiplier = 1 
    I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    W = tf.reshape(W, [-1, 3, 3]) * flow_multiplier
    A = W + I
  
    det = det3x3(A)
    det_loss = tf.nn.l2_loss(det - 1.0)
    # should be close to being orthogonal
  
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1 by minimizing
    # k1+1/k1+k2+1/k2+k3+1/k3.
    # to prevent NaN, minimize
    # k1+eps + (1+eps)^2/(k1+eps) + ...
    eps = 1e-5
    epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
    C = tf.matmul(A, A, True) + epsI
  
    s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    ortho_loss = tf.reduce_sum(ortho_loss)
  
          #0.1*det_loss + 0.1*ortho_loss
    #print('Mat loss:', det_loss + ortho_loss)
    return det_loss + ortho_loss #to_check


def loss_for_vector(target,b):
    """
    Fake function as on the vector displacement.
     We do not apply any loss 

    Parameters
    ----------
    target : keras layer
        keras layer containing the target.
        Keras always pass the target and the prediction to a loss function, 
        although here is not useful
    b : numpy array
        Displacement vecto.

    Returns
    -------
    float

    """

    return 0.0



#loss function to estimate similarity between the transformed img2 with img1
def similarity_loss(target,pred):
    """
    

    Parameters
    ----------
    target : keras layer
        Layer containing the target image.
    pred : keras layer
        Layer containing the movable image aligned.

    Returns
    -------
    raw_loss : float
        Output value of the cross-correlation.

    """

    #input is a set of two concatenated images
    #first the fixed image (true) and then the warped (predicted)
    fixed_img = target
    warped_img = pred
    #print('NaN in warped image:',tf.math.is_nan(warped_img))
    
    

    

    sizes = np.prod(fixed_img.shape.as_list()[1:])
    flatten1 = tf.reshape(fixed_img, [-1, sizes])
    flatten2 = tf.reshape(warped_img, [-1, sizes])

    mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
    #print('Mean1:',mean1)
    mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
    #print('Mean2:',mean2)
    var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
    #print('var1:',var1)
    var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
    #print('var2:',var2)    
    cov12 = tf.reduce_mean(
        (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
    #print('cov12:',cov12)
    
    pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    #print(pearson_r)
    raw_loss = 1 - pearson_r
    #print('Loss:',raw_loss)
    raw_loss = tf.reduce_sum(raw_loss)
    
    return raw_loss


#%% Interpolation

class Dense3DSpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, padding = False, **kwargs):
        self.padding = padding
        super(Dense3DSpatialTransformer, self).__init__(name='padded_output')

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of length 2 or 3. '
                            'First argument is the image, second is the offset field.')

        if len(input_shape[1]) != 5 or input_shape[1][4] != 3:
            raise Exception('Offset field must be one 5D tensor with 3 channels. '
                            'Got: ' + str(input_shape[1]))

        self.built = True

    def call(self, inputs):
        return self._transform(inputs[0], inputs[1][:, :, :, :, 1],
                               inputs[1][:, :, :, :, 0], inputs[1][:, :, :, :, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _transform(self, I, dx, dy, dz):

        batch_size = tf.shape(dx)[0]
        height = tf.shape(dx)[1]
        width = tf.shape(dx)[2]
        depth = tf.shape(dx)[3]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh, z_mesh = self._meshgrid(height, width, depth)
        x_mesh = tf.expand_dims(x_mesh, 0)
        y_mesh = tf.expand_dims(y_mesh, 0)
        z_mesh = tf.expand_dims(z_mesh, 0)

        x_mesh = tf.tile(x_mesh, [batch_size, 1, 1, 1])
        y_mesh = tf.tile(y_mesh, [batch_size, 1, 1, 1])
        z_mesh = tf.tile(z_mesh, [batch_size, 1, 1, 1])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, dtype='int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width, depth):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                                tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                                   tf.cast(height, tf.float32)-1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t = tf.tile(tf.expand_dims(x_t, 2), [1, 1, depth])
        y_t = tf.tile(tf.expand_dims(y_t, 2), [1, 1, depth])

        z_t = tf.linspace(0.0, tf.cast(depth, tf.float32)-1.0, depth)
        z_t = tf.expand_dims(tf.expand_dims(z_t, 0), 0)
        z_t = tf.tile(z_t, [height, width, 1])

        return x_t, y_t, z_t

    def _interpolate(self, im, x, y, z):
        if self.padding:
            im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "CONSTANT")

        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        depth = tf.shape(im)[3]
        channels = im.get_shape().as_list()[4]

        out_height = tf.shape(x)[1]
        out_width = tf.shape(x)[2]
        out_depth = tf.shape(x)[3]

        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        z = tf.reshape(z, [-1])

        padding_constant = 1 if self.padding else 0
        x = tf.cast(x, 'float32') + padding_constant
        y = tf.cast(y, 'float32') + padding_constant
        z = tf.cast(z, 'float32') + padding_constant

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        max_z = tf.cast(depth - 1, 'int32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self._repeat(tf.range(num_batch)*dim1,
                            out_height*out_width*out_depth)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        Ie = tf.gather(im_flat, idx_e)
        If = tf.gather(im_flat, idx_f)
        Ig = tf.gather(im_flat, idx_g)
        Ih = tf.gather(im_flat, idx_h)

        # and finally calculate interpolated values
        x1_f = tf.cast(x1, 'float32')
        y1_f = tf.cast(y1, 'float32')
        z1_f = tf.cast(z1, 'float32')

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = tf.expand_dims((dz * dx * dy), 1)
        wb = tf.expand_dims((dz * dx * (1-dy)), 1)
        wc = tf.expand_dims((dz * (1-dx) * dy), 1)
        wd = tf.expand_dims((dz * (1-dx) * (1-dy)), 1)
        we = tf.expand_dims(((1-dz) * dx * dy), 1)
        wf = tf.expand_dims(((1-dz) * dx * (1-dy)), 1)
        wg = tf.expand_dims(((1-dz) * (1-dx) * dy), 1)
        wh = tf.expand_dims(((1-dz) * (1-dx) * (1-dy)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,
                           we*Ie, wf*If, wg*Ig, wh*Ih])
        output = tf.reshape(output, tf.stack(
            [-1, out_height, out_width, out_depth, channels]))
        return output



#%% Generator

class DataGenerator(Sequence):
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
        X = np.empty((self.batch_size, *self.dim, 2*self.n_channels)) #training
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
            
            #mask the data
            # mov_data = scaled_mov_data#*mask_mov.get_fdata()
            # targ_data = scaled_targ_data#*mask_target.get_fdata()

           
            # we can assume that target and movable have the same dimensions
            pads = self._get_padding(movable_mat.shape)
            
            
            zeropad_movable = self._3Dpadding(scaled_mov_data,pads)
            zeropad_target = self._3Dpadding(scaled_targ_data,pads)
            
            X[i,] = self._concatenate_data(zeropad_target, zeropad_movable)
            Y[i,] = np.expand_dims(zeropad_target,axis=-1)
            # print(X.shape)
            # print(Y.shape)


        return X,Y
   
    
    
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
    

