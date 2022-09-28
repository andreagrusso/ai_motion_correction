# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:29:11 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
from tensorflow.keras.utils import Sequence

import numpy as np
import pydicom, warnings
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

import tensorflow.keras.backend as K
import itertools


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
    return 0.1*det_loss + 0.1*ortho_loss #to_check


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


class AffineFlow(tf.keras.layers.Layer):
    
    def __init__(self,name='AfffineFlow',**kwargs):
        super(AffineFlow, self).__init__(name=name)
        
    def call(self, inputs):
        
        W, b = inputs
        # tf.print(W)
        # tf.print(b)
        output = self._affine_flow(W, b)

        return output



    def _affine_flow(self,W,b):
        
        """
        Function to transform the W rotation parameters and b displacement
        parameters in an affine flow to apply to the movable image
        """

        #print(W.shape,b.shape)
        #dims = [el.astype('float') for el in dims]
        len1, len2, len3 = [128.0, 128.0, 128.0]
        #W, b, input_pair = inputs
        #len1, len2, len3 = input_pair.shape.as_list()[1:4]
        W = tf.reshape(W, [-1, 3, 3]) 
        #b = tf.reshape(b, [-1, 3])# * self.flow_multiplier
        #b = tf.reshape(b, [-1, 1, 1, 1, 3])
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


#%% Interpolation

class Dense3DSpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, padding = False, name='padded_output',**kwargs):
        self.padding = padding
        #self.name='padded_output'
        super(Dense3DSpatialTransformer, self).__init__(name=name)

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of length 2 or 3. '
                            'First argument is the image, second is the offset field.')

        if len(input_shape[1]) != 5 or input_shape[1][4] != 3:
            raise Exception('Offset field must be one 5D tensor with 3 channels. '
                            'Got: ' + str(input_shape[1]))

        self.built = True
        
    def get_config(self):
        return {"padding": self.padding}

    def call(self, inputs):
        return self._transform(inputs[0], inputs[1][:, :, :, :, 1],
                               inputs[1][:, :, :, :, 0], inputs[1][:, :, :, :, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _transform(self, I, dx, dy, dz):
        #print('Input:',tf.reduce_mean(I))

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
        
        #print(tf.reduce_mean(im))
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
        #print('Ouptut:',tf.reduce_mean(output))
        return output


#%%old networks
def AffineNeuralNetworkModel():
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same', 
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv1')(input_pair)
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv2')(conv1)
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(conv2)
    #conv3_1
    conv3_1 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv3_1')(conv3)
    #conv4 16*16*16
    conv4 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv4')(conv3_1)
    #conv4_1
    conv4_1 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4_1')(conv4)
    #conv5  8 * 8 * 8
    conv5 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv5')(conv4_1)
    #conv5_1
    conv5_1 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv5_1')(conv5)
    #conv6  4 * 4 * 4
    conv6 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(conv5_1)       
    #conv6_1  4 * 4 * 4
    conv6_1 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv6_1')(conv6)
    # rotation matrix
    W = tf.keras.layers.Conv3D(filters = 9 , kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', use_bias=False, 
                               #activity_regularizer=my_regularizer, 
                               name='rot_matrix')(conv6_1)
    # translation params
    b = tf.keras.layers.Conv3D(filters = 3, kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', use_bias=False, 
                               name='trans_matrix')(conv6_1)
    #affine
    #affine = tf.keras.layers.Lambda(affine_flow, name='affine_matrix')([W, b])
    affine = AffineFlow(name='affine_matrix',trainable=True)([W, b])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output',trainable=True)([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])


def AffineNeuralNetworkModel_Dense():
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same', 
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv1')(input_pair)
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv2')(conv1)
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(conv2)
    #conv3_1
    conv3_1 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv3_1')(conv3)
    #conv4 16*16*16
    conv4 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv4')(conv3_1)
    #conv4_1
    conv4_1 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4_1')(conv4)
    #conv5  8 * 8 * 8
    conv5 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv5')(conv4_1)
    #conv5_1
    conv5_1 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                     strides=1, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv5_1')(conv5)
    #conv6  4 * 4 * 4
    conv6 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(conv5_1)       
    #conv6_1  4 * 4 * 4
    conv6_1 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                      strides=2, padding='same',
                                      activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                      name='conv6_1')(conv6)
    # rotation matrix
    W = tf.keras.layers.Dense(units = 9, use_bias=False, 
                               #activity_regularizer=my_regularizer, 
                               name='rot_matrix')(conv6_1)
    # translation params
    b = tf.keras.layers.Dense(units = 3, use_bias=False, 
                               name='trans_matrix')(conv6_1)
    #affine
    #affine = tf.keras.layers.Lambda(affine_flow, name='affine_matrix')([W, b])
    affine = AffineFlow(name='affine_matrix',trainable=True)([W, b])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output',trainable=True)([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])


def ABIR():
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, 
                                             name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 8, kernel_size = 3, 
                                   strides=2, padding='same', 
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv1')(input_pair)
    #norm1 = tf.keras.layers.BatchNormalization()(conv1)
    
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv2')(conv1)
    #norm2 = tf.keras.layers.BatchNormalization()(conv2)
    
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(conv2)
    #norm3 = tf.keras.layers.BatchNormalization()(conv3)
    
    #conv3_1
    conv4 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=2, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4')(conv3)
    #norm4 = tf.keras.layers.BatchNormalization()(conv4)    
    
    #conv4 16*16*16
    conv5 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv5')(conv4)
    #norm5 = tf.keras.layers.BatchNormalization()(conv5)    

    #conv5  8 * 8 * 8
    conv6 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(conv5)
    #norm6 = tf.keras.layers.BatchNormalization()(conv6)    

    # rotation matrix
    W = tf.keras.layers.Conv3D(filters = 9 , kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', use_bias=False, 
                               #activity_regularizer=my_regularizer, 
                               name='rot_matrix')(conv6)
    W_d = tf.keras.layers.Dropout(0.3)(W)
    # translation params
    b = tf.keras.layers.Conv3D(filters = 3, kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', use_bias=False, 
                               name='trans_matrix')(conv6)
    b_d = tf.keras.layers.Dropout(0.3)(b)

    #affine
    #affine = tf.keras.layers.Lambda(affine_flow, name='affine_matrix')([W, b])
    affine = AffineFlow(name='affine_matrix',trainable=True)([W_d, b_d])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output',trainable=True)([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])

def ABIR2():
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, 
                                             name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 8, kernel_size = 3, 
                                   strides=2, padding='same', 
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv1')(input_pair)
    #norm1 = tf.keras.layers.BatchNormalization()(conv1)
    
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv2')(conv1)
    #norm2 = tf.keras.layers.BatchNormalization()(conv2)
    
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(conv2)
    #norm3 = tf.keras.layers.BatchNormalization()(conv3)
    
    #conv3_1
    conv4 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=2, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4')(conv3)
    #norm4 = tf.keras.layers.BatchNormalization()(conv4)    
    
    #conv4 16*16*16
    conv5 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv5')(conv4)
    #norm5 = tf.keras.layers.BatchNormalization()(conv5)    

    #conv5  8 * 8 * 8
    conv6 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(conv5)
    #norm6 = tf.keras.layers.BatchNormalization()(conv6)
    conv7 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv7')(conv6)    

    # rotation matrix
    W = tf.keras.layers.Dense(units=9, activation='linear', 
                              use_bias=True, name='rot_matrix')(conv7)
    # W = tf.keras.layers.Conv3D(filters = 9 , kernel_size = 2, 
    #                            strides=1, padding='valid', 
    #                            activation='linear', use_bias=False, 
    #                            #activity_regularizer=my_regularizer, 
    #                            name='rot_matrix')(conv6)
    W_d = tf.keras.layers.Dropout(0.3)(W)
    # translation params
    b = tf.keras.layers.Dense(units=3, activation='linear', 
                              use_bias=True, name='trans_matrix')(conv7)
    # b = tf.keras.layers.Conv3D(filters = 3, kernel_size = 2, 
    #                            strides=1, padding='valid', 
    #                            activation='linear', use_bias=False, 
    #                            name='trans_matrix')(conv6)
    b_d = tf.keras.layers.Dropout(0.3)(b)

    #affine
    #affine = tf.keras.layers.Lambda(affine_flow, name='affine_matrix')([W, b])
    affine = AffineFlow(name='affine_matrix',trainable=True)([W_d, b_d])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output',trainable=True)([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])


def ABIR3():
    
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, 
                                             name = 'input_pair')([fixed_input,movable_input])
    
   
    #conv1 64*64*64 first conv block
    conv1 = tf.keras.layers.Conv3D(filters = 8, kernel_size = 3, 
                                   strides=2, padding='same', 
                                   activation=None, 
                                   name='conv1')(input_pair)
    activation1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv1)
    norm1 = tf.keras.layers.BatchNormalization()(activation1)
    
    #conv2 32*32*32 2nd conv block
    conv2 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=None, 
                                   name='conv2')(norm1)
    activation2 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv2)
    norm2 = tf.keras.layers.BatchNormalization()(activation2)

    
    #conv3 3rd conv block 
    conv3 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(norm2)
    activation3 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv3)
    norm3 = tf.keras.layers.BatchNormalization()(activation3)  
    
    #conv4 4th conv block
    conv4 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=2, padding='same',
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4')(norm3)
    activation4 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv4)
    norm4 = tf.keras.layers.BatchNormalization()(activation4)  
    
    
    #conv5 5th conv block
    conv5 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv5')(norm4)
    activation5 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv5)
    norm5 = tf.keras.layers.BatchNormalization()(activation5)
    
    
    #conv6 6th conv block
    conv6 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(norm5)
    activation6 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv6)
    norm6 = tf.keras.layers.BatchNormalization()(activation6)
    
    #conv7 7th conv block
    conv7 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv7')(norm6) 
    activation7 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv7)
    norm7 = tf.keras.layers.BatchNormalization()(activation7)
    

    # rotation matrix (dense + dropput)
    W = tf.keras.layers.Dense(units=9, activation='linear', 
                              use_bias=True, name='rot_matrix')(norm7)
    W_d = tf.keras.layers.Dropout(0.3)(W)
    
    
    # translation params (dense + dropout)
    b = tf.keras.layers.Dense(units=3, activation='linear', 
                              use_bias=True, name='trans_matrix')(norm7)
    b_d = tf.keras.layers.Dropout(0.3)(b)

    #affine
    affine = AffineFlow(name='affine_matrix',trainable=True)([W_d, b_d])
    
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output',trainable=True)([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])

