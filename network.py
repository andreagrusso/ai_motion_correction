# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:36:55 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
from external_layers import  ParamsToAffineMatrix, SpatialTransformer, AffineFlow, Dense3DSpatialTransformer
from losses import regularizer_rot_matrix

#%% define a function to instiate the model

init_w = tf.keras.initializers.HeUniform()

def VoxMorphAffine():
    
    """
    Using VoxelMorph layers https://github.com/voxelmorph/voxelmorph and the simple
    workflow described in https://github.com/voxelmorph/voxelmorph/issues/371
    """

    
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, 
                                             name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 8, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv1')(input_pair)
    #norm1 = tf.keras.layers.BatchNormalization()(conv1)
    
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv2')(conv1)
    
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv3')(conv2)
    
    #conv3_1
    conv4 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=2, padding='same',
                                     use_bias = True,
                                     kernel_initializer=init_w,
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                     trainable=True,
                                     name='conv4')(conv3)
    
    #conv4 16*16*16
    conv5 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv5')(conv4)

    #conv5  8 * 8 * 8
    conv6 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv6')(conv5)
    
    conv7 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias = True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   trainable=True,
                                   name='conv7')(conv6)    

    # rotation params
    W = tf.keras.layers.Dense(units=9, 
                              activation='linear', 
                              trainable=True,
                              #use_bias = False,
                              name='rot_matrix')(conv7)

    #W_d = tf.keras.layers.Dropout(0.3)(W)
    
    # translation params
    b = tf.keras.layers.Dense(units=3, 
                              activation='linear',
                              trainable=True,
                              #use_bias = False,
                              name='trans_matrix')(conv7)
    #b_d = tf.keras.layers.Dropout(0.3)(b)

    
    #get the affine matrix with this layer
    affine_matrix = ParamsToAffineMatrix(name = 'affine_matrix')([b,W])
    #by default rotations are in degree
    #print(concat_params.shape)
    #print(movable_input.shape)
    #transform the affine in a dense shift
    #affine_flow = AffineToDenseShift(name='affine_flow')(affine_matrix)
    #this is done inside the SpatialTransformer layer
    
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = SpatialTransformer(name="padded_output")([movable_input,affine_matrix])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,affine_matrix])






def AffineNeuralNetworkModel():
    #define the two input tensors
    fixed_input = tf.keras.layers.Input((128,128,128,1),name='fixed_input')
    movable_input =  tf.keras.layers.Input((128,128,128,1),name='movable_input')
    
    input_pair = tf.keras.layers.Concatenate(axis=-1, name = 'input_pair')([fixed_input,movable_input])
    
    #conv1 64*64*64
    conv1 = tf.keras.layers.Conv3D(filters = 16, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv1')(input_pair)
    #conv2 32*32*32
    conv2 = tf.keras.layers.Conv3D(filters = 32, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv2')(conv1)
    #conv3
    conv3 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv3')(conv2)
    #conv3_1
    conv3_1 = tf.keras.layers.Conv3D(filters = 64, kernel_size = 3, 
                                     strides=1, padding='same',
                                     use_bias=True,
                                   kernel_initializer=init_w,
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv3_1')(conv3)
    #conv4 16*16*16
    conv4 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                                   name='conv4')(conv3_1)
    #conv4_1
    conv4_1 = tf.keras.layers.Conv3D(filters = 128, kernel_size = 3, 
                                     strides=1, padding='same',
                                   kernel_initializer=init_w,
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv4_1')(conv4)
    #conv5  8 * 8 * 8
    conv5 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv5')(conv4_1)
    #conv5_1
    conv5_1 = tf.keras.layers.Conv3D(filters = 256, kernel_size = 3, 
                                     strides=1, padding='same',
                                     use_bias=True,
                                   kernel_initializer=init_w,
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv5_1')(conv5)
    #conv6  4 * 4 * 4
    conv6 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                   strides=2, padding='same',
                                   use_bias=True,
                                   kernel_initializer=init_w,
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                   name='conv6')(conv5_1)       
    #conv6_1  4 * 4 * 4
    conv6_1 = tf.keras.layers.Conv3D(filters = 512, kernel_size = 3, 
                                     strides=1, padding='same',
                                     use_bias=True, 
                                   kernel_initializer=init_w,
                                     activation=tf.keras.layers.LeakyReLU(alpha=0.01), 
                                     name='conv6_1')(conv6)
    # rotation matrix
    W = tf.keras.layers.Conv3D(filters = 9 , kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', 
                               use_bias=False, 
                               #activity_regularizer=my_regularizer, 
                               name='rot_matrix')(conv6_1)
    # translation params
    b = tf.keras.layers.Conv3D(filters = 3, kernel_size = 2, 
                               strides=1, padding='valid', 
                               activation='linear', 
                               use_bias=False, 
                               name='trans_matrix')(conv6_1)
    #affine
    affine = AffineFlow(name='affine_matrix')([W, b])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output')([movable_input,affine])
    # padded_output = SpatialTransformer(name='padded_output')([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])