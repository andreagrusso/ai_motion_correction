# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:01:55 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import tensorflow as tf
#from tensorflow.keras.utils import plot_model

from utils_dcm_2inputs import  AffineFlow, Dense3DSpatialTransformer

#%% define a function to instiate the model

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
    affine = AffineFlow(name='affine_matrix')([W, b])
    #last layer for interpolation. Bring the moving image in the fixed image space
    padded_output = Dense3DSpatialTransformer(name='padded_output')([movable_input,affine])

    return tf.keras.Model(inputs=[fixed_input,movable_input], outputs=[padded_output,W,b])


