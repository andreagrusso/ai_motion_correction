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
import pydicom, warnings
from scipy.stats import zscore
#from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

import tensorflow.keras.backend as K
import itertools





#%% Neurite functions (used by VoxelMorph)



    

