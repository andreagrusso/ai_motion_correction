# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob, pickle
import numpy as np


def create_pairs(datadir):
    
#############TRANING####################

    training_files = []

    #multiple subjects
    subs = glob.glob(os.path.join(datadir,'train','sub*'))


    #first load, create time series, zscore, and re-write dcm
    for sub in subs:
        
        
        #each sub contains three different experiments
        experiments = glob.glob(os.path.join(sub,'*'))
        
        for exp in experiments:
            

            #get dcms files
            dcms = sorted(glob.glob(os.path.join(exp,'*.dcm')))   
            
            #random index for random target
            target_idx = np.random.randint(0,len(dcms))

            
            tmp_list = [[dcms[i],dcms[target_idx]] 
                        for i in range(0,len(dcms))]
        
            training_files += tmp_list

    #shuffle
    np.random.shuffle(training_files)
    
##########VALIDATION#################################### 
   
    validation_files = []

    #multiple subjects
    subs = glob.glob(os.path.join(datadir,'validation','sub*'))


    #first load, create time series, zscore, and re-write dcm
    for sub in subs:
        
        #each sub contains three different experiments
        experiments = glob.glob(os.path.join(sub,'*'))
        
        for exp in experiments:

            #get dcms files
            dcms = sorted(glob.glob(os.path.join(exp,'*.dcm'))) 
            
            #random index for random target
            target_idx = np.random.randint(0,len(dcms))
            
            tmp_list = [[dcms[i],dcms[target_idx]] 
                        for i in range(0,len(dcms))]
        
            validation_files += tmp_list

    #shuffle
    np.random.shuffle(validation_files)
    
    return training_files, validation_files
