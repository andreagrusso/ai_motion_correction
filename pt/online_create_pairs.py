# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob, pickle
import numpy as np


def create_pairs(datadir,nr_of_pairs):
    
    
    
#############TRANING####################

    training_files = []
    training_labels = []

    #multiple subjects
    subs = glob.glob(os.path.join(datadir,'train','sub*'))

    sub_label = 0
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
            training_labels += len(tmp_list)*[sub_label]
            sub_label += 1


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
    
    if nr_of_pairs==-1:
        
        #estimate the proportion of subject's data 
        class_sample_count = np.array([len(np.where(training_labels == lab)[0]) 
                                       for lab in np.unique(training_labels)])
        class_percentage = class_sample_count/np.sum(class_sample_count)
        
        return training_files, validation_files, class_percentage
    
    else:
        #estimate the proportion of subject's data 
        class_sample_count = np.array([len(np.where(training_labels[:nr_of_pairs] == lab)[0]) 
                                       for lab in np.unique(training_labels[:nr_of_pairs])])
        class_percentage = class_sample_count/np.sum(class_sample_count)
        
        return training_files[:nr_of_pairs], validation_files[:int(nr_of_pairs/2)], class_percentage



#%% testing pairs

def create_pairs_for_testing(datadir):
    
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
            target_idx = 0

            
            tmp_list = 100*[[dcms[1],dcms[target_idx]]]
            
            #return only the first vol as target and the second vol as movable
            #the second volume has usually low movements respect to the first
            #then it would be rotated with "known" affine
        
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
            target_idx = 0
            
            
            tmp_list = 100*[[dcms[1],dcms[target_idx]]]
        
            validation_files += tmp_list

    #shuffle
    np.random.shuffle(validation_files)
    
    
    ##########TESTING#################################### 
       
    testing_files = []

    #multiple subjects
    subs = glob.glob(os.path.join(datadir,'testing','sub*'))


    #first load, create time series, zscore, and re-write dcm
    for sub in subs:
        
        #each sub contains three different experiments
        experiments = glob.glob(os.path.join(sub,'*'))
        
        for exp in experiments:

            #get dcms files
            dcms = sorted(glob.glob(os.path.join(exp,'*.dcm'))) 
            
            #random index for random target
            target_idx = 0
            
            
            tmp_list = 100*[[dcms[1],dcms[target_idx]]]
        
            testing_files += tmp_list

    #shuffle
    np.random.shuffle(testing_files)
    
    return training_files, validation_files, testing_files