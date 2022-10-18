# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob, pickle
import numpy as np


#%%
datadir = '/home/ubuntu22/Desktop/ai_mc/dcm'
colabdir = '/content/drive/MyDrive/ColabNotebooks/ai_motion/data/dcm'    

#%%  Training files

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
        
        tmp_list = [[dcms[i],dcms[0]] for i in range(0,len(dcms))]
    
        training_files += tmp_list

#shuffle
np.random.shuffle(training_files)


pickle.dump(training_files,open(os.path.join(datadir,'dcm_training_set.pkl'),'wb'))

#prepare the same data for google colab
colab_training = []
for item in training_files:
    #replace the parent directory
    tmp_item0 = item[0].replace(datadir,colabdir)
    tmp_item1 = item[1].replace(datadir,colabdir)
    #replce '\\" with "/"
    colab_training.append([tmp_item0.replace("\\","/"),
                           tmp_item1.replace("\\","/")])

pickle.dump(colab_training,open(os.path.join(datadir,'dcm_colab_training_set.pkl'),'wb'))

#%% Validation files

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
        
        tmp_list = [[dcms[i],dcms[0]] for i in range(0,len(dcms))]
    
        validation_files += tmp_list

#shuffle
np.random.shuffle(validation_files)
pickle.dump(validation_files,open(os.path.join(datadir,'dcm_validation_set.pkl'),'wb'))

    
colab_validation = []
for item in validation_files:
    #replace the parent directory
    tmp_item0 = item[0].replace(datadir,colabdir)
    tmp_item1 = item[1].replace(datadir,colabdir)
    #replce '\\" with "/"
    colab_validation.append([tmp_item0.replace("\\","/"),
                           tmp_item1.replace("\\","/")]) 

pickle.dump(colab_validation,open(os.path.join(datadir,'dcm_colab_validation_set.pkl'),'wb'))

 
#%% Testing files

subs = glob.glob(os.path.join(datadir,'test','sub*'))


for sub in subs:
    
    #each sub contains three different experiments
    experiments = glob.glob(os.path.join(sub,'run*'))
    
    sub_name = os.path.basename(sub)
    
    for exp in experiments:
        
        exp_name = os.path.basename(exp)
        
        #get dcms files
        dcms = sorted(glob.glob(os.path.join(exp,'*.dcm')))        
        
        tmp_list = [[dcms[i],dcms[0]] for i in range(0,len(dcms))]
        
        dump_name = sub_name + '_' + exp_name

        pickle.dump(tmp_list,open(os.path.join(datadir,'dcm_'+dump_name+'_testing_set.pkl'),'wb'))
            
        #prepare data also for colab    
        colab_testing = []
        for item in tmp_list:
            #replace the parent directory
            tmp_item0 = item[0].replace(datadir,colabdir)
            tmp_item1 = item[1].replace(datadir,colabdir)
            #replce '\\" with "/"
            colab_testing.append([tmp_item0.replace("\\","/"),
                                    tmp_item1.replace("\\","/")])
            
        
        pickle.dump(colab_testing,open(os.path.join(datadir,'dcm_colab_'+dump_name+'_testing_set.pkl'),'wb'))    
        
        
        
        


       
