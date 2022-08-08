# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:08:30 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob
import pickle
import numpy as np 

#%%
datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction/train_dcm'
colabdir = '/content/drive/MyDrive/Colab Notebooks/ai_motion/data/train_dcm'


all_files = []
training_files_colab = []

#multiple subjects
subs = glob.glob(os.path.join(datadir,'P*'))

for sub in subs:
    
    #each sub contains three different experiments
    experiments = glob.glob(os.path.join(sub,'*'))
    
    for exp in experiments:
        
        #get dcms files
        dcms = sorted(glob.glob(os.path.join(exp,'DCM','*.dcm')))
        tmp_list = [[dcms[i],dcms[0]] for i in range(1,len(dcms))]
    
        all_files += tmp_list



#manipuate all the files to prepare for different training
train_lim = int(np.floor((len(all_files)/100)*70))
val_lim = int(np.floor((len(all_files)/100)*30))





#shuffle
np.random.shuffle(all_files)

training_files = all_files[:train_lim]
validation_files = all_files[train_lim:train_lim+val_lim]
#testing_files = all_files[train_lim+val_lim:]

pickle.dump(training_files,open(os.path.join(datadir,'dcm_training_set.pkl'),'wb'))
pickle.dump(validation_files,open(os.path.join(datadir,'dcm_validation_set.pkl'),'wb'))
#

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


    
colab_validation = []
for item in validation_files:
    #replace the parent directory
    tmp_item0 = item[0].replace(datadir,colabdir)
    tmp_item1 = item[1].replace(datadir,colabdir)
    #replce '\\" with "/"
    colab_validation.append([tmp_item0.replace("\\","/"),
                           tmp_item1.replace("\\","/")]) 

pickle.dump(colab_validation,open(os.path.join(datadir,'dcm_colab_validation_set.pkl'),'wb'))

 
#%% create the same file for the testing data

sub_test = glob.glob(os.path.join(datadir,'test_P*'))
all_testing_files = []

for sub in sub_test:
    
    #each sub contains three different experiments
    experiments = glob.glob(os.path.join(sub,'*'))
    
    for exp in experiments:
        
        #get dcms files
        dcms = sorted(glob.glob(os.path.join(exp,'DCM','*.dcm')))
        tmp_list = [[dcms[i],dcms[0]] for i in range(1,len(dcms))]
    
        all_testing_files += tmp_list

pickle.dump(all_testing_files,open(os.path.join(datadir,'dcm_testing_set.pkl'),'wb'))
   
colab_testing = []
for item in all_testing_files:
    #replace the parent directory
    tmp_item0 = item[0].replace(datadir,colabdir)
    tmp_item1 = item[1].replace(datadir,colabdir)
    #replce '\\" with "/"
    colab_testing.append([tmp_item0.replace("\\","/"),
                            tmp_item1.replace("\\","/")])
    

pickle.dump(colab_testing,open(os.path.join(datadir,'dcm_colab_testing_set.pkl'),'wb'))
       

        


