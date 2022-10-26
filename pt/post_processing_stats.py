# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:30:12 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import os, glob, time, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nibabel as nb
#import hiddenlayer as hl


from network import AffineNet
from generators import Create_dataset
from util_functions import output_processing, moco_movie

#%% Data

datadir = 'C:/Users/NeuroIm/Documents/data/ai_motion_correction'
model_dir = 'test'
sub = 'sub00_run1'
outdir = os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'ai')


#%% Load and plot the model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AffineNet()
model.load_state_dict(torch.load(os.path.join(datadir,'preliminary_nn_results',model_dir,'AffineNet_saved_model.pth')))
model.eval()


#hl.build_graph(model,(torch.zeros([1, 1, 128, 128, 128]),torch.zeros([1, 1, 128, 128, 128])))

model = model.to(device)

#%% Run the model

testing_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_'+sub+'_testing_set.pkl'),'rb'))
testing_set = Create_dataset(testing_files, (128,128,128))
testing_generator = torch.utils.data.DataLoader(testing_set, shuffle=False)

motion_params = np.empty((len(testing_set), 6))
aligned_data = []
mse = []
i=0
timing = []

diff_vol_pre = []
diff_vol_post = []

aligned_4D = []


for fixed_test, movable_test, orig_dim, world_affine in testing_generator: #just testing
    start = time.time()
    outputs = model(fixed_test['data'].type(torch.FloatTensor).to(device),
                    movable_test['data'].type(torch.FloatTensor).to(device))
    
    crop_vol, curr_motion, mse_post, mse_pre = output_processing(fixed_test['data'],
                                                         movable_test['data'],
                                                         outputs, 
                                                         orig_dim,
                                                         world_affine)
    aligned_4D.append(crop_vol)
    
    timing.append(time.time()-start)
    aligned_data.append(crop_vol)
    motion_params[i,:] = curr_motion
    mse += [mse_pre] + [mse_post]
    
    diff_vol_pre.append(abs(np.squeeze(movable_test['data'].cpu().detach().numpy()-fixed_test['data'].cpu().detach().numpy())))
    diff_vol_post.append(abs(np.squeeze(outputs[0].cpu().detach().numpy()-fixed_test['data'].cpu().detach().numpy())))
                        
    i +=1

#%% Model metrics

training_loss = pd.read_csv(os.path.join(datadir,'preliminary_nn_results',model_dir,'AffineNet_training_loss.csv'))
validation_loss = pd.read_csv(os.path.join(datadir,'preliminary_nn_results',model_dir,'AffineNet_validation_loss.csv'))

loss_df = pd.DataFrame([], columns=['Loss', 'Log loss','Epoch', 'Section'])
loss_df['Loss'] = list(np.vstack((training_loss['Loss'].values.reshape(-1,1),
                             validation_loss['Loss'].values.reshape(-1,1))))
loss_df['Log loss'] = list(np.vstack((np.log10(training_loss['Loss']).values.reshape(-1,1),
                             np.log10(validation_loss['Loss']).values.reshape(-1,1))))
loss_df['Epoch'] = list(np.vstack((training_loss['Epoch'].values.reshape(-1,1),
                             validation_loss['Epoch'].values.reshape(-1,1))))
loss_df['Section'] = ['Training' for i in range(len(training_loss))]+['Validation' for i in range(len(validation_loss))]


loss_df = loss_df.astype({"Epoch": int, "Loss": float, "Log loss": float})

sns.set_theme(style="darkgrid")

ax,f = plt.subplots(figsize=(15,10))
f=sns.lineplot(data = loss_df, 
             x='Epoch',y='Log loss', 
             hue='Section')
f.set_xticks(range(20))
plt.tight_layout()
plt.savefig(os.path.join(datadir,'preliminary_nn_results',model_dir,'model_losses.svg'), dpi=300)

#%% Subject's metric

mse_df = pd.DataFrame([], columns=['MSE', 'Vol','Section'])
mse_df['MSE'] = mse
mse_df['Section'] = len(motion_params)*['Pre','Post']
mse_df['Vol'] = list(np.arange(0, len(motion_params))) + list(np.arange(0, len(motion_params)))

mse_df = mse_df.astype({"Vol": int, "MSE": float, "Section": str})

sns.set_theme(style="darkgrid")
f, ax = plt.subplots(1,2, figsize=(15,10))

sns.lineplot(data = mse_df, 
             x='Vol',y='MSE', 
             hue='Section', ax=ax[0])
ax[1].plot(timing)
#f.set_xticks(range(len(pre_mse_sub00)))
plt.tight_layout()
plt.savefig(os.path.join(outdir,sub+'_mse_and_time.svg'), dpi=300)

#%%
spm_motion_file = glob.glob(os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'spm','rp*.txt'))
spm_motion = np.loadtxt(spm_motion_file[0])
spm_motion[:,3:] = np.rad2deg(spm_motion[:,3:])

df_motion = pd.DataFrame(np.vstack((motion_params,spm_motion)),
                         columns=['X trans', 'Y trans', 'Z trans',
                                  'X rot', 'Y rot', 'Z rot'])
df_motion['Algorithm'] = ['AI' for i in range(len(motion_params))]+['SPM' for i in range(len(motion_params))]
df_motion['Vol'] = list(np.arange(0, len(motion_params))) + list(np.arange(0, len(motion_params)))


f, ax = plt.subplots(2,3, figsize=(15,10))
sns.set_theme(style="darkgrid")

col =0
for i in range(2):
    for j in range(3):
        
        th = np.max(np.abs(df_motion[df_motion.columns[col]]))
        
        sns.lineplot(data = df_motion, 
                     x='Vol',y=df_motion.columns[col], 
                     hue='Algorithm',
                     ax = ax[i,j])
        ax[i,j].set(ylim=(-th-0.01,th+0.01))
        plt.title(df_motion.columns[col])
        col +=1
plt.tight_layout()
plt.savefig(os.path.join(outdir,sub+'_axis_motion.svg'), dpi=300)


#%% plot vol difference
import matplotlib as mpl
sns.set_theme(style="darkgrid")

f, ax = plt.subplots(2,4, figsize=(15,20))

index_plot = [np.random.randint(0, len(motion_params)) for i in range(4)]

index_plot.sort()   
for idx, vol in enumerate(index_plot):
    
    
    global_min= min([np.min(diff_vol_pre[vol][:,:,64]),
                     np.min(diff_vol_post[vol][:,:,64])])
    
    global_max = max([np.max(diff_vol_pre[vol][:,:,64]),
                     np.max(diff_vol_post[vol][:,:,64])])
        
    ax[0,idx].imshow((diff_vol_pre[vol][:,:,64]).T, cmap='Reds')
    ax[0,idx].set_title('Before alignment: vol n '+str(vol))
    norm = mpl.colors.Normalize(vmin=global_min,
                                vmax=global_max)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'), 
                 ax=ax[0,idx],
                 shrink=0.3)  
    
    ax[1,idx].imshow((diff_vol_post[vol][:,:,64]).T, cmap='Reds')
    ax[1,idx].set_title('After alignment: vol n '+str(vol))
    norm = mpl.colors.Normalize(vmin=global_min,
                                vmax=global_max)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'), 
                 ax=ax[1,idx],
                 shrink=0.3)
    plt.tight_layout()
    plt.suptitle('Difference between movable and fixed voulumes before and after alignment')
    
plt.savefig(os.path.join(outdir,sub+'_volume_differences.svg'), dpi=300)

#%% MoCo movie

aligned_4D = np.array(aligned_4D)
aligned_4D = np.moveaxis(aligned_4D,[0,1,2,3],[3,0,1,2])

pre_nii = glob.glob(os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'*nii'))
pre_nii_data = nb.load(pre_nii[0]).get_fdata()

moco_movie(np.array(aligned_4D), sub+'_post', outdir)
moco_movie(pre_nii_data, sub+'_pre', outdir)