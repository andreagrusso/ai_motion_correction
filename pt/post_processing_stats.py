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
import itertools

from moco_movie import moco_movie


from network import AffineNet
from generators import Create_dataset
from util_functions import output_processing, ants_moco, compare_affine_params, align_and_mse

#%% Data

datadir = '/home/ubuntu22/Desktop/ai_mc'
model_dir = 'affine_bs1_Multidp03_ep20'
sub = 'sub00_run1'
outdir = os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'ai')


#%% Load and plot the model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AffineNet()
model.load_state_dict(torch.load(os.path.join(datadir,'preliminary_nn_results',model_dir,'AffineNet_saved_model.pth')))
model.eval()

model = model.to(device)

#%% Run the model

testing_files = pickle.load(open(os.path.join(datadir,'dcm','dcm_'+sub+'_testing_set.pkl'),'rb'))
testing_set = Create_dataset(testing_files, (128,128,128))
testing_generator = torch.utils.data.DataLoader(testing_set, shuffle=False)

motion_params = np.empty((len(testing_set), 6))
#aligned_data = []
i=0
timing = np.empty((len(testing_set), 1))

diff_vol_pre = np.empty((128,128,128,len(testing_set)))
diff_vol_post = np.empty((128,128,128,len(testing_set)))

aligned_4D = []
all_affine = np.empty((3,4,len(testing_set)))
ai_output_mse = np.empty((len(testing_set), 1))


for fixed_test, movable_test, orig_dim in testing_generator: #just testing

    start = time.time()
    outputs = model(fixed_test['data'].to(device),
                    movable_test['data'].to(device))
    
    crop_vol, curr_affine, curr_motion = output_processing(movable_test,
                                                         outputs, 
                                                         orig_dim)
    aligned_4D.append(crop_vol)
    all_affine[...,i] = curr_affine
    
    timing[i,:] = time.time()-start
    motion_params[i,:] = curr_motion
    
    ai_output_mse[i,:] = align_and_mse(fixed_test['data'], movable_test['data'], 
                                curr_affine, orig_dim)

    
    diff_vol_pre[...,i] = abs(np.squeeze(movable_test['data'].cpu().detach().numpy()-fixed_test['data'].cpu().detach().numpy()))
    diff_vol_post[...,i] = abs(np.squeeze(outputs[0].cpu().detach().numpy()-fixed_test['data'].cpu().detach().numpy()))
                        
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
#plt.savefig(os.path.join(datadir,'preliminary_nn_results',model_dir,'model_losses.svg'), dpi=300)

#%% RMSE for both raw and ants aligned data
ai_mse = np.zeros_like(ai_output_mse) #this MSE is estimate after the alignnment with scipy
std_mse = np.zeros_like(ai_output_mse)
raw_mse = np.zeros_like(ai_output_mse)

pre_nii = glob.glob(os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'*nii'))

bwd_affine, fwd_affine, fw_ants_motion, bw_ants_motion = ants_moco(pre_nii[0], outdir)

raw_data = nb.load(pre_nii[0]).get_fdata()

for i in range(fwd_affine.shape[2]):
    
    raw_mse[i,:] = align_and_mse(raw_data[...,0], raw_data[...,i], 
                            np.empty([]), raw_data[...,0].shape)
    
    std_mse[i,:] = align_and_mse(raw_data[...,0], raw_data[...,i], 
                            fwd_affine[...,i], raw_data[...,0].shape)
    
    ai_mse[i,:] = align_and_mse(raw_data[...,0], raw_data[...,i], 
                            all_affine[...,i], raw_data[...,0].shape)
    







mse_df = pd.DataFrame([], columns=['MSE', 'Vol','Method'])
mse_df['MSE'] = list(np.vstack((raw_mse, ai_mse, ai_output_mse,std_mse)))
mse_df['Method'] = len(motion_params)*['RAW'] + len(motion_params)*['AI'] + len(motion_params)*['AI output'] + len(motion_params)*['STD']
mse_df['Vol'] = 4*list(np.arange(0, len(motion_params)))

mse_df = mse_df.astype({"Vol": int, "MSE": float, "Method": str})

sns.set_theme(style="darkgrid")
f, ax = plt.subplots(1,2, figsize=(15,10))

sns.lineplot(data = mse_df, 
              x='Vol',y='MSE', 
              hue='Method', ax=ax[0])
ax[1].plot(timing)
#f.set_xticks(range(len(pre_mse_sub00)))
plt.tight_layout()
plt.savefig(os.path.join(outdir,sub+'_mse_and_time.svg'), dpi=300)

#%%
# spm_motion_file = glob.glob(os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'spm','rp*.txt'))
# spm_motion = np.loadtxt(spm_motion_file[0])
# spm_motion[:,3:] = np.rad2deg(spm_motion[:,3:])

# df_motion = pd.DataFrame(np.vstack((motion_params,spm_motion)),
#                          columns=['X trans', 'Y trans', 'Z trans',
#                                   'X rot', 'Y rot', 'Z rot'])
# df_motion['Algorithm'] = ['AI' for i in range(len(motion_params))]+['SPM' for i in range(len(motion_params))]
# df_motion['Vol'] = list(np.arange(0, len(motion_params))) + list(np.arange(0, len(motion_params)))


# f, ax = plt.subplots(2,3, figsize=(15,10))
# sns.set_theme(style="darkgrid")

# col =0
# for i in range(2):
#     for j in range(3):
        
#         th = np.max(np.abs(df_motion[df_motion.columns[col]]))
        
#         sns.lineplot(data = df_motion, 
#                      x='Vol',y=df_motion.columns[col], 
#                      hue='Algorithm',
#                      ax = ax[i,j])
#         ax[i,j].set(ylim=(-th-0.01,th+0.01))
#         plt.title(df_motion.columns[col])
#         col +=1
# plt.tight_layout()
# #plt.savefig(os.path.join(outdir,sub+'_axis_motion.svg'), dpi=300)


# #%% plot vol difference
# import matplotlib as mpl
# sns.set_theme(style="darkgrid")

# f, ax = plt.subplots(2,4, figsize=(15,20))

# index_plot = [np.random.randint(0, len(motion_params)) for i in range(4)]

# index_plot.sort()   
# for idx, vol in enumerate(index_plot):
    
    
#     global_min= min([np.min(diff_vol_pre[vol][:,:,64]),
#                       np.min(diff_vol_post[vol][:,:,64])])
    
#     global_max = max([np.max(diff_vol_pre[vol][:,:,64]),
#                       np.max(diff_vol_post[vol][:,:,64])])
        
#     ax[0,idx].imshow((diff_vol_pre[vol][:,:,64]).T, cmap='Reds')
#     ax[0,idx].set_title('Before alignment: vol n '+str(vol))
#     norm = mpl.colors.Normalize(vmin=global_min,
#                                 vmax=global_max)
#     plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'), 
#                   ax=ax[0,idx],
#                   shrink=0.3)  
    
#     ax[1,idx].imshow((diff_vol_post[vol][:,:,64]).T, cmap='Reds')
#     ax[1,idx].set_title('After alignment: vol n '+str(vol))
#     norm = mpl.colors.Normalize(vmin=global_min,
#                                 vmax=global_max)
#     plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'), 
#                   ax=ax[1,idx],
#                   shrink=0.3)
#     plt.tight_layout()
#     plt.suptitle('Difference between movable and fixed voulumes before and after alignment')
    
# plt.savefig(os.path.join(outdir,sub+'_volume_differences.svg'), dpi=300)

# #%% get ANTS affine and compare

# all_affine = np.array(all_affine)
# all_affine = np.moveaxis(all_affine, [0,1,2],[2,0,1])

# pre_nii = glob.glob(os.path.join(datadir,'preliminary_nn_results',model_dir,sub,'*nii'))

# bwd_affine, fwd_affine, fw_ants_motion, bw_ants_motion = ants_moco(pre_nii[0], outdir)

# diff_bwd_affine = np.zeros_like(bwd_affine)
# diff_fwd_affine = np.zeros_like(fwd_affine)


# for i in range(all_affine.shape[-1]):
    
#     affine1 = all_affine[...,i]
    
#     #comparison with backward
#     affine2 = bwd_affine[...,i]
#     diff_bwd_affine[...,i] = compare_affine_params(affine1, affine2)
    
#     #comparison with forward
#     affine2 = fwd_affine[...,i]
#     diff_fwd_affine[...,i] = compare_affine_params(affine1, affine2)



# diff_fwd_affine = np.reshape(diff_fwd_affine, (diff_fwd_affine.shape[0]*diff_fwd_affine.shape[1],
#                                                diff_fwd_affine.shape[2])).T

# diff_bwd_affine = np.reshape(diff_bwd_affine, (diff_bwd_affine.shape[0]*diff_bwd_affine.shape[1],
#                                                diff_bwd_affine.shape[2])).T


# #index couple for plottin
# indices = np.indices((3,4))
# indices = np.reshape(indices,(2,12)).T



# f,ax = plt.subplots(3,4)
# max_val = np.max(abs(diff_bwd_affine))
# for j,idx in enumerate(indices):
#     ax[idx[0],idx[1]].plot(diff_bwd_affine[:,j])
#     ax[idx[0],idx[1]].set_title('A' + str(idx) +' diff')
#     ax[idx[0],idx[1]].set(ylim=(-max_val,max_val))
#     plt.tight_layout()

# plt.title('Diff from backward affine')



# f,ax = plt.subplots(3,4)
# max_val = np.max(abs(diff_fwd_affine))
# for j,idx in enumerate(indices):
#     ax[idx[0],idx[1]].plot(diff_fwd_affine[:,j])
#     ax[idx[0],idx[1]].set_title('A' + str(idx) +' diff')
#     ax[idx[0],idx[1]].set(ylim=(-max_val,max_val))
#     plt.tight_layout()

# plt.title('Diff from forward affine')

# f,ax = plt.subplots(1,1)
# plt.plot(motion_params[:,0])
# plt.plot(fw_ants_motion.T[:,0])
# plt.legend(['AI','ANTs'])

# df_motion = pd.DataFrame(np.vstack((motion_params,fw_ants_motion)),
#                           columns=['X trans', 'Y trans', 'Z trans',
#                                   'X rot', 'Y rot', 'Z rot'])
# df_motion['Algorithm'] = ['AI' for i in range(len(motion_params))]+['ANTs' for i in range(len(motion_params))]
# df_motion['Vol'] = list(np.arange(0, len(motion_params))) + list(np.arange(0, len(motion_params)))


# f, ax = plt.subplots(2,3, figsize=(15,10))
# sns.set_theme(style="darkgrid")

# col =0
# for i in range(2):
#     for j in range(3):
        
#         th = np.max(np.abs(df_motion[df_motion.columns[col]]))
        
#         sns.lineplot(data = df_motion, 
#                       x='Vol',y=df_motion.columns[col], 
#                       hue='Algorithm',
#                       ax = ax[i,j])
#         ax[i,j].set(ylim=(-th-0.01,th+0.01))
#         plt.title(df_motion.columns[col])
#         col +=1
# plt.tight_layout()
# plt.savefig(os.path.join(outdir,sub+'_axis_motion.svg'), dpi=300)

# #%% MoCo movie







# aligned_4D = np.array(aligned_4D)
# aligned_4D = np.transpose(aligned_4D, (1, 2, 3, 0))
# #aligned_4D = np.moveaxis(aligned_4D,[0,1,2,3],[1,2,3,0])

# pre_nii_data = nb.load(pre_nii[0]).get_fdata()

# data4movie = [np.array(aligned_4D), pre_nii_data]
# output_name = sub+'pre_and_post'

# moco_movie(data4movie, output_name, outdir)
