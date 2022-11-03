#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:57:34 2022

@author: ubuntu22
"""
import os, glob,imageio
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
import matplotlib.pyplot as plt

def moco_movie(datalist, sub_name, outdir):
#inspired by Alessandra Pizzuti's code
#https://github.com/27-apizzuti/Atomics/blob/main/MotionCorrection/moco_movies.py
    
    data_ai, data_std = datalist


    dumpFolder = os.path.join(outdir, 'moco_movie')
    
    if not os.path.exists(dumpFolder):
        os.mkdir(dumpFolder)
    
    #dataArr = nb.load(os.path.join(PATH_IN, '{}.nii.gz'.format(FILE_IN))).get_fdata()
    sliceNr = 32
    
    scaler = MinMaxScaler()
    #ai data are already scaled in [0-1]
    for i in range(data_std.shape[3]):
        
        data_std[...,i] = scaler.fit_transform(data_std[...,i].reshape(-1,1)).reshape(data_std[...,i].shape)




############# LOOP FOR ALL OTHER VOLUMES ######################################    
    for frame in range(data_ai.shape[-1]):
        
        slice_std = ndimage.rotate(data_std[:,:,int(sliceNr),frame],90)
        slice_ai = ndimage.rotate(data_ai[:,:,int(sliceNr),frame],90)
        
        f,ax = plt.subplots(1,2)
        f.set_facecolor((0, 0, 0))
        
        #ax[0].imshow(fixed_frame_std, cmap='Greys_r')
        ax[0].imshow(slice_std, cmap='Greys_r')
        ax[0].grid(False)
        ax[0].axis('off')
        #ax[1].imshow(fixed_frame_ai, cmap='Greys_r')
        ax[1].imshow(slice_ai, cmap='Greys_r')
        ax[1].grid(False)
        ax[1].axis('off')
               
        
        plt.savefig(os.path.join('{}'.format(dumpFolder), 'frame{}.png'.format(frame)))
        plt.close(f)
        #time.sleep(0.3)
    
        #imageio.imwrite(os.path.join('{}'.format(dumpFolder), 'frame{}.png'.format(frame)), img)
    
    
    files = sorted(glob.glob(os.path.join(outdir,'moco_movie','*.png')))
    print('Creating gif from {} images'.format(len(files)))

    
    writer = imageio.get_writer(os.path.join(outdir, '{}_movie.mp4'.format(sub_name)), fps=10)
    # Increase the fps to 24 or 30 or 60
    
    for file in files:
        filedata = imageio.imread(file)
        writer.append_data(filedata)
    writer.close()
    print('Deleting dump directory')
    os.system(f'rm -r {dumpFolder}')
    print('Done.')



