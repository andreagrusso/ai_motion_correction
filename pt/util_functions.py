# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:54:27 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""

import numpy as np
import os, torch, math, imageio, glob, shutil
import torchio as tio
from losses import Dice
import nibabel as nb
from scipy import ndimage
import ants
from nilearn.image import new_img_like 

from dicom2nifti.convert_dicom import dicom_array_to_nifti
import pydicom


def mosaic_to_mat(dcm_file):
    """ Snippets taken from https://dicom2nifti.readthedocs.io/ and arranged for
    our needs.
    """


    
    
    dicom_header = []
    dicom_header.append(pydicom.read_file(dcm_file,
                                 defer_size=None,
                                 stop_before_pixels=False,
                                 force=False))
    
    
      
    outfile = dcm_file[:-3]+'nii'
        
    
    #nii = dicom_array_to_nifti(dicom_header,outfile,reorient_nifti=True)
    nii = dicom_array_to_nifti(dicom_header,outfile)
    
    mat = nii['NII'].get_fdata()
    world_affine = nii['NII'].affine
    
    if os.path.isfile(nii['NII_FILE']):
        os.remove(nii['NII_FILE'])
    
    return mat, world_affine
    
    


def mat_to_mosaic(mosaic_dcm, data_matrix, outdir, idx_dcm, name):
    """
    

    Parameters
    ----------
    mosaic_dcm : dicom file from pydicom
        This is the original dicom file that will be used as a canvas 
        to write the new aligned data.
    data_matrix : numpy 3d data
        3D numpy matrix of the aligned data.
    outdir : string
        Directory where output will be saved.
    idx_dcm : integer
        Number of the dicom aligned compared to the length of the full
        time series.
    name : str
        Name for the new dcm.

    Returns
    -------
    None.

    """
            

    acq_matrix = np.array(mosaic_dcm.AcquisitionMatrix)
    acq_matrix = acq_matrix[acq_matrix!=0]
    vox_col, vox_row = mosaic_dcm.Columns, mosaic_dcm.Rows
    
    nr_slices = data_matrix.shape[2]
    
    old_pixel_array = mosaic_dcm.pixel_array
    new_pixel_array = np.zeros_like(old_pixel_array)
    
    col_idx = np.arange(0,vox_col+1,acq_matrix[1])
    row_idx = np.arange(0,vox_row+1,acq_matrix[0])
    
    i=0 #index to substract from the total number of slice
    for r, row_id in enumerate(row_idx[:-1]):
        
        if i==nr_slices-1:
            break
        
        #loop over columns
        for c,col_id in enumerate(col_idx[:-1]):
            
            new_pixel_array[row_id:row_idx[r+1],col_id:col_idx[c+1]] = data_matrix[:,:,i]
            i += 1
            
            if i==nr_slices-1:
                break
            
    
    #swap the old data with the new
    #mosaic_dcm.pixel_array = new_pixel_array
    mosaic_dcm.PixelData = new_pixel_array.tobytes()

    mosaic_dcm.save_as(os.path.join(outdir,name+str(idx_dcm)+'.dcm'))
        
    
    
    

def output_processing(fixed,movable,outputs,orig_dim, world_affine):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    dice_fn = Dice()
    mse_fn = torch.nn.MSELoss()
    
    data_tensor = outputs[0]
    matrix = np.squeeze(outputs[1].cpu().detach().numpy())
    world_affine = np.squeeze(world_affine.cpu().detach().numpy())
   
    
    matrix = ThetaToM(matrix, 128, 128, 128)
    #matrix = matrix[:-1,:]
    
    #trasnform to world coordinates
    matrix =  np.linalg.inv(world_affine) @ np.linalg.inv(matrix) @ world_affine
    matrix = matrix[:-1,:]

    #rot_params = outputs[1].cpu().detach().numpy()
    #trans_params = outputs[2].cpu().detach().numpy()
    
    orig_dim = [val.detach().numpy()[0] for val in orig_dim]
    
    motion_params = np.empty((1,6))
    
    
    #return volume to original dimension
    tensor = tio.ScalarImage(tensor=torch.squeeze(data_tensor,0))
    padding = tio.transforms.CropOrPad(tuple(orig_dim))
    crop_vol = padding(tensor)
    crop_vol = np.squeeze(crop_vol['data'].cpu().detach().numpy())
    
    #vec = np.zeros((1,4))
    #vec[-1] = 1
    #matrix = np.linalg.inv(np.vstack((matrix,vec)))
    #matrix = matrix[:-1,:]
    #motion parameters
    #motion_params[0,:3] = trans_params.reshape(1,-1)
    motion_params[0,:3] = matrix[:,-1].reshape(1,-1)

    
    #rotation params (hopefully in radians)
    #source Rigid Body Registration John Ashburner & Karl J. Friston
    #rot_mat = rot_params.reshape(3,3)+np.eye(3)
    rot_mat = matrix[:,:-1]
    q5 = np.arcsin(rot_mat[0,2]) #q5
    motion_params[0,4] = np.rad2deg(q5)
    
    
    if (abs(q5 - np.pi/2))**2 < 1e-9:
        
        q4 = 0
        motion_params[0,3] = np.rad2deg(q4)
        
        q6 = math.atan2(-rot_mat[1,0],
                        rot_mat[2,0]/rot_mat[0,2]) #q6
        motion_params[0,5] = np.rad2deg(q6)  
    
    else:
    
        q4 = math.atan2(rot_mat[1,2]/math.cos(q5),
                        rot_mat[2,2]/math.cos(q5)) #q4
        motion_params[0,3] = np.rad2deg(q4)
        
        q6 = math.atan2(rot_mat[0,1]/math.cos(q5),
                        rot_mat[0,0]/math.cos(q5)) #q6
        motion_params[0,5] = np.rad2deg(q6)
    
    
    #estimate the dice coefficient with the target
    fixed = fixed.to(device)
    #print(fixed.device)
    data_tensor = data_tensor.to(device)
    #print(data_tensor.device)
    movable = movable.to(device)
    #print(movable.device)
    
    dice_post = mse_fn(fixed,data_tensor)#dice_fn.loss(fixed,data_tensor)
    dice_post = dice_post.cpu().detach().numpy()
    
    #dice index with the original data
    dice_pre = mse_fn(fixed,movable)#dice_fn.loss(fixed,movable)
    dice_pre = dice_pre.cpu().detach().numpy()

    return crop_vol, matrix, motion_params, dice_post, dice_pre

    
    
#%% Coordinates transformation test

def get_N(W, H, D):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((4, 4), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[0, 2] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[1, 2] = 0
    N[2, 2] = 2.0 / D
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[2, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H, D):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H, D)
    return np.linalg.inv(N)



def ThetaToM(theta, w, h, d, return_inv=False):
    """convert theta matrix compatible with `torch.F.affine_grid` to affine warp matrix `M`
    compatible with `opencv.warpAffine`.

    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required

    Parameters
    ----------
    theta : np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    return_inv : False
        return M^-1 instead of M.

    Returns
    -------
    np.ndarray
        affine warp matrix `M` shaped [2, 3]
    """
    
    theta_aug = np.concatenate([theta, np.zeros((1, 4))], axis=0)
    theta_aug[-1, -1] = 1.0
    N = get_N(w, h, d)
    N_inv = get_N_inv(w, h, d)
    M = np.linalg.inv(theta_aug)
    M = N_inv @ M @ N
    if return_inv:
        M_inv = np.linalg.inv(M)
        return M_inv
    return M

#%% MoCo movie



def moco_movie(dataArr, sub_name, outdir):
#credit to Alessandra Pizzuti
#https://github.com/27-apizzuti/Atomics/blob/main/MotionCorrection/moco_movies.py



    dumpFolder = os.path.join(outdir, 'moco_movie')
    
    if not os.path.exists(dumpFolder):
        os.mkdir(dumpFolder)
    
    #dataArr = nb.load(os.path.join(PATH_IN, '{}.nii.gz'.format(FILE_IN))).get_fdata()
    sliceNr = 32
    globalMax = 0
    globalMin = 0
    
    for frame in range(dataArr.shape[3]):
        imgData = dataArr[:,:,int(sliceNr),frame]
        rotated_img = ndimage.rotate(imgData, 90)
    
        if np.amin(rotated_img) <= globalMin:
            globalMin = np.amin(rotated_img)
        if np.amax(rotated_img) >= globalMax:
            globalMax = np.amax(rotated_img)
            # // change the maxium with 75 percentile
    
    for frame in range(dataArr.shape[3]):
        imgData = dataArr[:,:,int(sliceNr),frame]
        rotated_img = ndimage.rotate(imgData, 90)
    
    
        rotated_img[0,0] = globalMax
        rotated_img[0,1] = globalMin
    
        rotated_img = (rotated_img - globalMin)/ (globalMax-globalMin)
        rotated_img = rotated_img.astype(np.float64)  # normalize the data to 0 - 1
        rotated_img = 255 * rotated_img # Now scale by 255
        img = rotated_img.astype(np.uint8)
    
        imageio.imwrite(os.path.join('{}'.format(dumpFolder), 'frame{}.png'.format(frame)), img)
    
    
    files = sorted(glob.glob(os.path.join(outdir,'moco_movie','*.png')))
    print('Creating gif from {} images'.format(len(files)))
    # images = []
    # for file in files:
    #     filedata = imageio.imread(file)
    #     images.append(filedata)
    
    # imageio.mimsave(os.path.join(PATH_IN, '{}_movie.gif'.format(FILE_IN)), images, duration = 1/10)
    
    writer = imageio.get_writer(os.path.join(outdir, '{}_movie.mp4'.format(sub_name)), fps=20)
    # Increase the fps to 24 or 30 or 60
    
    for file in files:
        filedata = imageio.imread(file)
        writer.append_data(filedata)
    writer.close()
    print('Deleting dump directory')
    os.system(f'rm -r {dumpFolder}')
    print('Done.')




def params2mat(params):
    
    mat = np.eye(4)
    
    #translation
    mat[:-1,-1] = params[9:]
    mat[:-1,:-1] = np.reshape(params[:9],(3,3))
    #get a 3x4 matrix
    mat = mat[:-1,:]
    
    return mat

def ants_motion(params):
    
    dx, dy, dz = params[9:]

    rot_x = np.arcsin(params[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(params[7] / cos_rot_x, params[8] / cos_rot_x)
    rot_z = np.arctan2(params[3] / cos_rot_x, params[0] / cos_rot_x)
    
    return np.array([dx, dy, dz, 
                     np.degrees(rot_x), np.degrees(rot_y), np.degrees(rot_z)]) 



def ants_moco(datafile, outdir):
    """
    Perform the motion correction with antspy.
    Credit to Alessandra Pizzuti
    Adpated from 
    https://github.com/27-apizzuti/Atomics/blob/main/MotionCorrection/my_ants_rigid_motion_correction.py
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    outdir : TYPE
        DESCRIPTION.

    Returns
    -------
    
    arrays of affine matrices

    """
    
    tmp_dir = os.path.join(outdir,'tmp')
    print('Create temporary directory')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    #get the name of the fiel with no extension (assuming no dots in the name)
    basename = os.path.basename(datafile).split('.')[0]
    
    #losd nifti 
    nii = nb.load(datafile)
    data = nii.get_fdata()
    #get the fixed (first vol)
    fixed = data[..., 0]
    
    
    #save as nifti the reference volume
    img = nb.Nifti1Image(fixed, header=nii.header, affine=nii.affine)
    nb.save(img, os.path.join(tmp_dir, '{}_ref_vol.nii.gz'.format(basename)))
    print('...Save {} in {}'.format('{}_ref_vol.nii.gz'.format(basename), os.path.join(tmp_dir)))
    fixed = ants.image_read(os.path.join(tmp_dir, '{}_ref_vol.nii.gz'.format(basename)))
    
    
    
    #prepare output data
    aligned_data = np.zeros_like(data)
    #backward matrices
    bwd_matrices = np.zeros((3,4,data.shape[-1]))
    #forward matrices
    fwd_matrices = np.zeros((3,4,data.shape[-1]))   
    
    bw_motion = np.empty((6, data.shape[-1]))
    fw_motion = np.empty((6, data.shape[-1]))

    for idx_vol in range(data.shape[-1]):
        print('Volume {}'.format(idx_vol))
        myvol = data[..., idx_vol]

        # Save individual volumes: this step is needed since ANTS registration input must be both ANTs object.
        img = nb.Nifti1Image(myvol, header=nii.header, affine=nii.affine)
        nb.save(img, os.path.join(tmp_dir, '{}_vol_{}.nii.gz'.format(basename, idx_vol)))
        print('...Save {} in {}'.format('{}_vol_{}.nii.gz'.format(basename, idx_vol), os.path.join(tmp_dir)))
        moving = ants.image_read(os.path.join(tmp_dir, '{}_vol_{}.nii.gz'.format(basename, idx_vol)))
    
        print('...Find trasformation matrix for {}, vol {}'.format(basename, idx_vol))
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform = 'Rigid')
        print('...Save transformation matrix')
        os.system(f"cp {mytx['fwdtransforms'][0]} {outdir}/{basename}_vol_{idx_vol}_affine.mat")
        bwd_params = ants.read_transform(mytx['invtransforms'][0]).parameters
        bw_motion[...,idx_vol] = ants_motion(bwd_params)
        
        bwd_matrices[...,idx_vol] = params2mat(bwd_params)
        fwd_params = ants.read_transform(mytx['fwdtransforms'][0]).parameters
        fwd_matrices[...,idx_vol] = params2mat(fwd_params)
        bw_motion[...,idx_vol] = ants_motion(fwd_params)

    
        # // Apply transformation
        mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, 
                                              transformlist=mytx['fwdtransforms'], 
                                              interpolator='bSpline')
        ants.image_write(mywarpedimage, os.path.join(tmp_dir, '{}_vol_{}_warped.nii.gz'.format(basename, idx_vol)))
        # Step needed to read the warped image
        nii2 = nb.load(os.path.join(tmp_dir, '{}_vol_{}_warped.nii.gz'.format(basename, idx_vol)))
        mywarp = nii2.get_fdata()
        aligned_data[..., idx_vol] = mywarp
        
    new_nii = new_img_like(nii,aligned_data)
    new_nii.to_filename(os.path.join(outdir,'{}_ants_warped.nii.gz'.format(basename)))
    print('ANTs aligned nifti saved!')
    print('... Removing temporary directory')
    shutil.rmtree(tmp_dir)
    
    
        
    return bwd_matrices, fwd_matrices, fw_motion, bw_motion
    
    
    
    
def compare_affine_params(affine1, affine2):
    
    return(affine1-affine2)