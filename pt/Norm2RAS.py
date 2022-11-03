#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:54:55 2022

@author: ubuntu22
"""

import numpy as np

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


def convert_mat(matrix, world_affine):
    #from norm coords to vox coords
    matrix = ThetaToM(matrix, 128, 128, 128)  
    #from vox coords to world RAS+ coords
    matrix =  np.linalg.inv(world_affine) @ np.linalg.inv(matrix) @ world_affine
    matrix = matrix[:-1,:]
    
    return matrix