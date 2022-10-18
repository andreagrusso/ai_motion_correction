# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 17:37:16 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import torch, math
import torch.nn.functional as F
import numpy as np


class Dice:
    """
    N-D dice for segmentation
    """
    

    def loss(self, y_true, y_pred):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
        
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice
    

#adapted from https://github.com/yuta-hi/pytorch_similarity/blob/master/torch_similarity/modules/normalized_cross_correlation.py
def NCC(x, y, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    x.to(device)
    y = y.to(device)
    
    shape = x.shape
    b = shape[0]

    # reshape
    x = x.reshape(b, -1) #x.view(b,-1)
    y = y.reshape(b, -1) #y.view(b,-1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    #ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    # if not return_map:
    #     return ncc

    return 1-ncc#, ncc_map




class regularizer_rot_matrix:
    
    def __init__(self):
        self.identity = torch.eye(3)
           
    
    
    
    def loss(self,matrix):
    
        """
        Function that combines the two losses applied to the rotation matrix
        One loss is on the orhtogonality to no scaling
        The other loss is on the determinant to ensure no flipping
        The two losses are summed
        
        Parameters
        ----------
        W : rotation parameters.
        
        Returns
        -------
        loss: float
            Sum of the two losses
        
        """
        
        #determinant should be close to 1
          #target that is an image is not used
          #print(A)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mse_loss = torch.nn.MSELoss()
        #the affine matrix created in the voxelmorph based neural network is the
        #one of the output of the network and it is a 3x4, thus already in the 
        #good shape for applying the transformation. Therefore, to apply the loss we
        #need only the rotation part (3x3 matrix)
        #print('W shape', W.shape)
        
        # det_loss = []
        # ortho_loss = []
        
        #loop across batch elements

        # if matrix.size!=3:
        #     matrix = torch.unsqueeze(matrix, 0)
        W = matrix[:,:,:-1].view(-1,3,3)#(matrix, (matrix.shape[0],3,3))
        A = W

        # W = torch.reshape(matrix, (matrix.shape[0],3,3))
        # I = self.identity.to(device)#[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        # A = W + I
          
        det = torch.det(A)
        det_loss = mse_loss(det,torch.ones(det.shape).to(device))

        #print(det_loss)

        # should be close to being orthogonal
          
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        # eps = 1e-5
        # epsI = eps*I#[[[eps * elem for elem in row] for row in Mat] for Mat in I]
        # C = torch.matmul(A, A) + epsI
          
        # s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        # #print(s1, s2, s3)
        # ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        # ortho_loss = torch.mean(ortho_loss)
        
        
        # print(det_loss)
        # print(ortho_loss)
            
        #A matrix is orthogonal if A*A = I
        #this code has been adapted from https://github.com/kevinzakka/pytorch-goodies#orthogonal-regularization 
        orth_loss = torch.zeros(1).to(device)
        reg = 1e-5
        param_flat = A.reshape(A.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat).to(device))
        sym -= torch.eye(param_flat.shape[0]).to(device)
        orth_loss = orth_loss + (reg * sym.abs().sum())
    

        tot_loss = torch.stack((det_loss,orth_loss[0]))
        
        
        return torch.sum(tot_loss) #to_check