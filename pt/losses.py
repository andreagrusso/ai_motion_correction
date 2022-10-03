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


def elem_sym_polys_of_eigen_values(M):
    
    #to calculate quickly the trace on batch. Sum across diagonal
    sigma1 = torch.einsum('bii->b', M) 
    
    
    sigma_2_a = torch.sum(torch.stack((M[:,0,0]*M[:,1,1],
                                      M[:,1,1]*M[:,2,2],
                                      M[:,2,2]*M[:,0,0])), dim=0)
    
    sigma_2_b = torch.sum(torch.stack((M[:,0,1]*M[:,1,0], 
                                        M[:,1,2]*M[:,2,1], 
                                        M[:,2,0]*M[:,0,2])),dim=0)
    
    sigma2 = torch.sum(torch.stack((sigma_2_a,-1*sigma_2_b)), dim=0) 
    



    sigma_3_a = torch.sum(torch.stack((M[:,0,0]*M[:,1,1]*M[:,2,2],
                                    M[:,0,1]*M[:,1,2]*M[:,2,0],
                                    M[:,0,2]*M[:,1,0]*M[:,2,1])), dim = 0)

    sigma_3_b = torch.sum(torch.stack((M[:,0,0]*M[:,1,2]*M[:,2,1],
                                M[:,0,1]*M[:,1,0]*M[:,2,2],
                                M[:,0,2]*M[:,1,1]*M[:,2,0])), dim = 0) 
                               
    sigma3 = torch.sum(torch.stack((sigma_3_a,-1*sigma_3_b)), dim=0)
    
    return sigma1, sigma2, sigma3

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win])#.to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1-torch.mean(cc)



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
         
        mse_loss = torch.nn.MSELoss()
        #the affine matrix created in the voxelmorph based neural network is the
        #one of the output of the network and it is a 3x4, thus already in the 
        #good shape for applying the transformation. Therefore, to apply the loss we
        #need only the rotation part (3x3 matrix)
        #print('W shape', W.shape)
        
        # det_loss = []
        # ortho_loss = []
        
        #loop across batch elements

        W = torch.reshape(matrix, (matrix.shape[0],3,3))
        #print('shape A',A.shape)
        #flow_multiplier = 1 
        I = self.identity#[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        #W = tf.reshape(W, [-1, 3, 3]) * flow_multiplier
        A = W + I
          
        det = torch.det(A)
        det_loss = mse_loss(det,torch.ones(det.shape))

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
        orth_loss = torch.zeros(1)
        reg = 1e-6
        param_flat = A.view(A.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0])
        orth_loss = orth_loss + (reg * sym.abs().sum())
    

        tot_loss = torch.stack((det_loss,orth_loss[0]))
        
        
        return torch.sum(tot_loss) #to_check