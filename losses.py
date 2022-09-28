# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:52:00 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


from external_layers import soft_quantize
from other_functions import det3x3, elem_sym_polys_of_eigen_values

#%%
def regularizer_rot_matrix(target,W):
    
    """
    Function that combines the two losses applied to the rotation matrix
    One loss is on the orhtogonality to no scaling
    The other loss is on the determinant to ensure no flipping
    The two losses are summed

    Parameters
    ----------
    target : keras layer
        keras layer containing the target.
        Keras always pass the target and the prediction to a loss function, 
        although here is not useful
    W : keras layer
        Keras layer indicating the rotation parameters.

    Returns
    -------
    loss: float
        Sum of the two losses

    """
    
    #determinant should be close to 1
      #target that is an image is not used
      #print(A)
      
    #the affine matrix created in the voxelmorph based neural network is the
    #one of the output of the network and it is a 3x4, thus already in the 
    #good shape for applying the transformation. Therefore, to apply the loss we
    #need only the rotation part (3x3 matrix)
    #print('W shape', W.shape)
    A = tf.reshape(W[:,:,:-1], [-1, 3, 3])#W[:,:,:-1].reshape(3,3)#
    #print('shape A',A.shape)
    #flow_multiplier = 1 
    I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    #W = tf.reshape(W, [-1, 3, 3]) * flow_multiplier
    #A = W + I
  
    det = det3x3(A)
    det_loss = tf.nn.l2_loss(det - 1.0)
    # should be close to being orthogonal
  
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1 by minimizing
    # k1+1/k1+k2+1/k2+k3+1/k3.
    # to prevent NaN, minimize
    # k1+eps + (1+eps)^2/(k1+eps) + ...
    eps = 1e-5
    epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
    C = tf.matmul(A, A, True) + epsI
  
    s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    ortho_loss = tf.reduce_sum(ortho_loss)
  
          #0.1*det_loss + 0.1*ortho_loss
    #print('Mat loss:', det_loss + ortho_loss)
    return det_loss + ortho_loss #to_check


# def voxmorph_loss_for_matrix2(target,W):
    
#     return 0.0

#%%
#LOSS VOXEL MORPH
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, signed=False):
        self.win = win
        self.eps = eps
        self.signed = signed

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        if self.signed:
            cc = cross / tf.sqrt(I_var * J_var + self.eps)
        else:
            # cc = (cross * cross) / (I_var * J_var)
            cc = (cross / I_var) * (cross / J_var)

        return cc

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute cc
        cc = self.ncc(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            cc = tf.reduce_mean(K.batch_flatten(cc), axis=-1)
        elif reduce == 'max':
            cc = tf.reduce_max(K.batch_flatten(cc), axis=-1)
        elif reduce is not None:
            raise ValueError(f'Unknown NCC reduction type: {reduce}')
        # loss
        return 1-cc


#%%
# MUTUAL INFORMATION LOSS
class MutualInformation:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes 
      (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo. 
      Multi-modal image registration with unsupervised deep learning. 
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    # TODO: add local MI by using patches. This is quite memory consuming, though.
    Includes functions that can compute mutual information between volumes, 
      between segmentations, or between a volume and a segmentation map
    mi = MutualInformation()
    mi.volumes      
    mi.segs         
    mi.volume_seg
    mi.channelwise
    mi.maps
    """

    def __init__(self,
                 bin_centers=None,
                 nb_bins=None,
                 soft_bin_alpha=None,
                 min_clip=None,
                 max_clip=None):
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters
        Args:
            bin_centers (np.float32, optional): Array or list of bin centers. Defaults to None.
            nb_bins (int, optional):  Number of bins. Defaults to 16 if bin_centers
                is not specified.
            soft_bin_alpha (int, optional): Alpha in RBF of soft quantization. Defaults
                to `1 / 2 * square(sigma)`.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        """

        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = tf.convert_to_tensor(bin_centers, dtype=tf.float32)
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / (self.nb_bins - 1)
            else:
                sigma = sigma_ratio * tf.reduce_mean(tf.experimental.numpy.diff(bin_centers))
            self.soft_bin_alpha = 1 / (2 * tf.square(sigma))
            print(self.soft_bin_alpha)

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes. 
        Algorithm: 
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of 
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        #print(x.shape)
        tensor_channels_y = K.shape(y)[-1]
        #print(y.shape)
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        tf.debugging.assert_equal(tensor_channels_x, 1, msg)
        tf.debugging.assert_equal(tensor_channels_y, 1, msg)

        # volume mi
        return K.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps. 
        Wraps maps()        
        Parameters:
            x and y:  [bs, ..., nb_labels]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps. 
        Wraps maps()        
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        tf.debugging.assert_equal(tf.minimum(tensor_channels_x, tensor_channels_y), 1, msg)
        # otherwise we don't know which one is which
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1, msg)

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])                       # [bs, ..., B]
        else:
            y = self._soft_sim_map(y[..., 0])                       # [bs, ..., B]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this 
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to 
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

        # reshape to [bs, V, C]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, C]
            y = tf.reshape(y, new_shape)                             # [bs, V, C]

        # move channels to first dimension
        ndims_k = len(x.shape)
        permute = [ndims_k - 1] + list(range(ndims_k - 1))
        cx = tf.transpose(x, permute)                                # [C, bs, V]
        cy = tf.transpose(y, permute)                                # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)                                  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                                  # [C, bs, V, B]

        # get mi
        map_fn = lambda x: self.maps(*x)
        cout = tf.map_fn(map_fn, [cxq, cyq], dtype=tf.float32)       # [C, bs]

        # permute back
        return tf.transpose(cout, [1, 0])                            # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains 
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output 
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each itemin the batch, so the joint probabilities 
        might be  different across inputs. In some cases, computing MI actoss the whole batch 
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the 
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
        tf.debugging.assert_non_negative(x)
        tf.debugging.assert_non_negative(y)

        eps = K.epsilon()

        # reshape to [bs, V, B]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, B1]
            y = tf.reshape(y, new_shape)                             # [bs, V, B2]

        # joint probability for each batch entry
        x_trans = tf.transpose(x, (0, 2, 1))                         # [bs, B1, V]
        pxy = K.batch_dot(x_trans, y)                                # [bs, B1, B2]
        pxy = pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)   # [bs, B1, B2]

        # x probability for each batch entry
        px = K.sum(x, 1, keepdims=True)                              # [bs, 1, B1]
        px = px / (K.sum(px, 2, keepdims=True) + eps)                # [bs, 1, B1]

        # y probability for each batch entry
        py = K.sum(y, 1, keepdims=True)                              # [bs, 1, B2]
        py = py / (K.sum(py, 2, keepdims=True) + eps)                # [bs, 1, B2]

        # independent xy probability
        px_trans = K.permute_dimensions(px, (0, 2, 1))               # [bs, B1, 1]
        pxpy = K.batch_dot(px_trans, py)                             # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = K.log(pxy / pxpy_eps + eps)                       # [bs, B1, B2]
        mi = K.sum(pxy * log_term, axis=[1, 2])                      # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image. 
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=True)               # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image. 
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=False)              # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        x_hist = self._soft_sim_map(x, **kwargs)                      # [bs, ..., B]
        x_hist_sum = K.sum(x_hist, -1, keepdims=True), K.epsilon()   # [bs, ..., B]
        x_prob = x_hist / (x_hist_sum)                               # [bs, ..., B]
        return x_prob
    
    
    
class MI_loss(MutualInformation):
    
    def loss(self,y_true,y_pred):
        return 1-self.volumes(y_true, y_pred)


#%% Losses for VTN-based network
#loss function to estimate similarity between the transformed img2 with img1
def similarity_loss(target,pred):
    """
    

    Parameters
    ----------
    target : keras layer
        Layer containing the target image.
    pred : keras layer
        Layer containing the movable image aligned.

    Returns
    -------
    raw_loss : float
        Output value of the cross-correlation.

    """

    #input is a set of two concatenated images
    #first the fixed image (true) and then the warped (predicted)
    fixed_img = target
    warped_img = pred
    #print('NaN in warped image:',tf.math.is_nan(warped_img))
    
    

    

    sizes = np.prod(fixed_img.shape.as_list()[1:])
    flatten1 = tf.reshape(fixed_img, [-1, sizes])
    flatten2 = tf.reshape(warped_img, [-1, sizes])

    mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
    #print('Mean1:',mean1)
    mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
    #print('Mean2:',mean2)
    var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
    #print('var1:',var1)
    var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
    #print('var2:',var2)    
    cov12 = tf.reduce_mean(
        (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
    #print('cov12:',cov12)
    
    pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))
    #print(pearson_r)
    raw_loss = 1 - pearson_r
    #print('Loss:',raw_loss)
    raw_loss = tf.reduce_mean(raw_loss)
    
    return raw_loss

#loss to ensure specific characteristic to the estimated matrix
def loss_for_matrix(target,W):
    
    """
    Function that combines the two losses applied to the rotation matrix
    One loss is on the orhtogonality to no scaling
    The other loss is on the determinant to ensure no flipping
    The two losses are summed

    Parameters
    ----------
    target : keras layer
        keras layer containing the target.
        Keras always pass the target and the prediction to a loss function, 
        although here is not useful
    W : keras layer
        Keras layer indicating the rotation parameters.

    Returns
    -------
    loss: float
        Sum of the two losses

    """
    
    #determinant should be close to 1
      #target that is an image is not used
      #print(A)
    flow_multiplier = 1 
    I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    W = tf.reshape(W, [-1, 3, 3]) * flow_multiplier
    A = W + I
  
    det = det3x3(A)
    det_loss = tf.nn.l2_loss(det - 1.0)
    # should be close to being orthogonal
  
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1 by minimizing
    # k1+1/k1+k2+1/k2+k3+1/k3.
    # to prevent NaN, minimize
    # k1+eps + (1+eps)^2/(k1+eps) + ...
    eps = 1e-5
    epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
    C = tf.matmul(A, A, True) + epsI
  
    s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    ortho_loss = tf.reduce_sum(ortho_loss)
  
          #0.1*det_loss + 0.1*ortho_loss
    #print('Mat loss:', det_loss + ortho_loss)
    return det_loss + ortho_loss #to_check


def loss_for_vector(target,b):
    """
    Fake function as on the vector displacement.
     We do not apply any loss 

    Parameters
    ----------
    target : keras layer
        keras layer containing the target.
        Keras always pass the target and the prediction to a loss function, 
        although here is not useful
    b : numpy array
        Displacement vecto.

    Returns
    -------
    float

    """

    return 0.0
#%% Metric (Dice coeffiecient)
class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        vol_axes = list(range(1, ndims + 1))

        top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
        bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

        div_no_nan = tf.math.divide_no_nan if hasattr(
            tf.math, 'divide_no_nan') else tf.div_no_nan  # pylint: disable=no-member
        dice = tf.reduce_mean(div_no_nan(top, bottom))
        return dice