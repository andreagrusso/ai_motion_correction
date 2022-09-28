# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:47:58 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import itertools, warnings




#%%############################################################################ 
#fucntions from Neurite used in our network
###############################################################################

# these two function have to replace the volshape_to:_meshgrid as we need to 
#use of the ij indexing that is not supported in volshape_to_meshgrid
def volshape_to_ndgrid(volshape, **kwargs):
    """
    compute Tensor ndgrid from a volume size
    Parameters:
        volshape: the volume size
        **args: "name" (optional)
    Returns:
        A list of Tensors
    See Also:
        ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return ndgrid(*linvec, **kwargs)


def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing
    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)
    Returns:
        A list of Tensors
    """
    return meshgrid(*args, indexing='ij', **kwargs)

###############################################################################
def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def meshgrid(*args, **kwargs):
    """
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921
    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """

    indexing = kwargs.pop("indexing", "xy")
    # name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output

def voxmorph_flatten(v):
    """
    flatten Tensor v
    Parameters:
        v: Tensor to be flattened
    Returns:
        flat Tensor
    """

    return tf.reshape(v, [-1])


def voxmorph_volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size
    Warning: this uses the tf.meshgrid convention, of 'xy' indexing.
    to use `ij` indexing, use the ndgrid equivalent
    Parameters:
        volshape: the volume size
        **args: "name" (optional)
    Returns:
        A list of Tensors
    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def voxmorph_interpn(vol, loc, interp_method='linear', fill_value=None):
    """
    N-D gridded interpolation in tensorflow
    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions
    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
        fill_value: value to use for points outside the domain. If None, the nearest
            neighbors will be used (default).
    Returns:
        new interpolated volume of the same size as the entries in loc
    If you find this function useful, please cite the original paper this was written for:
        VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
        G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
        IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 
        Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
        A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
        MedIA: Medical Image Analysis. (57). pp 226-236, 2019 
    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    if not loc.dtype.is_floating:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating else 'float32'
        loc = tf.cast(loc, target_loc_dtype)
    elif vol.dtype.is_floating and vol.dtype != loc.dtype:
        loc = tf.cast(loc, vol.dtype)

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    max_loc = [d - 1 for d in vol.get_shape().as_list()]

    # interpolate
    if interp_method == 'linear':
        # floor has to remain floating-point since we will use it in such operation
        loc0 = tf.floor(loc)

        # clip values
        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        # note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]

        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:

            # get nd values
            # note re: indices above volumes via
            #   https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's
            #   too expensive. Instead we fill the output with zero for the corresponding value.
            #   The CPU version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because gather_nd needs tf.stack which is slow :(
            idx = sub2ind2d(vol.shape[:-1], subs)
            vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
            vol_val = tf.gather(vol_reshape, idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest', \
            'method should be linear or nearest, got: %s' % interp_method
        roundloc = tf.cast(tf.round(loc), 'int32')
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind2d(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = tf.constant(fill_value, dtype=out_type)
        below = [tf.less(loc[..., d], 0) for d in range(nb_dims)]
        above = [tf.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)]
        out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
        interp_vol *= tf.cast(tf.logical_not(out_of_bounds), out_type)
        interp_vol += tf.cast(out_of_bounds, out_type) * fill_value

    # if only inputted volume without channels C, then return only that channel
    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol

def soft_quantize(x,
                  bin_centers=None,
                  nb_bins=16,
                  alpha=1,
                  min_clip=-np.inf,
                  max_clip=np.inf,
                  return_log=False):
    """
    (Softly) quantize intensities (values) in a given volume, based on RBFs. 
    In numpy this (hard quantization) is called "digitize".
    Specify bin_centers OR number of bins 
        (which will estimate bin centers based on a heuristic using the min/max of the image)
    Algorithm: 
    - create (or obtain) a set of bins
    - for each array element, that value v gets assigned to all bins with 
        a weight of exp(-alpha * (v - c)), where c is the bin center
    - return volume x nb_bins
    Parameters:
        x [bs, ...]: intensity image. 
        bin_centers (np.float32 or list, optional): bin centers for soft histogram.
            Defaults to None.
        nb_bins (int, optional): number of bins, if bin_centers is not specified. 
            Defaults to 16.
        alpha (int, optional): alpha in RBF.
            Defaults to 1.
        min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
        max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        return_log (bool, optional): [description]. Defaults to False.
    Returns:
        tf.float32: volume with one more dimension [bs, ..., B]
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    if bin_centers is not None:
        bin_centers = tf.convert_to_tensor(bin_centers, tf.float32)
        assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
        nb_bins = bin_centers.shape[0]
    else:
        if nb_bins is None:
            nb_bins = 16
        # get bin centers dynamically
        # TODO: perhaps consider an option to quantize by percentiles:
        #   minval = tfp.stats.percentile(x, 1)
        #   maxval = tfp.stats.percentile(x, 99)
        minval = K.min(x)
        maxval = K.max(x)
        bin_centers = tf.linspace(minval, maxval, nb_bins)

    # clipping at bin values
    x = x[..., tf.newaxis]                                                # [..., 1]
    x = tf.clip_by_value(x, min_clip, max_clip)

    # reshape bin centers to be (1, 1, .., B)
    new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
    bin_centers = K.reshape(bin_centers, new_shape)                       # [1, 1, ..., B]

    # compute image terms
    # TODO: need to go to log space? not sure
    bin_diff = K.square(x - bin_centers)                                  # [..., B]
    log = -alpha * bin_diff                                               # [..., B]

    if return_log:
        return log                                                        # [..., B]
    else:
        return K.exp(log)                                                 # [..., B]







#%%############################################################################
# Voxelmorph layer and functions

############################################################################### 
def is_affine_shape(shape):
    """
    Determins whether the given shape (single-batch) represents an
    affine matrix.
    Parameters:
        shape:  List of integers of the form [N, N+1], assuming an affine.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False


def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.
    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')

def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.
    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.
    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.
    Returns:
        Dense shift (warp) of shape (*shape, N).
    """

    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = volshape_to_ndgrid(shape) #USE THIS FUNCTION FOR ij INDEXING
    #mesh = voxmorph_volshape_to_meshgrid(shape, indexing=indexing) #function defined above
    mesh = [f if f.dtype == matrix.dtype else tf.cast(f, matrix.dtype) for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [voxmorph_flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype=matrix.dtype))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=ndims)


def params_to_affine_matrix(par,
                            deg=True,
                            shift_scale=False,
                            last_row=False,
                            ndims=3):
    """
    Constructs an affine transformation matrix from translation, rotation, scaling and shearing
    parameters in 2D or 3D. Supports batched inputs.
    Arguments:
        par: Parameters as a scalar, numpy array, TensorFlow tensor, or list or tuple of these.
            Elements of lists and tuples will be stacked along the last dimension, which
            corresponds to translations, rotations, scaling and shear. The size of the last
            axis must not exceed (N, N+1), for N dimensions. If the size is less than that,
            the missing parameters will be set to identity.
        deg: Whether the input rotations are specified in degrees. Defaults to True.
        shift_scale: Add 1 to any specified scaling parameters. This may be desirable
            when the parameters are estimated by a network. Defaults to False.
        last_row: Append the last row and return the full matrix. Defaults to False.
        ndims: Dimensionality of transform matrices. Must be 2 or 3. Defaults to 3.
    Returns:
        Affine transformation matrices as a (..., M, N+1) tensor, where M is N or N+1,
        depending on `last_row`.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), in press, 2021
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(par, (list, tuple)):
        par = tf.stack(par, axis=-1)
        #par = tf.concat(par, axis=-1)
        #print('par',par.shape)


    if not tf.is_tensor(par) or not par.dtype.is_floating:
        par = tf.cast(par, dtype='float32')

    # Add dimension to scalars
    if not par.shape.as_list():
        par = tf.reshape(par, shape=(1,))

    # Validate shape
    num_par = 6 if ndims == 2 else 12
    shape = par.shape.as_list()
    #print('shape list par',shape)
    if shape[-1] > num_par:
        raise ValueError(f'Number of params exceeds value {num_par} expected for dimensionality.')

    # Set defaults if incomplete and split by type
    width = np.zeros((len(shape), 2), dtype=np.int32)
    splits = (2, 1) * 2 if ndims == 2 else (3,) * 4
    for i in (2, 3, 4):
        width[-1, -1] = max(sum(splits[:i]) - shape[-1], 0)
        default = 1. if i == 3 and not shift_scale else 0.
        par = tf.pad(par, paddings=width, constant_values=default)
        shape = par.shape.as_list()
    shift, rot, scale, shear = tf.split(par, num_or_size_splits=splits, axis=-1)

    # Construct shear matrix
    s = tf.split(shear, num_or_size_splits=splits[-1], axis=-1)
    one, zero = tf.ones_like(s[0]), tf.zeros_like(s[0])
    if ndims == 2:
        mat_shear = tf.stack((
            tf.concat([one, s[0]], axis=-1),
            tf.concat([zero, one], axis=-1),
        ), axis=-2)
    else:
        mat_shear = tf.stack((
            tf.concat([one, s[0], s[1]], axis=-1),
            tf.concat([zero, one, s[2]], axis=-1),
            tf.concat([zero, zero, one], axis=-1),
        ), axis=-2)

    mat_scale = tf.linalg.diag(scale + 1. if shift_scale else scale)
    mat_rot = angles_to_rotation_matrix(rot, deg=deg, ndims=ndims)
    out = tf.matmul(mat_rot, tf.matmul(mat_scale, mat_shear))

    # Append translations
    shift = tf.expand_dims(shift, axis=-1)
    out = tf.concat((out, shift), axis=-1)

    # Append last row: store shapes as tensors to support batched inputs
    if last_row:
        shape_batch = tf.shape(shift)[:-2]
        shape_zeros = tf.concat((shape_batch, (1,), splits[:1]), axis=0)
        zeros = tf.zeros(shape_zeros, dtype=shift.dtype)
        shape_one = tf.concat((shape_batch, (1,), (1,)), axis=0)
        one = tf.ones(shape_one, dtype=shift.dtype)
        row = tf.concat((zeros, one), axis=-1)
        out = tf.concat([out, row], axis=-2)


    return tf.squeeze(out) if len(shape) < 2 else out


def angles_to_rotation_matrix(ang, deg=True, ndims=3):
    """
    Construct N-dimensional rotation matrices from angles, where N is 2 or 3. The direction of
    rotation for all axes follows the right-hand rule. The rotations are intrinsic, i.e. carried
    out in the body-centered frame of reference. Supports batched inputs.
    Arguments:
        ang: Input angles as a scalar, NumPy array, TensorFlow tensor, or list or tuple of these.
            Elements of lists and tuples will be stacked along the last dimension, which
            corresponds to the rotation axes (x, y, z in 3D), and its size must not exceed N.
            If the size is less than N, the missing angles will be set to zero.
        deg: Whether the input angles are specified in degrees. Defaults to True.
        ndims: Dimensionality of rotation matrices. Must be 2 or 3. Defaults to 3.
    Returns:
        ND rotation matrices as a (..., N, N) tensor.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), in press, 2021
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')

    if isinstance(ang, (list, tuple)):
        ang = tf.stack(ang, axis=-1)

    if not tf.is_tensor(ang) or not ang.dtype.is_floating:
        ang = tf.cast(ang, dtype='float32')

    # Add dimension to scalars
    if not ang.shape.as_list():
        ang = tf.reshape(ang, shape=(1,))

    # Validate shape
    num_ang = 1 if ndims == 2 else 3
    shape = ang.shape.as_list()
    if shape[-1] > num_ang:
        raise ValueError(f'Number of angles exceeds value {num_ang} expected for dimensionality.')

    # Set missing angles to zero
    width = np.zeros((len(shape), 2), dtype=np.int32)
    width[-1, -1] = max(num_ang - shape[-1], 0)
    ang = tf.pad(ang, paddings=width)

    # Compute sine and cosine
    if deg:
        ang *= np.pi / 180
    c = tf.split(tf.cos(ang), num_or_size_splits=num_ang, axis=-1)
    s = tf.split(tf.sin(ang), num_or_size_splits=num_ang, axis=-1)

    # Construct matrices
    if ndims == 2:
        out = tf.stack((
            tf.concat([c[0], -s[0]], axis=-1),
            tf.concat([s[0], c[0]], axis=-1),
        ), axis=-2)

    else:
        one, zero = tf.ones_like(c[0]), tf.zeros_like(c[0])
        rot_x = tf.stack((
            tf.concat([one, zero, zero], axis=-1),
            tf.concat([zero, c[0], -s[0]], axis=-1),
            tf.concat([zero, s[0], c[0]], axis=-1),
        ), axis=-2)
        rot_y = tf.stack((
            tf.concat([c[1], zero, s[1]], axis=-1),
            tf.concat([zero, one, zero], axis=-1),
            tf.concat([-s[1], zero, c[1]], axis=-1),
        ), axis=-2)
        rot_z = tf.stack((
            tf.concat([c[2], -s[2], zero], axis=-1),
            tf.concat([s[2], c[2], zero], axis=-1),
            tf.concat([zero, zero, one], axis=-1),
        ), axis=-2)
        out = tf.matmul(rot_x, tf.matmul(rot_y, rot_z))

    return tf.squeeze(out) if len(shape) < 2 else out

#%% Interpolation function from voxelmorph
def voxmorph_transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow
    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.
    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = voxmorph_volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return voxmorph_interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)



#%% Obtain an affine matrix from the outputs of the layer
class ParamsToAffineMatrix(tf.keras.layers.Layer):
    """
    Constructs an affine transformation matrix from translation, rotation, scaling and shearing
    parameters in 2D or 3D.
    If you find this layer useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), in press, 2021
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self, ndims=3, deg=True, shift_scale=True, last_row=False, **kwargs):
        """
        Parameters:
            ndims: Dimensionality of transform matrices. Must be 2 or 3.
            deg: Whether the input rotations are specified in degrees.
            shift_scale: Add 1 to any specified scaling parameters. This may be desirable
                when the parameters are estimated by a network.
            last_row: Whether to return a full matrix, including the last row.
        """
        self.ndims = ndims
        self.deg = deg
        self.shift_scale = shift_scale #SHIFT SCALE IS TREU TO HAVE 1 ON THE DIAGONAL
        self.last_row = last_row
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ndims': self.ndims,
            'deg': self.deg,
            'shift_scale': self.shift_scale,
            'last_row': self.last_row,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims + int(self.last_row), self.ndims + 1)

    def call(self, params):
        """
        Parameters:
            params: Parameters as a vector which corresponds to translations, rotations, scaling
                    and shear. The size of the last axis must not exceed (N, N+1), for N
                    dimensions. If the size is less than that, the missing parameters will be
                    set to the identity.
                    
        """
        #params is passed as list [translation, rotations]
        concat_params = tf.concat([tf.squeeze(params[0],axis=[1,2,3]),
                                   tf.squeeze(params[1],axis=[1,2,3])],axis=-1)
        
        
        return params_to_affine_matrix(par=concat_params,
                                             deg=self.deg,
                                             shift_scale=self.shift_scale,
                                             ndims=self.ndims,
                                             last_row=self.last_row)

#%% Obtain an "affine warp" from the affine matrix
class AffineToDenseShift(tf.keras.layers.Layer):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
        """
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'ndims': self.ndims,
            'shift_center': self.shift_center,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)

    def build(self, input_shape):
        validate_affine_shape(input_shape)

    def call(self, matrix):
        """
        Parameters:
            matrix: Affine matrix of shape [B, N, N+1].
        """
        single = lambda mat: affine_to_dense_shift(mat, self.shape,
                                                         shift_center=self.shift_center)
        return tf.map_fn(single, matrix)
    
    
#%% Interpolator
class SpatialTransformer(tf.keras.layers.Layer):
    """
    ND spatial transformer layer
    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.
    If you find this layer useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network.
    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):
        """
        Parameters: 
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
        """
        self.interp_method = interp_method
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = is_affine_shape(input_shape[1][1:])
        #print(self.is_affine)

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # convert affine matrix to warp field
        if self.is_affine:
            fun = lambda x: affine_to_dense_shift(x, vol.shape[1:-1],
                                                        shift_center=self.shift_center,
                                                        indexing=self.indexing)
            trf = tf.map_fn(fun, trf)


        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            a = tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)
            return a#tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        #print(inputs[0].shape, inputs[1].shape)
        return voxmorph_transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)
    
  
    
    
#%% LAYERS FOR VTN-based NETWORK

class AffineFlow(tf.keras.layers.Layer):
    
    def __init__(self,name='AfffineFlow',**kwargs):
        super(AffineFlow, self).__init__(name=name)
        
    def call(self, inputs):
        
        W, b = inputs
        # tf.print(W)
        # tf.print(b)
        output = self._affine_flow(W, b)

        return output



    def _affine_flow(self,W,b):
        
        """
        Function to transform the W rotation parameters and b displacement
        parameters in an affine flow to apply to the movable image
        """

        #print(W.shape,b.shape)
        #dims = [el.astype('float') for el in dims]
        len1, len2, len3 = [128.0, 128.0, 128.0]
        #W, b, input_pair = inputs
        #len1, len2, len3 = input_pair.shape.as_list()[1:4]
        W = tf.reshape(W, [-1, 3, 3]) 
        #b = tf.reshape(b, [-1, 3])# * self.flow_multiplier
        #b = tf.reshape(b, [-1, 1, 1, 1, 3])
        #print(W.shape,b.shape)
        xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
        xr = tf.reshape(xr, [1, -1, 1, 1, 1])
        yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
        yr = tf.reshape(yr, [1, 1, -1, 1, 1])
        zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
        zr = tf.reshape(zr, [1, 1, 1, -1, 1])
        wx = W[:, :, 0]
        wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
        wy = W[:, :, 1]
        wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
        wz = W[:, :, 2]
        wz = tf.reshape(wz, [-1, 1, 1, 1, 3])

        return (xr * wx + yr * wy) + (zr * wz + b)
    
    
class Dense3DSpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, padding = False, name='padded_output',**kwargs):
        self.padding = padding
        #self.name='padded_output'
        super(Dense3DSpatialTransformer, self).__init__(name=name)

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of length 2 or 3. '
                            'First argument is the image, second is the offset field.')

        if len(input_shape[1]) != 5 or input_shape[1][4] != 3:
            raise Exception('Offset field must be one 5D tensor with 3 channels. '
                            'Got: ' + str(input_shape[1]))

        self.built = True
        
    def get_config(self):
        return {"padding": self.padding}

    def call(self, inputs):
        return self._transform(inputs[0], inputs[1][:, :, :, :, 1],
                               inputs[1][:, :, :, :, 0], inputs[1][:, :, :, :, 2])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _transform(self, I, dx, dy, dz):
        #print('Input:',tf.reduce_mean(I))

        batch_size = tf.shape(dx)[0]
        height = tf.shape(dx)[1]
        width = tf.shape(dx)[2]
        depth = tf.shape(dx)[3]

        # Convert dx and dy to absolute locations
        x_mesh, y_mesh, z_mesh = self._meshgrid(height, width, depth)
        x_mesh = tf.expand_dims(x_mesh, 0)
        y_mesh = tf.expand_dims(y_mesh, 0)
        z_mesh = tf.expand_dims(z_mesh, 0)

        x_mesh = tf.tile(x_mesh, [batch_size, 1, 1, 1])
        y_mesh = tf.tile(y_mesh, [batch_size, 1, 1, 1])
        z_mesh = tf.tile(z_mesh, [batch_size, 1, 1, 1])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, dtype='int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(self, height, width, depth):
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                                tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                                   tf.cast(height, tf.float32)-1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t = tf.tile(tf.expand_dims(x_t, 2), [1, 1, depth])
        y_t = tf.tile(tf.expand_dims(y_t, 2), [1, 1, depth])

        z_t = tf.linspace(0.0, tf.cast(depth, tf.float32)-1.0, depth)
        z_t = tf.expand_dims(tf.expand_dims(z_t, 0), 0)
        z_t = tf.tile(z_t, [height, width, 1])

        return x_t, y_t, z_t

    def _interpolate(self, im, x, y, z):
        if self.padding:
            im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        
        #print(tf.reduce_mean(im))
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        depth = tf.shape(im)[3]
        channels = im.get_shape().as_list()[4]

        out_height = tf.shape(x)[1]
        out_width = tf.shape(x)[2]
        out_depth = tf.shape(x)[3]

        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        z = tf.reshape(z, [-1])

        padding_constant = 1 if self.padding else 0
        x = tf.cast(x, 'float32') + padding_constant
        y = tf.cast(y, 'float32') + padding_constant
        z = tf.cast(z, 'float32') + padding_constant

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        max_z = tf.cast(depth - 1, 'int32')

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1

        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self._repeat(tf.range(num_batch)*dim1,
                            out_height*out_width*out_depth)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        Ie = tf.gather(im_flat, idx_e)
        If = tf.gather(im_flat, idx_f)
        Ig = tf.gather(im_flat, idx_g)
        Ih = tf.gather(im_flat, idx_h)

        # and finally calculate interpolated values
        x1_f = tf.cast(x1, 'float32')
        y1_f = tf.cast(y1, 'float32')
        z1_f = tf.cast(z1, 'float32')

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = tf.expand_dims((dz * dx * dy), 1)
        wb = tf.expand_dims((dz * dx * (1-dy)), 1)
        wc = tf.expand_dims((dz * (1-dx) * dy), 1)
        wd = tf.expand_dims((dz * (1-dx) * (1-dy)), 1)
        we = tf.expand_dims(((1-dz) * dx * dy), 1)
        wf = tf.expand_dims(((1-dz) * dx * (1-dy)), 1)
        wg = tf.expand_dims(((1-dz) * (1-dx) * dy), 1)
        wh = tf.expand_dims(((1-dz) * (1-dx) * (1-dy)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,
                           we*Ie, wf*If, wg*Ig, wh*Ih])
        output = tf.reshape(output, tf.stack(
            [-1, out_height, out_width, out_depth, channels]))
        #print('Ouptut:',tf.reduce_mean(output))
        return output
