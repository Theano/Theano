"""
Convolution-like operations with sparse matrix multiplication.

To read about different sparse formats, see
U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""
# COPIED FROM hpu/icml09/sp.py
from __future__ import absolute_import, print_function, division
import numpy as np
import scipy
from scipy import sparse as scipy_sparse
from six.moves import xrange

import theano
import theano.sparse
from theano import sparse, gof, Op, tensor
from theano.sparse.basic import Remove0, remove0

# To maintain compatibility
from theano.sparse import (
    SpSum, sp_sum,
    ColScaleCSC, RowScaleCSC, col_scale, row_scale,
    Diag, diag, SquareDiagonal, square_diagonal,
    EnsureSortedIndices, ensure_sorted_indices, clean)


def register_specialize(lopt, *tags, **kwargs):
    theano.compile.optdb['specialize'].register(
        (kwargs and kwargs.pop('name')) or lopt.__name__, lopt, 'fast_run',
        *tags)


class ConvolutionIndices(Op):
    """Build indices for a sparse CSC matrix that could implement A
    (convolve) B.

       This generates a sparse matrix M, which generates a stack of
       image patches when computing the dot product of M with image
       patch. Convolution is then simply the dot product of (img x M)
       and the kernels.
    """
    __props__ = ()

    @staticmethod
    def conv_eval(inshp, kshp, strides=(1, 1), mode='valid'):
        (dx, dy) = strides
        return convolution_indices.evaluate(inshp, kshp, (dx, dy),
                                            mode=mode, ws=True)

    # img_shape and ker_shape are (height,width)
    @staticmethod
    def evaluate(inshp, kshp, strides=(1, 1), nkern=1, mode='valid', ws=True):
        """Build a sparse matrix which can be used for performing...
        * convolution: in this case, the dot product of this matrix
        with the input images will generate a stack of images
        patches. Convolution is then a tensordot operation of the
        filters and the patch stack.
        * sparse local connections: in this case, the sparse matrix
        allows us to operate the weight matrix as if it were
        fully-connected. The structured-dot with the input image gives
        the output for the following layer.

        :param ker_shape: shape of kernel to apply (smaller than image)
        :param img_shape: shape of input images
        :param mode: 'valid' generates output only when kernel and
                     image overlap overlap fully. Convolution obtained
                     by zero-padding the input
        :param ws: must be always True
        :param (dx,dy): offset parameter. In the case of no weight sharing,
                        gives the pixel offset between two receptive fields.
                        With weight sharing gives the offset between the
                        top-left pixels of the generated patches

        :rtype: tuple(indices, indptr, logical_shape, sp_type, out_img_shp)
        :returns: the structure of a sparse matrix, and the logical dimensions
                  of the image which will be the result of filtering.
        """
        if not ws:
            raise Exception("ws is obsolete and it must be always True")

        (dx, dy) = strides

        # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
        # in the first case, default nfeatures to 1
        if np.size(inshp) == 2:
            inshp = (1,) + inshp

        inshp = np.array(inshp)
        kshp = np.array(kshp)
        ksize = np.prod(kshp)

        kern = ksize - 1 - np.arange(ksize)

        # size of output image if doing proper convolution
        # (mode='full',dx=dy=0) outshp is the actual output shape
        # given the parameters
        fulloutshp = inshp[1:] + kshp - 1
        if mode == 'valid':
            s = -1
        else:
            s = 1
        outshp = np.int64(np.ceil((inshp[1:] + s * kshp - s * 1) \
                 / np.array([dy, dx], dtype='float')))
        if any(outshp <= 0):
            err = 'Invalid kernel', kshp, 'and/or step size', (dx, dy),\
                  'for given input shape', inshp
            raise ValueError(err)

        outsize = np.prod(outshp)
        insize = np.prod(inshp)

        # range of output units over which to iterate
        if mode == 'valid':
            lbound = np.array([kshp[0] - 1, kshp[1] - 1])
            ubound = lbound + (inshp[1:] - kshp + 1)
        else:
            lbound = np.zeros(2)
            ubound = fulloutshp

        # coordinates of image in "fulloutshp" coordinates
        topleft = np.array([kshp[0] - 1, kshp[1] - 1])
        # bound when counting the receptive field
        botright = topleft + inshp[1:]

        # sparse matrix specifics...
        if ws:
            spmatshp = (outsize * np.prod(kshp) * inshp[0], insize)
        else:
            spmatshp = (nkern * outsize, insize)
        spmat = scipy_sparse.lil_matrix(spmatshp)

        # loop over output image pixels
        z, zz = 0, 0

        # incremented every time we write something to the sparse
        # matrix this is used to track the ordering of filter tap
        # coefficient in sparse column ordering
        tapi, ntaps = 0, 0

        # Note: looping over the number of kernels could've been done
        # more efficiently as the last step (when writing to
        # spmat). However, this messes up the ordering of the column
        # values (order in which you write the values determines how
        # the vectorized data will get used later one)

        for fmapi in xrange(inshp[0]):  # loop over input features
            # loop over number of kernels (nkern=1 for weight sharing)
            for n in xrange(nkern):

                # FOR EACH OUTPUT PIXEL...
                # loop over output image height
                for oy in np.arange(lbound[0], ubound[0], dy):
                     # loop over output image width
                    for ox in np.arange(lbound[1], ubound[1], dx):

                       # kern[l] is filter value to apply at (oj,oi)
                       # for (iy,ix)
                        l = 0

                        # ... ITERATE OVER INPUT UNITS IN RECEPTIVE FIELD
                        for ky in oy + np.arange(kshp[0]):
                            for kx in ox + np.arange(kshp[1]):

                                # verify if we are still within image
                                # boundaries. Equivalent to
                                # zero-padding of the input image
                                if (all((ky, kx) >= topleft) and
                                    all((ky, kx) < botright)):

                                    # convert to "valid" input space
                                    # coords used to determine column
                                    # index to write to in sparse mat
                                    iy, ix = np.array((ky, kx)) - topleft
                                    # determine raster-index of input pixel...

                                    # taking into account multiple
                                    # input features
                                    col = iy * inshp[2] + ix + \
                                          fmapi * np.prod(inshp[1:])

                                    # convert oy,ox values to output
                                    # space coordinates
                                    if mode == 'full':
                                        (y, x) = (oy, ox)
                                    else:
                                        (y, x) = (oy, ox) - topleft
                                    # taking into account step size
                                    (y, x) = np.array([y, x]) / (dy, dx)

                                    # convert to row index of sparse matrix
                                    if ws:
                                        row = ((y * outshp[1] + x) *
                                               inshp[0] * ksize + l + fmapi *
                                               ksize)
                                    else:
                                        row = y * outshp[1] + x

                                    # Store something at that location
                                    # in sparse matrix.  The written
                                    # value is only useful for the
                                    # sparse case. It will determine
                                    # the way kernel taps are mapped
                                    # onto the sparse columns (idea of
                                    # kernel map)
                                    # n*... only for sparse
                                    spmat[row + n * outsize, col] = tapi + 1

                                    # total number of active taps
                                    # (used for kmap)
                                    ntaps += 1

                                # absolute tap index (total number of taps)
                                tapi += 1
                                # move on to next filter tap l=(l+1)%ksize
                                l += 1

        if spmat.format != 'csc':
            spmat = spmat.tocsc().sorted_indices()
        else:
            # BUG ALERT: scipy0.6 has bug where data and indices are written in
            # reverse column ordering.
            # Explicit call to sorted_indices removes this problem.
            spmat = spmat.sorted_indices()

        if ws:
            kmap = None
        else:
            kmap = np.zeros(ntaps, dtype='int')
            k = 0
            # print 'TEMPORARY BUGFIX: REMOVE !!!'
            for j in xrange(spmat.shape[1]):
                for i_idx in xrange(spmat.indptr[j], spmat.indptr[j + 1]):
                    if spmat.data[i_idx] != 0:
                        # this is == spmat[i,j] - 1
                        kmap[k] = spmat.data[i_idx] - 1
                        k += 1

        # when in valid mode, it is more efficient to store in sparse row
        # TODO: need to implement structured dot for csr matrix
        assert spmat.format == 'csc'
        sptype = 'csc'
        #sptype = 'csr' if mode=='valid' else 'csc'
        if 0 and mode == 'valid':
            spmat = spmat.tocsr()

        rval = (spmat.indices[:spmat.size],
                spmat.indptr, spmatshp, sptype, outshp)
        if kmap is not None:
            rval += (kmap,)

        return rval

    def perform(self, node, inputs, outputs):
        (inshp, kshp) = inputs
        (out_indices, out_indptr, spmat_shape) = outputs
        indices, indptr, spmatshp, outshp = self.evaluate(inshp, kshp)
        out_indices[0] = indices
        out_indptr[0] = indptr
        spmat_shape[0] = np.asarray(spmatshp)

convolution_indices = ConvolutionIndices()


def convolve(kerns, kshp, nkern, images, imgshp, step=(1, 1), bias=None,
             mode='valid', flatten=True):
    """Convolution implementation by sparse matrix multiplication.

    :note: For best speed, put the matrix which you expect to be
           smaller as the 'kernel' argument

    "images" is assumed to be a matrix of shape batch_size x img_size,
    where the second dimension represents each image in raster order

    If flatten is "False", the output feature map will have shape:

    .. code-block:: python

        batch_size x number of kernels x output_size

    If flatten is "True", the output feature map will have shape:

    .. code-block:: python

        batch_size x number of kernels * output_size

    .. note::

        IMPORTANT: note that this means that each feature map (image
        generate by each kernel) is contiguous in memory. The memory
        layout will therefore be: [ <feature_map_0> <feature_map_1>
        ... <feature_map_n>], where <feature_map> represents a
        "feature map" in raster order

    kerns is a 2D tensor of shape nkern x N.prod(kshp)

    :param kerns: 2D tensor containing kernels which are applied at every pixel
    :param kshp: tuple containing actual dimensions of kernel (not symbolic)
    :param nkern: number of kernels/filters to apply.
                  nkern=1 will apply one common filter to all input pixels
    :param images: tensor containing images on which to apply convolution
    :param imgshp: tuple containing image dimensions
    :param step: determines number of pixels between adjacent receptive fields
                 (tuple containing dx,dy values)
    :param mode: 'full', 'valid' see CSM.evaluate function for details
    :param sumdims: dimensions over which to sum for the tensordot operation.
                    By default ((2,),(1,)) assumes kerns is a nkern x kernsize
                    matrix and images is a batchsize x imgsize matrix
                    containing flattened images in raster order
    :param flatten: flatten the last 2 dimensions of the output. By default,
                    instead of generating a batchsize x outsize x nkern tensor,
                    will flatten to batchsize x outsize*nkern

    :return: out1, symbolic result
    :return: out2, logical shape of the output img (nkern,heigt,width)

    :TODO: test for 1D and think of how to do n-d convolutions
    """
    # start by computing output dimensions, size, etc
    kern_size = np.int64(np.prod(kshp))

    # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if np.size(imgshp) == 2:
        imgshp = (1,) + imgshp

    # construct indices and index pointers for sparse matrix, which,
    # when multiplied with input images will generate a stack of image
    # patches
    indices, indptr, spmat_shape, sptype, outshp = \
            convolution_indices.conv_eval(imgshp, kshp, step, mode)

    # build sparse matrix, then generate stack of image patches
    csc = theano.sparse.CSM(sptype)(np.ones(indices.size), indices,
                                    indptr, spmat_shape)
    patches = (sparse.structured_dot(csc, images.T)).T

    # compute output of linear classifier
    pshape = tensor.stack([images.shape[0] * tensor.as_tensor(np.prod(outshp)),\
                           tensor.as_tensor(imgshp[0] * kern_size)])
    patch_stack = tensor.reshape(patches, pshape, ndim=2)

    # kern is of shape: nkern x ksize*number_of_input_features
    # output is thus of shape: bsize*outshp x nkern
    output = tensor.dot(patch_stack, kerns.T)

    # add bias across each feature map (more efficient to do it now)
    if bias is not None:
        output += bias

    # now to have feature maps in raster order ...
    # go from bsize*outshp x nkern to bsize x nkern*outshp
    newshp = tensor.stack([images.shape[0],\
                           tensor.as_tensor(np.prod(outshp)),\
                           tensor.as_tensor(nkern)])
    tensout = tensor.reshape(output, newshp, ndim=3)
    output = tensor.DimShuffle((False,) * tensout.ndim, (0, 2, 1))(tensout)
    if flatten:
        output = tensor.flatten(output, 2)

    return output, np.hstack((nkern, outshp))


def max_pool(images, imgshp, maxpoolshp):
    """Implements a max pooling layer

    Takes as input a 2D tensor of shape batch_size x img_size and
    performs max pooling.  Max pooling downsamples by taking the max
    value in a given area, here defined by maxpoolshp. Outputs a 2D
    tensor of shape batch_size x output_size.

    :param images: 2D tensor containing images on which to apply convolution.
                   Assumed to be of shape batch_size x img_size
    :param imgshp: tuple containing image dimensions
    :param maxpoolshp: tuple containing shape of area to max pool over

    :return: out1, symbolic result (2D tensor)
    :return: out2, logical shape of the output
    """
    poolsize = np.int64(np.prod(maxpoolshp))

    # imgshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if np.size(imgshp) == 2:
        imgshp = (1,) + imgshp

    # construct indices and index pointers for sparse matrix, which,
    # when multiplied with input images will generate a stack of image
    # patches
    indices, indptr, spmat_shape, sptype, outshp = \
            convolution_indices.conv_eval(imgshp, maxpoolshp,
                                          maxpoolshp, mode='valid')

#    print 'XXXXXXXXXXXXXXXX MAX POOLING LAYER XXXXXXXXXXXXXXXXXXXX'
#    print 'imgshp = ', imgshp
#    print 'maxpoolshp = ', maxpoolshp
#    print 'outshp = ', outshp

    # build sparse matrix, then generate stack of image patches
    csc = theano.sparse.CSM(sptype)(np.ones(indices.size), indices,
                                    indptr, spmat_shape)
    patches = sparse.structured_dot(csc, images.T).T

    pshape = tensor.stack([images.shape[0] *\
                               tensor.as_tensor(np.prod(outshp)),
                           tensor.as_tensor(imgshp[0]),
                           tensor.as_tensor(poolsize)])
    patch_stack = tensor.reshape(patches, pshape, ndim=3)

    out1 = tensor.max(patch_stack, axis=2)

    pshape = tensor.stack([images.shape[0],
                           tensor.as_tensor(np.prod(outshp)),
                           tensor.as_tensor(imgshp[0])])
    out2 = tensor.reshape(out1, pshape, ndim=3)

    out3 = tensor.DimShuffle(out2.broadcastable, (0, 2, 1))(out2)

    return tensor.flatten(out3, 2), outshp
