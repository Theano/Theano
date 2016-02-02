"""
Contains a wrapper function for tensor.nnet.ConvOp, which can be used to perform
generic 2D convolution.

"""
from __future__ import absolute_import, print_function, division
import warnings

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv

import logging

__docformat__ = "restructuredtext en"


_logger = logging.getLogger("theano.tensor.signal.conv")


def conv2d(input, filters, image_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), **kargs):
    """
    signal.conv.conv2d performs a basic 2D convolution of the input with the
    given filters. The input parameter can be a single 2D image or a 3D tensor,
    containing a set of images. Similarly, filters can be a single 2D filter or
    a 3D tensor, corresponding to a set of 2D filters.

    Shape parameters are optional and will result in faster execution.

    Parameters
    ----------
    input : dmatrix of dtensor3
        Symbolic variable for images to be filtered.
    filters : dmatrix of dtensor3
        Symbolic variable containing filter values.
    border_mode: {'valid', 'full'}
        See scipy.signal.convolve2d.
    subsample
        Factor by which to subsample output.
    image_shape : tuple of length 2 or 3
        ([number images,] image height, image width).
    filter_shape : tuple of length 2 or 3
        ([number filters,] filter height, filter width).
    kwargs
        See theano.tensor.nnet.conv.conv2d.

    Returns
    -------
    symbolic 2D,3D or 4D tensor
        Tensor of filtered images, with shape
        ([number images,] [number filters,] image height, image width).

    """
    assert input.ndim in (2, 3)
    assert filters.ndim in (2, 3)

    # use shape information if it is given to us ###
    if filter_shape and image_shape:
        if input.ndim == 3:
            bsize = image_shape[0]
        else:
            bsize = 1
        imshp = (1,) + tuple(image_shape[-2:])

        if filters.ndim == 3:
            nkern = filter_shape[0]
        else:
            nkern = 1
        kshp = filter_shape[-2:]
    else:
        nkern, kshp = None, None
        bsize, imshp = None, None

    # reshape tensors to 4D, for compatibility with ConvOp ###
    if input.ndim == 3:
        sym_bsize = input.shape[0]
    else:
        sym_bsize = 1

    if filters.ndim == 3:
        sym_nkern = filters.shape[0]
    else:
        sym_nkern = 1

    new_input_shape = tensor.join(0, tensor.stack([sym_bsize, 1]), input.shape[-2:])
    input4D = tensor.reshape(input, new_input_shape, ndim=4)

    new_filter_shape = tensor.join(0, tensor.stack([sym_nkern, 1]), filters.shape[-2:])
    filters4D = tensor.reshape(filters, new_filter_shape, ndim=4)

    # perform actual convolution ###
    op = conv.ConvOp(output_mode=border_mode,
                     dx=subsample[0], dy=subsample[1],
                     imshp=imshp, kshp=kshp, nkern=nkern, bsize=bsize, **kargs)

    output = op(input4D, filters4D)

    # flatten to 3D tensor if convolving with single filter or single image
    if input.ndim == 2 and filters.ndim == 2:
        if theano.config.warn.signal_conv2d_interface:
            warnings.warn(
                "theano.tensor.signal.conv2d() now outputs a 2d tensor when both"
                " inputs are 2d. To disable this warning, set the Theano flag"
                " warn.signal_conv2d_interface to False",
                stacklevel=3)

        output = tensor.flatten(output.T, outdim=2).T
    elif input.ndim == 2 or filters.ndim == 2:
        output = tensor.flatten(output.T, outdim=3).T

    return output
