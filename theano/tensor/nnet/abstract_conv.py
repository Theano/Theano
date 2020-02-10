"""
Abstract conv interface
"""
from __future__ import absolute_import, print_function, division

import logging
from six import reraise, integer_types
import sys
try:
    from math import gcd
except ImportError:
    from fractions import gcd

import theano

from theano.tensor import as_tensor_variable, patternbroadcast
from theano.tensor import get_scalar_constant_value, NotScalarConstantError
from theano.tensor.opt import Assert
from theano.gof import Apply, Op

from six.moves import xrange

import warnings
import numpy as np

try:
    from scipy.signal.signaltools import _valfrommode, _bvalfromboundary, convolve
    from scipy.signal.sigtools import _convolve2d
    imported_scipy_signal = True
except ImportError:
    imported_scipy_signal = False


__docformat__ = "restructuredtext en"
_logger = logging.getLogger("theano.tensor.nnet.abstract_conv")


def get_conv_output_shape(image_shape, kernel_shape,
                          border_mode, subsample,
                          filter_dilation=None):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: tuple of int (symbolic or numeric) corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. For a normal convolution, its four (for 2D convolution)
        or five (for 3D convolution) elements must correspond respectively to :
        number of output channels, number of input channels, height and width
        (and possibly depth) of the kernel.
        For an unshared 2D convolution, its six channels must correspond to :
        number of output channels, height and width of the output, number of
        input channels, height and width of the kernel.
        None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric) or pairs of ints. If it is a string, it must be 'valid',
        'half' or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis. For asymmetric padding, provide a pair of ints for each dimension.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        espectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and width axis.
    Note - The shape of the convolution output does not depend on the 'unshared'
        or the 'num_groups' parameters.

    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image. None where undefined.

    """
    bsize, imshp = image_shape[0], image_shape[2:]

    convdim = len(image_shape) - 2
    nkern, kshp = kernel_shape[0], kernel_shape[-convdim:]

    if filter_dilation is None:
        filter_dilation = np.ones(len(subsample), dtype='int')

    if isinstance(border_mode, tuple):
        out_shp = tuple(get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode[i],
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    else:
        out_shp = tuple(get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode,
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    return (bsize, nkern) + out_shp


# filter dilation set by default to 1
# for compatibility with other tests.
def get_conv_shape_1axis(image_shape, kernel_shape, border_mode,
                         subsample, dilation=1):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    border_mode: string, int or tuple of 2 ints. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis. If it is a tuple, its two elements
        must correspond to the asymmetric padding (e.g., left and right) on
        the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis. None if undefined.

    """
    if None in [image_shape, kernel_shape, border_mode,
                subsample, dilation]:
        return None
    # Implicit dilated kernel shape
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "half":
        pad_l = pad_r = dil_kernel_shape // 2
    elif border_mode == "full":
        pad_l = pad_r = dil_kernel_shape - 1
    elif border_mode == "valid":
        pad_l = pad_r = 0
    else:
        if isinstance(border_mode, tuple):
            pad_l, pad_r = border_mode
        else:
            pad_l = pad_r = border_mode
        if pad_l < 0 or pad_r < 0:
            raise ValueError("border_mode must be >= 0")

    # In case of symbolic shape, we want to build the smallest graph
    # (image_shape + 2 * pad - dil_kernel_shape) // subsample + 1
    out_shp = (image_shape - dil_kernel_shape)
    if pad_l != 0:
        out_shp += pad_l
    if pad_r != 0:
        out_shp += pad_r
    if subsample != 1:
        out_shp = out_shp // subsample
    out_shp = out_shp + 1

    return out_shp


def get_conv_gradweights_shape(image_shape, top_shape,
                               border_mode, subsample,
                               filter_dilation=None,
                               num_groups=1, unshared=False):
    """
    This function tries to compute the kernel shape of convolution gradWeights.

    The weights shape can only be computed exactly when subsample is 1 and
    border_mode is not 'half'. If subsample is not 1 or border_mode is 'half',
    this function will return None.

    Parameters
    ----------
    image_shape: tuple of int corresponding to the input image shape. Its
        four (or five) elements must correspond respectively to: batch size,
        number of output channels, height and width of the image. None where
        undefined.
    top_shape: tuple of int (symbolic or numeric) corresponding to the top
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of output channels, height and width (and
        possibly depth) of the image. None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric) or pairs of ints. If it is a string, it must be 'valid',
        'half' or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis. For asymmetric padding, provide a pair of ints for each dimension.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        respectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and
        width axis.
    num_groups: An int which specifies the number of separate groups to
        be divided into.
    unshared: Boolean value. If true, unshared convolution will be performed,
        where a different filter is applied to each area of the input.

    Returns
    -------
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. Its four (or five) elements correspond respectively
        to: number of output channels, number of input channels, height and
        width (and possibly depth) of the kernel. None where undefined.

    """
    nkern, imshp = image_shape[1], image_shape[2:]
    nchan, topshp = top_shape[1], top_shape[2:]

    if filter_dilation is None:
        filter_dilation = np.ones(len(subsample), dtype='int')
    if num_groups > 1:
        nchan = nchan // num_groups

    if isinstance(border_mode, tuple):
        out_shp = tuple(get_conv_gradweights_shape_1axis(
            imshp[i], topshp[i], border_mode[i],
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    else:
        out_shp = tuple(get_conv_gradweights_shape_1axis(
            imshp[i], topshp[i], border_mode,
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    if unshared:
        return (nchan,) + top_shape[2:] + (nkern,) + out_shp
    else:
        return (nchan, nkern) + out_shp


def get_conv_gradweights_shape_1axis(image_shape, top_shape, border_mode,
                                     subsample, dilation):
    """
    This function tries to compute the image shape of convolution gradWeights.

    The weights shape can only be computed exactly when subsample is 1 and
    border_mode is not 'half'. If subsample is not 1 or border_mode is 'half',
    this function will return None.

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    top_shape: int or None. Corresponds to the top shape on a given axis.
        None if undefined.
    border_mode: string, int or tuple of 2 ints. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis. If it is a tuple, its two elements
        must correspond to the asymmetric padding (e.g., left and right) on
        the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.

    """
    if None in [image_shape, top_shape, border_mode,
                subsample, dilation]:
        return None
    if subsample != 1 or border_mode == "half":
        return None

    if border_mode == "full":
        kernel_shape = top_shape - image_shape
    elif border_mode == "valid":
        kernel_shape = image_shape - top_shape
    else:
        if isinstance(border_mode, tuple):
            pad_l, pad_r = border_mode
        else:
            pad_l = pad_r = border_mode
        if pad_l < 0 or pad_r < 0:
            raise ValueError("border_mode must be >= 0")

        kernel_shape = (image_shape + pad_l + pad_r - top_shape)

    if dilation > 1:
        kernel_shape = kernel_shape / dilation

    return kernel_shape + 1


def get_conv_gradinputs_shape(kernel_shape, top_shape,
                              border_mode, subsample,
                              filter_dilation=None,
                              num_groups=1):
    """
    This function tries to compute the image shape of convolution gradInputs.

    The image shape can only be computed exactly when subsample is 1.
    If subsample for a dimension is not 1, this function will return None for
    that dimension.

    Parameters
    ----------
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. Its four (or five) elements must correspond respectively
        to: number of output channels, number of input channels, height and
        width (and possibly depth) of the kernel. None where undefined.
    top_shape: tuple of int (symbolic or numeric) corresponding to the top
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of output channels, height and width (and
        possibly depth) of the image. None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric) or pairs of ints. If it is a string, it must be 'valid',
        'half' or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis. For asymmetric padding, provide a pair of ints for each dimension.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        respectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and
        width axis.
    num_groups: An int which specifies the number of separate groups to
        be divided into.
    Note - The shape of the convolution output does not depend on the 'unshared'
        parameter.

    Returns
    -------
    image_shape: tuple of int corresponding to the input image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image. None where undefined.

    """
    bsize, topshp = top_shape[0], top_shape[2:]

    convdim = len(top_shape) - 2
    nkern, kshp = kernel_shape[1], kernel_shape[-convdim:]

    if filter_dilation is None:
        filter_dilation = np.ones(len(subsample), dtype='int')
    if num_groups > 1:
        nkern = nkern * num_groups

    if isinstance(border_mode, tuple):
        out_shp = tuple(get_conv_gradinputs_shape_1axis(
            kshp[i], topshp[i], border_mode[i],
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    else:
        out_shp = tuple(get_conv_gradinputs_shape_1axis(
            kshp[i], topshp[i], border_mode,
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    return (bsize, nkern) + out_shp


def get_conv_gradinputs_shape_1axis(kernel_shape, top_shape, border_mode,
                                    subsample, dilation):
    """
    This function tries to compute the image shape of convolution gradInputs.

    The image shape can only be computed exactly when subsample is 1.
    If subsample is not 1, this function will return None.

    Parameters
    ----------
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    top_shape: int or None. Corresponds to the top shape on a given axis.
        None if undefined.
    border_mode: string, int or tuple of 2 ints. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis. If it is a tuple, its two elements
        must correspond to the asymmetric padding (e.g., left and right) on
        the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.

    """
    if None in [kernel_shape, top_shape, border_mode,
                subsample, dilation]:
        return None
    if subsample != 1:
        return None

    # Implicit dilated kernel shape
    dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "half":
        pad_l = pad_r = dil_kernel_shape // 2
    elif border_mode == "full":
        pad_l = pad_r = dil_kernel_shape - 1
    elif border_mode == "valid":
        pad_l = pad_r = 0
    else:
        if isinstance(border_mode, tuple):
            pad_l, pad_r = border_mode
        else:
            pad_l = pad_r = border_mode
        if pad_l < 0 or pad_r < 0:
            raise ValueError("border_mode must be >= 0")

    # In case of symbolic shape, we want to build the smallest graph
    # image_shape = (top_shape - 1) * s - 2 * pad + dil_kernel_shape + a
    # where 0 <= a < subsample, but we have checked that subsample == 1
    image_shape = (top_shape + dil_kernel_shape - 1)
    if pad_l > 0:
        image_shape -= pad_l
    if pad_r > 0:
        image_shape -= pad_r

    return image_shape


def check_conv_gradinputs_shape(image_shape, kernel_shape, output_shape,
                                border_mode, subsample,
                                filter_dilation=None):
    """
    This function checks if the given image shapes are consistent.

    Parameters
    ----------
    image_shape: tuple of int (symbolic or numeric) corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. Its four (or five) elements must correspond respectively
        to: number of output channels, number of input channels, height and
        width (and possibly depth) of the kernel. None where undefined.
    output_shape: tuple of int (symbolic or numeric) corresponding to the
        output shape. Its four (or five) elements must correspond respectively
        to: batch size, number of output channels, height and width
        (and possibly depth) of the output. None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric) or pairs of ints. If it is a string, it must be 'valid',
        'half' or 'full'. If it is a tuple, its two (or three) elements respectively
        correspond to the padding on height and width (and possibly depth)
        axis. For asymmetric padding, provide a pair of ints for each dimension.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        respectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    filter_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and
        width axis.

    Returns
    -------
    Returns False if a convolution with the given input shape, kernel shape
    and parameters would not have produced the given output shape.

    Returns True in all other cases: if the given output shape matches the
    computed output shape, but also if the shape could not be checked because
    because the shape contains symbolic values.

    """
    image_shape = tuple(image_shape)
    kernel_shape = tuple(kernel_shape)
    output_shape = tuple(output_shape)

    if len(image_shape) != len(kernel_shape) or len(image_shape) != len(output_shape):
        return False
    if len(image_shape) - 2 != len(subsample):
        return False
    if filter_dilation is not None and len(image_shape) - 2 != len(filter_dilation):
        return False

    # compute the predicted output shape
    computed_output_shape = get_conv_output_shape(
        image_shape, kernel_shape, border_mode, subsample, filter_dilation)

    # check if the given output shape matches the computed shape
    def check_dim(given, computed):
        if given is None or computed is None:
            return True
        try:
            given = get_scalar_constant_value(given)
            computed = get_scalar_constant_value(computed)
            return int(given) == int(computed)
        except NotScalarConstantError:
            # no answer possible, accept for now
            return True

    return all(check_dim(given, computed)
               for (given, computed) in zip(output_shape, computed_output_shape))


def assert_conv_shape(shape):
    """This function adds Assert nodes that check if shape is a valid convolution shape.

    The first two dimensions should be larger than or equal to zero. The convolution
    dimensions should be larger than zero.

    Parameters
    ----------
    shape: tuple of int (symbolic or numeric) corresponding to the input, output or
        kernel shape of a convolution. For input and output, the first elements should
        should be the batch size and number of channels. For kernels, the first and
        second elements should contain the number of input and output channels.
        The remaining dimensions are the convolution dimensions.

    Returns
    -------
    Returns a tuple similar to the given `shape`. For constant elements in `shape`,
    the function checks the value and raises a `ValueError` if the dimension is invalid.
    The elements that are not constant are wrapped in an `Assert` op that checks the
    dimension at run time.
    """
    out_shape = []
    for i, n in enumerate(shape):
        try:
            const_n = get_scalar_constant_value(n)
            if i < 2:
                if const_n < 0:
                    raise ValueError('The convolution would produce an invalid shape (dim[%d]: %d < 0).' % (i, const_n))
            else:
                if const_n <= 0:
                    raise ValueError('The convolution would produce an invalid shape (dim[%d]: %d <= 0).' % (i, const_n))
            out_shape.append(n)
        except NotScalarConstantError:
            if i < 2:
                assert_shp = Assert('The convolution would produce an invalid shape (dim[%d] < 0).' % i)
                out_shape.append(assert_shp(n, theano.tensor.ge(n, 0)))
            else:
                assert_shp = Assert('The convolution would produce an invalid shape (dim[%d] <= 0).' % i)
                out_shape.append(assert_shp(n, theano.tensor.gt(n, 0)))
    return tuple(out_shape)


def assert_shape(x, expected_shape, msg='Unexpected shape.'):
    """Wraps `x` in an `Assert` to check its shape.

    Parameters
    ----------
    x : Tensor
        x will be wrapped in an `Assert`.
    expected_shape : tuple or list
        The expected shape of `x`. The size of a dimension can be None,
        which means it will not be checked.
    msg : str
        The error message of the `Assert`.

    Returns
    -------
    Tensor
        `x` wrapped in an `Assert`. At execution time, this will throw an
        AssertionError if the shape of `x` does not match `expected_shape`.
        If `expected_shape` is None or contains only Nones, the function
        will return `x` directly.

    """
    if expected_shape is None or not theano.config.conv.assert_shape:
        return x
    shape = x.shape
    tests = []
    for i in range(x.ndim):
        if expected_shape[i] is not None:
            tests.append(theano.tensor.eq(shape[i], expected_shape[i]))
    if tests:
        return Assert(msg)(x, *tests)
    else:
        return x


def border_mode_to_pad(mode, convdim, kshp):
    """
    Computes a tuple for padding given the border_mode parameter

    Parameters
    ----------
    mode : int or tuple
        One of "valid", "full", "half", an integer, or a tuple where each
        member is either an integer or a tuple of 2 positive integers.
    convdim : int
        The dimensionality of the convolution.
    kshp : List/tuple of length 'convdim', indicating the size of the
        kernel in the spatial dimensions.

    Returns
    -------
    A tuple containing 'convdim' elements, each of which is a tuple of
    two positive integers corresponding to the padding on the left
    and the right sides respectively.

    """

    if isinstance(mode, tuple):
        if len(mode) != convdim:
            raise ValueError(
                'invalid border_mode {} which must be a '
                'tuple of length {}'.format(mode, convdim))
        border = ()
        for m in mode:
            if isinstance(m, integer_types) and m >= 0:
                border += ((m, m),)
            elif isinstance(m, tuple) and min(m) >= 0 and \
                    all(isinstance(b, integer_types) for b in m):
                if len(m) != 2:
                    raise NotImplementedError(
                        'Asymmetric padding not implemented '
                        'for {}d'.format(len(m)))
                border += ((m[0], m[1]),)
            else:
                raise ValueError(
                    'invalid border mode {}. The tuple can only contain '
                    'integers or tuples of length 2'.format(mode))
        pad = border
    elif mode == 'full':
        pad = tuple((kshp[i] - 1,) * 2 for i in range(convdim))
    elif mode == 'half':
        pad = tuple((kshp[i] // 2,) * 2 for i in range(convdim))
    elif mode == 'valid':
        pad = ((0, 0),) * convdim
    else:
        raise ValueError(
            'invalid border_mode {}, which must be either '
            '"valid", "full", "half", an integer or a tuple '
            'of length {}'.format(mode, convdim))
    return pad


def conv2d(input,
           filters,
           input_shape=None,
           filter_shape=None,
           border_mode='valid',
           subsample=(1, 1),
           filter_flip=True,
           filter_dilation=(1, 1),
           num_groups=1,
           unshared=False):
    """This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).

    Refer to :func:`nnet.conv2d <theano.tensor.nnet.conv2d>` for a more detailed documentation.
    """

    input = as_tensor_variable(input)
    filters = as_tensor_variable(filters)
    conv_op = AbstractConv2d(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation,
                             num_groups=num_groups,
                             unshared=unshared)
    return conv_op(input, filters)


def separable_conv2d(input,
                     depthwise_filters,
                     pointwise_filters,
                     num_channels,
                     input_shape=None,
                     depthwise_filter_shape=None,
                     pointwise_filter_shape=None,
                     border_mode='valid',
                     subsample=(1, 1),
                     filter_flip=True,
                     filter_dilation=(1, 1)):
    """
    This function will build the symbolic graph for depthwise
    convolutions which act separately on the input channels followed by
    pointwise convolution which mixes channels.

    Parameters
    ----------
    input: symbolic 4D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input rows, input columns).
        See the optional parameter ``input_shape``.

    depthwise_filters: symbolic 4D tensor
        Set of filters used depthwise convolution layer of shape
        (depthwise output channels, 1, filter rows, filter columns).

    pointwise_filters: symbolic 4D tensor
        Set of filters used pointwise convolution layer of shape
        (output channels, depthwise output channels, 1, 1).

    num_channels: int
        The number of channels of the input. Required for depthwise
        convolutions.

    input_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    depthwise_filter_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the depthwise filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    pointwise_filter_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the pointwise filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    border_mode: str, int or tuple of two int
        This applies only to depthwise convolutions
        Either of the following:

        ``'valid'``: apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter shape + 1
        ``'full'``: apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter rows // 2``
            rows and ``filter columns // 2`` columns, then perform a valid
            convolution. For filters with an odd number of rows and columns, this
            leads to the output shape being equal to the input shape.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2)``: pad input with a symmetric border of ``int1`` rows
            and ``int2`` columns, then perform a valid convolution.
        ``(int1, (int2, int3))`` or ``((int1, int2), int3)``:
            pad input with one symmetric border of `int1`` or ``int3``, and
            one asymmetric border of ``(int2, int3)`` or ``(int1, int2)``.
        ``((int1, int2), (int3, int4))``: pad input with an asymmetric
            border of ``(int1, int2)`` along one dimension and ``(int3, int4)``
            along the second dimension.

    subsample: tuple of len 2
        Factor by which to subsample the output.
        This applies only to depthwise convolutions

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a cross-correlation.

    filter_dilation: tuple of len 2
        Factor by which to subsample (stride) the input.
        This applies only to depthwise convolutions

    Returns
    -------
    Symbolic 4D tensor
        Set of feature maps generated by convolutional layer. Tensor is
        of shape (batch size, output channels, output rows, output columns)

"""

    input = as_tensor_variable(input)
    depthwise_filters = as_tensor_variable(depthwise_filters)
    conv_op = AbstractConv2d(imshp=input_shape,
                             kshp=depthwise_filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation,
                             num_groups=num_channels)

    if input_shape is None or depthwise_filter_shape is None:
        depthwise_op_shape = None
    else:
        depthwise_op_shape = conv_op.infer_shape(None, [input_shape, depthwise_filter_shape])[0]
    depthwise_op = conv_op(input, depthwise_filters)

    pointwise_op = conv2d(input=depthwise_op,
                          filters=pointwise_filters,
                          input_shape=depthwise_op_shape,
                          filter_shape=pointwise_filter_shape,
                          border_mode='valid',
                          subsample=(1, 1),
                          filter_flip=filter_flip,
                          filter_dilation=(1, 1),
                          num_groups=1)
    return pointwise_op


def separable_conv3d(input,
                     depthwise_filters,
                     pointwise_filters,
                     num_channels,
                     input_shape=None,
                     depthwise_filter_shape=None,
                     pointwise_filter_shape=None,
                     border_mode='valid',
                     subsample=(1, 1, 1),
                     filter_flip=True,
                     filter_dilation=(1, 1, 1)):
    """
    This function will build the symbolic graph for depthwise
    convolutions which act separately on the input channels followed by
    pointwise convolution which mixes channels.

    Parameters
    ----------
    input: symbolic 5D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input depth, input rows, input columns).
        See the optional parameter ``input_shape``.

    depthwise_filters: symbolic 5D tensor
        Set of filters used depthwise convolution layer of shape
        (depthwise output channels, 1, filter_depth, filter rows, filter columns).

    pointwise_filters: symbolic 5D tensor
        Set of filters used pointwise convolution layer of shape
        (output channels, depthwise output channels, 1, 1, 1).

    num_channels: int
        The number of channels of the input. Required for depthwise
        convolutions.

    input_shape: None, tuple/list of len 5 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    depthwise_filter_shape: None, tuple/list of len 5 of int or Constant variable
        The shape of the depthwise filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    pointwise_filter_shape: None, tuple/list of len 5 of int or Constant variable
        The shape of the pointwise filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    border_mode: str, int or tuple of three int
        This applies only to depthwise convolutions
        Either of the following:

        ``'valid'``: apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter shape + 1
        ``'full'``: apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter // 2``,
            then perform a valid convolution. For filters with an odd
            number of slices, rows and columns, this leads to the output
            shape being equal to the input shape.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2, int3)``
            pad input with a symmetric border of ``int1``, ``int2`` and
            ``int3`` columns, then perform a valid convolution.

    subsample: tuple of len 3
        This applies only to depthwise convolutions
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter x, y and z dimensions before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.

    filter_dilation: tuple of len 3
        Factor by which to subsample (stride) the input.
        Also called dilation elsewhere.

    Returns
    -------
    Symbolic 5D tensor
        Set of feature maps generated by convolutional layer. Tensor is
        of shape (batch size, output channels, output_depth,
        output rows, output columns)

    """

    input = as_tensor_variable(input)
    depthwise_filters = as_tensor_variable(depthwise_filters)
    conv_op = AbstractConv3d(imshp=input_shape,
                             kshp=depthwise_filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation,
                             num_groups=num_channels)

    if input_shape is None or depthwise_filter_shape is None:
        depthwise_op_shape = None
    else:
        depthwise_op_shape = conv_op.infer_shape(None, [input_shape, depthwise_filter_shape])[0]
    depthwise_op = conv_op(input, depthwise_filters)

    pointwise_op = conv3d(input=depthwise_op,
                          filters=pointwise_filters,
                          input_shape=depthwise_op_shape,
                          filter_shape=pointwise_filter_shape,
                          border_mode='valid',
                          subsample=(1, 1, 1),
                          filter_flip=filter_flip,
                          filter_dilation=(1, 1, 1),
                          num_groups=1)
    return pointwise_op


def conv3d(input,
           filters,
           input_shape=None,
           filter_shape=None,
           border_mode='valid',
           subsample=(1, 1, 1),
           filter_flip=True,
           filter_dilation=(1, 1, 1),
           num_groups=1):
    """
    This function will build the symbolic graph for convolving a mini-batch of a
    stack of 3D inputs with a set of 3D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).


    Parameters
    ----------
    input: symbolic 5D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input depth, input rows, input columns).
        See the optional parameter ``input_shape``.

    filters: symbolic 5D tensor
        Set of filters used in CNN layer of shape
        (output channels, input channels, filter depth, filter rows, filter columns).
        See the optional parameter ``filter_shape``.

    input_shape: None, tuple/list of len 5 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    filter_shape: None, tuple/list of len 5 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    border_mode: str, int or tuple of three int
        Either of the following:

        ``'valid'``: apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter shape + 1
        ``'full'``: apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter // 2``,
            then perform a valid convolution. For filters with an odd
            number of slices, rows and columns, this leads to the output
            shape being equal to the input shape.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2, int3)``
            pad input with a symmetric border of ``int1``, ``int2`` and
            ``int3`` columns, then perform a valid convolution.

    subsample: tuple of len 3
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter x, y and z dimensions before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.

    filter_dilation: tuple of len 3
        Factor by which to subsample (stride) the input.
        Also called dilation elsewhere.

    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    Returns
    -------
    Symbolic 5D tensor
        Set of feature maps generated by convolutional layer. Tensor is
        is of shape (batch size, output channels, output depth,
        output rows, output columns)

    Notes
    -----
        If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
        "caffe style convolution".

        This is only supported in Theano 0.8 or the development
        version until it is released.

    """
    input = as_tensor_variable(input)
    filters = as_tensor_variable(filters)
    conv_op = AbstractConv3d(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation,
                             num_groups=num_groups)
    return conv_op(input, filters)


def conv2d_grad_wrt_inputs(output_grad,
                           filters,
                           input_shape,
                           filter_shape=None,
                           border_mode='valid',
                           subsample=(1, 1),
                           filter_flip=True,
                           filter_dilation=(1, 1),
                           num_groups=1,
                           unshared=False):
    """Compute conv output gradient w.r.t its inputs

    This function builds the symbolic graph for getting the
    gradient of the output of a convolution (namely output_grad)
    w.r.t the input of the convolution, given a set of 2D filters
    used by the convolution, such that the output_grad is upsampled
    to the input_shape.

    Parameters
    ----------
    output_grad : symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input rows, input columns).  This is the tensor that
        will be upsampled or the output gradient of the convolution
        whose gradient will be taken with respect to the input of the
        convolution.
    filters: symbolic 4D or 6D tensor
        Set of filters used in CNN layer of shape
        (output channels, input channels, filter rows, filter columns)
        for normal convolution and
        (output channels, output rows, output columns, input channels,
        filter rows, filter columns)
        for unshared convolution.
        See the optional parameter ``filter_shape``.
    input_shape : [None/int/Constant] * 2 + [Tensor/int/Constant] * 2
        The shape of the input (upsampled) parameter.
        A tuple/list of len 4, with the first two dimensions
        being None or int or Constant and the last two dimensions being
        Tensor or int or Constant.
        Not Optional, since given the output_grad shape
        and the subsample values, multiple input_shape may be
        plausible.
    filter_shape : None or [None/int/Constant] * (4 or 6)
        The shape of the filters parameter. None or a tuple/list of len 4 or a
        tuple/list of len 6 (for unshared convolution)
        Optional, possibly used  to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that
        this element is not known at compile time.
    border_mode: str, int or a tuple of two ints or pairs of ints
        Either of the following:

          ``'valid'``
            apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter
            shape + 1

          ``'full'``
            apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1

          ``'half'``
            pad input with a symmetric border of ``filter rows // 2``
            rows and ``filter columns // 2`` columns, then perform a
            valid convolution. For filters with an odd number of rows
            and columns, this leads to the output shape being equal to
            the input shape. It is known as 'same' elsewhere.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2)``
            pad input with a symmetric border of ``int1`` rows and
            ``int2`` columns, then perform a valid convolution.

          ``(int1, (int2, int3))`` or ``((int1, int2), int3)``
            pad input with one symmetric border of `int1`` or ``int3``, and
            one asymmetric border of ``(int2, int3)`` or ``(int1, int2)``.

          ``((int1, int2), (int3, int4))``
            pad input with an asymmetric border of ``(int1, int2)`` along one dimension and ``(int3, int4)``
            along the second dimension.

    subsample : tuple of len 2
        The subsampling used in the forward pass.  Also called strides
        elsewhere.
    filter_flip : bool
        If ``True``, will flip the filter rows and columns before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.
    filter_dilation : tuple of len 2
        The filter dilation used in the forward pass.
        Also known as input striding.
    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately
    unshared: bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.

    Returns
    -------
    symbolic 4D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output rows, output
        columns)

    Notes
    -----

    :note: If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    filters = as_tensor_variable(filters)
    output_grad = as_tensor_variable(output_grad)

    # checking the type of input_shape
    for dim in [0, 1]:
        if not isinstance(input_shape[dim], (theano.tensor.TensorConstant,
                                             integer_types, type(None))):
            raise ValueError("input_shape[%d] must be a constant or None." %
                             dim)
    for dim in [2, 3]:
        if not isinstance(input_shape[dim], (theano.tensor.TensorVariable,
                                             theano.tensor.TensorConstant,
                                             integer_types)):
            raise ValueError("input_shape[%d] must be a symbolic variable,"
                             " a constant or None." %
                             dim)

    # checking the type of filter_shape
    if filter_shape is not None:
        if unshared:
            expected_dim = 6
        else:
            expected_dim = 4

        if len(filter_shape) != expected_dim:
            raise ValueError(
                "The lenght of filter_shape was %d, but we expected %d." % (
                    len(filter_shape), expected_dim))

        for dim in range(expected_dim):
            if not isinstance(filter_shape[dim], (theano.tensor.TensorConstant,
                                                  integer_types, type(None))):
                raise ValueError(
                    "filter_shape[%d] must be a constant or None" % dim)

    # setting the last two dimensions of input_shape to None, if
    # the type of these dimensions is TensorVariable.
    numerical_input_shape = list(input_shape)
    for dim in [2, 3]:
        if isinstance(input_shape[dim], theano.tensor.TensorVariable):
            numerical_input_shape[dim] = None

    grad_input_op = AbstractConv2d_gradInputs(imshp=numerical_input_shape,
                                              kshp=filter_shape,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              filter_flip=filter_flip,
                                              filter_dilation=filter_dilation,
                                              num_groups=num_groups,
                                              unshared=unshared)

    return grad_input_op(filters, output_grad, input_shape[-2:])


def conv3d_grad_wrt_inputs(output_grad,
                           filters,
                           input_shape,
                           filter_shape=None,
                           border_mode='valid',
                           subsample=(1, 1, 1),
                           filter_flip=True,
                           filter_dilation=(1, 1, 1),
                           num_groups=1):
    """Compute conv output gradient w.r.t its inputs

    This function builds the symbolic graph for getting the
    gradient of the output of a convolution (namely output_grad)
    w.r.t the input of the convolution, given a set of 3D filters
    used by the convolution, such that the output_grad is upsampled
    to the input_shape.

    Parameters
    ----------
    output_grad : symbolic 5D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input depth, input rows, input columns).  This is the
        tensor that will be upsampled or the output gradient of the
        convolution whose gradient will be taken with respect to the
        input of the convolution.
    filters : symbolic 5D tensor
        set of filters used in CNN layer of shape (output channels,
        input channels, filter depth, filter rows, filter columns).
        See the optional parameter ``filter_shape``.
    input_shape : [None/int/Constant] * 2 + [Tensor/int/Constant] * 2
        The shape of the input (upsampled) parameter.
        A tuple/list of len 5, with the first two dimensions
        being None or int or Constant and the last three dimensions being
        Tensor or int or Constant.
        Not Optional, since given the output_grad shape
        and the subsample values, multiple input_shape may be
        plausible.
    filter_shape : None or [None/int/Constant] * 5
        The shape of the filters parameter. None or a tuple/list of len 5.
        Optional, possibly used  to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that
        this element is not known at compile time.
    border_mode : str, int or tuple of three int
        Either of the following:

          ``'valid'``
            apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter
            shape + 1

          ``'full'``
            apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1

          ``'half'``
            pad input with a symmetric border of ``filter // 2``,
            then perform a valid convolution. For filters with an odd
            number of slices, rows and columns, this leads to the output
            shape being equal to the input shape. It is known as 'same'
            elsewhere.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2, int3)``
            pad input with a symmetric border of ``int1``, ``int2`` and
            ``int3`` columns, then perform a valid convolution.

    subsample : tuple of len 3
        The subsampling used in the forward pass.  Also called strides
        elsewhere.
    filter_flip : bool
        If ``True``, will flip the filter x, y and z dimensions before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.
    filter_dilation : tuple of len 3
        The filter dilation used in the forward pass.
        Also known as input striding.
    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    Returns
    -------
    symbolic 5D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output depth,
        output rows, output columns)

    Notes
    -----

    :note: If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    filters = as_tensor_variable(filters)
    output_grad = as_tensor_variable(output_grad)

    # checking the type of input_shape
    for dim in [0, 1]:
        assert isinstance(input_shape[dim], (theano.tensor.TensorConstant,
                                             integer_types, type(None)))
    for dim in [2, 3, 4]:
        assert isinstance(input_shape[dim], (theano.tensor.TensorVariable,
                                             theano.tensor.TensorConstant,
                                             integer_types))

    # checking the type of filter_shape
    if filter_shape is not None:
        for dim in [0, 1, 2, 3, 4]:
            assert isinstance(filter_shape[dim], (theano.tensor.TensorConstant,
                                                  integer_types, type(None)))

    # setting the last three dimensions of input_shape to None, if
    # the type of these dimensions is TensorVariable.
    numerical_input_shape = list(input_shape)
    for dim in [2, 3, 4]:
        if isinstance(input_shape[dim], theano.tensor.TensorVariable):
            numerical_input_shape[dim] = None

    grad_input_op = AbstractConv3d_gradInputs(imshp=numerical_input_shape,
                                              kshp=filter_shape,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              filter_flip=filter_flip,
                                              filter_dilation=filter_dilation,
                                              num_groups=num_groups)

    return grad_input_op(filters, output_grad, input_shape[-3:])


def conv2d_grad_wrt_weights(input,
                            output_grad,
                            filter_shape,
                            input_shape=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            filter_flip=True,
                            filter_dilation=(1, 1),
                            num_groups=1,
                            unshared=False):
    """Compute conv output gradient w.r.t its weights

    This function will build the symbolic graph for getting the
    gradient of the output of a convolution (output_grad) w.r.t its wights.

    Parameters
    ----------
    input : symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input rows, input columns).  This is the input of
        the convolution in the forward pass.
    output_grad : symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input rows, input columns).  This is the gradient of
        the output of convolution.
    filter_shape : [None/int/Constant] * (2 or 4) + [Tensor/int/Constant] * 2
        The shape of the filter parameter.  A tuple/list of len 4 or 6
        (for unshared), with the first two dimensions being None or int or
        Constant and the last two dimensions being Tensor or int or Constant.
        Not Optional, since given the output_grad shape and
        the input_shape, multiple filter_shape may be plausible.
    input_shape : None or [None/int/Constant] * 4
        The shape of the input parameter. None or a tuple/list of len 4.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify
        that this element is not known at compile time.
    border_mode: str, int or a tuple of two ints or pairs of ints
        Either of the following:

          ``'valid'``
            apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter
            shape + 1

          ``'full'``
            apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1

          ``'half'``
            pad input with a symmetric border of ``filter rows // 2``
            rows and ``filter columns // 2`` columns, then perform a
            valid convolution. For filters with an odd number of rows
            and columns, this leads to the output shape being equal to
            the input shape. It is known as 'same' elsewhere.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2)``
            pad input with a symmetric border of ``int1`` rows and
            ``int2`` columns, then perform a valid convolution.

          ``(int1, (int2, int3))`` or ``((int1, int2), int3)``
            pad input with one symmetric border of `int1`` or ``int3``, and
            one asymmetric border of ``(int2, int3)`` or ``(int1, int2)``.

          ``((int1, int2), (int3, int4))``
            pad input with an asymmetric border of ``(int1, int2)`` along
            one dimension and ``(int3, int4)`` along the second dimension.
    subsample : tuple of len 2
        The subsampling used in the forward pass of the convolutional
        operation.  Also called strides elsewhere.
    filter_flip : bool
        If ``True``, will flip the filter rows and columns before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.
    filter_dilation : tuple of len 2
        The filter dilation used in the forward pass.
        Also known as input striding.
    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately
    unshared: bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.

    Returns
    -------
    symbolic 4D tensor or 6D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output rows, output
        columns) for normal convolution and
        (output channels, output rows, output columns, input channels,
        filter rows, filter columns) for unshared convolution

    Notes
    -----

    :note: If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    input = as_tensor_variable(input)
    output_grad = as_tensor_variable(output_grad)

    # checking the type of filter_shape
    for dim in [0, 1]:
        assert isinstance(filter_shape[dim], (theano.tensor.TensorConstant,
                                              integer_types, type(None)))
    if unshared:
        for dim in [2, 3]:
            assert isinstance(filter_shape[dim], (theano.tensor.TensorConstant,
                                                  integer_types, type(None)))
    for dim in [-2, -1]:
        assert isinstance(filter_shape[dim], (theano.tensor.TensorVariable,
                                              theano.tensor.TensorConstant,
                                              integer_types))

    # checking the type of input_shape
    if input_shape is not None:
        for dim in [0, 1, 2, 3]:
            assert isinstance(input_shape[dim], (theano.tensor.TensorConstant,
                                                 integer_types, type(None)))

    # setting the last two dimensions of filter_shape to None, if
    # the type of these dimensions is TensorVariable.
    numerical_filter_shape = list(filter_shape)
    for dim in [-2, -1]:
        if isinstance(filter_shape[dim], theano.tensor.TensorVariable):
            numerical_filter_shape[dim] = None

    gradWeight_op = AbstractConv2d_gradWeights(imshp=input_shape,
                                               kshp=numerical_filter_shape,
                                               border_mode=border_mode,
                                               subsample=subsample,
                                               filter_flip=filter_flip,
                                               filter_dilation=filter_dilation,
                                               num_groups=num_groups,
                                               unshared=unshared)

    return gradWeight_op(input, output_grad, filter_shape[-2:])


def conv3d_grad_wrt_weights(input,
                            output_grad,
                            filter_shape,
                            input_shape=None,
                            border_mode='valid',
                            subsample=(1, 1, 1),
                            filter_flip=True,
                            filter_dilation=(1, 1, 1),
                            num_groups=1):
    """Compute conv output gradient w.r.t its weights

    This function will build the symbolic graph for getting the
    gradient of the output of a convolution (output_grad) w.r.t its weights.

    Parameters
    ----------
    input : symbolic 5D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input depth, input rows, input columns).  This is the input
        of the convolution in the forward pass.
    output_grad : symbolic 5D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input depth, input rows, input columns).  This is the
        gradient of the output of convolution.
    filter_shape : [None/int/Constant] * 2 + [Tensor/int/Constant] * 2
        The shape of the filter parameter.  A tuple/list of len 5, with the
        first two dimensions being None or int or Constant and the last three
        dimensions being Tensor or int or Constant.
        Not Optional, since given the output_grad shape and
        the input_shape, multiple filter_shape may be plausible.
    input_shape : None or [None/int/Constant] * 5
        The shape of the input parameter. None or a tuple/list of len 5.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify
        that this element is not known at compile time.
    border_mode : str, int or tuple of two ints
        Either of the following:

          ``'valid'``
            apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter
            shape + 1

          ``'full'``
            apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1

          ``'half'``
            pad input with a symmetric border of ``filter rows // 2``
            rows and ``filter columns // 2`` columns, then perform a
            valid convolution. For filters with an odd number of rows
            and columns, this leads to the output shape being equal to
            the input shape. It is known as 'same' elsewhere.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2, int3)``
            pad input with a symmetric border of ``int1``, ``int2`` and
            ``int3``, then perform a valid convolution.
    subsample : tuple of len 3
        The subsampling used in the forward pass of the convolutional
        operation.  Also called strides elsewhere.
    filter_flip : bool
        If ``True``, will flip the filters before sliding them over the
        input. This operation is normally referred to as a convolution,
        and this is the default. If ``False``, the filters are not
        flipped and the operation is referred to as a cross-correlation.
    filter_dilation : tuple of len 3
        The filter dilation used in the forward pass.
        Also known as input striding.
    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    Returns
    -------
    symbolic 5D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output time, output
        rows, output columns)

    Notes
    -----

    :note: If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    input = as_tensor_variable(input)
    output_grad = as_tensor_variable(output_grad)

    # checking the type of filter_shape
    for dim in [0, 1]:
        assert isinstance(filter_shape[dim], (theano.tensor.TensorConstant,
                                              integer_types, type(None)))
    for dim in [2, 3, 4]:
        assert isinstance(filter_shape[dim], (theano.tensor.TensorVariable,
                                              theano.tensor.TensorConstant,
                                              integer_types))

    # checking the type of input_shape
    if input_shape is not None:
        for dim in [0, 1, 2, 3, 4]:
            assert isinstance(input_shape[dim], (theano.tensor.TensorConstant,
                                                 integer_types, type(None)))

    # setting the last three dimensions of filter_shape to None, if
    # the type of these dimensions is TensorVariable.
    numerical_filter_shape = list(filter_shape)
    for dim in [2, 3, 4]:
        if isinstance(filter_shape[dim], theano.tensor.TensorVariable):
            numerical_filter_shape[dim] = None

    gradWeight_op = AbstractConv3d_gradWeights(imshp=input_shape,
                                               kshp=numerical_filter_shape,
                                               border_mode=border_mode,
                                               subsample=subsample,
                                               filter_flip=filter_flip,
                                               filter_dilation=filter_dilation,
                                               num_groups=num_groups)

    return gradWeight_op(input, output_grad, filter_shape[-3:])


def causal_conv1d(input,
                  filters,
                  filter_shape,
                  input_shape=None,
                  subsample=1,
                  filter_flip=True,
                  filter_dilation=1,
                  num_groups=1,
                  unshared=False):
    """
    Computes (dilated) causal convolution

    The output at time t depends only on the inputs till t-1. Used for
    modelling temporal data.
    See [WaveNet: A Generative Model for Raw Audio, section 2.1]
    (https://arxiv.org/abs/1609.03499).

    Parameters
    ----------
    input : symbolic 3D tensor
        mini-batch of feature vector stacks, of shape
        (batch_size, input_channels, input_length)
        See the optional parameter ``input_shape``
    filters : symbolic 3D tensor
        Set of filters used in the CNN, of shape
        (output_channels, input_channels, filter_length)
    filter_shape : [None/int/Constant] * 2 + [Tensor/int/Constant]
        The shape of the filters parameter.
        A tuple/list of len 3, with the first two dimensions
        being None or int or Constant and the last dimension being
        Tensor or int or Constant.
        Not optional, since the filter length is needed to calculate
        the left padding for causality.
    input_shape : None or [None/int/Constant] * 3
        The shape of the input parameter.
        None, or a tuple/list of len 3.
        Optional, possibly used to choose an optimal implementation.
    subsample : int
        The factor by which to subsample the output. Also called strides
        elsewhere.
    filter_dilation : int
        Factor by which to subsample (stride) the input. Also called
        dilation factor.
    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately
    unshared : bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.

    Returns
    -------
    Symbolic 3D tensor.
        Set of feature vectors generated by convolutional layer. Tensor is
        of shape (batch_size, output_channels, output_length)

    Notes
    -----

    :note: Currently, this is implemented with the 2D convolution ops.

    """

    input = as_tensor_variable(input)
    filters = as_tensor_variable(filters)

    if input.ndim != 3:
        raise ValueError('Input should be 3D for causal convolution.')
    if filters.ndim != 3:
        raise ValueError('Filters should be 3D for causal convolution')

    input = input.dimshuffle(0, 1, 2, 'x')
    filters = filters.dimshuffle(0, 1, 2, 'x')

    if input_shape is not None:
        assert(len(input_shape) == 3)
        input_shape = tuple(input_shape)
        input_shape += (1,)

    assert(len(filter_shape) == 3)
    filter_shape = tuple(filter_shape)
    filter_shape += (1,)

    left_pad = filter_dilation * (filter_shape[2] - 1)

    subsample = (subsample, 1)
    filter_dilation = (filter_dilation, 1)

    conv_op = AbstractConv2d(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=((left_pad, 0), 0),
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation,
                             num_groups=num_groups,
                             unshared=unshared)
    output = conv_op(input, filters)

    return output[:, :, :, 0]


def bilinear_kernel_2D(ratio, normalize=True):
    """Compute 2D kernel for bilinear upsampling

    This function builds the 2D kernel that can be used to upsample
    a tensor by the given ratio using bilinear interpolation.

    Parameters
    ----------
    ratio: int or Constant/Scalar Theano tensor of int* dtype
        the ratio by which an image will be upsampled by the returned filter
        in the 2D space.

    normalize: bool
        param normalize: indicates whether to normalize the kernel or not.
        Default is True.

    Returns
    -------
    symbolic 2D tensor
        the 2D kernels that can be applied to any given image to upsample it
        by the indicated ratio using bilinear interpolation in two dimensions.

    """

    if isinstance(ratio, tuple):
        ratio_h = ratio[1]
        ratio_v = ratio[0]
    else:
        ratio_h = ratio
        ratio_v = ratio
    hkern = bilinear_kernel_1D(ratio=ratio_h,
                               normalize=normalize).dimshuffle('x', 0)
    vkern = bilinear_kernel_1D(ratio=ratio_v,
                               normalize=normalize).dimshuffle(0, 'x')
    kern = hkern * vkern
    return kern


def bilinear_kernel_1D(ratio, normalize=True):
    """Compute 1D kernel for bilinear upsampling

    This function builds the 1D kernel that can be used to upsample
    a tensor by the given ratio using bilinear interpolation.

    Parameters
    ----------
    ratio: int or Constant/Scalar Theano tensor of int* dtype
        the ratio by which an image will be upsampled by the returned filter
        in the 2D space.

    normalize: bool
        param normalize: indicates whether to normalize the kernel or not.
        Default is True.

    Returns
    -------
    symbolic 1D tensor
        the 1D kernels that can be applied to any given image to upsample it
        by the indicated ratio using bilinear interpolation in one dimension.

    """

    T = theano.tensor
    half_kern = T.arange(1, ratio + 1, dtype=theano.config.floatX)
    kern = T.concatenate([half_kern, half_kern[-2::-1]])

    if normalize:
        kern /= T.cast(ratio, theano.config.floatX)
    return kern


def frac_bilinear_upsampling(input,
                             frac_ratio):
    """Compute bilinear upsampling
    This function will build the symbolic graph for upsampling
    a tensor by the given ratio using bilinear interpolation.

    Parameters
    ----------
    input: symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size,
        input channels, input rows, input columns) that will be upsampled.
    frac_ratio: tuple of int or tuple of tuples of int
        The tuple defining the fractional ratio by which the input is
        upsampled in the 2D space. One fractional ratio should be
        represented as (numerator, denominator). If row and col ratios are
        different frac_ratio should be a tuple of fractional ratios, i.e
        a tuple of tuples.
    Returns
    -------
    symbolic 4D tensor
        set of feature maps generated by bilinear upsampling. Tensor
        is of shape (batch size, num_input_channels, input row size * row ratio,
        input column size * column ratio). Each of these ratios can be fractional.
    Notes
    -----
    :note: The kernel used for bilinear interpolation is fixed (not learned).
    :note: When the upsampling frac_ratio numerator is even, the
        last row and column is repeated one extra time compared to the first
        row and column which makes the upsampled tensor asymmetrical on both
        sides. This does not happen when it is odd.
    """

    T = theano.tensor
    row, col = input.shape[2:]
    up_input = input.reshape((-1, 1, row, col))

    # define the upsampling ratio depending on the case
    if not isinstance(frac_ratio, tuple):
        raise ValueError("frac_ratio must be a tuple")
    else:
        if isinstance(frac_ratio[0], tuple):
            f_r = []
            for i, fr in enumerate(frac_ratio):
                p, q = fr
                div = gcd(p, q)
                f_r.append(tuple(np.array(fr) // div))
            frac_ratio = tuple(f_r)
            ratio = (frac_ratio[0][0], frac_ratio[1][0])
            subsample = (frac_ratio[0][1], frac_ratio[1][1])
        else:
            p, q = frac_ratio
            div = gcd(p, q)
            frac_ratio = tuple(np.array(frac_ratio) // div)
            ratio = (frac_ratio[0], frac_ratio[0])
            subsample = (frac_ratio[1], frac_ratio[1])

    # duplicate borders of the input
    concat_mat = T.concatenate((up_input[:, :, :1, :], up_input,
                                up_input[:, :, -1:, :]), axis=2)
    concat_mat = T.concatenate((concat_mat[:, :, :, :1], concat_mat,
                                concat_mat[:, :, :, -1:]), axis=3)

    # add padding for the pyramidal kernel
    double_pad = (2 * T.as_tensor([row, col]) - 1) * np.array(ratio) + 1
    pad = double_pad // 2

    # build pyramidal kernel
    kern = bilinear_kernel_2D(ratio=ratio)[np.newaxis, np.newaxis, :, :].astype(theano.config.floatX)

    # add corresponding padding
    pad_kern = T.concatenate((T.zeros(tuple(kern.shape[:2]) + (pad[0], kern.shape[-1]),
                                      dtype=theano.config.floatX),
                              kern,
                              T.zeros(tuple(kern.shape[:2]) + (double_pad[0] - pad[0], kern.shape[-1]),
                                      dtype=theano.config.floatX)),
                             axis=2)
    pad_kern = T.concatenate((T.zeros(tuple(pad_kern.shape[:3]) + (pad[1],), dtype=theano.config.floatX),
                              pad_kern,
                              T.zeros(tuple(pad_kern.shape[:3]) + (double_pad[1] - pad[1],),
                                      dtype=theano.config.floatX)),
                             axis=3)

    # upsample the input by passing it as kernel of conv and using filter_dilation
    upsamp = T.nnet.conv2d(pad_kern, concat_mat, border_mode='valid',
                           filter_dilation=ratio, subsample=subsample)
    return upsamp.reshape((input.shape[0], input.shape[1], upsamp.shape[2], upsamp.shape[3]))


def bilinear_upsampling(input,
                        ratio=None,
                        frac_ratio=None,
                        batch_size=None,
                        num_input_channels=None,
                        use_1D_kernel=True):
    """Compute bilinear upsampling
    This function will build the symbolic graph for upsampling
    a tensor by the given ratio using bilinear interpolation.

    Parameters
    ----------
    input: symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size,
        input channels, input rows, input columns) that will be upsampled.
    ratio: `int or Constant or Scalar Tensor of int* dtype`
        the ratio by which the input is upsampled in the 2D space (row and
        col size).
    frac_ratio: None, tuple of int or tuple of tuples of int
        The tuple defining the fractional ratio by which the input is
        upsampled in the 2D space. One fractional ratio should be
        represented as (numerator, denominator). If row and col ratios are
        different frac_ratio should be a tuple of fractional ratios, i.e
        a tuple of tuples.
    use_1D_kernel: bool
        if set to true, row and column will be upsampled separately by 1D
        kernels, otherwise they are upsampled together using a 2D kernel. The
        final result is the same, only the speed can differ, given factors such
        as upsampling ratio.
    Returns
    -------
    symbolic 4D tensor
        set of feature maps generated by bilinear upsampling. Tensor
        is of shape (batch size, num_input_channels, input row size * row ratio,
        input column size * column ratio). Each of these ratios can be fractional.
    Notes
    -----
    :note: The kernel used for bilinear interpolation is fixed (not learned).
    :note: When the upsampling ratio is even, the last row and column is
        repeated one extra time compared to the first row and column which makes
        the upsampled tensor asymmetrical on both sides. This does not happen when
        the upsampling ratio is odd.
    :note: This function must get either ratio or frac_ratio as parameter and
        never both at once.
    """

    if ratio and frac_ratio:
        raise ValueError("can't use ratio and frac_ratio together")
    if not (ratio or frac_ratio):
        raise ValueError("No ratio (or frac_ratio) provided")

    if frac_ratio:
        if use_1D_kernel:
            raise ValueError('For fractional ratios 1D kernel '
                             'method not implemented. You may want to pass '
                             'use_1D_kernel as False')
        # case of fractional 2D upsampling
        return frac_bilinear_upsampling(input, frac_ratio=frac_ratio)

    # the remaining case if integer ratio with use_1D_kernel
    T = theano.tensor
    try:
        up_bs = batch_size * num_input_channels
    except TypeError:
        up_bs = None
    row, col = input.shape[2:]
    up_input = input.reshape((-1, 1, row, col))

    # concatenating the first and last row and column
    # first and last row
    concat_mat = T.concatenate((up_input[:, :, :1, :], up_input,
                                up_input[:, :, -1:, :]), axis=2)
    # first and last col
    concat_mat = T.concatenate((concat_mat[:, :, :, :1], concat_mat,
                                concat_mat[:, :, :, -1:]), axis=3)
    concat_col = col + 2

    pad = 2 * ratio - (ratio - 1) // 2 - 1

    if use_1D_kernel:
        kern = bilinear_kernel_1D(ratio=ratio, normalize=True)
        # upsampling rows
        upsampled_row = conv2d_grad_wrt_inputs(output_grad=concat_mat,
                                               filters=kern[np.newaxis,
                                                            np.newaxis, :,
                                                            np.newaxis],
                                               input_shape=(up_bs, 1,
                                                            row * ratio,
                                                            concat_col),
                                               filter_shape=(1, 1, None, 1),
                                               border_mode=(pad, 0),
                                               subsample=(ratio, 1),
                                               filter_flip=True,
                                               filter_dilation=(1, 1))
        # upsampling cols
        upsampled_mat = conv2d_grad_wrt_inputs(output_grad=upsampled_row,
                                               filters=kern[np.newaxis,
                                                            np.newaxis,
                                                            np.newaxis, :],
                                               input_shape=(up_bs, 1,
                                                            row * ratio,
                                                            col * ratio),
                                               filter_shape=(1, 1, 1, None),
                                               border_mode=(0, pad),
                                               subsample=(1, ratio),
                                               filter_flip=True,
                                               filter_dilation=(1, 1))
    else:
        kern = bilinear_kernel_2D(ratio=ratio, normalize=True)
        upsampled_mat = conv2d_grad_wrt_inputs(output_grad=concat_mat,
                                               filters=kern[np.newaxis,
                                                            np.newaxis, :, :],
                                               input_shape=(up_bs, 1,
                                                            row * ratio,
                                                            col * ratio),
                                               filter_shape=(1, 1, None, None),
                                               border_mode=(pad, pad),
                                               subsample=(ratio, ratio),
                                               filter_flip=True,
                                               filter_dilation=(1, 1))

    return upsampled_mat.reshape((input.shape[0], input.shape[1],
                                  row * ratio, col * ratio))


class BaseAbstractConv(Op):
    """Base class for AbstractConv

    Parameters
    ----------
     convdim: The number of convolution dimensions (2 or 3).

     imshp: None, tuple/list of len ``(2 + convdim)`` of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        imshp is defined w.r.t the forward conv.

     kshp: None, tuple/list of len ``(2 + convdim)`` or ``(2 + 2 * convdim)``
        (for unshared) of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        kshp is defined w.r.t the forward conv.

    border_mode: str, int or a tuple of two ints or pairs of ints
        Either of the following:

        ``'valid'``: apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter shape + 1
        ``'full'``: apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter size // 2``
            in each convolution dimension, then perform a valid convolution.
            For filters with an odd filter size, this leads to the output
            shape being equal to the input shape.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2)``: (for 2D) pad input with a symmetric border of ``int1``,
            ``int2``, then perform a valid convolution.
        ``(int1, (int2, int3))`` or ``((int1, int2), int3)``: (for 2D)
            pad input with one symmetric border of `int1`` or ``int3``, and
            one asymmetric border of ``(int2, int3)`` or ``(int1, int2)``.
        ``((int1, int2), (int3, int4))``: (for 2D) pad input with an asymmetric
            border of ``(int1, int2)`` along one dimension and ``(int3, int4)``
            along the second dimension.
        ``(int1, int2, int3)``: (for 3D) pad input with a symmetric border of
            ``int1``, ``int2`` and ``int3``, then perform a valid convolution.

    subsample: tuple of len ``convdim``
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a
        cross-correlation.

    filter_dilation: tuple of len ``convdim``
        Factor by which to subsample (stride) the input.
        Also called dilation factor.

    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    unshared: bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.
    """
    check_broadcast = False
    __props__ = ('convdim', 'border_mode', 'subsample', 'filter_flip',
                 'imshp', 'kshp', 'filter_dilation', 'num_groups', 'unshared')

    def __init__(self, convdim,
                 imshp=None, kshp=None, border_mode="valid",
                 subsample=None, filter_flip=True, filter_dilation=None, num_groups=1,
                 unshared=False):

        self.convdim = convdim
        if convdim not in (2, 3):
            raise ValueError(
                'convolution dimension {} is not supported', convdim)

        if subsample is None:
            subsample = (1,) * convdim
        if filter_dilation is None:
            filter_dilation = (1,) * convdim

        if isinstance(border_mode, integer_types):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = (border_mode,) * convdim
        elif isinstance(border_mode, tuple):
            if len(border_mode) != convdim:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'tuple of length {}'.format(border_mode, convdim))
            new_border_mode = ()
            for mode in border_mode:
                if not((isinstance(mode, integer_types) and mode >= 0) or
                        (isinstance(mode, tuple) and len(mode) == 2 and min(mode) >= 0 and
                         all(isinstance(m, integer_types) for m in mode))):
                    raise ValueError(
                        'invalid border mode {}. The tuple can only contain integers '
                        ' or pairs of integers'.format(border_mode))
                if isinstance(mode, tuple):
                    if convdim != 2:
                        raise NotImplementedError(
                            'Asymmetric padding not implemented for {}D'.format(convdim))
                    if mode[0] == mode[1]:
                        mode = mode[0]
                new_border_mode += (mode,)
            border_mode = new_border_mode
        elif border_mode not in ('valid', 'full', 'half'):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a tuple '
                'of length {}'.format(border_mode, convdim))
        if isinstance(border_mode, tuple) and \
                all(mode == (0, 0) or mode == 0 for mode in border_mode):
            border_mode = 'valid'

        self.imshp = tuple(imshp) if imshp else (None,) * (2 + convdim)
        for imshp_i in self.imshp:
            if imshp_i is not None:
                # Components of imshp should be constant or ints
                try:
                    get_scalar_constant_value(imshp_i,
                                              only_process_constants=True)
                except NotScalarConstantError:
                    reraise(ValueError,
                            ValueError("imshp should be None or a tuple of "
                                       "constant int values"),
                            sys.exc_info()[2])
        if kshp:
            self.kshp = tuple(kshp)
        else:
            self.kshp = (None,) * ((2 + 2 * convdim) if unshared else (2 + convdim))
        for kshp_i in self.kshp:
            if kshp_i is not None:
                # Components of kshp should be constant or ints
                try:
                    get_scalar_constant_value(kshp_i,
                                              only_process_constants=True)
                except NotScalarConstantError:
                    reraise(ValueError,
                            ValueError("kshp should be None or a tuple of "
                                       "constant int values"),
                            sys.exc_info()[2])
        self.border_mode = border_mode
        self.filter_flip = filter_flip

        if len(subsample) != convdim:
            raise ValueError("subsample must have {} elements".format(convdim))
        self.subsample = tuple(subsample)
        if len(filter_dilation) != convdim:
            raise ValueError("filter_dilation must have {} elements".format(convdim))
        self.filter_dilation = tuple(filter_dilation)
        if num_groups < 1:
            raise ValueError("num_groups must have value greater than zero")
        self.num_groups = num_groups
        if unshared and self.convdim != 2:
            raise NotImplementedError('Unshared convolution not implemented for %dD'
                                      % self.convdim)
        self.unshared = unshared

    def do_constant_folding(self, node):
        # Disable constant folding since there is no implementation.
        # This may change in the future.
        return False

    def flops(self, inp, outp):
        """ Useful with the hack in profiling to print the MFlops"""
        if self.convdim == 2:
            # if the output shape is correct, then this gives the correct
            # flops for any direction, sampling, padding, and border mode
            inputs, filters = inp
            outputs, = outp
            assert inputs[1] == (filters[1] * self.num_groups)
            # nb mul and add by output pixel
            flops = filters[2] * filters[3] * 2
            # nb flops by output image
            flops *= outputs[2] * outputs[3]
            # nb patch multiplied
            flops *= inputs[1] * filters[0] * inputs[0] / self.num_groups
            return flops
        else:
            # TODO implement for convdim == 3
            raise NotImplementedError(
                'flops not implemented for convdim={}', self.convdim)

    def conv(self, img, kern, mode="valid", dilation=1, num_groups=1, unshared=False, direction="forward"):
        """
        Basic slow Python 2D or 3D convolution for DebugMode
        """
        if not imported_scipy_signal:
            raise NotImplementedError(
                "AbstractConv perform requires the python package"
                " for scipy.signal to be installed.")
        if not (mode in ('valid', 'full')):
            raise ValueError(
                'invalid mode {}, which must be either '
                '"valid" or "full"'.format(mode))
        if isinstance(dilation, integer_types):
            dilation = (dilation,) * self.convdim
        if len(dilation) != self.convdim:
            raise ValueError(
                'invalid dilation {}, expected {} values'.format(dilation,
                                                                 self.convdim))
        if unshared and direction == "backprop weights":
            if mode != "valid":
                raise ValueError('conv mode for unshared backprop wrt weights must be "valid"')
            # To allow the same format for the call to 'unshared2d' for all three directions,
            # the out_shape is shuffled here.
            # We do a transpose in the 'perform' function to bring it to the required shape
            out_shape = (img.shape[0], kern.shape[0],
                         kern.shape[2], kern.shape[3],
                         img.shape[2] - kern.shape[2] + 1,
                         img.shape[3] - kern.shape[3] + 1)
        else:
            out_shape = get_conv_output_shape(img.shape, kern.shape,
                                              mode, [1] * self.convdim, dilation)

        dil_kern_shp = kern.shape[:-self.convdim] + tuple(
            (kern.shape[-self.convdim + i] - 1) * dilation[i] + 1
            for i in range(self.convdim))
        dilated_kern = np.zeros(dil_kern_shp, dtype=kern.dtype)

        dilated_kern[(slice(None),) * (dilated_kern.ndim - self.convdim) +
                     tuple(slice(None, None, dilation[i]) for i in range(self.convdim))
                     ] = kern
        out = np.zeros(out_shape, dtype=img.dtype)

        if img.shape[1] % self.num_groups != 0:
            raise ValueError(
                'number of input channels must be divible by num_groups')
        if kern.shape[0] % self.num_groups != 0:
            raise ValueError(
                'number of filters must be divisible by num_groups')
        if img.shape[1] // num_groups != kern.shape[1]:
            raise ValueError(
                'the number of input channels in the kernel should '
                'specify the number of channels of 1 group')
        input_channel_offset = img.shape[1] // self.num_groups
        output_channel_offset = kern.shape[0] // self.num_groups

        if self.convdim == 2:
            val = _valfrommode(mode)
            bval = _bvalfromboundary('fill')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.ComplexWarning)
                for b in xrange(img.shape[0]):
                    for g in xrange(self.num_groups):
                        for n in xrange(output_channel_offset):
                            for im0 in xrange(input_channel_offset):
                                if unshared:
                                    out[b, g * output_channel_offset + n, ...] += self.unshared2d(img[b, g * input_channel_offset + im0, ...],
                                                                                                  dilated_kern[g * output_channel_offset + n, im0, ...],
                                                                                                  out_shape[2:], direction)
                                else:
                                    # some cast generates a warning here
                                    out[b, g * output_channel_offset + n, ...] += _convolve2d(img[b, g * input_channel_offset + im0, ...],
                                                                                              dilated_kern[g * output_channel_offset + n, im0, ...],
                                                                                              1, val, bval, 0)

        elif self.convdim == 3:
            if unshared:
                raise NotImplementedError('Unshared 3D convolution is not implemented')
            for b in xrange(img.shape[0]):
                for g in xrange(self.num_groups):
                    for n in xrange(output_channel_offset):
                        for im0 in xrange(input_channel_offset):
                            out[b, g * output_channel_offset + n, ...] += convolve(img[b, g * input_channel_offset + im0, ...],
                                                                                   dilated_kern[g * output_channel_offset + n,
                                                                                   im0, ...], mode)
        else:
            raise NotImplementedError('only 2D and 3D convolution are implemented')
        return out

    def unshared2d(self, inp, kern, out_shape, direction="forward"):
        '''
        Basic slow Python unshared 2d convolution.
        '''
        if self.convdim != 2:
            raise NotImplementedError('Unshared convolution not implemented for %dD'
                                      % self.convdim)
        out = np.zeros(out_shape, dtype=inp.dtype)

        if direction == "forward":
            for row in xrange(out_shape[0]):
                for col in xrange(out_shape[1]):
                    out[row, col] = np.sum(np.multiply(inp[row:row + kern.shape[2],
                                                       col:col + kern.shape[3]],
                                           kern[row, col, ::-1, ::-1]))
        elif direction == "backprop weights":
            for row in xrange(out_shape[0]):
                for col in xrange(out_shape[1]):
                    out[row, col, ...] = kern[row, col] * \
                        inp[row:row + out_shape[2], col:col + out_shape[3]]
        elif direction == "backprop inputs":
            for row in xrange(kern.shape[0]):
                for col in xrange(kern.shape[1]):
                    out[row:row + kern.shape[2], col:col + kern.shape[3]] += inp[row, col] * \
                        kern[row, col, ::-1, ::-1]
        else:
            raise ValueError("unshared2d: invalid value '{}' for 'direction'".format(direction))
        return out


class AbstractConv(BaseAbstractConv):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.
    """

    def __init__(self,
                 convdim,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=None,
                 filter_flip=True,
                 filter_dilation=None,
                 num_groups=1,
                 unshared=False):
        super(AbstractConv, self).__init__(convdim=convdim,
                                           imshp=imshp, kshp=kshp,
                                           border_mode=border_mode,
                                           subsample=subsample,
                                           filter_flip=filter_flip,
                                           filter_dilation=filter_dilation,
                                           num_groups=num_groups,
                                           unshared=unshared)

    def make_node(self, img, kern):
        # Make sure both inputs are Variables with the same Type
        if not isinstance(img, theano.Variable):
            img = as_tensor_variable(img)
        if not isinstance(kern, theano.Variable):
            kern = as_tensor_variable(kern)
        ktype = img.type.clone(dtype=kern.dtype,
                               broadcastable=kern.broadcastable)
        kern = ktype.filter_variable(kern)

        if img.type.ndim != 2 + self.convdim:
            raise TypeError('img must be %dD tensor' % (2 + self.convdim))

        if self.unshared:
            if kern.type.ndim != 2 + 2 * self.convdim:
                raise TypeError('kern must be %dD tensor for unshared convolution'
                                % (2 + 2 * self.convdim))
        else:
            if kern.type.ndim != 2 + self.convdim:
                raise TypeError('kern must be %dD tensor' % (2 + self.convdim))

        img = assert_shape(img, self.imshp,
                           'AbstractConv shape mismatch: shape of '
                           'image does not match given imshp.')
        kern = assert_shape(kern, self.kshp,
                            'AbstractConv shape mismatch: shape of '
                            'filters does not match given kshp.')

        broadcastable = [img.broadcastable[0],
                         kern.broadcastable[0]] + ([False] * self.convdim)
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, kern], [output])

    def perform(self, node, inp, out_):
        img, kern = inp
        img = np.asarray(img)
        kern = np.asarray(kern)

        dil_kernshp = tuple((kern.shape[-self.convdim + i] - 1) * self.filter_dilation[i] + 1
                            for i in range(self.convdim))
        if self.unshared and self.convdim != 2:
            raise NotImplementedError('Unshared convolution not implemented for %dD'
                                      % self.convdim)
        o, = out_
        mode = self.border_mode
        pad = border_mode_to_pad(mode, self.convdim, dil_kernshp)

        if any(p != (0, 0) for p in pad):
            mode = "valid"
            new_img = np.zeros((img.shape[0], img.shape[1]) +
                               tuple(img.shape[i + 2] + pad[i][0] + pad[i][1]
                                     for i in range(self.convdim)),
                               dtype=img.dtype)
            new_img[(slice(None), slice(None)) +
                    tuple(slice(pad[i][0], img.shape[i + 2] + pad[i][0])
                          for i in range(self.convdim))] = img
            img = new_img
        if not self.filter_flip:
            kern = kern[(slice(None),) * (kern.ndim - self.convdim) + (slice(None, None, -1),) * self.convdim]

        if self.unshared:
            out_shape = get_conv_output_shape(img.shape, kern.shape,
                                              mode, self.subsample, self.filter_dilation)
            if kern.shape[1:1 + self.convdim] != out_shape[2:2 + self.convdim]:
                raise ValueError('Kernel shape {} does not match '
                                 'computed output size {}'.format(kern.shape[1:1 + self.convdim],
                                                                  out_shape[2:2 + self.convdim]))
            if any(self.subsample[i] > 1 for i in range(self.convdim)):
                # Expand regions in kernel to correct for subsampling
                out_shape = get_conv_output_shape(img.shape, kern.shape,
                                                  mode, (1,) * self.convdim, self.filter_dilation)
                exp_kern_shp = kern.shape[:1] + out_shape[2:2 + self.convdim] + \
                    kern.shape[1 + self.convdim:]
                exp_kern = np.zeros(exp_kern_shp, dtype=kern.dtype)
                exp_kern[(slice(None),) +
                         tuple(slice(None, None, self.subsample[i]) for i in range(self.convdim)) +
                         (slice(None),) * (self.convdim + 1)] = kern
                kern = exp_kern
            # from (nFilters, out_rows, out_cols, nChannels, kH, kW)
            # to (nFilters, nChannels, out_rows, out_cols, kH, kW)
            axes_order = (0, 1 + self.convdim,) + tuple(range(1, 1 + self.convdim)) + \
                tuple(range(2 + self.convdim, kern.ndim))
            kern = kern.transpose(axes_order)

        conv_out = self.conv(img, kern, mode="valid", dilation=self.filter_dilation, num_groups=self.num_groups,
                             unshared=self.unshared)
        conv_out = conv_out[(slice(None), slice(None)) +
                            tuple(slice(None, None, self.subsample[i])
                                  for i in range(self.convdim))]
        o[0] = node.outputs[0].type.filter(conv_out)

    def R_op(self, inputs, eval_points):
        if self.num_groups > 1:
            raise NotImplementedError(
                'Rop not implemented for grouped convolutions')
        if self.unshared:
            raise NotImplementedError('Rop not implemented for unshared convolution')
        rval = None
        if eval_points[0] is not None:
            rval = self.make_node(eval_points[0], inputs[1]).outputs[0]
        if eval_points[1] is not None:
            if rval is None:
                rval = self.make_node(inputs[0], eval_points[1]).outputs[0]
            else:
                rval += self.make_node(inputs[0], eval_points[1]).outputs[0]
        return [rval]

    def infer_shape(self, node, input_shapes):
        imshp = input_shapes[0]
        kshp = input_shapes[1]

        # replace symbolic shapes with known constant shapes
        if self.imshp is not None:
            imshp = [imshp[i] if self.imshp[i] is None else self.imshp[i]
                     for i in range(2 + self.convdim)]
        if self.kshp is not None:
            if self.unshared:
                kshp = [kshp[i] if self.kshp[i] is None else self.kshp[i]
                        for i in range(2 + 2 * self.convdim)]
            else:
                kshp = [kshp[i] if self.kshp[i] is None else self.kshp[i]
                        for i in range(2 + self.convdim)]
        res = get_conv_output_shape(imshp, kshp, self.border_mode,
                                    self.subsample, self.filter_dilation)
        return [res]


class AbstractConv2d(AbstractConv):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1),
                 num_groups=1,
                 unshared=False):
        super(AbstractConv2d, self).__init__(convdim=2,
                                             imshp=imshp, kshp=kshp,
                                             border_mode=border_mode,
                                             subsample=subsample,
                                             filter_flip=filter_flip,
                                             filter_dilation=filter_dilation,
                                             num_groups=num_groups,
                                             unshared=unshared)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        # Don't add the assert again, as it was already added in the forward.
        d_bottom = AbstractConv2d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip,
                                             self.filter_dilation,
                                             num_groups=self.num_groups,
                                             unshared=self.unshared)(
            weights, top, bottom.shape[-2:], add_assert_shape=False)
        d_weights = AbstractConv2d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip,
                                               self.filter_dilation,
                                               num_groups=self.num_groups,
                                               unshared=self.unshared)(

            bottom, top, weights.shape[-2:], add_assert_shape=False)

        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        return d_bottom, d_weights


class AbstractConv3d(AbstractConv):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(AbstractConv3d, self).__init__(convdim=3,
                                             imshp=imshp, kshp=kshp,
                                             border_mode=border_mode,
                                             subsample=subsample,
                                             filter_flip=filter_flip,
                                             filter_dilation=filter_dilation,
                                             num_groups=num_groups)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_bottom = AbstractConv3d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip,
                                             self.filter_dilation,
                                             self.num_groups)(
            weights, top, bottom.shape[-3:])
        d_weights = AbstractConv3d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip,
                                               self.filter_dilation,
                                               self.num_groups)(

            bottom, top, weights.shape[-3:])

        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        return d_bottom, d_weights


class AbstractConv_gradWeights(BaseAbstractConv):
    """Gradient wrt. filters for `AbstractConv`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """
    def __init__(self,
                 convdim,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=None,
                 filter_flip=True,
                 filter_dilation=None,
                 num_groups=1,
                 unshared=False):
        super(AbstractConv_gradWeights, self).__init__(convdim=convdim,
                                                       imshp=imshp, kshp=kshp,
                                                       border_mode=border_mode,
                                                       subsample=subsample,
                                                       filter_flip=filter_flip,
                                                       filter_dilation=filter_dilation,
                                                       num_groups=num_groups,
                                                       unshared=unshared)

    # Update shape/height_width
    def make_node(self, img, topgrad, shape, add_assert_shape=True):
        # Make sure both inputs are Variables with the same Type
        if not isinstance(img, theano.Variable):
            img = as_tensor_variable(img)
        if not isinstance(topgrad, theano.Variable):
            topgrad = as_tensor_variable(topgrad)
        gtype = img.type.clone(dtype=topgrad.dtype,
                               broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if img.type.ndim != 2 + self.convdim:
            raise TypeError('img must be %dD tensor' % (2 + self.convdim))
        if topgrad.type.ndim != 2 + self.convdim:
            raise TypeError('topgrad must be %dD tensor' % (2 + self.convdim))
        if add_assert_shape:
            img = assert_shape(img, self.imshp,
                               'AbstractConv_gradWeights shape mismatch: shape of '
                               'image does not match given imshp.')

        shape = as_tensor_variable(shape)
        if self.unshared:
            broadcastable = [topgrad.broadcastable[1]] + ([False] * self.convdim) + \
                            [img.broadcastable[1]] + ([False] * self.convdim)
        else:
            broadcastable = [topgrad.broadcastable[1],
                             img.broadcastable[1]] + ([False] * self.convdim)
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        img, topgrad, shape = inp
        img = np.asarray(img)
        topgrad = np.asarray(topgrad)

        o, = out_

        if self.unshared and self.convdim != 2:
            raise NotImplementedError('Unshared convolution not implemented for %dD'
                                      % self.convdim)
        dil_shape = tuple((shape[i] - 1) * self.filter_dilation[i] + 1
                          for i in range(self.convdim))

        pad = border_mode_to_pad(self.border_mode, self.convdim, dil_shape)

        if any(p != (0, 0) for p in pad):
            new_img = np.zeros((img.shape[0], img.shape[1]) +
                               tuple(img.shape[i + 2] + pad[i][0] + pad[i][1]
                                     for i in range(self.convdim)),
                               dtype=img.dtype)
            new_img[(slice(None), slice(None)) +
                    tuple(slice(pad[i][0], img.shape[i + 2] + pad[i][0])
                          for i in range(self.convdim))] = img
            img = new_img

        if any(self.subsample[i] > 1 for i in range(self.convdim)):
            new_shape = ((topgrad.shape[0], topgrad.shape[1]) +
                         tuple(img.shape[i + 2] - dil_shape[i] + 1
                               for i in range(self.convdim)))
            new_topgrad = np.zeros((new_shape), dtype=topgrad.dtype)
            new_topgrad[(slice(None), slice(None)) +
                        tuple(slice(None, None, self.subsample[i])
                              for i in range(self.convdim))] = topgrad
            topgrad = new_topgrad

        axes_order = (1, 0) + tuple(range(2, self.convdim + 2))
        topgrad = topgrad.transpose(axes_order)
        img = img.transpose(axes_order)

        def correct_for_groups(mat):
            mshp0 = mat.shape[0] // self.num_groups
            mshp1 = mat.shape[1] * self.num_groups
            mat = mat.reshape((self.num_groups, mshp0) + mat.shape[1:])
            mat = mat.transpose((1, 0, 2) + tuple(range(3, 3 + self.convdim)))
            mat = mat.reshape((mshp0, mshp1) + mat.shape[-self.convdim:])
            return mat

        if self.num_groups > 1:
            img = correct_for_groups(img)

        if self.unshared:
            flip_kern = ((slice(None),) * (2 + self.convdim) +
                         (slice(None, None, -1),) * self.convdim)
            kern = self.conv(img, topgrad, mode="valid", num_groups=self.num_groups,
                             unshared=True, direction="backprop weights")
            if any(self.subsample[i] > 1 for i in range(self.convdim)):
                sub_slice = (slice(None),) * 2 + \
                    tuple(slice(None, None, self.subsample[i]) for i in range(0, self.convdim)) + \
                    (slice(None),) * self.convdim
                kern = kern[sub_slice]
            # from (nChannels, nFilters, out_rows, out_cols, kH, kW)
            # to (nFilters, out_rows, out_cols, nChannels, kH, kW)
            kern_axes = (1,) + tuple(range(2, self.convdim + 2)) + (0,) + \
                tuple(range(self.convdim + 2, kern.ndim))
        else:
            flip_topgrad = flip_kern = ((slice(None), slice(None)) +
                                        (slice(None, None, -1),) * self.convdim)
            topgrad = topgrad[flip_topgrad]
            kern = self.conv(img, topgrad, mode="valid", num_groups=self.num_groups)
            kern_axes = (1, 0) + tuple(range(2, self.convdim + 2))

        kern = kern.transpose(kern_axes)

        if any(self.filter_dilation[i] > 1 for i in range(self.convdim)):
            kern = kern[(slice(None),) * (kern.ndim - self.convdim) +
                        tuple(slice(None, None, self.filter_dilation[i])
                              for i in range(self.convdim))]

        if self.filter_flip:
            kern = kern[flip_kern]
        o[0] = node.outputs[0].type.filter(kern)

    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.kshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the kernel shape
        # from the shapes of inputs.
        imshp = input_shapes[0]
        topshp = input_shapes[1]

        if self.kshp:
            kshp = self.kshp
        else:
            if self.unshared:
                kshp = [None] * (2 + 2 * self.convdim)
            else:
                kshp = [None] * (2 + self.convdim)
        if self.unshared:
            fallback_kshp = ([topshp[1], topshp[2], topshp[3], imshp[1] // self.num_groups] +
                             [node.inputs[2][i] for i in range(self.convdim)])
            kshp = [fallback_kshp[i] if kshp[i] is None else kshp[i]
                    for i in range(2 + 2 * self.convdim)]
        else:
            fallback_kshp = ([topshp[1], imshp[1] // self.num_groups] +
                             [node.inputs[2][i] for i in range(self.convdim)])
            kshp = [fallback_kshp[i] if kshp[i] is None else kshp[i]
                    for i in range(2 + self.convdim)]
        return [kshp]


class AbstractConv2d_gradWeights(AbstractConv_gradWeights):
    """Gradient wrt. filters for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """
    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1),
                 num_groups=1,
                 unshared=False):
        super(AbstractConv2d_gradWeights, self).__init__(convdim=2,
                                                         imshp=imshp, kshp=kshp,
                                                         border_mode=border_mode,
                                                         subsample=subsample,
                                                         filter_flip=filter_flip,
                                                         filter_dilation=filter_dilation,
                                                         num_groups=num_groups,
                                                         unshared=unshared)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = AbstractConv2d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip,
                                             self.filter_dilation,
                                             self.num_groups,
                                             self.unshared)(weights,
                                                            top,
                                                            bottom.shape[-2:])
        d_top = AbstractConv2d(self.imshp,
                               self.kshp,
                               self.border_mode,
                               self.subsample,
                               self.filter_flip,
                               self.filter_dilation,
                               self.num_groups,
                               self.unshared)(bottom, weights)
        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_top = patternbroadcast(d_top, top.broadcastable)
        d_top = top.type.filter_variable(d_top)

        d_height_width = (theano.gradient.DisconnectedType()(),)
        return (d_bottom, d_top) + d_height_width


class AbstractConv3d_gradWeights(AbstractConv_gradWeights):
    """Gradient wrt. filters for `AbstractConv3d`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """
    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(AbstractConv3d_gradWeights, self).__init__(convdim=3,
                                                         imshp=imshp, kshp=kshp,
                                                         border_mode=border_mode,
                                                         subsample=subsample,
                                                         filter_flip=filter_flip,
                                                         filter_dilation=filter_dilation,
                                                         num_groups=num_groups)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = AbstractConv3d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip,
                                             self.filter_dilation,
                                             self.num_groups)(weights,
                                                              top,
                                                              bottom.shape[-3:])
        d_top = AbstractConv3d(self.imshp,
                               self.kshp,
                               self.border_mode,
                               self.subsample,
                               self.filter_flip,
                               self.filter_dilation,
                               self.num_groups)(bottom, weights)
        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_top = patternbroadcast(d_top, top.broadcastable)
        d_top = top.type.filter_variable(d_top)

        d_depth_height_width = (theano.gradient.DisconnectedType()(),)
        return (d_bottom, d_top) + d_depth_height_width


class AbstractConv_gradInputs(BaseAbstractConv):
    """Gradient wrt. inputs for `AbstractConv`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self,
                 convdim,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=None,
                 filter_flip=True,
                 filter_dilation=None,
                 num_groups=1,
                 unshared=False):
        super(AbstractConv_gradInputs, self).__init__(convdim=convdim,
                                                      imshp=imshp, kshp=kshp,
                                                      border_mode=border_mode,
                                                      subsample=subsample,
                                                      filter_flip=filter_flip,
                                                      filter_dilation=filter_dilation,
                                                      num_groups=num_groups,
                                                      unshared=unshared)

    # Update shape/height_width
    def make_node(self, kern, topgrad, shape, add_assert_shape=True):
        # Make sure both inputs are Variables with the same Type
        if not isinstance(kern, theano.Variable):
            kern = as_tensor_variable(kern)
        if not isinstance(topgrad, theano.Variable):
            topgrad = as_tensor_variable(topgrad)
        gtype = kern.type.clone(dtype=topgrad.dtype,
                                broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if self.unshared:
            if self.convdim != 2:
                raise NotImplementedError('Unshared convolution not implemented for %dD'
                                          % self.convdim)
            elif kern.type.ndim != 2 + 2 * self.convdim:
                raise TypeError('kern must be %dD tensor for unshared convolution'
                                % (2 + 2 * self.convdim))
        else:
            if kern.type.ndim != 2 + self.convdim:
                raise TypeError('kern must be %dD tensor' % (2 + self.convdim))

        if topgrad.type.ndim != 2 + self.convdim:
                raise TypeError('topgrad must be %dD tensor' % (2 + self.convdim))

        if add_assert_shape:
            kern = assert_shape(kern, self.kshp,
                                'AbstractConv_gradInputs shape mismatch: shape of '
                                'filters does not match given kshp.')

        shape = as_tensor_variable(shape)
        if self.num_groups > 1:
            broadcastable = [topgrad.type.broadcastable[0],
                             False] + ([False] * self.convdim)
        else:
            broadcastable = [topgrad.type.broadcastable[0],
                             kern.type.broadcastable[-self.convdim - 1]] + ([False] * self.convdim)
        output = kern.type.clone(broadcastable=broadcastable)()
        return Apply(self, [kern, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        kern, topgrad, shape = inp
        kern = np.asarray(kern)
        topgrad = np.asarray(topgrad)
        o, = out_

        if self.unshared and self.convdim != 2:
            raise NotImplementedError('Unshared convolution not implemented for %dD'
                                      % self.convdim)
        dil_kernshp = tuple((kern.shape[-self.convdim + i] - 1) * self.filter_dilation[i] + 1
                            for i in range(self.convdim))

        pad = border_mode_to_pad(self.border_mode, self.convdim, dil_kernshp)

        imshp = self.imshp[:] if self.imshp is not None else [None] * (2 + self.convdim)
        fallback_imshp = ([topgrad.shape[0], kern.shape[-self.convdim - 1]] +
                          [shape[i] for i in range(self.convdim)])
        imshp = [fallback_imshp[i] if imshp[i] is None else imshp[i]
                 for i in range(2 + self.convdim)]
        expected_topgrad_shape = get_conv_output_shape(
            imshp, kern.shape,
            self.border_mode, self.subsample, self.filter_dilation)
        if not tuple(expected_topgrad_shape) == tuple(topgrad.shape):
            raise ValueError(
                'invalid input_shape for gradInputs: the given input_shape '
                'would produce an output of shape {}, but the given topgrad '
                'has shape {}'.format(tuple(expected_topgrad_shape),
                                      tuple(topgrad.shape)))
        if any(self.subsample[i] > 1 for i in range(self.convdim)):
            new_shape = ((topgrad.shape[0], topgrad.shape[1]) +
                         tuple(shape[i] + pad[i][0] + pad[i][1] - dil_kernshp[i] + 1
                               for i in range(self.convdim)))
            new_topgrad = np.zeros((new_shape), dtype=topgrad.dtype)
            new_topgrad[(slice(None), slice(None)) +
                        tuple(slice(None, None, self.subsample[i])
                              for i in range(self.convdim))] = topgrad
            topgrad = new_topgrad

            if self.unshared:
                # Expand regions in kernel to correct for subsampling
                exp_kern_shp = kern.shape[:1] + topgrad.shape[2:] + kern.shape[1 + self.convdim:]
                exp_kern = np.zeros(exp_kern_shp, dtype=kern.dtype)
                exp_kern[(slice(None),) +
                         tuple(slice(None, None, self.subsample[i]) for i in range(self.convdim)) +
                         (slice(None),) * (self.convdim + 1)] = kern
                kern = exp_kern

        def correct_for_groups(mat):
            mshp0 = mat.shape[0] // self.num_groups
            mshp1 = mat.shape[-self.convdim - 1] * self.num_groups
            mat = mat.reshape((self.num_groups, mshp0) + mat.shape[1:])
            if self.unshared:
                # for 2D -> (1, 2, 3, 0, 4, 5, 6)
                mat = mat.transpose(tuple(range(1, 2 + self.convdim)) + (0,) +
                                    tuple(range(2 + self.convdim, mat.ndim)))
                mat = mat.reshape((mshp0,) + mat.shape[1:1 + self.convdim] + (mshp1,) + mat.shape[-self.convdim:])
            else:
                mat = mat.transpose((1, 0, 2) + tuple(range(3, 3 + self.convdim)))
                mat = mat.reshape((mshp0, mshp1) + mat.shape[-self.convdim:])
            return mat

        kern = correct_for_groups(kern)

        if self.unshared:
            # from (nFilters, out_rows, out_cols, nChannels, kH, kW)
            # to (nChannels, nFilters, out_rows, out_cols, kH, kW)
            axes_order = (1 + self.convdim, 0,) + tuple(range(1, 1 + self.convdim)) + \
                tuple(range(2 + self.convdim, kern.ndim))
            kern = kern.transpose(axes_order)
            if not self.filter_flip:
                kern = kern[(slice(None),) * (kern.ndim - self.convdim) +
                            (slice(None, None, -1),) * self.convdim]
            img = self.conv(topgrad, kern, mode="full", dilation=self.filter_dilation,
                            num_groups=self.num_groups, unshared=True, direction="backprop inputs")
        else:
            axes_order = (1, 0) + tuple(range(2, 2 + self.convdim))
            kern = kern.transpose(axes_order)
            flip_filters = ((slice(None), slice(None)) +
                            (slice(None, None, -1),) * self.convdim)
            if self.filter_flip:
                topgrad = topgrad[flip_filters]
            img = self.conv(topgrad, kern, mode="full", dilation=self.filter_dilation,
                            num_groups=self.num_groups)
            if self.filter_flip:
                img = img[flip_filters]

        if any(p != (0, 0) for p in pad):
            img = img[(slice(None), slice(None)) +
                      tuple(slice(pad[i][0], img.shape[i + 2] - pad[i][1])
                            for i in range(self.convdim))]
        o[0] = node.outputs[0].type.filter(img)

    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.imshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the image shape
        # from the shapes of inputs.
        kshp = input_shapes[0]
        topshp = input_shapes[1]
        imshp = self.imshp[:] if self.imshp is not None else [None] * (2 + self.convdim)
        if self.num_groups > 1:
            fallback_imshp = ([topshp[0], kshp[-self.convdim - 1] * self.num_groups] +
                              [node.inputs[2][i] for i in range(self.convdim)])
        else:
            fallback_imshp = ([topshp[0], kshp[-self.convdim - 1]] +
                              [node.inputs[2][i] for i in range(self.convdim)])
        imshp = [fallback_imshp[i] if imshp[i] is None else imshp[i]
                 for i in range(2 + self.convdim)]
        return [imshp]


class AbstractConv2d_gradInputs(AbstractConv_gradInputs):
    """Gradient wrt. inputs for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1),
                 num_groups=1,
                 unshared=False):
        super(AbstractConv2d_gradInputs, self).__init__(convdim=2,
                                                        imshp=imshp, kshp=kshp,
                                                        border_mode=border_mode,
                                                        subsample=subsample,
                                                        filter_flip=filter_flip,
                                                        filter_dilation=filter_dilation,
                                                        num_groups=num_groups,
                                                        unshared=unshared)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = AbstractConv2d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip,
                                               self.filter_dilation,
                                               self.num_groups,
                                               self.unshared)(
                                                   bottom, top,
                                                   weights.shape[-2:])
        d_top = AbstractConv2d(self.imshp, self.kshp,
                               self.border_mode,
                               self.subsample,
                               self.filter_flip,
                               self.filter_dilation,
                               self.num_groups,
                               self.unshared)(bottom, weights)
        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        d_top = patternbroadcast(d_top, top.broadcastable)
        d_top = top.type.filter_variable(d_top)

        d_height_width = (theano.gradient.DisconnectedType()(),)
        return (d_weights, d_top) + d_height_width


class AbstractConv3d_gradInputs(AbstractConv_gradInputs):
    """Gradient wrt. inputs for `AbstractConv3d`.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(AbstractConv3d_gradInputs, self).__init__(convdim=3,
                                                        imshp=imshp, kshp=kshp,
                                                        border_mode=border_mode,
                                                        subsample=subsample,
                                                        filter_flip=filter_flip,
                                                        filter_dilation=filter_dilation,
                                                        num_groups=num_groups)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = AbstractConv3d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip,
                                               self.filter_dilation,
                                               self.num_groups)(bottom, top,
                                                                weights.shape[-3:])
        d_top = AbstractConv3d(self.imshp, self.kshp,
                               self.border_mode,
                               self.subsample,
                               self.filter_flip,
                               self.filter_dilation,
                               self.num_groups)(bottom, weights)
        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        d_top = patternbroadcast(d_top, top.broadcastable)
        d_top = top.type.filter_variable(d_top)

        d_depth_height_width = (theano.gradient.DisconnectedType()(),)
        return (d_weights, d_top) + d_depth_height_width
