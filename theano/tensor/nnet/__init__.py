from __future__ import absolute_import, print_function, division
from .nnet import (
    CrossentropyCategorical1Hot, CrossentropyCategorical1HotGrad,
    CrossentropySoftmax1HotWithBiasDx, CrossentropySoftmaxArgmax1HotWithBias,
    LogSoftmax, Prepend_scalar_constant_to_each_row,
    Prepend_scalar_to_each_row, Softmax,
    SoftmaxGrad, SoftmaxWithBias,
    binary_crossentropy, sigmoid_binary_crossentropy,
    categorical_crossentropy, crossentropy_categorical_1hot,
    crossentropy_categorical_1hot_grad, crossentropy_softmax_1hot,
    crossentropy_softmax_1hot_with_bias,
    crossentropy_softmax_1hot_with_bias_dx,
    crossentropy_softmax_argmax_1hot_with_bias,
    crossentropy_softmax_max_and_argmax_1hot,
    crossentropy_softmax_max_and_argmax_1hot_with_bias,
    crossentropy_to_crossentropy_with_softmax,
    crossentropy_to_crossentropy_with_softmax_with_bias,
    graph_merge_softmax_with_crossentropy_softmax, h_softmax,
    logsoftmax, logsoftmax_op, prepend_0_to_each_row, prepend_1_to_each_row,
    prepend_scalar_to_each_row, relu, softmax, softmax_grad, softmax_graph,
    softmax_op, softmax_simplifier, softmax_with_bias, elu, selu,
    confusion_matrix, softsign)
from . import opt
from .conv import ConvOp
from .sigm import (softplus, sigmoid, sigmoid_inplace,
                   scalar_sigmoid, ultra_fast_sigmoid,
                   hard_sigmoid)
from .bn import batch_normalization


import warnings
from .abstract_conv import conv2d as abstract_conv2d
from .abstract_conv import conv2d_grad_wrt_inputs
from .abstract_conv import conv3d
from .abstract_conv import separable_conv2d


def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, filter_dilation=(1, 1), num_groups=1, unshared=False, **kwargs):
    """
    This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).


    Parameters
    ----------
    input: symbolic 4D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input rows, input columns).
        See the optional parameter ``input_shape``.

    filters: symbolic 4D or 6D tensor
        Set of filters used in CNN layer of shape
        (output channels, input channels, filter rows, filter columns)
        for normal convolution and
        (output channels, output rows, output columns, input channels,
        filter rows, filter columns)
        for unshared convolution.
        See the optional parameter ``filter_shape``.

    input_shape: None, tuple/list of len 4 or 6 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    filter_shape: None, tuple/list of len 4 or 6 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    border_mode: str, int or a tuple of two ints or pairs of ints
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
        ``(int1, int2)``: (for 2D) pad input with a symmetric border of ``int1``,
            ``int2``, then perform a valid convolution.
        ``(int1, (int2, int3))`` or ``((int1, int2), int3)``: (for 2D)
            pad input with one symmetric border of `int1`` or ``int3``, and
            one asymmetric border of ``(int2, int3)`` or ``(int1, int2)``.

    subsample: tuple of len 2
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a cross-correlation.

    image_shape: None, tuple/list of len 4 of int or Constant variable
        Deprecated alias for input_shape.

    filter_dilation: tuple of len 2
        Factor by which to subsample (stride) the input.
        Also called dilation elsewhere.

    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    unshared: bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.

    kwargs: Any other keyword arguments are accepted for backwards
            compatibility, but will be ignored.

    Returns
    -------
    Symbolic 4D tensor
        Set of feature maps generated by convolutional layer. Tensor is
        of shape (batch size, output channels, output rows, output columns)

    Notes
    -----
        If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

        This is only supported in Theano 0.8 or the development
        version until it is released.

        The parameter filter_dilation is an implementation of `dilated
        convolution <https://arxiv.org/pdf/1511.07122v3.pdf>`_.

    """

    if 'imshp_logical' in kwargs or 'kshp_logical' in kwargs:
        raise ValueError(
            "Keyword arguments 'imshp_logical' and 'kshp_logical' for conv2d "
            "are not supported anymore (and have not been a reliable way to "
            "perform upsampling). That feature is still available by calling "
            "theano.tensor.nnet.conv.conv2d() for the time being.")
    if len(kwargs.keys()) > 0:
        warnings.warn(str(kwargs.keys()) +
                      " are now deprecated in "
                      "`tensor.nnet.abstract_conv.conv2d` interface"
                      " and will be ignored.",
                      stacklevel=2)

    if image_shape is not None:
        warnings.warn("The `image_shape` keyword argument to "
                      "`tensor.nnet.conv2d` is deprecated, it has been "
                      "renamed to `input_shape`.",
                      stacklevel=2)
        if input_shape is None:
            input_shape = image_shape
        else:
            raise ValueError("input_shape and image_shape should not"
                             " be provided at the same time.")

    return abstract_conv2d(input, filters, input_shape, filter_shape,
                           border_mode, subsample, filter_flip,
                           filter_dilation, num_groups, unshared)


def conv2d_transpose(input, filters, output_shape, filter_shape=None,
                     border_mode='valid', input_dilation=(1, 1),
                     filter_flip=True, filter_dilation=(1, 1), num_groups=1, unshared=False):
    """
    This function will build the symbolic graph for applying a transposed
    convolution over a mini-batch of a stack of 2D inputs with a set of 2D
    filters.


    Parameters
    ----------
    input: symbolic 4D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input rows, input columns).
        See the optional parameter ``input_shape``.

    filters: symbolic 4D tensor
        Set of filters used in CNN layer of shape
        (input channels, output channels, filter rows, filter columns).
        See the optional parameter ``filter_shape``. **Note: the order for
        ``output_channels`` and ``input_channels`` is reversed with respect to
        ``conv2d``.**

    output_shape: tuple/list of len 4 of int or Constant variable
        The shape of the output of ``conv2d_transpose``. The last two elements
        are allowed to be ``tensor.scalar`` variables.

    filter_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    border_mode: str, int or tuple of two int
        Refers to the ``border_mode`` argument of the corresponding forward
        (non-transposed) convolution. See the argument description in
        ``conv2d``.  What was ``padding`` for the forward convolution means
        ``cropping`` the output of the transposed one. ``valid`` corresponds to
        no cropping, ``full`` to maximal cropping.

    input_dilation: tuple of len 2
        Corresponds to ``subsample`` (also called strides elsewhere) in the
        non-transposed convolution.

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a cross-correlation.

    filter_dilation: tuple of len 2
        Factor by which to subsample (stride) the input.
        Also called dilation elsewhere.

    num_groups : int
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    unshared: bool
        If true, then unshared or 'locally connected' convolution will be
        performed. A different filter will be used for each region of the
        input.
        Grouped unshared convolution is supported.

    Returns
    -------
    Symbolic 4D tensor
        Set of feature maps generated by the transposed convolution. Tensor is
        of shape (batch size, output channels, output rows, output columns)

    Notes
    -----
        If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

        This operation is also sometimes called "deconvolution".

        The parameter filter_dilation is an implementation of `dilated
        convolution <https://arxiv.org/pdf/1511.07122v3.pdf>`_.

    """

    return conv2d_grad_wrt_inputs(output_grad=input,
                                  filters=filters,
                                  input_shape=output_shape,
                                  filter_shape=filter_shape,
                                  border_mode=border_mode,
                                  subsample=input_dilation,
                                  filter_flip=filter_flip,
                                  filter_dilation=filter_dilation,
                                  num_groups=num_groups,
                                  unshared=unshared)
