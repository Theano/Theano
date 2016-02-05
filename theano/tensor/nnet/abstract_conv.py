"""
Abstract conv interface
"""

import logging
import theano

from theano.tensor import as_tensor_variable, patternbroadcast
from theano.gof import Apply, Op


__docformat__ = "restructuredtext en"
_logger = logging.getLogger("theano.tensor.nnet.abstract_conv")


def get_conv_output_shape(image_shape, kernel_shape,
                          border_mode, subsample):
    """
    This function compute the output shape of convolution operation.

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
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric). If it is a string, it must be 'valid', 'half' or 'full'.
        If it is a tuple, its two (or three) elements respectively correspond
        to the padding on height and width (and possibly depth) axis.
    subsample: tuple of int (symbolic or numeric). Its or three elements
        espectively correspond to the subsampling on height and width (and
        possibly depth) axis.

    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image. None where undefined.

    """
    bsize, imshp = image_shape[0], image_shape[2:]
    nkern, kshp = kernel_shape[0], kernel_shape[2:]
    if isinstance(border_mode, tuple):
        out_shp = tuple(get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode[i], subsample[i])
            for i in range(len(subsample)))
    else:
        out_shp = tuple(get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode, subsample[i])
            for i in range(len(subsample)))
    return (bsize, nkern) + out_shp


def get_conv_shape_1axis(image_shape, kernel_shape,
                         border_mode, subsample):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    border_mode: string or int. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.

    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis. None if undefined.

    """
    if None in [image_shape, kernel_shape, border_mode, subsample]:
        return None
    if border_mode == "half":
        pad = kernel_shape // 2
    elif border_mode == "full":
        pad = kernel_shape - 1
    elif border_mode == "valid":
        pad = 0
    else:
        pad = border_mode
        if pad < 0:
            raise ValueError("border_mode must be >= 0")
    out_shp = (image_shape + 2 * pad - kernel_shape) // subsample + 1

    return out_shp


def conv2d(input,
           filters,
           input_shape=None,
           filter_shape=None,
           border_mode='valid',
           subsample=(1, 1),
           filter_flip=True):
    """This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).

    Refer to :func:`nnet.conv2d <theano.tensor.nnet.conv2d>` for a more detailed documentation.
    """

    conv_op = AbstractConv2d(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip)
    return conv_op(input, filters)


def conv2d_grad_wrt_inputs(output_grad,
                           filters,
                           output_grad_shape=None,
                           input_shape=None,
                           filter_shape=None,
                           border_mode='valid',
                           subsample=(1, 1),
                           filter_flip=True):
    """This function builds the symbolic graph for getting the
    gradient of the output of a convolution (namely output_grad)
    w.r.t the input of the convolution, given a set of 2D filters
    used by the convolution, such that the output_grad is upsampled
    to the input shape.

    Parameters
    ----------
    output_grad : symbolic 4D tensor
        mini-batch of feature map stacks, of shape (batch size, input
        channels, input rows, input columns).  This is the tensor that
        will be upsampled or the output gradient of the convolution
        whose gradient will be taken with respect to the input of the
        convolution.  See the optional parameter
        ``output_grad_shape``.
    filters : symbolic 4D tensor
        set of filters used in CNN layer of shape (output channels,
        input channels, filter rows, filter columns).  See the
        optional parameter ``filter_shape``.
    output_grad_shape : list of 4 symbolic or real ints
        The shape of the output_grad parameter.  Optional, possibly
        used to choose an optimal implementation.  You can give
        ``None`` for any element of the list to specify that this
        element is not known at compile time.
    input_shape : list of 2 symbolic or real ints
        The shape (row and column size) of the input (upsampled)
        parameter.  Not Optional, since given the output_grad_shape
        and the subsample values, multiple input_shape may be
        plausible.
    filter_shape : list of 4 symbolic or real ints
        The shape of the filters parameter.  Optional, possibly used
        to choose an optimal implementation.  You can give ``None``
        for any element of the list to specify that this element is
        not known at compile time.
    border_mode : str, int or tuple of two int
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
            the input shape.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2)``
            pad input with a symmetric border of ``int1`` rows and
            ``int2`` columns, then perform a valid convolution.

    subsample : tuple of len 2
        The subsampling used in the forward pass.  Also called strides
        elsewhere.
    filter_flip : bool
        If ``True``, will flip the filter rows and columns before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.

    Returns
    -------
    symbolic 4D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output rows, output
        columns)

    Notes
    -----

    :note: If CuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    grad_input_op = AbstractConv2d_gradInputs(imshp=input_shape,
                                              kshp=filter_shape,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              filter_flip=filter_flip)

    return grad_input_op(filters, output_grad, input_shape)


def conv2d_grad_wrt_weights(input,
                            output_grad,
                            input_shape=None,
                            output_grad_shape=None,
                            filter_shape=None,
                            border_mode='valid',
                            subsample=(1, 1),
                            filter_flip=True):
    """This function will build the symbolic graph for getting the
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
    filters : symbolic 4D tensor.
        set of filters used in CNN layer of shape (output channels,
        input channels, filter rows, filter columns).  See the
        optional parameter ``filter_shape``.
    output_grad_shape : list of 4 ints or Constant variables
        The shape of the input parameter.  Optional, possibly used to
        choose an optimal implementation.  You can give ``None`` for
        any element of the list to specify that this element is not
        known at compile time.
    input_shape : list of 2 ints or Constant variables
        The shape of the input parameter.  This parameter indicates
        the row and column size of the input in the forward pass.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify
        that this element is not known at compile time.
    filter_shape : list of 4 ints or Constant variables
        The shape of the filters parameter.  Not Optional, since given
        the output_grad_shape and the input_shape, multiple
        filter_shape may be plausible.
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
            the input shape.

          ``int``
            pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.

          ``(int1, int2)``
            pad input with a symmetric border of ``int1`` rows and
            ``int2`` columns, then perform a valid convolution.

    subsample : tuple of len 2
        The subsampling used in the forward pass of the convolutional
        operation.  Also called strides elsewhere.
    filter_flip : bool
        If ``True``, will flip the filter rows and columns before
        sliding them over the input. This operation is normally
        referred to as a convolution, and this is the default. If
        ``False``, the filters are not flipped and the operation is
        referred to as a cross-correlation.

    Returns
    -------
    symbolic 4D tensor
        set of feature maps generated by convolutional layer. Tensor
        is of shape (batch size, output channels, output rows, output
        columns)

    Notes
    -----

    :note: If CuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

    :note: This is only supported in Theano 0.8 or the development
        version until it is released.

    """
    gradWeight_op = AbstractConv2d_gradWeights(imshp=input_shape,
                                               kshp=filter_shape,
                                               border_mode=border_mode,
                                               subsample=subsample,
                                               filter_flip=filter_flip)

    return gradWeight_op(input, output_grad, filter_shape)


class BaseAbstractConv2d(Op):
    """Base class for AbstractConv

    Define an abstract convolution op that will be replaced with the
    appropriate implementation

    Parameters
    ----------
     imshp: None, tuple/list of len 4 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        imshp is defined w.r.t the forward conv.

     kshp: None, tuple/list of len 4 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        kshp is defined w.r.t the forward conv.

     border_mode: str, int or tuple of two int
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

    subsample: tuple of len 2
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a
        cross-correlation.

    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_flip', 'imshp', 'kshp')

    def __init__(self,
                 imshp=None, kshp=None,
                 border_mode="valid", subsample=(1, 1),
                 filter_flip=True):

        if isinstance(border_mode, int):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if border_mode == (0, 0):
            border_mode = 'valid'
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))

        self.imshp = tuple(imshp) if imshp else None
        self.kshp = tuple(kshp) if kshp else None
        self.border_mode = border_mode
        self.filter_flip = filter_flip

        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        self.subsample = subsample

    def flops(self, inp, outp):
        """ Useful with the hack in profilemode to print the MFlops"""
        # if the output shape is correct, then this gives the correct
        # flops for any direction, sampling, padding, and border mode
        inputs, filters = inp
        outputs, = outp
        assert inputs[1] == filters[1]
        # nb mul and add by output pixel
        flops = filters[2] * filters[3] * 2
        # nb flops by output image
        flops *= outputs[2] * outputs[3]
        # nb patch multiplied
        flops *= inputs[1] * filters[0] * inputs[0]
        return flops

    def do_constant_folding(self, node):
        # Disable constant folding since there is no implementation.
        # This may change in the future.
        return False


class AbstractConv2d(BaseAbstractConv2d):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True):
        super(AbstractConv2d, self).__init__(imshp, kshp,
                                             border_mode, subsample,
                                             filter_flip)

    def make_node(self, img, kern):
        # Make sure both inputs have the same Type
        ktype = img.type.clone(dtype=kern.dtype,
                               broadcastable=kern.broadcastable)
        kern = ktype.filter_variable(kern)

        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.broadcastable[0],
                         kern.broadcastable[0],
                         False, False]
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, kern], [output])

    def perform(self, node, inp, out_):
        raise NotImplementedError(
            'AbstractConv2d theano optimization failed. '
            'Did you exclude both "conv_dnn" and "conv_gemm" from '
            'the optimizer? Is cudnn available and does the GPU support it?')

    def R_op(self, inputs, eval_points):
        rval = None
        if eval_points[0] is not None:
            rval = self.make_node(eval_points[0], inputs[1]).outputs[0]
        if eval_points[1] is not None:
            if rval is None:
                rval = self.make_node(inputs[0], eval_points[1]).outputs[0]
            else:
                rval += self.make_node(inputs[0], eval_points[1]).outputs[0]
        return [rval]

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_bottom = AbstractConv2d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip)(
            weights, top, bottom.shape[-2:])
        d_weights = AbstractConv2d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip)(

            bottom, top, weights.shape[-2:])

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

    def infer_shape(self, node, input_shapes):
        imshp = input_shapes[0]
        kshp = input_shapes[1]

        # replace symbolic shapes with known constant shapes
        if self.imshp is not None:
            imshp = [imshp[i] if self.imshp[i] is None else self.imshp[i]
                     for i in range(4)]
        if self.kshp is not None:
            kshp = [kshp[i] if self.kshp[i] is None else self.kshp[i]
                    for i in range(4)]
        res = get_conv_output_shape(imshp, kshp, self.border_mode,
                                    self.subsample)
        return [res]


class AbstractConv2d_gradWeights(BaseAbstractConv2d):
    """Gradient wrt. filters for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
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
                 filter_flip=True):
        super(AbstractConv2d_gradWeights, self).__init__(imshp, kshp,
                                                         border_mode,
                                                         subsample,
                                                         filter_flip)

    # Update shape/height_width
    def make_node(self, img, topgrad, shape):
        # Make sure both inputs have the same Type
        gtype = img.type.clone(dtype=topgrad.dtype,
                               broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        shape = as_tensor_variable(shape)
        broadcastable = [topgrad.broadcastable[1],
                         img.broadcastable[1],
                         False, False]
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        raise NotImplementedError(
            'AbstractConv2d_gradWeights theano optimization failed. '
            'Did you exclude both "conv_dnn" and "conv_gemm" from '
            'the optimizer?')

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = AbstractConv2d_gradInputs(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip)(
                                                 weights,
                                                 top,
                                                 bottom.shape[-2:])
        d_top = AbstractConv2d(self.imshp,
                               self.kshp,
                               self.border_mode,
                               self.subsample,
                               self.filter_flip)(bottom, weights)
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

    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.kshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the kernel shape
        # from the shapes of inputs.
        imshp = input_shapes[0]
        topshp = input_shapes[1]
        kshp = self.kshp[:] if self.kshp is not None else [None] * 4
        fallback_kshp = [topshp[1], imshp[1], node.inputs[2][0], node.inputs[2][1]]
        kshp = [fallback_kshp[i] if kshp[i] is None else kshp[i]
                for i in range(4)]
        return [kshp]


class AbstractConv2d_gradInputs(BaseAbstractConv2d):
    """Gradient wrt. inputs for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
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
                 filter_flip=True):
        super(AbstractConv2d_gradInputs, self).__init__(imshp, kshp,
                                                        border_mode,
                                                        subsample,
                                                        filter_flip)

    # Update shape/height_width
    def make_node(self, kern, topgrad, shape):
        # Make sure both inputs have the same Type
        gtype = kern.type.clone(dtype=topgrad.dtype,
                                broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        shape = as_tensor_variable(shape)
        broadcastable = [topgrad.type.broadcastable[0],
                         kern.type.broadcastable[1],
                         False, False]
        output = kern.type.clone(broadcastable=broadcastable)()
        return Apply(self, [kern, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        raise NotImplementedError(
            'AbstractConv2d_gradInputs theano optimization failed. '
            'Did you exclude both "conv_dnn" and "conv_gemm" from '
            'the optimizer?')

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = AbstractConv2d_gradWeights(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample)(
                                                   bottom, top,
                                                   weights.shape[-2:])
        d_top = AbstractConv2d(self.imshp, self.kshp,
                               self.border_mode, self.subsample)(
                                   bottom, weights)
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

    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.imshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the image shape
        # from the shapes of inputs.
        kshp = input_shapes[0]
        topshp = input_shapes[1]
        imshp = self.imshp[:] if self.imshp is not None else [None] * 4
        fallback_imshp = [topshp[0], kshp[1], node.inputs[2][0],
                          node.inputs[2][1]]
        imshp = [fallback_imshp[i] if imshp[i] is None else imshp[i]
                 for i in range(4)]
        return [imshp]
