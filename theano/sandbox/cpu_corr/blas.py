import copy
import os
import logging
_logger = logging.getLogger(__name__)

import theano
from theano import Apply
#from theano import tensor
from theano.compat.six import StringIO
from theano import gof
from theano.tensor import as_tensor_variable
from theano.tensor.blas_headers import (blas_header_text,
                                        blas_header_version)
from theano.tensor.blas import ldflags


class BaseCpuCorrMM(gof.Op):
    """Base class for `CpuCorrMM`, `CpuCorrMM_gradWeights` and
    `CpuCorrMM_gradInputs`. Cannot be used directly.

    :param border_mode: one of 'valid', 'full', 'half'; additionally, the
        padding size could be directly specified by an integer or a pair of
        integers
    :param subsample: perform subsampling of the output (default: (1, 1))
    :param pad: *deprecated*, now you should always use border_mode

    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample')

    def __init__(self, border_mode="valid", subsample=(1, 1), pad=(0, 0)):
        if pad != (0, 0):
            _logger.warning(
                'do not use pad for BaseCpuCorrMM; please set padding in '
                'border_mode parameter, see the docstring for more details')
            if border_mode != "valid":
                raise ValueError("border_mode must be 'valid'")
            border_mode = pad
        if isinstance(border_mode, int):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        self.subsample = subsample

    @property
    def pad(self):
        if self.border_mode != 'valid':
            return self.border_mode
        return (0, 0)

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample))

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

    def c_support_code(self):
        return blas_header_text()

    def c_libraried(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)
    
    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        return ['<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 1)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['corr_gemm.cpp']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code_helper(self, bottom, weights, top, direction, sub, height=None, width=None):
        """
        This generates the C code for CpuCorrMM (direction="forward"),
        CpuCorrMM_gradWeights (direction="backprop weights"), and
        CpuCorrMM_gradInputs (direction="backprop inputs").
        Depending on the direction, one of bottom, weights, top will
        receive the output, while the other two serve as inputs.

        :param bottom: Variable name of the input images in the forward pass,
            or the gradient of the input images in backprop wrt. inputs
        :param weights: Variable name of the filters in the forward pass,
            or the gradient of the filters in backprop wrt. weights
        :param top: Variable name of the output images / feature maps in the
            forward pass, or the gradient of the outputs in the backprop passes
        :param direction: "forward" to correlate bottom with weights and store
            results in top,
            "backprop weights" to do a valid convolution of bottom with top
            (swapping the first two dimensions) and store results in weights,
            and "backprop inputs" to do a full convolution of top with weights
            (swapping the first two dimensions) and store results in bottom.
        :param sub: Dictionary of substitutions useable to help generating the
            C code.
        :param height: If self.subsample[0] != 1, a variable giving the height
            of the filters for direction="backprop weights" or the height of
            the input images for direction="backprop inputs".

            If self.border_mode == 'half', a variable giving the height of the
            filters for direction="backprop weights".  Ignored otherwise.
        :param width: If self.subsample[1] != 1, a variable giving the width
            of the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".

            If self.border_mode == 'half', a variable giving the width of the
            filters for direction="backprop weights".  Ignored otherwise.
        """
        dH, dW = self.subsample
        if self.border_mode == "half":
            padH = padW = -1
        elif self.border_mode == "full":
            padH = padW = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = 0
        if direction == "forward":
            direction = 0
            out = top
        elif direction == "backprop weights":
            direction = 1
            out = weights
        elif direction == "backprop inputs":
            direction = 2
            out = bottom
        else:
            raise ValueError("direction must be one of 'forward', "
                    "'backprop weights', 'backprop inputs'")
        # When subsampling, we cannot unambiguously infer the height and width
        # of bottom and weights from top, so we require them to be given.
        # Similarly, when pad="half", we cannot infer the weight size.
        if ((direction != 0) and (dH != 1)) or ((direction == 1) and (padH == -1)):
            if not height:
                raise ValueError("height must be given for backprop with vertical sampling or pad='half'")
            height = '(*(npy_int*)(PyArray_DATA(%s)))' % height
        else:
            height = 'NULL'
        if ((direction != 0) and (dW != 1)) or ((direction == 1) and (padW == -1)):
            if not width:
                raise ValueError("width must be given for backprop with horizontal sampling or pad='half'")
            width = '(*(npy_int*)(PyArray_DATA(%s)))' % width
        else:
            width = 'NULL'
        sub = sub.copy()
        sub.update(locals())

        return """
    // Mandatory args
    int direction = %(direction)s;  // forward, bprop weights, bprop inputs

    // Optional args
    int dH = %(dH)s;
    int dW = %(dW)s;
    int padH = %(padH)s;
    int padW = %(padW)s;
    
    PyArray * bottom = %(bottom)s;
    PyArray * weights = %(weights)s;
    PyArray * top = %(top)s;
    PyArray * out2 = NULL;

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = PyArray_DIMS(weights)[2];
        kW = PyArray_DIMS(weights)[3];
    }
    else {
        if ((dH != 1) || (padH == -1)) {
            // vertical subsampling or half padding, kernel height is specified
            kH = %(height)s;
        }
        else if (padH == -2) {
            // vertical full padding, we can infer the kernel height
            kH = 2 - PyArray_DIMS(bottom)[2] + (PyArray_DIMS(top)[2] - 1) * dH;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = PyArray_DIMS(bottom)[2] + 2*padH - (PyArray_DIMS(top)[2] - 1) * dH;
        }
        if ((dW != 1) || (padW == -1)) {
            kW = %(width)s;
        }
        else if (padW == -2) {
            kW = 2 - PyArray_DIMS(bottom)[3] + (PyArray_DIMS(top)[3] - 1) * dW;
        }
        else {
            kW = PyArray_DIMS(bottom)[3] + 2*padW - (PyArray_DIMS(top)[3] - 1) * dW;
        }
    }

    // Auto-padding if requested
    if (padH == -1) {  // vertical half padding
        padH = kH / 2;
    }
    else if (padH == -2) {  // vertical full padding
        padH = kH - 1;
    }
    else if (padH < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCpuCorrMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCpuCorrMM: padW must be >= -2");
        %(fail)s
    }

    // Infer output shape
    int out_dim[4];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + 2*pad - weight) / sample + 1
        out_dim[0] = PyArray_DIMS(bottom)[0];
        out_dim[1] = PyArray_DIMS(weights)[0];
        out_dim[2] = (PyArray_DIMS(bottom)[2] + 2*padH - PyArray_DIMS(weights)[2]) / dH + 1;
        out_dim[3] = (PyArray_DIMS(bottom)[3] + 2*padW - PyArray_DIMS(weights)[3]) / dW + 1;
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width)
        // height and width: weights = bottom + 2*pad - (top - 1) * sample
        out_dim[0] = PyArray_DIMS(top)[1];
        out_dim[1] = PyArray_DIMS(bottom)[1];
        out_dim[2] = kH;  // already inferred further above
        out_dim[3] = kW;  // how convenient
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + weights - 2*pad
        out_dim[0] = PyArray_DIMS(top)[0];
        out_dim[1] = PyArray_DIMS(weights)[1];
        out_dim[2] = (dH != 1) ? %(height)s : (PyArray_DIMS(top)[2] - 1) * dH + PyArray_DIMS(weights)[2] - 2*padH;
        out_dim[3] = (dW != 1) ? %(width)s : (PyArray_DIMS(top)[3] - 1) * dW + PyArray_DIMS(weights)[3] - 2*padW;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseCpuCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    if ( !(%(out)s
           && %(out)s->nd==4
           && PyArray_is_c_contiguous(%(out)s)
           && PyArray_DIMS(%(out)s)[0]==out_dim[0]
           && PyArray_DIMS(%(out)s)[1]==out_dim[1]
           && PyArray_DIMS(%(out)s)[2]==out_dim[2]
           && PyArray_DIMS(%(out)s)[3]==out_dim[3]))
    {
        Py_XDECREF(%(out)s);
        %(out)s = (PyArray*)PyArray_NewDims(4,out_dim);
        if (NULL == %(out)s)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseCpuCorrMM: Failed to allocate output of %%d x %%d x %%d x %%d",
                    out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
            %(fail)s
        }
    }

    // Call CUDA code
    out2 = corrMM(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, padH, padW);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class CpuCorrMM(BaseCpuCorrMM):
    """GPU correlation implementation using Matrix Multiplication.

    :param border_mode: currently supports "valid" only; "full" can be
        simulated by setting `pad="full"` (at the cost of performance), or
        by using `CpuCorrMM_gradInputs`
    :param subsample: the subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `CpuCorrMM(...)(...)[:,:,::sv, ::sh]`,
        but faster.
        Set to `(1, 1)` to disable subsampling.
    :param pad: the width of a border of implicit zeros to pad the input
        image with. Should be a tuple with 2 elements giving the numbers of
        rows and columns to pad on each side, or "half" to set the padding
        to `(kernel_rows // 2, kernel_columns // 2)`, or "full" to set the
        padding to `(kernel_rows - 1, kernel_columns - 1)` at runtime.
        Set to `(0, 0)` to disable padding.

    :note: Currently, the Op requires the inputs, filters and outputs to be
        C-contiguous. Use :func:`gpu_contiguous
        <theano.sandbox.cuda.basic_ops.gpu_contiguous>` on these arguments
        if needed.

    :note: You can either enable the Theano flag `optimizer_including=conv_gemm`
        to automatically replace all convolution operations with `CpuCorrMM`
        or one of its gradients, or you can use it as a replacement for
        :func:`conv2d <theano.tensor.nnet.conv.conv2d>`, called as
        `CpuCorrMM(subsample=...)(image, filters)`. The latter is currently
        faster, but note that it computes a correlation -- if you need to
        compute a convolution, flip the filters as `filters[:,:,::-1,::-1]`.

    :warning: For 700 series Nvidia GPUs of compute capability 3.5 and CUDA 5.0
        to 6.0, there is a bug in CUBLAS' matrix multiplication function that
        can make CpuCorrMM or its gradients crash for some input and filter
        shapes. So if you have a Tesla K20, Tesla K40, Quadro K6000, GeForce GT
        640 (DDR5), GeForce GTX 780 (or Ti), GeForce GTX TITAN (or Black or Z)
        and experience a crash, switching to CUDA 6.5 or CUDA 4.2 should fix it.
        If this is not possible, changing the input or filter shapes (e.g., the
        batchsize or number of filters) may also work around the CUBLAS bug.
    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 pad=(0, 0)):
        super(CpuCorrMM, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, kern):
        # TODO broadcastable checks
        img = as_tensor_variable(img)
        kern = as_tensor_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [img.type()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(CpuCorrMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = CpuCorrMM_gradInputs(self.border_mode, self.subsample)(
            weights, top, bottom.shape[-2:])
        d_weights = CpuCorrMM_gradWeights(self.border_mode, self.subsample)(
            bottom, top, weights.shape[-2:])
        return d_bottom, d_weights


class CpuCorrMM_gradWeights(BaseCpuCorrMM):
    """Gradient wrt. filters for `CpuCorrMM`.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self, border_mode="valid",
            subsample=(1, 1),
            pad=(0, 0)):
        super(CpuCorrMM_gradWeights, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, topgrad, shape=None):
        img = as_tensor_variable(img)
        topgrad = as_tensor_variable(topgrad)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if self.subsample != (1, 1) or self.border_mode == "half":
            if shape is None:
                raise ValueError('shape must be given if subsample != (1, 1)'
                                 ' or border_mode == "half"')
            height_width = [shape[0], shape[1]]
        else:
            height_width = []

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False]
        return Apply(self, [img, topgrad] + height_width, [img.type()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width = inp[2:] or (None, None)
        weights, = out_
        direction = "backprop weights"
        return super(CpuCorrMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        weights = gpu_contiguous(weights)
        d_bottom = CpuCorrMM_gradInputs(self.border_mode, self.subsample)(
                weights, top, bottom.shape[-2:])
        d_top = CpuCorrMM(self.border_mode, self.subsample)(
                bottom, weights)
        d_height_width = (theano.gradient.DisconnectedType()(),) * 2 if len(inp) == 4 else ()
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class CpuCorrMM_gradInputs(BaseCpuCorrMM):
    """Gradient wrt. inputs for `CpuCorrMM`.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self, border_mode="valid",
            subsample=(1, 1),
            pad=(0, 0)):
        super(CpuCorrMM_gradInputs, self).__init__(border_mode, subsample, pad)

    def make_node(self, kern, topgrad, shape=None):
        kern = as_tensor_variable(kern)
        topgrad = as_tensor_variable(topgrad)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if self.subsample != (1, 1) and shape is None:
            raise ValueError('shape must be given if subsample != (1, 1)')
        height_width = [shape[0], shape[1]] if self.subsample != (1, 1) else []

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False]
        return Apply(self, [kern, topgrad] + height_width, [kern.type()])

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width = inp[2:] or (None, None)
        bottom, = out_
        direction = "backprop inputs"
        return super(CpuCorrMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        bottom = gpu_contiguous(bottom)
        d_weights = CpuCorrMM_gradWeights(self.border_mode, self.subsample)(
                bottom, top, weights.shape[-2:])
        d_top = CpuCorrMM(self.border_mode, self.subsample)(
                bottom, weights)
        d_height_width = (theano.gradient.DisconnectedType()(),) * 2 if len(inp) == 4 else ()
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width

