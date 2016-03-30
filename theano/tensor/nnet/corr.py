from __future__ import absolute_import, print_function, division
import os
import logging

from six import integer_types

import theano
from theano import Apply
from theano import gof
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.blas_headers import blas_header_text
from theano.tensor.blas import ldflags


_logger = logging.getLogger(__name__)


class BaseCorrMM(gof.Op):
    """
    Base class for `CorrMM`, `CorrMM_gradWeights` and
    `CorrMM_gradInputs`. Cannot be used directly.
    Parameters
    ----------
    border_mode : {'valid', 'full', 'half'}
        Additionally, the padding size could be directly specified by an integer
        or a pair of integers
    subsample
        Perform subsampling of the output (default: (1, 1)).

    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample')

    def __init__(self, border_mode="valid", subsample=(1, 1)):
        if isinstance(border_mode, integer_types):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            if len(border_mode) != 2 or border_mode[0] < 0 or border_mode[1] < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'pair of non-negative integers'.format(border_mode))
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
        self.subsample = tuple(subsample)

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

    def c_support_code(self):
        return blas_header_text()

    def c_libraries(self):
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
        return (1, 1)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        sub = {}
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')
        if dtype == 'float32':
            sub['gemm'] = 'sgemm_'
            sub['float_type'] = 'npy_float'
            sub['float_typenum'] = 'NPY_FLOAT'
            sub['n_bytes'] = 4
            sub['c_float_type'] = 'float'
        else:
            sub['gemm'] = 'dgemm_'
            sub['float_type'] = 'npy_double'
            sub['float_typenum'] = 'NPY_DOUBLE'
            sub['n_bytes'] = 8
            sub['c_float_type'] = 'double'
        files = ['corr_gemm.c']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                 for f in files]
        final_code = ''
        for code in codes:
            final_code += code
        return final_code % sub

    def c_code_helper(self, bottom, weights, top, direction, sub, height=None, width=None):
        """
        This generates the C code for CorrMM (direction="forward"),
        CorrMM_gradWeights (direction="backprop weights"), and
        CorrMM_gradInputs (direction="backprop inputs").
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
        if not theano.config.blas.ldflags:
            raise NotImplementedError("C code for CorrMM* classes need a blas library.")
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
        # Similarly, when border_mode="half", we cannot infer the weight size.
        if ((direction != 0) and (dH != 1)) or ((direction == 1) and (padH == -1)):
            if not height:
                raise ValueError("height must be given for backprop with vertical sampling or border_mode='half'")
            height = '(*(npy_int64 *)(PyArray_DATA(%s)))' % height
        else:
            height = '-1'
        if ((direction != 0) and (dW != 1)) or ((direction == 1) and (padW == -1)):
            if not width:
                raise ValueError("width must be given for backprop with horizontal sampling or border_mode='half'")
            width = '(*(npy_int64 *)(PyArray_DATA(%s)))' % width
        else:
            width = '-1'
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

    PyArrayObject * bottom = %(bottom)s;
    PyArrayObject * weights = %(weights)s;
    PyArrayObject * top = %(top)s;
    PyArrayObject * out2 = NULL;

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
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: padW must be >= -2");
        %(fail)s
    }

    // Infer output shape
    npy_intp out_dim[4];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + 2*pad - weight) / sample + 1
        out_dim[0] = (npy_intp)PyArray_DIMS(bottom)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[0];
        out_dim[2] = (npy_intp)((PyArray_DIMS(bottom)[2] + 2*padH - PyArray_DIMS(weights)[2]) / dH + 1);
        out_dim[3] = (npy_intp)((PyArray_DIMS(bottom)[3] + 2*padW - PyArray_DIMS(weights)[3]) / dW + 1);
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width)
        // height and width: weights = bottom + 2*pad - (top - 1) * sample
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[1];
        out_dim[1] = (npy_intp)PyArray_DIMS(bottom)[1];
        out_dim[2] = (npy_intp)kH;  // already inferred further above
        out_dim[3] = (npy_intp)kW;  // how convenient
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + weights - 2*pad
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[1];
        out_dim[2] = (npy_intp)((dH != 1) ? %(height)s : (PyArray_DIMS(top)[2] - 1) * dH + PyArray_DIMS(weights)[2] - 2*padH);
        out_dim[3] = (npy_intp)((dW != 1) ? %(width)s : (PyArray_DIMS(top)[3] - 1) * dW + PyArray_DIMS(weights)[3] - 2*padW);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    int typenum;
    if ( !(%(out)s
           && PyArray_NDIM(%(out)s)==4
           && PyArray_IS_C_CONTIGUOUS(%(out)s)
           && PyArray_DIMS(%(out)s)[0]==out_dim[0]
           && PyArray_DIMS(%(out)s)[1]==out_dim[1]
           && PyArray_DIMS(%(out)s)[2]==out_dim[2]
           && PyArray_DIMS(%(out)s)[3]==out_dim[3]))
    {
        Py_XDECREF(%(out)s);
        if (direction != 1) {
          typenum = PyArray_TYPE(weights);
        }
        else {
          typenum = PyArray_TYPE(bottom);
        }
        %(out)s = (PyArrayObject*)PyArray_EMPTY(4,
                                          out_dim,
                                          typenum,
                                          0);
        if (NULL == %(out)s)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                    (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
            %(fail)s
        }
    }

    // Call corrMM code
    out2 = corrMM(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, padH, padW);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class CorrMM(BaseCorrMM):
    """
    CPU correlation implementation using Matrix Multiplication.

    Parameters
    ----------
    border_mode
        The width of a border of implicit zeros to pad the
        input with. Must be a tuple with 2 elements giving the numbers of rows
        and columns to pad on each side, or a single integer to pad the same
        on all sides, or a string shortcut setting the padding at runtime:
        ``'valid'`` for ``(0, 0)`` (valid convolution, no padding), ``'full'``
        for ``(kernel_rows - 1, kernel_columns - 1)`` (full convolution),
        ``'half'`` for ``(kernel_rows // 2, kernel_columns // 2)`` (same
        convolution for odd-sized kernels). Note that the two widths are each
        applied twice, once per side (left and right, top and bottom).
    subsample
        The subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `CorrMM(...)(...)[:,:,::sv, ::sh]`,
        but faster.
        Set to `(1, 1)` to disable subsampling.

    """
    def __init__(self, border_mode="valid", subsample=(1, 1)):
        super(CorrMM, self).__init__(border_mode, subsample)

    def make_node(self, img, kern):
        img = as_tensor_variable(img)
        kern = as_tensor_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        dtype = img.type.dtype
        return Apply(self, [img, kern], [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        imshp = input_shape[0]
        kshp = input_shape[1]
        res = get_conv_output_shape(
            imshp,
            kshp,
            self.border_mode,
            self.subsample)
        return [res]

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(CorrMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_bottom = CorrMM_gradInputs(self.border_mode,
                                     self.subsample)(weights, top,
                                                     bottom.shape[-2:])
        d_weights = CorrMM_gradWeights(self.border_mode,
                                       self.subsample)(bottom, top,
                                                       weights.shape[-2:])
        return d_bottom, d_weights


class CorrMM_gradWeights(BaseCorrMM):
    """
    Gradient wrt. filters for `CorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on
    Theano's automatic differentiation or graph optimization to
    use it as needed.

    """

    def __init__(self, border_mode="valid", subsample=(1, 1)):
        super(CorrMM_gradWeights, self).__init__(border_mode, subsample)

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
            height_width = [as_tensor_variable(shape[0]).astype('int64'), as_tensor_variable(shape[1]).astype('int64')]
        else:
            height_width = []

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False]
        dtype = img.type.dtype
        return Apply(self, [img, topgrad] + height_width,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH = padW = -1
        elif self.border_mode == "full":
            padH = padW = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = 0
        dH, dW = self.subsample
        imshp = input_shape[0]
        topshp = input_shape[1]
        ssize, imshp = imshp[1], list(imshp[2:])
        nkern, topshp = topshp[1], list(topshp[2:])
        height_width = node.inputs[-2:]
        if ((dH != 1) or (padH == -1)):
            # vertical subsampling or half padding, kernel height is specified
            kH = height_width[0]
        elif padH == -2:
            # vertical full padding, we can infer the kernel height
            kH = 2 - imshp[0] + (topshp[0] - 1) * dH
        else:
            # explicit padding, we can infer the kernel height
            kH = imshp[0] + 2 * padH - (topshp[0] - 1) * dH
        if ((dW != 1) or (padW == -1)):
            kW = height_width[1]
        elif (padW == -2):
            kW = 2 - imshp[1] + (topshp[1] - 1) * dW
        else:
            kW = imshp[1] + 2 * padW - (topshp[1] - 1) * dW
        return [(nkern, ssize, kH, kW)]

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width = inp[2:] or (None, None)
        weights, = out_
        direction = "backprop weights"
        return super(CorrMM_gradWeights,
                     self).c_code_helper(bottom, weights, top, direction,
                                         sub, height, width)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = CorrMM_gradInputs(self.border_mode,
                                     self.subsample)(weights, top,
                                                     bottom.shape[-2:])
        d_top = CorrMM(self.border_mode,
                       self.subsample)(bottom, weights)
        d_height_width = ((theano.gradient.DisconnectedType()(),) * 2
                          if len(inp) == 4 else ())
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class CorrMM_gradInputs(BaseCorrMM):
    """
    Gradient wrt. inputs for `CorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on
    Theano's automatic differentiation or graph optimization to
    use it as needed.

    """

    def __init__(self, border_mode="valid", subsample=(1, 1)):
        super(CorrMM_gradInputs, self).__init__(border_mode, subsample)

    def make_node(self, kern, topgrad, shape=None):
        kern = as_tensor_variable(kern)
        topgrad = as_tensor_variable(topgrad)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if self.subsample != (1, 1) and shape is None:
            raise ValueError('shape must be given if subsample != (1, 1)')
        height_width = [as_tensor_variable(shape[0]).astype('int64'), as_tensor_variable(shape[1]).astype('int64')] if self.subsample != (1, 1) else []

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False]
        dtype = kern.type.dtype
        return Apply(self, [kern, topgrad] + height_width,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH = padW = -1
        elif self.border_mode == "full":
            padH = padW = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = 0
        dH, dW = self.subsample
        kshp = input_shape[0]
        topshp = input_shape[1]
        ssize, kshp = kshp[1], list(kshp[2:])
        bsize, topshp = topshp[0], list(topshp[2:])
        height_width = node.inputs[-2:]
        if padH == -1:
            padH = kshp[0] // 2
        elif padH == -2:
            padH = kshp[0] - 1
        elif padH < -2:
            raise ValueError('CorrMM_gradInputs: border_mode must be >= 0.')
        if padW == -1:
            padW = kshp[1] // 2
        elif padW == -2:
            padW = kshp[1] - 1
        elif padW < -2:
            raise ValueError('CorrMM_gradInputs: border_mode must be >= 0.')

        if dH != 1:
            out_shp0 = height_width[0]
        else:
            out_shp0 = (topshp[0] - 1) * dH + kshp[0] - 2 * padH
        if dW != 1:
            out_shp1 = height_width[1]
        else:
            out_shp1 = (topshp[1] - 1) * dW + kshp[1] - 2 * padW
        out_shp = (out_shp0, out_shp1)
        return [(bsize, ssize) + out_shp]

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width = inp[2:] or (None, None)
        bottom, = out_
        direction = "backprop inputs"
        return super(CorrMM_gradInputs,
                     self).c_code_helper(bottom, weights, top, direction, sub,
                                         height,
                                         width)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = CorrMM_gradWeights(self.border_mode,
                                       self.subsample)(bottom,
                                                       top,
                                                       weights.shape[-2:])
        d_top = CorrMM(self.border_mode,
                       self.subsample)(bottom, weights)
        d_height_width = ((theano.gradient.DisconnectedType()(),) *
                          2 if len(inp) == 4 else ())
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width
