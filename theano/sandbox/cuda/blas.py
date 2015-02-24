import copy
import os
import logging
_logger = logging.getLogger(__name__)

import theano
from theano import Apply
from theano import tensor
from theano.compat.six import StringIO
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.tensor import as_tensor_variable


class GpuDot22(GpuOp):
    """
    Implement dot(2d, 2d) on the gpu.
    """
    def __str__(self):
        return 'GpuDot22'

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        otype = CudaNdarrayType(
                (x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y], [otype()])

    def c_code_cache_version(self):
        return (1, 2)

    def c_code(self, node, nodename, inputs, outputs, sub):
        x, y = inputs
        z, = outputs
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", %(x)s->nd);
            %(fail)s;
        }
        if (%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", %(y)s->nd);
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
                  CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
                  CudaNdarray_HOST_DIMS(%(y)s)[1])
            || (CudaNdarray_HOST_STRIDES(%(z)s)[0] < 0)
            || (CudaNdarray_HOST_STRIDES(%(z)s)[1] < 0)
            || ((CudaNdarray_HOST_DIMS(%(z)s)[0] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[0] != 1)
                && (CudaNdarray_HOST_DIMS(%(z)s)[1] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[1] != 1)))
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(y)s)[1];
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s) ||
                CudaNdarray_alloc_contiguous(%(z)s, 2, dims))
            {
                if (%(z)s)
                {
                    Py_DECREF(%(z)s);
                    %(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(1.0f, %(x)s, %(y)s, 0.0f, %(z)s))
        {
            if (%(z)s)
            {
                Py_DECREF(%(z)s);
                %(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22 = GpuDot22()


class GpuDot22Scalar(GpuOp):
    """
    Implement dot(2d, 2d) * scalar on the gpu.
    """
    def __str__(self):
        return 'GpuDot22Scalar'

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y, a):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        if not tensor.blas._as_scalar(a):
            raise TypeError(a)
        otype = CudaNdarrayType(
                (x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y, a], [otype()])

    def c_code_cache_version(self):
        return (1, 2)

    def c_code(self, node, name, inputs, outputs, sub):
        x, y, a = inputs
        z, = outputs
        fail = sub['fail']
        return """
        #define REAL float
        float %(name)s_a = (PyArray_TYPE(%(a)s) == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(a)s))[0]);
        #undef REAL
        if (%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", %(x)s->nd);
            %(fail)s;
        }
        if (%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", %(y)s->nd);
            %(fail)s;
        }

        if ((NULL == %(z)s) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
              CudaNdarray_HOST_DIMS(%(x)s)[0]) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
              CudaNdarray_HOST_DIMS(%(y)s)[1])
            || (CudaNdarray_HOST_STRIDES(%(z)s)[0] < 0)
            || (CudaNdarray_HOST_STRIDES(%(z)s)[1] < 0)
            || ((CudaNdarray_HOST_DIMS(%(z)s)[0] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[0] != 1)
                && (CudaNdarray_HOST_DIMS(%(z)s)[1] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[1] != 1)))
        {
            //if (%(z)s) Py_DECREF(%(z)s);
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(y)s)[1];
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s) ||
                CudaNdarray_alloc_contiguous(%(z)s, 2, dims))
            {
                if (%(z)s)
                {
                    Py_DECREF(%(z)s);
                    %(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(%(name)s_a, %(x)s, %(y)s, 0.0f, %(z)s))
        {
            if (%(z)s)
            {
                Py_DECREF(%(z)s);
                %(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22scalar = GpuDot22Scalar()


class GpuGemm(GpuOp):
    """
    implement the gemm on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGemm{inplace}'
        else:
            return 'GpuGemm{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemm
        # is assumed to already have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return (4,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y, b = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """

        #define REAL float
        float %(name)s_a = (PyArray_TYPE(%(a)s) == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(a)s))[0]);

        float %(name)s_b = (PyArray_TYPE(%(b)s) == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(b)s))[0]);
        #undef REAL

        if (%(inplace)s
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] >= 0)
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] >= 0)
            && ((CudaNdarray_HOST_DIMS(%(z_in)s)[0] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 1)
                || (CudaNdarray_HOST_DIMS(%(z_in)s)[1] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] == 1)))
        {
            // The input has an appropriate layout, we work inplace
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (%(z_out)s->nd == 2)
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[1]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[1])
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] >= 0)
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] >= 0)
        // The following condition is needed as this is a condition by cublas
        // on the memory layout of the output it accepts.
                && ((CudaNdarray_HOST_DIMS(%(z_out)s)[0] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 1)
                    || (CudaNdarray_HOST_DIMS(%(z_out)s)[1] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] == 1)))
        {
            // The existing output has an appropriate layout,
            // copy the input data into it, then work inplace
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy the input, use the copy as output
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_gemm(%(name)s_a, %(x)s, %(y)s, %(name)s_b, %(z_out)s))
        {
            %(fail)s;
        }
        """

        return sio.getvalue() % locals()
gpu_gemm_no_inplace = GpuGemm(inplace=False)
gpu_gemm_inplace = GpuGemm(inplace=True)


class GpuGemv(GpuOp):
    """
    implement gemv on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGemv{inplace}'
        else:
            return 'GpuGemv{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemv
        # is assumed to already have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y, b = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """
        float %(name)s_alpha = ((dtype_%(a)s*)(PyArray_DATA(%(a)s)))[0];
        float %(name)s_beta = ((dtype_%(b)s*)(PyArray_DATA(%(b)s)))[0];

        if (%(inplace)s
            && ((CudaNdarray_HOST_STRIDES(%(z_in)s)[0] > 0)
                || ((CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 0)
                    && (CudaNdarray_HOST_DIMS(%(z_in)s)[0] == 1))))
        {
            // Work inplace on the input
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0] ==
                    CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && ((CudaNdarray_HOST_STRIDES(%(z_out)s)[0] > 0)
                    || ((CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 0)
                        && (CudaNdarray_HOST_DIMS(%(z_out)s)[0] == 1))))
        {
            // Work on the output
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_sgemv(%(name)s_alpha, %(x)s, %(y)s,
                              %(name)s_beta, %(z_out)s))
        {
            %(fail)s;
        }
        """
        return sio.getvalue() % locals()
gpu_gemv_no_inplace = GpuGemv(inplace=False)
gpu_gemv_inplace = GpuGemv(inplace=True)


class GpuGer(GpuOp):
    """
    implement ger on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGer{inplace}'
        else:
            return 'GpuGer{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y):
        # the more complicated error checking performed by tensor.ger is
        # assumed to already have been done
        return Apply(self, [z, a, x, y], [z.type()])

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """
        float %(name)s_alpha = ((dtype_%(a)s*)(PyArray_DATA(%(a)s)))[0];

        if (%(inplace)s
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] >= 0)
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] >= 0)
            && ((CudaNdarray_HOST_DIMS(%(z_in)s)[0] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 1)
                || (CudaNdarray_HOST_DIMS(%(z_in)s)[1] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] == 1)))
        {
            // The input has an appropriate layout, we work inplace
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (%(z_out)s->nd == 2)
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[1]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[1])
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] >= 0)
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] >= 0)
                && ((CudaNdarray_HOST_DIMS(%(z_out)s)[0] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 1)
                    || (CudaNdarray_HOST_DIMS(%(z_out)s)[1] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] == 1)))
        {
            // The existing output has an appropriate layout,
            // copy the input data into it, then work inplace
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy the input, use the copy as output
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_sger(%(name)s_alpha, %(x)s, %(y)s, %(z_out)s))
        {
            %(fail)s;
        }
        """
        return sio.getvalue() % locals()
gpu_ger_no_inplace = GpuGer(inplace=False)
gpu_ger_inplace = GpuGer(inplace=True)


class BaseGpuCorrMM(GpuOp):
    """Base class for `GpuCorrMM`, `GpuCorrMM_gradWeights` and
    `GpuCorrMM_gradInputs`. Cannot be used directly.

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
                'do not use pad for BaseGpuCorrMM; please set padding in'
                'border_mode, see the docstring for more details')
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

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 24)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['corr_gemm.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code_helper(self, bottom, weights, top, direction, sub, height=None, width=None):
        """
        This generates the C code for GpuCorrMM (direction="forward"),
        GpuCorrMM_gradWeights (direction="backprop weights"), and
        GpuCorrMM_gradInputs (direction="backprop inputs").
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
    
    CudaNdarray * bottom = %(bottom)s;
    CudaNdarray * weights = %(weights)s;
    CudaNdarray * top = %(top)s;
    CudaNdarray * out2 = NULL;

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = CudaNdarray_HOST_DIMS(weights)[2];
        kW = CudaNdarray_HOST_DIMS(weights)[3];
    }
    else {
        if ((dH != 1) || (padH == -1)) {
            // vertical subsampling or half padding, kernel height is specified
            kH = %(height)s;
        }
        else if (padH == -2) {
            // vertical full padding, we can infer the kernel height
            kH = 2 - CudaNdarray_HOST_DIMS(bottom)[2] + (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH;
        }
        if ((dW != 1) || (padW == -1)) {
            kW = %(width)s;
        }
        else if (padW == -2) {
            kW = 2 - CudaNdarray_HOST_DIMS(bottom)[3] + (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW;
        }
        else {
            kW = CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW;
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
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padW must be >= -2");
        %(fail)s
    }

    // Infer output shape
    int out_dim[4];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + 2*pad - weight) / sample + 1
        out_dim[0] = CudaNdarray_HOST_DIMS(bottom)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[0];
        out_dim[2] = (CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - CudaNdarray_HOST_DIMS(weights)[2]) / dH + 1;
        out_dim[3] = (CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - CudaNdarray_HOST_DIMS(weights)[3]) / dW + 1;
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width)
        // height and width: weights = bottom + 2*pad - (top - 1) * sample
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[1];
        out_dim[1] = CudaNdarray_HOST_DIMS(bottom)[1];
        out_dim[2] = kH;  // already inferred further above
        out_dim[3] = kW;  // how convenient
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + weights - 2*pad
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[1];
        out_dim[2] = (dH != 1) ? %(height)s : (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH + CudaNdarray_HOST_DIMS(weights)[2] - 2*padH;
        out_dim[3] = (dW != 1) ? %(width)s : (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW + CudaNdarray_HOST_DIMS(weights)[3] - 2*padW;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    if ( !(%(out)s
           && %(out)s->nd==4
           && CudaNdarray_is_c_contiguous(%(out)s)
           && CudaNdarray_HOST_DIMS(%(out)s)[0]==out_dim[0]
           && CudaNdarray_HOST_DIMS(%(out)s)[1]==out_dim[1]
           && CudaNdarray_HOST_DIMS(%(out)s)[2]==out_dim[2]
           && CudaNdarray_HOST_DIMS(%(out)s)[3]==out_dim[3]))
    {
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray*)CudaNdarray_NewDims(4,out_dim);
        if (NULL == %(out)s)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseGpuCorrMM: Failed to allocate output of %%d x %%d x %%d x %%d",
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


class GpuCorrMM(BaseGpuCorrMM):
    """GPU correlation implementation using Matrix Multiplication.

    :param border_mode: currently supports "valid" only; "full" can be
        simulated by setting `pad="full"` (at the cost of performance), or
        by using `GpuCorrMM_gradInputs`
    :param subsample: the subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `GpuCorrMM(...)(...)[:,:,::sv, ::sh]`,
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
        to automatically replace all convolution operations with `GpuCorrMM`
        or one of its gradients, or you can use it as a replacement for
        :func:`conv2d <theano.tensor.nnet.conv.conv2d>`, called as
        `GpuCorrMM(subsample=...)(image, filters)`. The latter is currently
        faster, but note that it computes a correlation -- if you need to
        compute a convolution, flip the filters as `filters[:,:,::-1,::-1]`.

    :warning: For 700 series Nvidia GPUs of compute capability 3.5 and CUDA 5.0
        to 6.0, there is a bug in CUBLAS' matrix multiplication function that
        can make GpuCorrMM or its gradients crash for some input and filter
        shapes. So if you have a Tesla K20, Tesla K40, Quadro K6000, GeForce GT
        640 (DDR5), GeForce GTX 780 (or Ti), GeForce GTX TITAN (or Black or Z)
        and experience a crash, switching to CUDA 6.5 or CUDA 4.2 should fix it.
        If this is not possible, changing the input or filter shapes (e.g., the
        batchsize or number of filters) may also work around the CUBLAS bug.
    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 pad=(0, 0)):
        super(GpuCorrMM, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, kern):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(GpuCorrMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = GpuCorrMM_gradInputs(self.border_mode, self.subsample)(
            weights, top, bottom.shape[-2:])
        d_weights = GpuCorrMM_gradWeights(self.border_mode, self.subsample)(
            bottom, top, weights.shape[-2:])
        return d_bottom, d_weights


class GpuCorrMM_gradWeights(BaseGpuCorrMM):
    """Gradient wrt. filters for `GpuCorrMM`.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self, border_mode="valid",
            subsample=(1, 1),
            pad=(0, 0)):
        super(GpuCorrMM_gradWeights, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, topgrad, shape=None):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
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
        return Apply(self, [img, topgrad] + height_width, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width = inp[2:] or (None, None)
        weights, = out_
        direction = "backprop weights"
        return super(GpuCorrMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        weights = gpu_contiguous(weights)
        d_bottom = GpuCorrMM_gradInputs(self.border_mode, self.subsample)(
                weights, top, bottom.shape[-2:])
        d_top = GpuCorrMM(self.border_mode, self.subsample)(
                bottom, weights)
        d_height_width = (theano.gradient.DisconnectedType()(),) * 2 if len(inp) == 4 else ()
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class GpuCorrMM_gradInputs(BaseGpuCorrMM):
    """Gradient wrt. inputs for `GpuCorrMM`.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self, border_mode="valid",
            subsample=(1, 1),
            pad=(0, 0)):
        super(GpuCorrMM_gradInputs, self).__init__(border_mode, subsample, pad)

    def make_node(self, kern, topgrad, shape=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if self.subsample != (1, 1) and shape is None:
            raise ValueError('shape must be given if subsample != (1, 1)')
        height_width = [shape[0], shape[1]] if self.subsample != (1, 1) else []

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False]
        return Apply(self, [kern, topgrad] + height_width, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width = inp[2:] or (None, None)
        bottom, = out_
        direction = "backprop inputs"
        return super(GpuCorrMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        bottom = gpu_contiguous(bottom)
        d_weights = GpuCorrMM_gradWeights(self.border_mode, self.subsample)(
                bottom, top, weights.shape[-2:])
        d_top = GpuCorrMM(self.border_mode, self.subsample)(
                bottom, weights)
        d_height_width = (theano.gradient.DisconnectedType()(),) * 2 if len(inp) == 4 else ()
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class BaseGpuCorr3dMM(GpuOp):
    """Base class for `GpuCorr3dMM`, `GpuCorr3dMM_gradWeights` and
    `GpuCorr3dMM_gradInputs`. Cannot be used directly."""
    __props__ = ('border_mode', 'subsample', 'pad')

    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 pad=(0, 0, 0)):
        if border_mode != "valid":
            raise ValueError("border_mode must be 'valid'")
        self.border_mode = border_mode
        if len(subsample) != 3:
            raise ValueError("subsample must have three elements")
        self.subsample = subsample
        if (pad not in ("half", "full")) and (len(pad) != 3):
            raise ValueError("pad must be 'half', 'full', or have three elements")
        self.pad = pad

    def __str__(self):
        return '%s{%s, %s, pad=%r}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            self.pad)

    def flops(self, inp, outp):
        """ Useful with the hack in profilemode to print the MFlops"""
        # if the output shape is correct, then this gives the correct
        # flops for any direction, sampling, padding, and border mode
        inputs, filters = inp
        outputs, = outp
        assert inputs[1] == filters[1]
        # nb mul and add by output pixel
        flops = filters[2] * filters[3] * filters[4] * 2
        # nb flops by output image
        flops *= outputs[2] * outputs[3] * outputs[4]
        # nb patch multiplied
        flops *= inputs[1] * filters[0] * inputs[0]
        return flops

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 23)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['corr3d_gemm.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code_helper(self, bottom, weights,
                      top, direction,
                      sub,
                      height=None, width=None, depth=None):
        """
        This generates the C code for GpuCorrMM (direction="forward"),
        GpuCorrMM_gradWeights (direction="backprop weights"), and
        GpuCorrMM_gradInputs (direction="backprop inputs").
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
            of the filters for direction="backprop weights" or the height of the
            input images for direction="backprop inputs".
            If self.pad == 'half', a variable giving the height of the filters
            for direction="backprop weights".
            Ignored otherwise.
        :param width: If self.subsample[1] != 1, a variable giving the width
            of the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".
            If self.pad == 'half', a variable giving the width of the filters
            for direction="backprop weights".
            Ignored otherwise.
        :param depth: If self.subsample[2] != 1, a variable giving the depth
            of the filters for direction="backprop weights" or the depth of the
            input images for direction="backprop inputs".
            If self.pad == 'half', a variable giving the depth of the filters
            for direction="backprop weights".
            Ignored otherwise.
        """
        if self.border_mode != "valid":
            raise ValueError("mode must be 'valid'")
        dH, dW, dD = self.subsample
        if self.pad == "half":
            padH = padW = padD = -1
        elif self.pad == "full":
            padH = padW = padD =-2
        else:
            padH, padW, padD = self.pad
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
        if ((direction != 0) and (dD != 1)) or ((direction == 1) and (padD == -1)):
            if not depth:
                raise ValueError("depth must be given for backprop with horizontal sampling or pad='half'")
            depth = '(*(npy_int*)(PyArray_DATA(%s)))' % depth
        else:
            depth = 'NULL'
        sub = sub.copy()
        sub.update(locals())

        return """
    // Mandatory args
    int direction = %(direction)s;  // forward, bprop weights, bprop inputs

    // Optional args
    int dH = %(dH)s;
    int dW = %(dW)s;
    int dD = %(dD)s;
    int padH = %(padH)s;
    int padW = %(padW)s;
    int padD = %(padD)s;

    CudaNdarray * bottom = %(bottom)s;
    CudaNdarray * weights = %(weights)s;
    CudaNdarray * top = %(top)s;
    CudaNdarray * out2 = NULL;

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW, kD;
    if (direction != 1)
    {
      // weight is an input variable, we can just read its shape
      kH = CudaNdarray_HOST_DIMS(weights)[2];
      kW = CudaNdarray_HOST_DIMS(weights)[3];
      kD = CudaNdarray_HOST_DIMS(weights)[4];
    }
    else
    {
      if ((dH != 1) || (padH == -1))
      {
         // vertical subsampling or half padding, kernel height is specified
         kH = %(height)s;
      }
      else if (padH == -2)
      {
        // vertical full padding, we can infer the kernel height
        kH = 2 - CudaNdarray_HOST_DIMS(bottom)[2] + (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH;
      }
      else
      {
        // explicit padding, we can infer the kernel height
        kH = CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH;
      }
      if ((dW != 1) || (padW == -1))
      {
        kW = %(width)s;
      }
      else if (padW == -2)
      {
         kW = 2 - CudaNdarray_HOST_DIMS(bottom)[3] + (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW;
      }
      else
      {
        kW = CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW;
      }
      if ((dD != 1) || (padD == -1))
      {
        kD = %(depth)s;
      }
      else if (padD == -2)
      {
         kD = 2 - CudaNdarray_HOST_DIMS(bottom)[4] + (CudaNdarray_HOST_DIMS(top)[4] - 1) * dD;
      }
      else
      {
        kD = CudaNdarray_HOST_DIMS(bottom)[4] + 2*padD - (CudaNdarray_HOST_DIMS(top)[4] - 1) * dD;
      }
    }

    // Auto-padding if requested
    if (padH == -1)
    { // vertical half padding
      padH = kH / 2;
    }
    else if (padH == -2)
    { // vertical full padding
      padH = kH - 1;
    }
    else if (padH < 0)
    {
      PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padH must be >= -2");
      %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
      padW = kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
      padW = kW - 1;
    }
    else if (padW < 0)
    {
      PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padW must be >= -2");
      %(fail)s
    }
    if (padD == -1)
    { // horizontal half padding
      padD = kD / 2;
    }
    else if (padD == -2)
    { // horizontal full padding
      padD = kD - 1;
    }
    else if (padD < 0)
    {
      PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padD must be >= -2");
      %(fail)s
    }

    // Infer output shape
    int out_dim[5];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width, depth)
        // height and width: top = (bottom + 2*pad - weight) / sample + 1
        out_dim[0] = CudaNdarray_HOST_DIMS(bottom)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[0];
        out_dim[2] = (CudaNdarray_HOST_DIMS(bottom)[2] + 2*padH - CudaNdarray_HOST_DIMS(weights)[2]) / dH + 1;
        out_dim[3] = (CudaNdarray_HOST_DIMS(bottom)[3] + 2*padW - CudaNdarray_HOST_DIMS(weights)[3]) / dW + 1;
        out_dim[4] = (CudaNdarray_HOST_DIMS(bottom)[4] + 2*padD - CudaNdarray_HOST_DIMS(weights)[4]) / dD + 1;
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width, depth)
        // height, width and depth: weights = bottom + 2*pad - (top-1) * sample
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[1];
        out_dim[1] = CudaNdarray_HOST_DIMS(bottom)[1];
        out_dim[2] = kH;  // already inferred further above
        out_dim[3] = kW;  // how convenient
        out_dim[4] = kD;
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width, depth)
        // height, width and depth: bottom = (top-1) * sample + weights - 2*pad
        out_dim[0] = CudaNdarray_HOST_DIMS(top)[0];
        out_dim[1] = CudaNdarray_HOST_DIMS(weights)[1];
        out_dim[2] = (dH != 1) ? %(height)s : (CudaNdarray_HOST_DIMS(top)[2] - 1) * dH + CudaNdarray_HOST_DIMS(weights)[2] - 2*padH;
        out_dim[3] = (dW != 1) ? %(width)s : (CudaNdarray_HOST_DIMS(top)[3] - 1) * dW + CudaNdarray_HOST_DIMS(weights)[3] - 2*padW;
        out_dim[4] = (dD != 1) ? %(depth)s : (CudaNdarray_HOST_DIMS(top)[4] - 1) * dD + CudaNdarray_HOST_DIMS(weights)[4] - 2*padD;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }



    // Prepare output array
    if (!(%(out)s
          && %(out)s->nd == 5
          && CudaNdarray_is_c_contiguous(%(out)s)
          && CudaNdarray_HOST_DIMS(%(out)s)[0] == out_dim[0]
          && CudaNdarray_HOST_DIMS(%(out)s)[1] == out_dim[1]
          && CudaNdarray_HOST_DIMS(%(out)s)[2] == out_dim[2]
          && CudaNdarray_HOST_DIMS(%(out)s)[3] == out_dim[3]
          && CudaNdarray_HOST_DIMS(%(out)s)[4] == out_dim[4]))
    {
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray*)CudaNdarray_NewDims(5, out_dim);
        if (NULL == %(out)s)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "BaseGpuCorr3dM: Failed to allocate output of %%d x %%d x %%d x %%d x %%d",
                    out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4]);
            %(fail)s
        }
    }

    // Call CUDA code
    out2 = corr3dMM(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, dD, padH, padW, padD);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class GpuCorr3dMM(BaseGpuCorr3dMM):
    """GPU correlation implementation using Matrix Multiplication.

    :warning: For 700 series Nvidia GPUs of compute capability 3.5 and CUDA 5.0
        to 6.0, there is a bug in CUBLAS' matrix multiplication function that
        can make GpuCorrMM or its gradients crash for some input and filter
        shapes. So if you have a Tesla K20, Tesla K40, Quadro K6000, GeForce GT
        640 (DDR5), GeForce GTX 780 (or Ti), GeForce GTX TITAN (or Black or Z)
        and experience a crash, switching to CUDA 6.5 or CUDA 4.2 should fix it.
        If this is not possible, changing the input or filter shapes (e.g., the
        batchsize or number of filters) may also work around the CUBLAS bug.
    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 pad=(0, 0, 0)):
        """
        :param border_mode: currently supports "valid" only; "full" can be
            simulated by setting `pad="full"` (at the cost of performance), or
            by using `GpuCorrMM_gradInputs`
        :param subsample: the subsample operation applied to each output image.
            Should be a tuple with 3 elements.
            `(sv, sh, sl)` is equivalent to `GpuCorrMM(...)(...)[:,:,::sv, ::sh, ::sl]`,
            but faster.
            Set to `(1, 1, 1)` to disable subsampling.
        :param pad: the width of a border of implicit zeros to pad the input
            image with. Should be a tuple with 3 elements giving the numbers of
            rows and columns to pad on each side, or "half" to set the padding
            to `(kernel_rows // 2, kernel_columns // 2, kernel_depth // 2)`, or "full" to set the
            padding to `(kernel_rows - 1, kernel_columns - 1, kernel_depth - 1)` at runtime.
            Set to `(0, 0, 0)` to disable padding.

        :note: Currently, the Op requires the inputs, filters and outputs to be
            C-contiguous. Use :func:`gpu_contiguous
            <theano.sandbox.cuda.basic_ops.gpu_contiguous>` on these arguments
            if needed.
        """
        super(GpuCorr3dMM, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, kern):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(GpuCorr3dMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode, self.subsample, self.pad)(
                weights, top, bottom.shape[-3:])
        d_weights = GpuCorr3dMM_gradWeights(self.border_mode, self.subsample, self.pad)(
                bottom, top, weights.shape[-3:])
        return d_bottom, d_weights


class GpuCorr3dMM_gradWeights(BaseGpuCorr3dMM):
    """Gradient wrt. filters for `GpuCorr3dMM`.

    :note: You will not want to use this directly, but rely on Theano's
        automatic differentiation or graph optimization to use it as needed.
    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 pad=(0, 0, 0)):
        super(GpuCorr3dMM_gradWeights, self).__init__(border_mode, subsample, pad)

    def make_node(self, img, topgrad, shape=None):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if shape is not None:
            shape = as_tensor_variable(shape)

        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if self.subsample != (1, 1, 1) or self.pad == "half":
            if shape is None:
                raise ValueError('shape must be given if subsample != (1, 1, 1), or pad == "half"')
            height_width_depth = [shape[0], shape[1], shape[2]]
        else:
            height_width_depth = []

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False, False]
        return Apply(self, [img, topgrad] + height_width_depth, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width, depth = inp[2:] or (None, None, None)
        weights, = out_
        direction = "backprop weights"
        return super(GpuCorr3dMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width, depth)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        weights = gpu_contiguous(weights)
        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode, self.subsample, self.pad)(weights, top, bottom.shape[-3:])
        d_top = GpuCorr3dMM(self.border_mode, self.subsample, self.pad)(
            bottom, weights)
        d_height_width_depth = (theano.gradient.DisconnectedType()(),) * 3 if len(inp) == 5 else ()
        return (d_bottom, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth

class GpuCorr3dMM_gradInputs(BaseGpuCorr3dMM):
    """Gradient wrt. inputs for `GpuCorr3dMM`.

    :note: You will not want to use this directly, but rely on Theano's
        automatic differentiation or graph optimization to use it as needed.
    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 pad=(0, 0, 0)):
        super(GpuCorr3dMM_gradInputs, self).__init__(border_mode, subsample, pad)

    def make_node(self, kern, topgrad, shape=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if self.subsample != (1, 1, 1) and shape is None:
            raise ValueError('shape must be given if subsample != (1, 1, 1)')
        height_width_depth = [shape[0], shape[1], shape[2]] if self.subsample != (1, 1, 1) else []

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False, False]
        return Apply(self, [kern, topgrad] + height_width_depth, [CudaNdarrayType(broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width, depth = inp[2:] or (None, None, None)
        bottom, = out_
        direction = "backprop inputs"
        return super(GpuCorr3dMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width, depth)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        bottom = gpu_contiguous(bottom)
        d_weights = GpuCorr3dMM_gradWeights(self.border_mode, self.subsample, self.pad)(
            bottom, top, weights.shape[-3:])
        d_top = GpuCorr3dMM(self.border_mode, self.subsample, self.pad)(
                bottom, weights)
        d_height_width_depth = (theano.gradient.DisconnectedType()(),) * 3 if len(inp) == 5 else ()
        return (d_weights, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth


##
# Not really a BLAS operation, but whatever.
#

class GpuConv(GpuOp):
    """
    Implement the batched and stacked 2d convolution on the gpu.
    """
    check_broadcast = False

    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode,
            subsample=(1, 1),
            logical_img_hw=None,
            logical_kern_hw=None,
            logical_kern_align_top=True,
            version=-1,
            direction_hint=None,
            verbose=0,
            kshp=None,
            imshp=None,
            max_threads_dim0=None,
            nkern=None,
            bsize=None,
            fft_opt=True):
        """
        :param version: each version of c_code implements many kernel for the
                        convolution. By default we try to guess the best one.
                        You can force one version with this parameter. This
                        parameter is used by the tests.
        :param direction_hint: 'forward', 'bprop weights' or 'bprop inputs'.
                        Serves as a hint for graph optimizers replacing
                        GpuConv by other implementations. If the GpuConv is
                        inserted automatically, we take its value from ConvOp.
        :param verbose: for value of 1,2 and 3. Print more information during
                        the execution of the convolution. Mostly used for
                        optimization or debugging.
        :param kshp:    The size of the kernel. If provided, can generate
                        faster code. If the GpuConv op is automatically
                        inserted,
                        we take its value automatically from the Conv op.
        :param imshp:   The size of the image. Not used for code generation but
                        allows to select an experimental new version in another
                        repo.
        :param max_threads_dim0: The maximum number of threads for the
                        block size dimensions 0 (blockDim.x) used by the
                        GPU function.
        :param nkern:   The number of kernels. Not used for this op, but can be
                        used by graph optimizers to select a more optimal
                        convolution implementation. If the GpuConv op is inserted
                        automatically, we take its value from the Conv op.
        :param bsize:   The batch size. Not used for this op, but can be
                        used by graph optimizers to select a more optimal
                        convolution implementation. If the GpuConv op is inserted
                        automatically, we take its value from the Conv op.
        :param fft_opt: deactivate fft_opt optimization at the op level when
                        set to False. Note that by default fft optimization
                        aren't enabled. See
                        :ref:`convolution documentation <libdoc_tensor_nnet_conv>`
                        to enable them.
        """
        self.border_mode = border_mode
        if version != -1:
            raise Exception(
                """GpuConv with version!=-1 is disabled as we do not
                test it anymore. It probably work, so you probably can
                just comment this error and use it. But we want to
                make sure you know about that. Also, this Op is pretty
                slow and isn't used by default anymore. We strongly
                suggest to use GpuCorrMM that is much faster and
                implement all the functionality (at a cost of some
                extra memory usage). If you can use cuDNN, that is
                even better.
                """)
        self.subsample = subsample
        if logical_img_hw is not None:
            h, w = logical_img_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_img_hw = tuple(logical_img_hw)
        self.logical_img_hw = logical_img_hw
        if logical_kern_hw is not None:
            h, w = logical_kern_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_hw = logical_kern_hw
        self.logical_kern_align_top = logical_kern_align_top
        self.version = version
        self.direction_hint = direction_hint
        self.verbose = verbose
        self.kshp = kshp
        self.imshp = imshp
        self.max_threads_dim0 = max_threads_dim0
        self.nkern = nkern
        self.bsize = bsize
        self.fft_opt = fft_opt

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.border_mode == other.border_mode \
            and self.subsample == other.subsample \
            and self.logical_img_hw == other.logical_img_hw \
            and self.logical_kern_hw == other.logical_kern_hw \
            and self.logical_kern_align_top == other.logical_kern_align_top \
            and self.version == other.version \
            and self.verbose == other.verbose \
            and self.kshp == other.kshp\
            and self.imshp == other.imshp\
            and self.max_threads_dim0 == other.max_threads_dim0

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "imshp"):
            self.imshp = None
        if not hasattr(self, "max_threads_dim0"):
            self.max_threads_dim0 = None
        if not hasattr(self, "direction_hint"):
            self.direction_hint = None

    def __hash__(self):
        # don't use hash(self.version) as hash(-1)==-2 and
        # hash(-2)==-2 in python!
        return hash(type(self)) \
            ^ hash(self.border_mode) \
            ^ hash(self.subsample) \
            ^ hash(self.logical_img_hw) \
            ^ hash(self.logical_kern_hw) \
            ^ hash(self.logical_kern_align_top) \
            ^ self.version \
            ^ hash(self.verbose) \
            ^ hash(self.kshp)\
            ^ hash(self.imshp)\
            ^ hash(self.max_threads_dim0)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.logical_img_hw),
            str(self.logical_kern_hw),
            str(self.logical_kern_align_top),
            str(self.imshp),
            str(self.kshp))

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
        images, kerns = inputs
        out, = outputs
        assert images[1] == kerns[1]
        flops = 0
        if self.border_mode == "valid":
            # nb mul and add by output pixel
            flops = kerns[2] * kerns[3] * 2
            # nb flops by output image
            flops *= out[2] * out[3]
            # nb patch multiplied
            flops *= images[1] * kerns[0] * images[0]
        else:
            flops = (images[0] * kerns[0] * images[1] *
                     kerns[2] * kerns[3] *
                     images[2] * images[3] * 2)
        return flops

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node_.op.max_threads_dim0 = prop['maxThreadsDim0']
        return super(GpuConv, node_.op).make_thunk(node_, storage_map,
                                                   compute_map, no_recycling)

    def c_compile_args(self):
        nb = 0
        if (self.kshp is not None) and (self.kshp[1] is not None):
            nb = self.kshp[1]
        return ['-DTHEANO_KERN_WID=' + str(nb)]  # ,'-g','-G']

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 22)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['conv_kernel.cu', 'conv_full_kernel.cu', 'conv.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code(self, node, nodename, inp, out_, sub):
        img, kern = inp
        out, = out_
        dx = self.subsample[0]
        dy = self.subsample[1]
        version = self.version
        verbose = self.verbose
        sub = sub.copy()
        max_threads_dim0 = self.max_threads_dim0
        if self.border_mode == "valid":
            bmode = 1
        else:
            assert self.border_mode == "full"
            bmode = 0
        if max_threads_dim0 is None:
            raise NotImplementedError("GpuConv.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")
        sub.update(locals())
        return """
    //Mandatory args
    int mode = %(bmode)s;

    //Optional args
    int version = %(version)s;
    int verbose = %(verbose)s;
    int dx = %(dx)s;
    int dy = %(dy)s;


    // TODO, make out be decref before we alloc out2!
    CudaNdarray * out2 = (CudaNdarray *)CudaNdarray_Conv(%(img)s, %(kern)s,
                                                         %(out)s, mode,
                                                         dx, dy,
                                                         version, verbose,
                                                         %(max_threads_dim0)s);
    Py_XDECREF(%(out)s);
    %(out)s = out2;

    if (%(out)s==NULL){
        %(fail)s
    }
""" % sub


class GpuDownsampleFactorMax(GpuOp):
    """
    Implement downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border=False):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError()
        if not x.type.ndim == 4:
            raise TypeError()
        return Apply(self, [x], [x.type()])

    #def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')
    def c_code_cache_version(self):
        return (6)

    def c_code(self, node, nodename, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        int dims[4], xdim2, xdim3;
        if (%(x)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError,
                            "GpuDownsampleFactorMax: rank error");
            %(fail)s;
        }
        xdim2 = CudaNdarray_HOST_DIMS(%(x)s)[2];
        xdim3 = CudaNdarray_HOST_DIMS(%(x)s)[3];
        dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
        dims[1] = CudaNdarray_HOST_DIMS(%(x)s)[1];
        dims[2] = xdim2 / %(ds0)s;
        dims[3] = xdim3 / %(ds1)s;
        if (! %(ignore_border)s)
        {
            dims[2] += (xdim2%%(%(ds0)s)?1:0);
            dims[3] += (xdim3%%(%(ds1)s)?1:0);
        }
        if(dims[3]>512){
            PyErr_Format(PyExc_ValueError,
                         "GpuDownsampleFactorMax: last dimention size of %%d"
                         " is bigger then 512. This case is not implemented.",
                         dims[3]);
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != dims[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != dims[1])
            || (CudaNdarray_HOST_DIMS(%(z)s)[2] != dims[2])
            || (CudaNdarray_HOST_DIMS(%(z)s)[3] != dims[3]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s)
                || CudaNdarray_alloc_contiguous(%(z)s, 4, dims))
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                PyErr_SetString(PyExc_ValueError,
                                "GpuDownsampleFactorMax:"
                                "Was not able to allocate output!");
                %(fail)s;
            }
        }
        {
            dim3 grid(std::min(dims[0] * dims[1], 65535),
                      dims[2]);
            //dim3 block(std::min(dims[3], 512));
            //TODO: implement this by supporting more outputs than threads
            dim3 block(dims[3]);
            if ((grid.x*grid.y) && dims[3])
            kMaxPool_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block,
                                                       xdim3*sizeof(float)>>>(
                dims[0], dims[1], dims[2], dims[3], xdim2, xdim3,
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s. (grid: %%i x %%i;"
                    " block: %%i x %%i x %%i)\\n",
                    "kMaxPool_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        return """
        template<int pf2, int pf3>
        __global__ void kMaxPool_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           float *z, int zS0, int zS1, int zS2, int zS3)
        {
            float cur_max, cur_x;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            for(int block_x_idx = blockIdx.x;
                block_x_idx < D0 * D1;
                block_x_idx += gridDim.x){

                int i0 = block_x_idx %% D0;
                int i1 = block_x_idx / D0;
                int i2 = blockIdx.y;

                extern __shared__ float xbuf[]; //size [xD3]

                for (int r2 = 0;
                     (r2 < pf2) && (%(ignore_border)s || (r2 + i2*pf2 < xD2));
                     ++r2)
                {
                    __syncthreads();
                    // load the current row of the image into shared memory
                    for (int j = tx; j < xD3; j += blockDim.x)
                    {
                        xbuf[j] = x[i0*xS0 + i1*xS1 + (i2*pf2+r2)*xS2 + j*xS3];
                    }
                    __syncthreads();

                    // initialize our max if this is the
                    // first row we're loading
                    cur_max = (r2 == 0) ? xbuf[tx*pf3] : cur_max;

                    // do a mini-reduction over the pf3 relevant elements
                    // in the current row

                    if (%(ignore_border)s)
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            cur_x = xbuf[tx*pf3+k];
                            cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            if (tx*pf3 + k < xD3)
                            {
                                cur_x = xbuf[tx*pf3+k];
                                cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                            }
                        }
                    }
                }

                z[i0*zS0 + i1*zS1 + i2*zS2 + tx*zS3] = cur_max;
            }
        }
        """ % locals()


class GpuDownsampleFactorMaxGrad(GpuOp):
    """
    Implement the grad of downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x, z, gz):
        return Apply(self, [x, z, gz], [x.type()])

    def c_code_cache_version(self):
        return (9,)

    def c_code(self, node, nodename, inp, out, sub):
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        if (%(x)s->nd != 4
            || %(z)s->nd != 4
            || %(gz)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (CudaNdarray_HOST_DIMS(%(gx)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[2] !=
                CudaNdarray_HOST_DIMS(%(x)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[3] !=
                CudaNdarray_HOST_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gx)s)
                || CudaNdarray_alloc_contiguous(%(gx)s, 4,
                                                CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            //TODO: supporting more output columns than threads
            // make sure we cover every x row when ignore border isset and
            // there's a border present to be ignored
            int needs_extra_z_col = %(ignore_border)s && (CudaNdarray_HOST_DIMS(%(x)s)[2] %% %(ds0)s);
            dim3 grid(std::min(CudaNdarray_HOST_DIMS(%(z)s)[0], 65535),
                      CudaNdarray_HOST_DIMS(%(z)s)[2] + (needs_extra_z_col ? 1 : 0));
            dim3 block(std::min(CudaNdarray_HOST_DIMS(%(x)s)[3], 512));

            kDownsampleMaxGrad_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[2],
                CudaNdarray_HOST_DIMS(%(z)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_DEV_DATA(%(gz)s),
                CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                CudaNdarray_DEV_DATA(%(gx)s),
                CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                CudaNdarray_HOST_STRIDES(%(gx)s)[3]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
    "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kDownsampleMaxGrad_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        # This code considers every position in the output z, andthen
        # computes the gradient for the input pixels that were
        # downsampled to that z-position. It does so by running along
        # every z row (sometimes plus one, to make sure every gx row
        # gets totally filled), and by running along every x col. This
        # code is not sensitive to the ignore_border flag along the
        # row dimension (since it runs for every position in the
        # output z), but it is sensitive along the col dimension.
        ignore_border = int(self.ignore_border)

        return """
        // ds0 is the downsampling factor in rows, ds1 in columns
        template<int ds0, int ds1>
        __global__ void kDownsampleMaxGrad_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           const float * z, int zS0, int zS1, int zS2, int zS3,
           const float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
           float *gx, int gxS0, int gxS1, int gxS2, int gxS3)
        {
            //  D0: number of image rows
            //  D1: number of image cols
            //  D2: number of z rows
            //  D3: number of z cols
            // xD2: number of x rows
            // xD3: number of x cols
            // various .S. variables are strides

            float cur_max, cur_x, my_z, my_gz;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            int bdimx = blockDim.x;

            for(int i0 = blockIdx.x;
                i0 < D0;
                i0 += gridDim.x){

                int i1 = 0;                // image col
                // row wrt z and/or gz, ranges from 0 to D2 - 1 OR D2
                // (as needed to cover all x rows)
                int i2 = blockIdx.y;
                int x_col = tx;            // col wrt x, ranges from 0 to xD3 - 1
                int z_col = x_col/ds1;     // z_col corresponding to this x_col


                //TODO: raise occupancy.  Use threadIdx.y to run several
                //      iterations of this i1 loop in parallel

                for (i1 = 0; i1 < D1; ++i1) // loop over images (same for z and x)
                {
                    for(int col_iter = 0;
                        (tx + col_iter * bdimx < xD3) ; col_iter++){

                        //The if inside is to don't do the division if we
                        // need only 1 col_iter

                        if(tx + bdimx < xD3)
                        {
                            x_col = tx + col_iter * bdimx;
                            z_col = x_col/ds1;
                        }

                        if (%(ignore_border)s && ((x_col >= ds1 * D3) || (i2 >= D2)))
                        {
                            // This happens only if x_col, or i2*ds0, was ignored
                            // (via ignore_border)
                            // TODO: if ignore_border is False, this is impossible
                            //        and we don't even need to generate this code.

                            my_gz = 0.0f;

                            //any fp number suffices for my_z, so we don't even
                            //need to set it to anything in particular.

                        }
                        else
                        {
                            // this is effectively:
                            // my_gz = gz[image_row][image_col][z_row][z_col]
                            // my_z  = z[image_row][image_col][z_row][z_col]
                            my_gz = gz[i0 * gzS0 + i1 * gzS1 + i2 * gzS2 +
                                       z_col*gzS3];
                            my_z =   z[i0 *  zS0 + i1 *  zS1 + i2 *  zS2 +
                                       z_col* zS3];
                        }
                        for (int x_row = i2*ds0;
                              (x_row < i2*ds0+ds0) && (x_row < xD2); ++x_row)
                        {
                            // this is effectively:
                            // gx[image_row][image_col][x_row][x_col]
                            //   = (my_z == x[image_row][image_col][
                            //                x_row][x_col]) ? my_gz : 0.0f;
                            gx[i0*gxS0 + i1*gxS1 + x_row*gxS2 + x_col*gxS3]
                               = (my_z == x[i0*xS0 + i1*xS1 + x_row*xS2 +
                                            x_col*xS3]) ? my_gz : 0.0f;
                        }

                    }
                }
            }
        }
        """ % locals()


class GpuDownsampleFactorMaxGradGrad(GpuOp):
    """
    Implement the grad of downsample with max on the gpu.
    """
    __props__ = ('ds', 'ignore_border')
    
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def make_node(self, x, z, gx):
        x = as_cuda_ndarray_variable(x)
        z = as_cuda_ndarray_variable(z)
        gx = as_cuda_ndarray_variable(gx)
        
        if x.type.ndim != 4:
            raise TypeError('x must be 4D tensor')
        if z.type.ndim != 4:
            raise TypeError('z must be 4D tensor')
        if gx.type.ndim != 4:
            raise TypeError('gx must be 4D tensor')
        
        return Apply(self, [x, z, gx], [x.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, nodename, inp, out, sub):
        x, z, gx = inp
        gz, = out
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        if (%(x)s->nd != 4
            || %(z)s->nd != 4
            || %(gx)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMaxGradGrad: rank error");
            %(fail)s;
        }
        if ((NULL == %(gz)s)
            || (CudaNdarray_HOST_DIMS(%(gz)s)[0] !=
                CudaNdarray_HOST_DIMS(%(z)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[1] !=
                CudaNdarray_HOST_DIMS(%(z)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[2] !=
                CudaNdarray_HOST_DIMS(%(z)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[3] !=
                CudaNdarray_HOST_DIMS(%(z)s)[3]))
        {
            Py_XDECREF(%(gz)s);
            %(gz)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gz)s)
                || CudaNdarray_alloc_contiguous(%(gz)s, 4,
                                                CudaNdarray_HOST_DIMS(%(z)s)))
            {
                Py_XDECREF(%(gz)s);
                %(gz)s = NULL;
                %(fail)s;
            }
        }
        {
            
            int needs_extra_z_col = %(ignore_border)s && (CudaNdarray_HOST_DIMS(%(x)s)[2] %% %(ds0)s);
            dim3 grid(std::min(CudaNdarray_HOST_DIMS(%(z)s)[0], 65535),
                      CudaNdarray_HOST_DIMS(%(z)s)[2] + (needs_extra_z_col ? 1 : 0));
            dim3 block(std::min(CudaNdarray_HOST_DIMS(%(x)s)[3], 512));

            kDownsampleMaxGradGrad_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[2],
                CudaNdarray_HOST_DIMS(%(z)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_DEV_DATA(%(gz)s),
                CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                CudaNdarray_DEV_DATA(%(gx)s),
                CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                CudaNdarray_HOST_STRIDES(%(gx)s)[3]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
    "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kDownsampleMaxGradGrad_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """
        // ds0 is the downsampling factor in rows, ds1 in columns
        template<int ds0, int ds1>
        __global__ void kDownsampleMaxGradGrad_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           const float * z, int zS0, int zS1, int zS2, int zS3,
           float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
           const float *gx, int gxS0, int gxS1, int gxS2, int gxS3)
        {
            //  D0: number of image rows
            //  D1: number of image cols
            //  D2: number of z rows
            //  D3: number of z cols
            // xD2: number of x rows
            // xD3: number of x cols
            // various .S. variables are strides

            float cur_max, cur_x, my_z, my_gx;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            int bdimx = blockDim.x;

            for(int i0 = blockIdx.x;
                i0 < D0;
                i0 += gridDim.x){

                int i1 = 0;                // image col
                // row wrt z and/or gz, ranges from 0 to D2 - 1 OR D2
                // (as needed to cover all x rows)
                int i2 = blockIdx.y;
                int x_col = tx;            // col wrt x, ranges from 0 to xD3 - 1
                int z_col = x_col/ds1;     // z_col corresponding to this x_col


                //TODO: raise occupancy.  Use threadIdx.y to run several
                //      iterations of this i1 loop in parallel

                for (i1 = 0; i1 < D1; ++i1) // loop over images (same for z and x)
                {
                    for(int col_iter = 0;
                        (tx + col_iter * bdimx < xD3) ; col_iter++){

                        //The if inside is to don't do the division if we
                        // need only 1 col_iter

                        if(tx + bdimx < xD3)
                        {
                            x_col = tx + col_iter * bdimx;
                            z_col = x_col/ds1;
                        }

                        my_z = z[i0 *  zS0 + i1 *  zS1 + i2 *  zS2 + z_col* zS3];

                        for (int x_row = i2*ds0;
                              (x_row < i2*ds0+ds0) && (x_row < xD2); ++x_row)
                        {
                            // my_gx = gx[image_row][image_col][x_row][x_col]
                            my_gx = gx[i0*gxS0 + i1*gxS1 + x_row*gxS2 + x_col*gxS3];
                            
                            if (my_z == x[i0*xS0 + i1*xS1 + x_row*xS2 + x_col*xS3]) {
                                gz[i0 *  gzS0 + i1 *  gzS1 + i2 *  gzS2 + z_col* gzS3] = my_gx;
                            }
                        }
                        

                    }
                }
            }
        }
        """ % locals()
