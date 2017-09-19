from __future__ import absolute_import, print_function, division
from six import integer_types

import theano
from theano import Apply, Op

from theano.compile import optdb
from theano.gof import LocalOptGroup, ParamsType
from theano.scalar import bool as bool_t
from theano.tensor.basic import as_tensor_variable
from theano.tensor.opt import in2out

from .basic_ops import (GpuArrayType, CGpuKernelBase,
                        as_gpuarray_variable, gpu_contiguous, infer_context_name, gpuarray_helper_inc_dir)
from .opt_util import inplace_allocempty

try:
    import pygpu
    from pygpu import blas
except ImportError as e:
    # To make sure theano is importable
    pass


class BlasOp(Op):
    def c_headers(self):
        return ['<blas_api.h>', '<numpy_compat.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), gpuarray_helper_inc_dir()]

    def c_init_code(self):
        return ['import_pygpu__blas();']


class GpuGemv(BlasOp):
    """
    Gemv on the GPU.

    """
    params_type = ParamsType(inplace=bool_t)
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, y, alpha, A, x, beta):
        ctx_name = infer_context_name(y, A, x)
        A = as_gpuarray_variable(A, ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        alpha = as_tensor_variable(alpha)
        beta = as_tensor_variable(beta)

        assert alpha.ndim == 0
        assert beta.ndim == 0
        assert A.ndim == 2
        assert x.ndim == 1
        assert y.ndim == 1
        assert A.dtype == x.dtype == y.dtype

        # float16 not supported
        expected = A.dtype
        assert theano.scalar.upcast(alpha.dtype,
                                    beta.dtype, expected) == expected
        alpha = alpha.astype(expected)
        beta = beta.astype(expected)
        return Apply(self, [y, alpha, A, x, beta], [y.type()])

    def perform(self, node, inputs, out_storage, params):
        y, alpha, A, x, beta = inputs
        inplace = params.inplace
        if inplace and y.strides[0] < 0:
            inplace = False
        if A.shape[1] == 0:
            out_storage[0][0] = pygpu.zeros(y.shape, dtype=y.dtype,
                                            context=y.context)
        else:
            out_storage[0][0] = blas.gemv(alpha, A, x, beta, y,
                                          overwrite_y=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], y=inp[0], alpha=inp[1], A=inp[2], x=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name,
                    params=sub['params'])
        code = """
               if (!%(params)s->inplace || %(y)s->ga.strides[0] <= 0) {
                 %(out)s = theano_try_copy(%(out)s, %(y)s);
                 if (%(out)s == NULL) {
                   %(fail)s
                 }
               } else {
                 Py_XDECREF(%(out)s);
                 %(out)s = %(y)s;
                 Py_INCREF(%(out)s);
               }
               """ % vars
        # in case of possible speed up using blas dot,
        # temporary hack A to 1D for vector-vector dot
        code += """
        if (PyGpuArray_DIM(%(A)s, 1) == 0) {
          int code;
          code = GpuArray_memset(&%(out)s->ga, 0);
          if (code != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Memset failed");
            %(fail)s
          }
        } else if ( PyGpuArray_DIM(%(A)s, 0) == 1
         &&((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0] == (dtype_%(alpha)s)1.
         &&((dtype_%(beta)s*)PyArray_DATA(%(beta)s))[0] == (dtype_%(beta)s)0.
        ) {
            %(out)s->ga.nd = 0;
            %(A)s->ga.nd = 1;
            %(A)s->ga.dimensions[0] = %(A)s->ga.dimensions[1];
            ssize_t a_stride0 = %(A)s->ga.strides[0];
            %(A)s->ga.strides[0] = %(A)s->ga.strides[1];
            if (pygpu_blas_rdot(%(x)s, %(A)s, %(out)s, 0) == -1) {
                %(fail)s
            }
            %(A)s->ga.strides[0] = a_stride0;
            %(out)s->ga.nd = 1;
            %(A)s->ga.nd = 2;
            %(A)s->ga.dimensions[0] = 1;
        } else if (
            pygpu_blas_rgemv(cb_no_trans,
            ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
            %(A)s, %(x)s,
            ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
            %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars

        return code

    def c_code_cache_version(self):
        return (10,)

gpugemv_no_inplace = GpuGemv(inplace=False)
gpugemv_inplace = GpuGemv(inplace=True)


class GpuGemm(BlasOp):
    """
    Gemm on the GPU.

    """
    params_type = ParamsType(inplace=bool_t)
    __props__ = ('inplace',)
    _f16_ok = True

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, C, alpha, A, B, beta):
        ctx_name = infer_context_name(C, A, B)
        A = as_gpuarray_variable(A, ctx_name)
        B = as_gpuarray_variable(B, ctx_name)
        C = as_gpuarray_variable(C, ctx_name)
        alpha = as_tensor_variable(alpha)
        beta = as_tensor_variable(beta)

        if not (A.dtype == B.dtype == C.dtype):
            raise TypeError(theano.tensor.blas.Gemm.E_mixed,
                            (A.dtype, B.dtype, C.dtype,
                             alpha.dtype, beta.dtype))
        if not A.dtype.startswith('float'):
            raise TypeError(theano.tensor.blas.Gemm.E_float, (A.dtype))

        if A.dtype == 'float16':
            expected = 'float32'
        else:
            expected = A.dtype
        assert theano.scalar.upcast(alpha.dtype,
                                    beta.dtype, expected) == expected
        alpha = alpha.astype(expected)
        beta = beta.astype(expected)

        assert alpha.ndim == 0
        assert beta.ndim == 0
        assert A.ndim == 2
        assert B.ndim == 2
        assert C.ndim == 2
        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs, params):
        C, alpha, A, B, beta = inputs
        inplace = params.inplace
        if inplace and not C.flags.forc:
            inplace = False
        outputs[0][0] = blas.gemm(alpha, A, B, beta, C,
                                  overwrite_c=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], C=inp[0], alpha=inp[1], A=inp[2], B=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name,
                    params=sub['params'])
        code = """
               if (!%(params)s->inplace || !GpuArray_ISONESEGMENT(&%(C)s->ga)) {
                 %(out)s = theano_try_copy(%(out)s, %(C)s);
                 if (%(out)s == NULL) {
                   %(fail)s
                 }
               } else {
                 Py_XDECREF(%(out)s);
                 %(out)s = %(C)s;
                 Py_INCREF(%(out)s);
               }
               if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                                    ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                                    %(A)s, %(B)s,
                                    ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                                    %(out)s, 0) == -1) {
                 %(fail)s
               }
        """ % vars

        return code

    def c_code_cache_version(self):
        return (7,)

gpugemm_no_inplace = GpuGemm(inplace=False)
gpugemm_inplace = GpuGemm(inplace=True)


class GpuGer(BlasOp):
    """
    Ger on the GPU.

    """
    params_type = ParamsType(inplace=bool_t)
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, A, alpha, x, y):
        ctx_name = infer_context_name(A, x, y)
        A = as_gpuarray_variable(A, ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        alpha = as_tensor_variable(alpha)
        if not(A.dtype == x.dtype == y.dtype):
            raise TypeError('ger requires matching dtypes',
                            (A.dtype, alpha.dtype, x.dtype, y.dtype))

        assert theano.scalar.upcast(alpha.dtype, A.dtype) == A.dtype
        alpha = alpha.astype(A.dtype)
        assert alpha.ndim == 0
        assert A.ndim == 2
        assert x.ndim == 1
        assert y.ndim == 1
        return Apply(self, [A, alpha, x, y], [A.type()])

    def perform(self, node, inp, out, params):
        A, alpha, x, y = inp
        inplace = params.inplace
        if inplace and not A.flags.forc:
            inplace = False
        out[0][0] = blas.ger(alpha, x, y, A,
                             overwrite_a=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], A=inp[0], alpha=inp[1], x=inp[2], y=inp[3],
                    fail=sub['fail'], name=name, params=sub['params'])
        code = """
               if (!%(params)s->inplace || !GpuArray_ISONESEGMENT(&%(A)s->ga)) {
                 %(out)s = theano_try_copy(%(out)s, %(A)s);
                 if (%(out)s == NULL) {
                   %(fail)s
                 }
               } else {
                 Py_XDECREF(%(out)s);
                 %(out)s = %(A)s;
                 Py_INCREF(%(out)s);
               }
               if (pygpu_blas_rger(((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                                %(x)s, %(y)s, %(out)s, 0) == -1) {
                 %(fail)s
               }
               """ % vars

        return code

    def c_code_cache_version(self):
        return (5,)


gpuger_no_inplace = GpuGer(inplace=False)
gpuger_inplace = GpuGer(inplace=True)


class GpuDot22(BlasOp):
    """
    Dot22 on the GPU.

    """
    _f16_ok = True
    __props__ = ()

    def make_node(self, x, y):
        ctx_name = infer_context_name(x, y)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.dtype == y.dtype
        otype = x.type.clone(
            broadcastable=(x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y], [otype()])

    def perform(self, node, inputs, outputs):
        x, y = inputs

        out = pygpu.empty((x.shape[0], y.shape[1]), dtype=x.dtype,
                          context=x.context)
        outputs[0][0] = blas.gemm(1., x, y, 0., out,
                                  overwrite_c=True)

    def c_code(self, node, name, inputs, outputs, sub):
        dtype = node.inputs[0].dtype
        typecode = pygpu.gpuarray.dtype_to_typecode(dtype)
        vars = dict(A=inputs[0], B=inputs[1], dtype=dtype, out=outputs[0],
                    typecode=typecode,
                    fail=sub['fail'], name=name)
        code = """
        double one = 1.;
        double zero = 0.;

        size_t dims[] = {0, 0};
        dims[0] = PyGpuArray_DIMS(%(A)s)[0];
        dims[1] = PyGpuArray_DIMS(%(B)s)[1];

        if (theano_prep_output(&%(out)s, 2, dims, %(typecode)s, GA_C_ORDER,
                               %(A)s->context)) {
            %(fail)s
        }

        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             one,
                             %(A)s, %(B)s,
                             zero,
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars

        return code

    def c_code_cache_version(self):
        return (5,)

gpu_dot22 = GpuDot22()


class GpuGemmBatch(BlasOp):
    params_type = ParamsType(inplace=bool_t)
    __props__ = ('inplace',)
    _f16_ok = True

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, C, alpha, A, B, beta):
        ctx_name = infer_context_name(C, A, B)
        A = as_gpuarray_variable(A, ctx_name)
        B = as_gpuarray_variable(B, ctx_name)
        C = as_gpuarray_variable(C, ctx_name)
        alpha = as_tensor_variable(alpha)
        if alpha.dtype == 'float16':
            alpha = alpha.astype('float32')
        beta = as_tensor_variable(beta)
        if beta.dtype == 'float16':
            beta = beta.astype('float32')
        assert alpha.ndim == 0
        assert beta.ndim == 0
        assert A.ndim == 3
        assert B.ndim == 3
        assert C.ndim == 3
        assert A.dtype == B.dtype == C.dtype
        if A.dtype in ('float32', 'float64'):
            assert A.dtype == alpha.dtype == beta.dtype
        else:
            assert 'float32' == alpha.dtype == beta.dtype
        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def c_headers(self):
        return super(GpuGemmBatch, self).c_headers() + ['<gpuarray/blas.h>']

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], C=inp[0], alpha=inp[1], A=inp[2], B=inp[3],
                    beta=inp[4], params=sub['params'],
                    fail=sub['fail'], name=name)
        code = """
        int err;
        if (%(params)s->inplace){
                   if (!GpuArray_ISONESEGMENT(&%(C)s->ga)) {
                     %(out)s = theano_try_copy(%(out)s, %(C)s);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     Py_XDECREF(%(out)s);
                     %(out)s = %(C)s;
                     Py_INCREF(%(out)s);
                   }
        } else {
                   %(out)s = theano_try_copy(%(out)s, %(C)s);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
        }
        err = GpuArray_rgemmBatch_3d(
            cb_no_trans, cb_no_trans,
            ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
            &%(A)s->ga, &%(B)s->ga,
            ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
            &%(out)s->ga, 0);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "%%s", GpuArray_error(&%(A)s->ga, err));
            %(fail)s;
        }
        """ % vars

        return code

    def c_code_cache_version(self):
        return (4,)

gpugemmbatch_no_inplace = GpuGemmBatch(inplace=False)
gpugemmbatch_inplace = GpuGemmBatch(inplace=True)


class BaseGpuCorrMM(CGpuKernelBase):
    """
    Base class for `GpuCorrMM`, `GpuCorrMM_gradWeights` and
    `GpuCorrMM_gradInputs`. Cannot be used directly.

    Parameters
    ----------
    border_mode : {'valid', 'full', 'half'}
        Additionally, the padding size could be directly specified by an integer,
        a pair of integers, or two pairs of integers.
    subsample
        Perform subsampling of the output (default: (1, 1)).
    filter_dilation
        Perform subsampling of the input, also known as dilation (default: (1, 1)).
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately (default : 1).
    unshared
        Perform unshared correlation (default: False)
    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_dilation', 'num_groups', 'unshared')
    _f16_ok = True

    def __init__(self, border_mode="valid", subsample=(1, 1),
                 filter_dilation=(1, 1), num_groups=1, unshared=False):
        if isinstance(border_mode, integer_types):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = ((border_mode, border_mode),) * 2
        elif isinstance(border_mode, tuple):
            if len(border_mode) != 2:
                raise ValueError(
                    'invalid border_mode {} which must be a '
                    'tuple of length 2'.format(border_mode))
            border = ()
            for mode in border_mode:
                if isinstance(mode, tuple) and len(mode) == 2 and \
                        min(mode) >= 0:
                    border += ((int(mode[0]), int(mode[1])),)
                elif mode >= 0:
                    border += ((int(mode), int(mode)),)
                else:
                    raise ValueError(
                        'invalid border mode {}. The tuple can only contain '
                        'integers or tuples of length 2'.format(border_mode))
            border_mode = border
        elif border_mode not in ('valid', 'full', 'half'):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a tuple '
                'of length 2'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        if len(filter_dilation) != 2:
            raise ValueError("filter_dilation must have two elements")
        self.subsample = tuple(subsample)
        self.filter_dilation = tuple(filter_dilation)
        if num_groups < 1:
            raise ValueError("Number of groups should be greater than 0")
        self.num_groups = num_groups
        CGpuKernelBase.__init__(self, ['c_code/corr_gemm.c'])
        self.unshared = unshared

    @property
    def pad(self):
        if self.border_mode != 'valid':
            return self.border_mode
        return ((0, 0),) * 2

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.filter_dilation),
            str(self.num_groups),
            str(self.unshared))

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'num_groups'):
            self.num_groups = 1

    def flops(self, inp, outp):
        """
        Useful with the hack in profilemode to print the MFlops.

        """
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

    def c_headers(self):
        return ["<gpuarray/array.h>", "<gpuarray/blas.h>", "gpuarray_helper.h"]

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code_cache_version(self):
        # Raise this whenever modifying the C code (including the file).
        return (12,)

    def c_code_helper(self, bottom, weights, top, direction, sub, height=None, width=None):
        """
        This generates the C code for GpuCorrMM (direction="forward"),
        GpuCorrMM_gradWeights (direction="backprop weights"), and
        GpuCorrMM_gradInputs (direction="backprop inputs").
        Depending on the direction, one of bottom, weights, top will
        receive the output, while the other two serve as inputs.

        Parameters
        ----------
        bottom
            Variable name of the input images in the forward pass,
            or the gradient of the input images in backprop wrt. inputs
        weights
            Variable name of the filters in the forward pass,
            or the gradient of the filters in backprop wrt. weights
        top
            Variable name of the output images / feature maps in the
            forward pass, or the gradient of the outputs in the backprop passes
        direction : {'forward', 'backprop weights', 'backprop inputs'}
            "forward" to correlate bottom with weights and store results in top,
            "backprop weights" to do a valid convolution of bottom with top
            (swapping the first two dimensions) and store results in weights,
            and "backprop inputs" to do a full convolution of top with weights
            (swapping the first two dimensions) and store results in bottom.
        sub
            Dictionary of substitutions useable to help generating the C code.
        height
            Required if self.subsample[0] != 1, a variable giving the height of
            the filters for direction="backprop weights" or the height of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the height
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.
        width
            Required if self.subsample[1] != 1, a variable giving the width of
            the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the width
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.

        """
        dH, dW = self.subsample
        dilH, dilW = self.filter_dilation
        numgroups = self.num_groups
        unshared = int(self.unshared)
        if self.border_mode == "half":
            padH_l = padH_r = padW_l = padW_r = -1
        elif self.border_mode == "full":
            padH_l = padH_r = padW_l = padW_r = -2
        elif isinstance(self.border_mode, tuple):
            (padH_l, padH_r), (padW_l, padW_r) = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH_l = padH_r = padW_l = padW_r = 0
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
        if height:
            height = '(*(npy_int*)(PyArray_DATA(%s)))' % height
        else:
            if ((direction != 0) and (dH != 1)) or ((direction == 1) and (padH_l == -1 or padH_r == -1)):
                raise ValueError("height must be given for backprop with vertical sampling or pad='half'")
            height = '-1'
        if width:
            width = '(*(npy_int*)(PyArray_DATA(%s)))' % width
        else:
            if ((direction != 0) and (dW != 1)) or ((direction == 1) and (padW_l == -1 or padW_r == -1)):
                raise ValueError("width must be given for backprop with horizontal sampling or pad='half'")
            width = '-1'

        sub = sub.copy()
        sub.update(locals())

        return """
    // Mandatory args
    int direction = %(direction)s;  // forward, bprop weights, bprop inputs

    // Optional args
    size_t dH = %(dH)s;
    size_t dW = %(dW)s;
    size_t dilH = %(dilH)s;
    size_t dilW = %(dilW)s;
    int padH_l = %(padH_l)s;
    int padH_r = %(padH_r)s;
    int padW_l = %(padW_l)s;
    int padW_r = %(padW_r)s;
    int numgroups = %(numgroups)s;
    int unshared = %(unshared)s;

    PyGpuArrayObject * bottom = %(bottom)s;
    PyGpuArrayObject * weights = %(weights)s;
    PyGpuArrayObject * top = %(top)s;
    PyGpuArrayObject * out2 = NULL;

    int wdim, odim;
    wdim = unshared ? 6 : 4;
    odim = 4; //Can be set to 6 later for unshared backprop wrt weights

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    size_t kH, kW, dil_kH, dil_kW;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = PyGpuArray_DIMS(weights)[wdim-2];
        kW = PyGpuArray_DIMS(weights)[wdim-1];
    }
    else {
        if (%(height)s != -1) {
            // kernel height is specified (perhaps vertical subsampling or half padding)
            kH = %(height)s;
        }
        else if (padH_l == -2 || padH_r == -2) {
            // vertical full padding, we can infer the kernel height
            kH = (2 - PyGpuArray_DIMS(bottom)[2] + (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = (PyGpuArray_DIMS(bottom)[2] + padH_l + padH_r - (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1 ;
        }
        if (%(width)s != -1) {
            kW = %(width)s;
        }
        else if (padW_l == -2 || padW_r == -2) {
            kW = (2 - PyGpuArray_DIMS(bottom)[3] + (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        else {
            kW = (PyGpuArray_DIMS(bottom)[3] + padW_l + padW_r - (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
    }

    // Implicit dilated kernel size
    dil_kH = (kH - 1) * dilH + 1;
    dil_kW = (kW - 1) * dilW + 1;

    // Auto-padding if requested
    if (padH_l == -1 || padH_r == -1) {  // vertical half padding
        padH_l = padH_r = dil_kH / 2;
    }
    else if (padH_l == -2 || padH_r == -2) {  // vertical full padding
        padH_l = padH_r = dil_kH - 1;
    }
    else if (padH_l < 0 || padH_r < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padH must be >= -2");
        %(fail)s
    }
    if (padW_l == -1 || padW_r == -1) {  // horizontal half padding
        padW_l = padW_r = dil_kW / 2;
    }
    else if (padW_l == -2 || padW_r == -2) {  // horizontal full padding
        padW_l = padW_r = dil_kW - 1;
    }
    else if (padW_l < 0 || padW_r < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padW must be >= -2");
        %(fail)s
    }

    // Infer output shape and type
    // The inferred shape can be negative.
    long long out_dim[6];
    size_t out_dim_size[6];
    out_dim[4] = out_dim[5] = 0; //Only used for unshared backprop wrt weights
    out_dim_size[4] = out_dim_size[5] = 0; //Same
    int out_typecode;
    PyGpuContextObject *out_context;
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + pad_l + pad_r - ((weight-1)*dil + 1)) / sample + 1
        out_dim[0] = PyGpuArray_DIMS(bottom)[0];
        out_dim[1] = PyGpuArray_DIMS(weights)[0];
        out_dim[2] = (PyGpuArray_DIMS(bottom)[2] + padH_l + padH_r - ((PyGpuArray_DIMS(weights)[wdim-2]-1)*dilH + 1)) / dH + 1;
        out_dim[3] = (PyGpuArray_DIMS(bottom)[3] + padW_l + padW_r - ((PyGpuArray_DIMS(weights)[wdim-1]-1)*dilW + 1)) / dW + 1;
        out_typecode = bottom->ga.typecode;
        out_context = bottom->context;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
        {
            if (unshared) {
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                             PyGpuArray_DIMS(weights)[4], PyGpuArray_DIMS(weights)[5],
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
                %(fail)s
            }
            else {
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
                %(fail)s
            }
        }
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width) or
        // (num_filters, top_height, top_width, num_channels, height, width) -> for unshared
        // height and width: weights = (bottom + 2*pad - (top - 1) * sample - 1) / dil + 1
        out_dim[0] = PyGpuArray_DIMS(top)[1];
        if (unshared){
            odim = 6;
            out_dim[1] = PyGpuArray_DIMS(top)[2];
            out_dim[2] = PyGpuArray_DIMS(top)[3];
        }
        out_dim[wdim-3] = PyGpuArray_DIMS(bottom)[1] / numgroups;
        out_dim[wdim-2] = kH;  // already inferred further above
        out_dim[wdim-1] = kW;  // how convenient
        out_typecode = top->ga.typecode;
        out_context = top->context;
        if (unshared) {
            if (out_dim[0] < 0 || out_dim[1] <= 0 || out_dim[2] <= 0 || out_dim[3] < 0
                    || out_dim[4] <= 0 || out_dim[5] <= 0){
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM backprop wrt. weights: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                             out_dim[4], out_dim[5],
                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        else {
             if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM backprop wrt. weights: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + (weights-1)*dil + 1 - 2*pad
        out_dim[0] = PyGpuArray_DIMS(top)[0];
        out_dim[1] = PyGpuArray_DIMS(weights)[wdim-3] * numgroups;
        out_dim[2] = (%(height)s != -1) ? %(height)s : (PyGpuArray_DIMS(top)[2] - 1) * dH + (PyGpuArray_DIMS(weights)[wdim-2]-1)*dilH + 1 - padH_l - padH_r;
        out_dim[3] = (%(width)s != -1) ? %(width)s : (PyGpuArray_DIMS(top)[3] - 1) * dW + (PyGpuArray_DIMS(weights)[wdim-1]-1)*dilW + 1 - padW_l - padW_r;
        out_typecode = top->ga.typecode;
        out_context = top->context;
        if (unshared) {
            if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM backprop wrt. inputs: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weight shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                             PyGpuArray_DIMS(weights)[4], PyGpuArray_DIMS(weights)[5],
                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        else {
            if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuCorrMM backprop wrt. inputs: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weight shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             out_dim[0], out_dim[1], out_dim[2], out_dim[3],
                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    out_dim_size[0] = (size_t)out_dim[0];
    out_dim_size[1] = (size_t)out_dim[1];
    out_dim_size[2] = (size_t)out_dim[2];
    out_dim_size[3] = (size_t)out_dim[3];

    if (odim == 6) {
        out_dim_size[4] = (size_t)out_dim[4];
        out_dim_size[5] = (size_t)out_dim[5];
    }

    // Prepare output array
    if (theano_prep_output(&%(out)s, odim, out_dim_size, out_typecode, GA_C_ORDER, out_context) != 0)
    {
        if (odim == 4) {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                    out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
        }
        if (odim == 6) {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld %%lld %%lld",
                    out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4], out_dim[5]);
        }
        %(fail)s
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga)) {
        PyErr_SetString(PyExc_ValueError, "Only contiguous outputs are supported.");
        %(fail)s
    }

    // Call GPU code
    out2 = corrMM(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, dilH, dilW,
                padH_l, padH_r, padW_l, padW_r, numgroups, unshared);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class GpuCorrMM(BaseGpuCorrMM):
    """
    GPU correlation implementation using Matrix Multiplication.

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
        convolution for odd-sized kernels).
        If it is a tuple containing 2 pairs of integers, then these specify
        the padding to be applied on each side ((left, right), (top, bottom)).
        Otherwise, each width is applied twice, once per side (left and right,
        top and bottom).
    subsample
        The subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `GpuCorrMM(...)(...)[:,:,::sv, ::sh]`,
        but faster.
        Set to `(1, 1)` to disable subsampling.
    filter_dilation
        The filter dilation operation applied to each input image.
        Should be a tuple with 2 elements.
        Set to `(1, 1)` to disable filter dilation.
    num_groups
        The number of distinct groups the image and kernel must be
        divided into.
        should be an int
        set to 1 to disable grouped convolution
    unshared
        Perform unshared correlation (default: False)

    Notes
    -----
    Currently, the Op requires the inputs, filters and outputs to be
    C-contiguous. Use :func:`gpu_contiguous
    <theano.gpuarray.basic_ops.gpu_contiguous>` on these arguments
    if needed.

    You can either enable the Theano flag `optimizer_including=conv_gemm`
    to automatically replace all convolution operations with `GpuCorrMM`
    or one of its gradients, or you can use it as a replacement for
    :func:`conv2d <theano.tensor.nnet.conv.conv2d>`, called as
    `GpuCorrMM(subsample=...)(image, filters)`. The latter is currently
    faster, but note that it computes a correlation -- if you need to
    compute a convolution, flip the filters as `filters[:,:,::-1,::-1]`.

    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1), num_groups=1, unshared=False):
        super(GpuCorrMM, self).__init__(border_mode, subsample,
                                        filter_dilation, num_groups, unshared)

    def make_node(self, img, kern):
        ctx_name = infer_context_name(img, kern)
        img = as_gpuarray_variable(img, ctx_name)
        kern = as_gpuarray_variable(kern, ctx_name)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if self.unshared:
            if kern.type.ndim != 6:
                raise TypeError('kern must be 6D tensor')
        else:
            if kern.type.ndim != 4:
                raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [GpuArrayType(dtype=img.dtype,
                                                      context_name=ctx_name,
                                                      broadcastable=broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(GpuCorrMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
                                        self.subsample,
                                        self.filter_dilation,
                                        self.num_groups,
                                        self.unshared)(
            weights, top, bottom.shape[-2:])
        d_weights = GpuCorrMM_gradWeights(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation,
                                          self.num_groups,
                                          self.unshared)(
            bottom, top, weights.shape[-2:])
        return d_bottom, d_weights


class GpuCorrMM_gradWeights(BaseGpuCorrMM):
    """
    Gradient wrt. filters for `GpuCorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1),
                 num_groups=1,
                 unshared=False):
        super(GpuCorrMM_gradWeights, self).__init__(border_mode,
                                                    subsample,
                                                    filter_dilation, num_groups,
                                                    unshared)

    def make_node(self, img, topgrad, shape=None):
        ctx_name = infer_context_name(img, topgrad)
        img = as_gpuarray_variable(img, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1) or self.border_mode == "half":
                raise ValueError('shape must be given if subsample != (1, 1)'
                                 ' or border_mode == "half"')
            height_width = []
        else:
            height_width = [shape[0], shape[1]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0

        if self.unshared:
            broadcastable = [topgrad.type.broadcastable[1], False, False,
                             img.type.broadcastable[1], False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                             False, False]
        return Apply(self, [img, topgrad] + height_width, [GpuArrayType(dtype=img.dtype,
                                                                        context_name=ctx_name,
                                                                        broadcastable=broadcastable)()])

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
        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
                                        self.subsample,
                                        self.filter_dilation,
                                        self.num_groups,
                                        self.unshared)(weights,
                                                       top,
                                                       bottom.shape[-2:])
        d_top = GpuCorrMM(
            self.border_mode, self.subsample, self.filter_dilation, self.num_groups, self.unshared)(bottom, weights)
        d_height_width = (
            theano.gradient.DisconnectedType()(),
            ) * 2 if len(inp) == 4 else ()
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class GpuCorrMM_gradInputs(BaseGpuCorrMM):
    """
    Gradient wrt. inputs for `GpuCorrMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1),
                 filter_dilation=(1, 1),
                 num_groups=1,
                 unshared=False):
        super(GpuCorrMM_gradInputs, self).__init__(border_mode, subsample,
                                                   filter_dilation, num_groups,
                                                   unshared)

    def make_node(self, kern, topgrad, shape=None):
        ctx_name = infer_context_name(kern, topgrad)
        kern = as_gpuarray_variable(kern, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        if self.unshared:
            if kern.type.ndim != 6:
                raise TypeError('kern must be 6D tensor')
        else:
            if kern.type.ndim != 4:
                raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1):
                raise ValueError('shape must be given if subsample != (1, 1)')
            height_width = []
        else:
            height_width = [shape[0], shape[1]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0

        if self.num_groups > 1:
            broadcastable = [topgrad.type.broadcastable[0], False,
                             False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[-3],
                             False, False]
        return Apply(self, [kern, topgrad] + height_width, [GpuArrayType(dtype=topgrad.dtype,
                                                                         context_name=ctx_name,
                                                                         broadcastable=broadcastable)()])

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
        d_weights = GpuCorrMM_gradWeights(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation,
                                          self.num_groups,
                                          self.unshared)(bottom,
                                                         top,
                                                         weights.shape[-2:])
        d_top = GpuCorrMM(self.border_mode,
                          self.subsample,
                          self.filter_dilation,
                          self.num_groups,
                          self.unshared)(bottom, weights)
        d_height_width = (
            theano.gradient.DisconnectedType()(),
            ) * 2 if len(inp) == 4 else ()
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class BaseGpuCorr3dMM(CGpuKernelBase):
    """
    Base class for `GpuCorr3dMM`, `GpuCorr3dMM_gradWeights` and
    `GpuCorr3dMM_gradInputs`. Cannot be used directly.

    Parameters
    ----------
    border_mode : {'valid', 'full', 'half'}
        Additionally, the padding size could be directly specified by an integer
        or a pair of integers
    subsample
        Perform subsampling of the output (default: (1, 1, 1)).
    filter_dilation
        Perform subsampling of the input, also known as dilation (default: (1, 1, 1)).
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately (default : 1).

    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_dilation', 'num_groups')
    _f16_ok = True

    def __init__(self, border_mode="valid", subsample=(1, 1, 1),
                 filter_dilation=(1, 1, 1), num_groups=1):
        if isinstance(border_mode, integer_types):
            border_mode = (border_mode, border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w, pad_d = map(int, border_mode)
            border_mode = (pad_h, pad_w, pad_d)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a tuple of'
                ' three integers'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 3:
            raise ValueError("subsample must have three elements")
        if len(filter_dilation) != 3:
            raise ValueError("filter_dilation must have three elements")
        self.subsample = tuple(subsample)
        self.filter_dilation = tuple(filter_dilation)
        if num_groups < 1:
            raise ValueError("Number of groups should be greater than 0")
        self.num_groups = num_groups
        CGpuKernelBase.__init__(self, ['c_code/corr3d_gemm.c'])

    @property
    def pad(self):
        if self.border_mode != 'valid':
            return self.border_mode
        return (0, 0, 0)

    def __str__(self):
        return '%s{%s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.filter_dilation),
            str(self.num_groups))

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'num_groups'):
            self.num_groups = 1

    def flops(self, inp, outp):
        """
        Useful with the hack in profilemode to print the MFlops.

        """
        # if the output shape is correct, then this gives the correct
        # flops for any direction, sampling, padding, and border mode
        inputs, filters = inp
        outputs, = outp
        assert inputs[1] == (filters[1] * self.num_groups)
        # nb mul and add by output pixel
        flops = filters[2] * filters[3] * filters[4] * 2
        # nb flops by output image
        flops *= outputs[2] * outputs[3] * outputs[4]
        # nb patch multiplied
        flops *= inputs[1] * filters[0] * inputs[0] / self.num_groups
        return flops

    def c_headers(self):
        return ["<gpuarray/array.h>", "<gpuarray/blas.h>", "gpuarray_helper.h"]

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code_cache_version(self):
        # raise this whenever modifying the code below.
        return (8,)

    def c_code_helper(self, bottom, weights, top, direction, sub,
                      height=None, width=None, depth=None):
        """
        This generates the C code for GpuCorr3dMM (direction="forward"),
        GpuCorr3dMM_gradWeights (direction="backprop weights"), and
        GpuCorr3dMM_gradInputs (direction="backprop inputs").
        Depending on the direction, one of bottom, weights, top will
        receive the output, while the other two serve as inputs.

        Parameters
        ----------
        bottom
            Variable name of the input images in the forward pass,
            or the gradient of the input images in backprop wrt. inputs
        weights
            Variable name of the filters in the forward pass,
            or the gradient of the filters in backprop wrt. weights
        top
            Variable name of the output images / feature maps in the
            forward pass, or the gradient of the outputs in the backprop passes
        direction : {'forward', 'backprop weights', 'backprop inputs'}
            "forward" to correlate bottom with weights and store results in top,
            "backprop weights" to do a valid convolution of bottom with top
            (swapping the first two dimensions) and store results in weights,
            and "backprop inputs" to do a full convolution of top with weights
            (swapping the first two dimensions) and store results in bottom.
        sub
            Dictionary of substitutions useable to help generating the C code.
        height
            Required if self.subsample[0] != 1, a variable giving the height of
            the filters for direction="backprop weights" or the height of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the height
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.
        width
            Required if self.subsample[1] != 1, a variable giving the width of
            the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the width
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.
        depth
            Required if self.subsample[2] != 1, a variable giving the depth of
            the filters for direction="backprop weights" or the depth of the
            input images for direction="backprop inputs".
            Required if self.border_mode == 'half', a variable giving the depth
            of the filters for direction="backprop weights".
            Not required otherwise, but if a value is given this will be checked.

        """
        dH, dW, dD = self.subsample
        dilH, dilW, dilD = self.filter_dilation
        numgroups = self.num_groups
        if self.border_mode == "half":
            padH = padW = padD = -1
        elif self.border_mode == "full":
            padH = padW = padD = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW, padD = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = padD = 0
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
        if height:
            height = '(*(npy_int*)(PyArray_DATA(%s)))' % height
        else:
            if ((direction != 0) and (dH != 1)) or ((direction == 1) and (padH == -1)):
                raise ValueError("height must be given for backprop with vertical sampling or pad='half'")
            height = '-1'
        if width:
            width = '(*(npy_int*)(PyArray_DATA(%s)))' % width
        else:
            if ((direction != 0) and (dW != 1)) or ((direction == 1) and (padW == -1)):
                raise ValueError("width must be given for backprop with horizontal sampling or pad='half'")
            width = '-1'
        if depth:
            depth = '(*(npy_int*)(PyArray_DATA(%s)))' % depth
        else:
            if ((direction != 0) and (dD != 1)) or ((direction == 1) and (padD == -1)):
                raise ValueError("depth must be given for backprop with horizontal sampling or pad='half'")
            depth = '-1'

        sub = sub.copy()
        sub.update(locals())

        return """
    // Mandatory args
    int direction = %(direction)s;  // forward, bprop weights, bprop inputs

    // Optional args
    size_t dH = %(dH)s;
    size_t dW = %(dW)s;
    size_t dD = %(dD)s;
    size_t dilH = %(dilH)s;
    size_t dilW = %(dilW)s;
    size_t dilD = %(dilD)s;
    int padH = %(padH)s;
    int padW = %(padW)s;
    int padD = %(padD)s;
    int numgroups = %(numgroups)s;

    PyGpuArrayObject * bottom = %(bottom)s;
    PyGpuArrayObject * weights = %(weights)s;
    PyGpuArrayObject * top = %(top)s;
    PyGpuArrayObject * out2 = NULL;

    // Obtain or infer kernel height, width and depth
    // (we need to know it early to be able to handle auto-padding)
    size_t kH, kW, kD, dil_kH, dil_kW, dil_kD;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = PyGpuArray_DIMS(weights)[2];
        kW = PyGpuArray_DIMS(weights)[3];
        kD = PyGpuArray_DIMS(weights)[4];
    }
    else {
        if (%(height)s != -1) {
            // kernel height is specified (perhaps vertical subsampling or half padding)
            kH = %(height)s;
        }
        else if (padH == -2) {
            // vertical full padding, we can infer the kernel height
            kH = (2 - PyGpuArray_DIMS(bottom)[2] + (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = (PyGpuArray_DIMS(bottom)[2] + 2*padH - (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1 ;
        }
        if (%(width)s != -1) {
            kW = %(width)s;
        }
        else if (padW == -2) {
            kW = (2 - PyGpuArray_DIMS(bottom)[3] + (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        else {
            kW = (PyGpuArray_DIMS(bottom)[3] + 2*padW - (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        if (%(depth)s != -1) {
            kD = %(depth)s;
        }
        else if (padD == -2) {
            kD = (2 - PyGpuArray_DIMS(bottom)[4] + (PyGpuArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
        }
        else {
            kD = (PyGpuArray_DIMS(bottom)[4] + 2*padD - (PyGpuArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
        }
    }

    // Implicit dilated kernel size
    dil_kH = (kH - 1) * dilH + 1;
    dil_kW = (kW - 1) * dilW + 1;
    dil_kD = (kD - 1) * dilD + 1;

    // Auto-padding if requested
    if (padH == -1) {  // vertical half padding
        padH = dil_kH / 2;
    }
    else if (padH == -2) {  // vertical full padding
        padH = dil_kH - 1;
    }
    else if (padH < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = dil_kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = dil_kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padW must be >= -2");
        %(fail)s
    }
    if (padD == -1) {  // depth half padding
        padD = dil_kD / 2;
    }
    else if (padD == -2) {  // depth full padding
        padD = dil_kD - 1;
    }
    else if (padD < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padD must be >= -2");
        %(fail)s
    }

    // Infer output shape and type
    // The inferred shape can be negative.
    long long out_dim[5];
    size_t out_dim_size[5];
    int out_typecode;
    PyGpuContextObject *out_context;
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width, depth)
        // height, width and depth: top = (bottom + 2*pad - ((weight-1)*dil + 1)) / sample + 1
        out_dim[0] = PyGpuArray_DIMS(bottom)[0];
        out_dim[1] = PyGpuArray_DIMS(weights)[0];
        out_dim[2] = (PyGpuArray_DIMS(bottom)[2] + 2*padH - ((PyGpuArray_DIMS(weights)[2]-1)*dilH + 1)) / dH + 1;
        out_dim[3] = (PyGpuArray_DIMS(bottom)[3] + 2*padW - ((PyGpuArray_DIMS(weights)[3]-1)*dilW + 1)) / dW + 1;
        out_dim[4] = (PyGpuArray_DIMS(bottom)[4] + 2*padD - ((PyGpuArray_DIMS(weights)[4]-1)*dilD + 1)) / dD + 1;
        out_typecode = bottom->ga.typecode;
        out_context = bottom->context;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorr3dMM: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                         PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                         PyGpuArray_DIMS(bottom)[4],
                         PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                         PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                         PyGpuArray_DIMS(weights)[4],
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4]);
            %(fail)s
        }
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width, depth)
        // height, width and depth: weights = (bottom + 2*pad - (top - 1) * sample - 1) / dil + 1
        out_dim[0] = PyGpuArray_DIMS(top)[1];
        out_dim[1] = PyGpuArray_DIMS(bottom)[1] / numgroups;
        out_dim[2] = kH;  // already inferred further above
        out_dim[3] = kW;  // how convenient
        out_dim[4] = kD;
        out_typecode = top->ga.typecode;
        out_context = top->context;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorr3dMM backprop wrt. weights: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
                         PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
                         PyGpuArray_DIMS(bottom)[4],
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4],
                         PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                         PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
                         PyGpuArray_DIMS(top)[4]);
            %(fail)s
        }
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width, depth)
        // height, width and depth: bottom = (top - 1) * sample + (weights-1)*dil + 1 - 2*pad
        out_dim[0] = PyGpuArray_DIMS(top)[0];
        out_dim[1] = PyGpuArray_DIMS(weights)[1] * numgroups;
        out_dim[2] = (%(height)s != -1) ? %(height)s : (PyGpuArray_DIMS(top)[2] - 1) * dH + (PyGpuArray_DIMS(weights)[2]-1)*dilH + 1 - 2*padH;
        out_dim[3] = (%(width)s != -1) ? %(width)s : (PyGpuArray_DIMS(top)[3] - 1) * dW + (PyGpuArray_DIMS(weights)[3]-1)*dilW + 1 - 2*padW;
        out_dim[4] = (%(depth)s != -1) ? %(depth)s : (PyGpuArray_DIMS(top)[4] - 1) * dD + (PyGpuArray_DIMS(weights)[4]-1)*dilD + 1 - 2*padD;
        out_typecode = top->ga.typecode;
        out_context = top->context;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuCorr3dMM backprop wrt. inputs: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4],
                         PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
                         PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
                         PyGpuArray_DIMS(weights)[4],
                         PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                         PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
                         PyGpuArray_DIMS(top)[4]);
            %(fail)s
        }
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    out_dim_size[0] = (size_t)out_dim[0];
    out_dim_size[1] = (size_t)out_dim[1];
    out_dim_size[2] = (size_t)out_dim[2];
    out_dim_size[3] = (size_t)out_dim[3];
    out_dim_size[4] = (size_t)out_dim[4];

    // Prepare output array
    if (theano_prep_output(&%(out)s, 5, out_dim_size, out_typecode, GA_C_ORDER, out_context) != 0)
    {
        PyErr_Format(PyExc_RuntimeError,
                "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld x %%lld",
                out_dim[0], out_dim[1], out_dim[2], out_dim[3], out_dim[4]);
        %(fail)s
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga)) {
        PyErr_SetString(PyExc_ValueError, "Only contiguous outputs are supported.");
        %(fail)s
    }

    // Call GPU code
    out2 = corr3dMM(%(bottom)s, %(weights)s, %(top)s, direction,
                    dH, dW, dD, dilH, dilW, dilD, padH, padW, padD, numgroups);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == %(out)s);

""" % sub


class GpuCorr3dMM(BaseGpuCorr3dMM):
    """
    GPU correlation implementation using Matrix Multiplication.

    Parameters
    ----------
    border_mode
        The width of a border of implicit zeros to pad the
        input with. Must be a tuple with 3 elements giving the width of
        the padding on each side, or a single integer to pad the same
        on all sides, or a string shortcut setting the padding at runtime:
        ``'valid'`` for ``(0, 0, 0)`` (valid convolution, no padding), ``'full'``
        for ``(kernel_rows - 1, kernel_columns - 1, kernel_depth - 1)``
        (full convolution), ``'half'`` for ``(kernel_rows // 2,
        kernel_columns // 2, kernel_depth // 2)`` (same convolution for
        odd-sized kernels). Note that the three widths are each
        applied twice, once per side (left and right, top and bottom, front
        and back).
    subsample
        The subsample operation applied to each output image. Should be a tuple
        with 3 elements. `(sv, sh, sl)` is equivalent to
        `GpuCorrMM(...)(...)[:,:,::sv, ::sh, ::sl]`, but faster.
        Set to `(1, 1, 1)` to disable subsampling.
    filter_dilation
        The filter dilation operation applied to each input image.
        Should be a tuple with 3 elements.
        Set to `(1, 1, 1)` to disable filter dilation.
    num_groups
        The number of distinct groups the image and kernel must be
        divided into.
        should be an int
        set to 1 to disable grouped convolution

    Notes
    -----
    Currently, the Op requires the inputs, filters and outputs to be
    C-contiguous. Use :func:`gpu_contiguous
    <theano.gpuarray.basic_ops.gpu_contiguous>` on these arguments
    if needed.

    You can either enable the Theano flag `optimizer_including=conv_gemm`
    to automatically replace all convolution operations with `GpuCorr3dMM`
    or one of its gradients, or you can use it as a replacement for
    :func:`conv2d <theano.tensor.nnet.conv.conv2d>`, called as
    `GpuCorr3dMM(subsample=...)(image, filters)`. The latter is currently
    faster, but note that it computes a correlation -- if you need to
    compute a convolution, flip the filters as `filters[:,:,::-1,::-1,::-1]`.

    """
    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(GpuCorr3dMM, self).__init__(border_mode, subsample,
                                          filter_dilation, num_groups)

    def make_node(self, img, kern):
        ctx_name = infer_context_name(img, kern)
        img = as_gpuarray_variable(img, ctx_name)
        kern = as_gpuarray_variable(kern, ctx_name)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False, False]
        return Apply(self, [img, kern], [GpuArrayType(dtype=img.dtype,
                                                      context_name=ctx_name,
                                                      broadcastable=broadcastable)()])

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        direction = "forward"
        return super(GpuCorr3dMM, self).c_code_helper(bottom, weights, top, direction, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)
        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation,
                                          self.num_groups)(
            weights, top, bottom.shape[-3:])
        d_weights = GpuCorr3dMM_gradWeights(self.border_mode,
                                            self.subsample,
                                            self.filter_dilation,
                                            self.num_groups)(
            bottom, top, weights.shape[-3:])
        return d_bottom, d_weights


class GpuCorr3dMM_gradWeights(BaseGpuCorr3dMM):
    """
    Gradient wrt. filters for `GpuCorr3dMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(GpuCorr3dMM_gradWeights, self).__init__(border_mode,
                                                      subsample,
                                                      filter_dilation,
                                                      num_groups)

    def make_node(self, img, topgrad, shape=None):
        ctx_name = infer_context_name(img, topgrad)
        img = as_gpuarray_variable(img, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if shape is None:
            if self.subsample != (1, 1, 1) or self.border_mode == "half":
                raise ValueError('shape must be given if subsample != (1, 1, 1)'
                                 ' or border_mode == "half"')
            height_width_depth = []
        else:
            height_width_depth = [shape[0], shape[1], shape[2]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0
            assert shape[2].ndim == 0

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False, False]
        return Apply(self, [img, topgrad] + height_width_depth,
                     [GpuArrayType(dtype=img.dtype,
                                   context_name=ctx_name,
                                   broadcastable=broadcastable)()])

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
        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode,
                                          self.subsample,
                                          self.filter_dilation,
                                          self.num_groups)(weights,
                                                           top,
                                                           bottom.shape[-3:])
        d_top = GpuCorr3dMM(
            self.border_mode, self.subsample, self.filter_dilation,
            self.num_groups)(bottom, weights)
        d_height_width_depth = (theano.gradient.DisconnectedType()(),)\
            * 3 if len(inp) == 5 else ()
        return (d_bottom, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth


class GpuCorr3dMM_gradInputs(BaseGpuCorr3dMM):
    """
    Gradient wrt. inputs for `GpuCorr3dMM`.

    Notes
    -----
    You will not want to use this directly, but rely on Theano's automatic
    differentiation or graph optimization to use it as needed.

    """

    def __init__(self, border_mode="valid",
                 subsample=(1, 1, 1),
                 filter_dilation=(1, 1, 1),
                 num_groups=1):
        super(GpuCorr3dMM_gradInputs, self).__init__(border_mode, subsample,
                                                     filter_dilation, num_groups)

    def make_node(self, kern, topgrad, shape=None):
        ctx_name = infer_context_name(kern, topgrad)
        kern = as_gpuarray_variable(kern, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if shape is None:
            if self.subsample != (1, 1, 1):
                raise ValueError('shape must be given if subsample != (1, 1, 1)')
            height_width_depth = []
        else:
            height_width_depth = [shape[0], shape[1], shape[2]]
            assert shape[0].ndim == 0
            assert shape[1].ndim == 0
            assert shape[2].ndim == 0

        if self.num_groups > 1:
            broadcastable = [topgrad.type.broadcastable[0], False,
                             False, False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[-4],
                             False, False, False]
        return Apply(self, [kern, topgrad] + height_width_depth,
                     [GpuArrayType(dtype=topgrad.dtype,
                                   context_name=ctx_name,
                                   broadcastable=broadcastable)()])

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
        d_weights = GpuCorr3dMM_gradWeights(self.border_mode,
                                            self.subsample,
                                            self.filter_dilation,
                                            self.num_groups)(bottom,
                                                             top,
                                                             weights.shape[-3:])
        d_top = GpuCorr3dMM(self.border_mode,
                            self.subsample,
                            self.filter_dilation,
                            self.num_groups)(bottom, weights)
        d_height_width_depth = (theano.gradient.DisconnectedType()(),)\
            * 3 if len(inp) == 5 else ()
        return (d_weights, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth


@inplace_allocempty(GpuGemv, 0)
def local_inplace_gpuagemv(node, inputs):
    return [gpugemv_inplace(*inputs)]


@inplace_allocempty(GpuGemm, 0)
def local_inplace_gpuagemm(node, inputs):
    return [gpugemm_inplace(*inputs)]


@inplace_allocempty(GpuGer, 0)
def local_inplace_gpuager(node, inputs):
    return [gpuger_inplace(*inputs)]


@inplace_allocempty(GpuGemmBatch, 0)
def local_inplace_gpuagemmbatch(node, inputs):
    return [gpugemmbatch_inplace(*inputs)]

gpuablas_opt_inplace = in2out(LocalOptGroup(local_inplace_gpuagemv,
                                            local_inplace_gpuagemm,
                                            local_inplace_gpuager,
                                            local_inplace_gpuagemmbatch),
                              name='gpuablas_opt_inplace')

optdb.register('InplaceGpuaBlasOpt',
               gpuablas_opt_inplace,
               70.0, 'fast_run', 'inplace', 'gpuarray')
