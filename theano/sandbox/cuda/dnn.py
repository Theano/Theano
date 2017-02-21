from __future__ import absolute_import, print_function, division
import os
import numpy
import warnings

from six import integer_types

import theano
from theano import Apply, tensor, config, Variable
from theano.scalar import as_scalar, constant, Log
from theano.gradient import DisconnectedType, grad_not_implemented
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.type import CDataType
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.tensor.nnet import LogSoftmax, SoftmaxGrad
from theano.tensor.nnet.abstract_conv import (get_conv_output_shape,
                                              assert_conv_shape)
from theano.tensor.signal.pool import (
    Pool, MaxPoolGrad, AveragePoolGrad)
from theano.tensor.nnet import bn
from theano.sandbox.cuda.type import CudaNdarrayType

from theano.sandbox.cuda import GpuOp, dnn_available
from theano.sandbox.cuda import dnn_version as version
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc, GpuAlloc,
                                           gpu_alloc_empty, GpuAllocEmpty,
                                           GpuElemwise)
from theano.sandbox.cuda.blas import (GpuConv, GpuDownsampleFactorMax,
                                      GpuDownsampleFactorMaxGrad)
from theano.sandbox.cuda.nnet import GpuSoftmax
from theano.sandbox.cuda.opt_util import (alpha_merge, output_merge,
                                          pad_dims, unpad_dims)
from theano.sandbox.cuda import gpu_seqopt, register_opt, register_inplace

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler

from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs,
                                              AbstractConv3d,
                                              AbstractConv3d_gradWeights,
                                              AbstractConv3d_gradInputs)


def c_define_tensor_desc(desc):
    return """
cudnnTensorDescriptor_t %(desc)s;
""" % dict(desc=desc)


def c_init_tensor_desc(desc, err, fail):
    return """
%(desc)s = NULL;
if ((%(err)s = cudnnCreateTensorDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
           ": %%s", cudnnGetErrorString(%(err)s));
%(fail)s
}
""" % dict(desc=desc, err=err, fail=fail)


def c_set_tensor4d(var, desc, err, fail):
    return """
{
    int str0, str1, str2, str3;
    str3 = CudaNdarray_HOST_STRIDES(%(var)s)[3]?CudaNdarray_HOST_STRIDES(%(var)s)[3]:1;
    str2 = CudaNdarray_HOST_STRIDES(%(var)s)[2]?CudaNdarray_HOST_STRIDES(%(var)s)[2]:CudaNdarray_HOST_DIMS(%(var)s)[3];
    str1 = CudaNdarray_HOST_STRIDES(%(var)s)[1]?CudaNdarray_HOST_STRIDES(%(var)s)[1]:CudaNdarray_HOST_DIMS(%(var)s)[2]*CudaNdarray_HOST_DIMS(%(var)s)[3];
    str0 = CudaNdarray_HOST_STRIDES(%(var)s)[0]?CudaNdarray_HOST_STRIDES(%(var)s)[0]:CudaNdarray_HOST_DIMS(%(var)s)[2]*CudaNdarray_HOST_DIMS(%(var)s)[3]*CudaNdarray_HOST_DIMS(%(var)s)[1];
%(err)s = cudnnSetTensor4dDescriptorEx(
    %(desc)s, CUDNN_DATA_FLOAT,
    CudaNdarray_HOST_DIMS(%(var)s)[0],
    CudaNdarray_HOST_DIMS(%(var)s)[1],
    CudaNdarray_HOST_DIMS(%(var)s)[2],
    CudaNdarray_HOST_DIMS(%(var)s)[3],
    str0, str1, str2, str3
);
if (%(err)s != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
    "could not set tensor4d descriptor: %%s"
    "shapes=%%d %%d %%d %%d strides=%%d %%d %%d %%d",
    cudnnGetErrorString(%(err)s),
    CudaNdarray_HOST_DIMS(%(var)s)[0],
    CudaNdarray_HOST_DIMS(%(var)s)[1],
    CudaNdarray_HOST_DIMS(%(var)s)[2],
    CudaNdarray_HOST_DIMS(%(var)s)[3],
    str0, str1, str2, str3
    );
    %(fail)s
}
}
        """ % dict(var=var, err=err, desc=desc, fail=fail)


def c_clean_tensor_desc(desc):
    return """
if(%(desc)s!= NULL)
cudnnDestroyTensorDescriptor(%(desc)s);
""" % dict(desc=desc)


class DnnBase(GpuOp, COp):
    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.

    """

    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False

    def __init__(self):
        COp.__init__(self, "dnn_base.c")

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compile_args(self):
        return ['-Wl,-rpath,' + config.dnn.library_path]

    def c_code_cache_version(self):
        return (super(DnnBase, self).c_code_cache_version(), version())


class GpuDnnConvDesc(GpuOp):
    """
    This Op builds a convolution descriptor for use in the other
    convolution operations.

    See the doc of :func:`dnn_conv` for a description of the parameters.

    """

    __props__ = ('border_mode', 'subsample', 'conv_mode', 'precision')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compiler(self):
        return NVCC_compiler

    def do_constant_folding(self, node):
        return False

    def __init__(self, border_mode, subsample=(1, 1), conv_mode='conv',
                 precision="float32"):
        if isinstance(border_mode, integer_types):
            border_mode = (border_mode,) * len(subsample)
        if isinstance(border_mode, tuple):
            assert len(border_mode) == len(subsample)
            border_mode = tuple(map(int, border_mode))
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode
        assert len(subsample) in [2, 3]
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

        assert precision in ['float16', 'float32', 'float64']
        self.precision = precision

    def make_node(self, img_shape, kern_shape):
        if img_shape.type.ndim != 1 or img_shape.type.dtype != 'int64':
            raise TypeError('img must be 1D shape tensor')
        if kern_shape.type.ndim != 1 or kern_shape.type.dtype != 'int64':
            raise TypeError('kern must be 1D shape tensor')

        node = Apply(self, [img_shape, kern_shape],
                     [CDataType("cudnnConvolutionDescriptor_t",
                                freefunc="cudnnDestroyConvolutionDescriptor")()])
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        return node

    def c_code(self, node, name, inputs, outputs, sub):
        img_shape, kern_shape = inputs
        desc, = outputs

        nb_dim = len(self.subsample)

        if isinstance(self.border_mode, tuple):
            pad_desc = tuple(map(int, self.border_mode))
            assert min(pad_desc) >= 0
            bmode = 1
        else:
            pad_desc = [0] * nb_dim

            if self.border_mode == "valid":
                bmode = 1
            elif self.border_mode == "half":
                bmode = 2
            else:
                assert self.border_mode == "full"
                bmode = 0

        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'

        pad_str = ", ".join([str(s) for s in pad_desc])
        subsample_str = ", ".join([str(s) for s in self.subsample])
        upscale_str = ", ".join(["1"] * nb_dim)

        if self.precision == 'float16':
            precision = 'CUDNN_DATA_HALF'
        elif self.precision == 'float32':
            precision = 'CUDNN_DATA_FLOAT'
        else:
            assert self.precision == 'float64'
            precision = 'CUDNN_DATA_DOUBLE'

        return """
{
  cudnnStatus_t err;

  if ((err = cudnnCreateConvolutionDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }

#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 30

  int pad[%(nb_dim)d] = {%(pad_str)s};
  int subsample[%(nb_dim)d] = {%(subsample_str)s};
  int upscale[%(nb_dim)d] = {%(upscale_str)s};

  // Adjust padding values if using full convolution
  if (%(bmode)d == 0) {
    pad[0] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 2) - 1;
    pad[1] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 3) - 1;
    if (%(nb_dim)d >= 3) {
        pad[2] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 4) - 1;
    }
  }
  // Adjust padding values if using half convolution
  else if (%(bmode)d == 2) {
    pad[0] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 2) / 2;
    pad[1] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 3) / 2;
    if (%(nb_dim)d >= 3) {
        pad[2] = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 4) / 2;
    }
  }

  err = cudnnSetConvolutionNdDescriptor(
  %(desc)s,
  %(nb_dim)d,
  pad, subsample, upscale,
  %(conv_flag)s, %(precision)s
  );
#else
  PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: CUDNN_VERSION must be >= 30");
#endif
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, img_shape=img_shape, kern_shape=kern_shape, desc=desc,
           bmode=bmode, conv_flag=conv_flag, fail=sub['fail'],
           pad_str=pad_str, subsample_str=subsample_str,
           upscale_str=upscale_str, nb_dim=nb_dim, precision=precision)

    def c_code_cache_version(self):
        return (4, version())

# scalar constants
_zero = constant(numpy.asarray(0.0, dtype='float32'))
_one = constant(numpy.asarray(1.0, dtype='float32'))


def ensure_float(val, default, name):
    if val is None:
        return default.clone()
    if not isinstance(val, Variable):
        val = constant(val)
    if hasattr(val, 'ndim') and val.ndim == 0:
        val = as_scalar(val)
    if not isinstance(val.type, theano.scalar.Scalar):
        raise TypeError("%s: expected a scalar value" % (name,))
    if not val.type.dtype == 'float32':
        raise TypeError("%s: type is not float32" % (name,))
    return val


class GpuDnnConv(DnnBase, COp):
    """
    The forward convolution.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'small', 'large', 'fft', 'fft_tiling', 'guess_once', 'winograd',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_fwd`.

    """

    __props__ = ('algo', 'inplace')
    __input_name__ = ('image', 'kernel', 'output',
                      'descriptor', 'alpha', 'beta')

    def __init__(self, workmem=None, inplace=False, algo=None):
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_fwd.c"],
                     "APPLY_SPECIFIC(conv_fwd)")

        if workmem is not None:
            warnings.warn(("GpuDnnConv: parameter 'workmem' is deprecated. "
                           "Use 'algo' instead."), stacklevel=3)
            assert algo is None
            self.algo = workmem
        else:
            if algo is None:
                algo = config.dnn.conv.algo_fwd
            self.algo = algo

        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

        if version() < (5000, 5000):
            if self.algo == 'winograd':
                raise RuntimeError("cuDNN winograd convolution requires "
                                   "cuDNN v5 or more recent")

        assert self.algo in ['none', 'small', 'large', 'fft', 'fft_tiling',
                             'winograd', 'guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'algo'):
            if hasattr(self, 'workmem'):
                self.algo = self.workmem
            else:
                self.algo = config.dnn.conv.algo_fwd
        if not hasattr(self, 'inplace'):
            self.inplace = False
        # Work around to reload old pickle.
        # We need to find the new file name and reload c code.
        self.load_c_code(["dnn_base.c", "dnn_conv_base.c", "dnn_fwd.c"])

    def get_op_params(self):
        if self.inplace:
            inpl_def = [('CONV_INPLACE', '1')]
        else:
            inpl_def = []

        choose_alg = '0'
        choose_alg_once = '0'
        choose_alg_time = '0'
        if version() == -1:
            alg = "0"
        else:
            if self.algo == 'none':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'
            elif self.algo == 'small':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
            elif self.algo == 'large':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'
            elif self.algo == 'direct':
                # need v2
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT'
            elif self.algo == 'fft':
                # need v3
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT'
            elif self.algo == 'fft_tiling':
                # need v4 for conv2d, need v5 for conv3d
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING'
            elif self.algo == 'winograd':
                # need v5
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD'
            elif self.algo in ['guess_once', 'guess_on_shape_change']:
                # The convolution implementation should be choosen according
                # to a heuristic
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
                choose_alg = '1'
                if self.algo == 'guess_once':
                    choose_alg_once = '1'
            elif self.algo in ['time_once', 'time_on_shape_change']:
                # The convolution implementation should be choosen by timing
                # every available implementation
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
                choose_alg = '1'
                choose_alg_time = '1'
                if self.algo == 'time_once':
                    choose_alg_once = '1'

        alg_def = ('CONV_ALGO', alg)
        alg_choose_def = ('CHOOSE_ALGO', choose_alg)
        alg_choose_once_def = ('CHOOSE_ALGO_ONCE', choose_alg_once)
        alg_choose_time_def = ('CHOOSE_ALGO_TIME', choose_alg_time)

        return [alg_def, alg_choose_def, alg_choose_once_def,
                alg_choose_time_def] + inpl_def

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be a 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, kern, output, desc, alpha, beta],
                     [output.type()])

    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConvGradI()(kerns, top, gpu_alloc_empty(*img.shape),
                                  desc)
        d_kerns = GpuDnnConvGradW()(img, top, gpu_alloc_empty(*kerns.shape),
                                    desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return [d_img * alpha, d_kerns * alpha, top * beta,
                DisconnectedType()(), d_alpha, d_beta]

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    @staticmethod
    def get_out_shape(ishape, kshape, border_mode, subsample):
        """
        This function computes the output shape for a convolution with
        the specified parameters. `ishape` and `kshape` can be symbolic
        or scalar.

        """
        return get_conv_output_shape(
            ishape,
            kshape,
            border_mode,
            subsample)

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConv3d(GpuDnnConv):
    """
    The forward convolution.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none',  'small', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_fwd`.

    """

    __props__ = ('algo', 'inplace')
    __input_name__ = ('image', 'kernel', 'output',
                      'descriptor', 'alpha', 'beta')

    def __init__(self, workmem=None, inplace=False, algo=None):
        if workmem is not None:
            warnings.warn(("GpuDnnConv3d: parameter 'workmem' is deprecated. "
                           "Use 'algo' instead."), stacklevel=3)
            assert algo is None
            algo = workmem

        good_algo = ['none', 'small', 'fft_tiling', 'winograd',
                     'guess_once', 'guess_on_shape_change',
                     'time_once', 'time_on_shape_change']
        if algo is None and config.dnn.conv.algo_fwd not in good_algo:
            algo = 'guess_once'
        elif algo is not None and algo not in good_algo:
            algo = 'guess_once'
        super(GpuDnnConv3d, self).__init__(inplace=inplace, algo=algo)

        assert self.algo in good_algo

        if version() < (5000, 5000):
            if self.algo == 'fft_tiling':
                raise RuntimeError("cuDNN 3d tiled-FFT convolution requires "
                                   "cuDNN v5 or more recent")
            elif self.algo == 'winograd':
                raise RuntimeError("cuDNN 3d winograd convolution requires "
                                   "cuDNN v5 or more recent")

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):

        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be a 5D tensor')
        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')
        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, kern, output, desc, alpha, beta],
                     [output.type()])

    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConv3dGradI()(kerns, top, gpu_alloc_empty(*img.shape),
                                    desc)
        d_kerns = GpuDnnConv3dGradW()(img, top, gpu_alloc_empty(*kerns.shape),
                                      desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return [d_img * alpha, d_kerns * alpha, top * beta,
                DisconnectedType()(), d_alpha, d_beta]

    @staticmethod
    def get_out_shape(ishape, kshape, border_mode, subsample):
        """
        This function computes the output shape for a convolution with
        the specified parameters.  `ishape` and `kshape` can be symbolic
        or scalar.
        """
        return get_conv_output_shape(
            ishape,
            kshape,
            border_mode,
            subsample)


class GpuDnnConvGradW(DnnBase, COp):
    """
    The convolution gradient with respect to the weights.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'deterministic', 'fft', 'small', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_filter`.

    """

    __props__ = ('algo', 'inplace',)
    __input_name__ = ('image', 'grad', 'output', 'descriptor', 'alpha', 'beta')

    def __init__(self, inplace=False, workmem=None, algo=None):
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_gw.c"],
                     "APPLY_SPECIFIC(conv_gw)")

        if workmem is not None:
            warnings.warn(("GpuDnnConvGradW: parameter 'workmem' is "
                           "deprecated. Use 'algo' instead."), stacklevel=3)
            assert algo is None
            self.algo = workmem
        else:
            if algo is None:
                algo = config.dnn.conv.algo_bwd_filter
            self.algo = algo

        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

        assert self.algo in ['none', 'deterministic', 'fft', 'small',
                             'guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'algo'):
            if hasattr(self, 'workmem'):
                self.algo = self.workmem
            else:
                self.algo = config.dnn.conv.algo_bwd_filter
        if not hasattr(self, 'inplace'):
            self.inplace = False
        self.load_c_code(["dnn_base.c", "dnn_conv_base.c", "dnn_gw.c"])

    def grad(self, inp, grads):
        img, top, output, desc, alpha, beta = inp
        kerns, = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnnConvGradI()(kerns, top, gpu_alloc_empty(*img.shape),
                                  desc)
        d_top = GpuDnnConv()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_img * alpha, d_top * alpha, kerns * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        if self.inplace:
            inplace_def = [('CONV_INPLACE', '1')]
        else:
            inplace_def = []

        choose_alg = '0'
        choose_alg_once = '0'
        choose_alg_time = '0'

        if version() == -1 or version() < (3000, 3000):
            alg = "0"
        else:
            if self.algo == 'none':
                # non-deterministic
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
            elif self.algo == 'deterministic':
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1'
            elif self.algo == 'fft':
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT'
            elif self.algo == 'small':
                # need v3, non-deterministic, small workspace
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3'
            elif self.algo in ['guess_once', 'guess_on_shape_change']:
                # The convolution implementation should be chosen according
                # to a heuristic
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
                choose_alg = '1'
                if self.algo == 'guess_once':
                    choose_alg_once = '1'
            elif self.algo in ['time_once', 'time_on_shape_change']:
                # The convolution implementation should be chosen according
                # to timing
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
                choose_alg = '1'
                choose_alg_time = '1'
                if self.algo == 'time_once':
                    choose_alg_once = '1'

        alg_def = ('CONV_ALGO', alg)
        alg_choose_def = ('CHOOSE_ALGO', choose_alg)
        alg_choose_once_def = ('CHOOSE_ALGO_ONCE', choose_alg_once)
        alg_choose_time_def = ('CHOOSE_ALGO_TIME', choose_alg_time)

        return inplace_def + [alg_def, alg_choose_def, alg_choose_once_def,
                              alg_choose_time_def]

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConv3dGradW(GpuDnnConvGradW):
    """
    The convolution gradient with respect to the weights.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'small', 'guess_once', 'guess_on_shape_change',
            'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_filter`.

    """

    __props__ = ('algo', 'inplace',)
    __input_name__ = ('image', 'grad', 'output', 'descriptor', 'alpha', 'beta')

    def __init__(self, inplace=False, workmem=None, algo=None):
        if workmem is not None:
            warnings.warn(("GpuDnnConv3dGradW: parameter 'workmem' is "
                           "deprecated. Use 'algo' instead."), stacklevel=3)
            assert algo is None
            algo = workmem
        good_algo = ['none', 'small',
                     'guess_once', 'guess_on_shape_change',
                     'time_once', 'time_on_shape_change']
        if version() < (5000, 5000) and algo == 'small':
            algo = 'guess_once'
        elif algo is None and config.dnn.conv.algo_bwd_filter not in good_algo:
            algo = 'guess_once'
        elif algo is not None and algo not in good_algo:
            algo = 'guess_once'
        super(GpuDnnConv3dGradW, self).__init__(inplace=inplace,
                                                algo=algo)
        assert self.algo in good_algo

    def grad(self, inp, grads):
        img, top, output, desc, alpha, beta = inp
        kerns, = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnnConv3dGradI()(kerns, top, gpu_alloc_empty(*img.shape),
                                    desc)
        d_top = GpuDnnConv3d()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_img * alpha, d_top * alpha, kerns * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        if self.algo != 'none' and desc.owner.op.subsample != (1, 1, 1):
            warnings.warn('cuDNN backward filter operation for 3D convolutions may produce bad results '
                          'with certain cuDNN algorithms depending on the compute capability of your GPU '
                          'if subsample is not (1, 1, 1). If you encounter problems, consider '
                          'setting the theano flag "dnn.conv.algo_bwd_filter" to "none".')
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, topgrad, output, desc, alpha, beta],
                     [output.type()])


class GpuDnnConvGradI(DnnBase, COp):
    """
    The convolution gradient with respect to the inputs.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'deterministic', 'fft', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_data`.

    """

    __props__ = ('algo', 'inplace',)
    __input_name__ = ('kernel', 'grad', 'output', 'descriptor', 'alpha',
                      'beta')

    def __init__(self, inplace=False, workmem=None, algo=None):
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_gi.c"],
                     "APPLY_SPECIFIC(conv_gi)")

        if workmem is not None:
            warnings.warn(("GpuDnnConvGradI: parameter 'workmem' is "
                           "deprecated. Use 'algo' instead."), stacklevel=3)
            assert algo is None
            self.algo = workmem
        else:
            if algo is None:
                algo = config.dnn.conv.algo_bwd_data
            self.algo = algo

        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

        if version() < (5000, 5000):
            if self.algo == 'winograd':
                raise RuntimeError("cuDNN's winograd convolution requires "
                                   "cuDNN v5 or more recent")

        assert self.algo in ['none', 'deterministic', 'fft', 'fft_tiling',
                             'winograd', 'guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'algo'):
            if hasattr(self, 'workmem'):
                self.algo = self.workmem
            else:
                self.algo = config.dnn.conv.algo_bwd_data
        if not hasattr(self, 'inplace'):
            self.inplace = False
        self.load_c_code(["dnn_base.c", "dnn_conv_base.c", "dnn_gi.c"])

    def grad(self, inp, grads):
        kerns, top, output, desc, alpha, beta = inp
        img, = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnnConvGradW()(img, top, gpu_alloc_empty(*kerns.shape),
                                    desc)
        d_top = GpuDnnConv()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_kerns * alpha, d_top * alpha, img * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        if self.inplace:
            inplace_def = [('CONV_INPLACE', '1')]
        else:
            inplace_def = []

        choose_alg = '0'
        choose_alg_once = '0'
        choose_alg_time = '0'

        if version() == -1 or version() < (3000, 3000):
            alg = "0"
        else:
            if self.algo == 'none':
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
            elif self.algo == 'deterministic':
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1'
            elif self.algo == 'fft':
                # need v3, big workspace
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT'
            elif self.algo == 'fft_tiling':
                # need v4, big workspace, but less then fft
                # need v5, for conv3d.
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING'
            elif self.algo == 'winograd':
                # need v5
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD'
            elif self.algo in ['guess_once', 'guess_on_shape_change']:
                # The convolution implementation should be chosen according
                # to a heuristic
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
                choose_alg = '1'
                if self.algo == 'guess_once':
                    choose_alg_once = '1'
            elif self.algo in ['time_once', 'time_on_shape_change']:
                # The convolution implementation should be chosen according
                # to timing
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
                choose_alg = '1'
                choose_alg_time = '1'
                if self.algo == 'time_once':
                    choose_alg_once = '1'

        alg_def = ('CONV_ALGO', alg)
        alg_choose_def = ('CHOOSE_ALGO', choose_alg)
        alg_choose_once_def = ('CHOOSE_ALGO_ONCE', choose_alg_once)
        alg_choose_time_def = ('CHOOSE_ALGO_TIME', choose_alg_time)

        return inplace_def + [alg_def, alg_choose_def, alg_choose_once_def,
                              alg_choose_time_def]

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [kern, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConv3dGradI(GpuDnnConvGradI):
    """
    The convolution gradient with respect to the inputs.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'deterministic, 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_data`.

    """

    __props__ = ('algo', 'inplace',)
    __input_name__ = ('kernel', 'grad', 'output', 'descriptor', 'alpha',
                      'beta')

    def __init__(self, inplace=False, workmem=None, algo=None):
        if workmem is not None:
            warnings.warn(("GpuDnnConv3dGradI: parameter 'workmem' is "
                           "deprecated. Use 'algo' instead."), stacklevel=3)
            assert algo is None
            algo = workmem

        good_algo = ['none', 'deterministic', 'fft_tiling', 'winograd',
                     'guess_once', 'guess_on_shape_change', 'time_once',
                     'time_on_shape_change']

        if algo is None and config.dnn.conv.algo_bwd_data not in good_algo:
            algo = 'guess_once'
        elif algo is not None and algo not in good_algo:
            algo = 'guess_once'
        super(GpuDnnConv3dGradI, self).__init__(inplace=inplace,
                                                algo=algo)
        assert self.algo in good_algo
        if version() < (5000, 5000):
            if self.algo == 'fft_tiling':
                raise RuntimeError("cuDNN 3d tiled-FFT convolution requires "
                                   "cuDNN v5 or more recent")
            elif self.algo == 'winograd':
                raise RuntimeError("cuDNN 3d winograd convolution requires "
                                   "cuDNN v5 or more recent")

    def grad(self, inp, grads):
        kerns, top, output, desc, alpha, beta = inp
        img, = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnnConv3dGradW()(img, top, gpu_alloc_empty(*kerns.shape),
                                      desc)
        d_top = GpuDnnConv3d()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_kerns * alpha, d_top * alpha, img * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [kern, topgrad, output, desc, alpha, beta],
                     [output.type()])


def dnn_conv(img, kerns, border_mode='valid', subsample=(1, 1),
             conv_mode='conv', direction_hint=None, workmem=None, algo=None,
             precision=None):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    Parameters
    ----------
    img
        Images to do the convolution over.
    kerns
        Convolution filters.
    border_mode
        One of 'valid', 'full', 'half'; additionally, the padding size can be
        directly specified by an integer or a pair of integers (as a tuple),
        specifying the amount of zero padding added to _both_ the top and
        bottom (first entry) and left and right (second entry) sides of
        the image.
    subsample
        Perform subsampling of the output (default: (1, 1)).
    conv_mode
        Perform convolution (kernels flipped) or cross-correlation.
        One of 'conv', 'cross' (default: 'conv').
    direction_hint
        Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1,1) and direction_hint is
        'bprop weights', it will use GpuDnnConvGradW.
        If border_mode is 'full', subsample is (1,1) and direction_hint is
        'bprop inputs', it will use GpuDnnConvGradI.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    workmem
        *deprecated*, use parameter algo instead.
    algo : {'none', 'small', 'large', 'fft', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Convolution implementation to use. Some of its  values may require certain
        versions of cuDNN to be installed. Default is the value of
        :attr:`config.dnn.conv.algo_fwd`.
    precision : {'as_input_f32', 'as_input', 'float16', 'float32', 'float64'}
        Description of the dtype in which the computation of the convolution
        should be done. Possible values are 'as_input', 'float16', 'float32'
        and 'float64'. Default is the value of
        :attr:`config.dnn.conv.precision`.

    """
    # For consistence, when using direction_hint too.
    if border_mode == (0, 0):
        border_mode = 'valid'

    # Establish dtype in which to perform the computation of the convolution
    if precision is None:
        precision = theano.config.dnn.conv.precision
    if precision == 'as_input' or precision == 'as_input_f32':
        nprec = theano.scalar.upcast(img.dtype, kerns.dtype)
        if nprec == 'float16' and precision == 'as_input_f32':
            precision = 'float32'
        else:
            precision = nprec

    # Check if deprecated param 'workmem' is used
    if workmem is not None:
        warnings.warn(("dnn_conv: parameter 'workmem' is deprecated. Use "
                       "'algo' instead."), stacklevel=3)
        assert algo is None
        algo = workmem

    # Ensure the value of direction_hint is supported
    assert direction_hint in [None, 'bprop weights', 'bprop inputs', 'forward']

    fgraph = getattr(img, 'fgraph', None) or getattr(kerns, 'fgraph', None)
    if (border_mode == 'valid' and subsample == (1, 1) and
            direction_hint == 'bprop weights'):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
        if conv_mode == 'conv':
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        out_shp = (shape_i(kerns, 1, fgraph),
                   shape_i(img, 1, fgraph),
                   shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
                   shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode='cross', precision=precision)(img.shape,
                                                                      out.shape)
        conv = GpuDnnConvGradW()(img, kerns, out, desc)
        return as_cuda_ndarray_variable(conv.dimshuffle(1, 0, 2, 3))

    elif (border_mode == 'full' and subsample == (1, 1) and
            direction_hint == 'bprop inputs'):
        # Special case: We are asked to use GpuDnnConvGradI. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img)
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        out_shp = (shape_i(img, 0, fgraph),
                   shape_i(kerns, 1, fgraph),
                   shape_i(img, 2, fgraph) + shape_i(kerns, 2, fgraph) - 1,
                   shape_i(img, 3, fgraph) + shape_i(kerns, 3, fgraph) - 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode=conv_mode, precision=precision)(out.shape,
                                                                        kerns.shape)
        return GpuDnnConvGradI()(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode, precision=precision)(img.shape,
                                                                    kerns.shape)
    desc_op = desc.owner.op
    out_shp = GpuDnnConv.get_out_shape(img.shape, kerns.shape,
                                       desc_op.border_mode,
                                       desc_op.subsample)
    out_shp = assert_conv_shape(out_shp)
    out = gpu_alloc_empty(*out_shp)
    return GpuDnnConv(algo=algo)(img, kerns, out, desc)


def dnn_conv3d(img, kerns, border_mode='valid', subsample=(1, 1, 1),
               conv_mode='conv', direction_hint=None, workmem=None,
               algo=None, precision=None):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bct01', that is 'batch', 'channel',
    'first dim', 'second dim', 'third dim' in that order.

    :param img: images to do the convolution over
    :param kerns: convolution filters
    :param border_mode: One of 'valid', 'full', 'half'; additionally, the
        padding size can be directly specified by an integer or a triplet of
        integers (as a tuple), specifying the amount of zero padding added to
        _both_ the top and bottom (first entry) and left and right (second
        entry) and front and back (third entry) sides of the volume.
    :param subsample: perform subsampling of the output (default: (1, 1, 1))
    :param conv_mode: perform convolution (kernels flipped) or
        cross-correlation. One of 'conv', 'cross'. (default: 'conv')
    :param direction_hint: Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1,1,1) and direction_hint is
        'bprop weights', it will use GpuDnnConvGradW.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    :param workmem: *deprecated*, use param algo instead
    :param algo: convolution implementation to use. Only 'none' is implemented
        for the conv3d. Default is the value of
        :attr:`config.dnn.conv.algo_fwd`.
    :param precision: dtype in which the computation of the convolution
        should be done. Possible values are 'as_input_f32', 'as_input',
        'float16', 'float32' and 'float64'. Default is the value of
        :attr:`config.dnn.conv.precision`.

    :warning: The cuDNN library only works with GPU that have a compute
        capability of 3.0 or higer.  This means that older GPU will not
        work with this Op.
    :warning: dnn_conv3d only works with cuDNN library 3.0

    """
    if border_mode == (0, 0, 0):
        border_mode = 'valid'

    # Establish dtype in which to perform the computation of the convolution
    if precision is None:
        precision = theano.config.dnn.conv.precision
    if precision == 'as_input' or precision == 'as_input_f32':
        nprec = theano.scalar.upcast(img.dtype, kerns.dtype)
        if nprec == 'float16' and precision == 'as_input_f32':
            precision = 'float32'
        else:
            precision = nprec

    # Check if deprecated param 'workmem' is used
    if workmem is not None:
        warnings.warn(("dnn_conv3d: parameter 'workmem' is deprecated. Use "
                       "'algo' instead."), stacklevel=3)
        assert algo is None
        algo = workmem

    # Ensure the value of direction_hint is supported
    assert direction_hint in [None, 'bprop weights', 'forward']

    fgraph = getattr(img, 'fgraph', None) or getattr(kerns, 'fgraph', None)
    if (border_mode == 'valid' and subsample == (1, 1, 1) and
            direction_hint == 'bprop weights'):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
        if conv_mode == 'conv':
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
        out_shp = (shape_i(kerns, 1, fgraph),
                   shape_i(img, 1, fgraph),
                   shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
                   shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1,
                   shape_i(img, 4, fgraph) - shape_i(kerns, 4, fgraph) + 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1, 1),
                              conv_mode='cross', precision=precision)(img.shape,
                                                                      out.shape)
        conv = GpuDnnConv3dGradW()(img, kerns, out, desc)
        return as_cuda_ndarray_variable(conv.dimshuffle(1, 0, 2, 3, 4))

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode, precision=precision)(img.shape,
                                                                    kerns.shape)
    desc_op = desc.owner.op
    out_shp = GpuDnnConv3d.get_out_shape(img.shape, kerns.shape,
                                         desc_op.border_mode,
                                         desc_op.subsample)
    out_shp = assert_conv_shape(out_shp)
    out = gpu_alloc_empty(*out_shp)
    return GpuDnnConv3d(algo=algo)(img, kerns, out, desc)


def dnn_gradweight(img, topgrad,
                   kerns_shp,
                   border_mode='valid', subsample=(1, 1),
                   conv_mode='conv'):
    """
    GPU convolution gradient with respect to weight using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    FIXME parameters doc

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """

    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    kerns_shp = theano.tensor.as_tensor_variable(kerns_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns_shp)
    out = gpu_alloc_empty(*kerns_shp)
    return GpuDnnConvGradW()(img, topgrad, out, desc)


def dnn_gradweight3d(img, topgrad,
                     kerns_shp,
                     border_mode='valid', subsample=(1, 1, 1),
                     conv_mode='conv'):
    """
    GPU convolution gradient with respect to weight using cuDNN from NVIDIA.

    The memory layout to use is 'bct01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    FIXME parameters doc

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """

    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    kerns_shp = theano.tensor.as_tensor_variable(kerns_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns_shp)
    out = gpu_alloc_empty(*kerns_shp)
    return GpuDnnConv3dGradW()(img, topgrad, out, desc)


def dnn_gradinput(kerns, topgrad,
                  img_shp,
                  border_mode='valid', subsample=(1, 1),
                  conv_mode='conv'):
    """
    GPU convolution gradient with respect to input using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    FIXME parameters doc

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """

    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)
    img_shp = theano.tensor.as_tensor_variable(img_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img_shp, kerns.shape)

    out = gpu_alloc_empty(*img_shp)
    return GpuDnnConvGradI()(kerns, topgrad, out, desc)


def dnn_gradinput3d(kerns, topgrad,
                    img_shp,
                    border_mode='valid', subsample=(1, 1),
                    conv_mode='conv'):
    """
    GPU convolution gradient with respect to input using cuDNN from NVIDIA.

    The memory layout to use is 'bct01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    FIXME parameters doc

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """

    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)
    img_shp = theano.tensor.as_tensor_variable(img_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img_shp, kerns.shape)

    out = gpu_alloc_empty(*img_shp)
    return GpuDnnConv3dGradI()(kerns, topgrad, out, desc)


class GpuDnnPoolDesc(GpuOp):
    """
    This Op builds a pooling descriptor for use in the other pooling operations.

    Parameters
    ----------
    ws
        Windows size.
    stride
        (dx, dy).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' correspond to 'average_inc_pad'.
    pad
        (pad_h, pad_w) padding information.
        pad_h is the number of zero-valued pixels added to each of the top and
        bottom borders.
        pad_w is the number of zero-valued pixels added to each of the left and
        right borders.

    Note
    ----
    Do not use anymore. Only needed to reload old pickled files.

    """

    __props__ = ('ws', 'stride', 'mode', 'pad')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compiler(self):
        return NVCC_compiler

    def do_constant_folding(self, node):
        return False

    def __init__(self, ws=(1, 1), stride=None, mode='max', pad=None):
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

        if stride is None:
            stride = (1,) * len(ws)
        if pad is None:
            pad = (0,) * len(ws)

        assert len(ws) == len(stride) and len(stride) == len(pad)
        assert len(ws) in (2, 3)
        self.ws = ws
        self.stride = stride
        self.pad = pad

    def get_ndim(self):
        return len(self.ws)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'pad'):
            self.pad = (0,) * self.get_ndim()

    def make_node(self):
        node = Apply(self, [],
                     [CDataType("cudnnPoolingDescriptor_t",
                                freefunc="cudnnDestroyPoolingDescriptor")()])
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        return node

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
        else:
            raise NotImplementedError("Unsupported pooling model.")

        return """
{
  cudnnStatus_t err;

  if ((err = cudnnCreatePoolingDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate pooling "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }
  {
    int win[%(nd)d] = {%(win)s};
    int pad[%(nd)d] = {%(pad)s};
    int str[%(nd)d] = {%(str)s};
    err = cudnnSetPoolingNdDescriptor_v4(
      %(desc)s, %(mode_flag)s,
      CUDNN_PROPAGATE_NAN, %(nd)d,
      win, pad, str);
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, desc=desc, mode_flag=mode_flag, fail=sub['fail'],
           nd=self.get_ndim(), win=', '.join(str(w) for w in self.ws),
           pad=', '.join(str(p) for p in self.pad),
           str=', '.join(str(s) for s in self.stride))

    def c_code_cache_version(self):
        return (3, version())


class GpuDnnPool(DnnBase):
    """
    Pooling.

    Parameters
    ----------
    img
        The image 4d or 5d tensor.
    ws
        Windows size.
    stride
        (dx, dy).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' correspond to 'average_inc_pad'.
    pad
        (padX, padY) padding information.
        padX is the size of the left and right borders,
        padY is the size of the top and bottom borders.

    """

    __props__ = ("mode",)

    def __init__(self, mode='max'):
        super(GpuDnnPool, self).__init__()
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

    def prepare_node(self, node, storage_map, compute_map, impl):
        super(GpuDnnPool, self).prepare_node(
            node, storage_map, compute_map, impl)

        if len(node.inputs) == 2:
            warnings.warn("Theano GPUDnnPoolGrad internal changed.", stacklevel=3)
            # Old interface
            self.mode = node.inputs[1].owner.op.mode
            ws = theano.tensor.constant(node.inputs[1].owner.op.ws)
            st = theano.tensor.constant(node.inputs[1].owner.op.stride)
            pad = theano.tensor.constant(node.inputs[1].owner.op.pad)
            node.inputs[1] = ws
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]

    def make_node(self, img, ws, stride, pad):
        img = as_cuda_ndarray_variable(img)
        assert (img.ndim in [4, 5])

        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [img, ws, stride, pad], [img.type()])

    def infer_shape(self, node, shape):
        w = node.inputs[1]
        s = node.inputs[2]
        p = node.inputs[3]

        ret = [shape[0][0], shape[0][1],
               (shape[0][2] + 2 * p[0] - w[0]) // s[0] + 1,
               (shape[0][3] + 2 * p[1] - w[1]) // s[1] + 1]
        if node.inputs[0].ndim == 5:
            ret.append((shape[0][4] + 2 * p[2] - w[2]) // s[2] + 1)
        return [ret]

    def c_support_code_struct(self, node, name):
        return """
cudnnTensorDescriptor_t input%(name)s;
cudnnTensorDescriptor_t output%(name)s;
cudnnPoolingDescriptor_t pool%(name)s;
""" % dict(name=name)

    def c_init_code_struct(self, node, name, sub):
        return """
cudnnStatus_t err%(name)s;
input%(name)s = NULL;
output%(name)s = NULL;
pool%(name)s = NULL;
if ((err%(name)s = cudnnCreateTensorDescriptor(&input%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&output%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}

if ((err%(name)s = cudnnCreatePoolingDescriptor(&pool%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate pooling "
                "descriptor: %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(name=name, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, name):
        return """
if (input%(name)s != NULL) { cudnnDestroyTensorDescriptor(input%(name)s); }
if (output%(name)s != NULL) { cudnnDestroyTensorDescriptor(output%(name)s); }
if (pool%(name)s != NULL) { cudnnDestroyPoolingDescriptor(pool%(name)s); }
""" % dict(name=name)

    def c_code(self, node, name, inputs, outputs, sub):
        ws = inputs[1]
        stride = inputs[2]
        pad = inputs[3]
        out, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
        else:
            raise NotImplementedError("Unsupported pooling model.")

        return """
cudnnStatus_t err;

int %(out)s_dims[5];

if (!CudaNdarray_is_c_contiguous(%(input)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

int win[%(nd)d];
int pad[%(nd)d];
int str[%(nd)d];
for(int i = 0; i < %(nd)d; i++) {
   win[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
}
for(int i = 0; i < %(nd)d; i++) {
   pad[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
}
for(int i = 0; i < %(nd)d; i++) {
   str[i] = *((npy_intp*)PyArray_GETPTR1(%(str)s, i));
}
err = cudnnSetPoolingNdDescriptor_v4(
    pool%(name)s, %(mode_flag)s,
    CUDNN_PROPAGATE_NAN, %(nd)d,
    win, pad, str);

if (err != CUDNN_STATUS_SUCCESS) {
PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                cudnnGetErrorString(err));
%(fail)s
}

%(out)s_dims[0] = CudaNdarray_HOST_DIMS(%(input)s)[0];
%(out)s_dims[1] = CudaNdarray_HOST_DIMS(%(input)s)[1];
%(out)s_dims[2] = (CudaNdarray_HOST_DIMS(%(input)s)[2] + (pad[0]*2) - win[0]) / str[0] + 1;
%(out)s_dims[3] = (CudaNdarray_HOST_DIMS(%(input)s)[3] + (pad[1]*2) - win[1]) / str[1] + 1;
if (%(nd)s == 3)
  %(out)s_dims[4] = (CudaNdarray_HOST_DIMS(%(input)s)[4] + (pad[2]*2) - win[2]) / str[2] + 1;

if (CudaNdarray_prep_output(&%(out)s, %(nd)s+2, %(out)s_dims) != 0)
{
  %(fail)s
}

// if input batch is empty, we return the empty output without calling cuDNN
// (which will fail on zero batch size).
// Ideally, "return success" here, but we don't have a %%(done)s, so just skip the call.
if (CudaNdarray_DIMS(%(input)s)[0] > 0) {
// Don't indent for keeping history

if (c_set_tensorNd(%(input)s, %(input_desc)s) != 0)
  %(fail)s

if (c_set_tensorNd(%(out)s, %(output_desc)s) != 0)
  %(fail)s

{
const float alpha = 1;
const float beta = 0;
err = cudnnPoolingForward(
_handle,
pool%(name)s,
&alpha,
%(input_desc)s, CudaNdarray_DEV_DATA(%(input)s),
&beta,
%(output_desc)s, CudaNdarray_DEV_DATA(%(out)s)
);
}
if (err != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnPoolingForward operation: %%s",
               cudnnGetErrorString(err));
  %(fail)s
}

} // Closes the batchdim > 0 check.
""" % dict(out=out, fail=sub['fail'],
           name=name, input=inputs[0],
           ws=ws, pad=pad, str=stride,
           nd=node.inputs[0].ndim - 2, input_desc="input" + name,
           output_desc="output" + name,
           mode_flag=mode_flag)

    def grad(self, inp, grads):
        img, ws, stride, pad = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        out = self(img, ws, stride, pad)

        g_out = GpuDnnPoolGrad(mode=self.mode)(img, out, grad, ws, stride, pad)

        return g_out, theano.gradient.DisconnectedType()(), theano.gradient.DisconnectedType()(), theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [0], [0], [0]]

    def c_code_cache_version(self):
        return (9, version())


class GpuDnnPoolGrad(DnnBase):
    """
    The pooling gradient.

    Parameters
    ----------
    inp
        The input of the pooling.
    out
        The output of the pooling in the forward.
    inp_grad
        Same size as out, but is the corresponding gradient information.
    ws
        Windows size.
    stride
        (dx, dy).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' correspond to 'average_inc_pad'.
    pad
        (padX, padY) padding information.
        padX is the size of the left and right borders,
        padY is the size of the top and bottom borders.
    """

    __props__ = ('mode',)

    def __init__(self, mode='max'):
        super(GpuDnnPoolGrad, self).__init__()
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 4:
            warnings.warn("Theano GPUDnnPoolGrad internal changed.", stacklevel=3)
            # Old interface
            self.mode = node.inputs[3].owner.op.mode
            ws = theano.tensor.constant(node.inputs[3].owner.op.ws)
            st = theano.tensor.constant(node.inputs[3].owner.op.stride)
            pad = theano.tensor.constant(node.inputs[3].owner.op.pad)
            node.inputs[3] = ws
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]

    def make_node(self, inp, out, inp_grad, ws, stride, pad):
        inp = as_cuda_ndarray_variable(inp)
        assert (inp.ndim in [4, 5])
        inp_grad = as_cuda_ndarray_variable(inp_grad)
        assert (inp_grad.ndim in [4, 5])
        out = as_cuda_ndarray_variable(out)
        assert(out.ndim in [4, 5])

        assert (inp_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, inp_grad, ws, stride, pad],
                     [inp.type()])

    def c_support_code_struct(self, node, name):
        return """
cudnnTensorDescriptor_t input%(name)s;
cudnnTensorDescriptor_t input_grad%(name)s;
cudnnTensorDescriptor_t output%(name)s;
cudnnTensorDescriptor_t output_grad%(name)s;
cudnnPoolingDescriptor_t pool%(name)s;
""" % dict(name=name)

    def c_init_code_struct(self, node, name, sub):
        return """
cudnnStatus_t err%(name)s;
input%(name)s = NULL;
input_grad%(name)s = NULL;
output%(name)s = NULL;
output_grad%(name)s = NULL;
pool%(name)s = NULL;
if ((err%(name)s = cudnnCreateTensorDescriptor(&input%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(input): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&input_grad%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(input_grad): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&output%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(output): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&output_grad%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(output_grad): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreatePoolingDescriptor(&pool%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate pooling descriptor "
               "(pool): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(name=name, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, name):
        return """
if (input%(name)s != NULL) { cudnnDestroyTensorDescriptor(input%(name)s); }
if (input_grad%(name)s != NULL) { cudnnDestroyTensorDescriptor(input_grad%(name)s); }
if (output%(name)s != NULL) { cudnnDestroyTensorDescriptor(output%(name)s); }
if (output_grad%(name)s != NULL) { cudnnDestroyTensorDescriptor(output_grad%(name)s); }
if (pool%(name)s != NULL) { cudnnDestroyPoolingDescriptor(pool%(name)s); }
""" % dict(name=name)

    def c_code(self, node, name, inputs, outputs, sub):
        # Here the name out and inp are based on the cudnn definition.
        # Not the definition of this class.
        # This make it complicated.
        out, inp, inp_grad, ws, stride, pad = inputs

        out_grad, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
        else:
            raise NotImplementedError("Unsupported pooling model.")

        return """
cudnnStatus_t err%(name)s;

if (!CudaNdarray_is_c_contiguous(%(input)s)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous inputs are supported.");
  %(fail)s
}

if (!CudaNdarray_is_c_contiguous(%(input_grad)s)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous input gradients are supported.");
  %(fail)s
}

if (!CudaNdarray_is_c_contiguous(%(output)s)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous outputs are supported.");
  %(fail)s
}

if (CudaNdarray_prep_output(&%(output_grad)s,
                            %(output)s->nd,
                            CudaNdarray_HOST_DIMS(%(output)s)) != 0)
{
  %(fail)s
}

// if input batch is empty, we return the empty output without calling cuDNN
// (which will fail on zero batch size).
// Ideally, "return success" here, but we don't have a %%(done)s, so just skip the call.
if (CudaNdarray_DIMS(%(input)s)[0] > 0) {
// Don't indent for keeping history

if (c_set_tensorNd(%(input)s, %(input_desc)s) != 0)
  %(fail)s
if (c_set_tensorNd(%(input_grad)s, %(input_grad_desc)s) != 0)
  %(fail)s
if (c_set_tensorNd(%(output)s, %(output_desc)s) != 0)
  %(fail)s

int win[%(nd)d];
int pad[%(nd)d];
int str[%(nd)d];
for(int i = 0; i < %(nd)d; i++) {
   win[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
}
for(int i = 0; i < %(nd)d; i++) {
   pad[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
}
for(int i = 0; i < %(nd)d; i++) {
   str[i] = *((npy_intp*)PyArray_GETPTR1(%(str)s, i));
}
err%(name)s = cudnnSetPoolingNdDescriptor_v4(
    pool%(name)s, %(mode_flag)s,
    CUDNN_PROPAGATE_NAN, %(nd)d,
    win, pad, str);

if (err%(name)s != CUDNN_STATUS_SUCCESS) {
PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                cudnnGetErrorString(err%(name)s));
%(fail)s
}

if (c_set_tensorNd(%(output_grad)s, %(output_grad_desc)s) != 0)
  %(fail)s

{
const float alpha = 1;
const float beta = 0;
err%(name)s = cudnnPoolingBackward(
_handle,
pool%(name)s,
&alpha,
%(input_desc)s, CudaNdarray_DEV_DATA(%(input)s),
%(input_grad_desc)s, CudaNdarray_DEV_DATA(%(input_grad)s),
%(output_desc)s, CudaNdarray_DEV_DATA(%(output)s),
&beta,
%(output_grad_desc)s, CudaNdarray_DEV_DATA(%(output_grad)s)
);
}
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPoolGrad: error doing operation: %%s.",
               cudnnGetErrorString(err%(name)s));
 %(fail)s
}

} // Closes the batchdim > 0 check.
""" % dict(output_grad=out_grad,
           fail=sub['fail'], name=name,
           input=inp, input_grad=inp_grad, output=out,
           input_desc="input" + name,
           input_grad_desc="input_grad" + name,
           output_desc="output" + name,
           output_grad_desc="output_grad" + name,
           mode_flag=mode_flag, nd=node.inputs[0].ndim - 2,
           ws=ws, pad=pad, str=stride)

    def c_code_cache_version(self):
        return (9, version())

    def infer_shape(self, node, shape):
        return [shape[0]]


def dnn_pool(img, ws, stride=None, mode='max', pad=None):
    """
    GPU pooling using cuDNN from NVIDIA.

    For 2D pooling, the memory layout to use is 'bc01', that is 'batch',
    'channel', 'first dim', 'second dim' in that order.

    For 3D pooling, the memory layout to use is 'bc012', that is 'batch',
    'channel', 'first dim', 'second dim', 'third dim'.

    Parameters
    ----------
    img
        Images to do the pooling over.
    ws
        Subsampling window size.  Should have 2 or 3 elements.
    stride
        Subsampling stride (default: (1, 1) or (1, 1, 1)).
    mode : {'max', 'average_inc_pad', 'average_exc_pad', 'sum'}
    pad
        Padding: (pad_h, pad_w) for 2D or (pad_h, pad_w, pad_d) for 3D.
        pad_h is the number of zero-valued pixels added to each of the top and
        bottom borders.
        pad_w is the number of zero-valued pixels added to each of the left
        and right borders.
        pad_d is the number of zero-valued pixels added to each of the front
        and back borders (3D pooling only).


    .. warning:: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.

    Notes
    -----
    This Op implements the ignore_border=True of max_pool_2d.

    """
    img = gpu_contiguous(img)
    if stride is None:
        stride = (1,) * len(ws)
    if pad is None:
        pad = (0,) * len(ws)
    if mode == "sum":
        ret = GpuDnnPool(mode="average_inc_pad")(img, ws, stride, pad)
        window_elem = theano.tensor.prod(ws).astype(ret.dtype)
        return as_cuda_ndarray_variable(ret * window_elem)

    return GpuDnnPool(mode=mode)(img, ws, stride, pad)


class GpuDnnSoftmaxBase(DnnBase):
    """
    Op for the cuDNN Softmax.

    Parameters
    ----------
    tensor_format
        Always set this to 'bc01'.
    algo : {'fast', 'accurate', 'log'}
        Indicating whether, respectively, computations should be optimized for
        speed, for accuracy, or if cuDNN should rather compute the log-softmax instead.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    __props__ = ('tensor_format', 'mode', 'algo')

    def __init__(self, tensor_format, algo, mode):
        if tensor_format != 'bc01':
            raise ValueError(
                "It was discovered that since December 2014, the "
                "tensor_format parameter was ignored and the equivalent of "
                "'bc01' is always used.  Since your code seems to be using "
                "another value, this might have affected previous results "
                "ran with this code.")
        DnnBase.__init__(self)
        self.tensor_format = tensor_format

        assert(algo in ('fast', 'accurate', 'log'))
        self.algo = algo

        assert(mode in ('instance', 'channel'))
        self.mode = mode

        self.tensor_4d_descs = [softmax_input
                                for softmax_input in self.softmax_inputs]
        self.tensor_4d_descs.append('softmax_output')

    def infer_shape(self, node, shape):
        if self.direction == 'forward':
            return [shape[0]]
        else:
            return [shape[1]]

    def c_support_code_struct(self, node, name):
        result = ''
        for id in self.tensor_4d_descs:
            result += c_define_tensor_desc('%s_%s' % (id, name))
        return result

    def c_init_code_struct(self, node, name, sub):
        result = """
cudnnStatus_t err%(name)s;
""" % dict(name=name)

        for id in self.tensor_4d_descs:
            result += c_init_tensor_desc('%s_%s' % (id, name), 'err' + name, sub['fail'])
        return result

    def c_cleanup_code_struct(self, node, name):
        result = ''
        for id in self.tensor_4d_descs:
            result += c_clean_tensor_desc('%s_%s' % (id, name))
        return result

    def c_code(self, node, name, inputs, outputs, sub):
        ins = inputs
        outs, = outputs

        if self.tensor_format == 'b01c':
            tensor_format = 1
        else:
            tensor_format = 0

        if self.mode == 'instance':
            mode = 1
        else:
            mode = 0

        if self.algo == 'fast':
            algo = "CUDNN_SOFTMAX_FAST"
        elif self.algo == "log":
            algo = "CUDNN_SOFTMAX_LOG"
        else:
            algo = "CUDNN_SOFTMAX_ACCURATE"

        # Setup configuration variables.
        result = """
cudnnStatus_t err%(name)s;
cudnnTensorFormat_t format%(name)s = CUDNN_TENSOR_NCHW;
if (%(tensor_format)d == 1)
  format%(name)s = CUDNN_TENSOR_NHWC;

cudnnSoftmaxAlgorithm_t algo%(name)s = %(algo)s;

cudnnSoftmaxMode_t mode%(name)s = CUDNN_SOFTMAX_MODE_CHANNEL;
if (%(mode)d == 1)
  mode%(name)s = CUDNN_SOFTMAX_MODE_INSTANCE;
""" % dict(name=name, tensor_format=tensor_format, mode=mode, algo=algo)

        # Validate the input and build the input variables.
        for input_idx, input_name in enumerate(self.softmax_inputs):
            result += c_set_tensor4d(ins[input_idx], input_name + "_" + name,
                                     "err" + name, sub['fail'])

        subs = dict(ins=ins[-1], outs=outs, fail=sub['fail'],
                    name=name)

        for idx, softmax_input in enumerate(self.softmax_inputs):
            subs['name%d' % idx] = softmax_input
            subs['ins%d' % idx] = inputs[idx]

        # Build and prepare the output variable.
        result += """
if (CudaNdarray_prep_output(&%(outs)s, 4, CudaNdarray_HOST_DIMS(%(ins)s)) != 0)
{
  %(fail)s
}
""" % subs
        result += c_set_tensor4d(outs,
                                 "softmax_output_" + name,
                                 "err" + name, sub['fail'])

        # Add on a call to the method that does the actual work.
        result += self.method() % subs

        return result

    def c_code_cache_version(self):
        return (0, 6, version())

    def method(self):
        raise NotImplementedError('GpuDnnSoftmaxBase::method')


class GpuDnnSoftmax(GpuDnnSoftmaxBase):
    """
    Op for the cuDNN Softmax.

    Parameters
    ----------
    tensor_format
        Always set to 'bc01'.
    algo : {'fast', 'accurate'}
        Indicating whether computations should be
        optimized for speed or accuracy respectively.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    direction = 'forward'
    softmax_inputs = ['softmax_input']

    def make_node(self, x):
        x = as_cuda_ndarray_variable(x)
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def method(self):
        return """
#ifndef CUDNN_VERSION
err%(name)s = cudnnSoftmaxForward(
  _handle,
  algo%(name)s,
  mode%(name)s,
  softmax_input_%(name)s,
  CudaNdarray_DEV_DATA(%(ins)s),
  softmax_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outs)s)
);
#else
{
const float alpha = 1.;
const float beta = 0.;
err%(name)s = cudnnSoftmaxForward(
  _handle,
  algo%(name)s,
  mode%(name)s,
  (void*) &alpha,
  softmax_input_%(name)s,
  CudaNdarray_DEV_DATA(%(ins)s),
  (void*) &beta,
  softmax_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outs)s)
);
}
#endif
"""

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = self(x)
        return [GpuDnnSoftmaxGrad(
            self.tensor_format,
            self.algo,
            self.mode
        )(g_sm, sm)]


class GpuDnnSoftmaxGrad(GpuDnnSoftmaxBase):
    """
    Op for the cuDNN SoftmaxGrad.

    Parameters
    ----------
    tensor_format
        Always set to 'bc01'.
    algo : {'fast', 'accurate'}
        Indicating whether computations should be
        optimized for speed or accuracy respectively.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    direction = 'backward'
    softmax_inputs = ['softmax_gout', 'softmax_input']

    def make_node(self, dy, sm):
        dy = as_cuda_ndarray_variable(dy)
        sm = as_cuda_ndarray_variable(sm)
        assert dy.ndim == 4
        assert sm.ndim == 4
        return Apply(self, [dy, sm], [sm.type()])

    def method(self):
        return """
#ifndef CUDNN_VERSION
err%(name)s = cudnnSoftmaxBackward(
  _handle,
  algo%(name)s,
  mode%(name)s,
  %(name1)s_%(name)s,
  CudaNdarray_DEV_DATA(%(ins1)s),
  %(name0)s_%(name)s,
  CudaNdarray_DEV_DATA(%(ins0)s),
  softmax_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outs)s)
);
#else
{
const float alpha = 1.;
const float beta = 0.;
err%(name)s = cudnnSoftmaxBackward(
  _handle,
  algo%(name)s,
  mode%(name)s,
  (void*) &alpha,
  %(name1)s_%(name)s,
  CudaNdarray_DEV_DATA(%(ins1)s),
  %(name0)s_%(name)s,
  CudaNdarray_DEV_DATA(%(ins0)s),
  (void*) &beta,
  softmax_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outs)s)
);
}
#endif
        """


class GpuDnnBatchNormBase(DnnBase):
    """
    Base Op for cuDNN Batch Normalization.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation (in this mode, bias and scale
        tensor dimensions are 1xCxHxW) or share normalization factors across
        spatial dimensions (in this mode, bias and scale tensor dimensions
        are 1xCx1x1).
    epsilon
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - r_a_factor) + batch mean * r_a_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - r_a_factor) + (m / (m - 1)) * batch var * r_a_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None.
    """

    __props__ = ('mode', 'epsilon')
    tensor_descs = []

    def __init__(self, mode='per-activation', epsilon=1e-4):
        DnnBase.__init__(self)

        if version() < (5000, 5000):
            raise RuntimeError("cuDNN Batch Normalization requires cuDNN v5")

        assert (mode in ('per-activation', 'spatial'))
        self.mode = mode

        assert (epsilon >= 1e-5)
        self.epsilon = epsilon

    def c_support_code_struct(self, node, name):
        result = ''
        for id in self.tensor_descs:
            result += c_define_tensor_desc('%s_%s' % (id, name))
        return result

    def c_init_code_struct(self, node, name, sub):
        result = """
cudnnStatus_t err%(name)s;
""" % dict(name=name)

        for id in self.tensor_descs:
            result += c_init_tensor_desc('%s_%s' % (id, name), 'err' + name, sub['fail'])
        return result

    def c_cleanup_code_struct(self, node, name):
        result = ''
        for id in self.tensor_descs:
            result += c_clean_tensor_desc('%s_%s' % (id, name))
        return result

    def c_code(self, node, name, inputs, outputs, sub):
        if self.mode == "spatial":
            mode = "CUDNN_BATCHNORM_SPATIAL"
        else:
            mode = "CUDNN_BATCHNORM_PER_ACTIVATION"

        # Setup configuration variables.
        result = """
cudnnStatus_t err%(name)s;
cudnnBatchNormMode_t mode%(name)s = %(mode)s;
double epsilon%(name)s = %(epsilon)e;
""" % dict(name=name,
           mode=mode,
           epsilon=self.epsilon)

        return result

    def c_code_cache_version(self):
        return (4, version())


class GpuDnnBatchNormInference(GpuDnnBatchNormBase):
    """
    Op for the cuDNN BatchNormalizationForwardInference function.
    See GpuDnnBatchNormBase for parameters.

    On application, takes input, scale, bias, mean and variance and produces:
    output = (input - mean) / sqrt(variance + epsilon) * scale + bias

    where mean and variance are usually some running averages over multiple
    batches computed during training.

    Note: scale, bias, mean and variance must follow the same tensor layout!
    """

    __props__ = ('mode', 'epsilon', 'inplace')
    tensor_descs = ['bn_input', 'bn_output', 'bn_params']

    def __init__(self, mode='per-activation', epsilon=1e-4, inplace=False):
        super(GpuDnnBatchNormInference, self).__init__(mode=mode, epsilon=epsilon)
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'inplace'):
            self.inplace = False

    def get_op_params(self):
        params = []
        if self.inplace:
            params.append(('INPLACE_OUTPUT', '1'))
        return params

    def infer_shape(self, node, shape):
        # output shape equals shape of x
        return [shape[0]]

    def make_node(self, x, scale, bias, estimated_mean, estimated_variance):
        x = as_cuda_ndarray_variable(x)
        scale = as_cuda_ndarray_variable(scale)
        bias = as_cuda_ndarray_variable(bias)
        estimated_mean = as_cuda_ndarray_variable(estimated_mean)
        estimated_variance = as_cuda_ndarray_variable(estimated_variance)
        assert x.ndim == scale.ndim == bias.ndim == estimated_mean.ndim == estimated_variance.ndim
        assert x.ndim in (4, 5)
        return Apply(self, [x, scale, bias, estimated_mean, estimated_variance],
                     [x.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        # super call to prepare common configuration
        result = super(GpuDnnBatchNormInference, self).c_code(node, name, inputs, outputs, sub)

        # give sensible names to inputs and outputs
        inp, scale, bias, est_mean, est_var = inputs
        outp, = outputs

        # call cuDNN function
        result += """
// set input tensor descriptors from input tensors
if (c_set_tensorNd(%(inp)s, bn_input_%(name)s) != 0)
{
    %(fail)s
}
if (c_set_tensorNd(%(scale)s, bn_params_%(name)s) != 0)
{
    %(fail)s
}

// build and prepare the output variable
#ifdef INPLACE_OUTPUT
  Py_XDECREF(%(outp)s);
  %(outp)s = %(inp)s;
  Py_INCREF(%(outp)s);
#else
if (CudaNdarray_prep_output(&%(outp)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(inp)s)) != 0)
{
    %(fail)s
}
#endif

// set output tensor descriptor from output tensor
if (c_set_tensorNd(%(outp)s, bn_output_%(name)s) != 0)
{
    %(fail)s
}

{
const float alpha = 1.;
const float beta = 0.;
err%(name)s = cudnnBatchNormalizationForwardInference(
  _handle,
  mode%(name)s,
  (void*) &alpha,
  (void*) &beta,
  bn_input_%(name)s,
  CudaNdarray_DEV_DATA(%(inp)s),
  bn_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outp)s),
  bn_params_%(name)s,
  CudaNdarray_DEV_DATA(%(scale)s),
  CudaNdarray_DEV_DATA(%(bias)s),
  CudaNdarray_DEV_DATA(%(est_mean)s),
  CudaNdarray_DEV_DATA(%(est_var)s),
  epsilon%(name)s
);
}
""" % dict(name=name, inp=inp, scale=scale, bias=bias, est_mean=est_mean,
           est_var=est_var, outp=outp, fail=sub['fail'])

        # add params
        define_macros, undef_macros = self.get_c_macros(node, name, check_input=False)
        result = """
%(define_macros)s
{
    %(code)s
}
%(undef_macros)s
""" % dict(code=result, define_macros=define_macros, undef_macros=undef_macros)

        return result

    def grad(self, inputs, grads):
        x, scale, bias, est_mean, est_var = inputs
        dy = grads[0]

        # add necessary broadcasts
        if self.mode == 'per-activation':
            axes = (0,)
        elif self.mode == 'spatial':
            axes = (0,) + tuple(range(2, x.ndim))
        scale, bias, est_mean, est_var = (theano.tensor.addbroadcast(t, *axes)
                                          for t in (scale, bias, est_mean, est_var))

        # define helper expressions
        est_var_eps = est_var + self.epsilon
        est_std = theano.tensor.sqrt(est_var_eps)
        two = theano.tensor.constant(2.)

        # define and return gradients
        dx = dy * (scale / est_std)
        dscale = (dy * (x - est_mean)).sum(axes, keepdims=True) / est_std
        dbias = dy.sum(axes, keepdims=True)
        dmean = -dy.sum(axes, keepdims=True) * (scale / est_std)
        dvar = -(dy * (x - est_mean)).sum(axes, keepdims=True) * (scale / (two * est_var_eps * est_std))
        return [dx, dscale, dbias, dmean, dvar]


class GpuDnnBatchNorm(GpuDnnBatchNormBase):
    """
    Op for the cuDNN BatchNormalizationForwardTraining function.
    See GpuDnnBatchNormBase for parameters.

    On application, takes input, scale, bias and produces:
    output = (input - mean) / sqrt(variance + epsilon) * scale + bias
    mean = input.mean(axis=axes, keepdims=True),
    invstd = 1. / sqrt(input.var(axis=axes, keepdims=True) + epsilon)

    where axes=0 if mode='per-activation', and axes=(0,2,3) if mode='spatial'

    Note: scale and bias must follow the same tensor layout!
    """

    __props__ = ('mode', 'epsilon', 'running_average_factor',
                 'running_averages', 'inplace_running_mean',
                 'inplace_running_var', 'inplace_output')
    tensor_descs = ['bn_input', 'bn_output', 'bn_params']

    def __init__(self, mode='per-activation', epsilon=1e-4,
                 running_average_factor=0,
                 running_averages=False, inplace_running_mean=False,
                 inplace_running_var=False, inplace_output=False):
        super(GpuDnnBatchNorm, self).__init__(mode=mode, epsilon=epsilon)
        self.running_average_factor = running_average_factor
        self.running_averages = running_averages
        self.inplace_output = inplace_output
        self.inplace_running_mean = inplace_running_mean
        self.inplace_running_var = inplace_running_var
        self.destroy_map = {}
        if self.inplace_output:
            self.destroy_map[0] = [0]
        if self.running_averages and self.inplace_running_mean:
            self.destroy_map[3] = [3]
        if self.running_averages and self.inplace_running_var:
            self.destroy_map[4] = [4]

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'running_average_factor'):
            self.running_average_factor = 0
        if not hasattr(self, 'running_averages'):
            self.running_averages = False
        if not (hasattr(self, 'inplace_running_mean') and
                hasattr(self, 'inplace_running_var') and
                hasattr(self, 'inplace_output')):
            self.inplace_running_mean = False
            self.inplace_running_var = False
            self.inplace_output = False
            self.destroy_map = {}

    def get_op_params(self):
        params = []
        if self.inplace_output:
            params.append(('INPLACE_OUTPUT', '1'))
        if self.running_averages:
            params.append(('RUNNING_AVERAGES', '1'))
            if self.inplace_running_mean:
                params.append(('INPLACE_RUNNING_MEAN', '1'))
            if self.inplace_running_var:
                params.append(('INPLACE_RUNNING_VAR', '1'))
        return params

    def infer_shape(self, node, shape):
        # first output equals shape of x
        # other outputs equal shape of scale
        return [shape[0]] + [shape[1]] * (len(node.outputs) - 1)

    def make_node(self, x, scale, bias,
                  running_mean=None, running_var=None):
        assert x.ndim == scale.ndim == bias.ndim
        assert x.ndim in (4, 5)
        assert self.running_averages == (running_mean is not None) == (running_var is not None)
        assert (running_mean is None or running_mean.ndim == x.ndim)
        assert (running_var is None or running_var.ndim == x.ndim)
        x = as_cuda_ndarray_variable(x)
        scale = as_cuda_ndarray_variable(scale)
        bias = as_cuda_ndarray_variable(bias)
        inputs = [x, scale, bias]
        output_types = [x.type(), scale.type(), scale.type()]
        if running_mean is not None and running_var is not None:
            inputs.append(as_cuda_ndarray_variable(running_mean))
            inputs.append(as_cuda_ndarray_variable(running_var))
            output_types.append(scale.type())
            output_types.append(scale.type())
        return Apply(self, inputs, output_types)

    def c_code(self, node, name, inputs, outputs, sub):
        # super call to prepare common configuration
        result = super(GpuDnnBatchNorm, self).c_code(node, name, inputs, outputs, sub)

        # give sensible names to inputs and outputs
        inp, scale, bias = inputs[:3]
        outp, x_mean, x_invstd = outputs[:3]
        if self.running_averages:
            running_average_factor = self.running_average_factor
            in_running_mean = inputs[3]
            in_running_var = inputs[4]
            out_running_mean = outputs[3]
            out_running_var = outputs[4]
        else:
            running_average_factor = 0.
            in_running_mean = 'NULL'
            in_running_var = 'NULL'
            out_running_mean = 'NULL'
            out_running_var = 'NULL'

        # set input tensor descriptors from input tensors
        result += """
// set input tensor descriptors from input tensors
if (c_set_tensorNd(%(inp)s, bn_input_%(name)s) != 0)
{
    %(fail)s
}
if (c_set_tensorNd(%(scale)s, bn_params_%(name)s) != 0)
{
    %(fail)s
}

// build and prepare the output variables
if ((CudaNdarray_prep_output(&%(outp)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(inp)s)) != 0) ||
    (CudaNdarray_prep_output(&%(x_mean)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0) ||
    (CudaNdarray_prep_output(&%(x_invstd)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0))
{
    %(fail)s
}
#ifdef RUNNING_AVERAGES
#ifdef INPLACE_RUNNING_MEAN
  Py_XDECREF(%(out_running_mean)s);
  CudaNdarray *running_mean%(name)s = %(in_running_mean)s;
  Py_INCREF(running_mean%(name)s);
#else
  if ((CudaNdarray_prep_output(&%(out_running_mean)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0) ||
      (CudaNdarray_CopyFromCudaNdarray(%(out_running_mean)s, %(in_running_mean)s) != 0))
  {
    %(fail)s
  }
  CudaNdarray *running_mean%(name)s = %(out_running_mean)s;
#endif
#ifdef INPLACE_RUNNING_VAR
  Py_XDECREF(%(out_running_var)s);
  CudaNdarray *running_var%(name)s = %(in_running_var)s;
  Py_INCREF(running_var%(name)s);
#else
  if ((CudaNdarray_prep_output(&%(out_running_var)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0) ||
      (CudaNdarray_CopyFromCudaNdarray(%(out_running_var)s, %(in_running_var)s) != 0))
  {
    %(fail)s
  }
  CudaNdarray *running_var%(name)s = %(out_running_var)s;
#endif
#endif

// set output tensor descriptor from output tensor
if (c_set_tensorNd(%(outp)s, bn_output_%(name)s) != 0)
{
    %(fail)s
}

{
const float alpha = 1.;
const float beta = 0.;
err%(name)s = cudnnBatchNormalizationForwardTraining(
  _handle,
  mode%(name)s,
  (void*) &alpha,
  (void*) &beta,
  bn_input_%(name)s,
  CudaNdarray_DEV_DATA(%(inp)s),
  bn_output_%(name)s,
  CudaNdarray_DEV_DATA(%(outp)s),
  bn_params_%(name)s,
  CudaNdarray_DEV_DATA(%(scale)s),
  CudaNdarray_DEV_DATA(%(bias)s),
#ifdef RUNNING_AVERAGES
  %(running_average_factor)f,
  CudaNdarray_DEV_DATA(running_mean%(name)s),
  CudaNdarray_DEV_DATA(running_var%(name)s),
#else
  0,
  NULL,
  NULL,
#endif
  epsilon%(name)s,
  CudaNdarray_DEV_DATA(%(x_mean)s),
  CudaNdarray_DEV_DATA(%(x_invstd)s)
);
}
#ifdef RUNNING_AVERAGES
  %(out_running_mean)s = running_mean%(name)s;
  %(out_running_var)s = running_var%(name)s;
#endif
""" % dict(name=name, inp=inp, scale=scale, bias=bias, outp=outp,
           x_mean=x_mean, x_invstd=x_invstd,
           running_average_factor=running_average_factor,
           in_running_mean=in_running_mean, in_running_var=in_running_var,
           out_running_mean=out_running_mean, out_running_var=out_running_var,
           fail=sub['fail'])

        # add params
        define_macros, undef_macros = self.get_c_macros(node, name, check_input=False)
        result = """
%(define_macros)s
{
    %(code)s
}
%(undef_macros)s
""" % dict(code=result, define_macros=define_macros, undef_macros=undef_macros)

        return result

    def grad(self, inputs, grads):
        x, scale, bias = inputs[:3]
        dy = grads[0]
        _, x_mean, x_invstd = self(*inputs)[:3]
        disconnected_outputs = []
        # Optional running_mean and running_var.
        for i in range(3, len(inputs)):
            disconnected_outputs.append(DisconnectedType()())
        return GpuDnnBatchNormGrad(self.mode, self.epsilon)(
            x, dy, scale, x_mean, x_invstd) + disconnected_outputs

    def connection_pattern(self, node):
        patterns = [[True, True, True],     # x
                    [True, True, True],     # scale
                    [True, True, True]]     # bias
        # Optional running_mean and running_var are only
        # connected to their new values.
        for i in range(3, len(node.inputs)):
            patterns[0].append(True)
            for pattern in patterns[1:]:
                pattern.append(False)
            patterns.append([False] * (i) + [True])
        return patterns


class GpuDnnBatchNormGrad(GpuDnnBatchNormBase):
    """
    Op for the cuDNN BatchNormalizationBackward function.
    See GpuDnnBatchNormBase for parameters.

    On application, takes input, dy, scale, mean, invstd and produces
    dinput, dscale and dbias. Note that it does not need the bias.

    Note: scale, mean and invstd must follow the same tensor layout!
    """

    tensor_descs = ['bn_input', 'bn_doutput', 'bn_dinput', 'bn_params']

    def infer_shape(self, node, shape):
        # first output equals shape of x
        # second and third output equal shape of scale
        return [shape[0], shape[2], shape[2]]

    def make_node(self, x, dy, scale, x_mean, x_invstd):
        x = as_cuda_ndarray_variable(x)
        dy = as_cuda_ndarray_variable(dy)
        scale = as_cuda_ndarray_variable(scale)
        x_mean = as_cuda_ndarray_variable(x_mean)
        x_invstd = as_cuda_ndarray_variable(x_invstd)
        assert x.ndim == dy.ndim == scale.ndim == x_mean.ndim == x_invstd.ndim
        assert x.ndim in (4, 5)
        return Apply(self, [x, dy, scale, x_mean, x_invstd], [x.type(), scale.type(), scale.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        # super call to prepare common configuration
        result = super(GpuDnnBatchNormGrad, self).c_code(node, name, inputs, outputs, sub)

        # give sensible names to inputs and outputs
        inp, doutp, scale, x_mean, x_invstd = inputs
        dinp, dscale, dbias = outputs

        # call cuDNN function
        result += """
// set input tensor descriptors from input tensors
if (c_set_tensorNd(%(inp)s, bn_input_%(name)s) != 0)
{
    %(fail)s
}
if (c_set_tensorNd(%(doutp)s, bn_doutput_%(name)s) != 0)
{
    %(fail)s
}
if (c_set_tensorNd(%(scale)s, bn_params_%(name)s) != 0)
{
    %(fail)s
}

// build and prepare the output variables
if ((CudaNdarray_prep_output(&%(dinp)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(inp)s)) != 0) ||
    (CudaNdarray_prep_output(&%(dscale)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0) ||
    (CudaNdarray_prep_output(&%(dbias)s, %(inp)s->nd, CudaNdarray_HOST_DIMS(%(scale)s)) != 0))
{
    %(fail)s
}

// set output tensor descriptor from output tensor
if (c_set_tensorNd(%(dinp)s, bn_dinput_%(name)s) != 0)
{
    %(fail)s
}

{
const float alphaData = 1.;
const float betaData = 0.;
const float alphaParam = 1.;
const float betaParam = 0.;
err%(name)s = cudnnBatchNormalizationBackward(
  _handle,
  mode%(name)s,
  (void*) &alphaData,
  (void*) &betaData,
  (void*) &alphaParam,
  (void*) &betaParam,
  bn_input_%(name)s,
  CudaNdarray_DEV_DATA(%(inp)s),
  bn_doutput_%(name)s,
  CudaNdarray_DEV_DATA(%(doutp)s),
  bn_dinput_%(name)s,
  CudaNdarray_DEV_DATA(%(dinp)s),
  bn_params_%(name)s,
  CudaNdarray_DEV_DATA(%(scale)s),
  CudaNdarray_DEV_DATA(%(dscale)s),
  CudaNdarray_DEV_DATA(%(dbias)s),
  epsilon%(name)s,
  CudaNdarray_DEV_DATA(%(x_mean)s),
  CudaNdarray_DEV_DATA(%(x_invstd)s)
);
}
""" % dict(name=name, inp=inp, doutp=doutp, scale=scale, x_mean=x_mean,
           x_invstd=x_invstd, dinp=dinp, dscale=dscale, dbias=dbias, fail=sub['fail'])

        return result


def dnn_batch_normalization_train(inputs, gamma, beta, mode='per-activation',
                                  epsilon=1e-4, running_average_factor=0.1,
                                  running_mean=None, running_var=None):
    """
    Performs batch normalization of the given inputs, using the mean and
    variance of the inputs.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation or share normalization factors
        across spatial dimensions (i.e., all dimensions past the second).
    gamma : tensor
        Learnable scale factors. Must match the dimensionality of `inputs`,
        but have sizes of `1` for all axes normalized over (i.e., in the first
        dimension for ``mode='per-activation'`, and additionally in all
        dimensions past the second for ``mode='spatial'``).
    beta : tensor
        Learnable biases. Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - r_a_factor) + batch mean * r_a_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - r_a_factor) + (m / (m - 1)) * batch var * r_a_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None.

    Returns
    -------
    out : tensor
        Batch-normalized inputs.
    mean : tensor
        Means of `inputs` across the normalization axes.
    invstd : tensor
        Inverse standard deviations of `inputs` across the normalization axes.
    new_running_mean : tensor
        New value of the running mean (only if both `running_mean` and
        `running_var` were given).
    new_running_var : tensor
        New value of the running variance (only if both `running_var` and
        `running_mean` were given).

    Notes
    -----
    Request cuDNN 5 and Theano 0.9dev2 or more recent.

    For 4d tensors, returned values are equivalent to:

    .. code-block:: python

        axes = 0 if mode == 'per-activation' else (0, 2, 3)
        mean = inputs.mean(axes, keepdims=True)
        var = inputs.var(axes, keepdims=True)
        invstd = T.inv(T.sqrt(var + epsilon))
        out = (inputs - mean) * gamma * invstd + beta

        m = T.cast(T.prod(inputs.shape) / T.prod(mean.shape), 'float32')
        running_mean = running_mean * (1 - running_average_factor) + \\
                       mean * running_average_factor
        running_var = running_var * (1 - running_average_factor) + \\
                      (m / (m - 1)) * var * running_average_factor

    For 5d tensors, the axes are (0, 2, 3, 4).
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError("gamma and beta must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, ndim))
    if (running_mean is None) != (running_var is None):
        raise ValueError("running_mean and running_var must either both be "
                         "given or both be None")
    if running_mean is not None and running_mean.ndim != ndim:
        raise ValueError("running_mean must be of the same dimensionality "
                         "as inputs; got %d instead of %d" %
                         (running_mean.ndim, ndim))
    if running_var is not None and running_var.ndim != ndim:
        raise ValueError("running_var must be of the same dimensionality "
                         "as inputs; got %d instead of %d" %
                         (running_var.ndim, ndim))
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %f" % epsilon)

    running_averages = (running_var is not None and running_var is not None)

    if ndim < 4:
        inputs = theano.tensor.shape_padright(inputs, 4 - ndim)
        gamma = theano.tensor.shape_padright(gamma, 4 - ndim)
        beta = theano.tensor.shape_padright(beta, 4 - ndim)
        if running_averages:
            running_mean = theano.tensor.shape_padright(running_mean, 4 - ndim)
            running_var = theano.tensor.shape_padright(running_var, 4 - ndim)
    elif ndim > 5:
        inputs_shape = inputs.shape
        params_shape = gamma.shape
        inputs = theano.tensor.flatten(inputs, 5)
        gamma = theano.tensor.flatten(gamma, 5)
        beta = theano.tensor.flatten(beta, 5)
        if running_averages:
            running_mean = theano.tensor.flatten(running_mean, 5)
            running_var = theano.tensor.flatten(running_var, 5)

    batchnorm_op = GpuDnnBatchNorm(mode=mode, epsilon=epsilon,
                                   running_average_factor=running_average_factor,
                                   running_averages=running_averages)
    if running_averages:
        out, mean, invstd, new_running_mean, new_running_var = batchnorm_op(
            gpu_contiguous(inputs), gpu_contiguous(gamma),
            gpu_contiguous(beta),
            running_mean=gpu_contiguous(running_mean),
            running_var=gpu_contiguous(running_var))
        if new_running_mean.broadcastable != running_mean.broadcastable:
            new_running_mean = tensor.patternbroadcast(new_running_mean, running_mean.broadcastable)
        if new_running_var.broadcastable != running_var.broadcastable:
            new_running_var = tensor.patternbroadcast(new_running_var, running_var.broadcastable)
        result = (out, mean, invstd, new_running_mean, new_running_var)
    else:
        result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
                              gpu_contiguous(beta))
    if ndim < 4:
        result = tuple(theano.tensor.flatten(r, ndim) for r in result)
    elif ndim > 5:
        result = (theano.tensor.reshape(result[0], inputs_shape),) + tuple(
            theano.tensor.reshape(r, params_shape) for r in result[1:])
    return result


def dnn_batch_normalization_test(inputs, gamma, beta, mean, var,
                                 mode='per-activation', epsilon=1e-4):
    """
    Performs batch normalization of the given inputs, using the given mean and
    variance.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation or share normalization factors
        across spatial dimensions (i.e., all dimensions past the second).
    gamma : tensor
        Scale factors. Must match the dimensionality of `inputs`, but have
        sizes of `1` for all axes normalized over (i.e., in the first dimension
        for ``mode='per-activation'`, and additionally in all dimensions past
        the second for ``mode='spatial'``).
    beta : tensor
        Biases. Must match the tensor layout of `gamma`.
    mean : tensor
        Means. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    var : tensor
        Variances. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).

    Returns
    -------
    out : tensor
        Batch-normalized inputs.

    Notes
    -----
    Request cuDNN 5 and Theano 0.9dev2 or more recent.

    For 4d tensors, the returned value is equivalent to:

    .. code-block:: python

        axes = (0,) if mode == 'per-activation' else (0, 2, 3)
        gamma, beta, mean, var = (T.addbroadcast(t, *axes)
                                  for t in (gamma, beta, mean, var))
        out = (inputs - mean) * gamma / T.sqrt(var + epsilon) + beta

    For 5d tensors, the axes would be (0, 2, 3, 4).
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError("gamma and beta must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, ndim))
    if mean.ndim != ndim or var.ndim != ndim:
        raise ValueError("mean and var must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (mean.ndim, var.ndim, ndim))
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %f" % epsilon)

    if ndim < 4:
        inputs = theano.tensor.shape_padright(inputs, 4 - ndim)
        gamma = theano.tensor.shape_padright(gamma, 4 - ndim)
        beta = theano.tensor.shape_padright(beta, 4 - ndim)
        mean = theano.tensor.shape_padright(mean, 4 - ndim)
        var = theano.tensor.shape_padright(var, 4 - ndim)
    elif ndim > 5:
        inputs_shape = inputs.shape
        inputs = theano.tensor.flatten(inputs, 5)
        gamma = theano.tensor.flatten(gamma, 5)
        beta = theano.tensor.flatten(beta, 5)
        mean = theano.tensor.flatten(mean, 5)
        var = theano.tensor.flatten(var, 5)
    batchnorm_op = GpuDnnBatchNormInference(mode=mode, epsilon=epsilon)
    result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
                          gpu_contiguous(beta), gpu_contiguous(mean),
                          gpu_contiguous(var))
    if ndim < 4:
        result = theano.tensor.flatten(result, ndim)
    elif ndim > 5:
        result = theano.tensor.reshape(result, inputs_shape)
    return result


# Intentation for history
if True:
    # @register_opt('cudnn')  # this optimizer is registered in opt.py instead.
    @local_optimizer([GpuConv])
    def local_conv_dnn(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuConv):
            if node.op.border_mode not in ['full', 'valid']:
                return
            img, kern = node.inputs
            border_mode = node.op.border_mode
            subsample = node.op.subsample
            direction_hint = node.op.direction_hint
            rval = dnn_conv(img, kern,
                            border_mode=border_mode, subsample=subsample,
                            direction_hint=direction_hint)
            if node.outputs[0].broadcastable != rval.broadcastable:
                rval = tensor.patternbroadcast(
                    rval, node.outputs[0].type.broadcastable)
            return [rval]

    # This optimizer is registered in opt.py as part of the meta-optimizer.
    # It tries exactly the opposite code path of what local_conv_dnn() uses,
    # because for some input/kernel shape configurations, this is faster.
    @local_optimizer([GpuConv])
    def local_conv_dnn_alternative(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuConv):
            border_mode = node.op.border_mode
            subsample = node.op.subsample
            if border_mode not in ['full', 'valid'] or subsample != (1, 1):
                return
            img, kern = node.inputs
            direction_hint = node.op.direction_hint
            if border_mode == 'full':
                # for a full convolution, try using the forward pass instead
                # of the backward pass wrt. inputs and vice versa
                if direction_hint == 'bprop inputs':
                    direction_hint = 'forward'
                else:
                    direction_hint = 'bprop inputs'
            elif border_mode == 'valid':
                # for a valid convolution, try using the backward pass wrt.
                # weights instead of the forward pass and vice versa
                if direction_hint == 'bprop weights':
                    direction_hint = 'forward'
                else:
                    direction_hint = 'bprop weights'
            rval = dnn_conv(img, kern,
                            border_mode=border_mode, subsample=subsample,
                            direction_hint=direction_hint)
            if node.outputs[0].broadcastable != rval.broadcastable:
                rval = tensor.patternbroadcast(
                    rval, node.outputs[0].type.broadcastable)
            return [rval]

    @local_optimizer([GpuDnnConv], inplace=True)
    def local_dnn_conv_inplace(node):
        if type(node.op) != GpuDnnConv or node.op.inplace:
            return
        inputs = list(node.inputs)
        dest = inputs[2]
        if (dest.owner and
                type(dest.owner.op) is GpuAllocEmpty and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc_empty(*dest.owner.inputs)
        elif (dest.owner and
                type(dest.owner.op) is GpuAlloc and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc(*dest.owner.inputs)
        return [GpuDnnConv(algo=node.op.algo, inplace=True)(*inputs)]

    @local_optimizer([GpuDnnConvGradW], inplace=True)
    def local_dnn_convgw_inplace(node):
        if type(node.op) != GpuDnnConvGradW or node.op.inplace:
            return
        inputs = list(node.inputs)
        dest = inputs[2]
        if (dest.owner and
                type(dest.owner.op) is GpuAllocEmpty and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc_empty(*dest.owner.inputs)
        elif (dest.owner and
                type(dest.owner.op) is GpuAlloc and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc(*dest.owner.inputs)
        return [GpuDnnConvGradW(inplace=True)(*inputs)]

    @local_optimizer([GpuDnnConvGradI], inplace=True)
    def local_dnn_convgi_inplace(node):
        if type(node.op) != GpuDnnConvGradI or node.op.inplace:
            return
        inputs = list(node.inputs)
        dest = inputs[2]
        if (dest.owner and
                type(dest.owner.op) is GpuAllocEmpty and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc_empty(*dest.owner.inputs)
        elif (dest.owner and
                type(dest.owner.op) is GpuAlloc and
                len(dest.clients) > 1):
            inputs[2] = gpu_alloc(*dest.owner.inputs)
        return [GpuDnnConvGradI(inplace=True)(*inputs)]

    optdb.register('local_dnn_conv_inplace',
                   tensor.opt.in2out(local_dnn_conv_inplace,
                                     local_dnn_convgw_inplace,
                                     local_dnn_convgi_inplace,
                                     name="local_dnn_conv_inplace"),
                   70.0, 'fast_run', 'inplace', 'gpu', 'cudnn')

    @register_opt('cudnn')
    @alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5)
    def local_dnn_conv_alpha_merge(node, *inputs):
        if not dnn_available() or version() == -1:
            return None
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5)
    def local_dnn_convw_alpha_merge(node, *inputs):
        if not dnn_available() or version() == -1:
            return None
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5)
    def local_dnn_convi_alpha_merge(node, *inputs):
        if not dnn_available() or version() == -1:
            return None
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2)
    def local_dnn_conv_output_merge(node, *inputs):
        inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2)
    def local_dnn_convw_output_merge(node, *inputs):
        inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2)
    def local_dnn_convi_output_merge(node, *inputs):
        inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
        return [node.op(*inputs)]

    @register_opt('cudnn')
    @local_optimizer([GpuDownsampleFactorMax])
    def local_pool_dnn(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuDownsampleFactorMax):
            if not node.op.ignore_border:
                return
            img, = node.inputs
            ds = node.op.ds
            return [dnn_pool(gpu_contiguous(img), ds, ds)]

    @register_opt('cudnn')
    @local_optimizer([Pool])
    def local_pool_dnn_alternative(node):
        if not dnn_available():
            return
        if isinstance(node.op, Pool):
            if not node.op.ignore_border:
                return
            img, ws, stride, pad = node.inputs
            nd = node.op.ndim
            mode = node.op.mode
            if nd not in (2, 3):
                return
            if (img.owner and isinstance(img.owner.op, HostFromGpu)):
                # dnn_pool expects exactly 2 non-pooling dimensions
                if img.ndim == nd + 2:
                    ret = dnn_pool(gpu_contiguous(img.owner.inputs[0]),
                                   ws, stride=stride, pad=pad, mode=mode)
                else:
                    input = gpu_contiguous(img.owner.inputs[0])
                    # reshape to 4D or 5D with 2 non-pooling dimensions
                    input_padded = pad_dims(input, 2, nd)
                    ret_padded = dnn_pool(input_padded,
                                          ws, stride=stride, pad=pad, mode=mode)
                    ret = unpad_dims(ret_padded, input, 2, nd)
                return [host_from_gpu(ret)]

    @register_opt('cudnn')
    @local_optimizer([GpuDownsampleFactorMaxGrad])
    def local_pool_dnn_grad(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuDownsampleFactorMaxGrad):
            if not node.op.ignore_border:
                return
            inp, out, inp_grad = node.inputs
            ds = node.op.ds

            return [GpuDnnPoolGrad(mode='max')(gpu_contiguous(inp),
                                               gpu_contiguous(out),
                                               gpu_contiguous(inp_grad),
                                               ds, ds, (0, 0))]

    @register_opt('cudnn')
    @local_optimizer([MaxPoolGrad])
    def local_pool_dnn_grad_stride(node):
        if not dnn_available():
            return
        if isinstance(node.op, MaxPoolGrad):
            if not node.op.ignore_border:
                return
            inp, out, inp_grad, ws, stride, pad = node.inputs
            nd = node.op.ndim
            mode = node.op.mode
            if nd not in (2, 3):
                return

            if ((inp.owner and isinstance(inp.owner.op, HostFromGpu)) or
                (out.owner and isinstance(out.owner.op, HostFromGpu)) or
                (inp_grad.owner and isinstance(inp_grad.owner.op,
                                               HostFromGpu))):
                # the GPU ops expect exactly 2 non-pooling dimensions
                if inp.ndim == nd + 2:
                    ret = GpuDnnPoolGrad(mode=mode)(gpu_contiguous(inp),
                                                    gpu_contiguous(out),
                                                    gpu_contiguous(inp_grad),
                                                    ws, stride, pad)
                else:
                    # reshape to 4D or 5D with 2 non-pooling dimensions
                    inp_padded = pad_dims(gpu_contiguous(inp), 2, nd)
                    out_padded = pad_dims(gpu_contiguous(out), 2, nd)
                    inp_grad_padded = pad_dims(gpu_contiguous(inp_grad), 2, nd)
                    ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
                                                           out_padded,
                                                           inp_grad_padded,
                                                           ws, stride, pad)
                    ret = unpad_dims(ret_padded, inp, 2, nd)
                return [host_from_gpu(ret)]

    @register_opt('cudnn')
    @local_optimizer([AveragePoolGrad])
    def local_avgpool_dnn_grad_stride(node):
        if not dnn_available():
            return
        if isinstance(node.op, AveragePoolGrad):
            if not node.op.ignore_border:
                return
            inp, inp_grad, ws, stride, pad = node.inputs
            nd = node.op.ndim
            mode = node.op.mode
            if nd not in (2, 3):
                return

            if ((inp.owner and isinstance(inp.owner.op, HostFromGpu)) or
                (inp_grad.owner and isinstance(inp_grad.owner.op,
                                               HostFromGpu))):
                # the GPU ops expect exactly 2 non-pooling dimensions
                if inp.ndim == nd + 2:
                    contiguous_inp_grad = gpu_contiguous(inp_grad)
                    ret = GpuDnnPoolGrad(mode=mode)(gpu_contiguous(inp),
                                                    contiguous_inp_grad,
                                                    contiguous_inp_grad,
                                                    ws, stride, pad)
                else:
                    inp_padded = pad_dims(gpu_contiguous(inp), 2, nd)
                    inp_grad_padded = pad_dims(gpu_contiguous(inp_grad), 2, nd)
                    ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
                                                           inp_grad_padded,
                                                           inp_grad_padded,
                                                           ws, stride, pad)
                    ret = unpad_dims(ret_padded, inp, 2, nd)
                return [host_from_gpu(ret)]

    @register_opt('cudnn')
    @local_optimizer([GpuSoftmax])
    def local_softmax_dnn(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuSoftmax):
            ins = node.inputs[0].dimshuffle(0, 1, 'x', 'x')
            ins = gpu_contiguous(ins)
            out = GpuDnnSoftmax('bc01', 'accurate', 'channel')(ins)
            out = as_cuda_ndarray_variable(out.dimshuffle(0, 1))
            return [out]

    @register_opt('cudnn', 'stabilize', 'fast_compile')
    # We put fast_compile as otherwise it won't be on the GPU.
    @local_optimizer([GpuElemwise, LogSoftmax])
    def local_log_softmax_dnn(node):
        # The log-softmax implementation is only available starting at cuDNN V3
        if not dnn_available():
            return

        if (isinstance(node.op, GpuElemwise) and
                isinstance(node.op.scalar_op, Log) and
                node.inputs[0].owner and
                isinstance(node.inputs[0].owner.op, GpuDnnSoftmax) and
                len(node.inputs[0].owner.out.clients) == 1):

            log_input = node.inputs[0]
            softmax_node = log_input.owner

            new_softmax_node = GpuDnnSoftmax(softmax_node.op.tensor_format,
                                             'log', softmax_node.op.mode)
            new_log_softmax = new_softmax_node(softmax_node.inputs[0])
            return [new_log_softmax]

        elif (isinstance(node.op, LogSoftmax) and node.inputs[0].owner and
              isinstance(node.inputs[0].owner.op, HostFromGpu)):
            if not dnn_available():
                return

            # Transform the input in the format expected by GpuDnnSoftmax
            inp = node.inputs[0].owner.inputs[0]
            if inp.ndim != 2:
                return
            inp = inp.dimshuffle(0, 1, 'x', 'x')

            # Apply GpuDnnSoftmax and return the result
            out = GpuDnnSoftmax('bc01', 'log', 'channel')(gpu_contiguous(inp))
            return [out.dimshuffle(0, 1)]

    class NoCuDNNRaise(Optimizer):
        def apply(self, fgraph):
            """ Raise a RuntimeError if cudnn can't be used"""
            if not dnn_available():
                # Make an assert error as we want Theano to fail, not
                # just skip this optimization.
                raise AssertionError(
                    "cuDNN optimization was enabled, but Theano was not able"
                    " to use it. We got this error: \n" +
                    dnn_available.msg)
    gpu_seqopt.register("NoCuDNNRaise", NoCuDNNRaise(), 0, 'cudnn')

    @register_opt('cudnn')
    @local_optimizer([SoftmaxGrad])
    def local_softmax_dnn_grad(node):
        if (isinstance(node.op, SoftmaxGrad) and
            ((node.inputs[0].owner and
              isinstance(node.inputs[0].owner.op, HostFromGpu)) or
             (node.inputs[1].owner and
                 isinstance(node.inputs[1].owner.op, HostFromGpu)))):
            if not dnn_available():
                return
            ins = []
            for n in node.inputs:
                if n.owner is not None:
                    if isinstance(n.owner.op, HostFromGpu):
                        n = n.owner.inputs[0]
                if n.ndim != 2:
                    return
                ins.append(n.dimshuffle(0, 'x', 1, 'x'))

            out = GpuDnnSoftmaxGrad(
                'bc01',
                'accurate',
                'instance',
            )(
                gpu_contiguous(ins[0]),
                gpu_contiguous(ins[1])
            )
            return [out.dimshuffle(0, 2)]


# AbstractConv Optimizations
@local_optimizer([AbstractConv2d, AbstractConv2d_gradWeights,
                  AbstractConv2d_gradInputs])
def local_abstractconv_cudnn(node):
    if (not isinstance(node.op, (AbstractConv2d,
                                 AbstractConv2d_gradWeights,
                                 AbstractConv2d_gradInputs))):
        return None
    if (node.op.filter_dilation != (1, 1)):
        return None

    inp1 = node.inputs[0]
    inp2 = node.inputs[1]

    if (not isinstance(inp1.type, CudaNdarrayType) or
            not isinstance(inp2.type, CudaNdarrayType)):
        return None

    if not dnn_available():
        return None

    if node.op.filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'
    if (isinstance(node.op, AbstractConv2d)):
        rval = dnn_conv(inp1, inp2,
                        border_mode=node.op.border_mode,
                        subsample=node.op.subsample,
                        direction_hint='forward',
                        conv_mode=conv_mode)
        return [rval]
    if (isinstance(node.op, AbstractConv2d_gradWeights)):
        shape = (inp2.shape[1], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1])
        rval = dnn_gradweight(inp1, inp2, shape,
                              border_mode=node.op.border_mode,
                              subsample=node.op.subsample,
                              conv_mode=conv_mode)
        return [rval]
    if (isinstance(node.op, AbstractConv2d_gradInputs)):
        shape = (inp2.shape[0], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1])
        rval = dnn_gradinput(inp1, inp2, shape,
                             border_mode=node.op.border_mode,
                             subsample=node.op.subsample,
                             conv_mode=conv_mode)
        return [rval]


@local_optimizer([AbstractConv3d,
                  AbstractConv3d_gradWeights,
                  AbstractConv3d_gradInputs])
def local_abstractconv3d_cudnn(node):
    if (not isinstance(node.op, (AbstractConv3d,
                                 AbstractConv3d_gradWeights,
                                 AbstractConv3d_gradInputs))):
        return None
    if (node.op.filter_dilation != (1, 1, 1)):
        return None

    inp1 = node.inputs[0]
    inp2 = node.inputs[1]

    if (not isinstance(inp1.type, CudaNdarrayType) or
            not isinstance(inp2.type, CudaNdarrayType)):
        return None

    if not dnn_available():
        return None

    if node.op.filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'
    if (isinstance(node.op, AbstractConv3d)):
        rval = dnn_conv3d(inp1, inp2,
                          border_mode=node.op.border_mode,
                          subsample=node.op.subsample,
                          direction_hint='forward',
                          conv_mode=conv_mode)
        return [rval]
    if (isinstance(node.op, AbstractConv3d_gradWeights)):
        shape = (inp2.shape[1], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1], node.inputs[2][2])
        rval = dnn_gradweight3d(inp1, inp2, shape,
                                border_mode=node.op.border_mode,
                                subsample=node.op.subsample,
                                conv_mode=conv_mode)
        return [rval]
    if (isinstance(node.op, AbstractConv3d_gradInputs)):
        shape = (inp2.shape[0], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1], node.inputs[2][2])
        rval = dnn_gradinput3d(inp1, inp2, shape,
                               border_mode=node.op.border_mode,
                               subsample=node.op.subsample,
                               conv_mode=conv_mode)
        return [rval]


@local_optimizer([bn.AbstractBatchNormTrain])
def local_abstract_batch_norm_train_cudnn(node):
    if not isinstance(node.op, bn.AbstractBatchNormTrain):
        return None

    x, scale, bias, epsilon, running_average_factor = node.inputs[:5]
    running_mean = node.inputs[5] if len(node.inputs) > 5 else None
    running_var = node.inputs[6] if len(node.inputs) > 6 else None

    # input on gpu?  TODO what about the output?
    x_on_gpu = (isinstance(x.type, CudaNdarrayType) or
                (x.owner and isinstance(x.owner.op, HostFromGpu)))
    if not x_on_gpu:
        return None

    # convert axes to cuDNN mode
    axes = tuple(node.op.axes)
    if axes == (0,):
        mode = 'per-activation'
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = 'spatial'
    else:
        return None

    try:
        eps = float(theano.tensor.get_scalar_constant_value(epsilon))
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None
    try:
        running_average_factor = float(theano.tensor.get_scalar_constant_value(running_average_factor))
    except theano.tensor.NotScalarConstantError:
        return None

    if not dnn_available():
        return None

    x = as_cuda_ndarray_variable(x)
    scale = as_cuda_ndarray_variable(scale)
    bias = as_cuda_ndarray_variable(bias)

    inputs = [x, scale, bias, mode, eps, running_average_factor]
    if running_mean is not None and running_var is not None:
        inputs.append(running_mean)
        inputs.append(running_var)

    results = list(dnn_batch_normalization_train(*inputs))

    # If the original output was on CPU, we have to transfer it
    for i in range(len(node.outputs)):
        if isinstance(node.outputs[i].type, tensor.TensorType):
            results[i] = tensor.as_tensor_variable(results[i])
    # TODO copy_stack_trace?
    return results


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_gpu_batch_norm_inplace_output(node):
    if isinstance(node.op, GpuDnnBatchNorm) and not node.op.inplace_output:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               epsilon=node.op.epsilon,
                               running_average_factor=node.op.running_average_factor,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=node.op.inplace_running_mean,
                               inplace_running_var=node.op.inplace_running_var,
                               inplace_output=True)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_gpu_batch_norm_inplace_running_mean(node):
    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_mean:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               epsilon=node.op.epsilon,
                               running_average_factor=node.op.running_average_factor,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=True,
                               inplace_running_var=node.op.inplace_running_var,
                               inplace_output=node.op.inplace_output)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_gpu_batch_norm_inplace_running_var(node):
    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_var:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               epsilon=node.op.epsilon,
                               running_average_factor=node.op.running_average_factor,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=node.op.inplace_running_mean,
                               inplace_running_var=True,
                               inplace_output=node.op.inplace_output)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNormInference], inplace=True)
def local_gpu_batch_norm_inference_inplace(node):
    if isinstance(node.op, GpuDnnBatchNormInference) and not node.op.inplace:
        return [GpuDnnBatchNormInference(mode=node.op.mode,
                                         epsilon=node.op.epsilon,
                                         inplace=True)(*node.inputs)]


def values_eq_approx_high_tol(a, b):
    """
    This fct is needed to don't have DebugMode raise useless
    errors due to rounding error.

    This happen as we reduce on the two last dimensions, so this
    can raise the absolute error if the number of elements we
    reduce on is significant.

    """
    return tensor.TensorType.values_eq_approx(a, b, atol=0.015)


@local_optimizer([bn.AbstractBatchNormTrainGrad])
def local_abstract_batch_norm_train_grad_cudnn(node):
    if not isinstance(node.op, bn.AbstractBatchNormTrainGrad):
        return None

    x, dy, scale, x_mean, x_invstd, epsilon = node.inputs

    # input on gpu?  TODO what about the output?
    x_on_gpu = (isinstance(x.type, CudaNdarrayType) or
                (x.owner and isinstance(x.owner.op, HostFromGpu)))
    dy_on_gpu = (isinstance(dy.type, CudaNdarrayType) or
                 (dy.owner and isinstance(dy.owner.op, HostFromGpu)))
    if not (x_on_gpu or dy_on_gpu):
        return None

    # convert axes to cuDNN mode
    axes = tuple(node.op.axes)
    if axes == (0,):
        mode = 'per-activation'
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = 'spatial'
    else:
        return None

    ndim = x.ndim
    if ndim < 4:
        x = theano.tensor.shape_padright(x, 4 - ndim)
        dy = theano.tensor.shape_padright(dy, 4 - ndim)
        scale = theano.tensor.shape_padright(scale, 4 - ndim)
        x_mean = theano.tensor.shape_padright(x_mean, 4 - ndim)
        x_invstd = theano.tensor.shape_padright(x_invstd, 4 - ndim)
    elif ndim > 5:
        x_shape = x.shape
        params_shape = scale.shape
        x = theano.tensor.flatten(x, 5)
        dy = theano.tensor.flatten(dy, 5)
        scale = theano.tensor.flatten(scale, 5)
        x_mean = theano.tensor.flatten(x_mean, 5)
        x_invstd = theano.tensor.flatten(x_invstd, 5)

    try:
        eps = float(theano.tensor.get_scalar_constant_value(epsilon))
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    if not dnn_available():
        return None

    x = as_cuda_ndarray_variable(x)
    dy = as_cuda_ndarray_variable(dy)
    scale = as_cuda_ndarray_variable(scale)
    x_mean = as_cuda_ndarray_variable(x_mean)
    x_invstd = as_cuda_ndarray_variable(x_invstd)

    g_wrt_inputs, g_wrt_scale, g_wrt_bias = \
        GpuDnnBatchNormGrad(mode, epsilon=eps)(x, dy, scale, x_mean, x_invstd)

    if ndim < 4:
        g_wrt_inputs = theano.tensor.flatten(g_wrt_inputs, ndim)
        g_wrt_scale = theano.tensor.flatten(g_wrt_scale, ndim)
        g_wrt_bias = theano.tensor.flatten(g_wrt_bias, ndim)
    elif ndim > 5:
        g_wrt_inputs = theano.tensor.reshape(g_wrt_inputs, x_shape)
        g_wrt_scale = theano.tensor.reshape(g_wrt_scale, params_shape)
        g_wrt_bias = theano.tensor.reshape(g_wrt_bias, params_shape)

    # If the original output was on CPU, we have to transfer it
    if isinstance(node.outputs[0].type, tensor.TensorType):
        g_wrt_inputs = tensor.as_tensor_variable(g_wrt_inputs)
    if isinstance(node.outputs[1].type, tensor.TensorType):
        g_wrt_scale = tensor.as_tensor_variable(g_wrt_scale)
    if isinstance(node.outputs[2].type, tensor.TensorType):
        g_wrt_bias = tensor.as_tensor_variable(g_wrt_bias)
    # TODO copy_stack_trace?

    g_wrt_inputs.tag.values_eq_approx = values_eq_approx_high_tol
    g_wrt_scale.tag.values_eq_approx = values_eq_approx_high_tol
    return [g_wrt_inputs, g_wrt_scale, g_wrt_bias]


@local_optimizer([bn.AbstractBatchNormInference])
def local_abstract_batch_norm_inference_cudnn(node):
    if not isinstance(node.op, bn.AbstractBatchNormInference):
        return None

    x, scale, bias, estimated_mean, estimated_variance, epsilon = node.inputs

    axes = tuple(node.op.axes)
    if axes == (0,):
        mode = 'per-activation'
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = 'spatial'
    else:
        return None

    # input on gpu?  TODO what about the output?
    x_on_gpu = (isinstance(x.type, CudaNdarrayType) or
                (x.owner and isinstance(x.owner.op, HostFromGpu)))
    if not x_on_gpu:
        return None

    try:
        eps = float(theano.tensor.get_scalar_constant_value(epsilon))
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    if not dnn_available():
        return None

    x = as_cuda_ndarray_variable(x)
    scale = as_cuda_ndarray_variable(scale)
    bias = as_cuda_ndarray_variable(bias)
    estimated_mean = as_cuda_ndarray_variable(estimated_mean)
    estimated_variance = as_cuda_ndarray_variable(estimated_variance)

    out = dnn_batch_normalization_test(x, scale, bias, estimated_mean, estimated_variance,
                                       mode, eps)

    # If the original output was on CPU, we have to transfer it
    # TODO copy_stack_trace?
    if isinstance(node.outputs[0].type, tensor.TensorType):
        return [tensor.as_tensor_variable(out)]
    else:
        return [out]
