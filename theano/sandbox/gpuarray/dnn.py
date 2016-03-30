from __future__ import absolute_import, print_function, division
import os
import warnings

import numpy
from six import integer_types

import theano
from theano import Op, Apply, tensor, config, Variable
from theano.scalar import as_scalar, constant, Log
from theano.tensor import as_tensor_variable
from theano.gradient import DisconnectedType, grad_not_implemented
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.cmodule import GCC_compiler
from theano.gof.type import CDataType, Generic
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.tensor.nnet import LogSoftmax, SoftmaxGrad
from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs,
                                              get_conv_output_shape)
from theano.tensor.signal.pool import (
    Pool, MaxPoolGrad, AveragePoolGrad)
from . import pygpu
from .type import get_context, gpu_context_type, list_contexts, GpuArrayType
from .basic_ops import (as_gpuarray_variable, infer_context_name,
                        gpu_contiguous, GpuAllocEmpty, empty_like)
from .elemwise import GpuElemwise

# These don't exist in gpuarray
# GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad
from .nnet import GpuSoftmax
from .opt import gpu_seqopt, register_opt, conv_groupopt, op_lifter
from .opt_util import alpha_merge, output_merge, inplace_allocempty


def raise_no_cudnn(msg="CuDNN is required for convolution and pooling"):
    raise RuntimeError(msg)


def _dnn_check_compile():
    preambule = """
#include <stdio.h>
#include <cudnn.h>
#include <cudnn_helper.h>
"""

    # No need for the context in here since we won't execute that code
    body = """
cudnnHandle_t _handle = NULL;
cudnnStatus_t err;
if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
  fprintf(stderr, "could not create cuDNN handle: %s",
          cudnnGetErrorString(err));
  return 1;
}
"""

    params = ["-l", "cudnn", "-I" + os.path.dirname(__file__)]
    if config.dnn.include_path:
        params.append("-I" + config.dnn.include_path)
    if config.dnn.library_path:
        params.append("-L" + config.dnn.library_path)
    if config.nvcc.compiler_bindir:
        params.extend(['--compiler-bindir', config.nvcc.compiler_bindir])
    # Do not run here the test program. It would run on the
    # default gpu, not the one selected by the user. If mixed
    # GPU are installed or if the GPUs are configured in
    # exclusive mode, this cause bad detection.
    avail, out, err = GCC_compiler.try_flags(
        params, preambule=preambule, body=body,
        try_run=False, output=True)

    if not avail:
        return False, ("Theano cannot compile with cuDNN. "
                       "We got this error:\n" + str(err))
    return True, None


def _dnn_check_version():
    v = version()
    if v < 3007:
        return False, (
            "You have an old release of CuDNN (or a release candidate) "
            "that isn't supported.  Please update to at least v3 final "
            "version.")

    return True, None


def dnn_present():
    if dnn_present.avail is not None:
        return dnn_present.avail
    if config.dnn.enabled == "False":
        dnn_present.msg = "disabled by dnn.enabled flag"
        dnn_present.avail = False

    if pygpu is None:
        dnn_present.msg = "PyGPU not available"
        dnn_present.avail = False
        return False

    dnn_present.avail, dnn_present.msg = _dnn_check_compile()
    if dnn_present.avail:
        dnn_present.avail, dnn_present.msg = _dnn_check_version()
        if not dnn_present.avail:
            raise RuntimeError(dnn_present.msg)

    if config.dnn.enabled == "True":
        if not dnn_present.avail:
            raise RuntimeError(
                "You enabled CuDNN, but we aren't able to use it: %s" %
                dnn_present.msg)

    return dnn_present.avail

dnn_present.avail = None
dnn_present.msg = None


def dnn_available(context_name):
    if not dnn_present():
        dnn_available.msg = dnn_present.msg
        return False

    ctx = get_context(context_name)

    if not ctx.kind == 'cuda':
        dnn_available.msg = "Not on a CUDA device."
        return False

    # This is a hack because bin_id is in the from of
    # "<something>_<major><minor>" for cuda devices.
    if ctx.bin_id[-2:] < b'30':
        dnn_available.msg = "Device not supported by cuDNN"
        return False

    return True

dnn_available.msg = None


class DnnBase(COp):

    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.

    """
    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False
    params_type = gpu_context_type

    def get_params(self, node):
        return node.outputs[0].type.context

    def __init__(self, files=None, c_func=None):
        if files is None:
            files = []
        COp.__init__(self, ["dnn_base.c"] + files, c_func)

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h', 'gpuarray_helper.h',
                'gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/util.h',
                'gpuarray/ext_cuda.h', 'gpuarray_api.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include(),
                config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn', 'gpuarray']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compile_args(self):
        return ['-Wl,-rpath,' + config.dnn.library_path]

    def c_code_cache_version(self):
        return (super(DnnBase, self).c_code_cache_version(), version())


class DnnVersion(Op):
    __props__ = ()

    def c_headers(self):
        return ['cudnn.h']

    def c_header_dirs(self):
        return [config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def c_compile_args(self):
        return ['-Wl,-rpath,' + config.dnn.library_path]

    def c_support_code(self):
        return """
#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#endif
"""

    def make_node(self):
        return Apply(self, [], [Generic()()])

    def c_code(self, node, name, inputs, outputs, sub):
        o = outputs[0]
        return """
        %(o)s = PyTuple_Pack(2, PyInt_FromLong(CUDNN_VERSION), PyInt_FromLong(cudnnGetVersion()));
        """ % locals()

    def do_constant_folding(self, node):
        # Needed as we do not want to cache this information.
        return False

    def c_code_cache_version(self):
        # Not needed, but make it clear that we do not want to cache this.
        return None


def version(raises=True):
    """
    Return the current cuDNN version we link with.

    This also does a check that the header version matches the runtime version.

    :raises: If True, raise an exception if CuDNN is not present or badly installed.
        Otherwise, return -1.
    """
    if not dnn_present():
        if raises:
            raise Exception(
                "We can't determine the cudnn version as it is not available",
                dnn_available.msg)
        else:
            return -1

    if version.v is None:
        f = theano.function([], DnnVersion()(),
                            theano.Mode(optimizer=None),
                            profile=False)
        v = f()
        if v[0] != v[1]:
            raise RuntimeError("Mixed dnn version. The header is version %s "
                               "while the library is version %s." % v)
        version.v = v[1]
    return version.v
version.v = None


class GpuDnnConvDesc(COp):

    """
    This Op builds a convolution descriptor for use in the other convolution
    operations.

    See the doc of :func:`dnn_conv` for a description of the parameters

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

    def do_constant_folding(self, node):
        return False

    def __init__(self, border_mode, subsample=(1, 1), conv_mode='conv',
                 precision="float32"):
        COp.__init__(self, ["conv_desc.c"], "APPLY_SPECIFIC(conv_desc)")

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
        assert len(subsample) in (2, 3)
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

        assert precision in ['float16', 'float32', 'float64']
        self.precision = precision

    def make_node(self, kern_shape):
        if kern_shape.type.ndim != 1 or kern_shape.type.dtype != 'int64':
            raise TypeError('kern must be 1D shape tensor')

        node = Apply(self, [kern_shape],
                     [CDataType("cudnnConvolutionDescriptor_t",
                                freefunc="cudnnDestroyConvolutionDescriptor")()])
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        return node

    def get_op_params(self):
        pad0 = '0'
        pad1 = '0'
        pad2 = '0'
        if isinstance(self.border_mode, tuple):
            pad0 = str(self.border_mode[0])
            pad1 = str(self.border_mode[1])
            if len(self.border_mode) > 2:
                pad2 = str(self.border_mode[2])
            bmode = '1'
        elif self.border_mode == "valid":
            bmode = '1'
        elif self.border_mode == "half":
            bmode = '2'
        elif self.border_mode == "full":
            bmode = '0'
        else:
            raise ValueError("Invalid value for border_mode")

        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'

        sub0 = str(self.subsample[0])
        sub1 = str(self.subsample[1])
        if len(self.subsample) > 2:
            sub2 = str(self.subsample[2])
        else:
            sub2 = '0'

        if self.precision == 'float16':
            precision = 'CUDNN_DATA_HALF'
        elif self.precision == 'float32':
            precision = 'CUDNN_DATA_FLOAT'
        else:
            assert self.precision == 'float64'
            precision = 'CUDNN_DATA_DOUBLE'

        return [('NB_DIMS', str(len(self.subsample))),
                ('BORDER_MODE', bmode),
                ('PAD_0', pad0), ('PAD_1', pad1), ('PAD_2', pad2),
                ('CONV_MODE', conv_flag),
                ('SUB_0', sub0), ('SUB_1', sub1), ('SUB_2', sub2),
                ('PRECISION', precision)]

    def c_code_cache_version(self):
        return (super(GpuDnnConvDesc, self).c_code_cache_version(), version())

# scalar constants
_zero = constant(numpy.asarray(0.0, dtype='float64'))
_one = constant(numpy.asarray(1.0, dtype='float64'))


def ensure_dt(val, default, name, dtype):
    if val is None:
        val = default.clone()
    if not isinstance(val, Variable):
        val = constant(val)
    if hasattr(val, 'ndim') and val.ndim == 0:
        val = as_scalar(val)
    if not isinstance(val.type, theano.scalar.Scalar):
        raise TypeError("%s: expected a scalar value" % (name,))
    if not val.type.dtype == dtype:
        val = val.astype(dtype)
    return val


class GpuDnnConv(DnnBase):

    """
    The forward convolution.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    algo : {'small', 'none', 'large', 'fft', 'fft_tiling', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_fwd`.

    """

    __props__ = ('algo', 'inplace')

    def __init__(self, algo=None, inplace=False):
        DnnBase.__init__(self, ["dnn_conv_base.c", "dnn_fwd.c"],
                         "APPLY_SPECIFIC(conv_fwd)")

        if algo is None:
            algo = config.dnn.conv.algo_fwd
        self.algo = algo

        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

        if version() < 3000:
            if self.algo == 'fft':
                raise RuntimeError("CuDNN FFT convolution requires CuDNN v3")
            elif self.algo in ['guess_once', 'guess_on_shape_change']:
                raise RuntimeError("CuDNN selection of convolution "
                                   "implementation based on heuristics "
                                   "requires CuDNN v3")
            elif self.algo in ['time_once', 'time_on_shape_change']:
                raise RuntimeError("CuDNN convolution timing requires CuDNN v3")

        # The fft_tiling implementation is only available from CuDNN V4 onward
        if version() < 4000:
            if self.algo == 'fft_tiling':
                raise RuntimeError("CuDNN tiled-FFT convolution requires "
                                   "CuDNN v4 or more recent")

        assert self.algo in ['none', 'small', 'large', 'fft', 'fft_tiling',
                             'guess_once', 'guess_on_shape_change',
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

    def get_op_params(self):
        defs = []
        if self.inplace:
            defs.append(('CONV_INPLACE', '1'))

        alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
        if self.algo == 'none':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'
        elif self.algo == 'small':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
        elif self.algo == 'large':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'
        elif self.algo == 'direct':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT'
        elif self.algo == 'fft':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT'
        elif self.algo == 'fft_tiling':
            # need v4
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING'
        defs.append(('CONV_ALGO', alg))

        if self.algo in ['guess_once', 'guess_on_shape_change',
                         'time_once', 'time_on_shape_change']:
            defs.append(('CHOOSE_ALGO', ''))
        if self.algo in ['guess_once', 'time_once']:
            defs.append(('CHOOSE_ONCE', ''))
        if self.algo in ['time_once', 'time_on_shape_change']:
            defs.append(('CHOOSE_TIME', ''))

        return defs

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):
        ctx_name = infer_context_name(img, kern, output)
        img = as_gpuarray_variable(img, ctx_name)
        kern = as_gpuarray_variable(kern, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if img.type.ndim not in (4, 5):
            raise TypeError('img must be 4D or 5D tensor')
        if kern.type.ndim not in (4, 5):
            raise TypeError('kern must be 4D or 5D tensor')
        if output.type.ndim not in (4, 5):
            raise TypeError('output must be a 4D or 5D tensor')

        if (img.type.ndim != kern.type.ndim or
                img.type.ndim != output.type.ndim):
            raise TypeError("The number of dimensions of "
                            "img, kern and output must match")

        if (img.type.ndim == 5 and
                self.algo in ['small', 'large', 'fft', 'fft_tiling']):
            raise ValueError("convolution algo %s can't be used for "
                             "3d convolutions", (self.algo,))

        if (not isinstance(desc.type, CDataType) or
                desc.type.ctype != 'cudnnConvolutionDescriptor_t'):
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_dt(alpha, _one, 'alpha', img.dtype)
        beta = ensure_dt(beta, _zero, 'beta', img.dtype)

        return Apply(self, [img, kern, output, desc, alpha, beta],
                     [output.type()])

    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConvGradI()(kerns, top, empty_like(img), desc)
        d_kerns = GpuDnnConvGradW()(img, top, empty_like(kerns), desc)
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

        # if ishape and/or kshape are not tuples or list, but rather symbolic
        # vectors, turn them into lists of symbolic scalars.
        if not isinstance(ishape, (list, tuple)):
            ishape = [ishape[i] for i in range(len(subsample) + 2)]
        if not isinstance(kshape, (list, tuple)):
            kshape = [kshape[i] for i in range(len(subsample) + 2)]

        return get_conv_output_shape(
            ishape,
            kshape,
            border_mode,
            subsample)

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConvGradW(DnnBase):

    """
    The convolution gradient with respect to the weights.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.

    """

    __props__ = ('algo', 'inplace')

    def __init__(self, inplace=False, algo=None):
        DnnBase.__init__(self, ["dnn_conv_base.c", "dnn_gw.c"],
                         "APPLY_SPECIFIC(conv_gw)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}
        if algo is None:
            algo = config.dnn.conv.algo_bwd_filter
        self.algo = algo

        assert self.algo in ['none', 'deterministic', 'fft', 'small',
                             'guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'inplace'):
            self.inplace = False
        if not hasattr(self, 'algo'):
            self.algo = config.dnn.conv.algo_bwd_filter

    def grad(self, inp, grads):
        img, top, output, desc, alpha, beta = inp
        kerns, = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnnConvGradI()(kerns, top, empty_like(img), desc)
        d_top = GpuDnnConv()(img, kerns, empty_like(top), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_img * alpha, d_top * alpha, kerns * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        defs = []
        if self.inplace:
            defs.append(('CONV_INPLACE', '1'))

        if version() < 3000:
            alg = '0'
        else:
            alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
            if self.algo == 'none':
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
            if self.algo == 'deterministic':
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1'
            if self.algo == 'fft':
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT'
            if self.algo == 'small':
                # non-deterministic, small workspace
                alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3'
            if self.algo in ['guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']:
                defs.append(('CHOOSE_ALGO', ''))
            if self.algo in ['guess_once', 'time_once']:
                defs.append(('CHOOSE_ONCE', ''))
            if self.algo in ['time_once', 'time_on_shape_change']:
                defs.append(('CHOOSE_TIME', ''))

        defs.append(('CONV_ALGO', alg))

        return defs

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        ctx_name = infer_context_name(img, topgrad, output)
        img = as_gpuarray_variable(img, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if img.type.ndim not in (4, 5):
            raise TypeError('img must be 4D or 5D tensor')
        if topgrad.type.ndim not in (4, 5):
            raise TypeError('topgrad must be 4D or 5D tensor')
        if output.type.ndim not in (4, 5):
            raise TypeError('output must be 4D or 5D tensor')

        if (img.type.ndim != topgrad.type.ndim or
                img.type.ndim != output.type.ndim):
            raise TypeError("The number of dimensions of "
                            "img, topgrad and output must match")

        if (img.type.ndim == 5 and
                self.algo in ['fft', 'deterministic', 'small']):
            raise ValueError("convolution algo %s can't be used for "
                             "3d convolutions", (self.algo,))

        if (not isinstance(desc.type, CDataType) or
                desc.type.ctype != 'cudnnConvolutionDescriptor_t'):
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_dt(alpha, _one, 'alpha', img.dtype)
        beta = ensure_dt(beta, _zero, 'beta', img.dtype)

        return Apply(self, [img, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConvGradI(DnnBase):

    """
    The convolution gradient with respect to the inputs.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.

    """

    __props__ = ('algo', 'inplace',)

    def __init__(self, inplace=False, algo=None):
        DnnBase.__init__(self, ["dnn_conv_base.c", "dnn_gi.c"],
                         "APPLY_SPECIFIC(conv_gi)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}
        if algo is None:
            algo = config.dnn.conv.algo_bwd_data
        self.algo = algo

        # The small-workspace implementation is only available from CuDNN V4
        # onward.
        if version() < 4000 and self.algo == 'fft_tiling':
            raise RuntimeError("CuDNN's tiled-FFT convolution requires CuDNN "
                               "v4 or more recent")

        assert self.algo in ['none', 'deterministic', 'fft', 'fft_tiling',
                             'guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'algo'):
            self.algo = config.dnn.conv.algo_bwd_data
        if not hasattr(self, 'inplace'):
            self.inplace = False

    def grad(self, inp, grads):
        kerns, top, output, desc, alpha, beta = inp
        img, = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnnConvGradW()(img, top, empty_like(kerns), desc)
        d_top = GpuDnnConv()(img, kerns, empty_like(top), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_kerns * alpha, d_top * alpha, img * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        defs = []
        if self.inplace:
            defs.append(('CONV_INPLACE', '1'))

        if version() < 3000:
            alg = '0'
        else:
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
            if self.algo == 'none':
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
            if self.algo == 'deterministic':
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1'
            if self.algo == 'fft':
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT'
            if self.algo == 'fft_tiling':
                # big workspace but less than fft
                alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING'

            if self.algo in ['guess_once', 'guess_on_shape_change',
                             'time_once', 'time_on_shape_change']:
                defs.append(('CHOOSE_ALGO', ''))
            if self.algo in ['guess_once', 'time_once']:
                defs.append(('CHOOSE_ONCE', ''))
            if self.algo in ['time_once', 'time_on_shape_change']:
                defs.append(('CHOOSE_TIME', ''))

        defs.append(('CONV_ALGO', alg))

        return defs

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        ctx_name = infer_context_name(kern, topgrad, output)
        kern = as_gpuarray_variable(kern, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if kern.type.ndim not in (4, 5):
            raise TypeError('kern must be 4D or 5D tensor')
        if topgrad.type.ndim not in (4, 5):
            raise TypeError('topgrad must be 4D or 5D tensor')
        if output.type.ndim not in (4, 5):
            raise TypeError('output must be 4D or 5D tensor')

        if (kern.type.ndim != topgrad.type.ndim or
                kern.type.ndim != output.type.ndim):
            raise TypeError("The number of dimensions of "
                            "kern, topgrad and output must match")

        if (kern.type.ndim == 5 and
                self.algo in ['fft', 'deterministic', 'fft_tiling']):
            raise ValueError("convolution algo %s can't be used for "
                             "3d convolutions", (self.algo,))

        if (not isinstance(desc.type, CDataType) or
                desc.type.ctype != 'cudnnConvolutionDescriptor_t'):
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_dt(alpha, _one, 'alpha', kern.dtype)
        beta = ensure_dt(beta, _zero, 'beta', kern.dtype)

        return Apply(self, [kern, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]


def dnn_conv(img, kerns, border_mode='valid', subsample=(1, 1),
             conv_mode='conv', direction_hint=None, workmem=None,
             algo=None, precision=None):
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
        One of 'valid', 'full', 'half'; additionally, the padding size
        could be directly specified by an integer or a pair of integers.
    subsample
        Perform subsampling of the output (default: (1, 1)).
    conv_mode
        Perform convolution (kernels flipped) or cross-correlation.
        One of 'conv', 'cross' (default: 'conv').
    direction_hint
        Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1, 1) and direction_hint is
        'bprop weights', it will use GpuDnnConvGradW.
        If border_mode is 'full', subsample is (1, 1) and direction_hint is
        *not* 'forward!', it will use GpuDnnConvGradI.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    algo : {'none', 'small', 'large', 'fft', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Convolution implementation to use. Some of its values may
        require certain versions of CuDNN to be installed. Default is
        the value of :attr:`config.dnn.conv.algo_fwd`.
    precision : {'as_input', 'float16', 'float32', 'float64'}
        Description of the dtype in which the computation of the convolution
        should be done. Possible values are 'as_input', 'float16', 'float32'
        and 'float64'. Default is the value of
        :attr:`config.dnn.conv.precision`.

    .. warning:: The cuDNN library only works with GPUs that have a compute
        capability of 3.0 or higer. This means that older GPUs will not
        work with this Op.

    """

    # Establish dtype in which to perform the computation of the convolution
    if precision is None:
        precision = theano.config.dnn.conv.precision
    if precision == 'as_input':
        precision = theano.scalar.upcast(img.dtype, kerns.dtype)

    if workmem is not None:
        if algo is not None:
            raise ValueError("You can't use both algo and workmem")
        warnings.warn("workmem is deprecated, use algo instead", stacklevel=2)
        algo = workmem
    fgraph = getattr(img, 'fgraph', None) or getattr(kerns, 'fgraph', None)
    ctx_name = infer_context_name(img, kerns)
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
        shape2 = shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1
        shape3 = shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1
        out = GpuAllocEmpty(img.dtype, ctx_name)(
            shape_i(kerns, 1, fgraph),
            shape_i(img, 1, fgraph), shape2, shape3)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode='cross', precision=precision)(out.shape)
        conv = GpuDnnConvGradW()(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)

    elif (border_mode == 'full' and subsample == (1, 1) and
          direction_hint != 'forward!'):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        shape2 = shape_i(img, 2, fgraph) + shape_i(kerns, 2, fgraph) - 1
        shape3 = shape_i(img, 3, fgraph) + shape_i(kerns, 3, fgraph) - 1
        out = GpuAllocEmpty(img.dtype, ctx_name)(shape_i(img, 0, fgraph),
                                                 shape_i(kerns, 1, fgraph),
                                                 shape2, shape3)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode=conv_mode, precision=precision)(kerns.shape)
        return GpuDnnConvGradI()(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode, precision=precision)(kerns.shape)
    desc_op = desc.owner.op
    out_shp = GpuDnnConv.get_out_shape(img.shape, kerns.shape,
                                       desc_op.border_mode,
                                       desc_op.subsample)
    out = GpuAllocEmpty(img.dtype, ctx_name)(*out_shp)
    return GpuDnnConv(algo=algo)(img, kerns, out, desc)


def dnn_gradweight(img, topgrad, kerns_shp, border_mode='valid',
                   subsample=(1, 1), conv_mode='conv'):
    ctx_name = infer_context_name(img, topgrad)
    img = as_gpuarray_variable(img, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    kerns_shp = as_tensor_variable(kerns_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(kerns_shp)
    out = GpuAllocEmpty(img.dtype, ctx_name)(*kerns_shp)
    return GpuDnnConvGradW()(img, topgrad, out, desc)


def dnn_gradinput(kerns, topgrad, img_shp, border_mode='valid',
                  subsample=(1, 1), conv_mode='conv'):
    ctx_name = infer_context_name(kerns, topgrad)
    kerns = as_gpuarray_variable(kerns, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)
    img_shp = as_tensor_variable(img_shp)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(kerns.shape)
    out = GpuAllocEmpty(kerns.dtype, ctx_name)(*img_shp)
    return GpuDnnConvGradI()(kerns, topgrad, out, desc)


class GpuDnnPoolDesc(Op):

    """
    This Op builds a pooling descriptor for use in the other
    pooling operations.

    `ws`, `stride` and `pad` must have the same length.

    Parameters
    ----------
    ws : tuple
        Window size.
    stride : tuple
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tuple
        (padX, padY) or (padX, padY, padZ)

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

    def do_constant_folding(self, node):
        return False

    def __init__(self, ws=(1, 1), stride=(1, 1), mode='max', pad=(0, 0)):
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

        assert len(ws) == len(stride) and len(stride) == len(pad)
        assert len(ws) in (2, 3)
        self.ws = ws
        self.stride = stride
        self.pad = pad

        if self.get_ndim() == 3 and version() < 3000:
            raise RuntimeError("CuDNN 3d pooling requires v3")
        if mode == 'average_exc_pad' and max(pad) > 0 and version() < 4004:
            raise RuntimeError(
                "CuDNN pooling mode 'average_exc_pad' requires at least v4")

    def get_ndim(self):
        return len(self.ws)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'pad'):
            self.pad = (0, 0)

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

  static const int win[%(nd)d] = {%(win)s};
  static const int pad[%(nd)d] = {%(pad)s};
  static const int str[%(nd)d] = {%(str)s};
  err = cudnnSetPoolingNdDescriptor(
    %(desc)s, %(mode_flag)s, %(nd)d,
    win, pad, str);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, desc=desc, mode_flag=mode_flag, fail=sub['fail'],
           nd=self.get_ndim(), win=', '.join(map(str, self.ws)),
           pad=', '.join(map(str, self.pad)),
           str=', '.join(map(str, self.stride)))

    def c_code_cache_version(self):
        return (3, version())


class GpuDnnPool(DnnBase):

    """
    Parameters
    ----------
    img
        The image 4d or 5d tensor.
    Parameters
    ----------
    ws : tensor variable
        Window size.
    stride : tensor variable
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tensor
        (padX, padY) or (padX, padY, padZ)
    """

    __props__ = ('mode',)

    def __init__(self, mode='max'):
        DnnBase.__init__(self, ["dnn_pool.c"], "APPLY_SPECIFIC(dnn_pool)")
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

    def get_op_params(self):
        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'

        return [('MODE_FLAG', mode_flag)]

    def make_node(self, img, ws, stride, pad):
        ctx_name = infer_context_name(img)
        img = as_gpuarray_variable(img, ctx_name)

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

        res = [shape[0][0], shape[0][1],
               (shape[0][2] + 2 * p[0] - w[0]) // s[0] + 1,
               (shape[0][3] + 2 * p[1] - w[1]) // s[1] + 1
               ]
        if node.inputs[0].ndim == 5:
            res.append((shape[0][4] + 2 * p[2] - w[2]) // s[2] + 1)
        return [res]

    def grad(self, inp, grads):
        img, ws, stride, pad = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        out = self(img, ws, stride, pad)

        g_out = GpuDnnPoolGrad(mode=self.mode)(img, out, grad, ws, stride, pad)

        return g_out, theano.gradient.DisconnectedType()(), theano.gradient.DisconnectedType()(), theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to parameters
        return [[1], [0], [0], [0]]


class GpuDnnPoolGrad(DnnBase):

    """
    The pooling gradient.

    Parameters
    ----------
    inp
        The input of the pooling.
    out
        The output of the pooling in the forward.
    out_grad
        Same size as out, but is the corresponding gradient information.
    ws : tensor variable
        Window size.
    stride : tensor variable
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tensor
        (padX, padY) or (padX, padY, padZ)

    """

    __props__ = ('mode',)

    def __init__(self, mode='max'):
        DnnBase.__init__(self, ["dnn_pool_grad.c"],
                         "APPLY_SPECIFIC(dnn_pool_grad)")
        if mode == 'average':
            mode = 'average_inc_pad'
        assert mode in ('max', 'average_inc_pad', 'average_exc_pad')
        self.mode = mode

    def get_op_params(self):
        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'

        return [('MODE_FLAG', mode_flag)]

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        ctx_name = infer_context_name(inp, out, out_grad)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim in [4, 5])
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out_grad.ndim in [4, 5])
        out = as_gpuarray_variable(out, ctx_name)
        assert(out.ndim in [4, 5])

        assert (out_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def infer_shape(self, node, shape):
        return [shape[0]]


def dnn_pool(img, ws, stride=(1, 1), mode='max', pad=(0, 0)):
    """
    GPU pooling using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    `ws`, `stride` and `pad` must have the same length.

    Parameters
    ----------
    img
        Images to do the pooling over.
    ws : tuple
        Subsampling window size.
    stride : tuple
        Subsampling stride (default: (1, 1)).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
    pad : tuple
        (padX, padY) or (padX, padY, padZ)
        default: (0, 0)

    .. warning:: The cuDNN library only works with GPU that have a compute
        capability of 3.0 or higer.  This means that older GPU will not
        work with this Op.

    Notes
    -----
    This Op implements the ignore_border=True of max_pool_2d.

    """
    img = gpu_contiguous(img)
    return GpuDnnPool(mode=mode)(img, ws, stride, pad)


class GpuDnnSoftmaxBase(DnnBase):

    """
    Op for the cuDNN Softmax.

    Parameters
    ----------
    algo
        'fast', 'accurate' or 'log' indicating whether, respectively,
        computations should be optimized for speed, for accuracy, or if CuDNN
        should rather compute the log-softmax instead.
    mode
        'instance' or 'channel' indicating whether the softmax should be
        computed per image across 'c01' or per spatial location '01' per
        image across 'c'.

    """

    __props__ = ('mode', 'algo')

    def __init__(self, algo, mode):
        DnnBase.__init__(self, [self.file], self.c_func)

        assert(algo in ('fast', 'accurate', 'log'))
        if algo == 'log' and version(raises=False) < 3000:
            raise RuntimeError("Need CuDNN v3 for log-softmax")
        self.algo = algo

        assert(mode in ('instance', 'channel'))
        self.mode = mode

    def infer_shape(self, node, shape):
        if self.direction == 'forward':
            return [shape[0]]
        else:
            return [shape[1]]

    def get_op_params(self):
        if self.mode == 'instance':
            mode = "CUDNN_SOFTMAX_MODE_INSTANCE"
        else:
            mode = "CUDNN_SOFTMAX_MODE_CHANNEL"

        if self.algo == 'fast':
            algo = "CUDNN_SOFTMAX_FAST"
        elif self.algo == 'log':
            algo = "CUDNN_SOFTMAX_LOG"
        else:
            algo = "CUDNN_SOFTMAX_ACCURATE"

        return [("SOFTMAX_MODE", mode), ("SOFTMAX_ALGO", algo)]


class GpuDnnSoftmax(GpuDnnSoftmaxBase):

    """
    Op for the cuDNN Softmax.

    algo
        'fast', 'accurate' or 'log' indicating whether, respectively,
        computations should be optimized for speed, for accuracy, or if CuDNN
        should rather compute the log-softmax instead.
    mode
        'instance' or 'channel' indicating whether the softmax should be
        computed per image across 'c01' or per spatial location '01' per
        image across 'c'.

    """
    direction = "forward"
    file = "dnn_softmax.c"
    c_func = "APPLY_SPECIFIC(softmax)"

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = self.make_node(x).outputs[0]
        return [GpuDnnSoftmaxGrad(
                self.algo,
                self.mode
                )(g_sm, sm)]


class GpuDnnSoftmaxGrad(GpuDnnSoftmaxBase):

    """
    Op for the cuDNN SoftmaxGrad.

    Parameters
    ----------
    algo
        'fast', 'accurate' or 'log' indicating whether, respectively,
        computations should be optimized for speed, for accuracy, or if CuDNN
        should rather compute the gradient of the log-softmax instead.
    mode
        'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.

    """
    direction = 'backward'
    file = "dnn_softmax_grad.c"
    c_func = "APPLY_SPECIFIC(softmax_grad)"

    def make_node(self, dy, sm):
        ctx_name = infer_context_name(dy, sm)
        dy = as_gpuarray_variable(dy, ctx_name)
        sm = as_gpuarray_variable(sm, ctx_name)
        assert dy.ndim == 4
        assert sm.ndim == 4
        return Apply(self, [dy, sm], [sm.type()])


@local_optimizer([AbstractConv2d, AbstractConv2d_gradWeights,
                  AbstractConv2d_gradInputs])
def local_abstractconv_cudnn(node):
    if (not isinstance(node.op, (AbstractConv2d,
                                 AbstractConv2d_gradWeights,
                                 AbstractConv2d_gradInputs))):
        return None

    inp1 = node.inputs[0]
    inp2 = node.inputs[1]

    if not isinstance(inp1.type, GpuArrayType):
        return None

    if not dnn_available(inp1.type.context_name):
        raise_no_cudnn()

    if node.op.filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'

    if isinstance(node.op, AbstractConv2d):
        rval = dnn_conv(inp1, inp2,
                        border_mode=node.op.border_mode,
                        subsample=node.op.subsample,
                        direction_hint='forward!',
                        conv_mode=conv_mode)
    if isinstance(node.op, AbstractConv2d_gradWeights):
        shape = (inp2.shape[1], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1])
        rval = dnn_gradweight(inp1, inp2, shape,
                              border_mode=node.op.border_mode,
                              subsample=node.op.subsample,
                              conv_mode=conv_mode)
    if isinstance(node.op, AbstractConv2d_gradInputs):
        shape = (inp2.shape[0], inp1.shape[1],
                 node.inputs[2][0], node.inputs[2][1])
        rval = dnn_gradinput(inp1, inp2, shape,
                             border_mode=node.op.border_mode,
                             subsample=node.op.subsample,
                             conv_mode=conv_mode)
    return [rval]

conv_groupopt.register('local_abstractconv_cudnn',
                       local_abstractconv_cudnn, 20,
                       'fast_compile', 'fast_run',
                       'gpuarray', 'conv_dnn', 'cudnn')


@inplace_allocempty(GpuDnnConv, 2)
def local_dnn_conv_inplace(node, inputs):
    return [GpuDnnConv(algo=node.op.algo, inplace=True)(*inputs)]


@inplace_allocempty(GpuDnnConvGradW, 2)
def local_dnn_convgw_inplace(node, inputs):
    return [GpuDnnConvGradW(algo=node.op.algo, inplace=True)(*inputs)]


@inplace_allocempty(GpuDnnConvGradI, 2)
def local_dnn_convgi_inplace(node, inputs):
    return [GpuDnnConvGradI(algo=node.op.algo, inplace=True)(*inputs)]

optdb.register('local_dnna_conv_inplace',
               tensor.opt.in2out(local_dnn_conv_inplace,
                                 local_dnn_convgw_inplace,
                                 local_dnn_convgi_inplace,
                                 name="local_dnna_conv_inplace"),
               70.0, 'fast_run', 'inplace', 'gpuarray', 'cudnn')


@register_opt('cudnn')
@alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5)
def local_dnn_conv_alpha_merge(node, *inputs):
    return [GpuDnnConv(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5)
def local_dnn_convw_alpha_merge(node, *inputs):
    return [GpuDnnConvGradW(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5)
def local_dnn_convi_alpha_merge(node, *inputs):
    return [GpuDnnConvGradI(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_conv_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConv(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convw_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradW(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convi_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradI(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@op_lifter([Pool])
def local_pool_dnn_alternative(node, ctx_name):
    if not dnn_available(ctx_name):
        raise_no_cudnn()
    if not node.op.ignore_border:
        return
    img, = node.inputs
    img = as_gpuarray_variable(img, ctx_name)
    ds = node.op.ds
    stride = node.op.st
    pad = node.op.padding
    mode = node.op.mode
    return dnn_pool(gpu_contiguous(img), ds, stride=stride, pad=pad, mode=mode)


@register_opt('cudnn')
@op_lifter([MaxPoolGrad])
def local_pool_dnn_grad_stride(node, ctx_name):
    if not dnn_available(ctx_name):
        raise_no_cudnn()
    if not node.op.ignore_border:
        return
    inp, out, out_grad = node.inputs
    inp = as_gpuarray_variable(inp, ctx_name)
    out = as_gpuarray_variable(out, ctx_name)
    out_grad = as_gpuarray_variable(out_grad, ctx_name)
    ds = node.op.ds
    st = node.op.st
    pad = node.op.padding
    mode = node.op.mode

    return GpuDnnPoolGrad(mode=mode)(gpu_contiguous(inp),
                                     gpu_contiguous(out),
                                     gpu_contiguous(out_grad),
                                     ds,
                                     st,
                                     pad)


@register_opt('cudnn')
@op_lifter([AveragePoolGrad])
def local_avg_pool_dnn_grad_stride(node, ctx_name):
    if not dnn_available(ctx_name):
        raise_no_cudnn()
    if not node.op.ignore_border:
        return
    inp, out_grad = node.inputs
    inp = as_gpuarray_variable(inp, ctx_name)
    out_grad = as_gpuarray_variable(out_grad, ctx_name)
    ds = node.op.ds
    st = node.op.st
    pad = node.op.padding
    mode = node.op.mode

    cg = gpu_contiguous(out_grad)

    # We reuse cg because CuDNN does not use the value of the `out`
    # argument but still checks its shape for average pooling. This
    # has been observed in v2 and v3 as far as I know.
    return GpuDnnPoolGrad(mode=mode)(gpu_contiguous(inp), cg, cg, ds, st, pad)


@register_opt('cudnn')
@local_optimizer([GpuSoftmax])
def local_softmax_dnn(node):
    if isinstance(node.op, GpuSoftmax):
        if not dnn_available(node.outputs[0].type.context_name):
            raise_no_cudnn()
        ins = node.inputs[0].dimshuffle(0, 1, 'x', 'x')
        ins = gpu_contiguous(ins)
        out = GpuDnnSoftmax('accurate', 'channel')(ins)
        out = as_gpuarray_variable(out.dimshuffle(0, 1), out.type.context_name)
        return [out]


@register_opt('cudnn')
@local_optimizer([GpuElemwise])
def local_log_softmax_dnn(node):
    # This looks for GpuDnnSoftmax so we know that we have cudnn.
    if (isinstance(node.op, GpuElemwise) and
            isinstance(node.op.scalar_op, Log) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, GpuDnnSoftmax) and
            len(node.inputs[0].clients) == 1):
        if version(raises=False) < 3000:
            # No log-softmax before cudnn v3
            raise_no_cudnn("Need CuDNN v3 for LogSoftmax")
        softmax_node = node.inputs[0].owner
        new_softmax = GpuDnnSoftmax('log', softmax_node.op.mode)
        return [new_softmax(softmax_node.inputs[0])]


@register_opt('cudnn')
@op_lifter([LogSoftmax])
def local_logsoftmax_to_dnn(node, ctx_name):
    # Transform the input in the format expected by GpuDnnSoftmax
    inp = node.inputs[0]
    if inp.ndim != 2:
        return
    if not dnn_available(ctx_name) or version(raises=False) < 3000:
        # No log-softmax before cudnn v3
        raise_no_cudnn("Need CuDNN v3 for LogSoftmax")

    inp = inp.dimshuffle(0, 1, 'x', 'x')
    inp.tag.context_name = ctx_name

    # Apply GpuDnnSoftmax and return the result
    out = GpuDnnSoftmax('log', 'channel')(gpu_contiguous(inp))
    return [out.dimshuffle(0, 1)]


class NoCuDNNRaise(Optimizer):

    def apply(self, fgraph):
        """
        Raise a error if cudnn can't be used.

        """
        for c in list_contexts():
            if not dnn_available(c):
                # Make an assert error as we want Theano to fail, not
                # just skip this optimization.
                raise AssertionError(
                    "cuDNN optimization was enabled, but Theano was not able "
                    "to use it for context " + c + ". We got this error: \n" +
                    dnn_available.msg)

gpu_seqopt.register("NoCuDNNRaise", NoCuDNNRaise(), 0, 'cudnn')


@register_opt('cudnn')
@op_lifter([SoftmaxGrad])
def local_softmax_dnn_grad(node, ctx_name):
    if not dnn_available(ctx_name):
        raise_no_cudnn("CuDNN needed for SoftmaxGrad")
    ins = []
    for n in node.inputs:
        n = as_gpuarray_variable(n, ctx_name)
        if n.ndim != 2:
            return
        ins.append(n.dimshuffle(0, 1, 'x', 'x'))

    out = GpuDnnSoftmaxGrad('accurate', 'channel')(
        gpu_contiguous(ins[0]), gpu_contiguous(ins[1]))
    return [out.dimshuffle(0, 1)]
