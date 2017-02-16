from __future__ import absolute_import, print_function, division
import ctypes
import os
import sys
import warnings

import numpy as np
from six import integer_types

import theano
from theano import Op, Apply, tensor, config, Variable
from theano.scalar import as_scalar, constant, Log, get_scalar_type
from theano.tensor import as_tensor_variable
from theano.gradient import DisconnectedType, grad_not_implemented
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.cmodule import GCC_compiler
from theano.gof.type import CDataType, Generic
from theano.compile import optdb
from theano.compile.ops import shape_i, shape_i_op
from theano.tensor.nnet import LogSoftmax, SoftmaxGrad
from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs,
                                              AbstractConv3d,
                                              AbstractConv3d_gradWeights,
                                              AbstractConv3d_gradInputs,
                                              get_conv_output_shape,
                                              assert_conv_shape)
from theano.tensor.signal.pool import (
    Pool, MaxPoolGrad, AveragePoolGrad)
from . import pygpu
from .type import (get_context, gpu_context_type, list_contexts,
                   GpuArraySharedVariable)
from .basic_ops import (as_gpuarray_variable, infer_context_name,
                        gpu_contiguous, gpu_alloc_empty,
                        empty_like, GpuArrayType, HostFromGpu)
from .elemwise import GpuElemwise

# These don't exist in gpuarray
# GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad
from .nnet import GpuSoftmax
from .opt import (gpu_seqopt, register_opt, pool_db, pool_db2,
                  op_lifter, register_opt2, register_inplace)

from .opt_util import alpha_merge, output_merge, inplace_allocempty, pad_dims, unpad_dims

from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_BWD_FILTER

try:
    from pygpu import gpuarray
except ImportError:
    pass


def _dnn_lib():
    if _dnn_lib.handle is None:
        import ctypes.util

        lib_name = ctypes.util.find_library('cudnn')
        if lib_name is None and sys.platform == 'win32':
            # Update these names when new versions of cudnn are supported.
            for name in ['cudnn64_5.dll']:
                lib_name = ctypes.util.find_library(name)
                if lib_name:
                    break
        if lib_name is None:
            raise RuntimeError('Could not find cudnn library (looked for v5[.1])')
        _dnn_lib.handle = ctypes.cdll.LoadLibrary(lib_name)
        cudnn = _dnn_lib.handle
        cudnn.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cudnn.cudnnCreate.restype = ctypes.c_int
        cudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
        cudnn.cudnnDestroy.restype = ctypes.c_int
    return _dnn_lib.handle

_dnn_lib.handle = None


def _make_handle(ctx):
    cudnn = _dnn_lib()
    handle = ctypes.c_void_p()
    with ctx:
        err = cudnn.cudnnCreate(ctypes.byref(handle))
    if err != 0:
        raise RuntimeError("error creating cudnn handle")
    return handle


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
    path_wrapper = "\"" if os.name == 'nt' else ""
    params = ["-l", "cudnn"]
    params.extend(['-I%s%s%s' % (path_wrapper, os.path.dirname(__file__), path_wrapper)])
    if config.dnn.include_path:
        params.extend(['-I%s%s%s' % (path_wrapper, config.dnn.include_path, path_wrapper)])
    if config.dnn.library_path:
        params.extend(['-L%s%s%s' % (path_wrapper, config.dnn.library_path, path_wrapper)])
    # Do not run here the test program. It would run on the
    # default gpu, not the one selected by the user. If mixed
    # GPU are installed or if the GPUs are configured in
    # exclusive mode, this cause bad detection.
    avail, out, err = GCC_compiler.try_flags(
        params, preambule=preambule, body=body,
        try_run=False, output=True)

    if not avail:
        return False, ("cannot compile with cuDNN. "
                       "We got this error:\n" + str(err))
    return True, None


def _dnn_check_version():
    v = version()
    if v < 5000:
        return False, "cuDNN version is too old. Update to v5, was %d." % v
    # 5200 should not print warning with cudnn 5.1 final.
    if v >= 5200:
        warnings.warn("Your cuDNN version is more recent than "
                      "Theano. If you encounter problems, try "
                      "updating Theano or downgrading cuDNN to "
                      "version 5.1.")
    return True, None


def dnn_present():
    if dnn_present.avail is not None:
        return dnn_present.avail
    if config.dnn.enabled == "False":
        dnn_present.msg = "Disabled by dnn.enabled flag"
        dnn_present.avail = False
        return False

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
                "You enabled cuDNN, but we aren't able to use it: %s" %
                dnn_present.msg)

    return dnn_present.avail

dnn_present.avail = None
dnn_present.msg = None


def dnn_available(context_name):
    if not dnn_present():
        dnn_available.msg = dnn_present.msg
        return False

    ctx = get_context(context_name)

    if not ctx.kind == b'cuda':
        dnn_available.msg = "Not on a CUDA device."
        return False

    # This is a hack because bin_id is in the from of
    # "<something>_<major><minor>" for cuda devices.
    if ctx.bin_id[-2:] < b'30':
        dnn_available.msg = "Device not supported"
        return False

    return True

dnn_available.msg = None

handle_type = CDataType('cudnnHandle_t', 'cudnnDestroy',
                        headers=['cudnn.h'],
                        header_dirs=[config.dnn.include_path],
                        libraries=['cudnn'],
                        lib_dirs=[config.dnn.library_path])


class DnnBase(COp):

    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.

    """
    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False
    params_type = handle_type

    def dnn_context(self, node):
        return node.outputs[0].type.context_name

    def get_params(self, node):
        ctx_name = self.dnn_context(node)
        ctx = get_context(ctx_name)
        if not hasattr(ctx, 'cudnn_handle_param'):
            ptr = ctx.cudnn_handle.value
            res = handle_type.make_value(ptr)
            ctx.cudnn_handle_param = res
        return ctx.cudnn_handle_param

    def __init__(self, files=None, c_func=None):
        if files is None:
            files = []
        COp.__init__(self, ["dnn_base.c"] + files, c_func)

    def c_headers(self):
        return ['gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/kernel.h',
                'gpuarray/util.h', 'gpuarray/ext_cuda.h', 'gpuarray_api.h',
                'numpy_compat.h', 'cudnn.h', 'cudnn_helper.h',
                'gpuarray_helper.h']

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
        return (super(DnnBase, self).c_code_cache_version(), version(), 1)


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

    :raises: If True, raise an exception if cuDNN is not present or badly installed.
        Otherwise, return -1.

    """
    if not dnn_present():
        if raises:
            raise RuntimeError(
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


def gpu_dnn_conv_desc(border_mode, subsample=(1, 1), conv_mode='conv',
                      precision="float32"):
    key = (border_mode, subsample, conv_mode, precision)
    if key not in gpu_dnn_conv_desc.cache:
        gpu_dnn_conv_desc.cache[key] = GpuDnnConvDesc(border_mode,
                                                      subsample,
                                                      conv_mode,
                                                      precision)
    return gpu_dnn_conv_desc.cache[key]
gpu_dnn_conv_desc.cache = {}


# scalar constants
_zero = constant(np.asarray(0.0, dtype='float64'))
_one = constant(np.asarray(1.0, dtype='float64'))


def ensure_dt(val, default, name, dtype):
    if dtype == 'float16':
        dtype = 'float32'
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
    descr :
        The convolution descriptor.
    algo : {'small', 'none', 'large', 'fft', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_fwd`.

    """
    _f16_ok = True
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

    def get_op_params(self):
        defs = []
        if self.inplace:
            defs.append(('CONV_INPLACE', '1'))

        alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
        if self.algo == 'none':  # 3d
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'
        elif self.algo == 'small':  # 3d
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
        elif self.algo == 'large':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'
        elif self.algo == 'direct':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT'
        elif self.algo == 'fft':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT'
        elif self.algo == 'fft_tiling':  # 3d
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING'
        elif self.algo == 'winograd':
            alg = 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD'
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

        if img.type.ndim == 5 and self.algo in ['large', 'fft']:
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

        d_img = gpu_dnn_conv_gradI()(kerns, top, empty_like(img), desc)
        d_kerns = gpu_dnn_conv_gradW()(img, top, empty_like(kerns), desc)
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


def gpu_dnn_conv(algo=None, inplace=False):
    key = (algo, inplace)
    if key not in gpu_dnn_conv.cache:
        gpu_dnn_conv.cache[key] = GpuDnnConv(algo, inplace)
    return gpu_dnn_conv.cache[key]
gpu_dnn_conv.cache = {}


class GpuDnnConvGradW(DnnBase):

    """
    The convolution gradient with respect to the weights.

    Parameters
    ----------
    image
    kernel
    descr :
        The convolution descriptor.
    algo : {'none', 'deterministic', 'fft', 'small', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_filter`.

    """
    _f16_ok = True
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

        assert self.algo in SUPPORTED_DNN_CONV_ALGO_BWD_FILTER

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

        d_img = gpu_dnn_conv_gradI()(kerns, top, empty_like(img), desc)
        d_top = gpu_dnn_conv()(img, kerns, empty_like(top), desc)
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

        alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
        if self.algo == 'none':  # 3d
            alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0'
        if self.algo == 'deterministic':
            alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1'
        if self.algo == 'fft':
            alg = 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT'
        if self.algo == 'small':  # 3d
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
        if img.type.ndim == 5 and self.algo != 'none' and desc.owner.op.subsample != (1, 1, 1):
            warnings.warn('cuDNN backward filter operation for 3D convolutions may produce bad results '
                          'with certain cuDNN algorithms depending on the compute capability of your GPU '
                          'if subsample is not (1, 1, 1). If you encounter problems, consider '
                          'setting the theano flag "dnn.conv.algo_bwd_filter" to "none".')
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
                self.algo in ['fft', 'deterministic']):
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


def gpu_dnn_conv_gradW(algo=None, inplace=False):
    key = (algo, inplace)
    if key not in gpu_dnn_conv_gradW.cache:
        gpu_dnn_conv_gradW.cache[key] = GpuDnnConvGradW(inplace, algo)
    return gpu_dnn_conv_gradW.cache[key]
gpu_dnn_conv_gradW.cache = {}


class GpuDnnConvGradI(DnnBase):
    """
    The convolution gradient with respect to the inputs.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    algo : {'none', 'deterministic', 'fft', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn.conv.algo_bwd_data`.

    """
    _f16_ok = True
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

        assert self.algo in ['none', 'deterministic', 'fft', 'fft_tiling',
                             'winograd', 'guess_once', 'guess_on_shape_change',
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

        d_kerns = gpu_dnn_conv_gradW()(img, top, empty_like(kerns), desc)
        d_top = gpu_dnn_conv()(img, kerns, empty_like(top), desc)
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

        alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
        if self.algo == 'none':  # 3d
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0'
        elif self.algo == 'deterministic':  # 3d
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1'
        elif self.algo == 'fft':
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT'
        elif self.algo == 'fft_tiling':  # 3d
            # big workspace but less than fft
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING'
        elif self.algo == 'winograd':
            alg = 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD'

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

        if kern.type.ndim == 5 and self.algo in ['fft']:
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


def gpu_dnn_conv_gradI(algo=None, inplace=False):
    key = (algo, inplace)
    if key not in gpu_dnn_conv_gradI.cache:
        gpu_dnn_conv_gradI.cache[key] = GpuDnnConvGradI(inplace, algo)
    return gpu_dnn_conv_gradI.cache[key]
gpu_dnn_conv_gradI.cache = {}


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
        require certain versions of cuDNN to be installed. Default is
        the value of :attr:`config.dnn.conv.algo_fwd`.
    precision : {'as_input_f32', 'as_input', 'float16', 'float32', 'float64'}
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
    if precision == 'as_input' or precision == 'as_input_f32':
        nprec = theano.scalar.upcast(img.dtype, kerns.dtype)
        if nprec == 'float16' and precision == 'as_input_f32':
            precision = 'float32'
        else:
            precision = nprec

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
        out_shp = (shape_i(kerns, 1, fgraph),
                   shape_i(img, 1, fgraph),
                   shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
                   shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode='cross', precision=precision)(out.shape)
        conv = gpu_dnn_conv_gradW()(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)

    elif (border_mode == 'full' and subsample == (1, 1) and
          direction_hint != 'forward!'):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        out_shp = (shape_i(img, 0, fgraph),
                   shape_i(kerns, 1, fgraph),
                   shape_i(img, 2, fgraph) + shape_i(kerns, 2, fgraph) - 1,
                   shape_i(img, 3, fgraph) + shape_i(kerns, 3, fgraph) - 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode=conv_mode, precision=precision)(kerns.shape)
        return gpu_dnn_conv_gradI()(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode, precision=precision)(kerns.shape)
    desc_op = desc.owner.op
    # We can use Shape_i and bypass the infer_shape here as this is on
    # the input of node and it will always be present.
    ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
    kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
    out_shp = get_conv_output_shape(ishape, kshape,
                                    desc_op.border_mode,
                                    desc_op.subsample)
    out_shp = assert_conv_shape(out_shp)
    out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
    return gpu_dnn_conv(algo=algo)(img, kerns, out, desc)


def dnn_conv3d(img, kerns, border_mode='valid', subsample=(1, 1, 1),
               conv_mode='conv', direction_hint=None,
               algo='none', precision=None):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc012', that is 'batch', 'channel',
    'first dim', 'second dim', 'third dim' in that order.

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
    algo : convolution implementation to use. Only 'none' is implemented
        for the conv3d. Default is the value of :attr:`config.dnn.conv.algo_fwd`.
    precision : {'as_input_f32', 'as_input', 'float16', 'float32', 'float64'}
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
    if precision == 'as_input' or precision == 'as_input_f32':
        nprec = theano.scalar.upcast(img.dtype, kerns.dtype)
        if nprec == 'float16' and precision == 'as_input_f32':
            precision = 'float32'
        else:
            precision = nprec

    fgraph = getattr(img, 'fgraph', None) or getattr(kerns, 'fgraph', None)
    ctx_name = infer_context_name(img, kerns)
    if (border_mode == 'valid' and subsample == (1, 1, 1) and
            direction_hint == 'bprop weights'):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
        if conv_mode == 'conv':
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
        out_shp = (shape_i(kerns, 1, fgraph),
                   shape_i(img, 1, fgraph),
                   shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
                   shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1,
                   shape_i(img, 4, fgraph) - shape_i(kerns, 4, fgraph) + 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1, 1),
                              conv_mode='cross', precision=precision)(out.shape)
        conv = gpu_dnn_conv_gradW()(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3, 4), ctx_name)

    elif (border_mode == 'full' and subsample == (1, 1, 1) and
          direction_hint != 'forward!'):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        out_shp = (shape_i(img, 0, fgraph),
                   shape_i(kerns, 1, fgraph),
                   shape_i(img, 2, fgraph) + shape_i(kerns, 2, fgraph) - 1,
                   shape_i(img, 3, fgraph) + shape_i(kerns, 3, fgraph) - 1,
                   shape_i(img, 4, fgraph) + shape_i(kerns, 4, fgraph) - 1)
        out_shp = assert_conv_shape(out_shp)
        out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1, 1),
                              conv_mode=conv_mode, precision=precision)(kerns.shape)
        return gpu_dnn_conv_gradI()(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode, precision=precision)(kerns.shape)
    desc_op = desc.owner.op
    # We can use Shape_i and bypass the infer_shape here as this is on
    # the input of node and it will always be present.
    ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
    kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
    out_shp = get_conv_output_shape(ishape, kshape,
                                    desc_op.border_mode,
                                    desc_op.subsample)
    out_shp = assert_conv_shape(out_shp)
    out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*out_shp)
    return gpu_dnn_conv(algo=algo)(img, kerns, out, desc)


def dnn_gradweight(img, topgrad, kerns_shp, border_mode='valid',
                   subsample=(1, 1), conv_mode='conv'):
    """
    TODO: document this
    """
    ctx_name = infer_context_name(img, topgrad)
    img = as_gpuarray_variable(img, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    kerns_shp = as_tensor_variable(kerns_shp)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode)(kerns_shp)
    out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*kerns_shp)
    return gpu_dnn_conv_gradW()(img, topgrad, out, desc)


def dnn_gradweight3d(img, topgrad, kerns_shp, border_mode='valid',
                     subsample=(1, 1, 1), conv_mode='conv'):
    """
    TODO: document this
    """
    ctx_name = infer_context_name(img, topgrad)
    img = as_gpuarray_variable(img, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    img = gpu_contiguous(img)
    topgrad = gpu_contiguous(topgrad)
    kerns_shp = as_tensor_variable(kerns_shp)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode)(kerns_shp)
    out = gpu_alloc_empty(ctx_name, dtype=img.dtype)(*kerns_shp)
    return gpu_dnn_conv_gradW()(img, topgrad, out, desc)


def dnn_gradinput(kerns, topgrad, img_shp, border_mode='valid',
                  subsample=(1, 1), conv_mode='conv'):
    """
    TODO: document this
    """
    ctx_name = infer_context_name(kerns, topgrad)
    kerns = as_gpuarray_variable(kerns, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)
    img_shp = as_tensor_variable(img_shp)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode)(kerns.shape)
    out = gpu_alloc_empty(ctx_name, kerns.dtype)(*img_shp)
    return gpu_dnn_conv_gradI()(kerns, topgrad, out, desc)


def dnn_gradinput3d(kerns, topgrad, img_shp, border_mode='valid',
                    subsample=(1, 1, 1), conv_mode='conv'):
    """
    TODO: document this
    """
    ctx_name = infer_context_name(kerns, topgrad)
    kerns = as_gpuarray_variable(kerns, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    kerns = gpu_contiguous(kerns)
    topgrad = gpu_contiguous(topgrad)
    img_shp = as_tensor_variable(img_shp)
    desc = gpu_dnn_conv_desc(border_mode=border_mode, subsample=subsample,
                             conv_mode=conv_mode)(kerns.shape)
    out = gpu_alloc_empty(ctx_name, kerns.dtype)(*img_shp)
    return gpu_dnn_conv_gradI()(kerns, topgrad, out, desc)


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

    Note
    ----
    Not used anymore. Only needed to reload old pickled files.
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

    err = cudnnSetPoolingNdDescriptor(%(desc)s, %(mode_flag)s, CUDNN_PROPAGATE_NAN, %(nd)d, win, pad, str);

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
        return (4, version())


class GpuDnnPool(DnnBase):

    """
    Parameters
    ----------
    img : tensor
        The image 4d or 5d tensor.
    ws : tensor
        Window size.
    stride : tensor
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tensor
        (padX, padY) or (padX, padY, padZ)

    """
    _f16_ok = True
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

    def L_op(self, inp, outputs, grads):
        img, ws, stride, pad = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        out, = outputs

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
    _f16_ok = True
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


def dnn_pool(img, ws, stride=None, mode='max', pad=None):
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
        Subsampling window size.  Should have 2 or 3 elements.
    stride : tuple
        Subsampling stride (default: (1, 1) or (1, 1, 1)).
    mode : {'max', 'average_inc_pad', 'average_exc_pad', 'sum'}
    pad : tuple
        (padX, padY) or (padX, padY, padZ)
        default: (0, 0) or (0, 0, 0)


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
        context_name = ret.type.context_name
        window_elem = theano.tensor.prod(ws).astype(ret.dtype)
        return as_gpuarray_variable(ret * window_elem, context_name)
    return GpuDnnPool(mode=mode)(img, ws, stride, pad)


class GpuDnnSoftmaxBase(DnnBase):

    """
    Op for the cuDNN Softmax.

    Parameters
    ----------
    algo : {'fast', 'accurate', 'log'}
        Indicating whether, respectively, computations should be optimized for
        speed, for accuracy, or if cuDNN should rather compute the log-softmax instead.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    __props__ = ('mode', 'algo')

    def __init__(self, algo, mode):
        DnnBase.__init__(self, [self.file], self.c_func)

        assert(algo in ('fast', 'accurate', 'log'))
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

    algo : {'fast', 'accurate', 'log'}
        Indicating whether, respectively, computations should be optimized for
        speed, for accuracy, or if cuDNN should rather compute the log-softmax instead.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """
    _f16_ok = True
    direction = "forward"
    file = "dnn_softmax.c"
    c_func = "APPLY_SPECIFIC(softmax)"

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def L_op(self, inp, outputs, grads):
        x, = inp
        g_sm, = grads
        sm, = outputs
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
        computations should be optimized for speed, for accuracy, or if cuDNN
        should rather compute the gradient of the log-softmax instead.
    mode
        'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.

    """
    _f16_ok = True
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


class GpuDnnBatchNorm(DnnBase):
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

    __props__ = ('mode', 'running_averages', 'inplace_running_mean',
                 'inplace_running_var', 'inplace_output')

    def __init__(self, mode='per-activation', running_averages=False,
                 inplace_running_mean=False, inplace_running_var=False,
                 inplace_output=False):
        DnnBase.__init__(self, ['dnn_batchnorm_base.c', 'dnn_batchnorm.c'],
                         'dnn_batchnorm_op')

        assert (mode in ('per-activation', 'spatial'))
        self.mode = mode
        self.running_averages = running_averages
        self.inplace_output = inplace_output
        self.inplace_running_mean = inplace_running_mean
        self.inplace_running_var = inplace_running_var
        self.destroy_map = {}
        if self.inplace_output:
            self.destroy_map[0] = [0]
        if self.running_averages and self.inplace_running_mean:
            self.destroy_map[3] = [5]
        if self.running_averages and self.inplace_running_var:
            self.destroy_map[4] = [6]

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
        params.append(('MODE', ("CUDNN_BATCHNORM_SPATIAL"
                                if self.mode == "spatial"
                                else "CUDNN_BATCHNORM_PER_ACTIVATION")))
        return params

    def infer_shape(self, node, shape):
        return [shape[0]] + [shape[1]] * (len(node.outputs) - 1)

    def make_node(self, x, scale, bias, epsilon=1e-4,
                  running_average_factor=0.1,
                  running_mean=None, running_var=None):
        assert x.ndim == scale.ndim == bias.ndim
        assert x.ndim in (4, 5)
        assert self.running_averages == (running_mean is not None) == (running_var is not None)
        assert (running_mean is None or running_mean.ndim == x.ndim)
        assert (running_var is None or running_var.ndim == x.ndim)
        ctx_name = infer_context_name(x, scale, bias)
        x = as_gpuarray_variable(x, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        bias = as_gpuarray_variable(bias, ctx_name)
        epsilon = as_scalar(epsilon).astype('float64')
        running_average_factor = as_scalar(running_average_factor).astype('float64')
        inputs = [x, scale, bias, epsilon, running_average_factor]
        output_types = [x.type(), scale.type(), scale.type()]
        if running_mean is not None and running_var is not None:
            inputs.append(as_gpuarray_variable(running_mean, ctx_name))
            inputs.append(as_gpuarray_variable(running_var, ctx_name))
            output_types.append(scale.type())
            output_types.append(scale.type())
        return Apply(self, inputs, output_types)

    def L_op(self, inputs, outputs, grads):
        x, scale, bias, epsilon, running_average_factor = inputs[:5]
        dy = grads[0]
        _, x_mean, x_invstd = outputs[:3]
        disconnected_outputs = [
            DisconnectedType()(),  # epsilon
            DisconnectedType()()]  # running_average_factor
        # Optional running_mean and running_var.
        for i in range(5, len(inputs)):
            disconnected_outputs.append(DisconnectedType()())
        return GpuDnnBatchNormGrad(self.mode)(
            x, dy, scale, x_mean, x_invstd, epsilon) + disconnected_outputs

    def connection_pattern(self, node):
        # Specificy that epsilon and running_average_factor are not connected to outputs.
        patterns = [[True, True, True],     # x
                    [True, True, True],     # scale
                    [True, True, True],     # bias
                    [False, False, False],  # epsilon
                    [False, False, False]]  # running_average_factor
        # Optional running_mean and running_var are only
        # connected to their new values.
        for i in range(5, len(node.inputs)):
            patterns[0].append(True)
            for pattern in patterns[1:]:
                pattern.append(False)
            patterns.append([False] * (3 + i - 5) + [True])
        return patterns


class GpuDnnBatchNormInference(DnnBase):
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
    """

    __props__ = ('mode', 'inplace')

    def __init__(self, mode='per-activation', inplace=False):
        DnnBase.__init__(self, ['dnn_batchnorm_base.c', 'dnn_batchnorm_inf.c'],
                         'dnn_batchnorm_op')

        assert (mode in ('per-activation', 'spatial'))
        self.mode = mode
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
        params.append(('MODE', ("CUDNN_BATCHNORM_SPATIAL"
                                if self.mode == "spatial"
                                else "CUDNN_BATCHNORM_PER_ACTIVATION")))
        return params

    def infer_shape(self, node, shape):
        return [shape[0]]

    def make_node(self, x, scale, bias, estimated_mean, estimated_variance, epsilon=1e-4):
        ctx_name = infer_context_name(x, scale, bias, estimated_mean,
                                      estimated_variance)
        x = as_gpuarray_variable(x, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        bias = as_gpuarray_variable(bias, ctx_name)
        estimated_mean = as_gpuarray_variable(estimated_mean, ctx_name)
        estimated_variance = as_gpuarray_variable(estimated_variance, ctx_name)
        epsilon = as_scalar(epsilon).astype('float64')
        assert x.ndim == scale.ndim == bias.ndim == estimated_mean.ndim == estimated_variance.ndim
        assert x.ndim in (4, 5)
        return Apply(self, [x, scale, bias, estimated_mean, estimated_variance, epsilon], [x.type()])

    def grad(self, inputs, grads):
        x, scale, bias, est_mean, est_var, epsilon = inputs
        dy = grads[0]

        if self.mode == "per-activation":
            axes = (0,)
        elif self.mode == "spatial":
            axes = (0,) + tuple(range(2, x.ndim))
        scale, bias, est_mean, est_var = (theano.tensor.addbroadcast(t, *axes)
                                          for t in (scale, bias, est_mean, est_var))

        # define helper expressions
        est_var_eps = est_var + epsilon
        est_std = theano.tensor.sqrt(est_var_eps)
        two = theano.tensor.constant(2.)

        # define and return gradients
        dx = dy * (scale / est_std)
        dscale = (dy * (x - est_mean)).sum(axes, keepdims=True) / est_std
        dbias = dy.sum(axes, keepdims=True)
        dmean = -dy.sum(axes, keepdims=True) * (scale / est_std)
        dvar = -(dy * (x - est_mean)).sum(axes, keepdims=True) * (scale / (two * est_var_eps * est_std))
        return [dx, dscale, dbias, dmean, dvar, DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that epsilon is not connected to outputs.
        return [[True], [True], [True], [True], [True], [False]]


class GpuDnnBatchNormGrad(DnnBase):
    __props__ = ('mode',)

    def __init__(self, mode='per-activation'):
        DnnBase.__init__(self, ['dnn_batchnorm_base.c', 'dnn_batchnorm_grad.c'],
                         'dnn_batchnorm_grad')

        assert (mode in ('per-activation', 'spatial'))
        self.mode = mode

    def get_op_params(self):
        params = []
        params.append(('MODE', ("CUDNN_BATCHNORM_SPATIAL"
                                if self.mode == "spatial"
                                else "CUDNN_BATCHNORM_PER_ACTIVATION")))
        return params

    def make_node(self, x, dy, scale, x_mean, x_invstd, epsilon=1e-4):
        ctx_name = infer_context_name(x, dy, scale, x_mean, x_invstd)
        x = as_gpuarray_variable(x, ctx_name)
        dy = as_gpuarray_variable(dy, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        x_mean = as_gpuarray_variable(x_mean, ctx_name)
        x_invstd = as_gpuarray_variable(x_invstd, ctx_name)
        epsilon = as_scalar(epsilon).astype('float64')
        assert x.ndim == dy.ndim == scale.ndim == x_mean.ndim == x_invstd.ndim
        assert x.ndim in (4, 5)
        return Apply(self, [x, dy, scale, x_mean, x_invstd, epsilon], [x.type(), scale.type(), scale.type()])

    def infer_shape(self, node, shape):
        return [shape[0], shape[2], shape[2]]

gpudata_type = CDataType('gpudata *', 'gpudata_release')
dropoutdesc_type = CDataType('cudnnDropoutDescriptor_t',
                             'cudnnDestroyDropoutDescriptor')


class GpuDnnDropoutOp(DnnBase):
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        DnnBase.__init__(self, ["dnn_dropout_fwd.c"], "dnn_dropout_fwd")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {1: [2]}

    def make_node(self, inp, descriptor, state):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        return Apply(self, [inp, descriptor, state],
                     [inp.type(), state.type(), gpudata_type()])

    def prepare_node(self, node, storage_map, compute_map):
        assert self.inplace, "GpuDnnDropoutOp not inplace"


class _DropoutDescriptor(DnnBase):
    __props__ = ('context_name',)

    def __init__(self, context_name):
        DnnBase.__init__(self, ["dnn_dropout_desc.c"], "dnn_dropout_desc")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, node):
        return False

    def make_node(self, dropout, seed, context_name):
        dropout = as_scalar(dropout).astype('float32')
        seed = as_scalar(seed).astype('uint64')

        assert context_name == self.context_name
        # This is a dirty hack to pass the context because params is
        # occupied by the cudnn handle
        context = gpu_context_type.make_constant(get_context(context_name))

        return Apply(self, [dropout, seed, context],
                     [dropoutdesc_type(),
                      GpuArrayType('uint8', (False,),
                                   context_name=context_name)()])

    def c_code_cache_version_apply(self, node):
        # disable the cache since we can't pickle contexts
        return None


def _make_dropout_desc(dropout, seed, context_name):
    desc, states = theano.function(
        [],
        _DropoutDescriptor(context_name)(dropout, seed, context_name),
        theano.Mode(optimizer=None),
        profile=False)()
    return desc, states


def dropout(x, dropout=0.0, seed=4242):
    desc, states = _make_dropout_desc(dropout, seed, x.type.context_name)
    y, odesc = GpuDnnDropoutOp()(x, desc)
    return y, desc, odesc, states

rnndesc_type = CDataType('cudnnRNNDescriptor_t',
                         'cudnnDestroyRNNDescriptor')


def as_i32(v):
    return as_scalar(v).astype('int32')


class _RNNDescriptor(DnnBase):
    __props__ = ('context_name',)

    def __init__(self, context_name):
        if version() < 5005:
            raise RuntimeError("cudnn RNN require cudnn v5 final or higher.")
        DnnBase.__init__(self, ["dnn_rnn_desc.c"], "dnn_rnn_desc")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, node):
        return False

    def make_node(self, hidden_size, num_layers, ddesc, input_mode,
                  direction_mode, rnn_mode, dtype):

        hidden_size = as_i32(hidden_size)
        num_layers = as_i32(num_layers)

        assert 5000 < version() < 5200, "Constants only work for cudnn 5, 5.1"

        if input_mode == 'linear':
            input_mode = as_i32(0)
        elif input_mode == 'skip':
            input_mode = as_i32(1)
        else:
            raise ValueError("input_mode")

        if direction_mode == 'unidirectional':
            direction_mode = as_i32(0)
        elif direction_mode == 'bidirectional':
            direction_mode = as_i32(1)
        else:
            raise ValueError("direction_mode")

        if rnn_mode == 'rnn_relu':
            rnn_mode = as_i32(0)
        elif rnn_mode == 'rnn_tanh':
            rnn_mode = as_i32(1)
        elif rnn_mode == 'lstm':
            rnn_mode = as_i32(2)
        elif rnn_mode == 'gru':
            rnn_mode = as_i32(3)
        else:
            raise ValueError("rnn_mode")

        dtype = as_i32(gpuarray.dtype_to_typecode(dtype))

        return Apply(self, [hidden_size, num_layers,
                            dropoutdesc_type.make_constant(ddesc),
                            input_mode, direction_mode, rnn_mode, dtype],
                     [rnndesc_type()])


def _make_rnn_desc(hidden_size, num_layers, ddesc, rnn_mode,
                   input_mode, direction_mode, dtype, context_name):
    desc = theano.function(
        [],
        _RNNDescriptor(context_name)(hidden_size, num_layers, ddesc,
                                     input_mode, direction_mode,
                                     rnn_mode, dtype),
        theano.Mode(optimizer=None),
        profile=False)()
    return desc


class _RNNParamSize(DnnBase):
    __props__ = ('context_name',)

    def __init__(self, context_name):
        DnnBase.__init__(self, ["dnn_rnn_paramsize.c"],
                         "dnn_rnn_paramsize")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, node):
        return False

    def make_node(self, desc, input_size, typecode):
        input_size = as_tensor_variable(input_size).astype('uint64')
        typecode = as_i32(typecode)
        return Apply(self, [rnndesc_type.make_constant(desc), input_size,
                            typecode],
                     [get_scalar_type('uint64')()])


def _get_param_size(desc, input_size, dtype, context_name):
    typecode = gpuarray.dtype_to_typecode(dtype)
    return theano.function(
        [],
        _RNNParamSize(context_name)(desc, input_size, typecode),
        theano.Mode(optimizer=None),
        profile=False)()


class _RNNSplitParams(DnnBase):
    __props__ = ('rnn_mode',)

    def __init__(self, rnn_mode):
        DnnBase.__init__(self)
        self.rnn_mode = rnn_mode

    def make_node(self, w, desc, layer, isize, typecode):
        w = as_gpuarray_variable(w, infer_context_name(w))
        assert w.ndim == 1
        layer = as_scalar(layer).astype('int32')
        isize = as_tensor_variable(isize).astype('uint64')
        assert isize.ndim == 1
        typecode = as_scalar(typecode).astype('int32')
        _1d = GpuArrayType(w.type.dtype, [False],
                           context_name=w.type.context_name)
        _2d = GpuArrayType(w.type.dtype, [False, False],
                           context_name=w.type.context_name)
        outputs = []
        if self.rnn_mode == 'rnn_relu' or self.rnn_mode == 'rnn_tanh':
            outputs.extend([_2d(), _1d()])  # input
            outputs.extend([_2d(), _1d()])  # recurrent
        elif self.rnn_mode == 'lstm':
            outputs.extend([_2d(), _1d()])  # input input
            outputs.extend([_2d(), _1d()])  # input forget
            outputs.extend([_2d(), _1d()])  # input newmem
            outputs.extend([_2d(), _1d()])  # input output
            outputs.extend([_2d(), _1d()])  # recur input
            outputs.extend([_2d(), _1d()])  # recur forget
            outputs.extend([_2d(), _1d()])  # recur newmem
            outputs.extend([_2d(), _1d()])  # recur output
        elif self.rnn_mode == 'gru':
            outputs.extend([_2d(), _1d()])  # input reset
            outputs.extend([_2d(), _1d()])  # input update
            outputs.extend([_2d(), _1d()])  # input newmem
            outputs.extend([_2d(), _1d()])  # recur reset
            outputs.extend([_2d(), _1d()])  # recur update
            outputs.extend([_2d(), _1d()])  # recur newmem

        return Apply(self, [w, layer, rnndesc_type.make_constant(desc),
                            isize, typecode], outputs)

    def c_code(self, node, name, inputs, outputs, sub):
        kw = dict(fail=sub['fail'], w=inputs[0], layer=inputs[1],
                  desc=inputs[2], isize=inputs[3], typecode=inputs[4],
                  handle=sub['params'])
        code = """
  cudnnTensorDescriptor_t xdesc;
  cudnnFilterDescriptor_t wdesc;
  cudnnFilterDescriptor_t odesc;
  size_t nshp[2];
  void *w;
  void *o;
  ptrdiff_t off;
  size_t bshp;
  cudnnStatus_t err;
  cudnnDataType_t dt;
  cudnnTensorFormat_t tf;
  int nd;
  int dims[3];
  int strs[3];

  if (PyArray_DIM(%(isize)s, 0) != 2) {
    PyErr_SetString(PyExc_ValueError, "input_size should be of length two");
    %(fail)s;
  }

  switch (%(typecode)s) {
  case GA_FLOAT:
    dt = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    dt = CUDNN_DATA_DOUBLE;
    break;
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "Unsupported data type");
    %(fail)s;
  }

  err = cudnnCreateTensorDescriptor(&xdesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create xdesc");
    %(fail)s;
  }

  dims[0] = *(npy_uint64 *)PyArray_GETPTR1(%(isize)s, 0);
  dims[1] = *(npy_uint64 *)PyArray_GETPTR1(%(isize)s, 1);
  dims[2] = 1;
  strs[0] = dims[2] * dims[1];
  strs[1] = dims[2];
  strs[2] = 1;

  err = cudnnSetTensorNdDescriptor(xdesc, dt, 3, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    PyErr_Format(PyExc_RuntimeError, "Could not set xdesc: %%s",
                 cudnnGetErrorString(err));
    %(fail)s;
  }

  if (c_make_filter(%(w)s, &wdesc)) {
    cudnnDestroyTensorDescriptor(xdesc);
    %(fail)s
  }

  err = cudnnCreateFilterDescriptor(&odesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "could not create odesc");
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    %(fail)s
  }

  w = PyGpuArray_DEV_DATA(%(w)s);
  nshp[0] = PyGpuArray_DIM(%(w)s, 0);
  nshp[1] = 1;
        """ % kw

        def get_params(id, m, b):
            kw2 = kw.copy()
            kw2['id'] = id
            kw2['m'] = m
            kw2['b'] = b
            return """
  err = cudnnGetRNNLinLayerBiasParams(%(handle)s, %(desc)s, %(layer)s, xdesc, wdesc, w, %(id)s, odesc, &o);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "can't fetch bias for id %(id)s");
    %(fail)s
  }
  off = (intptr_t)o - (intptr_t)w;
  assert(off >= 0 && "bias");

  err = cudnnGetFilterNdDescriptor(odesc, 3, &dt, &tf, &nd, dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "could not get bias shape for id %(id)s");
    %(fail)s;
  }
  // We assume that the typecode matches
  assert(dims[2] == 1);
  assert(dims[1] == 1);
  %(b)s = pygpu_view(%(w)s, Py_None);
  %(b)s->ga.offset = off;
  %(b)s->ga.dimensions[0] = dims[0];
  GpuArray_fix_flags(&%(b)s->ga);
  bshp = dims[0];

  err = cudnnGetRNNLinLayerMatrixParams(%(handle)s, %(desc)s, %(layer)s, xdesc, wdesc, w, %(id)s, odesc, &o);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "can't fetch matrix for id %(id)s");
    %(fail)s
  }
  off = (intptr_t)o - (intptr_t)w;
  assert(off >= 0 && "matrix");

  // This is 3d because of cudnn limitations.
  err = cudnnGetFilterNdDescriptor(odesc, 3, &dt, &tf, &nd, dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "could not get matrix shape for id %(id)s");
    %(fail)s;
  }

  assert(dims[1] == 1);
  assert(dims[2] == 1);
  // We assume that the typecode matches
  %(m)s = pygpu_reshape(%(w)s, 2, nshp, GA_F_ORDER, 1, -1);
  %(m)s->ga.offset = off;
  assert(dims[0] %% bshp == 0);
  %(m)s->ga.dimensions[0] = dims[0] / bshp;
  %(m)s->ga.dimensions[1] = bshp;
  %(m)s->ga.strides[1] = %(m)s->ga.dimensions[0] * gpuarray_get_elsize(%(m)s->ga.typecode);
  GpuArray_fix_flags(&%(m)s->ga);
            """ % kw2

        for i in range(len(outputs) // 2):
            code += get_params(i, outputs[2 * i], outputs[(2 * i) + 1])

        code += """
  cudnnDestroyTensorDescriptor(xdesc);
  cudnnDestroyFilterDescriptor(wdesc);
  cudnnDestroyFilterDescriptor(odesc);
        """
        return code

    def c_code_cache_version(self):
        return (3,)


def _split_rnn_params(w, desc, layer, input_size, dtype, rnn_mode):
    typecode = gpuarray.dtype_to_typecode(dtype)
    outs = _RNNSplitParams(rnn_mode)(w, desc, layer, input_size, typecode)
    outs = [theano.Out(o, borrow=True) for o in outs]
    return theano.function(
        [], outs,
        theano.Mode(optimizer=None),
        profile=False)()


class GpuDnnRNNOp(DnnBase):
    __props__ = ()
    _cop_num_inputs = 6
    _cop_num_outputs = 4

    def __init__(self, rnn_mode, direction_mode):
        DnnBase.__init__(self, ["dnn_rnn_fwd.c"], 'dnn_rnn_fwd')
        self.rnn_mode = rnn_mode
        if direction_mode == 'bidirectional':
            self.num_dirs = 2
        elif direction_mode == 'unidirectional':
            self.num_dirs = 1
        else:
            raise ValueError('direction_mode is invalid (got %s)' % (direction_mode,))

    def dnn_context(self, node):
        return node.outputs[1].type.context_name

    def make_node(self, desc, w, x, hx, cx=None):
        if cx is None:
            context_name = infer_context_name(w, x, hx)
        else:
            context_name = infer_context_name(w, x, hx, cx)

        w = as_gpuarray_variable(w, context_name)
        x = as_gpuarray_variable(x, context_name)
        hx = as_gpuarray_variable(hx, context_name)
        inputs = [desc, as_i32(self.num_dirs), w, x, hx]
        assert w.ndim == 1
        assert x.ndim == 3  # seqLength, minibatch, inputSize
        assert hx.ndim == 3  # numLayers, minibatch, hiddenSize * bidi

        if self.rnn_mode == 'lstm':
            cx = as_gpuarray_variable(cx, context_name)
            assert cx.ndim == 3  # numLayers, minibatch, hiddenSize * bidi
            inputs.append(cx)

        _3d = GpuArrayType(dtype=x.dtype, broadcastable=(False, False, False),
                           context_name=context_name)
        reserve = gpudata_type()
        y = _3d()  # seqLength, minibatch, hiddenSize * bidi
        hy = _3d()  # numLayers, miniBatch, hiddenSize * bidi
        outputs = [reserve, y, hy]

        if self.rnn_mode == 'lstm':
            cy = _3d()  # numLayers, miniBatch, hiddenSize * bidi
            outputs.append(cy)

        return Apply(self, inputs, outputs)

    def L_op(self, inputs, outputs, output_grads):
        desc, numDirs, w, x, hx = inputs[:5]
        cx = inputs[5] if len(inputs) == 6 else None
        reserve, y, hy = outputs[:3]
        _, dy, dhy = output_grads[:3]
        dcy = output_grads[3] if len(output_grads) == 4 else None
        # Since the op return two outputs which contain essentially
        # the same information, the user will most likely only use one
        # of them. This leads to the situation that the other is
        # considered "disconnected" by theano in the gradient.
        # However we know that this isn't really the case so we fix it
        # here.

        # If all the ys are disconnected, then you get a boring
        # gradient instead of an error.  But in that case you
        # shouldn't call this method anyway.
        if isinstance(dy.type, DisconnectedType):
            dy = as_gpuarray_variable(y.zeros_like(),
                                      context_name=y.type.context_name)
        if isinstance(dhy.type, DisconnectedType):
            dhy = None
        if dcy and isinstance(dcy.type, DisconnectedType):
            dcy = None
        dinputs = GpuDnnRNNGradInputs(rnn_mode=self.rnn_mode,
                                      grad_h=(dhy is not None),
                                      grad_c=(dcy is not None))(
            desc, x, y, dy, dhy, dcy, w, hx, cx, reserve, return_list=True)
        reserve2, dx, dhx = dinputs[:3]
        dw = GpuDnnRNNGradWeights()(
            desc, x, hx, y, reserve2, w)
        res = [DisconnectedType()(), DisconnectedType()(), dw, dx, dhx]
        if cx is not None:
            res.append(dinputs[3])  # dcx
        return res

    def connection_pattern(self, node):
        deconn = [[False] * len(node.outputs)] * 2
        conn = [[True] * len(node.outputs)] * (len(node.inputs) - 2)
        return deconn + conn


class GpuDnnRNNGradInputs(DnnBase):
    __props__ = ('rnn_mode', 'grad_c', 'grad_h')
    _cop_num_inputs = 10
    _cop_num_outputs = 4

    def __init__(self, rnn_mode, grad_h, grad_c):
        DnnBase.__init__(self, ['dnn_rnn_gi.c'], 'dnn_rnn_gi')
        self.rnn_mode = rnn_mode
        self.grad_h = grad_h
        self.grad_c = grad_c
        if self.grad_c:
            assert self.rnn_mode == 'lstm'

    def dnn_context(self, node):
        return node.outputs[1].type.context_name

    def make_node(self, desc, x, y, dy, dhy, dcy, w, hx, cx, reserve):
        # We trust the callers here
        xshp = as_scalar(x.shape[2]).astype('uint64')
        inputs = [desc, xshp, y, dy, w, hx, reserve]
        outputs = [reserve.type(), x.type(), hx.type()]
        if self.rnn_mode == 'lstm':
            inputs.append(cx)
            outputs.append(cx.type())
        if self.grad_h:
            inputs.append(dhy)
        if self.grad_c:
            inputs.append(dcy)

        return Apply(self, inputs, outputs)

    # We have special requirements so this is hooking into COp
    def format_c_function_args(self, inp, out):
        rinp = inp[:7]
        others = inp[7:]
        if self.rnn_mode == 'lstm':
            rinp.append(others.pop(0))
        else:
            rinp.append('NULL')
        if self.grad_h:
            rinp.append(others.pop(0))
        else:
            rinp.append('NULL')
        if self.grad_c:
            rinp.append(others.pop(0))
        else:
            rinp.append('NULL')
        assert len(others) == 0
        return COp.format_c_function_args(self, rinp, out)


class GpuDnnRNNGradWeights(DnnBase):
    __props__ = ()

    def __init__(self):
        DnnBase.__init__(self, ['dnn_rnn_gw.c'], 'dnn_rnn_gw')

    def make_node(self, desc, x, hx, y, reserve, w):
        # We trust the callers here
        wsize = as_scalar(w.shape[0]).astype('uint64')
        inputs = [desc, wsize, x, hx, y, reserve]
        outputs = [w.type()]
        return Apply(self, inputs, outputs)


class RNNBlock(object):
    """
    An object that allow us to use CuDNN v5 RNN implementation.
    TODO: make an example how to use. You can check Theano tests
    test_dnn_rnn_gru() and test_dnn_rnn_lstm() in the file
    theano/gpuarray/tests/test_dnn.py for now.


    Parameters
    ----------
    dtype : data type of computation
    hidden_size : int
    num_layers : int
    rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
        See cudnn documentation for ``cudnnRNNMode_t``.

    input_mode : {'linear', 'skip'}
        linear: input will be multiplied by a biased matrix
        skip: No operation is performed on the input.  The size must match the hidden size.
    direction_mode : {'unidirectional', 'bidirectional'}
        unidirectional: The network operates recurrently from the
                        first input to the last.

        bidirectional: The network operates from first to last then from last to first and concatenates the results at each layer.

    """

    def __init__(self, dtype, hidden_size, num_layers, rnn_mode,
                 input_mode='linear', direction_mode='unidirectional',
                 context_name=None):
        # This is not supported for any value other than 0, so don't change it
        ddesc, states = _make_dropout_desc(0, 4242, context_name)
        self.ddesc = ddesc
        self.dstates = states
        self.desc = _make_rnn_desc(hidden_size, num_layers,
                                   ddesc, rnn_mode, input_mode,
                                   direction_mode, dtype, context_name)
        self.rnn_mode = rnn_mode
        self.direction_mode = direction_mode
        self.context_name = context_name
        self.dtype = dtype

    def get_param_size(self, input_size):
        bytesize = _get_param_size(self.desc, input_size, self.dtype,
                                   self.context_name)
        bytesize = int(bytesize)
        assert bytesize % np.dtype(self.dtype).itemsize == 0
        return bytesize // np.dtype(self.dtype).itemsize

    def split_params(self, w, layer, input_size):
        if not isinstance(w, GpuArraySharedVariable):
            raise TypeError("split_params only works on gpuarray shared variables")
        return _split_rnn_params(w, self.desc, layer, input_size, self.dtype, self.rnn_mode)

    def apply(self, w, x, hx, cx=None):
        # Don't return the reserve as an output
        return GpuDnnRNNOp(self.rnn_mode, self.direction_mode)(
            rnndesc_type.make_constant(self.desc),
            w, x, hx, cx, return_list=True)[1:]


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
    Requires cuDNN 5 and Theano 0.9dev2 or more recent.

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

    running_averages = (running_mean is not None and running_var is not None)

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

    batchnorm_op = GpuDnnBatchNorm(mode=mode, running_averages=running_averages)
    if running_averages:
        out, mean, invstd, new_running_mean, new_running_var = batchnorm_op(
            gpu_contiguous(inputs), gpu_contiguous(gamma),
            gpu_contiguous(beta), epsilon=epsilon,
            running_average_factor=running_average_factor,
            running_mean=gpu_contiguous(running_mean),
            running_var=gpu_contiguous(running_var))
        if new_running_mean.broadcastable != running_mean.broadcastable:
            new_running_mean = tensor.patternbroadcast(new_running_mean, running_mean.broadcastable)
        if new_running_var.broadcastable != running_var.broadcastable:
            new_running_var = tensor.patternbroadcast(new_running_var, running_var.broadcastable)
        result = (out, mean, invstd, new_running_mean, new_running_var)
    else:
        result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
                              gpu_contiguous(beta), epsilon=epsilon)
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
    Requires cuDNN 5 and Theano 0.9dev2 or more recent.

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
    batchnorm_op = GpuDnnBatchNormInference(mode=mode)
    result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
                          gpu_contiguous(beta), gpu_contiguous(mean),
                          gpu_contiguous(var), epsilon=epsilon)
    if ndim < 4:
        result = theano.tensor.flatten(result, ndim)
    elif ndim > 5:
        result = theano.tensor.reshape(result, inputs_shape)
    return result


def local_abstractconv_cudnn_graph(op, context_name, inputs, outputs):
    if (not isinstance(op, (AbstractConv2d,
                            AbstractConv2d_gradWeights,
                            AbstractConv2d_gradInputs))):
        return

    if (op.filter_dilation != (1, 1)):
        return None

    inp1 = inputs[0]
    inp2 = inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    if op.filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'

    if isinstance(op, AbstractConv2d):
        rval = dnn_conv(inp1, inp2,
                        border_mode=op.border_mode,
                        subsample=op.subsample,
                        direction_hint='forward!',
                        conv_mode=conv_mode)
    elif isinstance(op, AbstractConv2d_gradWeights):
        shape = (inp2.shape[1], inp1.shape[1],
                 inputs[2][0], inputs[2][1])
        rval = dnn_gradweight(inp1, inp2, shape,
                              border_mode=op.border_mode,
                              subsample=op.subsample,
                              conv_mode=conv_mode)
    elif isinstance(op, AbstractConv2d_gradInputs):
        shape = (inp2.shape[0], inp1.shape[1],
                 inputs[2][0], inputs[2][1])
        rval = dnn_gradinput(inp1, inp2, shape,
                             border_mode=op.border_mode,
                             subsample=op.subsample,
                             conv_mode=conv_mode)
    return [rval]


def local_abstractconv3d_cudnn_graph(op, context_name, inputs, outputs):
    if (not isinstance(op, (AbstractConv3d,
                            AbstractConv3d_gradWeights,
                            AbstractConv3d_gradInputs))):
        return

    if (op.filter_dilation != (1, 1, 1)):
        return None

    inp1 = inputs[0]
    inp2 = inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    if op.filter_flip:
        conv_mode = 'conv'
    else:
        conv_mode = 'cross'

    if isinstance(op, AbstractConv3d):
        rval = dnn_conv3d(inp1, inp2,
                          border_mode=op.border_mode,
                          subsample=op.subsample,
                          direction_hint='forward!',
                          conv_mode=conv_mode)
    elif isinstance(op, AbstractConv3d_gradWeights):
        shape = (inp2.shape[1], inp1.shape[1],
                 inputs[2][0], inputs[2][1], inputs[2][2])
        rval = dnn_gradweight3d(inp1, inp2, shape,
                                border_mode=op.border_mode,
                                subsample=op.subsample,
                                conv_mode=conv_mode)
    elif isinstance(op, AbstractConv3d_gradInputs):
        shape = (inp2.shape[0], inp1.shape[1],
                 inputs[2][0], inputs[2][1], inputs[2][2])
        rval = dnn_gradinput3d(inp1, inp2, shape,
                               border_mode=op.border_mode,
                               subsample=op.subsample,
                               conv_mode=conv_mode)
    return [rval]


@local_optimizer([AbstractConv2d, AbstractConv3d])
def local_abstractconv_cudnn(node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if isinstance(node.op, AbstractConv2d):
        return local_abstractconv_cudnn_graph(node.op, ctx, node.inputs, node.outputs)
    elif isinstance(node.op, AbstractConv3d):
        return local_abstractconv3d_cudnn_graph(node.op, ctx, node.inputs, node.outputs)


@local_optimizer([AbstractConv2d_gradWeights, AbstractConv3d_gradWeights])
def local_abstractconv_gw_cudnn(node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if isinstance(node.op, AbstractConv2d_gradWeights):
        return local_abstractconv_cudnn_graph(node.op, ctx, node.inputs, node.outputs)
    elif isinstance(node.op, AbstractConv3d_gradWeights):
        return local_abstractconv3d_cudnn_graph(node.op, ctx, node.inputs, node.outputs)


@local_optimizer([AbstractConv2d_gradInputs, AbstractConv3d_gradInputs])
def local_abstractconv_gi_cudnn(node):
    ctx = infer_context_name(*node.inputs)
    if not isinstance(node.inputs[0].type, GpuArrayType):
        return
    if isinstance(node.op, AbstractConv2d_gradInputs):
        return local_abstractconv_cudnn_graph(node.op, ctx, node.inputs, node.outputs)
    elif isinstance(node.op, AbstractConv3d_gradInputs):
        return local_abstractconv3d_cudnn_graph(node.op, ctx, node.inputs, node.outputs)


@inplace_allocempty(GpuDnnConv, 2)
def local_dnn_conv_inplace(node, inputs):
    return [gpu_dnn_conv(algo=node.op.algo, inplace=True)(*inputs)]


@inplace_allocempty(GpuDnnConvGradW, 2)
def local_dnn_convgw_inplace(node, inputs):
    return [gpu_dnn_conv_gradW(algo=node.op.algo, inplace=True)(*inputs)]


@inplace_allocempty(GpuDnnConvGradI, 2)
def local_dnn_convgi_inplace(node, inputs):
    return [gpu_dnn_conv_gradI(algo=node.op.algo, inplace=True)(*inputs)]

optdb.register('local_dnna_conv_inplace',
               tensor.opt.in2out(local_dnn_conv_inplace,
                                 local_dnn_convgw_inplace,
                                 local_dnn_convgi_inplace,
                                 name="local_dnna_conv_inplace"),
               70.0, 'fast_run', 'inplace', 'gpuarray', 'cudnn')


@register_opt('cudnn')
@alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5)
def local_dnn_conv_alpha_merge(node, *inputs):
    return [gpu_dnn_conv(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5)
def local_dnn_convw_alpha_merge(node, *inputs):
    return [gpu_dnn_conv_gradW(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5)
def local_dnn_convi_alpha_merge(node, *inputs):
    return [gpu_dnn_conv_gradI(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_conv_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [gpu_dnn_conv(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convw_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [gpu_dnn_conv_gradW(algo=node.op.algo)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2)
def local_dnn_convi_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [gpu_dnn_conv_gradI(algo=node.op.algo)(*inputs)]


def local_gpua_pool_dnn_alternative(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    img, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    img = gpu_contiguous(as_gpuarray_variable(img, ctx_name))
    mode = op.mode
    # dnn_pool expects exactly 2 non-pooling dimensions
    if img.ndim == nd + 2:
        return dnn_pool(img, ws, stride=stride, pad=pad, mode=mode)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        img_padded = pad_dims(img, 2, nd)
        ret_padded = dnn_pool(img_padded, ws, stride=stride, pad=pad, mode=mode)
        return unpad_dims(ret_padded, img, 2, nd)
pool_db.register("local_gpua_pool_dnn_alternative",
                 op_lifter([Pool])(local_gpua_pool_dnn_alternative),
                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                 position=0)
pool_db2.register("local_gpua_pool_dnn_alternative",
                  local_optimizer([Pool])(local_gpua_pool_dnn_alternative),
                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                  position=0)


def local_gpua_pool_dnn_grad_stride(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    inp, out, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
    mode = op.mode

    # the GPU ops expect exactly 2 non-pooling dimensions
    if inp.ndim == nd + 2:
        return GpuDnnPoolGrad(mode=mode)(inp,
                                         out,
                                         out_grad,
                                         ws,
                                         stride,
                                         pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_padded = pad_dims(out, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
                                               out_padded,
                                               out_grad_padded,
                                               ws,
                                               stride,
                                               pad)
        return unpad_dims(ret_padded, inp, 2, nd)
pool_db.register("local_gpua_pool_dnn_grad_stride",
                 op_lifter([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                 position=0)
pool_db2.register("local_gpua_pool_dnn_grad_stride",
                  local_optimizer([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                  position=0)


def local_gpua_avg_pool_dnn_grad_stride(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    if not op.ignore_border:
        return
    inp, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
    mode = op.mode

    # the GPU ops expect exactly 2 non-pooling dimensions
    if inp.ndim == nd + 2:
        # We reuse out_grad because cuDNN does not use the value of the `out`
        # argument but still checks its shape for average pooling. This
        # has been observed in v2 and v3 as far as I know.
        return GpuDnnPoolGrad(mode=mode)(inp, out_grad, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
                                               out_grad_padded,
                                               out_grad_padded,
                                               ws,
                                               stride,
                                               pad)
        return unpad_dims(ret_padded, inp, 2, nd)
pool_db.register("local_gpua_avg_pool_dnn_grad_stride",
                 op_lifter([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                 position=0)
pool_db2.register("local_gpua_avg_pool_dnn_grad_stride",
                  local_optimizer([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
                  position=0)


@register_opt('cudnn', 'fast_compile')
@local_optimizer([GpuSoftmax])
def local_softmax_dnn(node):
    if isinstance(node.op, GpuSoftmax):
        if not dnn_available(node.outputs[0].type.context_name):
            return
        ins = node.inputs[0].dimshuffle(0, 1, 'x', 'x')
        ins = gpu_contiguous(ins)
        out = GpuDnnSoftmax('accurate', 'channel')(ins)
        out = as_gpuarray_variable(out.dimshuffle(0, 1), out.type.context_name)
        return [out]


@register_opt('cudnn', 'stabilize')
@local_optimizer([GpuElemwise])
def local_log_softmax_dnn(node):
    # This looks for GpuDnnSoftmax so we know that we have cudnn.
    if (isinstance(node.op, GpuElemwise) and
            isinstance(node.op.scalar_op, Log) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, GpuDnnSoftmax) and
            len(node.inputs[0].clients) == 1):
        softmax_node = node.inputs[0].owner
        new_softmax = GpuDnnSoftmax('log', softmax_node.op.mode)
        return [new_softmax(softmax_node.inputs[0])]


@register_opt('cudnn', 'fast_compile')
@op_lifter([LogSoftmax])
@register_opt2([LogSoftmax], 'fast_compile', 'cudnn')
def local_gpua_logsoftmax_to_dnn(op, ctx_name, inputs, outputs):
    # Transform the input in the format expected by GpuDnnSoftmax
    inp = inputs[0]
    if inp.ndim != 2:
        return
    if not dnn_available(ctx_name):
        return

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
                    "to use it for context " + str(c) + ". We got this error: \n" +
                    dnn_available.msg)

gpu_seqopt.register("NoCuDNNRaise", NoCuDNNRaise(), 0, 'cudnn')


@register_opt('cudnn', 'fast_compile')
@op_lifter([SoftmaxGrad])
@register_opt2([SoftmaxGrad], 'cudnn', 'fast_compile')
def local_gpua_softmax_dnn_grad(op, ctx_name, inputs, outputs):
    if not dnn_available(ctx_name):
        return
    ins = []
    for n in inputs:
        n = as_gpuarray_variable(n, ctx_name)
        if n.ndim != 2:
            return
        ins.append(n.dimshuffle(0, 'x', 1, 'x'))

    out = GpuDnnSoftmaxGrad('accurate', 'instance')(
        gpu_contiguous(ins[0]), gpu_contiguous(ins[1]))
    return [out.dimshuffle(0, 2)]


def local_abstract_batch_norm_train_cudnn(op, ctx_name, inputs, outputs):
    x, scale, bias, epsilon, running_average_factor = inputs[:5]
    running_mean = inputs[5] if len(inputs) > 5 else None
    running_var = inputs[6] if len(inputs) > 6 else None

    # convert axes to cuDNN mode
    axes = tuple(op.axes)
    if axes == (0,):
        mode = 'per-activation'
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = 'spatial'
    else:
        return None

    try:
        eps = theano.tensor.get_scalar_constant_value(epsilon)
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None
    try:
        running_average_factor = theano.tensor.get_scalar_constant_value(running_average_factor)
    except theano.tensor.NotScalarConstantError:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    bias = as_gpuarray_variable(bias, context_name=ctx)

    inputs = [x, scale, bias, mode, eps, running_average_factor]
    if running_mean is not None and running_var is not None:
        inputs.append(running_mean)
        inputs.append(running_var)

    results = list(dnn_batch_normalization_train(*inputs))

    return results


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_output(node):
    if isinstance(node.op, GpuDnnBatchNorm) and not node.op.inplace_output:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=node.op.inplace_running_mean,
                               inplace_running_var=node.op.inplace_running_var,
                               inplace_output=True)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_running_mean(node):
    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_mean:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=True,
                               inplace_running_var=node.op.inplace_running_var,
                               inplace_output=node.op.inplace_output)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNorm], inplace=True)
def local_batch_norm_inplace_running_var(node):
    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_var:
        return GpuDnnBatchNorm(mode=node.op.mode,
                               running_averages=node.op.running_averages,
                               inplace_running_mean=node.op.inplace_running_mean,
                               inplace_running_var=True,
                               inplace_output=node.op.inplace_output)(*node.inputs)


@register_inplace()
@local_optimizer([GpuDnnBatchNormInference], inplace=True)
def local_batch_norm_inference_inplace(node):
    if isinstance(node.op, GpuDnnBatchNormInference) and not node.op.inplace:
        return [GpuDnnBatchNormInference(mode=node.op.mode, inplace=True)(*node.inputs)]


def local_abstract_batch_norm_train_grad_cudnn(op, ctx_name, inputs, outputs):
    x, dy, scale, x_mean, x_invstd, epsilon = inputs

    # input on gpu?  TODO what about the output?
    x_on_gpu = (isinstance(x.type, GpuArrayType) or
                (x.owner and isinstance(x.owner.op, HostFromGpu)))
    dy_on_gpu = (isinstance(dy.type, GpuArrayType) or
                 (dy.owner and isinstance(dy.owner.op, HostFromGpu)))
    if not (x_on_gpu or dy_on_gpu):
        return None

    # convert axes to cuDNN mode
    axes = tuple(op.axes)
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
        eps = theano.tensor.get_scalar_constant_value(epsilon)
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    dy = as_gpuarray_variable(dy, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    x_mean = as_gpuarray_variable(x_mean, context_name=ctx)
    x_invstd = as_gpuarray_variable(x_invstd, context_name=ctx)

    g_wrt_inputs, g_wrt_scale, g_wrt_bias = \
        GpuDnnBatchNormGrad(mode)(x, dy, scale, x_mean, x_invstd, eps)

    if ndim < 4:
        g_wrt_inputs = theano.tensor.flatten(g_wrt_inputs, ndim)
        g_wrt_scale = theano.tensor.flatten(g_wrt_scale, ndim)
        g_wrt_bias = theano.tensor.flatten(g_wrt_bias, ndim)
    elif ndim > 5:
        g_wrt_inputs = theano.tensor.reshape(g_wrt_inputs, x_shape)
        g_wrt_scale = theano.tensor.reshape(g_wrt_scale, params_shape)
        g_wrt_bias = theano.tensor.reshape(g_wrt_bias, params_shape)

    return [g_wrt_inputs, g_wrt_scale, g_wrt_bias]


def local_abstract_batch_norm_inference_cudnn(op, ctx_name, inputs, outputs):
    x, scale, bias, estimated_mean, estimated_variance, epsilon = inputs

    axes = tuple(op.axes)
    if axes == (0,):
        mode = 'per-activation'
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = 'spatial'
    else:
        return None

    try:
        eps = theano.tensor.get_scalar_constant_value(epsilon)
    except theano.tensor.NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    bias = as_gpuarray_variable(bias, context_name=ctx)
    estimated_mean = as_gpuarray_variable(estimated_mean, context_name=ctx)
    estimated_variance = as_gpuarray_variable(estimated_variance, context_name=ctx)

    out = dnn_batch_normalization_test(x, scale, bias, estimated_mean, estimated_variance,
                                       mode, eps)

    return [out]
