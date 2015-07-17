import os
import numpy

import theano
from theano import Op, Apply, tensor, config, Variable
from theano.scalar import as_scalar, constant
from theano.gradient import DisconnectedType, grad_not_implemented
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.cmodule import GCC_compiler
from theano.gof.type import CDataType, Generic
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.configparser import AddConfigVar, EnumStr, StrParam
from theano.tensor.nnet import SoftmaxGrad
from theano.tensor.signal.downsample import (
    DownsampleFactorMax, DownsampleFactorMaxGrad)

from . import pygpu, init_dev
from .basic_ops import (as_gpuarray_variable,
                        gpu_contiguous, HostFromGpu,
                        GpuAllocEmpty, empty_like)
from .conv import GpuConv

# These don't exist in gpuarray
# GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad
from .nnet import GpuSoftmax
from .opt import gpu_seqopt, register_opt, conv_groupopt, op_lifter
from .opt_util import alpha_merge, output_merge

# This is to avoid conflict with the one in cuda/dnn.py
if not hasattr(config, 'dnn'):
    AddConfigVar('dnn.conv.workmem',
                 "Default value for the workmem attribute of cudnn "
                 "convolutions.",
                 EnumStr('small', 'none', 'large'),
                 in_c_key=False)

AddConfigVar('dnn.include_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(lambda: os.path.join(config.cuda.root, 'include')))

AddConfigVar('dnn.library_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(lambda: os.path.join(config.cuda.root, 'lib64')))


def dnn_available():
    if dnn_available.avail is not None:
        return dnn_available.avail
    if pygpu is None:
        dnn_available.msg = "PyGPU not available"
        dnn_available.avail = False
        return False
    if not init_dev.device.startswith('cuda'):
        dnn_available.msg = "Not on a CUDA device. Got %s." % init_dev.device
        dnn_available.avail = False
        return False
    # This is a hack because bin_id is in the from of
    # "sm_<major><minor>" for cuda devices.
    if pygpu.get_default_context().bin_id < 'sm_30':
        dnn_available.msg = "Device not supported by cuDNN"
        dnn_available.avail = False
    preambule = """
#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include <cudnn_helper.h>
"""

    body = """
cudnnHandle_t _handle = NULL;
cudnnStatus_t err;
if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
  fprintf(stderr, "could not create cuDNN handle: %s",
          cudnnGetErrorString(err));
  return 1;
}
"""
    # Do not run here the test program. It would run on the
    # default gpu, not the one selected by the user. If mixed
    # GPU are installed or if the GPUs are configured in
    # exclusive mode, this cause bad detection.
    comp, out, err = GCC_compiler.try_flags(
        ["-l", "cudnn", "-I" + os.path.dirname(__file__),
         "-I" + config.dnn.include_path,
         "-L" + config.dnn.library_path],
        preambule=preambule, body=body,
        try_run=False, output=True)

    dnn_available.avail = comp
    if not dnn_available.avail:
        dnn_available.msg = (
            "Theano cannot compile with cuDNN. We got this error:\n" +
            str(err))
    else:
        # If we can compile, check that we can import and run.
        v = version()
        if isinstance(v, tuple) and v[0] != v[1]:
            dnn_available.avail = False
            dnn_available.msg = ("Mixed dnn version. The header is"
                                 " from one version, but we link with"
                                 " a different version %s" % str(v))
            raise RuntimeError(dnn_available.msg)
        if version() == (20, 20):
            dnn_available.avail = False
            dnn_available.msg = (
                "You have installed a release candidate of CuDNN v2."
                " This isn't supported anymore."
                " Update to CuDNN v2 final version.")
            raise RuntimeError(dnn_available.msg)
    return dnn_available.avail


dnn_available.avail = None
dnn_available.msg = None


def c_set_tensor4d(var, desc, err, fail):
    return """
{
  cudnnDataType_t dt;
  size_t ds;
  switch (%(var)s->ga.typecode) {
  case GA_FLOAT:
    dt = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    dt = CUDNN_DATA_DOUBLE;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_tensor4d");
    return -1;
  }
  ds = gpuarray_get_elsize(%(var)s->ga.typecode);

  int str0, str1, str2, str3;
  // cudnn do not like 0s in strides
  str3 = PyGpuArray_STRIDES(%(var)s)[3]?PyGpuArray_STRIDES(%(var)s)[3]/ds:1;
  str2 = PyGpuArray_STRIDES(%(var)s)[2]?PyGpuArray_STRIDES(%(var)s)[2]/ds:PyGpuArray_DIMS(%(var)s)[3];
  str1 = PyGpuArray_STRIDES(%(var)s)[1]?PyGpuArray_STRIDES(%(var)s)[1]/ds:PyGpuArray_DIMS(%(var)s)[2]*PyGpuArray_DIMS(%(var)s)[3];
  str0 = PyGpuArray_STRIDES(%(var)s)[0]?PyGpuArray_STRIDES(%(var)s)[0]/ds:PyGpuArray_DIMS(%(var)s)[2]*PyGpuArray_DIMS(%(var)s)[3]*PyGpuArray_DIMS(%(var)s)[1];
  %(err)s = cudnnSetTensor4dDescriptorEx(
    %(desc)s, dt,
    PyGpuArray_DIMS(%(var)s)[0],
    PyGpuArray_DIMS(%(var)s)[1],
    PyGpuArray_DIMS(%(var)s)[2],
    PyGpuArray_DIMS(%(var)s)[3],
    str0, str1, str2, str3);

  if (%(err)s != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
    "could not set tensor4d descriptor: %%s",
    cudnnGetErrorString(%(err)s));
    %(fail)s
  }
}
        """ % dict(var=var, err=err, desc=desc, fail=fail)


class DnnBase(COp):
    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.
    """
    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False

    def __init__(self):
        COp.__init__(self, "dnn_base.c")

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h', 'gpuarray_helper.h',
                'gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/util.h',
                'gpuarray_api.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include(),
                config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn', 'gpuarray']

    def c_lib_dirs(self):
        return [config.dnn.library_path]


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
        #if defined(CUDNN_VERSION)
        %(o)s = PyTuple_Pack(2, PyInt_FromLong(CUDNN_VERSION), PyInt_FromLong(cudnnGetVersion()));
        #else
        %(o)s = PyInt_FromLong(-1);
        #endif
        """ % locals()

    def do_constant_folding(self, node):
        # Needed as we do not want to cache this information.
        return False

    def c_code_cache_version(self):
        # Not needed, but make it clear that we do not want to cache this.
        return None


def version():
    """return the current cuDNN version we compile with.

    This return a tuple with the header version and the library
    version we link with. For older cudnn version without version
    information, we return -1.

    """
    if not dnn_available():
        raise Exception(
            "We can't determine the cudnn version as it is not available",
            dnn_available.msg)

    if version.v is None:
        f = theano.function([], DnnVersion()(),
                            theano.Mode(optimizer=None),
                            profile=False)
        version.v = f()
    return version.v
version.v = None


class GpuDnnConvDesc(Op):
    """This Op builds a convolution descriptor for use in the other
    convolution operations.

    see the doc of :func:`dnn_conv` for a description of the parameters

    """
    __props__ = ('border_mode', 'subsample', 'conv_mode')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), config.dnn.include_path]

    def c_libraries(self):
        return ['cudnn']

    def c_lib_dirs(self):
        return [config.dnn.library_path]

    def __init__(self, border_mode, subsample=(1, 1), conv_mode='conv'):
        if isinstance(border_mode, int):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode
        assert len(subsample) == 2
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

    def make_node(self, img_shape, kern_shape):
        if img_shape.type.ndim != 1 or img_shape.type.dtype != 'int64':
            raise TypeError('img must be 1D shape tensor')
        if kern_shape.type.ndim != 1 or kern_shape.type.dtype != 'int64':
            raise TypeError('kern must be 1D shape tensor')

        return Apply(self, [img_shape, kern_shape],
                     [CDataType("cudnnConvolutionDescriptor_t",
                                freefunc="cudnnDestroyConvolutionDescriptor")()])

    def c_code(self, node, name, inputs, outputs, sub):
        img_shape, kern_shape = inputs
        desc, = outputs

        if isinstance(self.border_mode, tuple):
            pad_h_spec, pad_w_spec = map(int, self.border_mode)
            assert pad_h_spec >= 0 and pad_w_spec >= 0
            bmode = 2
        else:
            pad_h_spec = pad_w_spec = 0

            if self.border_mode == "valid":
                bmode = 1
            else:
                assert self.border_mode == "full"
                bmode = 0

        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'

        return """
{
  cudnnStatus_t err;
  int pad_h%(name)s;
  int pad_w%(name)s;

  if ((err = cudnnCreateConvolutionDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }

  if (%(bmode)d == 2) {
    pad_h%(name)s = %(pad_h_spec)d;
    pad_w%(name)s = %(pad_w_spec)d;
  } else if (%(bmode)d == 1) {
    pad_h%(name)s = 0;
    pad_w%(name)s = 0;
  } else if (%(bmode)d == 0) {
    pad_h%(name)s = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 2) - 1;
    pad_w%(name)s = *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 3) - 1;
  } else {
    PyErr_SetString(PyExc_ValueError, "bad border mode");
    %(fail)s
  }
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 20
  err = cudnnSetConvolution2dDescriptor(
  %(desc)s,
  pad_h%(name)s,
  pad_w%(name)s,
  %(subsx)d, %(subsy)d, 1, 1,
  %(conv_flag)s
  );
#else
  err = cudnnSetConvolutionDescriptorEx(
  %(desc)s,
  *(npy_int64 *)PyArray_GETPTR1(%(img_shape)s, 0),
  *(npy_int64 *)PyArray_GETPTR1(%(img_shape)s, 1),
  *(npy_int64 *)PyArray_GETPTR1(%(img_shape)s, 2),
  *(npy_int64 *)PyArray_GETPTR1(%(img_shape)s, 3),
  *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 0),
  *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 2),
  *(npy_int64 *)PyArray_GETPTR1(%(kern_shape)s, 3),
  pad_h%(name)s,
  pad_w%(name)s,
  %(subsx)d, %(subsy)d, 1, 1,
  %(conv_flag)s
  );
#endif
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, img_shape=img_shape, kern_shape=kern_shape, desc=desc,
           bmode=bmode, conv_flag=conv_flag, fail=sub['fail'],
           subsx=self.subsample[0], subsy=self.subsample[1],
           pad_h_spec=pad_h_spec, pad_w_spec=pad_w_spec)

    def c_code_cache_version(self):
        return (1, version())

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


class GpuDnnConv(DnnBase, COp):
    """
    The forward convolution.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor
    """
    __props__ = ('workmem', 'inplace')

    def __init__(self, workmem=None, inplace=False):
        """
        :param workmem: either 'none', 'small' or 'large'.  Default is
        the value of :attr:`config.dnn.conv.workmem`.
        """
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_fwd.c"],
                     "APPLY_SPECIFIC(conv_fwd)")
        if workmem is None:
            workmem = config.dnn.conv.workmem
        self.workmem = workmem
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}
        assert self.workmem in ['none', 'small', 'large']

    def get_op_params(self):
        if self.inplace:
            inpl_def = [('CONV_INPLACE', '1')]
        else:
            inpl_def = []
        if version() == -1:
            alg_def = ('CONV_ALGO', "0")
        else:
            if self.workmem == 'none':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'
            elif self.workmem == 'small':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'
            elif self.workmem == 'large':
                alg = 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'
            alg_def = ('CONV_ALGO', alg)
        return [alg_def] + inpl_def

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):
        img = as_gpuarray_variable(img)
        kern = as_gpuarray_variable(kern)
        output = as_gpuarray_variable(output)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be a 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
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
        the specified parameters.  `ishape` and `kshape` can be symbolic
        or scalar.
        """
        b = ishape[0]  # Number of inputs
        h = ishape[2]  # Height of input feature maps
        w = ishape[3]  # Width of input feature maps
        nb = kshape[0]  # Number of output feature maps
        kh = kshape[2]  # Height of each filter
        kw = kshape[3]  # Width of each filter

        sh, sw = subsample
        if border_mode == 'full':
            padh = kh - 1
            padw = kw - 1
        elif isinstance(border_mode, tuple):
            padh, padw = border_mode
        else:
            assert border_mode == 'valid'
            padh = 0
            padw = 0

        return (
            b, nb,
            (h + 2 * padh - kh) // sh + 1,
            (w + 2 * padw - kw) // sw + 1
        )

    def infer_shape(self, node, shape):
        return [shape[2]]


class GpuDnnConvGradW(DnnBase, COp):
    """
    The convolution gradient with respect to the weights.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_gw.c"],
                     "APPLY_SPECIFIC(conv_gw)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'inplace'):
            self.inplace = False

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
        if self.inplace:
            return [('CONV_INPLACE', '1')]
        else:
            return []

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        img = as_gpuarray_variable(img)
        topgrad = as_gpuarray_variable(topgrad)
        output = as_gpuarray_variable(output)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
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

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        COp.__init__(self, ["dnn_base.c", "dnn_conv_base.c", "dnn_gi.c"],
                     "APPLY_SPECIFIC(conv_gi)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

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
        if self.inplace:
            return [('CONV_INPLACE', '1')]
        else:
            return []

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        kern = as_gpuarray_variable(kern)
        topgrad = as_gpuarray_variable(topgrad)
        output = as_gpuarray_variable(output)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if output.type.ndim != 4:
            raise TypeError('output must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_dt(alpha, _one, 'alpha', kern.dtype)
        beta = ensure_dt(beta, _zero, 'beta', kern.dtype)

        return Apply(self, [kern, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]


def dnn_conv(img, kerns, border_mode='valid', subsample=(1, 1),
             conv_mode='conv', direction_hint=None, workmem=None):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    :param img: images to do the convolution over
    :param kerns: convolution filters
    :param border_mode: one of 'valid', 'full'; additionally, the padding size
        could be directly specified by an integer or a pair of integers
    :param subsample: perform subsampling of the output (default: (1, 1))
    :param conv_mode: perform convolution (kernels flipped) or cross-correlation.
        One of 'conv', 'cross'. (default: 'conv')
    :param direction_hint: Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1,1) and direction_hint is
        'bprop weights', it will use GpuDnnConvGradW.
        If border_mode is 'full', subsample is (1,1) and direction_hint is
        *not* 'forward!', it will use GpuDnnConvGradI.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    :param workmem: Specify the amount of working memory allowed.
      More memory is usually faster.  One of 'none', 'small' or
      'large'.  (default is None which takes its value from
      :attr:`config.dnn.conv.workmem`)


    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """
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
        shape2 = shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1
        shape3 = shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1
        out = GpuAllocEmpty(img.dtype)(shape_i(kerns, 1, fgraph),
                                       shape_i(img, 1, fgraph), shape2, shape3)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode='cross')(img.shape, out.shape)
        conv = GpuDnnConvGradW()(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3))

    elif (border_mode == 'full' and subsample == (1, 1) and
          direction_hint != 'forward!'):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v1 and v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        shape2 = shape_i(img, 2, fgraph) + shape_i(kerns, 2, fgraph) - 1
        shape3 = shape_i(img, 3, fgraph) + shape_i(kerns, 3, fgraph) - 1
        out = GpuAllocEmpty(img.dtype)(shape_i(img, 0, fgraph),
                                       shape_i(kerns, 1, fgraph),
                                       shape2, shape3)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode=conv_mode)(out.shape, kerns.shape)
        return GpuDnnConvGradI()(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    # contig_version will return a gpu_contiguous copy
    # if the img contains negative strides
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns.shape)
    desc_op = desc.owner.op
    out_shp = GpuDnnConv.get_out_shape(img.shape, kerns.shape,
                                       desc_op.border_mode,
                                       desc_op.subsample)
    out = GpuAllocEmpty(img.dtype)(*out_shp)
    return GpuDnnConv(workmem=workmem)(img, kerns, out, desc)


class GpuDnnPoolDesc(Op):
    """
    This Op builds a pooling descriptor for use in the other
    pooling operations.

    :param ws: windows size
    :param stride: (dx, dy)
    :param mode: 'max', 'average_inc_pad' or 'average_exc_pad'
        The old deprecated name 'average' correspond to 'average_inc_pad'
    :param pad: (padX, padY) padding information.
        padX is the size of the left and right borders,
        padY is the size of the top and bottom borders.
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
        assert len(ws) == 2
        self.ws = ws
        assert len(stride) == 2
        self.stride = stride
        assert len(stride) == 2
        self.pad = pad
        if (pad[0] != 0 or pad[1] != 0) and version() == -1:
            raise RuntimeError("CuDNN pooling with padding requires CuDNN v2")

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'pad'):
            self.pad = (0, 0)

    def make_node(self):
        if self.pad != (0, 0) and version() == -1:
            raise RuntimeError("CuDNN pooling with padding requires CuDNN v2")

        return Apply(self, [],
                     [CDataType("cudnnPoolingDescriptor_t",
                                freefunc="cudnnDestroyPoolingDescriptor")()])

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average_inc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        elif self.mode == "average_exc_pad":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
            if version() == -1:
                raise Exception("cudnn v1 do not support average_exc_pad")
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
#ifndef CUDNN_VERSION
  err = cudnnSetPoolingDescriptor(
  %(desc)s,
  %(mode_flag)s,
  %(wsX)d, %(wsY)d,
  %(stridex)d, %(stridey)d
  );
#else
  err = cudnnSetPooling2dDescriptor(
  %(desc)s,
  %(mode_flag)s,
  %(wsX)d, %(wsY)d,
  %(padX)d, %(padY)d,
  %(stridex)d, %(stridey)d
  );
#endif
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, desc=desc, mode_flag=mode_flag, fail=sub['fail'],
           wsX=self.ws[0], wsY=self.ws[1],
           stridex=self.stride[0], stridey=self.stride[1],
           padX=self.pad[0], padY=self.pad[1])

    def c_code_cache_version(self):
        return (2, version())


class GpuDnnPool(DnnBase):
    """
    Pooling.

    :param img: the image 4d tensor.
    :param desc: the pooling descriptor.
    """
    __props__ = ()

    def make_node(self, img, desc):
        img = as_gpuarray_variable(img)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnPoolingDescriptor_t':
            raise TypeError('desc must be cudnnPoolingDescriptor_t')

        return Apply(self, [img, desc],
                     [img.type()])

    def infer_shape(self, node, shape):
        desc = node.inputs[1].owner.op
        kh, kw = desc.ws
        sh, sw = desc.stride
        padh, padw = desc.pad
        return [(
            shape[0][0],
            shape[0][1],
            (shape[0][2] + 2 * padh - kh) // sh + 1,
            (shape[0][3] + 2 * padw - kw) // sw + 1
        )]

    def c_support_code_struct(self, node, name):
        return """
cudnnTensorDescriptor_t input%(name)s;
cudnnTensorDescriptor_t output%(name)s;
""" % dict(name=name)

    def c_init_code_struct(self, node, name, sub):
        return """
cudnnStatus_t err%(name)s;
input%(name)s = NULL;
output%(name)s = NULL;
if ((err%(name)s = cudnnCreateTensorDescriptor(&input%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&output%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(out): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(name=name, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, name):
        return """
if (input%(name)s != NULL) { cudnnDestroyTensorDescriptor(input%(name)s); }
if (output%(name)s != NULL) { cudnnDestroyTensorDescriptor(output%(name)s); }
""" % dict(name=name)

    def c_code(self, node, name, inputs, outputs, sub):
        desc = inputs[1]
        out, = outputs

        set_in = c_set_tensor4d(inputs[0], "input" + str(name),
                                'err' + name, sub['fail'])

        set_out = c_set_tensor4d(out, "output" + str(name),
                                 'err' + name, sub['fail'])

        return """
cudnnStatus_t err%(name)s;

size_t %(out)s_dims[4];

if (!GpuArray_IS_C_CONTIGUOUS(&%(input)s->ga)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

%(set_in)s

cudnnPoolingMode_t mode;
int wsX, wsY, vpad, hpad, strideX, strideY;
#ifndef CUDNN_VERSION
err%(name)s = cudnnGetPoolingDescriptor(
        %(desc)s, &mode,
        &wsX, &wsY,
        &strideX, &strideY);
#else
err%(name)s = cudnnGetPooling2dDescriptor(
        %(desc)s, &mode,
        &wsX, &wsY,
        &vpad, &hpad,
        &strideX, &strideY);
#endif

if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnGetPoolingDescriptor operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}

%(out)s_dims[0] = PyGpuArray_DIMS(%(input)s)[0];
%(out)s_dims[1] = PyGpuArray_DIMS(%(input)s)[1];
%(out)s_dims[2] = (PyGpuArray_DIMS(%(input)s)[2] + (vpad*2) - wsX) / strideX + 1;
%(out)s_dims[3] = (PyGpuArray_DIMS(%(input)s)[3] + (hpad*2) - wsY) / strideY + 1;

if (theano_prep_output(&%(out)s, 4, %(out)s_dims, %(input)s->ga.typecode,
                       GA_C_ORDER, pygpu_default_context()) != 0) {
  %(fail)s
}

%(set_out)s
#ifndef CUDNN_VERSION
err%(name)s = cudnnPoolingForward(
_handle,
%(desc)s,
%(input_desc)s, PyGpuArray_DEV_DATA(%(input)s),
%(output_desc)s, PyGpuArray_DEV_DATA(%(out)s)
);
#else
{
const float alpha = 1;
const float beta = 0;
err%(name)s = cudnnPoolingForward(
_handle,
%(desc)s,
&alpha,
%(input_desc)s, PyGpuArray_DEV_DATA(%(input)s),
&beta,
%(output_desc)s, PyGpuArray_DEV_DATA(%(out)s)
);
}
#endif
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnPoolingForward operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(out=out, desc=desc, fail=sub['fail'],
           name=name, set_in=set_in,
           set_out=set_out, input=inputs[0],
           input_desc="input" + name,
           output_desc="output" + name)

    def grad(self, inp, grads):
        img, desc = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        out = self(img, desc)

        g_out = GpuDnnPoolGrad()(img, out, grad, desc)

        return g_out, theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [0]]

    def c_code_cache_version(self):
        return (7, version())


class GpuDnnPoolGrad(DnnBase):
    """
    The pooling gradient.

    :param inp: the input of the pooling.
    :param out: the output of the pooling in the forward.
    :param inp_grad: same size as out, but is the corresponding gradient information.
    :param desc: The pooling descriptor.
    """
    __props__ = ()

    def make_node(self, inp, out, inp_grad, desc):
        inp = as_gpuarray_variable(inp)
        if inp.type.ndim != 4:
            raise TypeError('inp must be 4D tensor')

        inp_grad = as_gpuarray_variable(inp_grad)
        if inp_grad.type.ndim != 4:
            raise TypeError('inp_grad must be 4D tensor')

        out = as_gpuarray_variable(out)
        if out.type.ndim != 4:
            raise TypeError('out must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnPoolingDescriptor_t':
            raise TypeError('desc must be cudnnPoolingDescriptor_t')

        return Apply(self, [inp, out, inp_grad, desc],
                     [inp.type()])

    def c_support_code_struct(self, node, name):
        return """
cudnnTensorDescriptor_t input%(name)s;
cudnnTensorDescriptor_t input_grad%(name)s;
cudnnTensorDescriptor_t output%(name)s;
cudnnTensorDescriptor_t output_grad%(name)s;
""" % dict(name=name)

    def c_init_code_struct(self, node, name, sub):
        return """
cudnnStatus_t err%(name)s;
input%(name)s = NULL;
input_grad%(name)s = NULL;
output%(name)s = NULL;
output_grad%(name)s = NULL;
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
""" % dict(name=name, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, name):
        return """
if (input%(name)s != NULL) { cudnnDestroyTensorDescriptor(input%(name)s); }
if (input_grad%(name)s != NULL) { cudnnDestroyTensorDescriptor(input_grad%(name)s); }
if (output%(name)s != NULL) { cudnnDestroyTensorDescriptor(output%(name)s); }
if (output_grad%(name)s != NULL) { cudnnDestroyTensorDescriptor(output_grad%(name)s); }
""" % dict(name=name)

    def c_code(self, node, name, inputs, outputs, sub):
        # Here the name out and inp are based on the cudnn definition.
        # Not the definition of this class.
        # This make it complicated.
        out, inp, inp_grad, desc = inputs
        out_grad, = outputs

        set_in = "\n".join([
            c_set_tensor4d(inp, "input" + name,
                           'err' + name, sub['fail']),
            c_set_tensor4d(inp_grad, "input_grad" + name,
                           'err' + name, sub['fail']),
            c_set_tensor4d(out, "output" + name,
                           'err' + name, sub['fail'])
        ])

        set_out = c_set_tensor4d(out, "output_grad" + name,
                                 'err' + name, sub['fail'])

        return """
cudnnStatus_t err%(name)s;

if (!GpuArray_IS_C_CONTIGUOUS(&%(input)s->ga)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous inputs are supported.");
  %(fail)s
}

if (!GpuArray_IS_C_CONTIGUOUS(&%(input_grad)s->ga)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous input gradients are supported.");
  %(fail)s
}

if (!GpuArray_IS_C_CONTIGUOUS(&%(output)s->ga)) {
  PyErr_SetString(PyExc_ValueError,
                  "GpuDnnPoolGrad: Only contiguous outputs are supported.");
  %(fail)s
}

%(set_in)s

if (theano_prep_output(&%(output_grad)s, PyGpuArray_NDIM(%(output)s),
                       PyGpuArray_DIMS(%(output)s), %(output)s->ga.typecode,
                       GA_C_ORDER, pygpu_default_context()) != 0)
{
  %(fail)s
}

%(set_out)s
#ifndef CUDNN_VERSION
err%(name)s = cudnnPoolingBackward(
_handle,
%(desc)s,
%(input_desc)s, PyGpuArray_DEV_DATA(%(input)s),
%(input_grad_desc)s, PyGpuArray_DEV_DATA(%(input_grad)s),
%(output_desc)s, PyGpuArray_DEV_DATA(%(output)s),
%(output_grad_desc)s, PyGpuArray_DEV_DATA(%(output_grad)s)
);
#else
{
const float alpha = 1;
const float beta = 0;
err%(name)s = cudnnPoolingBackward(
_handle,
%(desc)s,
&alpha,
%(input_desc)s, PyGpuArray_DEV_DATA(%(input)s),
%(input_grad_desc)s, PyGpuArray_DEV_DATA(%(input_grad)s),
%(output_desc)s, PyGpuArray_DEV_DATA(%(output)s),
&beta,
%(output_grad_desc)s, PyGpuArray_DEV_DATA(%(output_grad)s)
);
}
#endif
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPoolGrad: error doing operation: %%s.",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(output_grad=out_grad, desc=desc,
           fail=sub['fail'],
           name=name, set_in=set_in,
           set_out=set_out, input=inp, input_grad=inp_grad, output=out,
           input_desc="input" + name,
           input_grad_desc="input_grad" + name,
           output_desc="output" + name,
           output_grad_desc="output_grad" + name)

    def c_code_cache_version(self):
        return (5, version())

    def infer_shape(self, node, shape):
        return [shape[0]]


def dnn_pool(img, ws, stride=(1, 1), mode='max', pad=(0, 0)):
    """
    GPU pooling using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    :param img: images to do the pooling over
    :param ws: subsampling window size
    :param stride: subsampling stride (default: (1, 1))
    :param mode: one of 'max', 'average_inc_pad' or 'average_exc_pad
        (default: 'max')
    :param pad: (padX, padY) padding information.
        padX is the size of the left and right borders,
        padY is the size of the top and bottom borders.

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    :note: This Op implements the ignore_border=True of max_pool_2d.
    """
    img = gpu_contiguous(img)
    desc = GpuDnnPoolDesc(ws=ws, stride=stride, mode=mode, pad=pad)()
    return GpuDnnPool()(img, desc)


class GpuDnnSoftmaxBase(DnnBase):
    """
    Op for the cuDNN Softmax.

    :param tensor_format: Whether the data format is 'bc01' or 'b01c'.
    :param algo: 'fast' or 'accurate' indicating whether computations should be
        optimized for speed or accuracy respectively.
    :param mode: 'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.
    """

    __props__ = ('tensor_format', 'mode', 'algo')

    def __init__(self, tensor_format, algo, mode):
        assert(tensor_format in ('bc01', 'b01c'))
        DnnBase.__init__(self)
        self.tensor_format = tensor_format

        assert(algo in ('fast', 'accurate'))
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

    def _define_tensor4d_desc(self, name, id):
        return """
cudnnTensorDescriptor_t %(id)s_%(name)s;
""" % dict(name=name, id=id)

    def _init_tensor4d_desc(self, name, id, fail):
        return """
%(id)s_%(name)s = NULL;
if ((err%(name)s = cudnnCreateTensorDescriptor(&%(id)s_%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               ": %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(name=name, id=id, fail=fail)

    def _clean_tensor4d_desc(self, name, id):
        return """
if(%(id)s_%(name)s!= NULL)
  cudnnDestroyTensorDescriptor(%(id)s_%(name)s);
""" % dict(name=name, id=id)

    def c_support_code_struct(self, node, name):
        result = ''
        for id in self.tensor_4d_descs:
            result += self._define_tensor4d_desc(name, id)
        return result

    def c_init_code_struct(self, node, name, sub):
        result = """
cudnnStatus_t err%(name)s;
""" % dict(name=name)

        for id in self.tensor_4d_descs:
            result += self._init_tensor4d_desc(name, id, sub['fail'])
        return result

    def c_cleanup_code_struct(self, node, name):
        result = ''
        for id in self.tensor_4d_descs:
            result += self._clean_tensor4d_desc(name, id)
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
            algo = 1
        else:
            algo = 0

        # Setup configuration variables.
        result = """
cudnnStatus_t err%(name)s;
cudnnTensorFormat_t format%(name)s = CUDNN_TENSOR_NCHW;
if (%(tensor_format)d == 1)
  format%(name)s = CUDNN_TENSOR_NHWC;

cudnnSoftmaxAlgorithm_t algo%(name)s = CUDNN_SOFTMAX_ACCURATE;
if (%(algo)d == 1)
  algo%(name)s = CUDNN_SOFTMAX_FAST;

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
if (theano_prep_output(&%(outs)s, PyGpuArray_NDIM(%(ins)s),
                       PyGpuArray_DIMS(%(ins)s), %(ins)s->ga.typecode,
                       GA_C_ORDER, pygpu_default_context()) != 0)
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
        return (0, 7, version())

    def method(self):
        raise NotImplementedError('GpuDnnSoftmaxBase::method')


class GpuDnnSoftmax(GpuDnnSoftmaxBase):
    """
    Op for the cuDNN Softmax.

    :param tensor_format: Whether the data format is 'bc01' or 'b01c'.
    :param algo: 'fast' or 'accurate' indicating whether computations should be
        optimized for speed or accuracy respectively.
    :param mode: 'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.
    """
    direction = 'forward'
    softmax_inputs = ['softmax_input']

    def make_node(self, x):
        x = as_gpuarray_variable(x)
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
  PyGpuArray_DEV_DATA(%(ins)s),
  softmax_output_%(name)s,
  PyGpuArray_DEV_DATA(%(outs)s)
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
  PyGpuArray_DEV_DATA(%(ins)s),
  (void*) &beta,
  softmax_output_%(name)s,
  PyGpuArray_DEV_DATA(%(outs)s)
);
}
#endif
"""

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = self.make_node(x).outputs[0]
        return [GpuDnnSoftmaxGrad(
            self.tensor_format,
            self.algo,
            self.mode
        )(g_sm, sm)]


class GpuDnnSoftmaxGrad(GpuDnnSoftmaxBase):
    """
    Op for the cuDNN SoftmaxGrad.

    :param tensor_format: Whether the data format is 'bc01' or 'b01c'.
    :param algo: 'fast' or 'accurate' indicating whether computations should be
        optimized for speed or accuracy respectively.
    :param mode: 'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.
    """
    direction = 'backward'
    softmax_inputs = ['softmax_gout', 'softmax_input']

    def make_node(self, dy, sm):
        dy = as_gpuarray_variable(dy)
        sm = as_gpuarray_variable(sm)
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
  PyGpuArray_DEV_DATA(%(ins1)s),
  %(name0)s_%(name)s,
  PyGpuArray_DEV_DATA(%(ins0)s),
  softmax_output_%(name)s,
  PyGpuArray_DEV_DATA(%(outs)s)
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
  PyGpuArray_DEV_DATA(%(ins1)s),
  %(name0)s_%(name)s,
  PyGpuArray_DEV_DATA(%(ins0)s),
  (void*) &beta,
  softmax_output_%(name)s,
  PyGpuArray_DEV_DATA(%(outs)s)
);
}
#endif
        """


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
            # of the backward pass wrt. inputs
            direction_hint = 'forward!'
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


conv_groupopt.register('local_conv_dnn', local_conv_dnn, 20,
                       'conv_dnn', 'fast_compile', 'fast_run', 'cudnn')


@local_optimizer([GpuDnnConv], inplace=True)
def local_dnn_conv_inplace(node):
    if type(node.op) != GpuDnnConv or node.op.inplace:
        return
    inputs = list(node.inputs)
    dest = inputs[2]
    if (dest.owner and
            isinstance(dest.owner.op, GpuAllocEmpty) and
            len(dest.clients) > 1):
        inputs[2] = GpuAllocEmpty(dest.owner.op.dtype)(*dest.owner.inputs)
    return [GpuDnnConv(workmem=node.op.workmem, inplace=True)(*inputs)]


@local_optimizer([GpuDnnConvGradW], inplace=True)
def local_dnn_convgw_inplace(node):
    if type(node.op) != GpuDnnConvGradW or node.op.inplace:
        return
    inputs = list(node.inputs)
    dest = inputs[2]
    if (dest.owner and
            isinstance(dest.owner.op, GpuAllocEmpty) and
            len(dest.clients) > 1):
        inputs[2] = GpuAllocEmpty(dest.owner.op.dtype)(*dest.owner.inputs)
    return [GpuDnnConvGradW(inplace=True)(*inputs)]


@local_optimizer([GpuDnnConvGradI], inplace=True)
def local_dnn_convgi_inplace(node):
    if type(node.op) != GpuDnnConvGradI or node.op.inplace:
        return
    inputs = list(node.inputs)
    dest = inputs[2]
    if (dest.owner and
            isinstance(dest.owner.op, GpuAllocEmpty) and
            len(dest.clients) > 1):
        inputs[2] = GpuAllocEmpty(dest.owner.op.dtype)(*dest.owner.inputs)
    return [GpuDnnConvGradI(inplace=True)(*inputs)]

optdb.register('local_dnna_conv_inplace',
               tensor.opt.in2out(local_dnn_conv_inplace,
                                 local_dnn_convgw_inplace,
                                 local_dnn_convgi_inplace,
                                 name="local_dnn_conv_inplace"),
               70.0, 'fast_run', 'inplace', 'gpuarray', 'cudnn')


@register_opt('cudnn')
@alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5, nd=4)
def local_dnn_conv_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnnConv(workmem=node.op.workmem)(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, nd=4)
def local_dnn_convw_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnnConvGradW()(*inputs)]


@register_opt('cudnn')
@alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, nd=4)
def local_dnn_convi_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnnConvGradI()(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2, nd=4)
def local_dnn_conv_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConv(workmem=node.op.workmem)(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2, nd=4)
def local_dnn_convw_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradW()(*inputs)]


@register_opt('cudnn')
@output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2, nd=4)
def local_dnn_convi_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnnConvGradI()(*inputs)]


@register_opt('cudnn')
@op_lifter([DownsampleFactorMax])
def local_pool_dnn_alternative(node):
    if not dnn_available():
        return
    if not node.op.ignore_border:
        return
    img, = node.inputs
    ds = node.op.ds
    stride = node.op.st
    pad = node.op.padding
    mode = node.op.mode
    return dnn_pool(gpu_contiguous(img.owner.inputs[0]),
                    ds, stride=stride, pad=pad, mode=mode)


@register_opt('cudnn')
@op_lifter([DownsampleFactorMaxGrad])
def local_pool_dnn_grad_stride(node):
    if not dnn_available():
        return
    if not node.op.ignore_border:
        return
    inp, out, inp_grad = node.inputs
    ds = node.op.ds
    st = node.op.st
    pad = node.op.padding
    mode = node.op.mode

    desc = GpuDnnPoolDesc(ws=ds, stride=st, mode=mode, pad=pad)()
    return GpuDnnPoolGrad()(gpu_contiguous(inp),
                            gpu_contiguous(out),
                            gpu_contiguous(inp_grad),
                            desc)


@register_opt('cudnn')
@local_optimizer([GpuSoftmax])
def local_softmax_dnn(node):
    if not dnn_available():
        return
    if isinstance(node.op, GpuSoftmax):
        ins = node.inputs[0].dimshuffle(0, 1, 'x', 'x')
        ins = gpu_contiguous(ins)
        out = GpuDnnSoftmax('bc01', 'accurate', 'channel')(ins)
        out = as_gpuarray_variable(out.dimshuffle(0, 1))
        return [out]


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
@op_lifter([SoftmaxGrad])
def local_softmax_dnn_grad(node):
    if not dnn_available():
        return
    ins = []
    for n in node.inputs:
        if isinstance(n.owner.op, HostFromGpu):
            n = n.owner.inputs[0]
        if n.ndim != 2:
            return
        ins.append(n.dimshuffle(0, 1, 'x', 'x'))

    out = GpuDnnSoftmaxGrad('bc01', 'accurate', 'channel')(
        gpu_contiguous(ins[0]), gpu_contiguous(ins[1]))
    return [out.dimshuffle(0, 1)]
