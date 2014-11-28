import os

import theano
from theano import Apply, gof, tensor
from theano.gof import Optimizer, local_optimizer
from theano.gof.type import CDataType
from theano.compat import PY3
from theano.tensor.nnet import SoftmaxGrad
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous, HostFromGpu)
from theano.sandbox.cuda.blas import (GpuConv, GpuDownsampleFactorMax,
                                      GpuDownsampleFactorMaxGrad)
from theano.sandbox.cuda.nnet import GpuSoftmax
from theano.sandbox.cuda import gpu_seqopt, register_opt

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler


def dnn_available():
    if dnn_available.avail is None:
        dev = theano.sandbox.cuda.active_device_number()
        if theano.sandbox.cuda.device_properties(dev)['major'] < 3:
            dnn_available.msg = "Device not supported by cuDNN"
            dnn_available.avail = False
        else:
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

            comp, run, out, err = gof.cmodule.GCC_compiler.try_flags(
                ["-l", "cudnn", "-I" + os.path.dirname(__file__),
                 "-I" + os.path.join(theano.config.cuda.root, 'include'),
                 "-L" + os.path.join(theano.config.cuda.root, 'lib64')],
                preambule=preambule, body=body,
                try_run=True, output=True)

            dnn_available.avail = comp and run
            if dnn_available.avail:
                dnn_available.msg = "cuDNN should work"
            else:
                dnn_available.msg = (
                    "Theano is not able to use cuDNN. We got this error: \n" +
                    str(err))
    return dnn_available.avail


dnn_available.avail = None
dnn_available.msg = None


def c_set_tensor4d(var, desc, err, fail):
    return """
%(err)s = cudnnSetTensor4dDescriptorEx(
    %(desc)s, CUDNN_DATA_FLOAT,
    CudaNdarray_HOST_DIMS(%(var)s)[0],
    CudaNdarray_HOST_DIMS(%(var)s)[1],
    CudaNdarray_HOST_DIMS(%(var)s)[2],
    CudaNdarray_HOST_DIMS(%(var)s)[3],
    CudaNdarray_HOST_STRIDES(%(var)s)[0]?CudaNdarray_HOST_STRIDES(%(var)s)[0]:CudaNdarray_HOST_DIMS(%(var)s)[2]*CudaNdarray_HOST_DIMS(%(var)s)[3]*CudaNdarray_HOST_DIMS(%(var)s)[1],
    CudaNdarray_HOST_STRIDES(%(var)s)[1]?CudaNdarray_HOST_STRIDES(%(var)s)[1]:CudaNdarray_HOST_DIMS(%(var)s)[2]*CudaNdarray_HOST_DIMS(%(var)s)[3],
    CudaNdarray_HOST_STRIDES(%(var)s)[2]?CudaNdarray_HOST_STRIDES(%(var)s)[2]:CudaNdarray_HOST_DIMS(%(var)s)[3],
    CudaNdarray_HOST_STRIDES(%(var)s)[3]?CudaNdarray_HOST_STRIDES(%(var)s)[3]:1
);
if (%(err)s != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set tensor4d descriptor: %%s",
    cudnnGetErrorString(%(err)s));
    %(fail)s
}
        """ % dict(var=var, err=err, desc=desc, fail=fail)


class DnnBase(GpuOp):
    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.
    """
    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_support_code(self):
        return """
cudnnHandle_t _handle = NULL;
"""

    def c_init_code(self):
        if PY3:
            error_out = "NULL"
        else:
            error_out = ""
        return ["""{
cudnnStatus_t err;
if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not create cuDNN handle: %%s",
               cudnnGetErrorString(err));
  return %s;
}
}""" % (error_out,)]


class GpuDnnConvDesc(GpuOp):
    """This Op builds a convolution descriptor for use in the other
    convolution operations.

    see the doc of :func:`dnn_conv` for a description of the parameters

    """
    __props__ = ('border_mode', 'subsample', 'conv_mode')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_compiler(self):
        return NVCC_compiler

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
                     [CDataType("cudnnConvolutionDescriptor_t")()])

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
        return (2,)


class GpuDnnConvBase(DnnBase):
    __props__ = ()

    def c_support_code_struct(self, node, struct_id):
        return """
cudnnTensor4dDescriptor_t input%(id)d;
cudnnTensor4dDescriptor_t output%(id)d;
cudnnFilterDescriptor_t kerns%(id)d;
""" % dict(id=struct_id)

    def c_init_code_struct(self, node, struct_id, sub):
        return """
cudnnStatus_t err%(id)d;
input%(id)d = NULL;
output%(id)d = NULL;
kerns%(id)d = NULL;
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&input%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&output%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(out): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateFilterDescriptor(&kerns%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %%s",
               cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, struct_id):
        return """
if (input%(id)d != NULL) {cudnnDestroyTensor4dDescriptor(input%(id)d);}
if (output%(id)d != NULL) {cudnnDestroyTensor4dDescriptor(output%(id)d);}
if (kerns%(id)d != NULL) {cudnnDestroyFilterDescriptor(kerns%(id)d);}
""" % dict(id=struct_id)

    def c_set_filter(self, var, desc, err, fail):
        return """
%(err)s = cudnnSetFilterDescriptor(
%(desc)s, CUDNN_DATA_FLOAT,
CudaNdarray_HOST_DIMS(%(var)s)[0],
CudaNdarray_HOST_DIMS(%(var)s)[1],
CudaNdarray_HOST_DIMS(%(var)s)[2],
CudaNdarray_HOST_DIMS(%(var)s)[3]
);
if (%(err)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set filter descriptor: %%s",
               cudnnGetErrorString(%(err)s));
  %(fail)s
}
""" % dict(var=var, desc=desc, err=err, fail=fail)

    def c_set_tensor4d(self, *arg):
        return c_set_tensor4d(*arg)

    def c_code(self, node, name, inputs, outputs, sub):
        desc = inputs[2]
        out, = outputs

        checks = []
        for v in inputs[:2]:
            checks.append("""
if (!CudaNdarray_is_c_contiguous(%s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %s
}
""" % (v, sub['fail']))

        sets = []
        for p, v, d in zip(inputs[:2], self.conv_inputs, self.conv_types[:2]):
            sets.append(getattr(self, 'c_set_'+d)(p, v + str(sub['struct_id']),
                                                  'err' + name, sub['fail']))

        set_out = getattr(self, 'c_set_' + self.conv_types[2])(
            out, self.conv_output + str(sub['struct_id']), 'err' + name,
            sub['fail'])

        return """
cudnnStatus_t err%(name)s;

%(checks)s

%(sets)s

{
  int out_dims[4];
  err%(name)s = cudnnGetOutputTensor4dDim(
  %(desc)s, %(path)s,
  &out_dims[0], &out_dims[1],
  &out_dims[2], &out_dims[3]
  );
  if (err%(name)s != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not get output sizes: %%s",
                 cudnnGetErrorString(err%(name)s));
    %(fail)s
  }
  // workaround for cudnn R1 bug
  if (%(path)s == CUDNN_CONVOLUTION_WEIGHT_GRAD &&
      (out_dims[0] != CudaNdarray_HOST_DIMS(%(input2)s)[1] ||
       out_dims[1] != CudaNdarray_HOST_DIMS(%(input1)s)[1])) {
    out_dims[0] = CudaNdarray_HOST_DIMS(%(input2)s)[1];
    out_dims[1] = CudaNdarray_HOST_DIMS(%(input1)s)[1];
    // This is a horrible hack that is unfortulately necessary
    int *dd = (int *)%(desc)s;
    out_dims[2] = dd[5];
    out_dims[3] = dd[6];
  }
  if (CudaNdarray_prep_output(&%(out)s, 4, out_dims) != 0) {
    %(fail)s
  }
}

%(set_out)s

err%(name)s = %(method)s(
_handle,
%(input1_desc)s, CudaNdarray_DEV_DATA(%(input1)s),
%(input2_desc)s, CudaNdarray_DEV_DATA(%(input2)s),
%(desc)s,
%(output_desc)s, CudaNdarray_DEV_DATA(%(out)s),
CUDNN_RESULT_NO_ACCUMULATE
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "error doing operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(out=out, desc=desc, fail=sub['fail'], id=sub['struct_id'],
           name=name, checks='\n'.join(checks), sets='\n'.join(sets),
           set_out=set_out, input1=inputs[0], input2=inputs[1],
           input1_desc=self.conv_inputs[0]+str(sub['struct_id']),
           input2_desc=self.conv_inputs[1]+str(sub['struct_id']),
           output_desc=self.conv_output+str(sub['struct_id']),
           method=self.conv_op, path=self.path_flag)

    def c_code_cache_version(self):
        return (8,)


class GpuDnnConv(GpuDnnConvBase):
    """
    The forward convolution.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """
    conv_inputs = 'input', 'kerns'
    conv_output = 'output'
    conv_types = 'tensor4d', 'filter', 'tensor4d'
    conv_op = 'cudnnConvolutionForward'
    path_flag = 'CUDNN_CONVOLUTION_FWD'

    def make_node(self, img, kern, desc):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        broadcastable = (img.type.broadcastable[0],
                         kern.type.broadcastable[0],
                         False, False)
        return Apply(self, [img, kern, desc],
                     [CudaNdarrayType(broadcastable)()])

    def grad(self, inp, grads):
        img, kerns, desc = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConvGradI()(kerns, top, desc)
        d_kerns = GpuDnnConvGradW()(img, top, desc)

        return d_img, d_kerns, theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [0]]


class GpuDnnConvGradW(GpuDnnConvBase):
    """
    The convolution gradient with respect to the weights.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """

    conv_inputs = 'input', 'output',
    conv_output = 'kerns'
    conv_types = 'tensor4d', 'tensor4d', 'filter'
    path_flag = 'CUDNN_CONVOLUTION_WEIGHT_GRAD'
    conv_op = 'cudnnConvolutionBackwardFilter'

    def grad(self, inp, grads):
        img, top, desc = inp
        kerns, = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnnConvGradI()(kerns, top, desc)
        d_top = GpuDnnConv()(img, kerns, desc)

        return d_img, d_top, theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [0]]

    def make_node(self, img, topgrad, desc):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        broadcastable = [topgrad.type.broadcastable[1],
                         img.type.broadcastable[1],
                         False, False]
        return Apply(self, [img, topgrad, desc],
                     [CudaNdarrayType(broadcastable)()])


class GpuDnnConvGradI(GpuDnnConvBase):
    """
    The convolution gradient with respect to the inputs.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """

    conv_inputs = 'kerns', 'output',
    conv_output = 'input'
    conv_types = 'filter', 'tensor4d', 'tensor4d'
    path_flag = 'CUDNN_CONVOLUTION_DATA_GRAD'
    conv_op = 'cudnnConvolutionBackwardData'

    def grad(self, inp, grads):
        kerns, top, desc = inp
        img, = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnnConvGradW()(img, top, desc)
        d_top = GpuDnnConv()(img, kerns, desc)

        return d_kerns, d_top, theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [0]]

    def make_node(self, kern, topgrad, desc):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        broadcastable = [topgrad.type.broadcastable[0],
                         kern.type.broadcastable[1],
                         False, False]
        return Apply(self, [kern, topgrad, desc],
                     [CudaNdarrayType(broadcastable)()])


def dnn_conv(img, kerns, border_mode='valid', subsample=(1, 1),
             conv_mode='conv', direction_hint=None):
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

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """
    if (border_mode == 'valid' and subsample == (1,1) and
        direction_hint == 'bprop weights'):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
        if conv_mode == 'conv':
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        shape = theano.tensor.stack(kerns.shape[1], img.shape[1],
                                    img.shape[2] - kerns.shape[2] + 1,
                                    img.shape[3] - kerns.shape[3] + 1)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode='cross')(img.shape, shape)
        conv = GpuDnnConvGradW()(img, kerns, desc)
        return as_cuda_ndarray_variable(conv.dimshuffle(1, 0, 2, 3))

    elif (border_mode == 'full' and subsample == (1, 1) and
          direction_hint != 'forward!'):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = 'cross' if conv_mode == 'conv' else 'conv'
        shape = theano.tensor.stack(img.shape[0], kerns.shape[1],
                                    img.shape[2] + kerns.shape[2] - 1,
                                    img.shape[3] + kerns.shape[3] - 1)
        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                              conv_mode=conv_mode)(shape, kerns.shape)
        return GpuDnnConvGradI()(kerns, img, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns.shape)
    return GpuDnnConv()(img, kerns, desc)


class GpuDnnPoolDesc(GpuOp):
    """
    This Op builds a pooling descriptor for use in the other
    pooling operations.

    :param ws: windows size
    :param stride: (dx, dy)
    :param mode: 'max' or 'average'
    """
    __props__ = ('ws', 'stride', 'mode')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_compiler(self):
        return NVCC_compiler

    def do_constant_folding(self, node):
        return False

    def __init__(self, ws=(1, 1), stride=(1, 1), mode='max'):
        assert mode in ('max', 'average')
        self.mode = mode
        assert len(ws) == 2
        self.ws = ws
        assert len(stride) == 2
        self.stride = stride

    def make_node(self):
        return Apply(self, [],
                     [CDataType("cudnnPoolingDescriptor_t")()])

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average":
            mode_flag = 'CUDNN_POOLING_AVERAGE'
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

  err = cudnnSetPoolingDescriptor(
  %(desc)s,
  %(mode_flag)s,    
  %(wsX)d, %(wsY)d,
  %(stridex)d, %(stridey)d
  );

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, desc=desc, mode_flag=mode_flag, fail=sub['fail'],
           wsX=self.ws[0], wsY=self.ws[1], stridex=self.stride[0],
           stridey=self.stride[1])

    def c_code_cache_version(self):
        return (1,)


class GpuDnnPool(DnnBase):
    """
    Pooling.

    :param img: the image 4d tensor.
    :param desc: the pooling descriptor.
    """
    __props__ = ()

    def make_node(self, img, desc):
        img = as_cuda_ndarray_variable(img)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnPoolingDescriptor_t':
            raise TypeError('desc must be cudnnPoolingDescriptor_t')

        return Apply(self, [img, desc],
                     [img.type()])

    def c_support_code_struct(self, node, struct_id):
        return """
cudnnTensor4dDescriptor_t input%(id)d;
cudnnTensor4dDescriptor_t output%(id)d;
""" % dict(id=struct_id)

    def c_init_code_struct(self, node, struct_id, sub):
        return """
cudnnStatus_t err%(id)d;
input%(id)d = NULL;
output%(id)d = NULL;
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&input%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&output%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(out): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, struct_id):
        return """
if (input%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(input%(id)d); }
if (output%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(output%(id)d); }
""" % dict(id=struct_id)

    def c_code(self, node, name, inputs, outputs, sub):
        desc = inputs[1]
        out, = outputs

        set_in = c_set_tensor4d(inputs[0], "input" + str(sub['struct_id']),
                                'err' + name, sub['fail'])

        set_out = c_set_tensor4d(out, "output" + str(sub['struct_id']),
                                 'err' + name, sub['fail'])

        return """
cudnnStatus_t err%(name)s;

int %(out)s_dims[4];

if (!CudaNdarray_is_c_contiguous(%(input)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

%(set_in)s

cudnnPoolingMode_t mode;
int wsX, wsY, strideX, strideY;

err%(name)s = cudnnGetPoolingDescriptor(%(desc)s, &mode, &wsX, &wsY, &strideX, &strideY);

if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnGetPoolingDescriptor operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}

%(out)s_dims[0] = CudaNdarray_HOST_DIMS(%(input)s)[0];
%(out)s_dims[1] = CudaNdarray_HOST_DIMS(%(input)s)[1];
%(out)s_dims[2] = (CudaNdarray_HOST_DIMS(%(input)s)[2] - wsX) / strideX + 1;
%(out)s_dims[3] = (CudaNdarray_HOST_DIMS(%(input)s)[3] - wsY) / strideY + 1;

if (CudaNdarray_prep_output(&%(out)s, 4, %(out)s_dims) != 0)
{
  %(fail)s
}

%(set_out)s

err%(name)s = cudnnPoolingForward(
_handle,
%(desc)s,
%(input_desc)s, CudaNdarray_DEV_DATA(%(input)s),
%(output_desc)s, CudaNdarray_DEV_DATA(%(out)s)
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnPoolingForward operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(out=out, desc=desc, fail=sub['fail'], id=sub['struct_id'],
           name=name, set_in=set_in,
           set_out=set_out, input=inputs[0],
           input_desc="input"+str(sub['struct_id']),
           output_desc="output"+str(sub['struct_id']))

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
        return (4,)


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
        inp = as_cuda_ndarray_variable(inp)
        if inp.type.ndim != 4:
            raise TypeError('inp must be 4D tensor')

        inp_grad = as_cuda_ndarray_variable(inp_grad)
        if inp_grad.type.ndim != 4:
            raise TypeError('inp_grad must be 4D tensor')

        out = as_cuda_ndarray_variable(out)
        if out.type.ndim != 4:
            raise TypeError('out must be 4D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnPoolingDescriptor_t':
            raise TypeError('desc must be cudnnPoolingDescriptor_t')

        return Apply(self, [inp, out, inp_grad, desc],
                     [inp.type()])

    def c_support_code_struct(self, node, struct_id):
        return """
cudnnTensor4dDescriptor_t input%(id)d;        
cudnnTensor4dDescriptor_t input_grad%(id)d;
cudnnTensor4dDescriptor_t output%(id)d;
cudnnTensor4dDescriptor_t output_grad%(id)d;
""" % dict(id=struct_id)

    def c_init_code_struct(self, node, struct_id, sub):
        return """
cudnnStatus_t err%(id)d;
input%(id)d = NULL;
input_grad%(id)d = NULL;
output%(id)d = NULL;
output_grad%(id)d = NULL;
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&input%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(input): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&input_grad%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(input_grad): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&output%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(output): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&output_grad%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError,
               "GpuDnnPoolGrad: could not allocate tensor4d descriptor "
               "(output_grad): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, struct_id):
        return """
if (input%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(input%(id)d); }
if (input_grad%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(input_grad%(id)d); }
if (output%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(output%(id)d); }
if (output_grad%(id)d != NULL) { cudnnDestroyTensor4dDescriptor(output_grad%(id)d); }
""" % dict(id=struct_id)

    def c_code(self, node, name, inputs, outputs, sub):
        # Here the name out and inp are based on the cudnn definition.
        # Not the definition of this class.
        # This make it complicated.
        out, inp, inp_grad, desc = inputs
        out_grad, = outputs

        set_in = "\n".join([
            c_set_tensor4d(inp, "input" + str(sub['struct_id']),
                           'err' + name, sub['fail']),
            c_set_tensor4d(inp_grad, "input_grad" + str(sub['struct_id']),
                           'err' + name, sub['fail']),
            c_set_tensor4d(out, "output" + str(sub['struct_id']),
                           'err' + name, sub['fail'])
        ])

        set_out = c_set_tensor4d(out, "output_grad" + str(sub['struct_id']),
                                 'err' + name, sub['fail'])

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

%(set_in)s

if (CudaNdarray_prep_output(&%(output_grad)s, 4,
                            CudaNdarray_HOST_DIMS(%(output)s)) != 0)
{
  %(fail)s
}

%(set_out)s

err%(name)s = cudnnPoolingBackward(
_handle,
%(desc)s,
%(input_desc)s, CudaNdarray_DEV_DATA(%(input)s),
%(input_grad_desc)s, CudaNdarray_DEV_DATA(%(input_grad)s),
%(output_desc)s, CudaNdarray_DEV_DATA(%(output)s),
%(output_grad_desc)s, CudaNdarray_DEV_DATA(%(output_grad)s)
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPoolGrad: error doing operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(output_grad=out_grad, desc=desc,
           fail=sub['fail'], id=sub['struct_id'],
           name=name, set_in=set_in,
           set_out=set_out, input=inp, input_grad=inp_grad, output=out,
           input_desc="input"+str(sub['struct_id']),
           input_grad_desc="input_grad"+str(sub['struct_id']),
           output_desc="output"+str(sub['struct_id']),
           output_grad_desc="output_grad"+str(sub['struct_id']))

    def c_code_cache_version(self):
        return (4,)


def dnn_pool(img, ws, stride=(1, 1), mode='max'):
    """
    GPU pooling using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    :param img: images to do the pooling over
    :param ws: subsampling window size
    :param stride: subsampling stride (default: (1, 1))
    :param mode: one of 'max', 'average' (default: 'max')

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    :note: This Op implements the ignore_border=True of max_pool_2d.
    """
    img = gpu_contiguous(img)
    desc = GpuDnnPoolDesc(ws=ws, stride=stride, mode=mode)()
    return GpuDnnPool()(img, desc)


class GpuDnnSoftmaxBase(DnnBase):
    """
    Op for the cuDNN Softmax.

    :param tensor_format: Whether the data format is 'bc01' or 'b01c'
    :param algo: 'fast' or 'accurate' indicating whether computations should be
        optimized for speed or accuracy respectively.
    :param mode: 'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spationali location '01' per
        image across 'c'.
    """

    __props__ = ('tensor_format', 'mode', 'algo')

    def __init__(self, tensor_format, algo, mode):
        assert(tensor_format in ('bc01', 'b01c'))
        self.tensor_format = tensor_format

        assert(algo in ('fast', 'accurate'))
        self.algo = algo

        assert(mode in ('instance', 'channel'))
        self.mode = mode

        self.tensor_4d_descs = [softmax_input
                                for softmax_input in self.softmax_inputs]
        self.tensor_4d_descs.append('softmax_output')

    def _define_tensor4d_desc(self, name, id):
        return """
cudnnTensor4dDescriptor_t %(name)s_%(id)d;
""" % dict(name=name, id=id)

    def _init_tensor4d_desc(self, name, id, fail):
        return """
%(name)s_%(id)d = NULL;
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&%(name)s_%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               ": %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(name=name, id=id, fail=fail)

    def _clean_tensor4d_desc(self, name, id):
        return """
if(%(name)s_%(id)d!= NULL)
  cudnnDestroyTensor4dDescriptor(%(name)s_%(id)d);
""" % dict(name=name, id=id)

    def c_support_code_struct(self, node, struct_id):
        result = ''
        for name in self.tensor_4d_descs:
            result += self._define_tensor4d_desc(name, struct_id)
        return result

    def c_init_code_struct(self, node, struct_id, sub):
        result = """
cudnnStatus_t err%(id)d;
""" % dict(id=struct_id)

        for name in self.tensor_4d_descs:
            result += self._init_tensor4d_desc(name, struct_id, sub['fail'])
        return result

    def c_cleanup_code_struct(self, node, struct_id):
        result = ''
        for name in self.tensor_4d_descs:
            result += self._clean_tensor4d_desc(name, struct_id)
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
cudnnTensorFormat_t format%(id)d = CUDNN_TENSOR_NCHW;
if (%(tensor_format)d == 1)
  format%(id)d = CUDNN_TENSOR_NHWC;

cudnnSoftmaxAlgorithm_t algo%(id)d = CUDNN_SOFTMAX_ACCURATE;
if (%(algo)d == 1)
  algo%(id)d = CUDNN_SOFTMAX_FAST;

cudnnSoftmaxMode_t mode%(id)d = CUDNN_SOFTMAX_MODE_CHANNEL;
if (%(mode)d == 1)
  mode%(id)d = CUDNN_SOFTMAX_MODE_INSTANCE;
""" % dict(id=sub['struct_id'], name=name,
           tensor_format=tensor_format, mode=mode, algo=algo)

        # Validate the input and build the input variables.
        for input_idx, input_name in enumerate(self.softmax_inputs):
            result += """
if (!CudaNdarray_is_c_contiguous(%(ins)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

err%(name)s = cudnnSetTensor4dDescriptor(
  %(input_name)s_%(id)d,
  format%(id)d,
  CUDNN_DATA_FLOAT,
  CudaNdarray_HOST_DIMS(%(ins)s)[0],
  CudaNdarray_HOST_DIMS(%(ins)s)[1],
  CudaNdarray_HOST_DIMS(%(ins)s)[2],
  CudaNdarray_HOST_DIMS(%(ins)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set tensor4d descriptor: %%%%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(id=sub['struct_id'], name=name, input_name=input_name,
           ins=ins[input_idx], fail=sub['fail'])

        # Build and prepare the output variable.
        result += """
if (CudaNdarray_prep_output(&%(outs)s, 4, CudaNdarray_HOST_DIMS(%(ins)s)) != 0)
{
  %(fail)s
}

err%(name)s = cudnnSetTensor4dDescriptor(
  softmax_output_%(id)d,
  format%(id)d,
  CUDNN_DATA_FLOAT,
  CudaNdarray_HOST_DIMS(%(outs)s)[0],
  CudaNdarray_HOST_DIMS(%(outs)s)[1],
  CudaNdarray_HOST_DIMS(%(outs)s)[2],
  CudaNdarray_HOST_DIMS(%(outs)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set out descriptor: %%%%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
"""

        # Add on a call to the method that does the actual work.
        result += self.method()

        subs = dict(ins=ins[-1], outs=outs, fail=sub['fail'],
                    id=sub['struct_id'], name=name)

        for idx, softmax_input in enumerate(self.softmax_inputs):
            subs['name%d' % idx] = softmax_input
            subs['ins%d' % idx] = inputs[idx]

        return result % subs

    def c_code_cache_version(self):
        return (0, 6)

    def method(self):
        raise NotImplementedError('GpuDnnSoftmaxBase::method')


class GpuDnnSoftmax(GpuDnnSoftmaxBase):
    softmax_inputs = ['softmax_input']

    def make_node(self, x):
        x = as_cuda_ndarray_variable(x)
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def method(self):
        return """
err%(name)s = cudnnSoftmaxForward(
  _handle,
  algo%(id)d,
  mode%(id)d,
  softmax_input_%(id)d,
  CudaNdarray_DEV_DATA(%(ins)s),
  softmax_output_%(id)d,
  CudaNdarray_DEV_DATA(%(outs)s)
);
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
    softmax_inputs = ['softmax_gout', 'softmax_input']

    def make_node(self, dy, sm):
        dy = as_cuda_ndarray_variable(dy)
        sm = as_cuda_ndarray_variable(sm)
        assert dy.ndim == 4
        assert sm.ndim == 4
        return Apply(self, [dy, sm], [sm.type.make_variable()])

    def method(self):
        return """
err%(name)s = cudnnSoftmaxBackward(
  _handle,
  algo%(id)d,
  mode%(id)d,
  %(name1)s_%(id)d,
  CudaNdarray_DEV_DATA(%(ins1)s),
  %(name0)s_%(id)d,
  CudaNdarray_DEV_DATA(%(ins0)s),
  softmax_output_%(id)d,
  CudaNdarray_DEV_DATA(%(outs)s)
);
"""


# Intentation for history
if True:
    #@register_opt('cudnn')  # this optimizer is registered in opt.py instead.
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
            return [dnn_conv(img, kern,
                             border_mode=border_mode, subsample=subsample,
                             direction_hint=direction_hint)]

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
            return [dnn_conv(img, kern,
                             border_mode=border_mode, subsample=subsample,
                             direction_hint=direction_hint)]

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
    @local_optimizer([GpuDownsampleFactorMaxGrad])
    def local_pool_dnn_grad(node):
        if not dnn_available():
            return
        if isinstance(node.op, GpuDownsampleFactorMaxGrad):
            if not node.op.ignore_border:
                return
            inp, out, inp_grad = node.inputs
            ds = node.op.ds

            desc = GpuDnnPoolDesc(ws=ds, stride=ds, mode="max")()
            return [GpuDnnPoolGrad()(gpu_contiguous(inp),
                                     gpu_contiguous(out),
                                     gpu_contiguous(inp_grad),
                                     desc)]

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
        if (
            isinstance(node.op, SoftmaxGrad)
            and (isinstance(node.inputs[0].owner.op, HostFromGpu)
                 or isinstance(node.inputs[1].owner.op, HostFromGpu))
        ):
            if not dnn_available():
                return
            ins = []
            for n in node.inputs:
                if isinstance(n.owner.op, HostFromGpu):
                    n = n.owner.inputs[0]
                ins.append(n.dimshuffle(0, 1, 'x', 'x'))

            out = GpuDnnSoftmaxGrad(
                'bc01',
                'accurate',
                'channel'
            )(
                gpu_contiguous(ins[0]),
                gpu_contiguous(ins[1])
            )
            return [out.dimshuffle(0, 1)]
