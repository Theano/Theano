import copy
import os

import theano
from theano import Apply, tensor
from theano.gof.type import CDataType
from theano.compat import PY3
from theano.compat.six import StringIO
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import (GpuOp, cuda_available, active_device_number,
                                 device_properties)
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.sandbox.cuda.blas import GpuConv
from theano.sandbox.cuda.nnet import GpuSoftmax

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler


def dnn_available():
    if dnn_available.avail is None:
        dev = active_device_number()
        if device_properties(dev)['major'] < 3:
            dnn_available.msg = "Device not supported by cuDNN"
            dnn_available.avail = False
        else:
            dnn_available.msg = "Can not find the cuDNN library"
            dnn_available.avail = theano.gof.cmodule.GCC_compiler.try_flags(
                ["-l", "cudnn"])
    return dnn_available.avail


dnn_available.avail = None
dnn_available.msg = None


class DnnBase(GpuOp):
    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.
    """
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
  PyErr_Format(PyExc_RuntimeError, "could not create cudnn handle: %%s",
               cudnnGetErrorString(err));
  return %s;
}
}""" % (error_out,)]


class GpuDnnConvDesc(GpuOp):
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
        assert border_mode in ('valid', 'full')
        self.border_mode = border_mode
        assert len(subsample) == 2
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

    def make_node(self, img_shape, kern_shape):
        if img_shape.type.ndim != 1 and img_shape.type.dtype != numpy.int64:
            raise TypeError('img must be 1D shape tensor')
        if kern_shape.type.ndim != 1 and kern_shape.type.dtype != numpy.int64:
            raise TypeError('kern must be 1D shape tensor')

        return Apply(self, [img_shape, kern_shape],
                     [CDataType("cudnnConvolutionDescriptor_t")()])

    def c_code(self, node, name, inputs, outputs, sub):
        img_shape, kern_shape = inputs
        desc, = outputs

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

  if (%(bmode)d == 1) {
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
           subsx=self.subsample[0], subsy=self.subsample[1])

    def c_code_cache_version(self):
        return (1,)


class GpuDnnConvBase(DnnBase):
    __props__ = ()

    def make_node(self, img, kern, desc):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = (img.type.broadcastable[0],
                         kern.type.broadcastable[0],
                         False, False)

        return Apply(self, [img, kern, desc],
                     [CudaNdarrayType(broadcastable)()])

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
cudnnDestroyTensor4dDescriptor(input%(id)d);
cudnnDestroyTensor4dDescriptor(output%(id)d);
cudnnDestroyFilterDescriptor(kerns%(id)d);
""" % dict(id=struct_id)

    def c_set_tensor4d(self, var, desc, err, fail):
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
        return (7,)


class GpuDnnConv(GpuDnnConvBase):
    conv_inputs = 'input', 'kerns'
    conv_output = 'output'
    conv_types = 'tensor4d', 'filter', 'tensor4d'
    conv_op = 'cudnnConvolutionForward'
    path_flag = 'CUDNN_CONVOLUTION_FWD'

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
    conv_inputs = 'input', 'output',
    conv_output = 'kerns'
    conv_types = 'tensor4d', 'tensor4d', 'filter'
    path_flag = 'CUDNN_CONVOLUTION_WEIGHT_GRAD'
    conv_op = 'cudnnConvolutionBackwardFilter'


class GpuDnnConvGradI(GpuDnnConvBase):
    conv_inputs = 'kerns', 'output',
    conv_output = 'input'
    conv_types = 'filter', 'tensor4d', 'tensor4d'
    path_flag = 'CUDNN_CONVOLUTION_DATA_GRAD'
    conv_op = 'cudnnConvolutionBackwardData'


def dnn_conv(img, kerns, border_mode='valid', subsample=(1, 1),
             conv_mode='conv'):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    :param img: images to do the convolution over
    :param kerns: convolution filters
    :param border_mode: one of 'valid', 'full' (default: 'valid')
    :param subsample: perform subsampling of the output (default: (1, 1))
    :param conv_mode: perform convolution (kernels flipped) or cross-correlation.  One of 'conv', 'cross'. (default: 'conv')

    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(img.shape, kerns.shape)
    return GpuDnnConv()(img, kerns, desc)


class GpuDnnSoftmax(DnnBase):
    """
    Op for the cuDNN Softmax.

    Parameters''
    -tensor_format: Whether the data format is 'bc01' or 'b01c'
    -algo: 'fast' or 'accurate' indicating whether computations should be
    optimized for speed or accuracy respectively.
    -mode: 'instance' or 'channel' indicating whether the softmax should be
    computed per image across 'c01' or per spationali location '01' per image
    across 'c'.
    """

    __props__ = ('tensor_format', 'mode', 'algo')

    def __init__(self, tensor_format, algo, mode):
        assert(tensor_format in ('bc01', 'b01c'))
        self.tensor_format = tensor_format

        assert(algo in ('fast', 'accurate'))
        self.algo = algo

        assert(mode in ('instance', 'channel'))
        self.mode = mode

    def make_node(self, x):
        x = as_cuda_ndarray_variable(x)
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def c_support_code_struct(self, node, struct_id):
        return """
cudnnTensor4dDescriptor_t softmax_input_%(id)d;
cudnnTensor4dDescriptor_t softmax_output_%(id)d;
""" % dict(id=struct_id)

    def c_init_code_struct(self, node, struct_id, sub):
        return """
softmax_input_%(id)d = NULL;
softmax_output_%(id)d = NULL;

cudnnStatus_t err%(id)d;
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&softmax_input_%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
if ((err%(id)d = cudnnCreateTensor4dDescriptor(&softmax_output_%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(out): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, struct_id):
        return """
if(softmax_input_%(id)d != NULL)
  cudnnDestroyTensor4dDescriptor(softmax_input_%(id)d);

if(softmax_output_%(id)d != NULL)
  cudnnDestroyTensor4dDescriptor(softmax_output_%(id)d);
""" % dict(id=struct_id)

    def c_code(self, node, name, inputs, outputs, sub):
        ins, = inputs
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

        return """
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

if (!CudaNdarray_is_c_contiguous(%(ins)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

err%(name)s = cudnnSetTensor4dDescriptor(
  softmax_input_%(id)d,
  format%(id)d,
  CUDNN_DATA_FLOAT,
  CudaNdarray_HOST_DIMS(%(ins)s)[0],
  CudaNdarray_HOST_DIMS(%(ins)s)[1],
  CudaNdarray_HOST_DIMS(%(ins)s)[2],
  CudaNdarray_HOST_DIMS(%(ins)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set tensor4d descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}

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
  PyErr_Format(PyExc_RuntimeError, "could not set out descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}

err%(name)s = cudnnSoftmaxForward(
  _handle,
  algo%(id)d,
  mode%(id)d,
  softmax_input_%(id)d,
  CudaNdarray_DEV_DATA(%(ins)s),
  softmax_output_%(id)d,
  CudaNdarray_DEV_DATA(%(outs)s)
);
""" % dict(ins=ins, outs=outs, tensor_format=tensor_format, mode=mode,
           algo=algo, fail=sub['fail'], id=sub['struct_id'], name=name)

    def c_code_cache_version(self):
        return (0, 3)


# We need this since other stuff from opt is not importable.
if cuda_available:

    from theano.sandbox.cuda.opt import (local_optimizer, gpu_contiguous,
                                         gpu_optimizer)

    @local_optimizer([GpuConv])
    def local_conv_dnn(node):
        if isinstance(node.op, GpuConv):
            if node.op.border_mode not in ['full', 'valid']:
                return
            img, kern = node.inputs
            border_mode = node.op.border_mode
            subsample = node.op.subsample
            return [dnn_conv(gpu_contiguous(img), gpu_contiguous(kern),
                             border_mode=border_mode, subsample=subsample)]

    gpu_optimizer.register("conv_cudnn", local_conv_dnn, 'cudnn')


    @local_optimizer([GpuSoftmax])
    def local_softmax_dnn(node):
        if isinstance(node.op, GpuSoftmax):
            ins = node.inputs[0].dimshuffle(0, 1, 'x', 'x')
            out = GpuDnnSoftmax('bc01', 'accurate', 'channel')(gpu_contiguous(ins))
            out = as_cuda_ndarray_variable(out.dimshuffle(0, 1))
            return [out]

    gpu_optimizer.register("softmax_cudnn", local_softmax_dnn, 'cudnn')
