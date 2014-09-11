import copy
import os

import theano
from theano import Apply
from theano import tensor
from theano.compat.six import StringIO
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.sandbox.cuda.blas import GpuConv

class GpuDnnConv(GpuOp):
    __props__ = ('border_mode',)

    def __init__(self, border_mode):
        self.border_mode = border_mode

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = (img.type.broadcastable[0],
                         kern.type.broadcastable[0],
                         False, False)

        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_support_code_struct(self, node, struct_id):
        return """
cudnnHandle_t handle%(id)d;
cudnnTensor4dDescriptor_t input%(id)d;
cudnnTensor4dDescriptor_t output%(id)d;
cudnnFilterDescriptor_t kerns%(id)d;
cudnnConvolutionDescriptor_t op%(id)d;
""" % dict(id=struct_id)

    def c_init_code_struct(self, node, struct_id, sub):
        return """
cudnnStatus_t err%(id)d;
if ((err%(id)d = cudnnCreate(&handle%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not create cudnn handle: %%s",
               cudnnGetErrorString(err%(id)d));
  %(fail)s
}
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
if ((err%(id)d = cudnnCreateConvolutionDescriptor(&op%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
               "descriptor: %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, struct_id):
        return """
cudnnDestroyTensor4dDescriptor(input%(id)d);
cudnnDestroyTensor4dDescriptor(output%(id)d);
cudnnDestroyFilterDescriptor(kerns%(id)d);
cudnnDestroyConvolutionDescriptor(op%(id)d);
cudnnDestroy(handle%(id)d);
""" % dict(id=struct_id)

    def c_code(self, node, name, inputs, outputs, sub):
        img, kern = inputs
        out, = outputs

        if self.border_mode == "valid":
            bmode = 1
        else:
            assert self.border_mode == "full"
            bmode = 0

        return """
cudnnStatus_t err%(name)s;
int pad_w%(name)s;
int pad_h%(name)s;

if (!CudaNdarray_is_c_contiguous(%(img)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

if (!CudaNdarray_is_c_contiguous(%(kerns)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous filters are supported.");
  %(fail)s
}

err%(name)s = cudnnSetTensor4dDescriptorEx(
input%(id)d, CUDNN_DATA_FLOAT,
CudaNdarray_HOST_DIMS(%(img)s)[0],
CudaNdarray_HOST_DIMS(%(img)s)[1],
CudaNdarray_HOST_DIMS(%(img)s)[2],
CudaNdarray_HOST_DIMS(%(img)s)[3],
CudaNdarray_HOST_STRIDES(%(img)s)[0],
CudaNdarray_HOST_STRIDES(%(img)s)[1],
CudaNdarray_HOST_STRIDES(%(img)s)[2],
CudaNdarray_HOST_STRIDES(%(img)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set tensor4d descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
err%(name)s = cudnnSetFilterDescriptor(
kerns%(id)d, CUDNN_DATA_FLOAT,
CudaNdarray_HOST_DIMS(%(kerns)s)[0],
CudaNdarray_HOST_DIMS(%(kerns)s)[1],
CudaNdarray_HOST_DIMS(%(kerns)s)[2],
CudaNdarray_HOST_DIMS(%(kerns)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set filter descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if (%(bmode)d == 1) {
  pad_h%(name)s = 0;
  pad_w%(name)s = 0;
} else if (%(bmode)d == 0) {
  pad_h%(name)s = CudaNdarray_HOST_DIMS(%(kerns)s)[2] - 1;
  pad_w%(name)s = CudaNdarray_HOST_DIMS(%(kerns)s)[3] - 1;
} else {
  PyErr_SetString(PyExc_ValueError, "bad border mode");
  %(fail)s
}
err%(name)s = cudnnSetConvolutionDescriptor(
op%(id)d, input%(id)d, kerns%(id)d, 
pad_h%(name)s,
pad_w%(name)s,
1, 1, 1, 1,
CUDNN_CONVOLUTION
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
{
int out_dims[4];
err%(name)s = cudnnGetOutputTensor4dDim(
op%(id)d, CUDNN_CONVOLUTION_FWD,
&out_dims[0], &out_dims[1],
&out_dims[2], &out_dims[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if (CudaNdarray_prep_output(&%(out)s, 4, out_dims) != 0) {
  %(fail)s
}
}
err%(name)s = cudnnSetTensor4dDescriptorEx(
output%(id)d, CUDNN_DATA_FLOAT,
CudaNdarray_HOST_DIMS(%(out)s)[0],
CudaNdarray_HOST_DIMS(%(out)s)[1],
CudaNdarray_HOST_DIMS(%(out)s)[2],
CudaNdarray_HOST_DIMS(%(out)s)[3],
CudaNdarray_HOST_STRIDES(%(out)s)[0],
CudaNdarray_HOST_STRIDES(%(out)s)[1],
CudaNdarray_HOST_STRIDES(%(out)s)[2],
CudaNdarray_HOST_STRIDES(%(out)s)[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set out descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
err%(name)s = cudnnConvolutionForward(
handle%(id)d,
input%(id)d, CudaNdarray_DEV_DATA(%(img)s),
kerns%(id)d, CudaNdarray_DEV_DATA(%(kerns)s),
op%(id)d,
output%(id)d, CudaNdarray_DEV_DATA(%(out)s),
CUDNN_RESULT_NO_ACCUMULATE
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "error doing operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(img=img, kerns=kern, out=out, bmode=bmode,
           fail=sub['fail'], id=sub['struct_id'], name=name)

    def c_code_cache_version(self):
        return (3,)


from theano.sandbox.cuda.opt import (local_optimizer, gpu_contiguous,
                                     gpu_optimizer)

@local_optimizer([GpuConv])
def local_conv_dnn(node):
    if isinstance(node.op, GpuConv):
        if (node.op.subsample != (1, 1) or
            node.op.border_mode not in ['full', 'valid']):
            return
        img, kern = node.inputs
        border_mode = node.op.border_mode
        return [GpuDnnConv(border_mode)(gpu_contiguous(img),
                                        gpu_contiguous(kern))]

gpu_optimizer.register("conv_cudnn", local_conv_dnn, 'cudnn')
