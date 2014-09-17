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
from theano.compat import PY3


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


class GpuDnnConvBase(DnnBase):
    __props__ = ('border_mode', 'subsample', 'conv_mode')

    def __init__(self, border_mode, subsample=(1, 1), conv_mode='conv'):
        assert border_mode in ('valid', 'full')
        self.border_mode = border_mode
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

    def __setstate__(self, props):
        self.__dict__.update(props)
        if not hasattr(self, 'conv_mode'):
            self.conv_mode = 'conv'
        if not hasattr(self, 'subsample'):
            self.subsample = (1, 1)

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = (img.type.broadcastable[0],
                         kern.type.broadcastable[0],
                         False, False)

        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def c_support_code_struct(self, node, struct_id):
        types = ['cudnn' + d.capitalize() + 'Descriptor_t'
                 for d in self.descriptors]
        elems = [t + ' param%d_%d;' % (i, struct_id)
                 for i, t in enumerate(types)]
        return ("cudnnConvolutionDescriptor_t op%d;\n" % (struct_id,) +
                '\n'.join(elems))

    def c_init_code_struct(self, node, struct_id, sub):
        vnames = ['param%d_%d' % (i, struct_id)
                  for i, t in enumerate(self.descriptors)]
        inits = [vname + '= NULL;' for vname in vnames]
        creates = []
        for d, var in zip(self.descriptors, vnames):
            creates.append("""
if ((err%(id)d = cudnnCreate%(d)sDescriptor(&%(var)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, d=d.capitalize(), var=var, fail=sub['fail']))

        return """
%(init)s
cudnnStatus_t err%(id)d;
%(create)s
if ((err%(id)d = cudnnCreateConvolutionDescriptor(&op%(id)d)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
               "descriptor: %%s", cudnnGetErrorString(err%(id)d));
  %(fail)s
}
""" % dict(id=struct_id, fail=sub['fail'], init='\n'.join(inits),
           create='\n'.join(creates))

    def c_cleanup_code_struct(self, node, struct_id):
        cleanups = ['cudnnDestroy%sDescriptor(param%d_%d);' % (d.capitalize(),
                                                               i, struct_id)
                    for i, d in enumerate(self.descriptors)]
        return """
%(cleanup)s
cudnnDestroyConvolutionDescriptor(op%(id)d);
""" % dict(id=struct_id, cleanup='\n'.join(cleanups))

    def c_set_tensor4d(self, var, desc, err, fail):
        return """
%(err)s = cudnnSetTensor4dDescriptorEx(
%(desc)s, CUDNN_DATA_FLOAT,
CudaNdarray_HOST_DIMS(%(var)s)[0],
CudaNdarray_HOST_DIMS(%(var)s)[1],
CudaNdarray_HOST_DIMS(%(var)s)[2],
CudaNdarray_HOST_DIMS(%(var)s)[3],
CudaNdarray_HOST_STRIDES(%(var)s)[0],
CudaNdarray_HOST_STRIDES(%(var)s)[1],
CudaNdarray_HOST_STRIDES(%(var)s)[2],
CudaNdarray_HOST_STRIDES(%(var)s)[3]
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
        param0, param1 = inputs
        out, = outputs

        if self.border_mode == "valid":
            bmode = 1
        else:
            assert self.border_mode == "full"
            bmode = 0

        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'


        vnames = ['param%d_%d' % (i, sub['struct_id'])
                  for i, t in enumerate(self.descriptors)]
        checks = []
        for v in (param0, param1):
            checks.append("""
if (!CudaNdarray_is_c_contiguous(%s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %s
}
""" % (v, sub['fail']))

        sets = []
        for p, v, d in zip((param0, param1), vnames[:-1],
                           self.descriptors[:-1]):
            sets.append(getattr(self, 'c_set_'+d)(p, v, 'err'+name,
                                                  sub['fail']))

        set_out = getattr(self, 'c_set_'+self.descriptors[-1])(
            out, vnames[-1], 'err'+name, sub['fail'])

        return """
cudnnStatus_t err%(name)s;
int pad_w%(name)s;
int pad_h%(name)s;

%(checks)s

%(sets)s

if (%(bmode)d == 1) {
  pad_h%(name)s = 0;
  pad_w%(name)s = 0;
} else if (%(bmode)d == 0) {
  pad_h%(name)s = CudaNdarray_HOST_DIMS(%(param1)s)[2] - 1;
  pad_w%(name)s = CudaNdarray_HOST_DIMS(%(param1)s)[3] - 1;
} else {
  PyErr_SetString(PyExc_ValueError, "bad border mode");
  %(fail)s
}
err%(name)s = cudnnSetConvolutionDescriptor(
op%(id)d, param0_%(id)d, param1_%(id)d,
pad_h%(name)s,
pad_w%(name)s,
%(subsx)d, %(subsy)d, 1, 1,
%(conv_flag)s
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
{
int out_dims[4];
err%(name)s = cudnnGetOutputTensor4dDim(
op%(id)d, %(path)s,
&out_dims[0], &out_dims[1],
&out_dims[2], &out_dims[3]
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "could not get output sizes: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if (CudaNdarray_prep_output(&%(out)s, 4, out_dims) != 0) {
  %(fail)s
}
}

%(set_out)s

err%(name)s = %(method)s(
_handle,
param0_%(id)d, CudaNdarray_DEV_DATA(%(param0)s),
param1_%(id)d, CudaNdarray_DEV_DATA(%(param1)s),
op%(id)d,
param2_%(id)d, CudaNdarray_DEV_DATA(%(out)s),
CUDNN_RESULT_NO_ACCUMULATE
);
if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError, "error doing operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(param0=param0, param1=param1, out=out, bmode=bmode,
           conv_flag=conv_flag, fail=sub['fail'], id=sub['struct_id'],
           name=name, checks='\n'.join(checks), sets='\n'.join(sets),
           subsx=self.subsample[0], subsy=self.subsample[1],
           set_out=set_out, method=self.conv_op, path=self.path_flag)

    def c_code_cache_version(self):
        return (6,)


class GpuDnnConv(GpuDnnConvBase):
    descriptors = ('tensor4d', 'filter', 'tensor4d')
    path_flag = 'CUDNN_CONVOLUTION_FWD'
    conv_op ='cudnnConvolutionForward'

    def grad(self, inp, grads):
        img, kerns = inp
        top, = grads

        d_img = GpuDnnConvGradI(self.border_mode, self.conv_mode)(kerns, top)
        d_kerns = GpuDnnConvGradW(self.border_mode, self.conv_mode)(img, top)

        return d_img, d_kerns


class GpuDnnConvGradW(GpuDnnConvBase):
    descriptors = ('tensor4d', 'tensor4d', 'filter')
    path_flag = 'CUDNN_CONVOLUTION_WEIGHT_GRAD'
    conv_op = 'cudnnConvolutionBackwardFilter'


class GpuDnnConvGradI(GpuDnnConvBase):
    descriptors = ('filter', 'tensor4d', 'tensor4d')
    path_flag = 'CUDNN_CONVOLUTION_DATA_GRAD'
    conv_op = 'cudnnConvolutionBackwardData'


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
        return [GpuDnnConv(border_mode, subsample)(gpu_contiguous(img),
                                                   gpu_contiguous(kern))]

gpu_optimizer.register("conv_cudnn", local_conv_dnn, 'cudnn')
