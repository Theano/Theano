import copy
import logging
import StringIO
import sys

import numpy

import theano
from theano import Op, Type, Apply, Variable, Constant, Generic
from theano import tensor, scalar, config

from theano.gof.python25 import all, any

from theano.sandbox.cuda import GpuOp, device_properties
from theano.sandbox.cuda.basic_ops import (HostFromGpu, GpuFromHost, host_from_gpu,
        gpu_from_host)
from theano.sandbox.cuda.opt import gpu_seqopt

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.opt import gpu_seqopt

_logger_name = 'theano.sandbox.cuda.async'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())  # TO REMOVE


##################################
# Asynchronous GPU Communication #
##################################

class HostFromGpuSend(GpuOp):
    """
    Start the transfer from gpu to the cpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'HostFromGpuSend'

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype,
                                    broadcastable=x.broadcastable)(),
                                 theano.Variable(Generic())])

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out, event = outputs
        fail = sub['fail']
        eventName = "%s_event"%out
        return """
        cudaEvent_t *%(eventName)s = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        PyObject *%(event)s = PyCObject_FromVoidPtr((void *)(%(eventName)s), &free_cudaEvent);
        cudaEventCreate(%(eventName)s);
        %(out)s = (PyArrayObject *) CudaNdarray_CreateArrayObj(%(inp)s);
        cudaEventRecord(*%(eventName)s, 0);
        """ % locals()

    def c_code_cache_version(self):
        return (1,)

class HostFromGpuWait(GpuOp):
    """
    Implement the transfer from gpu to the cpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'HostFromGpuWait'

    def make_node(self, x, event):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x, event], [tensor.TensorType(dtype=x.dtype,
                                         broadcastable=x.broadcastable)()])

    def infer_shape(self, node, xshp):
        return [xshp[0]]

    def c_code(self, node, name, inputs, outputs, sub):
        print inputs
        print type(inputs)
        inp, event = inputs
        out = outputs[0]
        fail = sub['fail']
        # eventName = "%s_event"%event
        return """
        cudaEventSynchronize(*(cudaEvent_t)(PyCObject_AsVoidPtr(%(event)s)));

        %(out)s = %(inp)s;
        if(!%(out)s){
            %(fail)s;
        }
        """ % locals()

    def c_code_cache_version(self):
        return (1,)

class GpuFromHostSend(GpuOp):
    """
    Start the transfer from cpu to the gpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'GpuFromHost'

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable,
                                                 dtype=x.dtype)(),
                                 theano.Variable(Generic())])

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out, event = outputs
        fail = sub['fail']
        eventName = "%s_event"%out
        return """
        int err = 0;
        cudaEvent_t *%(eventName)s = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        PyObject *%(event)s = PyCObject_FromVoidPtr((void *)(%(eventName)s), &free_cudaEvent);
        cudaEventCreate(%(eventName)s);
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray*) CudaNdarray_New();
        if(!%(out)s){
            %(fail)s;
        }
        err = CudaNdarray_CopyFromArray(%(out)s, %(inp)s);

        // This should probably happen after synchronization
        if(err){
            %(fail)s;
        }
        cudaEventRecord(*%(eventName)s, 0);
        """ % locals()

    def c_code_cache_version(self):
        return (1,)

class GpuFromHostWait(GpuOp):
    """
    Wait for completion of the transfer from cpu to the gpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'GpuFromHostWait'

    def make_node(self, x, event):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x, event],
                    [CudaNdarrayType(broadcastable=x.broadcastable,
                                     dtype=x.dtype)()])

    def infer_shape(self, node, xshp):
        return [xshp[0]]

    def c_code(self, node, name, inputs, outputs, sub):
        inp, event = inputs
        out = outputs[0]
        fail = sub['fail']
        # eventName = "GpuFromHost_%s_event"%inp

        return """
        cudaEventSynchronize(*(cudaEvent_t*)(PyCObject_AsVoidPtr(%(event)s)));
        %(out)s = %(inp)s;
        """ % locals()

    def c_code_cache_version(self):
        return (1,)

# TODO Register
@theano.gof.local_optimizer([host_from_gpu, gpu_from_host])
def local_async_gpu(node):
    if isinstance(node.op, HostFromGpu):
        return [HostFromGpuWait()(*HostFromGpuSend()(node.inputs[0]))]
    if isinstance(node.op, GpuFromHost):
        return [GpuFromHostWait()(*GpuFromHostSend()(node.inputs[0]))]
    return False
async_optimizer = theano.gof.TopoOptimizer(local_async_gpu)


# gpu_seqopt.register('local_async_gpu', local_async_gpu, 3, 'fast_run', 'gpu')

#gpu_seqopt.register('local_async_gpu',
#                    theano.tensor.opt.in2out(local_async_gpu), 3,
#                    'fast_run', 'gpu')
gpu_seqopt.register('local_async_gpu', theano.tensor.opt.in2out([local_async_gpu]), 3,
        'fast_run', 'gpu')
