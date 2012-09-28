import copy
import logging
import StringIO
import sys
import os

import numpy

import theano
from theano import Apply, Variable, Generic
from theano import tensor, config

from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (HostFromGpu, GpuFromHost,
                                           host_from_gpu, gpu_from_host)
from theano.sandbox.cuda.opt import gpu_seqopt

from theano.sandbox.cuda.type import CudaNdarrayType
import cuda_ndarray
from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
from theano.gof.sched import key_to_cmp

_logger_name = 'theano.sandbox.cuda.async'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())  # TO REMOVE


##################################
# Asynchronous GPU Communication #
##################################

class GpuAsyncTransferOp(GpuOp):
    """ Shared code between Gpu Asynchronous Transfer Ops

    See Also:
        HostFromGpuSend
        HostFromGpuWait
        GpuFromHostSend
        GpuFromHostWait
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def c_code_cache_version(self):
        return ()

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<cuda.h>']

    def c_header_dirs(self):
        """Override `CLinkerOp.c_headers` """
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret
    def c_lib_dirs(self):
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_libraries(self):
        # returning cublas because the cuda_ndarray.cuh header
        # includes calls to SetVector and cublasGetError
        return ['cudart', 'cublas']

    def c_compiler(self):
        return NVCC_compiler


class HostFromGpuSend(GpuAsyncTransferOp):
    """
    Start the transfer from gpu to the cpu.
    """

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype,
                                    broadcastable=x.broadcastable)(),
                                 theano.Variable(Generic())])

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out, event = outputs
        fail = sub['fail']
        eventName = "%s_event" % out
        return """
        cudaEvent_t *%(eventName)s = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        %(event)s = PyCObject_FromVoidPtr((void *)(%(eventName)s), &free_cudaEvent);
        cudaEventCreate(%(eventName)s);
        %(out)s = (PyArrayObject *) CudaNdarray_CreateArrayObj(%(inp)s);
        cudaEventRecord(*%(eventName)s, 0);
        """ % locals()


class HostFromGpuWait(GpuAsyncTransferOp):
    """
    Implement the transfer from gpu to the cpu.
    """
    def make_node(self, x, event):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x, event], [tensor.TensorType(dtype=x.dtype,
                                         broadcastable=x.broadcastable)()])
    view_map = {0: [0]}

    def c_code(self, node, name, inputs, outputs, sub):
        inp, event = inputs
        out = outputs[0]
        fail = sub['fail']
        # eventName = "%s_event"%event
        return """
        cudaError_t err = cudaSuccess;
        cudaEvent_t* tmp = (cudaEvent_t*)(PyCObject_AsVoidPtr(%(event)s));
        if(!tmp){
           PyErr_Format(PyExc_ValueError,
                        "HostFromGpuWait: Received bad event ptr %%p", tmp);
           %(fail)s;
        }
        err = cudaEventSynchronize(*tmp);
        if(err != cudaSuccess){
            %(fail)s;
        }

        %(out)s = %(inp)s;
        Py_INCREF(%(inp)s);
        if(!%(out)s){
            %(fail)s;
        }
        """ % locals()

class GpuFromHostSend(GpuAsyncTransferOp):
    """
    Start the transfer from cpu to the gpu.
    """
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable,
                                                 dtype=x.dtype)(),
                                 theano.Variable(Generic())])

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out, event = outputs
        fail = sub['fail']
        eventName = "%s_event" % out
        return """
        int err = 0;
        cudaEvent_t *%(eventName)s = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        %(event)s = PyCObject_FromVoidPtr((void *)(%(eventName)s), &free_cudaEvent);
        cudaEventCreate(%(eventName)s);
        // Py_XDECREF(%(out)s);
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


class GpuFromHostWait(GpuAsyncTransferOp):
    """
    Wait for completion of the transfer from cpu to the gpu.
    """

    def make_node(self, x, event):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x, event],
                    [CudaNdarrayType(broadcastable=x.broadcastable,
                                     dtype=x.dtype)()])
    view_map = {0: [0]}

    def c_code(self, node, name, inputs, outputs, sub):
        inp, event = inputs
        out = outputs[0]
        fail = sub['fail']
        # eventName = "GpuFromHost_%s_event"%inp

        return """
        cudaError_t err = cudaSuccess;
        cudaEvent_t* tmp = (cudaEvent_t*)(PyCObject_AsVoidPtr(%(event)s));
        if(!tmp){
           PyErr_Format(PyExc_ValueError,
                        "HostFromGpuWait: Received bad event ptr %%p", tmp);
           %(fail)s;
        }
        err = cudaEventSynchronize(*tmp);
        if(err != cudaSuccess){
            %(fail)s;
        }

        %(out)s = %(inp)s;
        Py_INCREF(%(inp)s);
        """ % locals()

@theano.gof.local_optimizer([host_from_gpu, gpu_from_host])
def local_async_gpu(node):
    if isinstance(node.op, HostFromGpu):
        return [HostFromGpuWait()(*HostFromGpuSend()(node.inputs[0]))]
    if isinstance(node.op, GpuFromHost):
        return [GpuFromHostWait()(*GpuFromHostSend()(node.inputs[0]))]
    return False

async_optimizer = theano.gof.TopoOptimizer(local_async_gpu)

gpu_seqopt.register('local_async_gpu',
                    theano.tensor.opt.in2out(local_async_gpu),
                    3, 'fast_run', 'gpu')

# GPU Scheduling Comparators

def send_wait(a):
    """ Wait as long as possible on Waits. Start Send/Recvs early """
    if isinstance(a.op, (GpuFromHostWait, HostFromGpuWait)):
        return 1
    if isinstance(a.op, (GpuFromHostSend, HostFromGpuSend)):
        return -1
    return 0

def gpu_ops_first(a):
    """ Do GpuOps first. They don't block. """
    if isinstance(a.op, GpuOp):
        return -1
    return 0

def send_in_order(a, b):
    """ Send variables in the order in which they are needed """
    from theano.gof.sched import make_dependence_cmp
    dependence = make_dependence_cmp()
    if ((isinstance(a, GpuFromHostSend) and isinstance(b, GpuFromHostSend)) or
        (isinstance(a, HostFromGpuSend) and isinstance(b, HostFromGpuSend))):
        return dependence(a.inputs[0].owner, b.inputs[0].owner)
    return 0

def tiebreaker(a):
    """
    Break ties by the maximum var-name on which the node depends
    """
    if not a or not a.inputs:
        return 'ZZZ'
    return max(inp.name or tiebreaker(inp.owner) for inp in a.inputs)

gpu_cmps = [key_to_cmp(send_wait),
            key_to_cmp(gpu_ops_first),
            send_in_order,
            key_to_cmp(tiebreaker)]
