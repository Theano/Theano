import unittest

from theano import scalar
from theano.gof import FunctionGraph
from theano.gof.python25 import all, any

from theano.tensor.tests.test_elemwise import test_Broadcast

from theano.sandbox.gpuarray.tests.test_basic_ops import rand_gpuarray
from theano.sandbox.gpuarray.elemwise import GpuElemwise
from theano.sandbox.gpuarray.type import GpuArrayType

from pygpu.array import gpuarray

class test_gpu_Broadcast(test_Broadcast):
    op = GpuElemwise
    type = GpuArrayType
    
    def rand_val(self, shp):
        return rand_gpuarray(*shp, **dict(cls=gpuarray))

    # no c_code() yet
    #cop = GpuElemwise
    #ctype = GpuArrayType

    #def rand_cval(self, shp):
    #    return rand_gpuarray(*shp, **dict(cls=gpuarray))

