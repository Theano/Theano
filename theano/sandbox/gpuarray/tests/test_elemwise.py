import unittest

from theano import scalar, gof
from theano.gof import FunctionGraph
from theano.gof.python25 import all, any
from theano.tests.unittest_tools import SkipTest

from theano.tensor.tests.test_elemwise import (test_Broadcast, test_DimShuffle,
                                               test_CAReduce)

from theano.sandbox.gpuarray.tests.test_basic_ops import rand_gpuarray
from theano.sandbox.gpuarray.elemwise import (GpuElemwise, GpuDimShuffle,
                                              GpuCAReduce)
from theano.sandbox.gpuarray.type import GpuArrayType

from pygpu.array import gpuarray

# This is acutally a test for GpuElemwise
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


class test_GpuDimShuffle(test_DimShuffle):
    op = GpuDimShuffle

class test_GpuCAReduce(test_CAReduce):
    dtypes = ["float32"]
    bin_dtypes = ["uint8", "int8"]
    op = GpuCAReduce
    reds = [scalar.add, scalar.mul]

    def test_perform(self):
        for dtype in self.dtypes + self.bin_dtypes:
            for op in self.reds:
                self.with_linker(gof.PerformLinker(), op, dtype=dtype)

    def test_perform_nan(self):
        for dtype in self.dtypes:
            for op in self.reds:
                self.with_linker(gof.PerformLinker(), op, dtype=dtype,
                                 test_nan=True)

    def test_c(self):
        raise SkipTest("no C code")

    def test_c_nan(self):
        raise SkipTest("no C code")
