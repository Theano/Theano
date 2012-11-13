from nose.plugins.skip import SkipTest

import numpy
import theano.tensor as T

import theano.sandbox.gpuarray
if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu disabled")

from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.basic_ops import (host_from_gpu, gpu_from_host)

from pygpu import gpuarray

def rand_gpuarray(shape, dtype):
    return gpuarray.array(numpy.random.rand(*shape), dtype=dtype)

def test_transfer():
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
    
    av = numpy.asarray(numpy.random.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av, kind=g.type.kind, context=g.type.context)
    
    f = theano.function([a], gpu_from_host(a))
    fv = f(av)
    assert GpuArrayType.values_eq_approx(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert numpy.allclose(fv, av)
