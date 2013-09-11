import operator

import theano
from theano.compile import DeepCopyOp

from theano.sandbox.gpuarray.tests.test_basic_ops import rand_gpuarray

from theano.sandbox.gpuarray.type import GpuArrayType

def test_deep_copy():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')

    f = theano.function([g], g)

    assert isinstance(f.maker.fgraph.toposort()[0].op, DeepCopyOp)

    res = f(a)

    assert GpuArrayType.values_eq(res, a)
