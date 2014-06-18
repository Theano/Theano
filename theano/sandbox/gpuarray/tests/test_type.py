import operator

import numpy

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


def test_values_eq_approx():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')
    assert GpuArrayType.values_eq_approx(a, a)
    b = a.copy()
    b[0] = numpy.asarray(b[0]) + 1.
    assert not GpuArrayType.values_eq_approx(a, b)
    b = a.copy()
    b[0] = -numpy.asarray(b[0])
    assert not GpuArrayType.values_eq_approx(a, b)


def test_specify_shape():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')
    f = theano.function([g], theano.tensor.specify_shape(g, [20]))
    f(a)
