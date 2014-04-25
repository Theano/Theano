import numpy

import theano
from theano import tensor
from theano.tests import unittest_tools as utt
import theano.sandbox.gpuarray
from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.basic_ops import GpuAlloc, GpuReshape, gpu_alloc
from theano.sandbox.gpuarray.elemwise import GpuCAReduceCuda
from theano.sandbox.gpuarray.tests.test_basic_ops import (
    rand_gpuarray, mode_with_gpu, mode_without_gpu
    )
from theano.tests.unittest_tools import SkipTest

def test_flatten():
    m = theano.tensor.fmatrix()
    f = theano.function([m], m.flatten(), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val)
    assert res.shape == val.shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    m = theano.tensor.tensor3()
    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11, 12).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.reshape(10, -1))
    assert res.shape == val.reshape(10, -1).shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]


def test_sum_prod():
    for method in ['sum']:
        m = theano.tensor.fmatrix()
        f = theano.function([m], getattr(m, method)(), mode=mode_with_gpu)
        val = numpy.random.rand(10, 11).astype("float32")
        res = f(val)
        utt.assert_allclose(res, val.sum())
        assert res.shape == ()
        assert GpuCAReduceCuda in [type(node.op)
                                   for node in f.maker.fgraph.toposort()]


def test_local_gpualloc_memset_0():
    i = theano.tensor.iscalar()
    z = numpy.zeros((1,), dtype='float32')
    o = numpy.ones((1,), dtype='float32')
    ones = numpy.ones((2,), dtype='float32')

    # Test with 0
    a = gpu_alloc(z, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc) and topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 0).all()

    # Test with 1
    a = gpu_alloc(o, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 1).all()

    # Test with 1, 1
    a = gpu_alloc(ones, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(2)) == 1).all()


def test_rebroadcast():
    d = numpy.random.rand(10, 10).astype('float32')
    v = theano.tensor.fmatrix()
    up = tensor.unbroadcast(v.sum().dimshuffle('x', 'x'), 0, 1)
    f = theano.function([v], [up], mode=mode_with_gpu)

    f(d)

    topo = f.maker.fgraph.toposort()
    rebrs = [node for node in topo if isinstance(node.op, tensor.Rebroadcast)]
    assert len(rebrs) == 1
    rebr = rebrs[0]

    assert isinstance(rebr.inputs[0].type, GpuArrayType)
    assert isinstance(rebr.outputs[0].type, GpuArrayType)
