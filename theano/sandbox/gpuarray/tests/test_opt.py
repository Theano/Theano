import numpy

import theano
from theano.tests import unittest_tools as utt
from theano.sandbox.gpuarray.basic_ops import GpuAlloc, GpuReshape, gpu_alloc
from theano.sandbox.gpuarray.elemwise import GpuCAReduce
import theano.sandbox.gpuarray
from theano.tests.unittest_tools import SkipTest

if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available and not theano.sandbox.gpuarray.pygpu_activated:
    if not cuda_ndarray.use.device_number:
        cuda_ndarray.use('gpu')
    theano.sandbox.gpuarray.init_dev('cuda')

if not theano.sandbox.gpuarray.pygpu_activated:
    raise SkipTest("pygpu disabled")

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')


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
        assert GpuCAReduce in [type(node.op)
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
