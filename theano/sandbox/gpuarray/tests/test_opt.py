import numpy

import theano
from theano.tests import unittest_tools as utt
from theano.sandbox.gpuarray.basic_ops import GpuReshape
import theano.sandbox.gpuarray

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
    val = numpy.random.rand(10,11).astype("float32")
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
