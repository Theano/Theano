import numpy as np
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

from theano.sandbox.cuda.async import (local_async_gpu, GpuFromHostWait,
                                       HostFromGpuWait,
                                       async_optimizer)
from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
import theano
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


def test_async_to_gpu():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)
    assert len(gx2) == 1
    gx2 = gx2[0]
    assert (gx.dtype, type(gx)) == (gx2.dtype, type(gx2))
    assert isinstance(gx2.owner.op, GpuFromHostWait)


def test_async_to_host():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    y = theano.sandbox.cuda.host_from_gpu(gx)
    y2 = local_async_gpu.transform(y.owner)
    assert len(y2) == 1
    y2 = y2[0]

    assert (y.dtype, type(y)) == (y2.dtype, type(y2))
    assert isinstance(y2.owner.op, HostFromGpuWait)


def test_compile():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    theano.function([x], gx2, mode=theano.Mode(optimizer=None, linker='c|py'))


def test_execute():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    f = theano.function([x], gx2, mode=theano.Mode(optimizer=None,
                                                   linker='c|py'))
    xx = np.ones((5, 5), dtype=x.dtype)
    f(xx)


def test_optimizer():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    fgraph = theano.FunctionGraph([x], [gx])
    async_optimizer.optimize(fgraph)


def test_optimizer2():
    """ Test that the optimization is correctly registered"""
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    mode = theano.Mode(linker='c|py').including("local_async_gpu", 'gpu')
    f = theano.function([x], gx, mode=mode)
    xx = np.ones((5, 5), dtype=x.dtype)
    f(xx)
