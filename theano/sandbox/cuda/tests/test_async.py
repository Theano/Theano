from theano.sandbox.cuda.async import local_async_gpu, GpuFromHostWait
from theano.sandbox.cuda.basic_ops import gpu_from_host
import theano

def test_async_to_gpu():
    x = theano.tensor.matrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    assert (gx.dtype, type(gx)) == (gx2.dtype, type(gx2))
    assert isinstance(gx2.owner.op, GpuFromHostWait)

def test_transfer():
    x = theano.tensor.matrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    theano.function([x], gx2, mode=theano.Mode(optimizer=None, linker='c|py'))
