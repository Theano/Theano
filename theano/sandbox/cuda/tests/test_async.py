from theano.sandbox.cuda.basic_ops import async_gpu, GpuFromHostWait
import theano

def test_async_to_gpu():
    x = theano.tensor.matrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = async_gpu.transform(gx.owner)

    assert (gx.dtype, type(gx)) == (gx2.dtype, type(gx2))
    assert isinstance(gx2.owner.op, GpuFromHostWait)
