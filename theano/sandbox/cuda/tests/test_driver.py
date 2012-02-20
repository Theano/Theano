import numpy
import theano

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as cuda
import theano.sandbox.cuda.basic_ops as B


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def test_nvidia_driver1():
    """ Some nvidia driver give bad result for reduction
        This execute some reduction test to ensure it run correctly
    """
    a = numpy.random.rand(10000).astype("float32")
    A = cuda.shared_constructor(a)
    f = theano.function(inputs=[], outputs=A.sum(), mode=mode_with_gpu)
    topo = f.maker.env.toposort()
    assert len(topo) == 2
    assert sum(isinstance(node.op, B.GpuSum) for node in topo) == 1
    if not numpy.allclose(f(), a.sum()):
        raise Exception("The nvidia driver version installed with this OS "
                        "does not give good results for reduction."
                        "Installing the nvidia driver available on the same "
                        "download page as the cuda package will fix the "
                        "problem: http://developer.nvidia.com/cuda-downloads")


def test_nvidia_driver2():
    """ Test that the gpu device is initialized by theano when
        we manually make a shared variable on the gpu.

        The driver should always be tested during theano initialization
        of the gpu device
    """
    a = numpy.random.rand(10000).astype("float32")
    cuda.shared_constructor(a)
    assert theano.sandbox.cuda.use.device_number is not None


def test_nvidia_driver3():
    """ Test that the gpu device is initialized by theano when
        we build a function with gpu op.

        The driver should always be tested during theano initialization
        of the gpu device
    """
    var = cuda.fvector()
    f = theano.function([var], var + 1, mode=mode_with_gpu)
    topo = f.maker.env.toposort()
    assert any([isinstance(node.op, cuda.GpuElemwise) for node in topo])
    assert theano.sandbox.cuda.use.device_number is not None

# TODO make sure the test_nvidia_driver test are executed when we make manually
# a CudaNdarray like this: cuda.CudaNdarray.zeros((5,4))
