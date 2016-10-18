from __future__ import absolute_import, print_function, division
import numpy
import theano

# Skip test if cuda_ndarray is not available.
try:
    from nose.plugins.skip import SkipTest
    import theano.sandbox.cuda as cuda_ndarray
    if cuda_ndarray.cuda_available is False:
        raise SkipTest('Optional package cuda disabled')
except ImportError:
    # To have the GPU back-end work without nose, we need this file to
    # be importable without nose.
    pass
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
    f = theano.function(inputs=[], outputs=A.sum(), mode=mode_with_gpu,
                        profile=False)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    if sum(isinstance(node.op, B.GpuCAReduce) for node in topo) != 1:
        msg = '\n\t'.join(
            ['Expected exactly one occurrence of GpuCAReduce ' +
             'but got:'] + [str(app) for app in topo])
        raise AssertionError(msg)
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
    f = theano.function([var], var + 1, mode=mode_with_gpu,
                        profile=False)
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(node.op, cuda.GpuElemwise) for node in topo])
    assert theano.sandbox.cuda.use.device_number is not None


def test_nvcc_cast():
    """Test that the casting behaviour is correct.

    Some versions of nvcc, in particular the one in 6.5.14, return an incorrect
    value in this case.

    Reported by Zijung Zhang at
    https://groups.google.com/d/topic/theano-dev/LzHtP2OWeRE/discussion
    """
    var = theano.tensor.fvector()
    f = theano.function([var], -1. * (var > 0), mode=mode_with_gpu)
    if not numpy.allclose(f([-1, 0, 1]), [0, 0, -1]):
        raise Exception(
            "The version of nvcc that Theano detected on your system "
            "has a bug during conversion from integers to floating point. "
            "Installing CUDA 7.0 (or more recent) should fix the problem.")


# TODO make sure the test_nvidia_driver test are executed when we make manually
# a CudaNdarray like this: cuda.CudaNdarray.zeros((5,4))
