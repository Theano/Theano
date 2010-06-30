import numpy
import theano
from theano import shared, function
import theano.tensor as T
from neighbours import images2neibs, neibs2images, GpuImages2Neibs
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda

if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

def test_neibs():
    shape = (100,40,18,18)
    images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
    neib_shape = T.as_tensor_variable((2,2))#(array((2,2), dtype='float32'))

    f = function([], images2neibs(images, neib_shape), mode=mode_without_gpu)

    #print images.value
    neibs = f()
    #print neibs
    g = function([], neibs2images(neibs, neib_shape, images.shape), mode=mode_without_gpu)
    
    #print g()
    assert numpy.allclose(images.value,g())

def test_neibs_gpu():
    if cuda.cuda_available == False:
       raise SkipTest('Optional package cuda disabled')
   
    shape = (100,40,18,18)
    images = shared(numpy.arange(numpy.prod(shape), dtype='float32').reshape(shape))
    neib_shape = T.as_tensor_variable((2,2))#(array((2,2), dtype='float32'))

    from theano.sandbox.cuda.basic_ops import gpu_from_host

    f = function([], images2neibs(images,neib_shape),
                 mode=mode_with_gpu)
    f_gpu = function([], images2neibs(images,neib_shape),
                 mode=mode_with_gpu)
    assert any([isinstance(node.op,GpuImages2Neibs) for node in f_gpu.maker.env.toposort()])
    #print images.value
    neibs = numpy.asarray(f_gpu())
    assert numpy.allclose(neibs,f())
    #print neibs
    g = function([], neibs2images(neibs, neib_shape, images.shape), mode=mode_with_gpu)
    assert any([isinstance(node.op,GpuImages2Neibs) for node in f.maker.env.toposort()])
    #print numpy.asarray(g())
    assert numpy.allclose(images.value,g())


def speed_neibs():
    shape = (100,40,18,18)
    images = shared(numpy.arange(numpy.prod(shape), dtype='float32').reshape(shape))
    neib_shape = T.as_tensor_variable((2,2))#(array((2,2), dtype='float32'))

    from theano.sandbox.cuda.basic_ops import gpu_from_host

    f = function([], images2neibs(images,neib_shape))#, mode=mode_without_gpu)
  
    for i in range(1000):
        f()
        


if __name__ == '__main__':
    test_neibs_gpu()
    test_neibs()
