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
    assert any([isinstance(node.op,GpuImages2Neibs) for node in f.maker.env.toposort()])
    #print images.value
    res1=[[[[  0.,   1.,   4.,   5.],
         [  2.,   3.,   6.,   7.],
         [  8.,   9.,  12.,  13.],
         [ 10.,  11.,  14.,  15.],
         [ 16.,  17.,  20.,  21.],
         [ 18.,  19.,  22.,  23.],
         [ 24.,  25.,  28.,  29.],
         [ 26.,  27.,  30.,  31.],
         [ 32.,  33.,  36.,  37.],
         [ 34.,  35.,  38.,  39.],
         [ 40.,  41.,  44.,  45.],
         [ 42.,  43.,  46.,  47.],
         [ 48.,  49.,  52.,  53.],
         [ 50.,  51.,  54.,  55.],
         [ 56.,  57.,  60.,  61.],
         [ 58.,  59.,  62.,  63.]]]]
    neibs = numpy.asarray(f())
    numpy.allclose(neibs,res1)
    #print neibs
    g = function([], neibs2images(neibs, neib_shape, images.shape), mode=mode_with_gpu)
    assert any([isinstance(node.op,GpuImages2Neibs) for node in f.maker.env.toposort()])
    #print numpy.asarray(g())
    assert numpy.allclose(images.value,g())

if __name__ == '__main__':
    test_neibs_gpu()
    test_neibs()
