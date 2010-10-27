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

def test_neibs_bad_shape():
    shape = (2,3,10,10)
    images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
    neib_shape = T.as_tensor_variable((3,2))

    try:
        f = function([], images2neibs(images, neib_shape), mode=mode_without_gpu)
        neibs = f()
        #print neibs
        assert False,"An error was expected"
    except TypeError:
        pass

    shape = (2,3,10,10)
    images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
    neib_shape = T.as_tensor_variable((2,3))

    try:
        f = function([], images2neibs(images, neib_shape), mode=mode_without_gpu)
        neibs = f()
        #print neibs
        assert False,"An error was expected"
    except TypeError:
        pass

def test_neibs_manual():
    shape = (2,3,4,4)
    images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
    neib_shape = T.as_tensor_variable((2,2))

    f = function([], images2neibs(images, neib_shape), mode=mode_without_gpu)

    #print images.value
    neibs = f()
    print neibs
    assert numpy.allclose(neibs,[[ 0,  1,  4,  5],
       [ 2,  3,  6,  7],
       [ 8,  9, 12, 13],
       [10, 11, 14, 15],
       [16, 17, 20, 21],
       [18, 19, 22, 23],
       [24, 25, 28, 29],
       [26, 27, 30, 31],
       [32, 33, 36, 37],
       [34, 35, 38, 39],
       [40, 41, 44, 45],
       [42, 43, 46, 47],
       [48, 49, 52, 53],
       [50, 51, 54, 55],
       [56, 57, 60, 61],
       [58, 59, 62, 63],
       [64, 65, 68, 69],
       [66, 67, 70, 71],
       [72, 73, 76, 77],
       [74, 75, 78, 79],
       [80, 81, 84, 85],
       [82, 83, 86, 87],
       [88, 89, 92, 93],
       [90, 91, 94, 95]])
    g = function([], neibs2images(neibs, neib_shape, images.shape), mode=mode_without_gpu)
    
    print g()
    assert numpy.allclose(images.value,g())


def test_neibs_manual_step():
    shape = (2,3,5,5)
    images = shared(numpy.asarray(numpy.arange(numpy.prod(shape)).reshape(shape),dtype='float32'))
    neib_shape = T.as_tensor_variable((3,3))
    neib_step = T.as_tensor_variable((2,2))
    modes = [mode_without_gpu]
    if cuda.cuda_available:
        modes.append(mode_with_gpu)
    for mode in modes:
        f = function([], images2neibs(images, neib_shape, neib_step), mode=mode)

    #print images.value
        neibs = f()
        print neibs
        assert numpy.allclose(neibs,
      [[  0,   1,   2,   5,   6,   7,  10,  11,  12],
       [  2,   3,   4,   7,   8,   9,  12,  13,  14],
       [ 10,  11,  12,  15,  16,  17,  20,  21,  22],
       [ 12,  13,  14,  17,  18,  19,  22,  23,  24],
       [ 25,  26,  27,  30,  31,  32,  35,  36,  37],
       [ 27,  28,  29,  32,  33,  34,  37,  38,  39],
       [ 35,  36,  37,  40,  41,  42,  45,  46,  47],
       [ 37,  38,  39,  42,  43,  44,  47,  48,  49],
       [ 50,  51,  52,  55,  56,  57,  60,  61,  62],
       [ 52,  53,  54,  57,  58,  59,  62,  63,  64],
       [ 60,  61,  62,  65,  66,  67,  70,  71,  72],
       [ 62,  63,  64,  67,  68,  69,  72,  73,  74],
       [ 75,  76,  77,  80,  81,  82,  85,  86,  87],
       [ 77,  78,  79,  82,  83,  84,  87,  88,  89],
       [ 85,  86,  87,  90,  91,  92,  95,  96,  97],
       [ 87,  88,  89,  92,  93,  94,  97,  98,  99],
       [100, 101, 102, 105, 106, 107, 110, 111, 112],
       [102, 103, 104, 107, 108, 109, 112, 113, 114],
       [110, 111, 112, 115, 116, 117, 120, 121, 122],
       [112, 113, 114, 117, 118, 119, 122, 123, 124],
       [125, 126, 127, 130, 131, 132, 135, 136, 137],
       [127, 128, 129, 132, 133, 134, 137, 138, 139],
       [135, 136, 137, 140, 141, 142, 145, 146, 147],
       [137, 138, 139, 142, 143, 144, 147, 148, 149]])
        #g = function([], neibs2images(neibs, neib_shape, images.shape), mode=mode_without_gpu)
        
        #print g()
        #assert numpy.allclose(images.value,g())


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
