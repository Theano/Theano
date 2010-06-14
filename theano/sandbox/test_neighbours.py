from numpy import *
import theano
from theano import shared, function
import theano.tensor as T
from neighbours import images2neibs, neibs2images

def neibs_test():

    images = shared(arange(2*2*4*4, dtype='float32').reshape(2,2,4,4))
    neib_shape = shared(array((2,2), dtype='float32'))

    f = function([], images2neibs(images, neib_shape))

    print images.value
    neibs = f()
    print neibs
    g = function([], neibs2images(neibs, neib_shape, images.shape))
    
    print g()
    assert allclose(images.value,g())
