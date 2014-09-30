import unittest
import theano
import theano.tensor as T
import numpy
import cPickle
from collections import OrderedDict
floatX = 'float32'

def test_pickle_unpickle():
    # Test if pick and unpickling a theano function with
    # shared variables work
    x1 = T.fmatrix('x1')
    x2 = T.fmatrix('x2')
    x3 = theano.shared(numpy.ones((10,10),dtype=floatX))
    x4 = theano.shared(numpy.ones((10,10),dtype=floatX))
    y = T.sum(T.sum(T.sum(x1**2+x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = theano.function([x1,x2],y, updates=updates)

    pkl_path = open('thean_fn.pkl','wb')
    cPickle.dump(f, pkl_path, -1)
    pkl_path = open('thean_fn.pkl','r')
    f_ = cPickle.load(pkl_path)

    in1 = numpy.ones((10, 10), dtype=floatX)
    in2 = numpy.ones((10, 10), dtype=floatX)
        
    assert f(in1, in2) == f_(in1, in2)
    
    print f(in1, in2)
    
if __name__ == '__main__':
    test_pickle_unpickle()
