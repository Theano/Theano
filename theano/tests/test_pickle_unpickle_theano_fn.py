"""
This script tests the pickle and unpickle of theano functions.
When a compiled theano has shared vars, their values are also being pickled.

Side notes useful for debugging:
The pickling tools theano uses is here:
theano.compile.function_module._pickle_Function()
theano.compile.function_module._pickle_FunctionMaker()
Whether reoptimize the pickled function graph is handled by
FunctionMaker.__init__()
The config option is in configdefaults.py

This note is written by Li Yao.
"""
import unittest
import numpy
import cPickle
from collections import OrderedDict
floatX = 'float32'
import theano
import theano.tensor as T
    
def test_pickle_unpickle_with_reoptimization():
    x1 = T.fmatrix('x1')
    x2 = T.fmatrix('x2')
    x3 = theano.shared(numpy.ones((10,10),dtype=floatX))
    x4 = theano.shared(numpy.ones((10,10),dtype=floatX))
    y = T.sum(T.sum(T.sum(x1**2+x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = theano.function([x1,x2],y, updates=updates)

    # now pickle the compiled theano fn
    pkl_path = open('thean_fn.pkl','wb')
    cPickle.dump(f, pkl_path, -1)
    
    in1 = numpy.ones((10, 10), dtype=floatX)
    in2 = numpy.ones((10, 10), dtype=floatX)
    print 'the desired value is ',f(in1, in2)
    
    # test unpickle with optimization
    theano.config.reoptimize_unpickled_function=True # the default is True
    pkl_path = open('thean_fn.pkl','r')
    f_ = cPickle.load(pkl_path)
    print 'got value ', f_(in1, in2)
    assert f(in1, in2) == f_(in1, in2)
    
def test_pickle_unpickle_without_reoptimization():
    x1 = T.fmatrix('x1')
    x2 = T.fmatrix('x2')
    x3 = theano.shared(numpy.ones((10,10),dtype=floatX))
    x4 = theano.shared(numpy.ones((10,10),dtype=floatX))
    y = T.sum(T.sum(T.sum(x1**2+x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = theano.function([x1,x2],y, updates=updates)

    # now pickle the compiled theano fn
    pkl_path = open('thean_fn.pkl','wb')
    cPickle.dump(f, pkl_path, -1)

    # compute f value
    in1 = numpy.ones((10, 10), dtype=floatX)
    in2 = numpy.ones((10, 10), dtype=floatX)
    print 'the desired value is ',f(in1, in2)
    
    # test unpickle without optimization
    theano.config.reoptimize_unpickled_function=False # the default is True
    pkl_path = open('thean_fn.pkl','r')
    f_ = cPickle.load(pkl_path)
    print 'got value ', f_(in1, in2)
    assert f(in1, in2) == f_(in1, in2)
    
if __name__ == '__main__':
    test_pickle_unpickle_with_reoptimization()
    test_pickle_unpickle_without_reoptimization()
