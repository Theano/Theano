import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import theano_cuda_ndarray as tcn


def test_elemwise0():

    a = tcn.shared_constructor(numpy.random.rand(4,4), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, a+b)])

    a0 = a.value * 1.0
    print 'BEFORE ADD', a.value
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.ones((4,4)))
    print 'AFTER ADD', a.value

    assert numpy.all(a0 + 1.0 == a.value)

def test_elemwise1():
    """ Several kinds of elemwise expressions with no broadcasting, non power-of-two shape """

    shape = (3,4)
    a = tcn.shared_constructor(numpy.random.rand(*shape)+0.5, 'a')
    b = tensor.fmatrix()

    #let debugmode catch any mistakes
    print >> sys.stderr, "STARTING FUNCTION 1"
    f = pfunc([b], [], updates=[(a, b**a)])
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.random.rand(*shape)+0.3)

    print >> sys.stderr, "STARTING FUNCTION 2"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, tensor.exp(b**a))])
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.random.rand(*shape)+0.3)

    print >> sys.stderr, "STARTING FUNCTION 3"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, a+b * tensor.exp(b**a))])
    f(numpy.random.rand(*shape)+0.3)

def test_elemwise2():
    """ Several kinds of elemwise expressions with dimension permutations """
    rng = numpy.random.RandomState(int(time.time()))
    print 'random?', rng.rand(3)
    shape = (3,5)
    for pattern in [(0,1), (1,0)]:
        a = tcn.shared_constructor(rng.rand(*shape), name=None)
        b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
        f = pfunc([b], [], updates=[(a, (a+b).dimshuffle(pattern))])
        has_elemwise = False
        for i, node in enumerate(f.maker.env.toposort()):
            print >> sys.stderr, i, node
            has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
        assert not has_elemwise
        #let debugmode catch errors
        print >> sys.stderr, 'pattern', pattern
        f(rng.rand(*shape)*.3)
    
    shape = (3,4,5,6)
    a = tcn.shared_constructor(rng.rand(*shape), 'a')
    b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
    f = pfunc([b], [], updates=[(a, (a+b).dimshuffle([2,0,3,1]) *
        tensor.exp(b**a).dimshuffle([2,0,3,1]))])
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(rng.rand(*shape))

def test_elemwise3():
    """ Several kinds of elemwise expressions with dimension permutations and broadcasting"""
    
    shape = (3,4,5,6)
    a = tcn.shared_constructor(numpy.random.rand(*shape), 'a')
    b = tensor.fvector()
    print b.type
    print tensor.constant(1).type
    print (1 + b).type
    print (1 + b**a).type
    print tensor.exp((1 + b**a)).type
    f = pfunc([b], [], updates=[(a, (a+b).dimshuffle([2,0,3,1]) * tensor.exp(1 +
        b**a).dimshuffle([2,0,3,1]))])
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stderr, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(numpy.random.rand(6))

def test_elemwise4():
    """ Test that two vectors can be broadcast to form an outer product (by performing rank-1 matrix update"""
    
    shape = (3,4)
    a = tcn.shared_constructor(numpy.random.rand(*shape), 'a')
    b = tensor.fvector()
    c = tensor.fvector()
    f = pfunc([b,c], [], updates=[(a, (a+b.dimshuffle('x', 0)*c.dimshuffle(0, 'x')))])
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stderr, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(numpy.random.rand(4), numpy.random.rand(3))
