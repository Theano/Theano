import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import theano.sandbox.cuda as tcn
import cuda_ndarray

def tes_use():
    tcn.use()

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


def speed_elemwise_collapse():
    """ used to time if the collapse of ccontiguous dims are usefull """
    
    shape = (30,40,50,600)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,::2,:,:]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    v = v[:,::2,:,:]
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    t1=time.time()
    for i in range(100):
       #let debugmode catch errors
       f(v)
    t2=time.time()

def speed_elemwise_collapse2():
    """ used to test the speed up of the generalised collapse of ccontiguous dims"""
    
    shape = (30,40,50,600)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,:,:,::2]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    v = v[:,:,:,::2]
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    t1=time.time()
    for i in range(100):
       #let debugmode catch errors
       f(v)
    t2=time.time()

def test_elemwise_collapse():
    """ Test when all inputs have one(and the same) broadcastable dimension """
    
    shape = (4,5,60)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0,'x',1,2)
    b = tcn.CudaNdarrayType((False, True, False, False))()
    c = a3+b
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(shape[0],1,*shape[1:]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(shape[0],1,*shape[1:])+v)
    print "Expected collapse of all dimensions"

def test_elemwise_collapse2():
    """ Test when only one inputs have one broadcastable dimension """
    
    shape = (4,5,60)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0,'x',1,2)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(shape[0],5,*shape[1:]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(shape[0],1,*shape[1:])+v)
    print "Expected collapse to 3 dimensions"

def test_elemwise_collapse3():
    """ Test when only one inputs have two broadcastable dimension at each ends """
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x',0,1,'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(5,shape[0],shape[1],4),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,shape[0],shape[1],1)+v)
    print "Expected collapse to 3 dimensions"

def test_elemwise_collapse4():
    """ Test when only one inputs have two broadcastable dimension at each ends and we add a scalar"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x',0,1,'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b+2)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(5,shape[0],shape[1],4),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,shape[0],shape[1],1)+v+2)
    print "Expected collapse to 3 dimensions"


def test_elemwise_collapse5():
    """ Test when only one inputs have two broadcastable dimension at the beginning and we add a scalar"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x','x',0,1)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b+2)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(5,4,shape[0],shape[1]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,1,shape[0],shape[1])+v+2)
    print "Expected collapse to 2 dimensions"

def test_elemwise_collapse6():
    """ Test when all inputs have two broadcastable dimension at the beginning"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x','x',0,1)
    b = tcn.CudaNdarrayType((True, True, False, False))()
    f = pfunc([b], [a3+b])

    v = numpy.asarray(numpy.random.rand(1,1,shape[0],shape[1]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,1,shape[0],shape[1])+v)
    print "Expected collapse to c contiguous"


def test_elemwise_collapse7(atol=1e-6):
    """ Test when one input have one broadcastable dimension and the other is a scalar"""
    
    shape = (5,4,1)
    a = cuda_ndarray.CudaNdarray(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a.copy(), 'a')
    a3 = a2.dimshuffle(0, 'x', 1, 2)
    f = pfunc([], [a3+2])


    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    #let debugmode catch errors
    out=f()[0]
    ans=(a+2).reshape(shape[0],1,shape[1],shape[2])
    assert numpy.allclose(out,ans, atol=atol)
    print "Expected collapse to c contiguous"


