import sys, time

from theano import shared
from theano.compile.pfunc import pfunc
from theano import tensor

import numpy
import theano
import theano.tensor as T

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_enabled == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as tcn
import theano.sandbox.cuda as cuda
import theano.compile.mode
from theano.tests import unittest_tools as utt

mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

def tes_use():
    tcn.use()

def test_sum():
    """
    test sum pattern 1, 11, 10, 100, 110, 001, 111, 1011, 1111
    TODO: test with broadcast
    """

    for shape, pattern in [((5,),[0]),
                           ((5,4),[0,1]),((5,4),[0]),
                           ((5,4,3),[0]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[0,1,2]),
                           ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3])]:
        a = tensor.TensorType('float32',(False,)*len(shape))()
        b = T.Sum(pattern)(a)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val,dtype='float32')
        f = theano.function([a],b, mode=mode_with_gpu)
        f2 = theano.function([a],b)
        assert tcn.GpuSum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f2.maker.env.toposort()]
        assert numpy.allclose(f2(val),f(val))
        

        #test with broadcast
    for shape, pattern in [((5,),[0]),
                           ((5,4),[0,1]),((5,4),[0]),
                           ((5,4,3),[0]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[0,1,2]),
                           ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3])]:
        shape = numpy.asarray(shape)*2
        a = tensor.TensorType('float32',(False,)*len(shape))()
        a2 = tcn.CudaNdarrayType((False,)*len(shape))()
        b = T.Sum(pattern)(a)
        b2 = T.Sum(pattern)(a2)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val,dtype='float32')
        val2 = cuda.CudaNdarray(val)
        if len(shape)==1:
            val = val[::2]
            val2 = val2[::2]
        elif len(shape)==2:
            val = val[::2,::2]
            val2 = val2[::2,::2]
        elif len(shape)==3:
            val = val[::2,::2,::2]
            val2 = val2[::2,::2,::2]
        elif len(shape)==4:
            val = val[::2,::2,::2,::2]
            val2 = val2[::2,::2,::2,::2]
        f = theano.function([a],b)
        f2 = theano.function([a2],b2, mode=mode_with_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f2.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert numpy.allclose(f2(val2),f(val))
        

def test_reshape():

    a = tcn.CudaNdarrayType((False,))()
    b = tcn.CudaNdarrayType((False,False))()
    c = T.reshape(a, [2,3])

    #basic
    f = theano.function([a], c)
    fv = f(cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32')))
    assert numpy.all(fv == numpy.asarray([[0,1,2], [3,4,5]]))

    #test that it works without inplace operations
    a_val = cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32'))
    a_val_copy = cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32'))
    b_val = cuda_ndarray.CudaNdarray(theano._asarray([[0,1,2],[3,4,5]],dtype='float32'))

    f_sub = theano.function([a,b], c-b)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    #test that it works with inplace operations
    a_val = theano._asarray([0,1,2,3,4,5], dtype='float32')
    a_val_copy = theano._asarray([0,1,2,3,4,5], dtype='float32')
    b_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float32')

    f_sub = theano.function([a,b], c-b)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    # verify gradient
    def just_vals(v):
        return T.Reshape(2)(v, theano._asarray([2,3], dtype='int32'))
    utt.verify_grad(just_vals, [a_val])

def test_elemwise0():

    a = tcn.shared_constructor(numpy.random.rand(4,4), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, a+b)], mode=mode_with_gpu)

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
    print >> sys.stdout, "STARTING FUNCTION 1"
    f = pfunc([b], [], updates=[(a, b**a)], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.random.rand(*shape)+0.3)

    print >> sys.stdout, "STARTING FUNCTION 2"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, tensor.exp(b**a))], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.random.rand(*shape)+0.3)

    print >> sys.stdout, "STARTING FUNCTION 3"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, a+b * tensor.exp(b**a))], mode=mode_with_gpu)
    f(numpy.random.rand(*shape)+0.3)

def test_elemwise2():
    """ Several kinds of elemwise expressions with dimension permutations """
    rng = numpy.random.RandomState(int(time.time()))
    print 'random?', rng.rand(3)
    shape = (3,5)
    for pattern in [(0,1), (1,0)]:
        a = tcn.shared_constructor(rng.rand(*shape), name=None)
        b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
        f = pfunc([b], [], updates=[(a, (a+b).dimshuffle(pattern))], mode=mode_with_gpu)
        has_elemwise = False
        for i, node in enumerate(f.maker.env.toposort()):
            print >> sys.stdout, i, node
            has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
        assert not has_elemwise
        #let debugmode catch errors
        print >> sys.stdout, 'pattern', pattern
        f(rng.rand(*shape)*.3)
    
    shape = (3,4,5,6)
    a = tcn.shared_constructor(rng.rand(*shape), 'a')
    b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
    f = pfunc([b], [], updates=[(a, (a+b).dimshuffle([2,0,3,1]) *
        tensor.exp(b**a).dimshuffle([2,0,3,1]))], mode=mode_with_gpu)
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
        b**a).dimshuffle([2,0,3,1]))], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stdout, i, node
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
    f = pfunc([b,c], [], updates=[(a, (a+b.dimshuffle('x', 0)*c.dimshuffle(0, 'x')))], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stdout, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(numpy.random.rand(4), numpy.random.rand(3))


def speed_elemwise_collapse():
    """ used to time if the collapse of ccontiguous dims are usefull """
    
    shape = (30,40,50,600)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,::2,:,:]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(*shape),dtype='float32')
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
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,:,:,::2]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(*shape),dtype='float32')
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
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0,'x',1,2)
    b = tcn.CudaNdarrayType((False, True, False, False))()
    c = a3+b
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(shape[0],1,*shape[1:]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(shape[0],1,*shape[1:])+v)
    print "Expected collapse of all dimensions"

def test_elemwise_collapse2():
    """ Test when only one inputs have one broadcastable dimension """
    
    shape = (4,5,60)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0,'x',1,2)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(shape[0],5,*shape[1:]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(shape[0],1,*shape[1:])+v)
    print "Expected collapse to 3 dimensions"

def test_elemwise_collapse3():
    """ Test when only one inputs have two broadcastable dimension at each ends """
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x',0,1,'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b)
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(5,shape[0],shape[1],4),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,shape[0],shape[1],1)+v)
    print "Expected collapse to 3 dimensions"

def test_elemwise_collapse4():
    """ Test when only one inputs have two broadcastable dimension at each ends and we add a scalar"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x',0,1,'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b+2)
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(5,shape[0],shape[1],4),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,shape[0],shape[1],1)+v+2)
    print "Expected collapse to 3 dimensions"

def test_elemwise_collapse5():
    """ Test when only one inputs have two broadcastable dimension at the beginning and we add a scalar"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x','x',0,1)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3+b+2)
    f = pfunc([b], [c])


    v = theano._asarray(numpy.random.rand(5,4,shape[0],shape[1]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,1,shape[0],shape[1])+v+2)
    print "Expected collapse to 2 dimensions"

def test_elemwise_collapse6():
    """ Test when all inputs have two broadcastable dimension at the beginning"""
    
    shape = (4,5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x','x',0,1)
    b = tcn.CudaNdarrayType((True, True, False, False))()
    f = pfunc([b], [a3+b])

    v = theano._asarray(numpy.random.rand(1,1,shape[0],shape[1]),dtype='float32')
    v=cuda_ndarray.CudaNdarray(v)
    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f(v)[0]
    assert numpy.allclose(out,a.reshape(1,1,shape[0],shape[1])+v)
    print "Expected collapse to c contiguous"


def test_elemwise_collapse7(atol=1e-6):
    """ Test when one input have one broadcastable dimension and the other is a scalar"""
    
    shape = (5,4,1)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a.copy(), 'a')
    a3 = a2.dimshuffle(0, 'x', 1, 2)
    f = pfunc([], [a3+2])

    if False:
        for id,n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out=f()[0]
    ans=(a+2).reshape(shape[0],1,shape[1],shape[2])
    assert numpy.allclose(out,ans, atol=atol)
    print "Expected collapse to c contiguous"
