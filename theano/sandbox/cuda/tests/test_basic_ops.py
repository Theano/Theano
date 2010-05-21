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
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as tcn
import theano.sandbox.cuda as cuda
import theano.compile.mode
from theano.tests import unittest_tools as utt

if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

def tes_use():
    tcn.use()

def test_sum():
    """
    test sum pattern 1, 11, 10, 01, 100, 110, 011, 001, 111, 0011, 0111, 1011, 1111
    TODO: test with broadcast
    """
    for shape, pattern in [((100,3,1300),[1]),
                           ((0,),[0]),((5,),[0]),
                           ((0,0),[0,1]),((1,0),[0,1]),((5,4),[0,1]),((33,31),[0,1]),((5,4),[1]),((5,4),[0]),#need something bigger then 32 for some opt test.
                           ((5,4,3),[0]),((5,4,3),[1]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[1,2]),((5,4,3),[0,1,2]),
                           ((0,0,0,0),[0,1,2,3]),
                           ((5,4,3,20),[2,3]), ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3]),((5,4,3,2),[1,2,3]),
                           ((5,4,3,10,11),[1,2])]:
        a = tensor.TensorType('float32',(False,)*len(shape))()
        b = T.Sum(pattern)(a)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val,dtype='float32')
        f = theano.function([a],b, mode=mode_with_gpu)
        f2 = theano.function([a],b, mode=mode_without_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f2.maker.env.toposort()]
        if val.size==0:
            assert f2(val)==f(val)
        else:
            assert numpy.allclose(f2(val),f(val))
        

        #test with dimshuffle
        #we shuffle the 2 outer dims.
    for shape, pattern in [#((5,),[0]),
                           ((5,4),[0,1]),((5,4),[0]),
                           ((5,4,3),[0]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[0,1,2]),
                           ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3])]:
        a = tensor.TensorType('float32',(False,)*len(shape))()
        dim_pattern = range(len(shape))
        dim_pattern[0]=1
        dim_pattern[1]=0
        a = a.dimshuffle(dim_pattern)
        b = T.Sum(pattern)(a)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val,dtype='float32')
        f = theano.function([a],b, mode=mode_with_gpu)
        f2 = theano.function([a],b, mode=mode_without_gpu)
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
        f = theano.function([a],b, mode=mode_without_gpu)
        f2 = theano.function([a2],b2, mode=mode_with_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f2.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert numpy.allclose(f2(val2),f(val))
        

def test_reshape():

    a = tcn.CudaNdarrayType((False,))()
    b = tcn.CudaNdarrayType((False,False))()
    c = T.reshape(a, [2,3])

    #basic
    f = theano.function([a], c, mode=mode_without_gpu)
    fv = f(cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32')))
    assert numpy.all(fv == numpy.asarray([[0,1,2], [3,4,5]]))

    #test that it works without inplace operations
    a_val = cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32'))
    a_val_copy = cuda_ndarray.CudaNdarray(theano._asarray([0,1,2,3,4,5],dtype='float32'))
    b_val = cuda_ndarray.CudaNdarray(theano._asarray([[0,1,2],[3,4,5]],dtype='float32'))

    f_sub = theano.function([a,b], c-b, mode=mode_without_gpu)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    #test that it works with inplace operations
    a_val = theano._asarray([0,1,2,3,4,5], dtype='float32')
    a_val_copy = theano._asarray([0,1,2,3,4,5], dtype='float32')
    b_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float32')

    f_sub = theano.function([a,b], c-b, mode=mode_without_gpu)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    # verify gradient
    def just_vals(v):
        return T.Reshape(2)(v, theano._asarray([2,3], dtype='int32'))
    utt.verify_grad(just_vals, [a_val])

def test_elemwise_empty():
    #test with 0 element
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(0,0), dtype='float32'), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, a+b)], mode=mode_with_gpu)
    f2 = pfunc([b], [], updates=[(a, a+b)], mode=mode_without_gpu)

    a0 = a.value * 1.0
    f(numpy.ones((0,0)))

    assert numpy.all(a0 + 1.0 == a.value)

def test_elemwise0():

    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(4,4), dtype='float32'), 'a')

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
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape), dtype='float32')+0.5, 'a')
    b = tensor.fmatrix()

    #let debugmode catch any mistakes
    print >> sys.stdout, "STARTING FUNCTION 1"
    f = pfunc([b], [], updates=[(a, b**a)], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32')+0.3)

    print >> sys.stdout, "STARTING FUNCTION 2"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, tensor.exp(b**a))], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32')+0.3)

    print >> sys.stdout, "STARTING FUNCTION 3"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, a+b * tensor.exp(b**a))], mode=mode_with_gpu)
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32')+0.3)

def test_elemwise2():
    """ Several kinds of elemwise expressions with dimension permutations """
    rng = numpy.random.RandomState(int(time.time()))
    print 'random?', rng.rand(3)
    shape = (3,5)
    for pattern in [(0,1), (1,0)]:
        a = tcn.shared_constructor(theano._asarray(rng.rand(*shape),dtype='float32'), name=None)
        b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
        f = pfunc([b], [], updates=[(a, (a+b).dimshuffle(pattern))], mode=mode_with_gpu)
        has_elemwise = False
        for i, node in enumerate(f.maker.env.toposort()):
            print >> sys.stdout, i, node
            has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
        assert not has_elemwise
        #let debugmode catch errors
        print >> sys.stdout, 'pattern', pattern
        f(theano._asarray(rng.rand(*shape),dtype='float32')*.3)
    
    shape = (3,4,5,6)
    a = tcn.shared_constructor(theano._asarray(rng.rand(*shape),dtype='float32'), 'a')
    b = tensor.Tensor(dtype='float32', broadcastable=[0]*len(shape))()
    f = pfunc([b], [], updates=[(a, (a+b).dimshuffle([2,0,3,1]) *
        tensor.exp(b**a).dimshuffle([2,0,3,1]))], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(theano._asarray(rng.rand(*shape),dtype='float32'))

def test_elemwise3():
    """ Several kinds of elemwise expressions with dimension permutations and broadcasting"""
    
    shape = (3,4,5,6)
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape), dtype='float32'), 'a')
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
    f(theano._asarray(numpy.random.rand(6), dtype='float32'))

def test_elemwise4():
    """ Test that two vectors can be broadcast to form an outer product (by performing rank-1 matrix update"""
    
    shape = (3,4)
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape), dtype='float32'), 'a')
    b = tensor.fvector()
    c = tensor.fvector()
    f = pfunc([b,c], [], updates=[(a, (a+b.dimshuffle('x', 0)*c.dimshuffle(0, 'x')))], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stdout, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(theano._asarray(numpy.random.rand(4), dtype='float32'), theano._asarray(numpy.random.rand(3), dtype='float32'))


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


def test_hostfromgpu_shape_i():
    """
    Test that the shape is lifted over hostfromgpu
    """
    pass

    m = mode_with_gpu.including('local_dot_to_dot22','local_dot22_to_dot22scalar','specialize')
    a=T.fmatrix('a')
    ca=theano.sandbox.cuda.var.CudaNdarrayType((False,False))()

    av=numpy.asarray(numpy.random.rand(5,4),dtype='float32')
    cv=cuda.CudaNdarray(numpy.asarray(numpy.random.rand(5,4),dtype='float32'))

    f = theano.function([a],cuda.basic_ops.gpu_from_host(a), mode=m)
    assert cuda.basic_ops.gpu_from_host in [x.op for x in f.maker.env.toposort()]
    f = theano.function([a],cuda.basic_ops.gpu_from_host(a).shape, mode=m)
    topo = f.maker.env.toposort()
    assert isinstance(topo[0].op,T.opt.Shape_i)
    assert isinstance(topo[1].op,T.opt.Shape_i)
    assert isinstance(topo[2].op,T.opt.MakeVector)
    assert tuple(f(av))==(5,4)



    f = theano.function([ca],cuda.basic_ops.host_from_gpu(ca), mode=m)
    assert cuda.basic_ops.host_from_gpu in [x.op for x in f.maker.env.toposort()]
    f = theano.function([ca],cuda.basic_ops.host_from_gpu(ca).shape, mode=m)
    topo = f.maker.env.toposort()
    assert isinstance(topo[0].op,T.opt.Shape_i)
    assert isinstance(topo[1].op,T.opt.Shape_i)
    assert isinstance(topo[2].op,T.opt.MakeVector)
    assert tuple(f(cv))==(5,4)

# -----------------------------------------------------------------------

import theano.sandbox.cuda as cuda_ndarray
from theano.sandbox.cuda.basic_ops import gpu_join, GpuDimShuffle

def test_gpujoin_twomatrices_joincolumns():
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    _b = numpy.asarray([[5,6,7],[8,9,10]],dtype='float32')
    a = tcn.shared_constructor(_a)
    b = tcn.shared_constructor(_b)

    c = gpu_join(1,a,b)

    f = theano.function([], c)

    assert numpy.all(f() == numpy.concatenate([_a,_b], axis=1))

def test_gpujoin_twomatrices_badshapes():
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    _b = numpy.asarray([[5,6,7],[8,9,10]],dtype='float32')
    a = tcn.shared_constructor(_a)
    b = tcn.shared_constructor(_b)

    # try to join on dimension 0 where they don't agree (2!=3)
    c = gpu_join(0,a,b)

    f = theano.function([], c)

    try:
        f()
        assert False
    except ValueError:
        assert True




def test_gpujoin_preserves_broadcasting():
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    _b = numpy.asarray([[5,6,7],[8,9,10]],dtype='float32')
    a = tcn.shared_constructor(_a)
    b = tcn.shared_constructor(_b)

    # [0,0] : the two original dims were non-broadcastable
    # [1,x,0]: new order and broadcastability
    gpu_dimshuffle = GpuDimShuffle([0,0], [1,'x',0])

    a_shuffled = gpu_dimshuffle(a)
    b_shuffled = gpu_dimshuffle(b)

    c = gpu_join(0,a_shuffled,b_shuffled)

    assert c.type.broadcastable == (False,True,False)

    f = theano.function([], c, mode=mode_with_gpu)
    
    res = f()

    a_reshaped = numpy.asarray([[[1,3]],[[2,4]]], dtype='float32')
    b_reshaped = numpy.asarray([[[5,8]],[[6,9]],[[7,10]]], dtype='float32')

    concat = numpy.concatenate([a_reshaped,b_reshaped], axis=0)

    assert numpy.all(res == concat)


def test_gpujoin_assert_cndas():
    # this will end up being an ndarray, as it's float64
    _a = numpy.asarray([[1,2],[3,4]],dtype='float64')
    a = theano.shared(_a)

    try:
        c = gpu_join(1,a)
        # can't "assert False" here, as we want the assertion 
        # error from gpu_join
    except AssertionError:
        assert True
        return

    assert False
    
def test_gpujoin_no_rebroadcast():
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    a = tcn.shared_constructor(_a)
    f = theano.function([],T.join(1,a))
    l = f.maker.env.toposort()
    assert not any([isinstance(x.op,T.Rebroadcast) for x in l])

if __name__ == '__main__':
    test_gpujoin_twomatrices_joincolumns()
    test_gpujoin_assert_cndas()
    test_gpujoin_preserves_broadcasting()
    test_gpujoin_twomatrices_badshapes()



