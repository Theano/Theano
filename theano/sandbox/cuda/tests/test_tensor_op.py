"""
This file test tensor op that should also operate on CudaNdaray.
"""
import copy
from nose.plugins.skip import SkipTest

import numpy

import theano
from theano import tensor
import theano.tensor as T

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_shape_i():
    x = cuda.ftensor3()
    v = cuda.CudaNdarray(numpy.zeros((3,4,5),dtype='float32'))
    f = theano.function([x],x.shape[1])
    topo = f.maker.env.toposort()
    assert f(v)==4
    if theano.config.mode!='FAST_COMPILE':
        assert len(topo)==1
        assert isinstance(topo[0].op,T.opt.Shape_i)

def test_shape():
    x = cuda.ftensor3()
    v = cuda.CudaNdarray(numpy.zeros((3,4,5),dtype='float32'))
    f = theano.function([x],x.shape)
    topo = f.maker.env.toposort()
    assert numpy.all(f(v)==(3,4,5))
    if theano.config.mode!='FAST_COMPILE':
        assert len(topo)==4
        assert isinstance(topo[0].op,T.opt.Shape_i)
        assert isinstance(topo[1].op,T.opt.Shape_i)
        assert isinstance(topo[2].op,T.opt.Shape_i)
        assert isinstance(topo[3].op,T.opt.MakeVector)

def test_softmax_optimizations():
    from theano.tensor.nnet.nnet import softmax, crossentropy_categorical_1hot
    x = tensor.fmatrix('x')
    one_of_n = tensor.lvector('one_of_n')
    op = crossentropy_categorical_1hot

    xe = op(x, one_of_n)

    env = theano.gof.Env(
        [x, one_of_n],
        [op(softmax(x), one_of_n)])
    assert env.outputs[0].owner.op == op

    mode_with_gpu.optimizer.optimize(env)

    assert str(env.outputs[0].owner.op) == 'OutputGuard'
    assert env.outputs[0].owner.inputs[0].owner.op == cuda.host_from_gpu
    assert env.outputs[0].owner.inputs[0].owner.inputs[0].owner.op == cuda.nnet.gpu_crossentropy_softmax_argmax_1hot_with_bias

def test_may_share_memory_cuda():
    from theano.misc.may_share_memory import may_share_memory
    a = cuda.CudaNdarray(numpy.zeros((3,4),dtype='float32'))
    b = cuda.CudaNdarray(numpy.zeros((3,4),dtype='float32'))
    na = numpy.zeros((3,4))
    nb = numpy.zeros((3,4))
    va = a.view()
    vb = b.view()
    ra = a.reshape((4,3))
    rb = b.reshape((4,3))

    #can't test the transpose as ta._strides = is not implemented
    #manual transpose of a
    #ta = a.reshape((4,3))
    #ta._strides = (ta._strides[1],ta._strides[0])#not implemented
    #elem_size=elem_size = numpy.zeros(0,dtype=a.dtype).dtype.itemsize
    #ta.gpudata += ta.size*elem_size

    for a_,b_,rep in [(a,a,True),(b,b,True),(a,b,False),
                      (a,na,False),(b,nb,False),(na,b,False),(nb,a,False),
                      (a,va,True),(b,vb,True),(va,b,False),(a,vb,False),
                      (a,ra,True),(b,rb,True),(ra,b,False),(a,rb,False),
                      ]:
        assert may_share_memory(a_,b_)==rep
        assert may_share_memory(b_,a_)==rep

    #test that it raise error when needed.
    for a_,b_,rep in [(a,(0,),False),(a,1,False),(a,None,False)]:
        assert may_share_memory(a_,b_,False)==rep
        assert may_share_memory(b_,a_,False)==rep
        try:
            may_share_memory(a_,b_)
            raise Exception("An error was expected")
        except TypeError:
            pass
        try:
            may_share_memory(b_,a_)
            raise Exception("An error was expected")
        except TypeError:
            pass


def test_deepcopy():
    a = cuda.fmatrix()
    a_v = cuda.CudaNdarray(numpy.zeros((3, 4), dtype='float32'))

    # We force the c code to check that we generate c code
    mode = theano.Mode("c", mode_with_gpu.optimizer)
    f = theano.function([a], a, mode=mode)
    theano.printing.debugprint(f)
    out = f(a_v)
    assert out is not a_v
    assert numpy.allclose(numpy.asarray(a_v), numpy.asarray(out))

    # We force the python linker as the default code should work for this op
    mode = theano.Mode("py", mode_with_gpu.optimizer)
    f = theano.function([a], a, mode=mode)
    theano.printing.debugprint(f)
    out = f(a_v)
    assert out is not a_v
    assert numpy.allclose(numpy.asarray(a_v), numpy.asarray(out))
