"""
This file test tensor op that should also operate on CudaNdaray.
"""
import numpy

from theano import tensor

import theano
import theano.tensor as T

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
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

def test_grad_sqrt_sum():
    """
    This trigered a bug in the past.
    """
    I = T.ftensor4().reshape((1,1,28,28))

    W = cuda.shared_constructor(numpy.asarray(
            numpy.random.random((1,1,10,10)),dtype='float32'))

    C = T.sqrt(T.sum(W[:,:,1:1,1:1]**2))# This line was causing a bug.
    #C = T.sqrt(T.sum(W[:,:,1:2,1:2]**2))# This line work

    g = T.grad(C, W)

    bog = theano.function([I], C,
                          updates = {W:W-g}, mode=mode_with_gpu)

    theano.printing.debugprint(bog)
    print bog.maker.env.toposort()

    print "BEFORE"
    bog(numpy.asarray(numpy.ones((1,1,28,28)),dtype='float32'))
    print "AFTER"
