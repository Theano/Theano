import sys, time
from theano.compile.sharedvalue import shared
from theano.compile.pfunc import pfunc
from theano import tensor
import theano
import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.compile.mode
from theano.sandbox.cuda.type import CudaNdarrayType

if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

import theano.sandbox.cuda as cuda


def test_no_shared_var_graph():
    """Test that the InputToGpuOptimizer optimizer make graph that don't have shared variable compiled too.
    """
    a=tensor.fmatrix()
    b=tensor.fmatrix()
    f = theano.function([a,b],[a+b], mode=mode_with_gpu)
    l = f.maker.env.toposort()
    assert len(l)==4
    assert numpy.any(isinstance(x.op,cuda.GpuElemwise) for x in l)
    assert numpy.any(isinstance(x.op,cuda.GpuFromHost) for x in l)
    assert numpy.any(isinstance(x.op,cuda.HostFromGpu) for x in l)

def test_int_pow():
    a = CudaNdarrayType([False])()

    f = theano.function([a], (a*4).sum(), mode=mode_with_gpu)

    op_names = [n.op.__class__.__name__ for n in f.maker.env.toposort()]
    assert op_names == ['GpuSum', 'GpuElemwise', 'HostFromGpu']

    f = theano.function([a], tensor.pow(a,4).sum(), mode=mode_with_gpu)
    op_names = [n.op.__class__.__name__ for n in f.maker.env.toposort()]
    assert op_names == ['GpuElemwise', 'GpuSum', 'HostFromGpu']

    #theano.printing.debugprint(f)


def test_softmax():
    x = tensor.fmatrix()

    f = theano.function([x],tensor.nnet.nnet.Softmax()(x), mode=mode_with_gpu)
    f2 = theano.function([x],tensor.nnet.nnet.Softmax()(x), mode=mode_without_gpu)
    assert isinstance(f.maker.env.toposort()[1].op,cuda.nnet.GpuSoftmax)
    xv=numpy.random.rand(7,8)
    assert numpy.allclose(f(xv),f2(xv))


def test_softmax_with_bias():
    x = tensor.fmatrix()
    b = tensor.fvector()

    f = theano.function([x,b],tensor.nnet.nnet.SoftmaxWithBias()(x,b), mode=mode_with_gpu)
    f2 = theano.function([x,b],tensor.nnet.nnet.SoftmaxWithBias()(x,b), mode=mode_without_gpu)
    assert isinstance(f.maker.env.toposort()[2].op,cuda.nnet.GpuSoftmaxWithBias)
    xv=numpy.random.rand(7,8)
    bv=numpy.random.rand(8)
    assert numpy.allclose(f(xv,bv),f2(xv,bv))



def test_opt_gpujoin_joinvectors_elemwise_than_minusone():
    # from a bug in normal sampling
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    _b = numpy.asarray([[5,6,7],[8,9,10]],dtype='float32')
    a = theano.shared(_a)
    b = theano.shared(_b)

    c = tensor.join(1,a,b)

    f = theano.function([], c)

    #theano.printing.debugprint(f)

    graph_nodes = f.maker.env.toposort()

    assert isinstance(graph_nodes[-1].op, cuda.HostFromGpu)
    assert isinstance(graph_nodes[-2].op, cuda.GpuJoin)

    assert numpy.all(f() == numpy.concatenate([_a,_b], axis=1))



if __name__ == '__main__':
    test_opt_gpujoin_onlyajoin()
    test_opt_gpujoin_joinvectors_elemwise_than_minusone()
