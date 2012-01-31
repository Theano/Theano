import sys, time, unittest

import numpy
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

from theano.compile.pfunc import pfunc
from theano import config, tensor
import theano

from theano.tests import unittest_tools as utt

import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

from theano.sandbox.cuda.type import CudaNdarrayType

if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


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

def test_gpualloc():
    '''
    This tests tries to catch the scenario when, due to infer_shape,
    the input of the alloc changes from tesnor scalar to a constant
    1. In this case the original constracted broadcastable pattern will
    have a False for that dimension, but the new broadcastable pattern
    that will be inserted by gpualloc will have  a True since it knows the
    dimension is 1 and therefore broadcastable.
    '''

    x = theano.shared(numpy.ones(3,dtype='float32'), 'x')
    m = (x).dimshuffle(['x',0])
    v = tensor.alloc(1., *m.shape)
    f = theano.function([], v+x)
    l = f.maker.env.toposort()
    assert numpy.any(ininstance(x.op, cuda.GpuAlloc) for x in l )



def test_softmax():
    x = tensor.fmatrix()

    f = theano.function([x],tensor.nnet.nnet.Softmax()(x), mode=mode_with_gpu)
    f2 = theano.function([x],tensor.nnet.nnet.Softmax()(x), mode=mode_without_gpu)
    assert isinstance(f.maker.env.toposort()[1].op,cuda.nnet.GpuSoftmax)
    xv=numpy.random.rand(7,8).astype('float32')
    assert numpy.allclose(f(xv),f2(xv))


def test_softmax_with_bias():
    x = tensor.fmatrix()
    b = tensor.fvector()

    f = theano.function([x,b],tensor.nnet.nnet.SoftmaxWithBias()(x,b), mode=mode_with_gpu)
    f2 = theano.function([x,b],tensor.nnet.nnet.SoftmaxWithBias()(x,b), mode=mode_without_gpu)
    assert isinstance(f.maker.env.toposort()[2].op,cuda.nnet.GpuSoftmaxWithBias)
    xv=numpy.random.rand(7,8).astype('float32')
    bv=numpy.random.rand(8).astype('float32')
    assert numpy.allclose(f(xv,bv),f2(xv,bv))

def test_opt_gpujoin_onlyajoin():
    # from a bug in normal sampling
    _a = numpy.asarray([[1,2],[3,4]],dtype='float32')
    _b = numpy.asarray([[5,6,7],[8,9,10]],dtype='float32')
    a = cuda.shared_constructor(_a)
    b = cuda.shared_constructor(_b)

    c = tensor.join(1,a,b)

    f = theano.function([], c, mode=mode_with_gpu)

    #theano.printing.debugprint(f)

    f()

    graph_nodes = f.maker.env.toposort()

    assert isinstance(graph_nodes[-1].op, cuda.HostFromGpu)
    assert isinstance(graph_nodes[-2].op, cuda.GpuJoin)

    assert numpy.all(f() == numpy.concatenate([_a,_b], axis=1))



def test_opt_gpujoin_joinvectors_elemwise_then_minusone():
    # from a bug in gpu normal sampling
    _a = numpy.asarray([1,2,3,4],dtype='float32')
    _b = numpy.asarray([5,6,7,8],dtype='float32')
    a = cuda.shared_constructor(_a)
    b = cuda.shared_constructor(_b)

    a_prime = tensor.cos(a)
    b_prime = tensor.sin(b)

    c = tensor.join(0,a_prime,b_prime)

    d = c[:-1]

    f = theano.function([], d, mode=mode_with_gpu)

    #theano.printing.debugprint(f)

    graph_nodes = f.maker.env.toposort()

    assert isinstance(graph_nodes[-1].op, cuda.HostFromGpu)
    assert isinstance(graph_nodes[-2].op, cuda.GpuSubtensor)
    assert isinstance(graph_nodes[-3].op, cuda.GpuJoin)

    concat = numpy.concatenate([numpy.cos(_a),numpy.sin(_b)],axis=1)
    concat = concat[:-1]

    assert numpy.allclose(numpy.asarray(f()), concat)

def test_print_op():
    """ Test that print ops don't block gpu optimization"""
    b = tensor.fmatrix()
    f = theano.function([b],theano.printing.Print()(b)*2, mode=mode_with_gpu)
    #theano.printing.debugprint(f)
    #print f.maker.env.toposort()
#[GpuFromHost(<TensorType(float32, matrix)>), <theano.printing.Print object at 0x3581210>(GpuFromHost.0), GpuElemwise{mul}(CudaNdarray{[[ 2.]]}, <theano.printing.Print object at 0x3581210>.0), HostFromGpu(GpuElemwise{mul}.0)]
    topo = f.maker.env.toposort()
    assert topo[0].op == cuda.gpu_from_host
    assert isinstance(topo[1].op, theano.printing.Print)
    assert isinstance(topo[2].op, cuda.GpuElemwise)
    assert topo[3].op == cuda.host_from_gpu
    f(numpy.random.random((5,5)).astype('float32'))

def test_huge_elemwise_fusion():
    """ Test the the GpuElemwise fusion work correctly
        We check that we fuse one node with part of its input
        in case their is too many inputs and that would make it bust the 256
        bytes limits.
    """
    shape = (2,3,4,5,6)
    ttype = tensor.tensor(dtype='float32',broadcastable=(False,)*len(shape))
    vars = [tensor.tanh(ttype) for x in range(10)]
    f = pfunc(vars, [vars[0]-vars[1]-vars[2]-vars[3]-vars[4]-vars[5]-vars[6]], mode=mode_with_gpu)
    topo = f.maker.env.toposort()
    #theano.printing.debugprint(f)
    #for i, node in enumerate(topo):
    #    print >> sys.stdout, i, node
    assert len(topo)==10
    assert sum([isinstance(node.op, cuda.GpuElemwise) for node in topo])==2
    assert isinstance(topo[7].op.scalar_op,theano.scalar.basic.Sub)
    assert isinstance(topo[8].op.scalar_op,theano.scalar.basic.Composite)
    #let debugmode catch errors
    gen = lambda : theano._asarray(numpy.random.rand(*shape), dtype='float32')
    f(gen(),gen(),gen(),gen(),gen(),gen(),gen(),gen(),gen(),gen())

    # Test the case where we can't put the computation on the gpu! their is too many
    # dimensions to the input to have 2 inputs to the op!

    shape = (1,2,3,4,5,6,7,2,2,3,2,1,2,2,2,)
    ttype = tensor.tensor(dtype='float32',broadcastable=(False,)*len(shape))
    vars = [tensor.tanh(ttype) for x in range(10)]
    f = pfunc(vars, [vars[0]-vars[1]-vars[2]-vars[3]-vars[4]-vars[5]-vars[6]], mode=mode_with_gpu)
    topo = f.maker.env.toposort()
    #theano.printing.debugprint(f)
    assert len(topo) == 1
    assert sum([isinstance(node.op, cuda.GpuElemwise) for node in topo]) == 0
    assert sum([isinstance(node.op, tensor.Elemwise) for node in topo]) == 1
    #let debugmode catch errors
    gen = lambda: theano._asarray(numpy.random.rand(*shape), dtype='float32')
    f(gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen(), gen())

    def gen(shape):
        return theano._asarray(numpy.random.rand(*shape), dtype='float32')

    max_var = 16  # excluded
    for shape in [(2,),
                  (2, 2),
                  (2, 2, 2),
                  (2, 2, 2, 2),
                  (2, 2, 2, 2, 2),  # 5d
                  (2, 2, 2, 2, 2, 2),
#                  (2, 2, 2, 2, 2, 2, 2),
#                  (2, 2, 2, 2, 2, 2, 2, 2),
#                  (2, 2, 2, 1, 1, 1, 1, 2, 2),  # 9d
                  ]:
        vals = [cuda.shared_constructor(gen(shape)) for x in range(max_var)]
        for use_tan in [True, False]:
            if use_tan:
                vars = [tensor.tanh(x) for x in vals]
            else:
                vars = vals
            for nb_var in range(1, max_var):
                out = reduce(lambda x, y: x + y, vars[:nb_var])
                if not isinstance(out.type, CudaNdarrayType):
                    out = cuda.gpu_from_host(out)
                f = pfunc([], [out], mode=mode_with_gpu)
                topo = f.maker.env.toposort()
                #print shape, nb_var, use_tan, len(topo)
                assert (sum([isinstance(node.op, cuda.GpuElemwise)
                             for node in topo]) == len(topo) or
                        (nb_var == 1 and use_tan == False))
                assert sum([isinstance(node.op, tensor.Elemwise)
                            for node in topo]) == 0

                #let debugmode catch errors
                f()


def test_elemwise_fusion():
    """ Test the the GpuElemwise fusion work correctly"""
    shape = (3,4)
    a = cuda.shared_constructor(theano._asarray(numpy.random.rand(*shape), dtype='float32'), 'a')
    b = tensor.fmatrix()
    c = tensor.fmatrix()
    f = pfunc([b,c], [a+b+c], mode=mode_with_gpu)
    topo = f.maker.env.toposort()
    for i, node in enumerate(topo):
        print >> sys.stdout, i, node
    assert len(topo)==4
    assert isinstance(topo[2].op.scalar_op,theano.scalar.basic.Composite)
    #let debugmode catch errors
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32'), theano._asarray(numpy.random.rand(*shape), dtype='float32'))


class test_local_gpu_tensordot(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed())

    def test_transfer(self):
        tensor1 = self.rng.rand(20, 10, 5, 8).astype('float32')
        tensor2 = self.rng.rand(5, 8, 20).astype('float32')
        tensor3 = self.rng.rand(8, 20, 5).astype('float32')

        x = tensor.ftensor4('x')
        y = tensor.ftensor3('y')

        tdot1 = tensor.tensordot(x, y, 2)
        f1 = theano.function([x, y], tdot1, mode=mode_with_gpu)
        topo1 = f1.maker.env.toposort()
        assert topo1[-1].op == cuda.host_from_gpu
        # Let DebugMode debug
        f1(tensor1, tensor2)

        tdot2 = tensor.tensordot(x, y, axes=[(0, 3), (1, 0)])
        f2 = theano.function([x, y], tdot2, mode=mode_with_gpu)
        topo2 = f2.maker.env.toposort()
        assert topo2[-1].op == cuda.host_from_gpu
        f2(tensor1, tensor3)

        tdot3 = tensor.tensordot(x, y, axes=[(0, 3, 2), (1, 0, 2)])
        f3 = theano.function([x, y], tdot3, mode=mode_with_gpu)
        topo3 = f3.maker.env.toposort()
        assert topo3[-1].op == cuda.host_from_gpu
        f3(tensor1, tensor3)


if __name__ == '__main__':
    test_gpualloc()
    test_opt_gpujoin_onlyajoin()
    test_opt_gpujoin_joinvectors_elemwise_then_minusone()
