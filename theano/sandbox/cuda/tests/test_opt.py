from __future__ import absolute_import, print_function, division
import operator
import sys
import unittest

import numpy
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises

import theano
from six.moves import reduce
from theano.compile.pfunc import pfunc
from theano import config, tensor
import theano.tensor.tests.test_nlinalg
import theano.tensor.tests.test_opt as test_opt

from theano.tests.breakpoint import PdbBreakpoint
from theano.tests import unittest_tools as utt

import theano.sandbox.cuda as cuda

if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda.cula as cula

from theano.sandbox.cuda import basic_ops
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.scalar.basic_scipy import erfinv

from theano.tensor.nnet.blocksparse import sparse_block_dot
from theano.sandbox.cuda.blocksparse import GpuSparseBlockGemv, GpuSparseBlockOuter

imported_scipy_special = False
try:
    import scipy.special
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_no_shared_var_graph():
    """Test that the InputToGpuOptimizer optimizer make graph that don't have shared variable compiled too.
    """
    a = tensor.fmatrix()
    b = tensor.fmatrix()
    f = theano.function([a, b], [a + b], mode=mode_with_gpu)
    l = f.maker.fgraph.toposort()
    assert len(l) == 4
    assert numpy.any(isinstance(x.op, cuda.GpuElemwise) for x in l)
    assert numpy.any(isinstance(x.op, cuda.GpuFromHost) for x in l)
    assert numpy.any(isinstance(x.op, cuda.HostFromGpu) for x in l)


def test_local_assert():
    x = theano.tensor.fmatrix()
    a = theano.tensor.opt.assert_op(x, theano.tensor.eq(x, 0).any())
    f = theano.function([x], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, theano.tensor.opt.Assert)]
    assert len(a_op) == 1
    assert isinstance(a_op[0].inputs[0].type, CudaNdarrayType)


def test_local_remove_all_assert():
    x = theano.tensor.fmatrix()
    a = theano.tensor.opt.assert_op(x, theano.tensor.eq(x, 0).any())

    # By default `unsafe` should not be there
    f = theano.function([x], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, theano.tensor.opt.Assert)]
    assert len(a_op) == 1

    # Put `unsafe`
    f = theano.function([x], a, mode=mode_with_gpu.including('unsafe'))
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, theano.tensor.opt.Assert)]
    assert len(a_op) == 0

    # Remove `unsafe`
    f = theano.function([x], a, mode=mode_with_gpu.excluding('unsafe'))
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, theano.tensor.opt.Assert)]
    assert len(a_op) == 1


def test_local_gpu_contiguous_gpu_contiguous():
    a = tensor.fmatrix()
    o1 = basic_ops.gpu_contiguous(a)
    o2 = basic_ops.gpu_contiguous(o1)
    f1 = theano.function([a], o1, mode=mode_with_gpu)
    f2 = theano.function([a], o2, mode=mode_with_gpu)
    assert 1 == len([node for node in f1.maker.fgraph.toposort()
                     if isinstance(node.op, basic_ops.GpuContiguous)])
    assert 1 == len([node for node in f2.maker.fgraph.toposort()
                     if isinstance(node.op, basic_ops.GpuContiguous)])


def test_local_assert_no_cpu_op():
    numpy.random.seed(1)
    m = numpy.random.uniform(-1, 1, (10, 10)).astype("float32")
    ms = cuda.shared_constructor(m, name="m_shared")
    out = theano.tensor.tanh(ms).dot(ms.T)

    mode_local_assert = mode_with_gpu.including("assert_no_cpu_op")
    mode_local_assert = mode_local_assert.excluding("local_gpu_elemwise_0")
    mode_local_assert = mode_local_assert.excluding("local_gpu_elemwise_1")

    old = config.assert_no_cpu_op
    old2 = config.on_opt_error
    # If the flag is raise
    try:
        config.assert_no_cpu_op = 'raise'
        config.on_opt_error = 'ignore'

        assert_raises(AssertionError, theano.function,
                        [], out, mode=mode_local_assert)
    finally:
        config.assert_no_cpu_op = old
        config.on_opt_error = old2

    # If the flag is ignore
    try:
        config.assert_no_cpu_op = 'ignore'
        theano.function([], out, mode=mode_local_assert)
    finally:
        config.assert_no_cpu_op = old


def test_int_pow():
    a = CudaNdarrayType([False])()

    f = theano.function([a], (a*4).sum(), mode=mode_with_gpu)

    op_names = [n.op.__class__.__name__ for n in f.maker.fgraph.toposort()]
    assert op_names == ['GpuCAReduce', 'GpuElemwise', 'HostFromGpu']

    f = theano.function([a], tensor.pow(a, 4).sum(), mode=mode_with_gpu)
    op_names = [n.op.__class__.__name__ for n in f.maker.fgraph.toposort()]
    assert op_names == ['GpuElemwise', 'GpuCAReduce', 'HostFromGpu']


def test_gpualloc():
    '''
    This tests tries to catch the scenario when, due to infer_shape,
    the input of the alloc changes from tensor scalar to a constant
    1. In this case the original constracted broadcastable pattern will
    have a False for that dimension, but the new broadcastable pattern
    that will be inserted by gpualloc will have  a True since it knows the
    dimension is 1 and therefore broadcastable.
    '''

    x = theano.shared(numpy.ones(3, dtype='float32'), 'x')
    m = (x).dimshuffle(['x', 0])
    v = tensor.alloc(1., *m.shape)
    f = theano.function([], v + x,
                        mode=mode_with_gpu.excluding("local_elemwise_alloc"))
    l = f.maker.fgraph.toposort()
    assert numpy.any([isinstance(x.op, cuda.GpuAlloc) for x in l])


def test_gpuallocempty():

    f_gpu = theano.function([], tensor.AllocEmpty('float32')(2,3),
                        mode=mode_with_gpu)
    l_gpu = f_gpu.maker.fgraph.toposort()

    assert numpy.any([isinstance(x.op, basic_ops.GpuAllocEmpty) for x in l_gpu])

    f_cpu = theano.function([], tensor.AllocEmpty('int32')(2,3))
    l_cpu = f_cpu.maker.fgraph.toposort()
    assert not numpy.any([isinstance(x.op, basic_ops.GpuAllocEmpty) for x in l_cpu])

class Test_local_elemwise_alloc(test_opt.Test_local_elemwise_alloc):
    dtype = 'float32'

    def setUp(self):
        super(Test_local_elemwise_alloc, self).setUp()
        self.fast_run_mode = mode_with_gpu

        # self.vec = tensor.vector('vec', dtype=dtype)
        # self.mat = tensor.matrix('mat', dtype=dtype)
        # self.tens = tensor.tensor3('tens', dtype=dtype)

        # self.alloc_wo_dep = basic_ops.gpu_alloc(self.vec, 2, 2)
        # self.alloc_w_dep = basic_ops.gpu_alloc(self.vec, *self.mat.shape)

        self.alloc_wo_dep = basic_ops.gpu_alloc(self.vec, 2, 2)
        self.alloc_w_dep = basic_ops.gpu_alloc(self.vec, *self.mat.shape)
        self.alloc_w_dep_tens = basic_ops.gpu_alloc(
            self.vec,
            self.tens.shape[0],
            self.tens.shape[1]
        )
        self.tv_wo_dep = basic_ops.gpu_alloc(self.vec, 5, 5)
        self.tm_wo_dep = basic_ops.gpu_alloc(self.mat, 5, 5, 5)
        self.s = tensor.iscalar('s')
        self.tv_w_dep = basic_ops.gpu_alloc(self.vec, self.s, self.s)
        self.tm_w_dep = basic_ops.gpu_alloc(self.mat, 5, 5, 5)
        self.row = tensor.row(dtype=self.dtype)
        self.o = basic_ops.gpu_alloc(self.row, 5, 5)

    def _verify_alloc_count(self, f, count):
        assert(
            sum([isinstance(elem.op, basic_ops.GpuAlloc)
                 for elem in f.maker.fgraph.toposort()
                 if elem.op is not None]) == count
        )


def test_alloc_memset_0():
    i = tensor.iscalar()
    z = numpy.zeros((1,), dtype='float32')
    o = numpy.ones((1,), dtype='float32')
    ones = numpy.ones((2,), dtype='float32')

    # Test with 0
    a = basic_ops.gpu_alloc(cuda.gpu_from_host(tensor.constant(z)), i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, basic_ops.GpuAlloc) and topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 0).all()

    # Test with 1
    a = basic_ops.gpu_alloc(cuda.gpu_from_host(tensor.constant(o)), i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, basic_ops.GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 1).all()

    # Test with 1, 1
    a = basic_ops.gpu_alloc(cuda.gpu_from_host(tensor.constant(ones)), i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, basic_ops.GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(2)) == 1).all()


def test_gpuspecifyshape():
    x = cuda.shared_constructor(numpy.ones(3, dtype='float32'), 'x')
    m = theano.tensor.specify_shape(x + numpy.float32(1), (3,))
    f = theano.function([], updates=[(x, m * numpy.float32(2))],
                        mode=mode_with_gpu)
    l = f.maker.fgraph.toposort()
    assert not numpy.any([isinstance(x.op, cuda.HostFromGpu) for x in l])


def test_softmax():
    x = tensor.fmatrix()

    f = theano.function([x], tensor.nnet.nnet.Softmax()(x),
                        mode=mode_with_gpu.excluding('cudnn'))
    f2 = theano.function([x], tensor.nnet.nnet.Softmax()(x),
                         mode=mode_without_gpu)
    assert isinstance(f.maker.fgraph.toposort()[1].op, cuda.nnet.GpuSoftmax)
    xv = numpy.random.rand(7, 8).astype('float32')
    assert numpy.allclose(f(xv), f2(xv))


def test_softmax_with_bias():
    x = tensor.fmatrix()
    b = tensor.fvector()

    f = theano.function([x, b], tensor.nnet.nnet.SoftmaxWithBias()(x, b),
                        mode=mode_with_gpu)
    f2 = theano.function([x, b], tensor.nnet.nnet.SoftmaxWithBias()(x, b),
                         mode=mode_without_gpu)
    assert isinstance(f.maker.fgraph.toposort()[2].op,
                      cuda.nnet.GpuSoftmaxWithBias)
    xv = numpy.random.rand(7, 8).astype('float32')
    bv = numpy.random.rand(8).astype('float32')
    assert numpy.allclose(f(xv, bv), f2(xv, bv))


def test_opt_gpujoin_onlyajoin():
    # from a bug in normal sampling
    _a = numpy.asarray([[1, 2], [3, 4]], dtype='float32')
    _b = numpy.asarray([[5, 6, 7], [8, 9, 10]], dtype='float32')
    a = cuda.shared_constructor(_a)
    b = cuda.shared_constructor(_b)

    c = tensor.join(1, a, b)

    f = theano.function([], c, mode=mode_with_gpu)

    f()

    graph_nodes = f.maker.fgraph.toposort()

    assert isinstance(graph_nodes[-1].op, cuda.HostFromGpu)
    assert isinstance(graph_nodes[-2].op, cuda.GpuJoin)

    assert numpy.all(f() == numpy.concatenate([_a, _b], axis=1))

    # test mixed dtype
    _b = numpy.asarray([[5, 6, 7], [8, 9, 10]], dtype='float64')
    b = theano.tensor.constant(_b)

    c = tensor.join(1, a, b)

    f = theano.function([], c, mode=mode_with_gpu)

    f()

    graph_nodes = f.maker.fgraph.toposort()
    assert isinstance(graph_nodes[-1].op, theano.tensor.Join)

    assert numpy.all(f() == numpy.concatenate([_a, _b], axis=1))


def test_opt_gpujoin_joinvectors_elemwise_then_minusone():
    # from a bug in gpu normal sampling
    _a = numpy.asarray([1, 2, 3, 4], dtype='float32')
    _b = numpy.asarray([5, 6, 7, 8], dtype='float32')
    a = cuda.shared_constructor(_a)
    b = cuda.shared_constructor(_b)

    a_prime = tensor.cos(a)
    b_prime = tensor.sin(b)

    c = tensor.join(0, a_prime, b_prime)

    d = c[:-1]

    f = theano.function([], d, mode=mode_with_gpu)

    graph_nodes = f.maker.fgraph.toposort()

    assert isinstance(graph_nodes[-1].op, cuda.HostFromGpu)
    assert isinstance(graph_nodes[-2].op, cuda.GpuSubtensor)
    assert isinstance(graph_nodes[-3].op, cuda.GpuJoin)

    concat = numpy.concatenate([numpy.cos(_a), numpy.sin(_b)], axis=0)
    concat = concat[:-1]

    assert numpy.allclose(numpy.asarray(f()), concat)


def test_opt_gpujoin_joinvectors_negativeaxes():
    """
    Test that negative axis concatenation works as expected.
    """

    # Test case for one-dimensional vectors
    rng = numpy.random.RandomState(22)
    x1 = rng.rand(5)
    x2 = rng.rand(10)
    t1 = cuda.shared_constructor(numpy.asarray(x1, "float32"))
    t2 = cuda.shared_constructor(numpy.asarray(x2, "float32"))

    t = tensor.concatenate([t1, t2], axis=-1)
    f = theano.function(inputs=[], outputs=t)

    assert(numpy.allclose(f(), numpy.concatenate([x1, x2], axis=-1)))

    # Test case for two-dimensional vectors
    x1 = rng.rand(5, 10)
    x2 = rng.rand(10, 10)
    t1 = cuda.shared_constructor(numpy.asarray(x1, "float32"))
    t2 = cuda.shared_constructor(numpy.asarray(x2, "float32"))

    t = tensor.concatenate([t1, t2], axis=-2)
    f = theano.function(inputs=[], outputs=t)

    assert(numpy.allclose(f(), numpy.concatenate([x1, x2], axis=-2)))

    # Now check that a value error is raised when vectors don't match
    # along the negative concatenation axis
    try:
        t = tensor.concatenate([t1, t2], axis=-1)
        f = theano.function(inputs=[], outputs=t)
        f()
        assert(False)
    except ValueError:
        assert(True)

    # Finally check that a value error is raised when negative
    # axis is larger in absolute value than smallest number of dims
    try:
        t = tensor.concatenate([t1, t2], axis=-3)
        f = theano.function(inputs=[], outputs=t)
        f()
        assert(False)
    except IndexError:
        assert(True)


def test_local_gpu_subtensor():
    # Test shared forced on CPU.
    t = tensor._shared(numpy.zeros(20, "float32"))
    f = theano.function([], t[3:4], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, cuda.GpuSubtensor) for node in topo])

    # Test graph input.
    t = tensor.fmatrix()
    f = theano.function([t], t[3:4], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, cuda.GpuSubtensor) for node in topo])

    # Test multiple use of the input
    # We want the subtensor to be on the GPU to prevent multiple transfer.
    t = tensor.fmatrix()
    f = theano.function([t], [t[3:4], t+1], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert not any([type(node.op) is tensor.Subtensor for node in topo])
    assert any([isinstance(node.op, cuda.GpuSubtensor) for node in topo])

    # Test multiple use of the input + input as output
    # We want the subtensor to be on the GPU to prevent multiple transfer.
    t = tensor.fmatrix()
    f = theano.function([t], [t[3:4], t+1, t], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert not any([type(node.op) is tensor.Subtensor for node in topo])
    assert any([isinstance(node.op, cuda.GpuSubtensor) for node in topo])

    # Test shared forced on CPU end we do computation on the output of
    # the subtensor.
    t = tensor._shared(numpy.zeros(20, "float32"))
    f = theano.function([], t[3:4]+1, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, cuda.GpuSubtensor) for node in topo])
    assert any([isinstance(node.op, cuda.GpuElemwise) for node in topo])


def test_local_gpu_split():
    """ Test that the GpuSplit op is being applied and works """
    # Construct symbolic split
    x = tensor.fvector()
    splits = tensor.lvector()
    ra, rb, rc = tensor.split(x, splits, n_splits=3, axis=0)
    # Compile function to use CPU
    f = theano.function([x, splits], [ra, rb, rc], mode=mode_without_gpu)
    # Get values for CPU version
    cpu_res = f([0, 1, 2, 3, 4, 5], [3, 2, 1])
    l = f.maker.fgraph.toposort()
    # Ensure that one op is theano.tensor.Split
    assert any([isinstance(o.op, theano.tensor.Split) for o in l])
    # GPU version
    f = theano.function([x, splits], [ra, rb, rc], mode=mode_with_gpu)
    gpu_res = f([0, 1, 2, 3, 4, 5], [3, 2, 1])
    l = f.maker.fgraph.toposort()
    assert any([isinstance(o.op, cuda.GpuSplit) for o in l])
    # Check equality
    assert all([(cpu == gpu).all() for cpu, gpu in zip(cpu_res, gpu_res)])

    # Test the other path of the optimizer, when it is the output that
    # is moved to the GPU.
    ra = cuda.gpu_from_host(ra)
    f = theano.function([x, splits], [ra, rb, rc],
                        mode=mode_with_gpu.excluding("InputToGpuOptimizer"))
    gpu_res = f([0, 1, 2, 3, 4, 5], [3, 2, 1])
    l = f.maker.fgraph.toposort()
    assert any([isinstance(o.op, cuda.GpuSplit) for o in l])
    # Check equality
    assert all([(cpu == gpu).all() for cpu, gpu in zip(cpu_res, gpu_res)])

    # Test that split with only 1 output work
    ra = tensor.split(x, splits, n_splits=1, axis=0)
    f = theano.function([x, splits], [ra], mode=mode_without_gpu)
    cpu_res = f([0, 1, 2, 3, 4, 5], [6])
    l = f.maker.fgraph.toposort()
    # Ensure that no op is theano.tensor.Split or GpuSplit, they get
    # optimized away.
    assert not any([isinstance(o.op, (theano.tensor.Split,
                                      cuda.GpuSplit)) for o in l])
    # GPU version
    f = theano.function([x, splits], [ra], mode=mode_with_gpu)
    gpu_res = f([0, 1, 2, 3, 4, 5], [6])
    l = f.maker.fgraph.toposort()
    assert not any([isinstance(o.op, (theano.tensor.Split,
                                      cuda.GpuSplit)) for o in l])
    # Check equality
    assert all([(cpu == gpu).all() for cpu, gpu in zip(cpu_res, gpu_res)])


def test_print_op():
    """ Test that print ops don't block gpu optimization"""
    b = tensor.fmatrix()
    f = theano.function([b], theano.printing.Print()(b)*2, mode=mode_with_gpu)
    # theano.printing.debugprint(f)
    # print f.maker.fgraph.toposort()
#[GpuFromHost(<TensorType(float32, matrix)>), <theano.printing.Print object at 0x3581210>(GpuFromHost.0), GpuElemwise{mul}(CudaNdarray{[[ 2.]]}, <theano.printing.Print object at 0x3581210>.0), HostFromGpu(GpuElemwise{mul}.0)]
    topo = f.maker.fgraph.toposort()
    assert topo[0].op == cuda.gpu_from_host
    assert isinstance(topo[1].op, theano.printing.Print)
    assert isinstance(topo[2].op, cuda.GpuElemwise)
    assert topo[3].op == cuda.host_from_gpu
    f(numpy.random.random((5, 5)).astype('float32'))


def test_pdbbreakpoint_op():
    """ Test that PdbBreakpoint ops don't block gpu optimization"""
    b = tensor.fmatrix()

    # Create a function composed of a breakpoint followed by
    # some computation
    condition = tensor.gt(b.sum(), 0)
    b_monitored = PdbBreakpoint(name='TestBreakpoint')(condition, b)
    output = b_monitored ** 2

    f = theano.function([b], output, mode=mode_with_gpu)

    # Ensure that, in the compiled function, the computation following the
    # breakpoint has been moved to the gpu.
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[-2].op, cuda.GpuElemwise)
    assert topo[-1].op == cuda.host_from_gpu


def test_local_gpu_elemwise_careduce():
    x = theano.tensor.fmatrix()
    o = (x * x).sum()
    f = theano.function([x], o, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert topo[1].op.pre_scalar_op == theano.scalar.sqr
    data = numpy.random.rand(3, 4).astype('float32')
    utt.assert_allclose(f(data), (data * data).sum())

    o = (x * x).sum(axis=1)
    f = theano.function([x], o, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert topo[1].op.pre_scalar_op == theano.scalar.sqr
    utt.assert_allclose(f(data), (data * data).sum(axis=1))


def test_huge_elemwise_fusion():
    """ Test the the GpuElemwise fusion work correctly
        We check that we fuse one node with part of its input
        in case their is too many inputs and that would make it bust the 256
        bytes limits.
    """
    shape = (2, 3, 4, 5, 6)
    ttype = tensor.tensor(dtype='float32', broadcastable=(False,) * len(shape))
    gpu_ptr_size = theano.sandbox.cuda.opt.get_device_type_sizes()['gpu_ptr_size']
    if gpu_ptr_size == 8:
        nb_in = 7
        len_topo = 10
    elif gpu_ptr_size == 4:
        nb_in = 8
        len_topo = 11
    else:
        raise Exception("Unexpected value for gpu_ptr_size", gpu_ptr_size)
    vars = [tensor.tanh(ttype) for x in range(nb_in)]
    f = pfunc(vars, [reduce(operator.sub, vars)], mode=mode_with_gpu)

    topo = f.maker.fgraph.toposort()
    assert len(topo) == len_topo
    assert sum([isinstance(node.op, cuda.GpuElemwise) for node in topo]) == 2
    assert isinstance(topo[-3].op.scalar_op, theano.scalar.basic.Sub)
    assert isinstance(topo[-2].op.scalar_op, theano.scalar.basic.Composite)
    # let debugmode catch errors
    gen = lambda: theano._asarray(numpy.random.rand(*shape), dtype='float32')
    f(*[gen() for i in range(nb_in)])

    # Test the case where we can't put the computation on the gpu! their is too
    # many dimensions to the input to have 2 inputs to the op!

    shape = (1, 2, 3, 4, 5, 6, 7, 2, 2, 3, 2, 1, 2, 2, 2,)
    ttype = tensor.tensor(dtype='float32', broadcastable=(False,) * len(shape))
    vars = [tensor.tanh(ttype) for x in range(7)]
    f = pfunc(vars, [vars[0] - vars[1] - vars[2] - vars[3] - vars[4] -
                     vars[5] - vars[6]], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert sum([isinstance(node.op, cuda.GpuElemwise) for node in topo]) == 0
    assert sum([isinstance(node.op, tensor.Elemwise) for node in topo]) == 1
    # let debugmode catch errors
    gen = lambda: theano._asarray(numpy.random.rand(*shape), dtype='float32')
    f(gen(), gen(), gen(), gen(), gen(), gen(), gen())

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
                topo = f.maker.fgraph.toposort()
                # print shape, nb_var, use_tan, len(topo)
                assert (sum([isinstance(node.op, cuda.GpuElemwise)
                             for node in topo]) == len(topo) or
                        (nb_var == 1 and use_tan is False))
                assert sum([isinstance(node.op, tensor.Elemwise)
                            for node in topo]) == 0

                # let debugmode catch errors
                f()


def test_local_gpu_elemwise_0():
    """
    Test local_gpu_elemwise_0 when there is a dtype upcastable to float32
    """
    a = tensor.bmatrix()
    b = tensor.fmatrix()
    c = tensor.fmatrix()

    a_v = (numpy.random.rand(4, 5) * 10).astype("int8")
    b_v = (numpy.random.rand(4, 5) * 10).astype("float32")
    c_v = (numpy.random.rand(4, 5) * 10).astype("float32")

    # Due to optimization order, this composite is created when all
    # the op are on the gpu.
    f = theano.function([a, b, c], a + b + c, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, cuda.GpuElemwise) for node in topo) == 1
    assert sum(isinstance(node.op, tensor.Elemwise) for node in topo) == 1
    utt.assert_allclose(f(a_v, b_v, c_v), a_v + b_v + c_v)

    # Now test with the composite already on the cpu before we move it
    # to the gpu
    a_s = theano.scalar.int8()
    b_s = theano.scalar.float32()
    c_s = theano.scalar.float32()
    out_s = theano.scalar.Composite([a_s, b_s, c_s], [a_s + b_s + c_s])
    out_op = tensor.Elemwise(out_s)
    f = theano.function([a, b, c], out_op(a, b, c), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, cuda.GpuElemwise) for node in topo) == 1
    assert sum(isinstance(node.op, tensor.Elemwise) for node in topo) == 1
    utt.assert_allclose(f(a_v, b_v, c_v), a_v + b_v + c_v)

    # Test multiple output
    a_s = theano.scalar.float32()
    a = tensor.fmatrix()
    from theano.scalar.basic import identity
    out_s = theano.scalar.Composite([a_s, b_s, c_s],
                                    [identity(a_s), identity(c_s), identity(b_s)])
    outs_op = tensor.Elemwise(out_s)
    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, cuda.GpuElemwise) for node in topo) == 1
    assert sum(isinstance(node.op, tensor.Elemwise) for node in topo) == 0
    out = f(a_v, b_v, c_v)
    utt.assert_allclose(out[0], a_v)
    utt.assert_allclose(out[1], c_v)
    utt.assert_allclose(out[2], b_v)

    # Test multiple output
    out_s = theano.scalar.Composite([a_s, b_s, c_s], [a_s + b_s, a_s * c_s])
    outs_op = tensor.Elemwise(out_s)
    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, cuda.GpuElemwise) for node in topo) == 1
    assert sum(isinstance(node.op, tensor.Elemwise) for node in topo) == 0
    out = f(a_v, b_v, c_v)
    utt.assert_allclose(out[0], a_v + b_v)
    utt.assert_allclose(out[1], a_v * c_v)

    # Test non-contiguous input
    c = cuda.shared_constructor(c_v)
    f = theano.function([a, b], outs_op(a[::2], b[::2], c[::2]),
                        mode=mode_with_gpu)
    out = f(a_v, b_v)
    utt.assert_allclose(out[0], a_v[::2] + b_v[::2])
    utt.assert_allclose(out[1], a_v[::2] * c_v[::2])


def test_elemwise_fusion():
    """ Test the the GpuElemwise fusion work correctly"""
    shape = (3, 4)
    a = cuda.shared_constructor(theano._asarray(numpy.random.rand(*shape),
                                                dtype='float32'), 'a')
    b = tensor.fmatrix()
    c = tensor.fmatrix()
    f = pfunc([b, c], [a + b + c], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    for i, node in enumerate(topo):
        print(i, node, file=sys.stdout)
    assert len(topo) == 4
    assert isinstance(topo[2].op.scalar_op, theano.scalar.basic.Composite)
    # let debugmode catch errors
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32'),
      theano._asarray(numpy.random.rand(*shape), dtype='float32'))


import theano.tests.test_ifelse


class TestIfElse(theano.tests.test_ifelse.test_ifelse):
    dtype = "float32"
    mode = mode_with_gpu
    cast_output = staticmethod(basic_ops.as_cuda_ndarray_variable)
    shared = staticmethod(cuda.shared_constructor)

    def get_ifelse(self, n):
        return theano.ifelse.IfElse(n, gpu=True, as_view=True)


def test_incsubtensor_mixed():

    # This catches a bug that occurred when incrementing
    # a float32 tensor by a float64 tensor.
    # The result is defined to be float32, so it is OK
    # to downcast the float64 increment in order to
    # transfer it to the GPU.
    # The bug was that the optimization called GpuFromHost
    # without casting first, causing the optimization to
    # fail.
    X = tensor.fmatrix()
    Y = tensor.dmatrix()
    Z = tensor.inc_subtensor(X[0:1, 0:1], Y)
    f = theano.function([X, Y], Z, mode=mode_with_gpu)
    packed, = f.maker.fgraph.inputs[1].clients
    client, idx = packed
    print(client)
    assert isinstance(client.op, tensor.Elemwise)
    assert isinstance(client.op.scalar_op, theano.scalar.Cast)
    packed, = client.outputs[0].clients
    client, idx = packed
    assert isinstance(client.op, cuda.GpuFromHost)


def test_erfinvgpu():
    """ Test that local_gpu_elemwise_0 replaces Erfinv with ErfinvGPU """
    x = tensor.fmatrix()
    f = theano.function([x], tensor.Elemwise(erfinv)(x), mode=mode_with_gpu)
    f2 = theano.function([x], tensor.Elemwise(erfinv)(x),
                         mode=mode_without_gpu)
    assert isinstance(f.maker.fgraph.toposort()[1].op, cuda.GpuElemwise)
    assert isinstance(f.maker.fgraph.toposort()[1].op.scalar_op,
                      cuda.elemwise.ErfinvGPU)
    xv = numpy.random.rand(7, 8).astype('float32')
    if imported_scipy_special:
        assert numpy.allclose(f(xv), f2(xv))


def test_local_gpu_solve():

    if not cula.cula_available:
        raise SkipTest('Optional dependency CULA not available')

    numpy.random.seed(1)

    def cmp(a_shp, b_shp):
        a0 = numpy.random.uniform(-0.4, 0.4,
                                  a_shp).astype('float32')
        a = cuda.shared_constructor(a0, 'a')

        b0 = numpy.random.uniform(-0.4, 0.4,
                                  b_shp).astype('float32')
        b = cuda.shared_constructor(b0, 'b')

        f = pfunc([], tensor.slinalg.solve(a, b), mode=mode_with_gpu)

        assert isinstance(f.maker.fgraph.toposort()[1].inputs[0].owner.op,
                          cuda.cula.GpuSolve)

        assert cuda.opt.local_gpu_solve.transform(
            tensor.slinalg.solve(a, b).owner)
        out = f()
        assert numpy.allclose(numpy.dot(a0, out), b0)

    cmp((6, 6), (6, 1))
    cmp((5, 5), (5, 1))


def test_local_gpu_dot_to_dot22dot():
    def cmp(a_shp, b_shp):
        a0 = numpy.random.rand(*a_shp).astype('float32')
        a = cuda.shared_constructor(a0, 'a')
        b0 = numpy.random.rand(*b_shp).astype('float32')
        b = cuda.shared_constructor(b0, 'b')

        f = pfunc([], tensor.dot(a, b), mode=mode_with_gpu)
        assert cuda.opt.local_gpu_dot_to_dot22.transform(
            tensor.dot(a, b).owner)
        out = f()

        assert numpy.allclose(numpy.dot(a0, b0), out)

        # Try with a matrix equal to a0, but with strides in both dims
        a.set_value(a0)
        a.set_value(
            a.get_value(borrow=True,
                        return_internal_type=True)[::-1],
            borrow=True)
        f()

    cmp((4,), (4, 5))
    cmp((3, 4), (4,))


def test_blocksparse_gpu_gemv_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o, mode=mode_with_gpu)

    assert sum(1 for n in f.maker.fgraph.apply_nodes
               if isinstance(n.op, GpuSparseBlockGemv)) == 1


def test_blocksparse_gpu_outer_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], [o, tensor.grad(o.sum(),
                                                               wrt=W)],
                        mode=mode_with_gpu)

    assert sum(1 for n in f.maker.fgraph.apply_nodes
               if isinstance(n.op, GpuSparseBlockOuter)) == 1


class test_diag(theano.tensor.tests.test_nlinalg.test_diag):
    mode = mode_with_gpu
    shared = staticmethod(cuda.shared_constructor)
    floatX = 'float32'
    type = CudaNdarrayType

    def __init__(self, name):
        super(theano.tensor.tests.test_nlinalg.test_diag,
              self).__init__(name)


class Test_GpuReshape(test_opt.Test_Reshape):
    def setUp(self):
        self.mode = mode_with_gpu
        self.op = basic_ops.GpuReshape


def test_local_abstractconv_gemm():
    """ We test it here as this is the optimization only that we test.
    This test gh-4036"""
    image = tensor.ftensor4()
    W = tensor.ftensor4()
    conv = tensor.nnet.conv2d(image,
                         W,
                         input_shape=(1, 32, 32, 32),
                         filter_shape=(32, 32, 3, 3),
                         border_mode='half')
    f = theano.function([image, W], [conv], mode=mode_with_gpu)
    f(numpy.random.rand(1, 32, 32, 32).astype('float32'),
      numpy.random.rand(32, 32, 3, 3).astype('float32'))

if __name__ == '__main__':
    test_gpualloc()
    test_opt_gpujoin_onlyajoin()
    test_opt_gpujoin_joinvectors_elemwise_then_minusone()
    test_opt_gpujoin_joinvectors_negativeaxes()
