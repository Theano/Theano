import unittest
from itertools import izip
from copy import copy, deepcopy

import numpy
import theano
import theano.tensor as T
from theano.tensor import TensorType
from theano.tensor.basic import alloc
from theano.tensor.tests.test_basic import (
    rand, safe_make_node, T_reshape, T_Join_and_Split
    )
from theano.tests.unittest_tools import SkipTest
from numpy.testing.noseclasses import KnownFailureTest

import theano.sandbox.gpuarray

if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")

# If you are writing a new test file, don't copy this code, but rather
# import stuff from this file (like mode_with_gpu) to reuse it.
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available and not theano.sandbox.gpuarray.pygpu_activated:
    if not cuda_ndarray.use.device_number:
        #We should not enable all the use like the flag device=gpu,
        #as many tests don't work in that setup.
        cuda_ndarray.use('gpu',
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False)
    theano.sandbox.gpuarray.init_dev('cuda')

if not theano.sandbox.gpuarray.pygpu_activated:
    raise SkipTest("pygpu disabled")

from theano.sandbox.gpuarray.type import (GpuArrayType,
                                          gpuarray_shared_constructor)
from theano.sandbox.gpuarray.basic_ops import (
    host_from_gpu, gpu_from_host,
    gpu_alloc, GpuAlloc,
    gpu_from_cuda,
    cuda_from_gpu, HostFromGpu,
    GpuFromHost, GpuReshape,
    gpu_join, GpuJoin, GpuSplit, GpuEye, gpu_contiguous)
from theano.sandbox.gpuarray.subtensor import GpuSubtensor

from theano.tests import unittest_tools as utt
utt.seed_rng()
rng = numpy.random.RandomState(seed=utt.fetch_seed())

from pygpu import gpuarray

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')


def may_fail(msg, EClass):
    """Mark a test that requires very specific conditions to work to
       mask a specific exception class."""
    def test_decorator(f):
        def wrapper():
            try:
                f()
            except Exception, e:
                if isinstance(e, EClass):
                    raise KnownFailureTest(msg, e)
                raise
        wrapper.__name__ = f.__name__
        return wrapper
    return test_decorator


def inplace_func(inputs, outputs, mode=None, allow_input_downcast=False,
                 on_unused_input='raise', name=None):
    if mode is None:
        mode = mode_with_gpu
    return theano.function(inputs, outputs, mode=mode,
                           allow_input_downcast=allow_input_downcast,
                           accept_inplace=True,
                           on_unused_input=on_unused_input, name=name)


def fake_shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    from theano.tensor.sharedvar import tensor_constructor, scalar_constructor
    for c in (gpuarray_shared_constructor, tensor_constructor,
              scalar_constructor):
        try:
            return c(value, name=name, strict=strict,
                     allow_downcast=allow_downcast, **kwargs)
        except TypeError:
            continue


def rand_gpuarray(*shape, **kwargs):
    r = rng.rand(*shape) * 2 - 1
    dtype = kwargs.pop('dtype', theano.config.floatX)
    cls = kwargs.pop('cls', None)
    if len(kwargs) != 0:
        raise TypeError('Unexpected argument %s', kwargs.keys()[0])
    return gpuarray.array(r, dtype=dtype, cls=cls)


def makeTester(name, op, gpu_op, cases, checks=None, mode_gpu=mode_with_gpu,
               mode_nogpu=mode_without_gpu, skip=False, eps=1e-10):
    if checks is None:
        checks = {}

    _op = op
    _gpu_op = gpu_op
    _cases = cases
    _skip = skip
    _checks = checks

    class Checker(unittest.TestCase, utt.TestOptimizationMixin):
        op = staticmethod(_op)
        gpu_op = staticmethod(_gpu_op)
        cases = _cases
        skip = _skip
        checks = _checks

        def setUp(self):
            eval(self.__class__.__module__ + '.' + self.__class__.__name__)

        def test_all(self):
            if skip:
                raise SkipTest(skip)

            for testname, inputs in cases.items():
                self.run_case(testname, inputs)

        def run_case(self, testname, inputs):
            inputs_ref = [theano.shared(inp) for inp in inputs]
            inputs_tst = [theano.shared(inp) for inp in inputs]

            try:
                node_ref = safe_make_node(self.op, *inputs_ref)
                node_tst = safe_make_node(self.op, *inputs_tst)
            except Exception, exc:
                err_msg = ("Test %s::%s: Error occured while making "
                           "a node with inputs %s") % (self.gpu_op, testname,
                                                       inputs)
                exc.args += (err_msg,)
                raise

            try:
                f_ref = inplace_func([], node_ref.outputs, mode=mode_nogpu)
                f_tst = inplace_func([], node_tst.outputs, mode=mode_gpu)
            except Exception, exc:
                err_msg = ("Test %s::%s: Error occured while trying to "
                           "make a Function") % (self.gpu_op, testname)
                exc.args += (err_msg,)
                raise

            self.assertFunctionContains1(f_tst, self.gpu_op)

            ref_e = None
            try:
                expecteds = f_ref()
            except Exception, exc:
                ref_e = exc

            try:
                variables = f_tst()
            except Exception, exc:
                if ref_e is None:
                    err_msg = ("Test %s::%s: exception when calling the "
                               "Function") % (self.gpu_op, testname)
                    exc.args += (err_msg,)
                    raise
                else:
                    # if we raised an exception of the same type we're good.
                    if isinstance(exc, type(ref_e)):
                        return
                    else:
                        err_msg = ("Test %s::%s: exception raised during test "
                                   "call was not the same as the reference "
                                   "call (got: %s, expected %s)") % \
                                   (self.gpu_op, testname, type(exc),
                                    type(ref_e))
                        exc.args += (err_msg,)
                        raise

            for i, (variable, expected) in \
                    enumerate(izip(variables, expecteds)):
                if variable.dtype != expected.dtype or \
                        variable.shape != expected.shape or \
                        not TensorType.values_eq_approx(variable,
                                                        expected):
                    self.fail(("Test %s::%s: Output %s gave the wrong "
                               "value. With inputs %s, expected %s "
                               "(dtype %s), got %s (dtype %s).") % (
                            self.op, testname, i, inputs, expected,
                            expected.dtype, variable, variable.dtype))

            for description, check in self.checks.items():
                if not check(inputs, variables):
                    self.fail(("Test %s::%s: Failed check: %s "
                               "(inputs were %s, ouputs were %s)") %
                              (self.op, testname, description,
                               inputs, variables))

    Checker.__name__ = name
    return Checker


def test_transfer_cpu_gpu():
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')

    av = numpy.asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av)

    f = theano.function([a], gpu_from_host(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert numpy.all(fv == av)


def test_transfer_strided():
    # This is just to ensure that it works in theano
    # libgpuarray has a much more comprehensive suit of tests to
    # ensure correctness
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')

    av = numpy.asarray(rng.rand(5, 8), dtype='float32')
    gv = gpuarray.array(av)

    av = av[:, ::2]
    gv = gv[:, ::2]

    f = theano.function([a], gpu_from_host(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert numpy.all(fv == av)


@may_fail("Op fails if both contexts are not the same and it's rare "
          "that the tests will be run this way", ValueError)
def test_transfer_cuda_gpu():
    import theano.sandbox.cuda as cuda_ndarray
    if cuda_ndarray.cuda_available is False:
        raise SkipTest("Can't test interaction with cuda if cuda not present")
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
    c = cuda_ndarray.CudaNdarrayType((False, False))('c')

    av = theano._asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av)
    cv = cuda_ndarray.CudaNdarray(av)
    gvs = gv[:, ::-2]
    cvs = cv[:, ::-2]

    f = theano.function([c], gpu_from_cuda(c))
    fv = f(cv)
    assert GpuArrayType.values_eq_approx(fv, gv)

    fvs = f(cvs)
    assert GpuArrayType.values_eq_approx(fvs, gvs)

    f = theano.function([g], cuda_from_gpu(g))
    fv = f(gv)
    assert cuda_ndarray.CudaNdarrayType.values_eq_approx(fv, cv)

    fvs = f(gvs)
    assert cuda_ndarray.CudaNdarrayType.values_eq_approx(fvs, cvs)


def gpu_alloc_expected(x, *shp):
    g = gpuarray.empty(shp, dtype=x.dtype)
    g[:] = x
    return g

GpuAllocTester = makeTester(
    name="GpuAllocTester",
    op=alloc,
    gpu_op=gpu_alloc,
    cases=dict(
        correct01=(rand(), numpy.int32(7)),
# just gives a DeepCopyOp with possibly wrong results on the CPU
#        correct01_bcast=(rand(1), numpy.int32(7)),
        correct02=(rand(), numpy.int32(4), numpy.int32(7)),
        correct12=(rand(7), numpy.int32(4), numpy.int32(7)),
        correct13=(rand(7), numpy.int32(2), numpy.int32(4),
                   numpy.int32(7)),
        correct23=(rand(4, 7), numpy.int32(2), numpy.int32(4),
                   numpy.int32(7)),
        bad_shape12=(rand(7), numpy.int32(7), numpy.int32(5)),
        )
)


class TestAlloc(theano.tensor.tests.test_basic.TestAlloc):
    dtype = "float32"
    mode = mode_with_gpu
    shared = staticmethod(gpuarray_shared_constructor)
    allocs = [GpuAlloc, GpuAlloc, T.Alloc]


def test_shape():
    x = GpuArrayType(dtype='float32', broadcastable=[False, False, False])()
    v = gpuarray.zeros((3, 4, 5), dtype='float32')
    f = theano.function([x], x.shape)
    topo = f.maker.fgraph.toposort()
    assert numpy.all(f(v) == (3, 4, 5))
    if theano.config.mode != 'FAST_COMPILE':
        assert len(topo) == 4
        assert isinstance(topo[0].op, T.opt.Shape_i)
        assert isinstance(topo[1].op, T.opt.Shape_i)
        assert isinstance(topo[2].op, T.opt.Shape_i)
        assert isinstance(topo[3].op, T.opt.MakeVector)
    mode = mode_with_gpu.excluding("local_shape_to_shape_i")
    f = theano.function([x], x.shape, mode=mode)
    topo = f.maker.fgraph.toposort()
    assert numpy.all(f(v) == (3, 4, 5))
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Shape)


def test_gpu_contiguous():
    a = T.fmatrix('a')
    i = T.iscalar('i')
    a_val = numpy.asarray(numpy.random.rand(4, 5), dtype='float32')
    f = theano.function([a, i], gpu_contiguous(a[::i]),
                        mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(node.op, GpuSubtensor) for node in topo])
    assert f(a_val, 1).flags.c_contiguous
    assert f(a_val, 2).flags.c_contiguous
    assert f(a_val, 2).flags.c_contiguous


class G_reshape(T_reshape):
    def shortDescription(self):
        return None

    def __init__(self, name):
        T_reshape.__init__(self, name,
                           shared=gpuarray_shared_constructor,
                           op=GpuReshape,
                           mode=mode_with_gpu,
                           # avoid errors with limited devices
#                             dtype='float32',
                           ignore_topo=(HostFromGpu, GpuFromHost,
                                        theano.compile.DeepCopyOp,
                                        theano.sandbox.gpuarray.elemwise.GpuElemwise,
                                        theano.tensor.opt.Shape_i,
                                        theano.tensor.opt.MakeVector))
        assert self.op == GpuReshape


class G_Join_and_Split(T_Join_and_Split):
    def setUp(self):
        super(G_Join_and_Split, self).setUp()
        self.mode = mode_with_gpu.excluding('constant_folding')
        self.join_op = GpuJoin
        self.split_op = GpuSplit
        # Use join instead of MakeVector since there is no MakeVector on GPU
        self.make_vector_op = GpuJoin
        # this is to avoid errors with limited devices
        self.floatX = 'float32'
        self.hide_error = theano.config.mode not in ['DebugMode', 'DEBUG_MODE']
        self.shared = gpuarray_shared_constructor

    def test_gpusplit_opt(self):
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        m = self.shared(rng.rand(4, 6).astype(self.floatX))
        o = T.Split(2)(m, 0, [2, 2])
        f = theano.function([], o, mode=self.mode)
        assert any([isinstance(node.op, self.split_op)
                    for node in f.maker.fgraph.toposort()])
        o1, o2 = f()
        assert numpy.allclose(o1, m.get_value(borrow=True)[:2])
        assert numpy.allclose(o2, m.get_value(borrow=True)[2:])


def test_gpujoin_gpualloc():
    a = T.fmatrix('a')
    a_val = numpy.asarray(numpy.random.rand(4, 5), dtype='float32')
    b = T.fmatrix('b')
    b_val = numpy.asarray(numpy.random.rand(3, 5), dtype='float32')

    f = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)) + 4,
                        mode=mode_without_gpu)
    f_gpu = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)),
                            mode=mode_with_gpu)
    f_gpu2 = theano.function([a, b], T.join(0, T.zeros_like(a),
                                            T.ones_like(b)) + 4,
                             mode=mode_with_gpu)
    assert sum([node.op == T.alloc for node in f.maker.fgraph.toposort()]) == 2
    assert sum([node.op == T.join for node in f.maker.fgraph.toposort()]) == 1
    assert sum([isinstance(node.op, GpuAlloc)
                for node in f_gpu.maker.fgraph.toposort()]) == 2
    assert sum([node.op == gpu_join
                for node in f_gpu.maker.fgraph.toposort()]) == 1
    assert sum([isinstance(node.op, GpuAlloc)
                for node in f_gpu2.maker.fgraph.toposort()]) == 2
    assert sum([node.op == gpu_join
                for node in f_gpu2.maker.fgraph.toposort()]) == 1
    assert numpy.allclose(f(a_val, b_val), f_gpu2(a_val, b_val))


def test_gpueye():
    def check(dtype, N, M_=None):
        # Theano does not accept None as a tensor.
        # So we must use a real value.
        M = M_
        # Currently DebugMode does not support None as inputs even if this is
        # allowed.
        if M is None:
            M = N
        N_symb = T.iscalar()
        M_symb = T.iscalar()
        k_symb = numpy.asarray(0)
        out = T.eye(N_symb, M_symb, k_symb, dtype=dtype)
        f = theano.function([N_symb, M_symb],
                            out,
                            mode=mode_with_gpu)
        result = numpy.asarray(f(N, M))
        assert numpy.allclose(result, numpy.eye(N, M_, dtype=dtype))
        assert result.dtype == numpy.dtype(dtype)
        assert any([isinstance(node.op, GpuEye)
                    for node in f.maker.fgraph.toposort()])

    for dtype in ['float32', 'int32']:
        yield check, dtype, 3
        # M != N, k = 0
        yield check, dtype, 3, 5
        yield check, dtype, 5, 3


def test_hostfromgpu_shape_i():
    """
    Test that the shape is lifted over hostfromgpu
    """

    m = mode_with_gpu.including('local_dot_to_dot22',
                                'local_dot22_to_dot22scalar',
                                'specialize')
    a = T.fmatrix('a')
    ca = theano.sandbox.gpuarray.type.GpuArrayType('float32', (False, False))()
    av = numpy.asarray(numpy.random.rand(5, 4), dtype='float32')
    cv = gpuarray.asarray(numpy.random.rand(5, 4),
                          dtype='float32')

    gpu_from_host = theano.sandbox.gpuarray.basic_ops.gpu_from_host
    host_from_gpu = theano.sandbox.gpuarray.basic_ops.host_from_gpu
    f = theano.function([a], gpu_from_host(a), mode=m)
    assert gpu_from_host in [x.op
                             for x in f.maker.fgraph.toposort()]
    f = theano.function([a], gpu_from_host(a).shape, mode=m)
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[0].op, T.opt.Shape_i)
    assert isinstance(topo[1].op, T.opt.Shape_i)
    assert isinstance(topo[2].op, T.opt.MakeVector)
    assert tuple(f(av)) == (5, 4)

    f = theano.function([ca], host_from_gpu(ca), mode=m)
    assert host_from_gpu in [x.op
                             for x in f.maker.fgraph.toposort()]
    f = theano.function([ca], host_from_gpu(ca).shape, mode=m)
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[0].op, theano.compile.Shape_i)
    assert isinstance(topo[1].op, theano.compile.Shape_i)
    assert isinstance(topo[2].op, theano.tensor.opt.MakeVector)
    assert tuple(f(cv)) == (5, 4)
