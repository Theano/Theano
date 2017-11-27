from __future__ import absolute_import, print_function, division
import unittest
from theano.compat import izip

from six import iteritems

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import TensorType
from theano.tensor.basic import alloc

# Don't import test classes otherwise they get tested as part of the file
from theano.tensor.tests import test_basic
from theano.tensor.tests.test_basic import rand, safe_make_node
from theano.tests import unittest_tools as utt

from ..type import (GpuArrayType, get_context,
                    gpuarray_shared_constructor)
from ..basic_ops import (
    host_from_gpu, HostFromGpu, GpuFromHost, GpuReshape, GpuToGpu,
    GpuAlloc, GpuAllocEmpty, GpuContiguous,
    gpu_join, GpuJoin, GpuSplit, GpuEye, gpu_contiguous)
from ..elemwise import GpuDimShuffle, GpuElemwise
from ..subtensor import GpuSubtensor

from .config import mode_with_gpu, mode_without_gpu, test_ctx_name

from pygpu import gpuarray

utt.seed_rng()
rng = np.random.RandomState(seed=utt.fetch_seed())


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
        raise TypeError('Unexpected argument %s', list(kwargs.keys())[0])
    return gpuarray.array(r, dtype=dtype, cls=cls,
                          context=get_context(test_ctx_name))


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
                from nose.plugins.skip import SkipTest
                raise SkipTest(skip)

            for testname, inputs in iteritems(cases):
                for _ in range(len(inputs)):
                    if type(inputs[_]) is float:
                        inputs[_] = np.asarray(inputs[_],
                                               dtype=theano.config.floatX)
                self.run_case(testname, inputs)

        def run_case(self, testname, inputs):
            inputs_ref = [theano.shared(inp) for inp in inputs]
            inputs_tst = [theano.shared(inp) for inp in inputs]

            try:
                node_ref = safe_make_node(self.op, *inputs_ref)
                node_tst = safe_make_node(self.op, *inputs_tst)
            except Exception as exc:
                err_msg = ("Test %s::%s: Error occurred while making "
                           "a node with inputs %s") % (self.gpu_op, testname,
                                                       inputs)
                exc.args += (err_msg,)
                raise

            try:
                f_ref = inplace_func([], node_ref.outputs, mode=mode_nogpu)
                f_tst = inplace_func([], node_tst.outputs, mode=mode_gpu)
            except Exception as exc:
                err_msg = ("Test %s::%s: Error occurred while trying to "
                           "make a Function") % (self.gpu_op, testname)
                exc.args += (err_msg,)
                raise

            self.assertFunctionContains1(f_tst, self.gpu_op)

            ref_e = None
            try:
                expecteds = f_ref()
            except Exception as exc:
                ref_e = exc

            try:
                variables = f_tst()
            except Exception as exc:
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
                                   "call (got: %s, expected %s)" %
                                   (self.gpu_op, testname, type(exc),
                                    type(ref_e)))
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
                               "(dtype %s), got %s (dtype %s)." %
                               (self.op, testname, i, inputs, expected,
                                expected.dtype, variable, variable.dtype)))

            for description, check in iteritems(self.checks):
                if not check(inputs, variables):
                    self.fail(("Test %s::%s: Failed check: %s "
                               "(inputs were %s, ouputs were %s)") %
                              (self.op, testname, description,
                               inputs, variables))

    Checker.__name__ = name
    if hasattr(Checker, '__qualname__'):
        Checker.__qualname__ = name
    return Checker


def test_transfer_cpu_gpu():
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')

    av = np.asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av, context=get_context(test_ctx_name))

    f = theano.function([a], GpuFromHost(test_ctx_name)(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert np.all(fv == av)


def test_transfer_gpu_gpu():
    g = GpuArrayType(dtype='float32', broadcastable=(False, False),
                     context_name=test_ctx_name)()

    av = np.asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av, context=get_context(test_ctx_name))
    mode = mode_with_gpu.excluding('cut_gpua_host_transfers', 'local_cut_gpua_host_gpua')
    f = theano.function([g], GpuToGpu(test_ctx_name)(g), mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuToGpu)
    fv = f(gv)
    assert GpuArrayType.values_eq(fv, gv)


def test_transfer_strided():
    # This is just to ensure that it works in theano
    # libgpuarray has a much more comprehensive suit of tests to
    # ensure correctness
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')

    av = np.asarray(rng.rand(5, 8), dtype='float32')
    gv = gpuarray.array(av, context=get_context(test_ctx_name))

    av = av[:, ::2]
    gv = gv[:, ::2]

    f = theano.function([a], GpuFromHost(test_ctx_name)(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert np.all(fv == av)


def gpu_alloc_expected(x, *shp):
    g = gpuarray.empty(shp, dtype=x.dtype, context=get_context(test_ctx_name))
    g[:] = x
    return g


GpuAllocTester = makeTester(
    name="GpuAllocTester",
    # The +1 is there to allow the lift to the GPU.
    op=lambda *args: alloc(*args) + 1,
    gpu_op=GpuAlloc(test_ctx_name),
    cases=dict(
        correct01=(rand(), np.int32(7)),
        # just gives a DeepCopyOp with possibly wrong results on the CPU
        # correct01_bcast=(rand(1), np.int32(7)),
        correct02=(rand(), np.int32(4), np.int32(7)),
        correct12=(rand(7), np.int32(4), np.int32(7)),
        correct13=(rand(7), np.int32(2), np.int32(4),
                   np.int32(7)),
        correct23=(rand(4, 7), np.int32(2), np.int32(4),
                   np.int32(7)),
        bad_shape12=(rand(7), np.int32(7), np.int32(5)),
        )
)


class TestAlloc(test_basic.TestAlloc):
    dtype = "float32"
    mode = mode_with_gpu
    shared = staticmethod(gpuarray_shared_constructor)
    allocs = [GpuAlloc(test_ctx_name), GpuAlloc(test_ctx_name), T.Alloc()]


def test_alloc_empty():
    for dt in ['float32', 'int8']:
        f = theano.function([], GpuAllocEmpty(dt, context_name=test_ctx_name)(2, 3))
        assert len(f.maker.fgraph.apply_nodes) == 1
        out = f()
        assert out.shape == (2, 3)
        assert out.dtype == dt

    f = theano.function([], [GpuAllocEmpty('uint64', test_ctx_name)(3, 2),
                             GpuAllocEmpty('uint64', test_ctx_name)(3, 2)])
    out = f()
    assert out[0].shape == (3, 2)
    assert out[0].dtype == 'uint64'
    assert out[1].shape == (3, 2)
    assert out[1].dtype == 'uint64'
    assert len([node for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, GpuAllocEmpty)]) == 1


def test_shape():
    x = GpuArrayType(dtype='float32', broadcastable=[False, False, False])()
    v = gpuarray.zeros((3, 4, 5), dtype='float32', context=get_context(test_ctx_name))
    f = theano.function([x], x.shape)
    topo = f.maker.fgraph.toposort()
    assert np.all(f(v) == (3, 4, 5))
    if theano.config.mode != 'FAST_COMPILE':
        assert len(topo) == 4
        assert isinstance(topo[0].op, T.opt.Shape_i)
        assert isinstance(topo[1].op, T.opt.Shape_i)
        assert isinstance(topo[2].op, T.opt.Shape_i)
        assert isinstance(topo[3].op, T.opt.MakeVector)
    mode = mode_with_gpu.excluding("local_shape_to_shape_i")
    f = theano.function([x], x.shape, mode=mode)
    topo = f.maker.fgraph.toposort()
    assert np.all(f(v) == (3, 4, 5))
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Shape)


def test_gpu_contiguous():
    a = T.fmatrix('a')
    i = T.iscalar('i')
    a_val = np.asarray(np.random.rand(4, 5), dtype='float32')
    # The reshape is needed otherwise we make the subtensor on the CPU
    # to transfer less data.
    f = theano.function([a, i], gpu_contiguous(a.reshape((5, 4))[::i]),
                        mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([isinstance(node.op, GpuSubtensor) for node in topo])
    assert any([isinstance(node.op, GpuContiguous) for node in topo])
    assert f(a_val, 1).flags.c_contiguous
    assert f(a_val, 2).flags.c_contiguous
    assert f(a_val, 2).flags.c_contiguous


class G_reshape(test_basic.T_reshape):
    def shortDescription(self):
        return None

    def __init__(self, name):
        test_basic.T_reshape.__init__(
            self, name,
            shared=gpuarray_shared_constructor,
            op=GpuReshape,
            mode=mode_with_gpu,
            ignore_topo=(HostFromGpu, GpuFromHost,
                         theano.compile.DeepCopyOp,
                         GpuDimShuffle, GpuElemwise,
                         theano.tensor.opt.Shape_i,
                         theano.tensor.opt.MakeVector))
        assert self.op == GpuReshape


class G_comparison(test_basic.test_comparison):
    def setUp(self):
        utt.seed_rng()
        self.mode = mode_with_gpu
        self.shared = gpuarray_shared_constructor
        self.dtypes = ['float64', 'float32']


class G_Join_and_Split(test_basic.T_Join_and_Split):
    def setUp(self):
        super(G_Join_and_Split, self).setUp()
        self.mode = mode_with_gpu.excluding('constant_folding')
        self.join_op = GpuJoin()
        self.split_op_class = GpuSplit
        # Use join instead of MakeVector since there is no MakeVector on GPU
        self.make_vector_op = GpuJoin()
        # this is to avoid errors with limited devices
        self.floatX = 'float32'
        self.hide_error = theano.config.mode not in ['DebugMode', 'DEBUG_MODE']

        def shared(x, **kwargs):
            return gpuarray_shared_constructor(x, target=test_ctx_name,
                                               **kwargs)
        self.shared = shared

    def test_gpusplit_opt(self):
        # Test that we move the node to the GPU
        # Also test float16 computation at the same time.
        rng = np.random.RandomState(seed=utt.fetch_seed())
        m = self.shared(rng.rand(4, 6).astype('float16'))
        o = T.Split(2)(m, 0, [2, 2])
        assert o[0].dtype == 'float16'
        f = theano.function([], o, mode=self.mode)
        assert any([isinstance(node.op, self.split_op_class)
                    for node in f.maker.fgraph.toposort()])
        o1, o2 = f()
        assert np.allclose(o1, m.get_value(borrow=True)[:2])
        assert np.allclose(o2, m.get_value(borrow=True)[2:])


def test_gpujoin_gpualloc():
    a = T.fmatrix('a')
    a_val = np.asarray(np.random.rand(4, 5), dtype='float32')
    b = T.fmatrix('b')
    b_val = np.asarray(np.random.rand(3, 5), dtype='float32')

    f = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)) + 4,
                        mode=mode_without_gpu)
    f_gpu = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)),
                            mode=mode_with_gpu)
    f_gpu2 = theano.function([a, b], T.join(0, T.zeros_like(a),
                                            T.ones_like(b)) + 4,
                             mode=mode_with_gpu)
    assert sum([node.op == T.alloc for node in f.maker.fgraph.toposort()]) == 2
    assert sum([node.op == T.join_ for node in f.maker.fgraph.toposort()]) == 1
    assert sum([isinstance(node.op, GpuAlloc)
                for node in f_gpu.maker.fgraph.toposort()]) == 2
    assert sum([node.op == gpu_join
                for node in f_gpu.maker.fgraph.toposort()]) == 1
    assert sum([isinstance(node.op, GpuAlloc)
                for node in f_gpu2.maker.fgraph.toposort()]) == 2
    assert sum([node.op == gpu_join
                for node in f_gpu2.maker.fgraph.toposort()]) == 1
    assert np.allclose(f(a_val, b_val), f_gpu2(a_val, b_val))


def test_gpueye():
    def check(dtype, N, M_=None, k=0):
        # Theano does not accept None as a tensor.
        # So we must use a real value.
        M = M_
        # Currently DebugMode does not support None as inputs even if this is
        # allowed.
        if M is None:
            M = N
        N_symb = T.iscalar()
        M_symb = T.iscalar()
        k_symb = T.iscalar()
        out = T.eye(N_symb, M_symb, k_symb, dtype=dtype) + np.array(1).astype(dtype)
        f = theano.function([N_symb, M_symb, k_symb],
                            out,
                            mode=mode_with_gpu)

        result = np.asarray(f(N, M, k)) - np.array(1).astype(dtype)
        assert np.allclose(result, np.eye(N, M_, k, dtype=dtype))
        assert result.dtype == np.dtype(dtype)
        assert any([isinstance(node.op, GpuEye)
                    for node in f.maker.fgraph.toposort()])

    for dtype in ['float32', 'int32', 'float16']:
        yield check, dtype, 3
        # M != N, k = 0
        yield check, dtype, 3, 5
        yield check, dtype, 5, 3
        # N == M, k != 0
        yield check, dtype, 3, 3, 1
        yield check, dtype, 3, 3, -1
        # N < M, k != 0
        yield check, dtype, 3, 5, 1
        yield check, dtype, 3, 5, -1
        # N > M, k != 0
        yield check, dtype, 5, 3, 1
        yield check, dtype, 5, 3, -1
        # k > M, -k > N, k > M, k > N
        yield check, dtype, 5, 3, 3
        yield check, dtype, 3, 5, 3
        yield check, dtype, 5, 3, -3
        yield check, dtype, 3, 5, -3
        yield check, dtype, 5, 3, 6
        yield check, dtype, 3, 5, -6


def test_hostfromgpu_shape_i():
    # Test that the shape is lifted over hostfromgpu

    m = mode_with_gpu.including('local_dot_to_dot22',
                                'local_dot22_to_dot22scalar',
                                'specialize')
    a = T.fmatrix('a')
    ca = theano.gpuarray.type.GpuArrayType('float32', (False, False))()
    av = np.asarray(np.random.rand(5, 4), dtype='float32')
    cv = gpuarray.asarray(np.random.rand(5, 4),
                          dtype='float32',
                          context=get_context(test_ctx_name))

    f = theano.function([a], GpuFromHost(test_ctx_name)(a), mode=m)
    assert any(isinstance(x.op, GpuFromHost)
               for x in f.maker.fgraph.toposort())
    f = theano.function([a], GpuFromHost(test_ctx_name)(a).shape, mode=m)
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


def test_Gpujoin_inplace():
    # Test Gpujoin to work inplace.
    #
    # This function tests the case when several elements are passed to the
    # Gpujoin function but all except one of them are empty. In this case
    # Gpujoin should work inplace and the output should be the view of the
    # non-empty element.
    s = T.lscalar()
    data = np.array([3, 4, 5], dtype=theano.config.floatX)
    x = gpuarray_shared_constructor(data, borrow=True)
    z = T.zeros((s,))

    join = GpuJoin(view=0)
    c = join(0, x, z)

    f = theano.function([s], theano.Out(c, borrow=True))
    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
        assert x.get_value(borrow=True, return_internal_type=True) is f(0)
    assert np.allclose(f(0), [3, 4, 5])
