from __future__ import absolute_import, print_function, division
import numpy as np
from nose.tools import assert_raises

import theano
from theano import tensor
import theano.tensor.slinalg as slinalg
from theano.tests.breakpoint import PdbBreakpoint
from theano.tests import unittest_tools as utt, test_ifelse
from theano.tensor.tests import test_basic

import theano.gpuarray
from .. import basic_ops
from ..type import GpuArrayType, gpuarray_shared_constructor, get_context
from ..basic_ops import (
    GpuAlloc, GpuAllocEmpty, GpuReshape, GpuFromHost, host_from_gpu)
from ..blas import GpuGemm
from ..elemwise import GpuCAReduceCuda, GpuCAReduceCPY, GpuElemwise
from ..subtensor import GpuSubtensor
from ..linalg import GpuCusolverSolve, cusolver_available

from .config import mode_with_gpu, mode_without_gpu, test_ctx_name, SkipTest


def test_local_assert():
    x = theano.tensor.fmatrix()
    a = theano.tensor.opt.assert_op(x, theano.tensor.eq(x, 0).any())
    f = theano.function([x], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    a_op = [n for n in topo if isinstance(n.op, theano.tensor.opt.Assert)]
    assert len(a_op) == 1
    assert isinstance(a_op[0].inputs[0].type, GpuArrayType)


def test_local_remove_all_assert():
    x = theano.tensor.fmatrix()
    a = theano.tensor.opt.assert_op(x, theano.tensor.eq(x, 0).any())

    # By default `unsafe` should not be there
    f = theano.function([x], a, mode=mode_with_gpu.excluding('unsafe'))
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


def test_local_gpu_contiguous():
    a = tensor.fmatrix()
    o = tensor.extra_ops.cpu_contiguous(a)
    f = theano.function([a], o, mode=mode_with_gpu)
    assert 1 == len([node for node in f.maker.fgraph.toposort()
                     if isinstance(node.op, basic_ops.GpuContiguous)])
    f([[2.]])


def test_flatten():
    m = theano.tensor.fmatrix()
    f = theano.function([m], m.flatten(), mode=mode_with_gpu)
    val = np.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]
    val = np.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = np.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val)
    assert res.shape == val.shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    m = theano.tensor.tensor3()
    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = np.random.rand(10, 11, 12).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.reshape(10, -1))
    assert res.shape == val.reshape(10, -1).shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]


def test_reduce():
    kind = get_context(test_ctx_name).kind

    for method, param in [('sum', dict(acc_dtype='float32')),
                          ('prod', dict(acc_dtype='float32')),
                          ('max', {}), ('min', {})]:
        m = theano.tensor.fmatrix()
        f = theano.function([m], getattr(m, method)(axis=0,
                                                    **param),
                            mode=mode_with_gpu)
        val = np.random.rand(10, 11).astype("float32")
        res = f(val)
        utt.assert_allclose(res, getattr(val, method)(axis=0))
        assert res.shape == (11,)
        topo = f.maker.fgraph.toposort()
        ops = [type(node.op) for node in topo]

        if kind == b'opencl' and method in ["max", "min"]:
            assert not(GpuCAReduceCuda in ops or GpuCAReduceCPY in ops)
        else:
            assert GpuCAReduceCuda in ops or GpuCAReduceCPY in ops


def test_local_gpualloc_memset_0():
    i = theano.tensor.iscalar()
    z = np.zeros((1,), dtype='float32')
    o = np.ones((1,), dtype='float32')
    ones = np.ones((2,), dtype='float32')

    # Test with 0 from CPU op.
    # Should not be transfered as the only client is the output
    a = tensor.alloc(z, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, theano.tensor.Alloc)
    assert (np.asarray(f(6)) == 0).all()

    # Test with 0 from CPU op.
    # Should be transfered as it is used by another op.
    a = tensor.alloc(z, i)
    f = theano.function([i], a.cumsum(), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[0].op, GpuAlloc)
    assert (np.asarray(f(6)) == 0).all()

    # Test with 0
    a = GpuAlloc(test_ctx_name)(z, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc) and topo[0].op.memset_0
    assert (np.asarray(f(6)) == 0).all()

    # Test with 1
    a = GpuAlloc(test_ctx_name)(o, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (np.asarray(f(6)) == 1).all()

    # Test with 1, 1
    a = GpuAlloc(test_ctx_name)(ones, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (np.asarray(f(2)) == 1).all()


def test_local_gpualloc_empty():
    i = theano.tensor.iscalar()
    ii = theano.tensor.iscalar()

    # Test with vector
    # Should not be moved as the only client is the output
    a = tensor.AllocEmpty('float32')(i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, theano.tensor.AllocEmpty)
    # This return not initilized data, so we can only check the shape
    assert f(3).shape == (3,)

    # Test with vector
    # Should be moved
    a = tensor.AllocEmpty('float32')(i)
    f = theano.function([i], a.cumsum(), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[0].op, GpuAllocEmpty)
    # This return not initilized data, so we can only check the shape
    assert f(3).shape == (3,)

    # Test with matrix
    a = tensor.AllocEmpty('float32')(i, ii)
    f = theano.function([i, ii], a.cumsum(axis=0), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[0].op, GpuAllocEmpty)
    # This return not initilized data, so we can only check the shape
    assert f(3, 4).shape == (3, 4)


def test_rebroadcast():
    d = np.random.rand(10, 10).astype('float32')
    v = theano.tensor.fmatrix()
    up = tensor.unbroadcast(v.sum().dimshuffle('x', 'x'), 0, 1)
    f = theano.function([v], [up], mode=mode_with_gpu)

    f(d)

    topo = f.maker.fgraph.toposort()
    rebrs = [node for node in topo if isinstance(node.op, tensor.Rebroadcast)]
    assert len(rebrs) == 1
    rebr = rebrs[0]

    assert isinstance(rebr.inputs[0].type, GpuArrayType)
    assert isinstance(rebr.outputs[0].type, GpuArrayType)


class TestSpecifyShape(test_basic.TestSpecifyShape):
    mode = mode_with_gpu
    input_type = GpuArrayType


class test_gpu_ifelse(test_ifelse.test_ifelse):
    mode = mode_with_gpu

    @staticmethod
    def cast_output(v):
        return basic_ops.as_gpuarray_variable(v, test_ctx_name)
    shared = staticmethod(gpuarray_shared_constructor)

    def get_ifelse(self, n):
        return theano.ifelse.IfElse(n, gpu=True, as_view=True)

    def test_lifter_with_inputs_of_graph(self):
        x = tensor.vector()
        cond = tensor.iscalar()
        f = theano.function([x, cond],
                            theano.ifelse.ifelse(cond, x.mean(), x.sum()),
                            mode=mode_with_gpu)
        assert f(np.float32([1, 2, 3]), 0) == 6

        x = tensor.vector()
        cond = tensor.scalar()
        f = theano.function([x, cond],
                            theano.ifelse.ifelse(cond, x.mean(), x.sum()),
                            mode=mode_with_gpu)
        assert f(np.float32([1, 2, 3]), 0) == 6


def test_print_op():
    """ Test that print ops don't block gpu optimization"""
    b = tensor.fmatrix()
    f = theano.function([b], theano.printing.Print()(b) * 2,
                        mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[0].op, GpuFromHost)
    assert isinstance(topo[1].op, theano.printing.Print)
    assert isinstance(topo[2].op, GpuElemwise)
    assert topo[3].op == host_from_gpu
    f(np.random.random((5, 5)).astype('float32'))


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
    assert isinstance(topo[-2].op, GpuElemwise)
    assert topo[-1].op == host_from_gpu


def test_local_gpu_elemwise_careduce():
    x = theano.tensor.matrix()
    o = (x * x).sum()
    f = theano.function([x], o, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert topo[1].op.pre_scalar_op == theano.scalar.sqr
    data = np.random.rand(3, 4).astype(theano.config.floatX)
    utt.assert_allclose(f(data), (data * data).sum())

    o = (x * x).sum(axis=1)
    f = theano.function([x], o, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert topo[1].op.pre_scalar_op == theano.scalar.sqr
    utt.assert_allclose(f(data), (data * data).sum(axis=1))


def test_local_lift_dot22scalar():
    x = tensor.matrix()
    y = tensor.matrix()
    a = tensor.scalar()
    o = tensor.blas.Dot22Scalar()(x, y, a)
    f_cpu = theano.function([x, y, a], o)
    f_gpu = theano.function([x, y, a], o, mode=mode_with_gpu)
    assert not any(isinstance(n.op, tensor.blas.Dot22Scalar)
                   for n in f_gpu.maker.fgraph.apply_nodes)
    assert any(isinstance(n.op, GpuGemm)
               for n in f_gpu.maker.fgraph.apply_nodes)
    x_val = np.random.random((2, 3)).astype(theano.config.floatX)
    y_val = np.random.random((3, 4)).astype(theano.config.floatX)
    a_val = 0.5
    utt.assert_allclose(f_cpu(x_val, y_val, a_val), f_gpu(x_val, y_val, a_val))


def test_local_gpu_subtensor():
    # Test shared forced on CPU.
    t = tensor._shared(np.zeros(20, "float32"))
    f = theano.function([], t[3:4], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])

    # Test graph input.
    t = tensor.fmatrix()
    f = theano.function([t], t[3:4], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])

    # Test multiple use of the input
    # We want the subtensor to be on the GPU to prevent multiple transfer.
    t = tensor.fmatrix()
    f = theano.function([t], [t[3:4], t + 1], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert not any([type(node.op) is tensor.Subtensor for node in topo])
    assert any([isinstance(node.op, GpuSubtensor) for node in topo])

    # Test multiple use of the input + input as output
    # We want the subtensor to be on the GPU to prevent multiple transfer.
    t = tensor.fmatrix()
    f = theano.function([t], [t[3:4], t + 1, t], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert not any([type(node.op) is tensor.Subtensor for node in topo])
    assert any([isinstance(node.op, GpuSubtensor) for node in topo])

    # Test shared forced on CPU end we do computation on the output of
    # the subtensor.
    t = tensor._shared(np.zeros(20, "float32"))
    f = theano.function([], t[3:4] + 1, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])
    # Our optimizer isn't smart enough to move to the GPU Elemwise.
    # If it where just a little bit smarter, it could wrongly move it to the GPU.
    # If it where super smart, it would know it should not move it to the GPU.
    assert any([isinstance(node.op, tensor.Elemwise) for node in topo])


def test_local_gpu_elemwise():
    """
    Test local_gpu_elemwise when there is a dtype upcastable to float32
    """
    a = tensor.bmatrix()
    b = tensor.fmatrix()
    c = tensor.fmatrix()

    a_v = (np.random.rand(4, 5) * 10).astype("int8")
    b_v = (np.random.rand(4, 5) * 10).astype("float32")
    c_v = (np.random.rand(4, 5) * 10).astype("float32")

    # Due to optimization order, this composite is created when all
    # the op are on the gpu.
    f = theano.function([a, b, c], a + b + c, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
    assert sum(type(node.op) == tensor.Elemwise for node in topo) == 0
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
    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
    assert sum(type(node.op) == tensor.Elemwise for node in topo) == 0
    utt.assert_allclose(f(a_v, b_v, c_v), a_v + b_v + c_v)

    return  # Not yet implemeted
    # Test multiple output
    a_s = theano.scalar.float32()
    a = tensor.fmatrix()
    from theano.scalar.basic import identity
    out_s = theano.scalar.Composite([a_s, b_s, c_s],
                                    [identity(a_s), identity(c_s), identity(b_s)])
    outs_op = tensor.Elemwise(out_s)
    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
    assert sum(type(node.op) == tensor.Elemwise for node in topo) == 0
    out = f(a_v, b_v, c_v)
    utt.assert_allclose(out[0], a_v)
    utt.assert_allclose(out[1], c_v)
    utt.assert_allclose(out[2], b_v)

    # Test multiple output
    out_s = theano.scalar.Composite([a_s, b_s, c_s], [a_s + b_s, a_s * b_s])
    outs_op = tensor.Elemwise(out_s)
    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
    assert sum(type(node.op) == tensor.Elemwise for node in topo) == 0
    out = f(a_v, b_v, c_v)
    utt.assert_allclose(out[0], a_v + b_v)
    utt.assert_allclose(out[1], a_v * c_v)

    # Test non-contiguous input
    c = gpuarray_shared_constructor(np.asarray(c_v, dtype='float32'))
    f = theano.function([a, b], outs_op(a[::2], b[::2], c[::2]),
                        mode=mode_with_gpu)
    out = f(a_v, b_v)
    utt.assert_allclose(out[0], a_v[::2] + b_v[::2])
    utt.assert_allclose(out[1], a_v[::2] * c_v[::2])


def test_many_arg_elemwise():
    # this test checks whether the + and * elemwise ops can handle
    # extremely large numbers of arguments on gpu
    rng = np.random.RandomState([1, 2, 3])

    for num_args in [75]:
        for op_to_test in [theano.tensor.add, theano.tensor.mul]:
            for nb_dim in [2, 3, 4, 5, 7]:
                shapes = [rng.randint(1, 5) for i in range(nb_dim)]
                args = [np.cast['float32'](rng.randn(*shapes))
                        for arg in range(0, num_args)]

                symb_args = [theano.tensor.TensorType('float32',
                                                      (False,) * nb_dim)()
                             for arg in range(0, num_args)]

                outputs = []
                for mode in [mode_with_gpu, mode_without_gpu]:
                    # test the optijmization local_gpu_elemwise_0
                    f = theano.function(
                        symb_args, op_to_test(*symb_args),
                        mode=mode.excluding("local_gpu_elemwise_1"))
                    outputs.append(f(*args))
                    # assert that the test was done on the gpu.
                    if mode is mode_with_gpu:
                        assert any([isinstance(node.op, GpuElemwise)
                                    for node in f.maker.fgraph.apply_nodes])

                    # test the optijmization local_gpu_elemwise_1
                    f = theano.function(
                        symb_args,
                        GpuFromHost(test_ctx_name)(op_to_test(*symb_args)),
                        mode=mode.excluding("local_gpu_elemwise_0"))
                    out = f(*args)
                    # assert that the test was done on the gpu.
                    if mode is mode_with_gpu:
                        assert any([isinstance(node.op, GpuElemwise)
                                    for node in f.maker.fgraph.apply_nodes])
                    utt.assert_allclose(out, outputs[-1])

                results_gpu, results_cpu = outputs

                utt.assert_allclose(results_gpu, results_cpu)


def test_local_lift_abstractconv_gpu_shape():
    prev = theano.config.on_opt_error
    try:
        theano.config.on_opt_error = 'raise'
        s = tensor.ivector()
        a = tensor.ftensor4()
        b = tensor.ftensor4()
        c = tensor.nnet.abstract_conv.AbstractConv2d_gradWeights()(a, b, s)
        theano.function([s, a, b], c, mode=mode_with_gpu)
    finally:
        theano.config.on_opt_error = prev


def test_local_assert_no_cpu_op():
    rng = np.random.RandomState(utt.fetch_seed())
    m = rng.uniform(-1, 1, (10, 10)).astype("float32")
    ms = gpuarray_shared_constructor(m, name="m_shared")
    out = theano.tensor.tanh(ms).dot(ms.T)

    mode_local_assert = mode_with_gpu.including("assert_no_cpu_op")
    mode_local_assert = mode_local_assert.excluding("local_gpua_elemwise")

    old = theano.config.assert_no_cpu_op
    old2 = theano.config.on_opt_error
    # If the flag is raise
    try:
        theano.config.assert_no_cpu_op = 'raise'
        theano.config.on_opt_error = 'ignore'

        assert_raises(AssertionError, theano.function,
                      [], out, mode=mode_local_assert)
    finally:
        theano.config.assert_no_cpu_op = old
        theano.config.on_opt_error = old2

    # If the flag is ignore
    try:
        theano.config.assert_no_cpu_op = 'ignore'
        theano.function([], out, mode=mode_local_assert)
    finally:
        theano.config.assert_no_cpu_op = old


def test_no_complex():
    width_var = tensor.cscalar()
    freq_var = tensor.fscalar()
    signal_var = tensor.fscalar()
    stft_out = tensor.exp(width_var * freq_var) * signal_var
    theano.function([width_var, freq_var, signal_var], stft_out,
                    mode=mode_with_gpu)


def test_local_lift_solve():
    if not cusolver_available:
        raise SkipTest('No cuSolver')
    A = tensor.fmatrix()
    b = tensor.fmatrix()
    o = slinalg.solve(A, b)
    f_cpu = theano.function([A, b], o)
    f_gpu = theano.function([A, b], o, mode=mode_with_gpu)
    assert not any(isinstance(n.op, slinalg.Solve)
                   for n in f_gpu.maker.fgraph.apply_nodes)
    assert any(isinstance(n.op, GpuCusolverSolve)
               for n in f_gpu.maker.fgraph.apply_nodes)
    A_val = np.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
    b_val = np.random.uniform(-0.4, 0.4, (5, 3)).astype("float32")
    utt.assert_allclose(f_cpu(A_val, b_val), f_gpu(A_val, b_val))


def test_local_gpua_advanced_incsubtensor():
    # test a corner case reported at gh-5589
    target = tensor.ftensor4()
    y = target.dimshuffle(1, 0, 2, 3).flatten(ndim=1)
    w = tensor.ones_like(y)
    w = tensor.set_subtensor(w[tensor.eq(y, 1.0).nonzero()], 100)
    w = tensor.set_subtensor(w[tensor.eq(y, -1.0).nonzero()], 0)
    theano.function([target], w)
