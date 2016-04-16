from __future__ import absolute_import, print_function, division
import numpy

import theano
from theano import tensor
from theano.tests.breakpoint import PdbBreakpoint
from theano.tests import unittest_tools as utt, test_ifelse
from theano.tensor.tests import test_basic

import theano.sandbox.gpuarray
from .. import basic_ops
from ..type import GpuArrayType, gpuarray_shared_constructor, get_context
from ..basic_ops import (
    GpuAlloc, GpuAllocEmpty, GpuReshape, GpuFromHost, host_from_gpu)
from ..blas import GpuGemm
from ..elemwise import GpuCAReduceCuda, GpuCAReduceCPY, GpuElemwise
from ..subtensor import GpuSubtensor

from .config import mode_with_gpu, test_ctx_name


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


def test_flatten():
    m = theano.tensor.fmatrix()
    f = theano.function([m], m.flatten(), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val.flatten())
    assert res.shape == val.flatten().shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11).astype("float32")
    res = f(val)
    utt.assert_allclose(res, val)
    assert res.shape == val.shape
    assert GpuReshape in [type(node.op)
                          for node in f.maker.fgraph.toposort()]

    m = theano.tensor.tensor3()
    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
    val = numpy.random.rand(10, 11, 12).astype("float32")
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
        val = numpy.random.rand(10, 11).astype("float32")
        res = f(val)
        utt.assert_allclose(res, getattr(val, method)(axis=0))
        assert res.shape == (11,)
        topo = f.maker.fgraph.toposort()
        ops = [type(node.op) for node in topo]

        if kind == 'opencl' and method in ["max", "min"]:
            assert not(GpuCAReduceCuda in ops or GpuCAReduceCPY in ops)
        else:
            assert GpuCAReduceCuda in ops or GpuCAReduceCPY in ops


def test_local_gpualloc_memset_0():
    i = theano.tensor.iscalar()
    z = numpy.zeros((1,), dtype='float32')
    o = numpy.ones((1,), dtype='float32')
    ones = numpy.ones((2,), dtype='float32')

    # Test with 0 from CPU op.
    a = tensor.alloc(z, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert isinstance(topo[0].op, GpuAlloc) and topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 0).all()

    # Test with 0
    a = GpuAlloc(test_ctx_name)(z, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc) and topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 0).all()

    # Test with 1
    a = GpuAlloc(test_ctx_name)(o, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(6)) == 1).all()

    # Test with 1, 1
    a = GpuAlloc(test_ctx_name)(ones, i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, GpuAlloc)
    assert not topo[0].op.memset_0
    assert (numpy.asarray(f(2)) == 1).all()


def test_local_gpualloc_empty():
    i = theano.tensor.iscalar()
    ii = theano.tensor.iscalar()

    # Test with vector
    a = tensor.AllocEmpty('float32')(i)
    f = theano.function([i], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert isinstance(topo[0].op, GpuAllocEmpty)
    # This return not initilized data, so we can only check the shape
    assert f(3).shape == (3,)

    # Test with matrix
    a = tensor.AllocEmpty('float32')(i, ii)
    f = theano.function([i, ii], a, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert isinstance(topo[0].op, GpuAllocEmpty)
    # This return not initilized data, so we can only check the shape
    assert f(3, 4).shape == (3, 4)


def test_rebroadcast():
    d = numpy.random.rand(10, 10).astype('float32')
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
    assert isinstance(topo[-2].op, GpuElemwise)
    assert topo[-1].op == host_from_gpu


def test_local_gpu_elemwise_careduce():
    x = theano.tensor.matrix()
    o = (x * x).sum()
    f = theano.function([x], o, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert topo[1].op.pre_scalar_op == theano.scalar.sqr
    data = numpy.random.rand(3, 4).astype(theano.config.floatX)
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
    x_val = numpy.random.random((2, 3)).astype(theano.config.floatX)
    y_val = numpy.random.random((3, 4)).astype(theano.config.floatX)
    a_val = 0.5
    utt.assert_allclose(f_cpu(x_val, y_val, a_val), f_gpu(x_val, y_val, a_val))


def test_local_gpu_subtensor():
    # Test shared forced on CPU.
    t = tensor._shared(numpy.zeros(20, "float32"))
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
    t = tensor._shared(numpy.zeros(20, "float32"))
    f = theano.function([], t[3:4] + 1, mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert any([type(node.op) is tensor.Subtensor for node in topo])
    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])
    assert any([isinstance(node.op, GpuElemwise) for node in topo])


def test_local_gpu_elemwise():
    """
    Test local_gpu_elemwise when there is a dtype upcastable to float32
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
    c = gpuarray_shared_constructor(numpy.asarray(c_v, dtype='float32'))
    f = theano.function([a, b], outs_op(a[::2], b[::2], c[::2]),
                        mode=mode_with_gpu)
    out = f(a_v, b_v)
    utt.assert_allclose(out[0], a_v[::2] + b_v[::2])
    utt.assert_allclose(out[1], a_v[::2] * c_v[::2])


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
