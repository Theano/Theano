from __future__ import absolute_import, print_function, division
import numpy as np
import unittest

import theano
from theano import tensor
from theano.compile import DeepCopyOp
from theano.tensor.tests import test_subtensor, test_basic
from theano.tests import unittest_tools as utt

from ..basic_ops import HostFromGpu, GpuFromHost, GpuContiguous
from ..elemwise import GpuDimShuffle
from ..subtensor import (GpuIncSubtensor, GpuSubtensor,
                         GpuAdvancedSubtensor1,
                         GpuAdvancedSubtensor,
                         GpuAdvancedBooleanSubtensor,
                         GpuAdvancedIncSubtensor,
                         GpuAdvancedIncSubtensor1,
                         GpuAdvancedIncSubtensor1_dev20,
                         GpuExtractDiag,
                         GpuAllocDiag)
from ..type import gpuarray_shared_constructor

from .config import mode_with_gpu, test_ctx_name


class G_subtensor(test_subtensor.T_subtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        def shared(x, **kwargs):
            return gpuarray_shared_constructor(x, target=test_ctx_name,
                                               **kwargs)

        test_subtensor.T_subtensor.__init__(
            self, name,
            shared=shared,
            sub=GpuSubtensor,
            inc_sub=GpuIncSubtensor,
            adv_sub1=GpuAdvancedSubtensor1,
            adv_incsub1=GpuAdvancedIncSubtensor1,
            adv_sub=GpuAdvancedSubtensor,
            adv_bool_sub=GpuAdvancedBooleanSubtensor,
            dimshuffle=GpuDimShuffle,
            mode=mode_with_gpu,
            # avoid errors with limited devices
            dtype='float32',
            ignore_topo=(HostFromGpu, GpuFromHost,
                         DeepCopyOp, GpuContiguous))
        # GPU opt can't run in fast_compile only.
        self.fast_compile = False
        assert self.sub == GpuSubtensor


class G_subtensorF16(test_subtensor.T_subtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        def shared(x, **kwargs):
            return gpuarray_shared_constructor(x, target=test_ctx_name,
                                               **kwargs)

        test_subtensor.T_subtensor.__init__(
            self, name,
            shared=shared,
            sub=GpuSubtensor,
            inc_sub=GpuIncSubtensor,
            adv_sub1=GpuAdvancedSubtensor1,
            adv_incsub1=GpuAdvancedIncSubtensor1,
            adv_sub=GpuAdvancedSubtensor,
            adv_bool_sub=GpuAdvancedBooleanSubtensor,
            dimshuffle=GpuDimShuffle,
            mode=mode_with_gpu,
            # avoid errors with limited devices
            dtype='float16',  # use floatX?
            ignore_topo=(HostFromGpu, GpuFromHost,
                         DeepCopyOp, GpuContiguous))
        # GPU opt can't run in fast_compile only.
        self.fast_compile = False
        assert self.sub == GpuSubtensor


def test_advinc_subtensor1():
    # Test the second case in the opt local_gpu_advanced_incsubtensor1
    for shp in [(3, 3), (3, 3, 3)]:
        shared = gpuarray_shared_constructor
        xval = np.arange(np.prod(shp), dtype='float32').reshape(shp) + 1
        yval = np.empty((2,) + shp[1:], dtype='float32')
        yval[:] = 10
        x = shared(xval, name='x')
        y = tensor.tensor(dtype='float32',
                          broadcastable=(False,) * len(shp),
                          name='y')
        expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
        f = theano.function([y], expr, mode=mode_with_gpu)
        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
                    for node in f.maker.fgraph.toposort()]) == 1
        rval = f(yval)
        rep = xval.copy()
        np.add.at(rep, [0, 2], yval)
        assert np.allclose(rval, rep)


def test_advinc_subtensor1_dtype():
    # Test the mixed dtype case
    shp = (3, 4)
    for dtype1, dtype2 in [('float32', 'int8'), ('float32', 'float64'),
                           ('uint64', 'int8'), ('int64', 'uint8'),
                           ('float16', 'int8'), ('float16', 'float64'),
                           ('float16', 'float16')]:
        shared = gpuarray_shared_constructor
        xval = np.arange(np.prod(shp), dtype=dtype1).reshape(shp) + 1
        yval = np.empty((2,) + shp[1:], dtype=dtype2)
        yval[:] = 10
        x = shared(xval, name='x')
        y = tensor.tensor(dtype=yval.dtype,
                          broadcastable=(False,) * len(yval.shape),
                          name='y')
        expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
        f = theano.function([y], expr, mode=mode_with_gpu)
        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1_dev20)
                    for node in f.maker.fgraph.toposort()]) == 1
        rval = f(yval)
        rep = xval.copy()
        np.add.at(rep, [[0, 2]], yval)
        assert np.allclose(rval, rep)


@theano.change_flags(deterministic='more')
def test_deterministic_flag():
    shp = (3, 4)
    for dtype1, dtype2 in [('float32', 'int8')]:
        shared = gpuarray_shared_constructor
        xval = np.arange(np.prod(shp), dtype=dtype1).reshape(shp) + 1
        yval = np.empty((2,) + shp[1:], dtype=dtype2)
        yval[:] = 10
        x = shared(xval, name='x')
        y = tensor.tensor(dtype=yval.dtype,
                          broadcastable=(False,) * len(yval.shape),
                          name='y')
        expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
        f = theano.function([y], expr, mode=mode_with_gpu)
        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
                    for node in f.maker.fgraph.toposort()]) == 1
        rval = f(yval)
        rep = xval.copy()
        np.add.at(rep, [[0, 2]], yval)
        assert np.allclose(rval, rep)


def test_advinc_subtensor1_vector_scalar():
    # Test the case where x is a vector and y a scalar
    shp = (3,)
    for dtype1, dtype2 in [('float32', 'int8'), ('float32', 'float64'),
                           ('float16', 'int8'), ('float16', 'float64'),
                           ('float16', 'float16'), ('int8', 'int8'),
                           ('int16', 'int16')]:
        shared = gpuarray_shared_constructor
        xval = np.arange(np.prod(shp), dtype=dtype1).reshape(shp) + 1
        yval = np.asarray(10, dtype=dtype2)
        x = shared(xval, name='x')
        y = tensor.tensor(dtype=yval.dtype,
                          broadcastable=(False,) * len(yval.shape),
                          name='y')
        expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
        f = theano.function([y], expr, mode=mode_with_gpu)

        assert sum([isinstance(node.op, (GpuAdvancedIncSubtensor1_dev20,
                                         GpuAdvancedIncSubtensor1))
                    for node in f.maker.fgraph.toposort()]) == 1
        rval = f(yval)
        rep = xval.copy()
        rep[[0, 2]] += yval
        assert np.allclose(rval, rep)


def test_incsub_f16():
    shp = (3, 3)
    shared = gpuarray_shared_constructor
    xval = np.arange(np.prod(shp), dtype='float16').reshape(shp) + 1
    yval = np.empty((2,) + shp[1:], dtype='float16')
    yval[:] = 2
    x = shared(xval, name='x')
    y = tensor.tensor(dtype='float16',
                      broadcastable=(False,) * len(shp),
                      name='y')
    expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
    f = theano.function([y], expr, mode=mode_with_gpu)
    assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
                for node in f.maker.fgraph.toposort()]) == 1
    rval = f(yval)
    rep = xval.copy()
    np.add.at(rep, [[0, 2]], yval)
    assert np.allclose(rval, rep)

    expr = tensor.inc_subtensor(x[1:], y)
    f = theano.function([y], expr, mode=mode_with_gpu)
    assert sum([isinstance(node.op, GpuIncSubtensor)
                for node in f.maker.fgraph.toposort()]) == 1
    rval = f(yval)
    rep = xval.copy()
    rep[1:] += yval
    assert np.allclose(rval, rep)


def test_incsub_offset():
    # Test for https://github.com/Theano/Theano/issues/5670

    # Build a GPU variable which value will have an offset (x1)
    x = gpuarray_shared_constructor(np.zeros(5, dtype=theano.config.floatX))
    x1 = x[1:]
    # Use inc_subtensor on it
    y = tensor.vector()
    z = tensor.inc_subtensor(x1[2:], y)
    # Use updates so that inc_subtensor can happen inplace
    f = theano.function([y], z, updates={x: z}, mode=mode_with_gpu)
    utt.assert_allclose(
        f([1, 2]),
        np.array([0, 0, 1, 2], dtype=theano.config.floatX))


class G_advancedsubtensor(test_subtensor.TestAdvancedSubtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        test_subtensor.TestAdvancedSubtensor.__init__(
            self, name,
            shared=gpuarray_shared_constructor,
            sub=GpuAdvancedSubtensor,
            inc_sub=GpuAdvancedIncSubtensor,
            mode=mode_with_gpu,
            # avoid errors with limited devices
            dtype='float32',  # floatX?
            ignore_topo=(HostFromGpu, GpuFromHost,
                         DeepCopyOp))
        # GPU opt can't run in fast_compile only.
        self.fast_compile = False
        assert self.sub == GpuAdvancedSubtensor


class G_advancedsubtensorF16(test_subtensor.TestAdvancedSubtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        test_subtensor.TestAdvancedSubtensor.__init__(
            self, name,
            shared=gpuarray_shared_constructor,
            sub=GpuAdvancedSubtensor,
            mode=mode_with_gpu,
            # avoid errors with limited devices
            dtype='float16',  # floatX?
            ignore_topo=(HostFromGpu, GpuFromHost,
                         DeepCopyOp))
        # GPU opt can't run in fast_compile only.
        self.fast_compile = False
        assert self.sub == GpuAdvancedSubtensor


def test_adv_subtensor():
    # Test the advancedsubtensor on gpu.
    shp = (2, 3, 4)
    shared = gpuarray_shared_constructor
    xval = np.arange(np.prod(shp), dtype=theano.config.floatX).reshape(shp)
    idx1, idx2 = tensor.ivectors('idx1', 'idx2')
    idxs = [idx1, None, slice(0, 2, 1), idx2, None]
    x = shared(xval, name='x')
    expr = x[idxs]
    f = theano.function([idx1, idx2], expr, mode=mode_with_gpu)
    assert sum([isinstance(node.op, GpuAdvancedSubtensor)
               for node in f.maker.fgraph.toposort()]) == 1
    idx1_val = [0, 1]
    idx2_val = [0, 1]
    rval = f(idx1_val, idx2_val)
    rep = xval[idx1_val, None, slice(0, 2, 1), idx2_val, None]
    assert np.allclose(rval, rep)


class test_gpuextractdiag(unittest.TestCase):
    def test_extractdiag_opt(self):
        x = tensor.matrix()
        fn = theano.function([x], tensor.ExtractDiag()(x), mode=mode_with_gpu)
        assert any([isinstance(node.op, GpuExtractDiag)
                    for node in fn.maker.fgraph.toposort()])

    def test_matrix(self):
        x = tensor.matrix()
        np_x = np.arange(77).reshape(7, 11).astype(theano.config.floatX)
        fn = theano.function([x], GpuExtractDiag()(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np_x.diagonal())
        fn = theano.function([x], GpuExtractDiag(2)(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np_x.diagonal(2))
        fn = theano.function([x], GpuExtractDiag(-3)(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np_x.diagonal(-3))

    def test_tensor(self):
        x = tensor.tensor4()
        np_x = np.arange(30107).reshape(7, 11, 17, 23).astype(theano.config.floatX)
        for offset, axis1, axis2 in [
                (1, 0, 1), (-1, 0, 1), (0, 1, 0), (-2, 1, 0),
                (-3, 1, 0), (-2, 2, 0), (3, 3, 0), (-1, 3, 2),
                (2, 2, 3), (-1, 2, 1), (1, 3, 1), (-1, 1, 3)]:
            assert np.allclose(
                GpuExtractDiag(offset, axis1, axis2)(x).eval({x: np_x}),
                np_x.diagonal(offset, axis1, axis2))

    def test_tensor_float16(self):
        x = tensor.tensor4()
        np_x = np.arange(30107).reshape(7, 11, 17, 23).astype('float16')
        for offset, axis1, axis2 in [
                (1, 0, 1), (-1, 0, 1), (0, 1, 0), (-2, 1, 0),
                (-3, 1, 0), (-2, 2, 0), (3, 3, 0), (-1, 3, 2),
                (2, 2, 3), (-1, 2, 1), (1, 3, 1), (-1, 1, 3)]:
            assert np.allclose(
                GpuExtractDiag(offset, axis1, axis2)(x).eval({x: np_x}),
                np_x.diagonal(offset, axis1, axis2))


class TestGpuAllocDiag(test_basic.TestAllocDiag):
    def __init__(self, name):
        test_basic.TestAllocDiag.__init__(
            self, name,
            alloc_diag=GpuAllocDiag,
            mode=mode_with_gpu
        )


class test_gpuallocdiag(unittest.TestCase):
    def test_allocdiag_opt(self):
        x = tensor.vector()
        fn = theano.function([x], tensor.AllocDiag()(x), mode=mode_with_gpu)
        assert any([isinstance(node.op, GpuAllocDiag)
                    for node in fn.maker.fgraph.toposort()])

    def test_matrix(self):
        x = tensor.vector()
        np_x = np.arange(7).astype(theano.config.floatX)
        fn = theano.function([x], GpuAllocDiag()(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np.diag(np_x))
        fn = theano.function([x], GpuAllocDiag(2)(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np.diag(np_x, 2))
        fn = theano.function([x], GpuAllocDiag(-3)(x), mode=mode_with_gpu)
        assert np.allclose(fn(np_x), np.diag(np_x, -3))

    def test_grad(self):
        x = tensor.vector()
        np_x = np.random.randn(7).astype(theano.config.floatX)

        # offset = 0 case:
        mtx_x = GpuAllocDiag()(x)
        sum_mtx_x = tensor.sum(mtx_x)
        grad_x = tensor.grad(sum_mtx_x, x)
        grad_mtx_x = tensor.grad(sum_mtx_x, mtx_x)

        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)

        computed_grad_x = fn_grad_x(np_x)
        computed_grad_mtx_x = fn_grad_mtx_x(np_x)
        true_grad_x = np.diagonal(computed_grad_mtx_x, 0)
        assert np.allclose(computed_grad_x, true_grad_x)

        # offset > 0 case:
        mtx_x = GpuAllocDiag(2)(x)
        sum_mtx_x = tensor.sum(mtx_x)
        grad_x = tensor.grad(sum_mtx_x, x)
        grad_mtx_x = tensor.grad(sum_mtx_x, mtx_x)

        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)

        computed_grad_x = fn_grad_x(np_x)
        computed_grad_mtx_x = fn_grad_mtx_x(np_x)
        true_grad_x = np.diagonal(computed_grad_mtx_x, 2)
        assert np.allclose(computed_grad_x, true_grad_x)

        # offset < 0 case:
        mtx_x = GpuAllocDiag(-3)(x)
        sum_mtx_x = tensor.sum(mtx_x)
        grad_x = tensor.grad(sum_mtx_x, x)
        grad_mtx_x = tensor.grad(sum_mtx_x, mtx_x)

        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)

        computed_grad_x = fn_grad_x(np_x)
        computed_grad_mtx_x = fn_grad_mtx_x(np_x)
        true_grad_x = np.diagonal(computed_grad_mtx_x, -3)
        assert np.allclose(computed_grad_x, true_grad_x)

        # assert
