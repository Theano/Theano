from __future__ import absolute_import, print_function, division
import os

import nose
import numpy as np

import theano
from theano.compat import PY3
from theano import config
from theano.compile import DeepCopyOp, Rebroadcast, ViewOp
from theano.misc.pkl_utils import CompatUnpickler

# Disabled for now
# from theano.tensor.tests.test_sharedvar import makeSharedTester

from .config import test_ctx_name
from .test_basic_ops import rand_gpuarray
from ..type import GpuArrayType, gpuarray_shared_constructor

import pygpu


def test_deep_copy():
    for dtype in ['float16', 'float32']:
        a = rand_gpuarray(20, dtype=dtype)
        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')

        f = theano.function([g], g)

        assert isinstance(f.maker.fgraph.toposort()[0].op, DeepCopyOp)

        res = f(a)

        assert GpuArrayType.values_eq(res, a)


def test_view():
    for dtype in ['float16', 'float32']:
        a = rand_gpuarray(20, dtype=dtype)
        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')

        m = theano.compile.get_default_mode().excluding("local_view_op")
        f = theano.function([g], ViewOp()(g), mode=m)

        assert isinstance(f.maker.fgraph.toposort()[0].op, ViewOp)

        res = f(a)

        assert GpuArrayType.values_eq(res, a)


def test_rebroadcast():
    for dtype in ['float16', 'float32']:
        a = rand_gpuarray(1, dtype=dtype)
        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')

        f = theano.function([g], Rebroadcast((0, True))(g))

        assert isinstance(f.maker.fgraph.toposort()[0].op, Rebroadcast)

        res = f(a)

        assert GpuArrayType.values_eq(res, a)


def test_values_eq_approx():
    a = rand_gpuarray(20, dtype='float32')
    assert GpuArrayType.values_eq_approx(a, a)
    b = a.copy()
    b[0] = np.asarray(b[0]) + 1.
    assert not GpuArrayType.values_eq_approx(a, b)
    b = a.copy()
    b[0] = -np.asarray(b[0])
    assert not GpuArrayType.values_eq_approx(a, b)


def test_specify_shape():
    for dtype in ['float16', 'float32']:
        a = rand_gpuarray(20, dtype=dtype)
        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')
        f = theano.function([g], theano.tensor.specify_shape(g, [20]))
        f(a)


def test_filter_float():
    theano.compile.shared_constructor(gpuarray_shared_constructor)
    try:
        s = theano.shared(np.array(0.0, dtype='float32'),
                          target=test_ctx_name)
        theano.function([], updates=[(s, 0.0)])
    finally:
        del theano.compile.sharedvalue.shared.constructors[-1]


def test_filter_variable():
    # Test that filter_variable accepts more restrictive broadcast
    gpu_row = GpuArrayType(dtype=theano.config.floatX,
                           broadcastable=(True, False))
    gpu_matrix = GpuArrayType(dtype=theano.config.floatX,
                              broadcastable=(False, False))
    r = gpu_row()
    m = gpu_matrix.filter_variable(r)
    assert m.type == gpu_matrix

    # On CPU as well
    r = theano.tensor.row()
    m = gpu_matrix.filter_variable(r)
    assert m.type == gpu_matrix


def test_gpuarray_shared_scalar():
    # By default, we don't put scalar as shared variable on the GPU
    nose.tools.assert_raises(
        TypeError, gpuarray_shared_constructor, np.asarray(1, dtype='float32'))

    # But we can force that
    gpuarray_shared_constructor(np.asarray(1, dtype='float32'),
                                target=test_ctx_name)


def test_unpickle_gpuarray_as_numpy_ndarray_flag0():
    # Test when pygpu isn't there for unpickle are in test_pickle.py
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = False

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'GpuArray.pkl'

        with open(os.path.join(testfile_dir, fname), 'rb') as fp:
            if PY3:
                u = CompatUnpickler(fp, encoding="latin1")
            else:
                u = CompatUnpickler(fp)
            mat = u.load()
            assert isinstance(mat, pygpu.gpuarray.GpuArray)
            assert np.asarray(mat)[0] == -42.0
    finally:
        config.experimental.unpickle_gpu_on_cpu = oldflag

# These tests are disabled because they expect the impossible
'''
@makeSharedTester(
    shared_constructor_=gpuarray_shared_constructor,
    dtype_=theano.config.floatX,
    get_value_borrow_true_alias_=True,
    shared_borrow_true_alias_=True,
    set_value_borrow_true_alias_=True,
    set_value_inplace_=True,
    set_cast_value_inplace_=False,
    shared_constructor_accept_ndarray_=True,
    internal_type_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
                                         cls=pygpu._array.ndgpuarray),
    test_internal_type_=lambda a: isinstance(a, pygpu.gpuarray.GpuArray),
    theano_fct_=theano.tensor.exp,
    ref_fct_=np.exp,
    cast_value_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
                                      cls=pygpu._array.ndgpuarray))
class test_shared_options(object):
        pass


@makeSharedTester(
    shared_constructor_=gpuarray_shared_constructor,
    dtype_=theano.config.floatX,
    get_value_borrow_true_alias_=False,
    shared_borrow_true_alias_=False,
    set_value_borrow_true_alias_=False,
    set_value_inplace_=True,
    set_cast_value_inplace_=True,
    shared_constructor_accept_ndarray_=True,
    internal_type_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
                                         cls=pygpu._array.ndgpuarray),
    test_internal_type_=lambda a: isinstance(a, pygpu.gpuarray.GpuArray),
    theano_fct_=theano.tensor.exp,
    ref_fct_=np.exp,
    cast_value_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
                                      cls=pygpu._array.ndgpuarray))
class test_shared_options2(object):
    pass
'''


def test_set_value_non_contiguous():
    s = gpuarray_shared_constructor(
        np.asarray([[1., 2.], [1., 2.], [5, 6]]))
    s.set_value(s.get_value(borrow=True, return_internal_type=True)[::2],
                borrow=True)
    assert not s.get_value(borrow=True,
                           return_internal_type=True).flags["C_CONTIGUOUS"]
    # In the past, this failed
    s.set_value([[0, 0], [1, 1]])
