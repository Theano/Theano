from __future__ import absolute_import, print_function, division
import os

import numpy

import theano
from theano.compat import PY3
from theano import config
from theano.compile import DeepCopyOp
from theano.misc.pkl_utils import CompatUnpickler

from .config import test_ctx_name
from .test_basic_ops import rand_gpuarray
from ..type import GpuArrayType, gpuarray_shared_constructor

import pygpu


def test_deep_copy():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')

    f = theano.function([g], g)

    assert isinstance(f.maker.fgraph.toposort()[0].op, DeepCopyOp)

    res = f(a)

    assert GpuArrayType.values_eq(res, a)


def test_values_eq_approx():
    a = rand_gpuarray(20, dtype='float32')
    assert GpuArrayType.values_eq_approx(a, a)
    b = a.copy()
    b[0] = numpy.asarray(b[0]) + 1.
    assert not GpuArrayType.values_eq_approx(a, b)
    b = a.copy()
    b[0] = -numpy.asarray(b[0])
    assert not GpuArrayType.values_eq_approx(a, b)


def test_specify_shape():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')
    f = theano.function([g], theano.tensor.specify_shape(g, [20]))
    f(a)


def test_filter_float():
    theano.compile.shared_constructor(gpuarray_shared_constructor)
    try:
        s = theano.shared(numpy.array(0.0, dtype='float32'),
                          target=test_ctx_name)
        theano.function([], updates=[(s, 0.0)])
    finally:
        del theano.compile.sharedvalue.shared.constructors[-1]


def test_unpickle_gpuarray_as_numpy_ndarray_flag0():
    """ Test when pygpu isn't there for unpickle are in test_pickle.py"""
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
            assert numpy.asarray(mat)[0] == -42.0
    finally:
        config.experimental.unpickle_gpu_on_cpu = oldflag
