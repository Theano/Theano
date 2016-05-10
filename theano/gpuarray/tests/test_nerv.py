from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest

import numpy

from theano import function
from theano.tests import unittest_tools as utt
from theano.tensor import vector, matrix, dot

from .config import mode_with_gpu
from ..nerv import Gemm16, nerv


def test_gemm16_swap():
    if nerv is None:
        raise SkipTest("nervanagpu not available")
    v = vector(dtype='float16')
    m = matrix(dtype='float16')
    m2 = matrix(dtype='float16')
    m32 = matrix(dtype='float32')

    # test that we don't try to replace anything but matrix x matrix in float16
    f = function([v, m], dot(v, m), mode=mode_with_gpu)
    assert len([node for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, Gemm16)]) == 0
    f = function([m32, m], dot(m32, m), mode=mode_with_gpu)
    assert len([node for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, Gemm16)]) == 0

    f = function([m, m2], dot(m, m2), mode=mode_with_gpu)
    assert len([node for node in f.maker.fgraph.apply_nodes
                if isinstance(node.op, Gemm16)]) == 1


def test_gemm16_value():
    if nerv is None:
        raise SkipTest("nervanagpu not available")
    m = matrix(dtype='float16')
    m2 = matrix(dtype='float16')

    f = function([m, m2], dot(m, m2), mode=mode_with_gpu)

    v1 = numpy.random.random((3, 4)).astype('float16')
    v2 = numpy.random.random((4, 2)).astype('float16')

    of = f(v1, v2)
    on = numpy.dot(v1, v2)

    utt.assert_allclose(of, on)
