from __future__ import absolute_import, print_function, division

import numpy

import unittest

import theano
from theano import config, function, tensor
from theano.sandbox import multinomial
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import theano.tests.unittest_tools as utt

from .config import mode_with_gpu
from ..multinomial import (GPUAMultinomialFromUniform,
                           GPUAMultinomialWOReplacementFromUniform)


def test_multinomial_0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    for dtype in ['int64', 'float32', 'auto']:

        m = theano.sandbox.multinomial.MultinomialFromUniform(dtype)(p, u)

        # the m*2 allows the multinomial to reuse output
        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode_with_gpu)

        assert any([type(node.op) is GPUAMultinomialFromUniform
                    for node in f.maker.fgraph.toposort()])

        # test that both first and second samples can be drawn
        utt.assert_allclose(f([[1, 0], [0, 1]], [.1, .1]),
                            [[2, 0], [0, 2]])

        # test that both second labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.31, .31])
        utt.assert_allclose(r, [[0, 2], [0, 2]])

        # test that both first labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.21, .21])
        utt.assert_allclose(r, [[0, 2], [2, 0]])

        # change the size to make sure output gets reallocated ok
        # and also make sure that the GPU version doesn't screw up the
        # transposed-ness
        r = f([[.2, .8]], [.25])
        utt.assert_allclose(r, [[0, 2]])


# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
def test_multinomial_large():
    # DEBUG_MODE will test this on GPU
    p = tensor.fmatrix()
    u = tensor.fvector()
    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(p, u)
    f = function([p, u], m * 2, allow_input_downcast=True, mode=mode_with_gpu)
    assert any([type(node.op) is GPUAMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])

    pval = numpy.arange(10000 * 4,
                        dtype='float32').reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    mval = f(pval, uval)

    assert mval.shape == pval.shape
    if config.cast_policy == 'custom':
        assert mval.dtype == pval.dtype
    elif config.cast_policy == 'numpy+floatX':
        assert mval.dtype == config.floatX
    elif config.cast_policy == 'numpy':
        assert mval.dtype == 'float64'
    else:
        raise NotImplementedError(config.cast_policy)
    utt.assert_allclose(mval.sum(axis=1), 2)
    asdf = numpy.asarray([0, 0, 2, 0]) + 0 * pval
    utt.assert_allclose(mval, asdf)  # broadcast over all rows


def test_gpu_opt_dtypes():
    # Test if the returned samples are of the datatype specified
    for dtype in ['uint32', 'float32', 'int64', 'float64']:
        p = tensor.fmatrix()
        u = tensor.fvector()
        m = theano.sandbox.multinomial.MultinomialFromUniform(dtype)(p, u)

        f = function([p, u], m, allow_input_downcast=True, mode=mode_with_gpu)
        assert any([type(node.op) is GPUAMultinomialFromUniform
                    for node in f.maker.fgraph.toposort()])
        pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
        pval = pval / pval.sum(axis=1)[:, None]
        uval = numpy.ones_like(pval[:, 0]) * 0.5
        samples = f(pval, uval)
        assert samples.dtype == dtype, "%s != %s" % (samples.dtype, dtype)


def test_gpu_opt():
    # Does have some overlap with test_multinomial_0

    # We test the case where we put the op on the gpu when the output
    # is moved to the gpu.
    p = tensor.fmatrix()
    u = tensor.fvector()
    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(p, u)
    assert m.dtype == 'float32', m.dtype

    f = function([p, u], m, allow_input_downcast=True, mode=mode_with_gpu)
    assert any([type(node.op) is GPUAMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    f(pval, uval)

    # Test with a row, it was failing in the past.
    r = tensor.frow()
    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(r, u)
    assert m.dtype == 'float32', m.dtype

    f = function([r, u], m, allow_input_downcast=True, mode=mode_with_gpu)
    assert any([type(node.op) is GPUAMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(1 * 4, dtype='float32').reshape((1, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    f(pval, uval)


class test_OP_wor(unittest.TestCase):

    def test_select_distinct(self):
        """
        Tests that MultinomialWOReplacementFromUniform always selects distinct elements
        """
        p = tensor.fmatrix()
        u = tensor.fvector()
        n = tensor.iscalar()
        m = multinomial.MultinomialWOReplacementFromUniform('auto')(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 1000
        all_indices = range(n_elements)
        numpy.random.seed(12345)
        for i in [5, 10, 50, 100, 500, n_elements]:
            uni = numpy.random.rand(i).astype(config.floatX)
            pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
            pvals /= pvals.sum(1)
            res = f(pvals, uni, i)
            res = numpy.squeeze(res)
            assert len(res) == i, res
            assert numpy.all(numpy.in1d(numpy.unique(res), all_indices)), res

    def test_fail_select_alot(self):
        """
        Tests that MultinomialWOReplacementFromUniform fails when asked to sample more
        elements than the actual number of elements
        """
        p = tensor.fmatrix()
        u = tensor.fvector()
        n = tensor.iscalar()
        m = multinomial.MultinomialWOReplacementFromUniform('auto')(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 200
        numpy.random.seed(12345)
        uni = numpy.random.rand(n_selected).astype(config.floatX)
        pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        self.assertRaises(ValueError, f, pvals, uni, n_selected)

    def test_select_proportional_to_weight(self):
        """
        Tests that MultinomialWOReplacementFromUniform selects elements, on average,
        proportional to the their probabilities
        """
        p = tensor.fmatrix()
        u = tensor.fvector()
        n = tensor.iscalar()
        m = multinomial.MultinomialWOReplacementFromUniform('auto')(p, u, n)

        f = function([p, u, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 10
        mean_rtol = 0.0005
        numpy.random.seed(12345)
        pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        avg_pvals = numpy.zeros((n_elements,), dtype=config.floatX)

        for rep in range(10000):
            uni = numpy.random.rand(n_selected).astype(config.floatX)
            res = f(pvals, uni, n_selected)
            res = numpy.squeeze(res)
            avg_pvals[res] += 1
        avg_pvals /= avg_pvals.sum()
        avg_diff = numpy.mean(abs(avg_pvals - pvals))
        assert avg_diff < mean_rtol, avg_diff


class test_function_wor(unittest.TestCase):

    def test_select_distinct(self):
        """
        Tests that multinomial_wo_replacement always selects distinct elements
        """
        th_rng = RandomStreams(12345)

        p = tensor.fmatrix()
        n = tensor.iscalar()
        m = th_rng.multinomial_wo_replacement(pvals=p, n=n)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 1000
        all_indices = range(n_elements)
        numpy.random.seed(12345)
        for i in [5, 10, 50, 100, 500, n_elements]:
            pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
            pvals /= pvals.sum(1)
            res = f(pvals, i)
            res = numpy.squeeze(res)
            assert len(res) == i
            assert numpy.all(numpy.in1d(numpy.unique(res), all_indices)), res

    def test_fail_select_alot(self):
        """
        Tests that multinomial_wo_replacement fails when asked to sample more
        elements than the actual number of elements
        """
        th_rng = RandomStreams(12345)

        p = tensor.fmatrix()
        n = tensor.iscalar()
        m = th_rng.multinomial_wo_replacement(pvals=p, n=n)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 200
        numpy.random.seed(12345)
        pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        self.assertRaises(ValueError, f, pvals, n_selected)

    def test_select_proportional_to_weight(self):
        """
        Tests that multinomial_wo_replacement selects elements, on average,
        proportional to the their probabilities
        """
        th_rng = RandomStreams(12345)

        p = tensor.fmatrix()
        n = tensor.iscalar()
        m = th_rng.multinomial_wo_replacement(pvals=p, n=n)

        f = function([p, n], m, allow_input_downcast=True)

        n_elements = 100
        n_selected = 10
        mean_rtol = 0.0005
        numpy.random.seed(12345)
        pvals = numpy.random.randint(1, 100, (1, n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        avg_pvals = numpy.zeros((n_elements,), dtype=config.floatX)

        for rep in range(10000):
            res = f(pvals, n_selected)
            res = numpy.squeeze(res)
            avg_pvals[res] += 1
        avg_pvals /= avg_pvals.sum()
        avg_diff = numpy.mean(abs(avg_pvals - pvals))
        assert avg_diff < mean_rtol


def test_gpu_opt_wor():
    # We test the case where we put the op on the gpu when the output
    # is moved to the gpu.
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.MultinomialWOReplacementFromUniform('auto')(p, u, n)
    assert m.dtype == 'int64', m.dtype

    f = function([p, u, n], m, allow_input_downcast=True, mode=mode_with_gpu)
    assert any([type(node.op) is GPUAMultinomialWOReplacementFromUniform
                for node in f.maker.fgraph.toposort()])
    n_samples = 3
    pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones(pval.shape[0] * n_samples) * 0.5
    f(pval, uval, n_samples)

    # Test with a row, it was failing in the past.
    r = tensor.frow()
    m = multinomial.MultinomialWOReplacementFromUniform('auto')(r, u, n)
    assert m.dtype == 'int64', m.dtype

    f = function([r, u, n], m, allow_input_downcast=True, mode=mode_with_gpu)
    assert any([type(node.op) is GPUAMultinomialWOReplacementFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(1 * 4, dtype='float32').reshape((1, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    f(pval, uval, 1)
