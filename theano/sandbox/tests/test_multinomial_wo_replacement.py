from __future__ import absolute_import, print_function, division
import numpy
from theano import config, function, tensor
from theano.sandbox import multinomial
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import unittest


class test_OP(unittest.TestCase):

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
            assert len(res) == i
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


class test_function(unittest.TestCase):

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
