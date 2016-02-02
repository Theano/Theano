from __future__ import absolute_import, print_function, division
import unittest

import numpy
import numpy.random

import theano
from theano.tests import unittest_tools as utt

'''
  Different tests that are not connected to any particular Op, or
  functionality of Theano. Here will go for example code that we will
  publish in papers, that we should ensure that it will remain
  operational

'''


class T_scipy(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.orig_floatX = theano.config.floatX

    def tearDown(self):
        theano.config.floatX = self.orig_floatX

    def test_scipy_paper_example1(self):
        a = theano.tensor.vector('a')  # declare variable
        b = a + a**10                  # build expression
        f = theano.function([a], b)    # compile function
        assert numpy.all(f([0, 1, 2]) == numpy.array([0, 2, 1026]))

    def test_scipy_paper_example2(self):
        ''' This just sees if things compile well and if they run '''
        # PREAMPBLE
        T = theano.tensor
        shared = theano.shared
        function = theano.function
        rng = numpy.random
        theano.config.floatX = 'float64'

        #
        # ACTUAL SCRIPT FROM PAPER

        x = T.matrix()
        y = T.vector()
        w = shared(rng.randn(100))
        b = shared(numpy.zeros(()))

        # Construct Theano expression graph
        p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
        xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
        prediction = p_1 > 0.5
        cost = xent.mean() + 0.01 * (w ** 2).sum()
        gw, gb = T.grad(cost, [w, b])

        # Compile expressions to functions
        train = function(
            inputs=[x, y],
            outputs=[prediction, xent],
            updates=[(w, w - 0.1 * gw), (b, b - 0.1 * gb)])
        function(inputs=[x], outputs=prediction)

        N = 4
        feats = 100
        D = (rng.randn(N, feats), rng.randint(size=4, low=0, high=2))
        training_steps = 10
        for i in range(training_steps):
            pred, err = train(D[0], D[1])


if __name__ == '__main__':
    unittest.main()
