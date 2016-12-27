from __future__ import absolute_import, print_function, division

import numpy

import theano.tensor as tensor
from theano import config
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import relu


def test_relu():
    x = tensor.ftensor4('x')
    seed = utt.fetch_seed()
    rng = numpy.random.RandomState(seed)
    X = rng.randn(20, 20, 30, 30).astype(config.floatX)

    # test the base case, without custom alpha value
    y = relu(x).eval({x: X})
    assert numpy.allclose(y, numpy.maximum(X, 0))

    # test for different constant alpha values (also outside of [0, 1])
    for alpha in 0, 0.3, 1, 2, -0.3, -1, -2:
        y = relu(x, alpha).eval({x: X})
        assert numpy.allclose(y, numpy.where(X > 0, X, alpha * X))

    def mp(input):
       return relu(input, alpha)

    imval = rng.rand(2, 3, 3, 4) * 10.0
    for alpha in (0.0, 0.3, 1.0, -0.3, -1.0):
        utt.verify_grad(mp, [imval], rng=rng)

if __name__ == "__main__":
    test_relu()
