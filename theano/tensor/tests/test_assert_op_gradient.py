import numpy

import theano
import theano.tensor as T
from theano.tensor.opt import Assert


def test_assert_op_gradient():
    x = T.vector('x')
    assert_op = Assert()
    cost = T.sum(assert_op(x, x.size < 2))
    grad = T.grad(cost, x)
    func = theano.function([x], grad)

    x_val = numpy.ones(shape=(1,))
    assert func(x_val) == 1
