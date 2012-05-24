# TODO implement grad for higher dimension

import theano
import numpy as np
from theano import tensor as T
from theano.tests import unittest_tools as utt


class DiffOp(theano.Op):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[n] = a[n+1] - a[n]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff for vector.

    Parameter:
    x -- Input vector.

    Keywords arguments:
    n -- The number of times values are differenced, default is 1.

    """

    def __init__(self, n=1):
        self.n = n
        # self.axis = axis

    def __eq__(self, other):
        return (type(self) == type(other) and
               self.n == other.n)
               # self.axis == other.axis

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n)  # ^ hash(self.axis)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.diff(x, self.n)  # axis

    def grad(self, inputs, outputs_gradients):
        z = outputs_gradients[0]

        def _grad_helper(z):
            pre = T.concatenate([[0.], z])  # Prepend 0
            app = T.concatenate([z, [0.]])  # Append 0
            return pre - app

        for k in range(self.n):  # Apply grad recursively
            z = _grad_helper(z)
        return [z]

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        out_shape = list(i0_shapes)
        out_shape[0] = out_shape[0] - self.n  # Axis
        return [out_shape]

    def __str__(self):
        return self.__class__.__name__


def diff(x, n=1):  # Axis
    return DiffOp(n=n)(x)


class TestDiffOp(utt.InferShapeTester):
    nb = 10  # Number of time iterating for n

    def setUp(self):
        super(TestDiffOp, self).setUp()
        self.op_class = DiffOp
        self.op = DiffOp()

    def test_diffOp(self):
        x = T.dvector('x')
        a = np.random.random(500)

        f = theano.function([x], diff(x))
        assert np.allclose(np.diff(a), f(a))

        # Test n
        for k in range(TestDiffOp.nb):
            g = theano.function([x], diff(x, n=k))
            assert np.allclose(np.diff(a, n=k), g(a))

    def test_infer_shape(self):
        x = T.dvector('x')

        self._compile_and_check([x],
                                [self.op(x)],
                                [np.random.random(500)],
                                self.op_class)

        for k in range(TestDiffOp.nb):
            self._compile_and_check([x],
                                    [DiffOp(n=k)(x)],
                                    [np.random.random(500)],
                                    self.op_class)

    def test_grad(self):
        x = T.vector('x')
        a = np.random.random(500)

        gf = theano.function([x], T.grad(T.sum(diff(x)), x))
        utt.verify_grad(self.op, [a])

        # Test n
        for k in range(TestDiffOp.nb):
            dg = theano.function([x], T.grad(T.sum(diff(x, n=k)), x))
            utt.verify_grad(DiffOp(n=k), [a])
