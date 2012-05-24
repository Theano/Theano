import theano
import numpy as np
from theano import tensor as T
from theano.tests import unittest_tools as utt


class BinCountOp(theano.Op):
    """Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest
    value in x. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of x). Each bin gives the
    number of occurrences of its index value in x. If weights is
    specified the input array is weighted by it, i.e. if a value n
    is found at position i, out[n] += weight[i] instead of out[n] += 1.
    Wraping of numpy.bincount

    Parameter:
    x -- 1 dimension, nonnegative ints

    Keywords arguments:
    weights -- Weights, array of the same shape as x.
    minlength -- A minimum number of bins for the output array.

    """
    compatible_type = ('int8', 'int16', 'int32', 'int64',
                       'uint8', 'uint16', 'uint32', 'uint64')

    def __init__(self, weights=None, minlength=None):
        self.weights = weights
        self.minlength = minlength

    def __eq__(self, other):
        return (type(self) == type(other) and
               self.weights == other.weights and
               self.minlength == other.minlength)

    def __hash__(self):
        h = 0
        if self.weights != None:
            for k in range(len(self.weights)):
                h = h ^ hash(self.weights[k])
        return hash(type(self)) ^ h ^ hash(self.minlength)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.dtype not in BinCountOp.compatible_type:
            raise TypeError("Inputs must be integers.")
        if x.ndim != 1:
            raise TypeError("Inputs must be of dimension 1.")
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        if x.dtype not in BinCountOp.compatible_type:
            raise TypeError("Inputs must be integers.")
        if x.ndim != 1:
            raise TypeError("Input must be of dimension 1.")
        z = output_storage[0]
        z[0] = np.bincount(x, self.weights, self.minlength)

    def grad(self, inputs, outputs_gradients):
        return [None for i in inputs]  # Non differentiable

    def infer_shape(self, node, ins_shapes):
        inputs = node.inputs[0]
        m = T.max(inputs) + 1
        if self.minlength != None:
            m = T.max(T.stack(m, self.minlength))
        return [[m]]

    def __str__(self):
        return self.__class__.__name__


def bincount(x, weights=None, minlength=None):
    return BinCountOp(weights=weights, minlength=minlength)(x)


class TestBinCountOp(utt.InferShapeTester):
    def setUp(self):
        super(TestBinCountOp, self).setUp()
        self.op_class = BinCountOp
        self.op = BinCountOp()

    def test_bincountOp(self):
        x = T.lvector('x')
        a = np.random.random_integers(50, size=(25))
        w = np.random.random((25,))

        f1 = theano.function([x], bincount(x))
        f2 = theano.function([x], bincount(x, weights=w))
        f3 = theano.function([x], bincount(x, minlength=23))

        assert (np.bincount(a) == f1(a)).all
        assert (np.bincount(a, weights=w) == f2(a)).all
        assert (np.bincount(a, minlength=23) == f3(a)).all

    def test_infer_shape(self):
        x = T.lvector('x')

        self._compile_and_check([x],
                                [self.op(x)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)

        w = np.random.random((25,))
        self._compile_and_check([x],
                                [bincount(x, weights=w)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)

        self._compile_and_check([x],
                                [bincount(x, minlength=60)],
                                [np.random.random_integers(50, size=(25,))],
                                self.op_class)
