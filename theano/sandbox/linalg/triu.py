import numpy
from theano.gof import Op, Apply
from theano import tensor, function
from theano.tests import unittest_tools as utt


class Triu(Op):
    """
    Upper triangle of an array.

    An instance of this class returns a copy of an array with elements above
    the k-th diagonal zeroed.

    Parameters:

    m: array_like, shape (M, N). Input array.

    k: int, optional. Diagonal below which to zero elements. k = 0 (the
        default) is the main diagonal, k < 0 is below it and k > 0 is
        above.

    Returns:

    ndarray, shape (M, N). Upper triangle of m, of same shape and
        data-type as m.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "%s" % self.__class__.__name__

    def make_node(self, m, k):
        m = tensor.as_tensor_variable(m)
        k = tensor.as_tensor_variable(k)
        if m.ndim != 2:
            raise TypeError('%s input array must have two dimensions'
                            % self.__class__.__name__)
        elif k.ndim != 0:
            raise TypeError('%s location of boundary must be a scalar'
                            % self.__class__.__name__)
        elif (not k.dtype.startswith('int')) and \
              (not k.dtype.startswith('uint')):
            raise TypeError('%s location of boundary must be an integer'
                            % self.__class__.__name__)
        return Apply(self, [m, k], [m.type()])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def perform(self, node, inputs, output_storage):
        m, k = inputs
        output_storage[0][0] = numpy.triu(m, k)

    def grad(self, inputs, cost_grad):
        m, k = inputs
        ingrad = tensor.as_tensor_variable(cost_grad[0])
        outgrad_a = ingrad * triu(tensor.ones((m.shape[0], m.shape[1])), k)
        return [outgrad_a, None]


def triu(m, k=0):

    if k is None:
        return Triu()(m, 0)
    else:
        return Triu()(m, k)


class TestTriu(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestTriu, self).setUp()
        self.op_class = Triu
        self.op = triu

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.iscalar()
        f = function([x, y], self.op(x, y))
        f_def = function([x], self.op(x))

        for shp in [(2, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            out = f_def(m)
            assert numpy.allclose(out, numpy.triu(m))

            for k in [0, 1, 2, -1]:
                out = f(m, k)
                assert numpy.allclose(out, numpy.triu(m, k))

    def test_gradient(self):
        def helper(x):
            return self.op(x, k)

        for shp in [(5, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            utt.verify_grad(self.op, [m], n_tests=1, rng=TestTriu.rng)

            for k in [0, 1, 2, -1]:
                utt.verify_grad(helper, [m], n_tests=1, rng=TestTriu.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.iscalar()

        for shp in [(2, 5), (1, 4), (4, 1)]:
            m = numpy.random.rand(*shp)
            self._compile_and_check([x], [self.op(x)],
                                [m], self.op_class)
            for k in [0, 1, 2, -1]:
                self._compile_and_check([x, y], [self.op(x, y)],
                                [m, k], self.op_class)


if __name__ == "__main__":
    t = TestTriu('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
