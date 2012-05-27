import numpy
from theano import tensor, gof, function, scalar
from theano.sandbox.linalg.ops import diag
from theano.tests import unittest_tools as utt


class FillDiagonalCopy(gof.Op):
    """
    An instance of this class returns a copy of an array with all elements of
    the main diagonal set to a specified scalar value.

    inputs:

    a : Rectangular array of at least two dimensions.
    val : Scalar value to fill the diagonal whose type must be compatible with
    that of array 'a' (i.e. 'val' must not be an upcasting of 'a').

    output:

    An array identical to 'a' except that its main diagonal is filled with
    scalar 'val'. (For an array 'a' with a.ndim >= 2, the main diagonal is the
    list of locations a[i, i, ..., i] (i.e. with indices all identical).)
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash_(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val):
        a = tensor.as_tensor_variable(a)
        val = tensor.as_tensor_variable(val)
        if a.ndim < 2:
            raise TypeError('%s: first parameter must have at least'
                            ' two dimensions' % self.__class__.__name__)
        elif val.ndim != 0:
            raise TypeError('%s: second parameter must be a scalar'
                            % self.__class__.__name__)
        val = tensor.cast(val, dtype=scalar.upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError('%s: type of second parameter must be compatible'
                          ' with first\'s' % self.__class__.__name__)
        return gof.Apply(self, [a, val], [a.type()])

    def perform(self, node, inputs, output_storage):
        if inputs[0].ndim < 2:
            raise TypeError('%s: first parameter must have at least'
                            ' two dimensions' % self.__class__.__name__)
        elif inputs[1].ndim != 0:
            raise TypeError('%s: second parameter must be a scalar'
                            % self.__class__.__name__)
        a = inputs[0].copy()
        val = inputs[1]
        numpy.fill_diagonal(a, val)
        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Note: The gradient is currently implemented for matrices
        only.
        """
        a, val = inp
        grad = cost_grad[0]
        if (a.dtype == 'complex64' or a.dtype == 'complex128' or
            val.dtype == 'complex64' or val.dtype == 'complex128'):
            return [None, None]
        elif a.ndim > 2:
            raise NotImplementedError('%s: gradient is currently implemented'
                            ' for matrices only' % self.__class__.__name__)
        wr_a = grad.copy()
        wr_a = fill_diagonal(wr_a, 0)  # valid for any number of dimensions
        wr_val = diag(grad).sum()  # diag is only valid for matrices
        return [wr_a, wr_val]


def fill_diagonal(in_a, in_val):
    localop = FillDiagonalCopy()
    return localop(in_a, in_val)


class TestFillDiagonalCopy(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestFillDiagonalCopy, self).setUp()
        self.op_class = FillDiagonalCopy
        self.op = fill_diagonal

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.dscalar()
        f = function([x, y], fill_diagonal(x, y))
        g = function([x], diag(x))
        a = numpy.random.rand(8, 5)
        val = numpy.random.rand()
        out = f(a, val)
        numpy.fill_diagonal(a, val)
        # remember that numpy.fill_diagonal works in place
        assert numpy.allclose(out, a)
        
    def test_gradient(self):
        #  TODO: check why gradient wrto val does not match when a has more rows
        #  than cols: might be problem with testing procedure
        utt.verify_grad(fill_diagonal, [numpy.random.rand(5, 8),
                                        numpy.random.rand()],
                        n_tests=1, rng=TestFillDiagonalCopy.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dscalar()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(8, 5), numpy.random.rand()],
                                self.op_class)


if __name__ == "__main__":
    t = TestFillDiagonalCopy('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
