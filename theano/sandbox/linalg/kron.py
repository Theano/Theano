import numpy
from theano.gof import Op, Apply
from theano import tensor, function
from theano.tests import unittest_tools as utt

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False


class Kron(Op):
    """
    Kronecker product of a and b.

    Parameters:

    a: array, shape (M, N)
    b: array, shape (P, Q)

    Returns:

    A: array, shape (M*P, N*Q)

    The result is the block matrix:

    a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
    a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
    ...
    a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "%s" % self.__class__.__name__

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)

        if a.ndim != 2 or b.ndim != 2:
            raise TypeError('%s: inputs must have two dimensions' %
                            self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(a[0, 0] * b[0, 0]).dtype,
                                     broadcastable=(False, False))()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        shape_a, shape_b = in_shapes
        return [[shape_a[0] * shape_b[0], shape_a[1] * shape_b[1]]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs
        output_storage[0][0] = scipy.linalg.kron(a, b)

    def grad(self, inputs, cost_grad):
        raise NotImplementedError('%s: gradient is not currently'
                ' implemented' % self.__class__.__name__)

kron = Kron()


class TestKron(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestKron, self).setUp()
        self.op_class = Kron
        self.op = kron

    def test_perform(self):
        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], kron(x, y))

        for shp0 in [(8, 6), (5, 8)]:
            for shp1 in [(5, 7), (3, 3)]:
                a = numpy.random.rand(*shp0)
                b = numpy.random.rand(*shp1)
                out = f(a, b)
                assert numpy.allclose(out, scipy.linalg.kron(a, b))

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dmatrix()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(8, 5),
                                 numpy.random.rand(3, 7)],
                                self.op_class)
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(2, 5),
                                 numpy.random.rand(6, 3)],
                                self.op_class)

if __name__ == "__main__":
    t = TestKron('setUp')
    t.setUp()
    t.test_perform()
    t.test_infer_shape()
