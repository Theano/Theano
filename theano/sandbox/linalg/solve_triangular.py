import numpy
from theano.gof import Op, Apply
from theano import tensor, function, config
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import matrix_inverse, kron, triu, tril

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False


class SolveTriangular(Op):
    """
    An instance of this class solves the matrix equation a x = b for x where
    'a' is triangular.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    unit_diagonal : (boolean) If True, diagonal elements of A are assumed to be
        1 and will not be referenced.
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
        performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, lower=False, unit_diagonal=False, overwrite_b=False):

        self.lower = lower
        self.unit_diagonal = unit_diagonal
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.lower == other.lower and
                self.unit_diagonal == other.unit_diagonal and
                self.overwrite_b == other.overwrite_b)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.lower) ^
                hash(self.unit_diagonal) ^ hash(self.overwrite_b))

    def props(self):
        return (self.lower, self.unit_diagonal, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s}" % (self.__class__.__name__,
                "lower=".join(str(self.lower)),
                "unit_diagonal".join(str(self.unit_diagonal)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'SolveTriangular{%s}' % str(self.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the SolveTriangular op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        if a.ndim != 2 or  b.ndim > 2 or b.ndim == 0:
            raise TypeError('%s: inputs have improper dimensions:\n'
                    '\'a\' must have two,'
                    ' \'b\' must have either one or two' %
                            self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(a * b).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs
        if a.shape[0] != a.shape[1] or a.shape[1] != b.shape[0]:
            raise TypeError('%s: inputs have improper lengths' %
                            self.__class__.__name__)
        try:
            output_storage[0][0] = scipy.linalg.solve_triangular(a, b,
                            trans=0, lower=self.lower,
                       unit_diagonal=self.unit_diagonal,
                             overwrite_b=self.overwrite_b, debug=False)
        except:
            raise  Exception('%s: array \'a\' is singular'
                             % self.__class__.__name__)

    def grad(self, inputs, cost_grad):
        """
        Notes:
        1. The gradient is computed under the assumption that perturbations
        of the input array respect triangularity, i.e. partial derivatives wrt
        triangular region are zero.
        2. In contrast with the usual mathematical presentation, in order to
        apply theano's 'reshape' function wich implements row-order (i.e. C
        order), the differential expressions below have been derived based on
        the row-vectorizations of inputs 'a' and 'b'.

        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK
        """

        a, b = inputs
        ingrad = cost_grad
        ingrad = tensor.as_tensor_variable(ingrad)
        shp_a = (tensor.shape(inputs[0])[1],
                               tensor.shape(inputs[0])[1])
        I_M = tensor.eye(*shp_a)
        if self.lower:
            inv_a = solve_triangular(a, I_M, lower=True)
            tri_M = tril(tensor.ones(shp_a))
        else:
            inv_a = solve_triangular(a, I_M, lower=False)
            tri_M = triu(tensor.ones(shp_a))
        if b.ndim == 1:
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            prod_a_b = tensor.shape_padleft(prod_a_b)
            jac_veca = kron(inv_a, prod_a_b)
            jac_b = inv_a
            outgrad_veca = tensor.tensordot(ingrad, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_b = tensor.tensordot(ingrad, jac_b, axes=1).flatten(ndim=1)
        else:
            ingrad_vec = ingrad.flatten(ndim=1)
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            jac_veca = kron(inv_a, prod_a_b)
            I_N = tensor.eye(tensor.shape(inputs[1])[1],
                               tensor.shape(inputs[1])[1])
            jac_vecb = kron(inv_a, I_N)
            outgrad_veca = tensor.tensordot(ingrad_vec, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0])) * tri_M
            outgrad_vecb = tensor.tensordot(ingrad_vec, jac_vecb, axes=1)
            outgrad_b = tensor.reshape(outgrad_vecb,
                        (inputs[1].shape[0], inputs[1].shape[1]))
        return [outgrad_a, outgrad_b]


def solve_triangular(a, b, lower=False, unit_diagonal=False,
                             overwrite_b=False):
    return SolveTriangular(lower=lower, unit_diagonal=unit_diagonal,
                           overwrite_b=overwrite_b)(a, b)


#TODO: Optimizations to replace multiplication by matrix inverse
#      with Ops solve() or solve_triangular()


class TestSolveTriangular(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):

        super(TestSolveTriangular, self).setUp()
        self.op_class = SolveTriangular
        self.op = solve_triangular

    def test_perform(self):

        init = numpy.random.rand(4, 4)
        x = tensor.dmatrix()
        for val in [True, False]:
            if val:
                a = numpy.tril(init)
            else:
                a = numpy.triu(init)
                
            y = tensor.dmatrix()
            f = function([x, y], self.op(x, y, lower=val))
            for shp in [(4, 5), (4, 1)]:
                b = numpy.random.rand(*shp)
                out = f(a, b)
                assert numpy.allclose(out,
                        scipy.linalg.solve_triangular(a, b, lower=val))

            y = tensor.dvector()
            f = function([x, y], self.op(x, y, lower=val))
            b = numpy.random.rand(4)
            out = f(a, b)
            assert numpy.allclose(out,
                    scipy.linalg.solve_triangular(a, b, lower=val))

    def test_gradient(self):
        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(5, 5)),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(4, 4)),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=True),
                        [numpy.tril(numpy.random.rand(4, 4)),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolveTriangular.rng)
        
        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(5, 5)),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolveTriangular.rng)

        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(4, 4)),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolveTriangular.rng)

        utt.verify_grad(SolveTriangular(lower=False),
                        [numpy.triu(numpy.random.rand(4, 4)),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolveTriangular.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dmatrix()

        a55 = numpy.random.rand(5, 5)
        a44 = numpy.random.rand(4, 4)
        b52 = numpy.random.rand(5, 2)
        b41 = numpy.random.rand(4, 1)
        b4 = numpy.random.rand(4)

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.tril(a55), b52], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.tril(a44), b41], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a55), b52], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a44), b41], self.op_class)

        y = tensor.dvector()

        self._compile_and_check([x, y], [self.op(x, y, lower=True)],
                                [numpy.triu(a44), b4], self.op_class)

        self._compile_and_check([x, y], [self.op(x, y, lower=False)],
                                [numpy.triu(a44), b4], self.op_class)


if __name__ == "__main__":
    t = TestSolveTriangular('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
