import numpy
import theano
from theano.gof import Op, Apply
from theano import tensor

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
    (notice that a[i,j]*b is itself a matrix of the same shape as b)

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
            "Scipy not available. Scipy is needed for the Kron op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)

        if (not a.ndim == 2 or not b.ndim == 2):
            raise TypeError('%s: inputs must have two dimensions' %
                            self.__class__.__name__)

        out_var = tensor.TensorType(dtype=theano.scalar.upcast(a, b),
                                     broadcastable=(False, False))()
        return Apply(self, [a, b], [out_var])

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
