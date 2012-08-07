import numpy
from theano.gof import Op, Apply
from theano import tensor, function
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import matrix_inverse, kron

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False


class Solve(Op):
    """
    Solves the matrix equation a x = b for x.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    sym_pos: (boolean) Assume a is symmetric and positive definite.
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    overwrite_a: (boolean) Allow overwriting data in a (may enhance
    performance).
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
    performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
        self.sym_pos = sym_pos
        self.lower = lower
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and self.sym_pos == other.sym_pos and
                self.lower == other.lower and
                self.overwrite_a == other.overwrite_a and
                self.overwrite_b == other.overwite_b)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.sym_pos) ^ hash(self.lower) ^
                hash(self.overwrite_a) ^ hash(self.overwrite_b))

    def props(self):
        return (self.sym_pos, self.lower, self.overwrite_a, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s, %s}" % (self.__class__.__name__,
                "sym_pos=".join(str(self.sym_pos)),
                "lower=".join(str(self.lower)),
                "overwrite_a".join(str(self.overwrite_a)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
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
            output_storage[0][0] = scipy.linalg.solve(a, b, self.sym_pos,
                        self.lower, self.overwrite_a, self.overwrite_b)
        except Exception, e:
            e.args = e.args + ('array \'a\' might be singular',) 
            raise 

    def grad(self, inputs, cost_grad):
        """
        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK

        Note: In contrast with the usual mathematical presentation, in order
        to apply theano's 'reshape' function wich implements row-order
        (i.e. C order), the differential expressions below have been derived
        around the row-vectorizations of inputs 'a' and 'b'.
        """

        a, b = inputs
        ingrad = cost_grad
        ingrad = tensor.as_tensor_variable(ingrad)
        inv_a = matrix_inverse(a)

        if b.ndim == 1:
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            prod_a_b = tensor.shape_padleft(prod_a_b)
            jac_veca = kron(inv_a, prod_a_b)
            jac_b = inv_a
            outgrad_veca = tensor.tensordot(ingrad, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0]))
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
                        (inputs[0].shape[0], inputs[0].shape[0]))
            outgrad_vecb = tensor.tensordot(ingrad_vec, jac_vecb, axes=1)
            outgrad_b = tensor.reshape(outgrad_vecb,
                        (inputs[1].shape[0], inputs[1].shape[1]))

        return [outgrad_a, outgrad_b]


def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
    localop = Solve(sym_pos, lower, overwrite_a, overwrite_b)
    return localop(a, b)


#TODO: Optimizations to replace multiplication by matrix inverse
#      with Ops solve() or solve_triangular()


