import numpy
from theano.gof import Op, Apply
from theano import tensor, function
from theano.tests import unittest_tools as utt


class Tril(Op):
    """
    Lower triangle of an array.

    An instance of this class returns a copy of an array with elements below
    the k-th diagonal zeroed.

    Parameters:

    m: array_like, shape (M, N). Input array.

    k: int, optional. Diagonal above which to zero elements. k = 0 (the
        default) is the main diagonal, k < 0 is below it and k > 0 is
        above.

    Returns:

    ndarray, shape (M, N). Lower triangle of m, of same shape and
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
        if not k:
            k = 0
        output_storage[0][0] = numpy.tril(m, k)

    def grad(self, inputs, cost_grad):
        m, k = inputs
        ingrad = tensor.as_tensor_variable(cost_grad[0])
        outgrad_a = ingrad * tril(tensor.ones((m.shape[0], m.shape[1])), k)
        return [outgrad_a, None]


def tril(m, k=0):

    if k is None:
        return Tril()(m, 0)
    else:
        return Tril()(m, k)

