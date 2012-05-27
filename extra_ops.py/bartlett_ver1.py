import numpy
from theano import gof, tensor, function
from theano.tests import unittest_tools as utt


class Bartlett(gof.Op):
    """
    An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    input : (integer scalar) Number of points in the output window. If zero or
    less, an empty vector is returned.

    output : (vector of doubles) The triangular window, with the maximum value
    normalized to one (the value one appears only if the number of samples is
    odd), with the first and last samples equal to zero.
    """

    def __init__(self):
        gof.Op.__init__(self)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, M):
        M = tensor.as_tensor_variable(M)
        if M.ndim != 0:
            raise TypeError('%s only works on scalar input'
                            % self.__class__.__name__)
        elif not M.dtype.startswith('int'):
        # dtype is a theano attribute here
            raise TypeError('%s only works on integer input'
                            % self.__class__.__name__)
        return gof.Apply(self, [M], [tensor.dvector()])

    def perform(self, node, inputs, out_):
        M = inputs[0]
        if not M.dtype.name.startswith('int'):
        # dtype is an instance of numpy dtype class here
            raise TypeError('%s only works on integers'
                            % self.__class__.__name__)
        out, = out_
        out[0] = numpy.bartlett(M)

    def infer_shape(self, node, in_shapes):
        M = node.inputs[0]
        return [[M]]

    def grad(self, inputs, output_grads):
        return [None for i in inputs]


def bartlett(M):
    localop = Bartlett()
    return localop(M)


class TestBartlett(utt.InferShapeTester):

    def setUp(self):
        super(TestBartlett, self).setUp()
        self.op_class = Bartlett
        self.op = bartlett

    def test_perform(self):
        x = tensor.lscalar()
        f = function([x], self.op(x))
        M = numpy.random.random_integers(3, 50, size=())
        assert numpy.allclose(f(M), numpy.bartlett(M))

    def test_infer_shape(self):
        x = tensor.lscalar()
        self._compile_and_check([x], [self.op(x)],
                                [numpy.random.random_integers(3, 50, size=())],
                                self.op_class)

if __name__ == "__main__":
    t = TestBartlett('setUp')
    t.setUp()
    t.test_perform()
    t.test_infer_shape()
