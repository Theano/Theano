import numpy
from theano import gof, tensor
import unittest

class Solve(gof.Op):
    """
    Find the solution to the linear equation Ax=b,
    where A is a 2d matrix and b is a 1d or 2d matrix.
    It use numpy.solve to find the solution.
    """

    def make_node(self, A, b):
        if not isinstance(A, gof.Variable) or not A.type==tensor.matrix().type:
            raise TypeError("We expected that A had a matrix type")
        if not isinstance(B, gof.Variable) or not B.type==tensor.matrix().type:
            raise TypeError("We expected that B had a matrix type")

        node = gof.Apply(op=self, inputs=[A, B], outputs=[tensor.matrix()])
        return node

    def perform(self, node, (A, B), (output, )):
        ret=numpy.solve(A,B)
        output[0]=ret

    def grad(self, (theta, A, B), (gtheta,)):
        raise NotImplementedError()

solve = Solve()


class T_solve(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed(666))

    def test0(self):
        A=self.rng.randn(5,5)
        b=numpy.array(range(5),dtype=float)
        x=numpy.linalg.solve(A,b)
        Ax = numpy.dot(A,x)
        are = tensor.numeric_grad.abs_rel_err(Ax, b)
        self.failUnless(numpy.all(are < 1.0e-5), (are, Ax, b))
        #print A,b
        #print numpy.dot(A,x)

