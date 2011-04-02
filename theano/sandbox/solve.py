import numpy, scipy.linalg
from theano import gof, tensor, scalar
import unittest


class Solve(gof.Op):
    """
    Find the solution to the linear equation Ax=b,
    where A is a 2d matrix and b is a 1d or 2d matrix.
    It use numpy.solve to find the solution.
    """

    #TODO: Add class options to use the performance-enhancing flags
    #     sym_pos, lower, overwrite_a, overwrite_b

    #TODO: Add C code that calls the underlying LAPACK routines
    #      and keeps a memory workspace from call to call as a non-default Op output

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, A, b):
        A_ = tensor.as_tensor_variable(A)
        b_ = tensor.as_tensor_variable(b)
        if A_.broadcastable != (False, False):
            raise TypeError("A must be a matrix", A_.type)
        if b_.broadcastable not in ((False,), (True, False), (False, False)):
            raise TypeError("b must be a matrix or vector", b_.type)
        odtype = scalar.upcast(A_.dtype, b_.dtype)
        otype = tensor.TensorType(broadcastable=b_.broadcastable, dtype=odtype)
        return gof.Apply(op=self, inputs=[A, B], outputs=[otype()])

    def perform(self, node, inp, out):
        A, b = inp
        output, = out
        ret=scipy.linalg.solve(A,b)
        if ret.dtype != node.outputs[0].dtype:
            print >> sys.stderr, "WARNING: Solve.perform() required cast."
            ret = theano._asarray(ret, dtype=node.outputs[0].dtype)
        output[0]=ret

solve = Solve()


## TODO: test dtype conversion
## TODO: test that invalid types are rejected by make_node
## TODO: test that each valid type for A and b works correctly
from theano.tests import unittest_tools as utt
class T_solve(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed(666))

    def test0(self):
        A=self.rng.randn(5,5)
        b=numpy.array(range(5),dtype=float)
        x=scipy.linalg.solve(A,b)
        Ax = numpy.dot(A,x)
        are = tensor.numeric_grad.abs_rel_err(Ax, b)
        self.assertTrue(numpy.all(are < 1.0e-5), (are, Ax, b))
        #print A,b
        #print numpy.dot(A,x)
