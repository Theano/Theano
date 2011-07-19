"""
 WRITE ME

 Tests for the R operator / L operator

ops without:

    PermuteRowElements
    Tile
    AdvancedSubtensor
    TensorDot
    Outer
    Prod
    MulwithoutZeros
    ProdWithoutZeros


list of ops that support R-op:
    * Alloc
    * Split
    * ARange
    * ScalarFromTensor
    * Shape
    * SpecifyShape
    * MaxAndArgmax
    * Subtensor
    * IncSubtensor
    * Rebroadcast
    * Join
    * Reshape
    * Flatten
    * AdvancedSubtensor1
    * AdvancedIncSubtensor1
    * AdvancedIncSubtensor
    * Dot
    * DimShuffle
    * Elemwise
    * Sum
    * Softmax
    * Scan






"""

import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
import theano.tensor as TT
import numpy
from theano.gof import Op, Apply

'''
Special Op created to test what happens when you have one op that is not
differentiable in the computational graph
'''
class BreakRop(Op):
    """
    @note: Non-differentiable.
    """
    def __hash__(self):
        return hash(type(self))
    def __eq__(self, other):
        return type(self) == type(other)
    def make_node(self, x):
        return Apply(self, [x], [x.type()])
    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        out[0] = x
    def grad(self, inp, grads):
        return [None]
    def R_op(self, inputs, eval_points):
        return [None]

    def test_specifyshape(self):
        rng  = numpy.random.RandomState(utt.fetch_seed())
        vx = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)
        vv = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)

        x  = TT.vector('x')
        v  = TT.vector('v')
        y  = TT.specify_shape(x, (5,))
        yv = TT.Rop(y,x,v)
        rop_f = function([x,v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(x.shape[0]),
                           non_sequences = [y,x])
        sy = TT.dot(J, v)

        scan_f = function([x,v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2)


class test_lop(unittest.TestCase):

    def test_specifyshape(self):
        rng  = numpy.random.RandomState(utt.fetch_seed())
        vx = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)
        vv = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)

        x  = TT.vector('x')
        v  = TT.vector('v')
        y  = TT.specify_shape(x, (5,))
        yv = TT.Lop(y,x,v)
        rop_f = function([x,v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(x.shape[0]),
                           non_sequences = [y,x])
        sy = TT.dot(v, J)

        scan_f = function([x,v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2)


