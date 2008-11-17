from xlogx import xlogx

import unittest
from theano import compile
from theano import gradient

from theano.tensor import as_tensor
import theano._test_tensor as TT

import random
import numpy.random

class T_XlogX(unittest.TestCase):
    def test0(self):
        x = as_tensor([1, 0])
        y = xlogx(x)
        y = compile.eval_outputs([y])
        self.failUnless(numpy.all(y == numpy.asarray([0, 0.])))
    def test1(self):
        class Dummy(object):
            def make_node(self, a):
                return [xlogx(a)[:,2]]
        TT.verify_grad(self, Dummy(), [numpy.random.rand(3,4)])


if __name__ == '__main__':
    unittest.main()
