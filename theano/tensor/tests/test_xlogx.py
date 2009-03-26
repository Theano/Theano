from theano.tensor.xlogx import xlogx

import unittest

import theano
from theano.tensor import as_tensor_variable
import test_basic as TT

import random
import numpy.random
from theano.tests import unittest_tools

class T_XlogX(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

    def test0(self):
        x = as_tensor_variable([1, 0])
        y = xlogx(x)
        f = theano.function([], [y])
        self.failUnless(numpy.all(f() == numpy.asarray([0, 0.])))
    def test1(self):
#        class Dummy(object):
#            def make_node(self, a):
#                return [xlogx(a)[:,2]]
        TT.verify_grad(xlogx, [numpy.random.rand(3,4)])


if __name__ == '__main__':
    unittest.main()
