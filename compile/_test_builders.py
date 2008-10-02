
import unittest
import gof, gof.opt

from theano import compile
from theano.compile.function_module import *
from theano.scalar import *

from theano import tensor
from theano import tensor as T
import random
import numpy as N

from theano.compile.builders import *


class T_OpFromGraph(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN')
        f = op(x, y, z) - op(y, z, x)
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(8.0 == fn(xv, yv, zv))
        assert numpy.all(8.0 == fn(xv, yv, zv))
    
    def test_size_changes(self):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = OpFromGraph([x, y], [e], mode='FAST_RUN')
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 3)), N.ones((3, 4))*3, N.ones((4, 5))*5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
    
    def test_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN', grad_depth = 2)
        f = op(x, y, z)
        f = f - T.grad(f, y)
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(11.0 == fn(xv, yv, zv))


if __name__ == '__main__':
    unittest.main()

