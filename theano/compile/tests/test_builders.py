import numpy
import unittest

from theano import config

from theano.compile import function

from theano import tensor
from theano import tensor as T

from theano.compile.builders import OpFromGraph


class T_OpFromGraph(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN')
        f = op(x, y, z) - op(y, z, x)  # (1+3*5=array of 16) - (3+1*5=array of 8)
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX)*3
        zv = numpy.ones((2, 2), dtype=config.floatX)*5
        #print function, function.__module__
        #print fn.maker.fgraph.toposort()
        fn(xv, yv, zv)
        assert numpy.all(8.0 == fn(xv, yv, zv))
        assert numpy.all(8.0 == fn(xv, yv, zv))

    def test_size_changes(self):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = OpFromGraph([x, y], [e], mode='FAST_RUN')
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 3), dtype=config.floatX)
        yv = numpy.ones((3, 4), dtype=config.floatX)*3
        zv = numpy.ones((4, 5), dtype=config.floatX)*5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)

    def test_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN', grad_depth=2)
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX)*3
        zv = numpy.ones((2, 2), dtype=config.floatX)*5
        assert numpy.all(11.0 == fn(xv, yv, zv))


if __name__ == '__main__':
    unittest.main()
