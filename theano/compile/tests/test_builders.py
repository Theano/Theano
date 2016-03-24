from __future__ import absolute_import, print_function, division
import numpy

from theano import config, shared

from theano.compile import function

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theano.compile.builders import OpFromGraph

from theano.tests import unittest_tools


class T_OpFromGraph(unittest_tools.InferShapeTester):

    def test_straightforward(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX) * 3
        zv = numpy.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        fn(xv, yv, zv)
        assert numpy.all(8.0 == fn(xv, yv, zv))
        assert numpy.all(8.0 == fn(xv, yv, zv))

    def test_size_changes(self):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = OpFromGraph([x, y], [e])
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 3), dtype=config.floatX)
        yv = numpy.ones((3, 4), dtype=config.floatX) * 3
        zv = numpy.ones((4, 5), dtype=config.floatX) * 5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)

    def test_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX) * 3
        zv = numpy.ones((2, 2), dtype=config.floatX) * 5
        assert numpy.all(11.0 == fn(xv, yv, zv))

    def test_grad_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX) * 3
        zv = numpy.ones((2, 2), dtype=config.floatX) * 5
        assert numpy.allclose(6.0, fn(xv, yv, zv))

    def test_shared(self):
        x, y, z = T.matrices('xyz')
        s = shared(numpy.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = OpFromGraph([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX) * 3
        zv = numpy.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        assert numpy.allclose(8.0, fn(xv, yv, zv))
        assert numpy.allclose(8.0, fn(xv, yv, zv))

    def test_shared_grad(self):
        x, y, z = T.matrices('xyz')
        s = shared(numpy.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = OpFromGraph([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = numpy.ones((2, 2), dtype=config.floatX)
        yv = numpy.ones((2, 2), dtype=config.floatX) * 3
        zv = numpy.ones((2, 2), dtype=config.floatX) * 5
        assert numpy.allclose(11.0 + s.get_value(), fn(xv, yv, zv))

        # grad again the shared variable
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), s)
        fn = function([x, y, z], f)
        assert numpy.allclose(15.0 + s.get_value(),
                              fn(xv, yv, zv))

    def test_connection_pattern(self):
        # Basic case
        x, y, z = T.matrices('xyz')
        out1 = x * y
        out2 = y * z

        op1 = OpFromGraph([x, y, z], [out1, out2])
        results = op1.connection_pattern(None)
        expect_result = [[True, False],
                         [True, True],
                         [False, True]]
        assert results == expect_result

        # Graph with ops that don't have a 'full' connection pattern
        # and with ops that have multiple outputs
        m, n, p, q = T.matrices('mnpq')
        o1, o2 = op1(m, n, p)
        out1, out2 = op1(o1, q, o2)
        op2 = OpFromGraph([m, n, p, q], [out1, out2])

        results = op2.connection_pattern(None)
        expect_result = [[True, False],
                         [True, True],
                         [False, True],
                         [True, True]]
        assert results == expect_result

        # Inner graph where some computation doesn't rely on explicit inputs
        srng = RandomStreams(seed=234)
        rv_u = srng.uniform((2, 2))
        x, y = T.matrices('xy')
        out1 = x + rv_u
        out2 = y + 3
        out3 = 3 + rv_u
        op3 = OpFromGraph([x, y], [out1, out2, out3])

        results = op3.connection_pattern(None)
        expect_result = [[True, False, False],
                         [False, True, False],
                         [True, False, True]]
        assert results == expect_result

    def test_infer_shape(self):
        x = T.matrix('x')
        y = T.matrix('y')
        o1 = x + y
        o2 = x * y
        op_graph = OpFromGraph([x, y], [o1, o2])

        q = T.matrix('q')
        p = T.matrix('p')
        self._compile_and_check([q, p],
                                op_graph(q, p),
                                [numpy.ones([3, 4], dtype=config.floatX),
                                 numpy.ones([3, 4], dtype=config.floatX)],
                                OpFromGraph)
