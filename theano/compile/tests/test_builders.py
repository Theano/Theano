from __future__ import absolute_import, print_function, division
import numpy as np

from theano import config, shared

from theano.compile import function

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theano.compile.builders import OpFromGraphInline, OpFromGraphPrecompiled

from theano.tests import unittest_tools

test_params = unittest_tools.parameterized.expand(
    [(OpFromGraphInline,), (OpFromGraphPrecompiled,)])


class T_OpFromGraph(unittest_tools.InferShapeTester):

    @test_params
    def test_straightforward(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        fn(xv, yv, zv)
        assert np.all(8.0 == fn(xv, yv, zv))
        assert np.all(8.0 == fn(xv, yv, zv))

    @test_params
    def test_size_changes(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = cls_ofg([x, y], [e])
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv = np.ones((2, 3), dtype=config.floatX)
        yv = np.ones((3, 4), dtype=config.floatX) * 3
        zv = np.ones((4, 5), dtype=config.floatX) * 5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)

    @test_params
    def test_grad(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.all(11.0 == fn(xv, yv, zv))

    @test_params
    def test_grad_grad(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.allclose(6.0, fn(xv, yv, zv))

    @test_params
    def test_shared(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        s = shared(np.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        assert np.allclose(8.0, fn(xv, yv, zv))
        assert np.allclose(8.0, fn(xv, yv, zv))

    @test_params
    def test_shared_grad(self, cls_ofg):
        x, y, z = T.matrices('xyz')
        s = shared(np.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.allclose(11.0 + s.get_value(), fn(xv, yv, zv))

        # grad again the shared variable
        f = op(x, y, z)
        f = f - T.grad(T.sum(f), s)
        fn = function([x, y, z], f)
        assert np.allclose(15.0 + s.get_value(),
                           fn(xv, yv, zv))

    @test_params
    def test_grad_override(self, cls_ofg):
        x, y = T.vectors('xy')

        def go(inps, gs):
            x, y = inps
            g = gs[0]
            return [g*y*2, g*x*1.5]
        # no override case is coverd in "grad" test

        # single override case
        op_mul = cls_ofg([x, y], [x*y], grad_overrides=go)
        xx, yy = T.vector('xx'), T.vector('yy')
        zz = T.sum(op_mul(xx, yy))
        dx, dy = T.grad(zz, [xx, yy])
        fn = function([xx, yy], [dx, dy])
        xv = numpy.random.rand(16).astype(config.floatX)
        yv = numpy.random.rand(16).astype(config.floatX)
        dxv, dyv = fn(xv, yv)
        assert numpy.allclose(yv*2, dxv)
        assert numpy.allclose(xv*1.5, dyv)

        # list override case
        def go1(inps, gs):
            x, w, b = inps
            g = gs[0]
            return g*w*2

        def go2(inps, gs):
            x, w, b = inps
            g = gs[0]
            return g*x*1.5

        w, b = T.vectors('wb')
        # we make the 3rd gradient default (no override)
        op_linear = cls_ofg([x, w, b], [x*w+b], grad_overrides=[go1, go2])
        xx, ww, bb = T.vector('xx'), T.vector('yy'), T.vector('bb')
        zz = T.sum(op_linear(xx, ww, bb))
        dx, dw, db = T.grad(zz, [xx, ww, bb])
        fn = function([xx, ww, bb], [dx, dw, db])
        xv = numpy.random.rand(16).astype(config.floatX)
        wv = numpy.random.rand(16).astype(config.floatX)
        bv = numpy.random.rand(16).astype(config.floatX)
        dxv, dwv, dbv = fn(xv, wv, bv)
        assert numpy.allclose(wv*2, dxv)
        assert numpy.allclose(xv*1.5, dwv)
        assert numpy.allclose(numpy.ones(16, dtype=config.floatX), dbv)

    @test_params
    def test_nested(self, cls_ofg):
        x, y = T.vectors('xy')
        u, v = x+y, x-y
        op_ft = cls_ofg([x, y], [u, v])
        op_ift = cls_ofg([x, y], [u/2, v/2])

        xx, yy = T.vector('xx'), T.vector('yy')
        xx2, yy2 = op_ift(*op_ft(xx, yy))
        fn = function([xx, yy], [xx2, yy2])

        xv = numpy.random.rand(16).astype(config.floatX)
        yv = numpy.random.rand(16).astype(config.floatX)
        xv2, yv2 = fn(xv, yv)
        assert numpy.allclose(xv, xv2)
        assert numpy.allclose(yv, yv2)

    @test_params
    def test_connection_pattern(self, cls_ofg):
        # Basic case
        x, y, z = T.matrices('xyz')
        out1 = x * y
        out2 = y * z

        op1 = cls_ofg([x, y, z], [out1, out2])
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
        op2 = cls_ofg([m, n, p, q], [out1, out2])

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
        op3 = cls_ofg([x, y], [out1, out2, out3])

        results = op3.connection_pattern(None)
        expect_result = [[True, False, False],
                         [False, True, False],
                         [True, False, True]]
        assert results == expect_result

    @test_params
    def test_infer_shape(self, cls_ofg):
        x = T.matrix('x')
        y = T.matrix('y')
        o1 = x + y
        o2 = x * y
        op_graph = cls_ofg([x, y], [o1, o2])

        q = T.matrix('q')
        p = T.matrix('p')
        # we don't want check_topo for inline ops
        # since the inline op is replaced during optimization
        is_compile = not issubclass(cls_ofg, OpFromGraphInline)
        self._compile_and_check([q, p],
                                op_graph(q, p),
                                [np.ones([3, 4], dtype=config.floatX),
                                 np.ones([3, 4], dtype=config.floatX)],
                                cls_ofg,
                                check_topo=is_compile)
