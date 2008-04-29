
import unittest

from gof import Result, Op, Env
import gof

from scalar import *


def inputs():
    return floats('xyz')


class _test_ScalarOps(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        g = Env([x, y], [e])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 1.5


class _test_composite(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = Env([x, y], [c.out])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 1.5

    def test_with_constants(self):
        x, y, z = inputs()
        e = mul(add(70.0, y), div(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        assert "70.0" in c.op.c_code(c, 'dummy', ['x', 'y'], ['z'], dict(id = 0))
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = Env([x, y], [c.out])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 36.0

    def test_many_outputs(self):
        x, y, z = inputs()
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        C = Composite([x, y, z], [e0, e1, e2])
        c = C.make_node(x, y, z)
        # print c.c_code(['x', 'y', 'z'], ['out0', 'out1', 'out2'], dict(id = 0))
        g = Env([x, y, z], c.outputs)
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0, 3.0) == [6.0, 7.0, 0.5]
        

if __name__ == '__main__':
    unittest.main()




