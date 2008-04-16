
import unittest

from gof import Result, Op, Env, modes
import gof

from scalar import *


def inputs():
    x = modes.build(as_scalar(1.0, 'x'))
    y = modes.build(as_scalar(2.0, 'y'))
    z = modes.build(as_scalar(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_ScalarOps(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        g = env([x, y], [e])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 1.5


class _test_composite(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        C = composite([x, y], [e])
        c = C(x, y)
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        c.perform()
        assert c.outputs[0].data == 1.5
        g = env([x, y], [c.out])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 1.5

    def test_with_constants(self):
        x, y, z = inputs()
        e = mul(add(70.0, y), div(x, y))
        C = composite([x, y], [e])
        c = C(x, y)
        assert "70.0" in c.c_code(['x', 'y'], ['z'], dict(id = 0))
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        c.perform()
        assert c.outputs[0].data == 36.0
        g = env([x, y], [c.out])
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0) == 36.0

    def test_many_outputs(self):
        x, y, z = inputs()
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        C = composite([x, y, z], [e0, e1, e2])
        c = C(x, y, z)
        # print c.c_code(['x', 'y', 'z'], ['out0', 'out1', 'out2'], dict(id = 0))
        c.perform()
        assert c.outputs[0].data == 6.0
        assert c.outputs[1].data == 7.0
        assert c.outputs[2].data == 0.5
        g = env([x, y, z], c.outputs)
        fn = gof.DualLinker(g).make_function()
        assert fn(1.0, 2.0, 3.0) == [6.0, 7.0, 0.5]
        

if __name__ == '__main__':
    unittest.main()




