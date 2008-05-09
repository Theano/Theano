
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


class _test_logical(unittest.TestCase):
    def test_gt(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [x > y])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a>b))

    def test_lt(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [x < y])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a<b))

    def test_le(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [x <= y])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a<=b))

    def test_ge(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [x >= y])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a>=b))

    def test_eq(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [eq(x,y)])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a==b))

    def test_neq(self):
        x, y, z = inputs()
        fn = gof.DualLinker(Env([x,y], [neq(x,y)])).make_function()
        for a,b in ((3.,9), (3,0.9), (3,3)):
            self.failUnless(fn(a,b) == (a!=b))


    def test_or(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [x|y])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == (a|b), (a,b))

    def test_xor(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [x^y])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == (a ^ b), (a,b))

    def test_and(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [and_(x, y)])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == (a & b), (a,b))

        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [x & y])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == (a & b), (a,b))
    
    def test_not(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [invert(x)])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == ~a, (a,))

        x, y, z = ints('xyz')
        fn = gof.DualLinker(Env([x,y], [~x])).make_function()
        for a,b in ((0,1), (0,0), (1,0), (1,1)):
            self.failUnless(fn(a,b) == ~a, (a,))

if __name__ == '__main__':
    unittest.main()




