"""
These routines are not well-tested. They are also old.
OB says that it is not important to test them well because Scalar Ops
are rarely used by themselves, instead they are the basis for Tensor Ops
(which should be checked thoroughly). Moreover, Scalar will be changed
to use numpy's scalar routines.
If you do want to rewrite these tests, bear in mind:
  * You don't need to use Composite.
  * FunctionGraph and DualLinker are old, use compile.function instead.
"""


from __future__ import absolute_import, print_function, division

import unittest
import numpy as np

import theano
from theano.gof import FunctionGraph
from theano import gof
from theano.tests import unittest_tools as utt

from theano.scalar.basic import (floats, float16, float32, float64,
                                 ints, int8, int32, complex64, uint8,
                                 ComplexError, IntDiv, TrueDiv,
                                 Composite, add, div_proxy,
                                 and_, eq, neq, invert, mul, Scalar, InRange,
                                 cast, constant, switch)
from theano.scalar.basic import (
    true_div, inv, log, log2, log10, log1p, exp, exp2, expm1, sqrt, deg2rad,
    rad2deg, cos, arccos, sin, arcsin, tan, arctan, arctan2, cosh, arccosh,
    sinh, arcsinh, tanh, arctanh)


def inputs():
    return floats('xyz')


class test_ScalarOps(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div_proxy(x, y))
        g = FunctionGraph([x, y], [e])
        fn = gof.DualLinker().accept(g).make_function()
        assert fn(1.0, 2.0) == 1.5

    # This test is moved to theano.tensor.tests.test_basic.py:test_mod
    # We move it their as under ubuntu the c_extract call of theano.scalar
    # call PyInt_check and it fail under some os. If work in other case.
    # As we use theano.scalar normally, but we use theano.tensor.scalar
    # that is not important. Also this make the theano fct fail at call time
    # so this is not a silent bug.
    # --> This is why it is purposedly named 'tes_mod' instead of 'test_mod'.
    def tes_mod(self):
        """
        We add this test as not all language and C implementation give the same
        signe to the result. This check that the c_code of `Mod` is implemented
        as Python. That is what we want.
        """
        x, y = ints('xy')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x % y])).make_function()
        for a, b in ((0, 1), (1, 1), (0, -1), (1, -1), (-1, -1),
                     (1, 2), (-1, 2), (1, -2), (-1, -2),
                     (5, 3), (-5, 3), (5, -3), (-5, -3)
                     ):
            self.assertTrue(fn(a, b) == a % b, (a,))


def has_f16(comp):
    if any(v.type == float16 for v in comp.fgraph.variables):
        return True
    return False


class test_composite(unittest.TestCase):
    def test_composite_clone_float32(self):
        w = int8()
        x = float16()
        y = float32()
        cz = Composite([x, y], [tanh(x + cast(y, 'float16'))])
        c = Composite([w, x, y], [cz(x, y) - cz(x, y)**2 +
                                  cast(x, 'int16') + cast(x, 'float32') +
                                  cast(w, 'float16') -
                                  constant(np.float16(1.0))])
        assert has_f16(c)
        nc = c.clone_float32()
        assert not has_f16(nc)

        v = uint8()
        w = float16()
        x = float16()
        y = float16()
        z = float16()

        c = Composite([v, w, x, y, z], [switch(v, mul(w, x, y), z)])

        assert has_f16(c)
        nc = c.clone_float32()
        assert not has_f16(nc)

    def test_straightforward(self):
        x, y, z = inputs()
        e = mul(add(x, y), div_proxy(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = FunctionGraph([x, y], [c.out])
        fn = gof.DualLinker().accept(g).make_function()
        assert fn(1.0, 2.0) == 1.5

    def test_flatten(self):
        # Test that we flatten multiple Composite.
        x, y, z = inputs()
        C = Composite([x, y], [x + y])
        CC = Composite([x, y], [C(x * y, y)])
        assert not isinstance(CC.outputs[0].owner.op, Composite)

        # Test with multiple outputs
        CC = Composite([x, y, z], [C(x * y, y), C(x * z, y)])
        # We don't flatten that case.
        assert isinstance(CC.outputs[0].owner.op, Composite)

    def test_with_constants(self):
        x, y, z = inputs()
        e = mul(add(70.0, y), div_proxy(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        assert "70.0" in c.op.c_code(c, 'dummy', ['x', 'y'], ['z'], dict(id=0))
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = FunctionGraph([x, y], [c.out])
        fn = gof.DualLinker().accept(g).make_function()
        assert fn(1.0, 2.0) == 36.0

    def test_many_outputs(self):
        x, y, z = inputs()
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        C = Composite([x, y, z], [e0, e1, e2])
        c = C.make_node(x, y, z)
        # print c.c_code(['x', 'y', 'z'], ['out0', 'out1', 'out2'], dict(id = 0))
        g = FunctionGraph([x, y, z], c.outputs)
        fn = gof.DualLinker().accept(g).make_function()
        assert fn(1.0, 2.0, 3.0) == [6.0, 7.0, 0.5]

    def test_composite_printing(self):
        x, y, z = floats('xyz')
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        e3 = x // 5
        e4 = -x
        e5 = x - y
        e6 = x ** y + (-z)
        e7 = x % 3
        C = Composite([x, y, z], [e0, e1, e2, e3, e4, e5, e6, e7])
        c = C.make_node(x, y, z)
        g = FunctionGraph([x, y, z], c.outputs)
        gof.DualLinker().accept(g).make_function()

        assert str(g) == ('[*1 -> Composite{((i0 + i1) + i2),'
                          ' (i0 + (i1 * i2)), (i0 / i1), '
                          '(i0 // Constant{5}), '
                          '(-i0), (i0 - i1), ((i0 ** i1) + (-i2)),'
                          ' (i0 % Constant{3})}(x, y, z), '
                          '*1::1, *1::2, *1::3, *1::4, *1::5, *1::6, *1::7]')

    def test_make_node_continue_graph(self):
        # This is a test for a bug (now fixed) that disabled the
        # local_gpu_elemwise_0 optimization and printed an
        # optimization warning on the terminal.

        # We test that Composite.make_node accept as inputs Variable
        # some that represent existing computation.

        si0 = theano.scalar.int8()
        si1 = theano.scalar.int8()
        si2 = theano.scalar.float32()
        sout = (si0 * si1) / si2
        sop = theano.scalar.Composite([si0, si1, si2],
                                      [sout])
        si0 = theano.scalar.int8()
        si1 = theano.scalar.int8()
        si2 = theano.scalar.float32()
        si3 = theano.scalar.float32()
        sop.make_node(si0 * si3, si1, si2)


class test_logical(unittest.TestCase):
    def test_gt(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x > y])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a > b))

    def test_lt(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x < y])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a < b))

    def test_le(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x <= y])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a <= b))

    def test_ge(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x >= y])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a >= b))

    def test_eq(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [eq(x, y)])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a == b))

    def test_neq(self):
        x, y, z = inputs()
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [neq(x, y)])).make_function()
        for a, b in ((3., 9), (3, 0.9), (3, 3)):
            self.assertTrue(fn(a, b) == (a != b))

    def test_or(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x | y])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == (a | b), (a, b))

    def test_xor(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x ^ y])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == (a ^ b), (a, b))

    def test_and(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [and_(x, y)])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == (a & b), (a, b))

        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [x & y])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == (a & b), (a, b))

    def test_not(self):
        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [invert(x)])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == ~a, (a,))

        x, y, z = ints('xyz')
        fn = gof.DualLinker().accept(FunctionGraph([x, y], [~x])).make_function()
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            self.assertTrue(fn(a, b) == ~a, (a,))


# This class does not inherit from unittest.TestCase, because it would
# interfere with the "yield" mechanism that automatically generates test, see
# http://stackoverflow.com/questions/6689537/nose-test-generators-inside-class
# Therefore, it needs to be named "test_..." or "Test_...", so nose can pick
# it up by name, otherwise the tests would not be executed.
class test_upgrade_to_float(object):
    # Test for Ops whose output has to be floating point, even when all
    # inputs are ints.
    # In particular, when the inputs are int8, the output should be
    # at least float32, not float16.

    unary_ops_vals = [
        (inv, list(range(-127, 0)) + list(range(1, 127))),
        (sqrt, list(range(0, 128))),
        (log, list(range(1, 128))),
        (log2, list(range(1, 128))),
        (log10, list(range(1, 128))),
        (log1p, list(range(0, 128))),
        (exp, list(range(-127, 89))),
        (exp2, list(range(-127, 89))),
        (expm1, list(range(-127, 89))),
        (deg2rad, list(range(-127, 128))),
        (rad2deg, list(range(-127, 128))),
        (cos, list(range(-127, 128))),
        (arccos, list(range(-1, 2))),
        (cosh, list(range(-89, 90))),
        (arccosh, list(range(1, 128))),
        (sin, list(range(-127, 128))),
        (arcsin, list(range(-1, 2))),
        (sinh, list(range(-89, 90))),
        (arcsinh, list(range(-127, 128))),
        (tan, list(range(-3, 4))),
        (arctan, list(range(-127, 128))),
        (tanh, list(range(-127, 128))),
        (arctanh, [0])]

    binary_ops_vals = [
        (arctan2, list(range(-127, 128)), list(range(-127, 128)))]

    @staticmethod
    def _test_unary(unary_op, x_range):
        xi = int8('xi')
        xf = float32('xf')

        ei = unary_op(xi)
        fi = theano.function([xi], ei)

        ef = unary_op(xf)
        ff = theano.function([xf], ef)

        for x_val in x_range:
            outi = fi(x_val)
            outf = ff(x_val)

            assert outi.dtype == outf.dtype, 'incorrect dtype'
            assert np.allclose(outi, outf), 'insufficient precision'

    @staticmethod
    def _test_binary(binary_op, x_range, y_range):
        xi = int8('xi')
        yi = int8('yi')
        xf = float32('xf')
        yf = float32('yf')

        ei = binary_op(xi, yi)
        fi = theano.function([xi, yi], ei)

        ef = binary_op(xf, yf)
        ff = theano.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                assert outi.dtype == outf.dtype, 'incorrect dtype'
                assert np.allclose(outi, outf), 'insufficient precision'

    def test_true_div(self):
        # true_div's upcast policy is not exactly "upgrade_to_float",
        # so the test is a little bit different
        x_range = list(range(-127, 128))
        y_range = list(range(-127, 0)) + list(range(1, 127))

        xi = int8('xi')
        yi = int8('yi')
        xf = Scalar(theano.config.floatX)('xf')
        yf = Scalar(theano.config.floatX)('yf')

        ei = true_div(xi, yi)
        fi = theano.function([xi, yi], ei)

        ef = true_div(xf, yf)
        ff = theano.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                assert outi.dtype == outf.dtype, 'incorrect dtype'
                assert np.allclose(outi, outf), 'insufficient precision'

    def test_unary(self):
        # Automatically define all individual unary tests
        for unary_op, x_range in self.unary_ops_vals:
            test_name = 'test_%s' % unary_op.name

            def test():
                self._test_unary(unary_op, x_range)
            test.description = test_name
            yield test

    def test_binary(self):
        # Automatically define all individual binary tests
        for binary_op, x_range, y_range in self.binary_ops_vals:
            test_name = 'test_%s' % binary_op.name

            def test():
                self._test_binary(binary_op, x_range, y_range)
            test.description = test_name
            yield test


class test_complex_mod(unittest.TestCase):
    """Make sure % fails on complex numbers."""

    def test_fail(self):
        x = complex64()
        y = int32()
        try:
            x % y
            assert False
        except ComplexError:
            pass


class test_div(unittest.TestCase):
    def test_0(self):
        a = int8()
        b = int32()
        c = complex64()
        d = float64()
        f = float32()

        assert isinstance((a // b).owner.op, IntDiv)
        assert isinstance((b // a).owner.op, IntDiv)
        assert isinstance((b / d).owner.op, TrueDiv)
        assert isinstance((b / f).owner.op, TrueDiv)
        assert isinstance((f / a).owner.op, TrueDiv)
        assert isinstance((d / b).owner.op, TrueDiv)
        assert isinstance((d / f).owner.op, TrueDiv)
        assert isinstance((f / c).owner.op, TrueDiv)
        assert isinstance((a / c).owner.op, TrueDiv)


def test_grad_gt():
    x = float32(name='x')
    y = float32(name='y')
    z = x > y
    g = theano.gradient.grad(z, y)
    assert g.eval({y: 1.}) == 0.


def test_grad_switch():

    # This is a code snippet from the mailing list
    # It caused an assert to be raised due to the
    # switch op's grad method not handling integer
    # inputs correctly

    x = theano.tensor.matrix()
    c = theano.tensor.matrix()

    s = theano.tensor.switch(c, x, 0)
    l = s.sum()

    theano.gradient.grad(l, x)


def test_grad_identity():
    # Check that the grad method of Identity correctly handles int dytpes
    x = theano.tensor.imatrix('x')
    # tensor_copy is Elemwise{Identity}
    y = theano.tensor.tensor_copy(x)
    l = y.sum(dtype=theano.config.floatX)
    theano.gradient.grad(l, x)


def test_grad_inrange():
    for bound_definition in [(True, True), (False, False)]:
        # Instantiate op, and then take the gradient
        op = InRange(*bound_definition)
        x = theano.tensor.fscalar('x')
        low = theano.tensor.fscalar('low')
        high = theano.tensor.fscalar('high')
        out = op(x, low, high)
        gx, glow, ghigh = theano.tensor.grad(out, [x, low, high])

        # We look if the gradient are equal to zero
        # if x is lower than the lower bound,
        # equal to the lower bound, between lower and higher bound,
        # equal to the higher bound and higher than the higher
        # bound.
        # Mathematically we should have an infinite gradient when
        # x is equal to the lower or higher bound but in that case
        # Theano defines the gradient to be zero for stability.
        f = theano.function([x, low, high], [gx, glow, ghigh])
        utt.assert_allclose(f(0, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(1, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(2, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(5, 1, 5), [0, 0, 0])
        utt.assert_allclose(f(7, 1, 5), [0, 0, 0])


def test_grad_abs():
    a = theano.tensor.fscalar("a")
    b = theano.tensor.nnet.relu(a)
    c = theano.grad(b, a)
    f = theano.function([a], c, mode=theano.Mode(optimizer=None))
    # Currently Theano return 0.5, but it isn't sure it won't change
    # in the futur.
    ret = f(0.)
    assert ret == 0.5, ret

# Testing of Composite is done in tensor/tests/test_opt.py
# in test_fusion, TestCompositeCodegen


def test_constant():
    c = constant(2, name='a')
    assert c.name == 'a'
    assert c.dtype == 'int8'
    c = constant(2, dtype='float32')
    assert c.name is None
    assert c.dtype == 'float32'


if __name__ == '__main__':
    unittest.main()
