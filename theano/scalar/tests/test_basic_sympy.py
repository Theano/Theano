from __future__ import absolute_import, print_function, division

import theano
from theano.scalar.basic_sympy import SymPyCCode
from theano.scalar.basic import floats

from nose.plugins.skip import SkipTest

try:
    import sympy
    xs = sympy.Symbol('x')
    ys = sympy.Symbol('y')
except ImportError:
    raise SkipTest('optional package sympy disabled')

xt, yt = floats('xy')


def test_SymPyCCode():
    if not theano.config.cxx:
        raise SkipTest("Need cxx for this test")

    op = SymPyCCode([xs, ys], xs + ys)
    e = op(xt, yt)
    g = theano.gof.FunctionGraph([xt, yt], [e])
    fn = theano.gof.CLinker().accept(g).make_function()
    assert fn(1.0, 2.0) == 3.0


def test_grad():
    op = SymPyCCode([xs], xs**2)
    zt = op(xt)
    ztprime = theano.grad(zt, xt)
    assert ztprime.owner.op.expr == 2 * xs


def test_multivar_grad():
    op = SymPyCCode([xs, ys], xs ** 2 + ys ** 3)
    zt = op(xt, yt)
    dzdx, dzdy = theano.grad(zt, [xt, yt])
    assert dzdx.owner.op.expr == 2 * xs
    assert dzdy.owner.op.expr == 3 * ys ** 2
