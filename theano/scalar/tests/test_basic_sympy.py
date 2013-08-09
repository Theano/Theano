from theano.scalar.basic_sympy import SymPyCCode
from theano.scalar.basic import floats
import theano

try:
    import sympy
    xs = sympy.Symbol('x')
    ys = sympy.Symbol('y')
except ImportError:
    sympy = False

xt, yt = floats('xy')

def test_SymPyCCode():
    if not sympy:       return
    op = SymPyCCode([xs, ys], xs + ys)
    e = op(xt, yt)
    g = theano.gof.FunctionGraph([xt, yt], [e])
    fn = theano.gof.CLinker().accept(g).make_function()
    assert fn(1.0, 2.0) == 3.0

def test_grad():
    if not sympy:       return
    op = SymPyCCode([xs], xs**2)
    zt = op(xt)
    ztprime = theano.grad(zt, xt)
    assert ztprime.owner.op.expr == 2*xs

def test_multivar_grad():
    if not sympy:       return
    op = SymPyCCode([xs, ys], xs**2 + ys**2)
    zt = op(xt, yt)
    dzdx, dzdy = theano.grad(zt, [xt, yt])
    assert dzdx.owner.op.expr == 2*xs
    assert dzdy.owner.op.expr == 2*ys
