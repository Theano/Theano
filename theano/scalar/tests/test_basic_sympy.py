from theano.scalar.basic_sympy import SymPyCCode
from theano.scalar.basic import floats
from theano.gof import FunctionGraph
from theano import gof
import sympy

def test_SymPyCCode():
    xs = sympy.Symbol('x')
    ys = sympy.Symbol('y')
    op = SymPyCCode([xs, ys], xs + ys)
    xt, yt = floats('xy')
    e = op(xt, yt)
    g = FunctionGraph([xt, yt], [e])
    fn = gof.CLinker().accept(g).make_function()
    assert fn(1.0, 2.0) == 3.0
