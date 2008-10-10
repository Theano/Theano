
from basic import _scal_elemwise, _transpose_inplace
from .. import scalar as scal
import elemwise

def _scal_inplace(symbol):
    """Replace a symbol definition with an elementwise version of the corresponding scalar Op"""
    symbolname = symbol.__name__
    inplace = symbolname.endswith('_inplace')

    if inplace:
        scalar_op = getattr(scal, symbolname[:-len('_inplace')])
        inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
        rval = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name=symbolname)
    else:
        scalar_op = getattr(scal, symbolname)
        rval = elemwise.Elemwise(scalar_op, name=symbolname)

    if getattr(symbol, '__doc__', False):
        rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

    #for the meaning of this see the ./epydoc script
    # it makes epydoc display rval as if it were a function, not an object
    rval.__epydoc_asRoutine = symbol
    rval.__module__ = 'theano.tensor.inplace'

    return rval


@_scal_inplace
def lt_inplace(a,b):
    """a < b (inplace on a)"""

@_scal_inplace
def gt_inplace(a,b):
    """a > b (inplace on a)"""

@_scal_inplace
def le_inplace(a,b):
    """a <= b (inplace on a)"""

@_scal_inplace
def ge_inplace(a,b):
    """a >= b (inplace on a)"""

@_scal_inplace
def eq_inplace(a,b):
    """a == b (inplace on a)"""

@_scal_inplace
def neq_inplace(a,b):
    """a != b (inplace on a)"""

@_scal_inplace
def and__inplace(a,b):
    """bitwise a & b (inplace on a)"""

@_scal_inplace
def or__inplace(a,b):
    """bitwise a | b (inplace on a)"""

@_scal_inplace
def xor_inplace(a,b):
    """bitwise a ^ b (inplace on a)"""

@_scal_inplace
def invert_inplace(a):
    """bitwise ~a (inplace on a)"""

@_scal_inplace
def abs__inplace(a):
    """|`a`| (inplace on `a`)"""

@_scal_inplace
def exp_inplace(a):
    """e^`a` (inplace on `a`)"""

@_scal_inplace
def neg_inplace(a):
    """-a (inplace on a)"""

@_scal_inplace
def inv_inplace(a):
    """1.0/a (inplace on a)"""

@_scal_inplace
def log_inplace(a):
    """base e logarithm of a (inplace on a)"""

@_scal_inplace
def log2_inplace(a):
    """base 2 logarithm of a (inplace on a)"""

@_scal_inplace
def sgn_inplace(a):
    """sign of `a` (inplace on `a`)"""

@_scal_inplace
def sqr_inplace(a):
    """square of `a` (inplace on `a`)"""

@_scal_inplace
def sqrt_inplace(a):
    """square root of `a` (inplace on `a`)"""

@_scal_inplace
def cos_inplace(a):
    """cosine of `a` (inplace on `a`)"""

@_scal_inplace
def sin_inplace(a):
    """sine of `a` (inplace on `a`)"""

@_scal_inplace
def tan_inplace(a):
    """tangent of `a` (inplace on `a`)"""

@_scal_inplace
def cosh_inplace(a):
    """hyperbolic cosine of `a` (inplace on `a`)"""

@_scal_inplace
def sinh_inplace(a):
    """hyperbolic sine of `a` (inplace on `a`)"""

@_scal_inplace
def tanh_inplace(a):
    """hyperbolic tangent of `a` (inplace on `a`)"""

@_scal_inplace
def second_inplace(a):
    """Fill `a` with `b`"""

fill_inplace = second_inplace

@_scal_inplace
def add_inplace(a, b):
    """elementwise addition (inplace on `a`)"""

@_scal_inplace
def sub_inplace(a, b):
    """elementwise subtraction (inplace on `a`)"""

@_scal_inplace
def mul_inplace(a, b):
    """elementwise multiplication (inplace on `a`)"""

@_scal_inplace
def div_inplace(a, b):
    """elementwise division (inplace on `a`)"""

@_scal_inplace
def mod_inplace(a, b):
    """elementwise modulo (inplace on `a`)"""

@_scal_inplace
def pow_inplace(a, b):
    """elementwise power (inplace on `a`)"""

transpose_inplace = _transpose_inplace
"""WRITEME"""


