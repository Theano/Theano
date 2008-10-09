
from basic import _scal_elemwise, _transpose_inplace

@_scal_elemwise
def lt_inplace(a,b):
    """a < b (inplace on a)"""

@_scal_elemwise
def gt_inplace(a,b):
    """a > b (inplace on a)"""

@_scal_elemwise
def le_inplace(a,b):
    """a <= b (inplace on a)"""

@_scal_elemwise
def ge_inplace(a,b):
    """a >= b (inplace on a)"""

@_scal_elemwise
def eq_inplace(a,b):
    """a == b (inplace on a)"""

@_scal_elemwise
def neq_inplace(a,b):
    """a != b (inplace on a)"""

@_scal_elemwise
def and__inplace(a,b):
    """bitwise a & b (inplace on a)"""

@_scal_elemwise
def or__inplace(a,b):
    """bitwise a | b (inplace on a)"""

@_scal_elemwise
def xor_inplace(a,b):
    """bitwise a ^ b (inplace on a)"""

@_scal_elemwise
def invert_inplace(a):
    """bitwise ~a (inplace on a)"""

@_scal_elemwise
def abs__inplace(a):
    """|`a`| (inplace on `a`)"""

@_scal_elemwise
def exp_inplace(a):
    """e^`a` (inplace on `a`)"""

@_scal_elemwise
def neg_inplace(a):
    """-a (inplace on a)"""

@_scal_elemwise
def inv_inplace(a):
    """1.0/a (inplace on a)"""

@_scal_elemwise
def log_inplace(a):
    """base e logarithm of a (inplace on a)"""

@_scal_elemwise
def log2_inplace(a):
    """base 2 logarithm of a (inplace on a)"""

@_scal_elemwise
def sgn_inplace(a):
    """sign of `a` (inplace on `a`)"""

@_scal_elemwise
def sqr_inplace(a):
    """square of `a` (inplace on `a`)"""

@_scal_elemwise
def sqrt_inplace(a):
    """square root of `a` (inplace on `a`)"""

@_scal_elemwise
def cos_inplace(a):
    """cosine of `a` (inplace on `a`)"""

@_scal_elemwise
def sin_inplace(a):
    """sine of `a` (inplace on `a`)"""

@_scal_elemwise
def tan_inplace(a):
    """tangent of `a` (inplace on `a`)"""

@_scal_elemwise
def cosh_inplace(a):
    """hyperbolic cosine of `a` (inplace on `a`)"""

@_scal_elemwise
def sinh_inplace(a):
    """hyperbolic sine of `a` (inplace on `a`)"""

@_scal_elemwise
def tanh_inplace(a):
    """hyperbolic tangent of `a` (inplace on `a`)"""

@_scal_elemwise
def second_inplace(a):
    """Fill `a` with `b`"""

fill_inplace = second_inplace

@_scal_elemwise
def add_inplace(a, b):
    """elementwise addition (inplace on `a`)"""

@_scal_elemwise
def sub_inplace(a, b):
    """elementwise subtraction (inplace on `a`)"""

@_scal_elemwise
def mul_inplace(a, b):
    """elementwise multiplication (inplace on `a`)"""

@_scal_elemwise
def div_inplace(a, b):
    """elementwise division (inplace on `a`)"""

@_scal_elemwise
def mod_inplace(a, b):
    """elementwise modulo (inplace on `a`)"""

@_scal_elemwise
def pow_inplace(a, b):
    """elementwise power (inplace on `a`)"""

transpose_inplace = _transpose_inplace
"""WRITEME"""


