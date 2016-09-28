from __future__ import absolute_import, print_function, division
from theano import scalar as scal
from . import elemwise
from theano import printing
from theano.printing import pprint


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

    # for the meaning of this see the ./epydoc script
    # it makes epydoc display rval as if it were a function, not an object
    rval.__epydoc_asRoutine = symbol
    rval.__module__ = 'theano.tensor.inplace'

    pprint.assign(rval, printing.FunctionPrinter(symbolname.replace('_inplace', '=')))
    return rval


@_scal_inplace
def lt_inplace(a, b):
    """a < b (inplace on a)"""


@_scal_inplace
def gt_inplace(a, b):
    """a > b (inplace on a)"""


@_scal_inplace
def le_inplace(a, b):
    """a <= b (inplace on a)"""


@_scal_inplace
def ge_inplace(a, b):
    """a >= b (inplace on a)"""


@_scal_inplace
def eq_inplace(a, b):
    """a == b (inplace on a)"""


@_scal_inplace
def neq_inplace(a, b):
    """a != b (inplace on a)"""


@_scal_inplace
def and__inplace(a, b):
    """bitwise a & b (inplace on a)"""


@_scal_inplace
def or__inplace(a, b):
    """bitwise a | b (inplace on a)"""


@_scal_inplace
def xor_inplace(a, b):
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
def exp2_inplace(a):
    """2^`a` (inplace on `a`)"""


@_scal_inplace
def expm1_inplace(a):
    """e^`a` - 1 (inplace on `a`)"""


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
def log1p_inplace(a):
    """log(1+a)"""


@_scal_inplace
def log2_inplace(a):
    """base 2 logarithm of a (inplace on a)"""


@_scal_inplace
def log10_inplace(a):
    """base 10 logarithm of a (inplace on a)"""


@_scal_inplace
def sgn_inplace(a):
    """sign of `a` (inplace on `a`)"""


@_scal_inplace
def ceil_inplace(a):
    """ceil of `a` (inplace on `a`)"""


@_scal_inplace
def floor_inplace(a):
    """floor of `a` (inplace on `a`)"""


@_scal_inplace
def trunc_inplace(a):
    """trunc of `a` (inplace on `a`)"""


@_scal_inplace
def round_half_to_even_inplace(a):
    """round_half_to_even_inplace(a) (inplace on `a`)"""


@_scal_inplace
def round_half_away_from_zero_inplace(a):
    """round_half_away_from_zero_inplace(a) (inplace on `a`)"""


@_scal_inplace
def sqr_inplace(a):
    """square of `a` (inplace on `a`)"""


@_scal_inplace
def sqrt_inplace(a):
    """square root of `a` (inplace on `a`)"""


@_scal_inplace
def deg2rad_inplace(a):
    """convert degree `a` to radian(inplace on `a`)"""


@_scal_inplace
def rad2deg_inplace(a):
    """convert radian `a` to degree(inplace on `a`)"""


@_scal_inplace
def cos_inplace(a):
    """cosine of `a` (inplace on `a`)"""


@_scal_inplace
def arccos_inplace(a):
    """arccosine of `a` (inplace on `a`)"""


@_scal_inplace
def sin_inplace(a):
    """sine of `a` (inplace on `a`)"""


@_scal_inplace
def arcsin_inplace(a):
    """arcsine of `a` (inplace on `a`)"""


@_scal_inplace
def tan_inplace(a):
    """tangent of `a` (inplace on `a`)"""


@_scal_inplace
def arctan_inplace(a):
    """arctangent of `a` (inplace on `a`)"""


@_scal_inplace
def arctan2_inplace(a, b):
    """arctangent of `a` / `b` (inplace on `a`)"""


@_scal_inplace
def cosh_inplace(a):
    """hyperbolic cosine of `a` (inplace on `a`)"""


@_scal_inplace
def arccosh_inplace(a):
    """hyperbolic arc cosine of `a` (inplace on `a`)"""


@_scal_inplace
def sinh_inplace(a):
    """hyperbolic sine of `a` (inplace on `a`)"""


@_scal_inplace
def arcsinh_inplace(a):
    """hyperbolic arc sine of `a` (inplace on `a`)"""


@_scal_inplace
def tanh_inplace(a):
    """hyperbolic tangent of `a` (inplace on `a`)"""


@_scal_inplace
def arctanh_inplace(a):
    """hyperbolic arc tangent of `a` (inplace on `a`)"""


@_scal_inplace
def erf_inplace(a):
    """error function"""


@_scal_inplace
def erfc_inplace(a):
    """complementary error function"""


@_scal_inplace
def erfcx_inplace(a):
    """scaled complementary error function"""


@_scal_inplace
def gamma_inplace(a):
    """gamma function"""


@_scal_inplace
def gammaln_inplace(a):
    """log gamma function"""


@_scal_inplace
def psi_inplace(a):
    """derivative of log gamma function"""


@_scal_inplace
def chi2sf_inplace(x, k):
    """chi squared survival function"""


@_scal_inplace
def j0_inplace(a):
    """Bessel function of the 0'th kind"""


@_scal_inplace
def j1_inplace(a):
    """Bessel function of the 0'th kind"""


@_scal_inplace
def second_inplace(a):
    """Fill `a` with `b`"""

fill_inplace = second_inplace
pprint.assign(fill_inplace, printing.FunctionPrinter('fill='))


@_scal_inplace
def maximum_inplace(a, b):
    """elementwise addition (inplace on `a`)"""


@_scal_inplace
def minimum_inplace(a, b):
    """elementwise addition (inplace on `a`)"""


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
def true_div_inplace(a, b):
    """elementwise division (inplace on `a`)"""


@_scal_inplace
def int_div_inplace(a, b):
    """elementwise division (inplace on `a`)"""


@_scal_inplace
def mod_inplace(a, b):
    """elementwise modulo (inplace on `a`)"""


@_scal_inplace
def pow_inplace(a, b):
    """elementwise power (inplace on `a`)"""


@_scal_inplace
def conj_inplace(a):
    """elementwise conjugate (inplace on `a`)"""

pprint.assign(add_inplace, printing.OperatorPrinter('+=', -2, 'either'))
pprint.assign(mul_inplace, printing.OperatorPrinter('*=', -1, 'either'))
pprint.assign(sub_inplace, printing.OperatorPrinter('-=', -2, 'left'))
pprint.assign(neg_inplace, printing.OperatorPrinter('-=', 0, 'either'))
pprint.assign(true_div_inplace, printing.OperatorPrinter('/=', -1, 'left'))
pprint.assign(int_div_inplace, printing.OperatorPrinter('//=', -1, 'left'))
pprint.assign(pow_inplace, printing.OperatorPrinter('**=', 1, 'right'))


def transpose_inplace(x, **kwargs):
    "Perform a transpose on a tensor without copying the underlying storage"
    dims = list(range(x.ndim - 1, -1, -1))
    return elemwise.DimShuffle(x.broadcastable, dims, inplace=True)(x)
