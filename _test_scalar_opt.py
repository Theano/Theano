

import unittest

from gof import Result, Op, Env, modes
import gof

from scalar import *
from scalar_opt import *


def inputs():
    x = Scalar('float64', name = 'x')
    y = Scalar('float64', name = 'y')
    z = Scalar('float64', name = 'z')
    a = Scalar('float64', name = 'a')
    return x, y, z

def more_inputs():
    a = Scalar('float64', name = 'a')
    b = Scalar('float64', name = 'b')
    c = Scalar('float64', name = 'c')
    d = Scalar('float64', name = 'd')
    return a, b, c, d


class _test_opts(unittest.TestCase):

    def test_pow_to_sqr(self):
        x, y, z = inputs()
        e = x ** 2.0
        g = Env([x], [e])
        assert str(g) == "[Pow(x, 2.0)]"
        gof.ConstantFinder().optimize(g)
        pow2sqr_float.optimize(g)
        assert str(g) == "[Sqr(x)]"


# class _test_canonize(unittest.TestCase):

#     def test_muldiv(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
# #        e = (2.0 * x) / (2.0 * y)
# #        e = (2.0 * x) / (4.0 * y)
# #        e = x / (y / z)
# #        e = (x * y) / x
# #        e = (x / y) * (y / z) * (z / x)
# #        e = (a / b) * (b / c) * (c / d)
# #        e = (a * b) / (b * c) / (c * d)
# #        e = 2 * x / 2
# #        e = x / y / x
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn).optimize(g)
#         print g
        
#     def test_plusmin(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
# #        e = x - x
# #        e = (2.0 + x) - (2.0 + y)
# #        e = (2.0 + x) - (4.0 + y)
# #        e = x - (y - z)
# #        e = (x + y) - x
# #        e = (x - y) + (y - z) + (z - x)
# #        e = (a - b) + (b - c) + (c - d)
# #        e = x + -y
# #        e = a - b - b + a + b + c + b - c
#         e = x + log(y) - x + y
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_both(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
#         e0 = (x * y / x)
#         e = e0 + e0 - e0
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn).optimize(g)
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_group_powers(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
# #        e = x * exp(y) * exp(z)
# #        e = x * pow(x, y) * pow(x, z)
# #        e = pow(x, y) / pow(x, z)
# #        e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0)
# #        e = pow(x - x, y)
# #        e = pow(x, 2.0 + y - 7.0)
# #        e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) / pow(x, z)
# #        e = pow(x, 2.0 + y - 7.0 - z)
# #        e = x ** y / x ** y
# #        e = x ** y / x ** (y - 1.0)
#         e = exp(x) * a * exp(y) / exp(z)
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn, group_powers).optimize(g)
#         print g
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g
#         pow2one_float.optimize(g)
#         pow2x_float.optimize(g)
#         print g
        


if __name__ == '__main__':
    unittest.main()

