

import unittest
from pprint import *
import tensor as T
import gof

class _test_pp(unittest.TestCase):

    def check(self, expr, expected):
        s = pp.process(expr)
        self.failUnless(s == expected, "for (%s), pp produced (%s)" % (expected, s))

    def test_operator_precedence(self):
        x, y, z = T.matrices('xyz')
        self.check(x + y * z, "x + y * z")
        self.check((x + y) * z, "(x + y) * z")
        self.check(x * y ** z, "x * y ** z")
        self.check(z ** x * y, "z ** x * y")
        self.check((x * y) ** z, "(x * y) ** z")
        self.check(z ** (x * y), "z ** (x * y)")

    def test_unary_minus_precedence(self):
        x, y, z = T.matrices('xyz')
        self.check(-x+y, "-x + y")
        self.check(-(x*y), "-(x * y)")
        self.check((-x)*y, "-x * y")
        self.check(x*-y, "x * -y")
        self.check(x/-y, "x / -y")
        self.check(-x**y, "-x ** y")
        self.check((-x)**y, "(-x) ** y")
        self.check(-(x**y), "-x ** y")
        self.check(x**-y, "x ** (-y)")

    def test_parenthesizing(self):
        x, y, z = T.matrices('xyz')
        self.check(x * (y * z), "x * y * z")
        self.check(x / (y / z) / x, "x / (y / z) / x")
        self.check((x ** y) ** z, "(x ** y) ** z")
        self.check(x / (y * z), "x / (y * z)")
        self.check(x * (y / z), "x * y / z")
        self.check(x / y * z, "x / y * z")



if __name__ == '__main__':
    unittest.main()
