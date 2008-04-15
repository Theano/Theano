

import unittest

from gof import Result, Op, Env, modes
import gof

from scalar import *
from scalar_opt import *


def inputs():
    x = Scalar('float64', name = 'x')
    y = Scalar('float64', name = 'y')
    z = Scalar('float64', name = 'z')
    return x, y, z


class _test_opts(unittest.TestCase):

    def test_pow_to_sqr(self):
        x, y, z = inputs()
        e = x ** 2.0
        g = Env([x], [e])
        assert str(g) == "[Pow(x, 2.0)]"
        gof.ConstantFinder().optimize(g)
        opt2.optimize(g)
        assert str(g) == "[Sqr(x)]"


if __name__ == '__main__':
    unittest.main()
