from __future__ import absolute_import, print_function, division

import unittest
import scipy.ndimage
import numpy as np
from numpy.testing import assert_array_equal

import theano
import theano.tensor as T
from theano.tensor.scipy_ndimage import (spline_filter1d, spline_filter,
                                         SplineFilter1DGrad)
from theano.tests.unittest_tools import verify_grad


class TestSplineFilter1D(unittest.TestCase):
    def test_spline_filter1d(self):
        x = T.matrix()
        for shape in ((4, 3), (10, 1)):
            x_val = np.random.uniform(size=shape).astype(theano.config.floatX)
            for order in range(5):
                for axis in (-1, 0, 1):
                    # Theano implementation
                    f = theano.function([x], spline_filter1d(x, order, axis))
                    res = f(x_val)

                    # compare with SciPy function
                    res_ref = scipy.ndimage.spline_filter1d(x_val, order, axis)
                    assert_array_equal(res, res_ref)

                    # first-order gradient
                    def fn(x_):
                        return spline_filter1d(x_, order, axis)
                    verify_grad(fn, [x_val])

                    # second-order gradient
                    if order > 1:
                        def fn_grad(x_):
                            return SplineFilter1DGrad(order, axis)(x_)
                        verify_grad(fn_grad, [x_val])

    def test_spline_filter(self):
        x = T.tensor3()
        x_val = np.random.uniform(size=(4, 3, 9)).astype(theano.config.floatX)
        order = 2
        f = theano.function([x], spline_filter(x, order))
        res = f(x_val)
        res_ref = scipy.ndimage.spline_filter(x_val, order)
        assert_array_equal(res, res_ref)
