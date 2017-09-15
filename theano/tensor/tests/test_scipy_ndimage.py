from __future__ import absolute_import, print_function, division

import unittest
from nose.plugins.skip import SkipTest
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.scipy_ndimage import (zoom, ZoomShiftGrad,
                                         spline_filter1d, spline_filter,
                                         SplineFilter1DGrad)
import theano.tests.unittest_tools as utt

try:
    import scipy.ndimage
    imported_scipy = True
except ImportError:
    # some tests won't work
    imported_scipy = False


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

                    if imported_scipy:
                        # Compare with SciPy function
                        res_ref = scipy.ndimage.spline_filter1d(x_val, order, axis)
                        utt.assert_allclose(res, res_ref)

                    # First-order gradient
                    def fn(x_):
                        return spline_filter1d(x_, order, axis)
                    utt.verify_grad(fn, [x_val])

                    # Second-order gradient
                    if order > 1:
                        def fn_grad(x_):
                            return SplineFilter1DGrad(order, axis)(x_)
                        utt.verify_grad(fn_grad, [x_val])

    def test_spline_filter(self):
        if not imported_scipy:
            raise SkipTest('SciPy ndimage not available')

        x = T.tensor3()
        x_val = np.random.uniform(size=(4, 3, 9)).astype(theano.config.floatX)
        order = 2
        f = theano.function([x], spline_filter(x, order))
        res = f(x_val)
        res_ref = scipy.ndimage.spline_filter(x_val, order)
        utt.assert_allclose(res, res_ref)


class TestZoomShift(unittest.TestCase):
    def test_zoom(self):
        test_cases = (([1, 1],   'constant', 2.0, True),
                      ([0.5, 2], 'constant', 2.0, True),
                      ([0.3, 1], 'nearest',  0.0, True),
                      ([1, 2.3], 'reflect',  0.0, False),
                      ([2, 2],   'mirror',   0.0, False),
                      ([2, 2],   'wrap',     0.0, False))

        x = T.matrix()
        for shape in ((4, 3), (10, 15), (1, 1)):
            x_val = np.random.uniform(size=shape).astype(theano.config.floatX)
            for (zoom_ar, mode, cval, prefilter) in test_cases:
                for order in range(5):
                    # Theano implementation
                    f = theano.function([x], zoom(x, zoom=zoom_ar, order=order, mode=mode,
                                                  cval=cval, prefilter=prefilter))
                    res = f(x_val)

                    if imported_scipy:
                        # Compare with SciPy function
                        res_ref = scipy.ndimage.zoom(x_val, zoom=zoom_ar, order=order, mode=mode,
                                                      cval=cval, prefilter=prefilter)
                        utt.assert_allclose(res, res_ref)

                    if len(res) > 0:
                        # First-order gradient
                        def fn(x_):
                            return zoom(x_, zoom=zoom_ar, order=order, mode=mode,
                                        cval=cval, prefilter=prefilter)
                        utt.verify_grad(fn, [x_val])

                        # The ops work internally use inverted values for zoom_ar.
                        # This is usually handled by the zoom(...) helper,
                        # but we compute it here so we can call ZoomShiftGrad directly.
                        zoom_ar_in_op = [float(ii - 1) / float(oo - 1) if oo > 1 else 1
                                         for ii, oo in zip(shape, res.shape)]

                        # Second-order gradient
                        def fn_grad(y_):
                            return ZoomShiftGrad(order=order, mode=mode)(
                                    y_, x_val.shape, zoom_ar_in_op, None, cval=cval)
                        utt.verify_grad(fn_grad, [res])
