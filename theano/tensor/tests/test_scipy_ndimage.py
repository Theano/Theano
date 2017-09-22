from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.scipy_ndimage import (zoom, shift,
                                         ZoomShift, ZoomShiftGrad,
                                         spline_filter1d, spline_filter,
                                         SplineFilter1D,
                                         SplineFilter1DGrad)
import theano.tests.unittest_tools as utt

try:
    import scipy.ndimage
    imported_scipy = True
except ImportError:
    # some tests won't work
    imported_scipy = False


class TestSplineFilter1D(utt.InferShapeTester):
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

                    if theano.config.mode != 'FAST_COMPILE':
                        # First-order gradient
                        def fn(x_):
                            return spline_filter1d(x_, order, axis)
                        utt.verify_grad(fn, [x_val])

                        # Second-order gradient
                        if order > 1:
                            def fn_grad(x_):
                                return SplineFilter1DGrad(order, axis)(x_)
                            utt.verify_grad(fn_grad, [x_val])

    def test_spline_filter1d_infer_shape(self):
        x = T.matrix()
        x_val = np.random.uniform(size=(4, 3)).astype(theano.config.floatX)
        self._compile_and_check([x],
                                [spline_filter1d(x, order=2, axis=0)],
                                [x_val], SplineFilter1D)
        if theano.config.mode != 'FAST_COMPILE':
            self._compile_and_check([x],
                                    [SplineFilter1DGrad(order=2, axis=0)(x)],
                                    [x_val], SplineFilter1DGrad)

    def test_spline_filter(self):
        if not imported_scipy:
            raise SkipTest('SciPy ndimage not available')

        x = T.tensor3()
        x_val = np.random.uniform(size=(4, 3, 9)).astype(theano.config.floatX)
        order = 2
        y = spline_filter(x, order)
        f = theano.function([x], y)
        res = f(x_val)
        res_ref = scipy.ndimage.spline_filter(x_val, order)
        utt.assert_allclose(res, res_ref)

        if theano.config.mode != 'FAST_COMPILE':
            # some of the ops should be inplace
            nodes = [n for n in f.maker.fgraph.toposort()
                     if isinstance(n.op, SplineFilter1D)]
            assert any(n.op.inplace for n in nodes)

        y_grad = T.tensor3()
        x_grad = T.grad(None, wrt=x, known_grads={y: y_grad})
        fg = theano.function([y_grad], x_grad)

        if theano.config.mode != 'FAST_COMPILE':
            # some of the ops should be inplace
            nodes = [n for n in fg.maker.fgraph.toposort()
                     if isinstance(n.op, SplineFilter1DGrad)]
            assert any(n.op.inplace for n in nodes)


class TestZoomShift(utt.InferShapeTester):
    def _compute_zoom_for_op(self, input_shape, output_shape):
        # compute the internal zoom parameters for ZoomShift and ZoomShiftGrad
        return [float(ii - 1) / float(oo - 1) if oo > 1 else 1
                for ii, oo in zip(input_shape, output_shape)]

    def test_zoom(self):
        test_cases = (([1, 1], 'constant', 2.0, True),
                      ([0.5, 2], 'constant', 2.0, True),
                      ([0.3, 1], 'nearest', 0.0, True),
                      ([1, 2.3], 'reflect', 0.0, False),
                      (2, 'mirror', 0.0, False),
                      (2, 'wrap', 0.0, False))

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
                        # Recompute the zoom factors to avoid Python 2.7/3 rounding differences
                        adjusted_zoom_ar = (np.round(np.array(x_val.shape).astype('float64') *
                                                     np.array(zoom_ar).astype('float64')) /
                                            np.array(x_val.shape).astype('float64'))

                        # Compare with SciPy function
                        res_ref = scipy.ndimage.zoom(x_val, zoom=adjusted_zoom_ar,
                                                     order=order, mode=mode,
                                                     cval=cval, prefilter=prefilter)
                        utt.assert_allclose(res, res_ref)

                    if len(res) > 0 and theano.config.mode != 'FAST_COMPILE':
                        # First-order gradient
                        def fn(x_):
                            # verify_grad makes the any axis with length == 1 broadcastable
                            x_ = T.patternbroadcast(x_, (False,) * x_.ndim)
                            return zoom(x_, zoom=zoom_ar, order=order, mode=mode,
                                        cval=cval, prefilter=prefilter)
                        utt.verify_grad(fn, [x_val])

                        # The ops internally use inverted values for zoom_ar.
                        # This is usually handled by the zoom(...) helper,
                        # but we compute it here so we can call ZoomShiftGrad directly.
                        zoom_ar_in_op = self._compute_zoom_for_op(shape, res.shape)

                        # Second-order gradient
                        def fn_grad(y_):
                            # verify_grad makes the any axis with length == 1 broadcastable
                            y_ = T.patternbroadcast(y_, (False,) * y_.ndim)
                            return ZoomShiftGrad(order=order, mode=mode)(
                                y_, x_val.shape, zoom_ar_in_op, None, cval=cval)
                        utt.verify_grad(fn_grad, [res])

    def test_zoom_axis(self):
        x = T.tensor4()
        shape = (4, 3, 2, 5)
        zoom_ar = [2, 3]
        axes = [1, 3]
        x_val = np.random.uniform(size=shape).astype(theano.config.floatX)

        # compute result
        y = zoom(x.dimshuffle(0, 1, 2, 3, 'x'), zoom=zoom_ar, order=2, axes=axes)
        f = theano.function([x], y.dimshuffle(0, 1, 2, 3))
        res = f(x_val)

        # reference: loop over all images
        zoom_factors = [1] * len(shape)
        for axis, zoom_factor in zip(axes, zoom_ar):
            zoom_factors[axis] = zoom_factor
        expected_shape = [ss * ff for ss, ff in zip(shape, zoom_factors)]
        no_zoom_axes = [axis for axis in range(len(shape)) if axis not in axes]

        img = T.TensorType(theano.config.floatX, ((False,) * len(no_zoom_axes)))()
        f_single = theano.function([img], zoom(img, zoom=zoom_ar, order=2))

        res_ref = np.zeros(expected_shape)
        # iterate over the images
        for no_zoom_idx in np.ndindex(*[shape[axis] for axis in no_zoom_axes]):
            # find out where this image lives
            image_slice = [slice(None)] * len(shape)
            for axis, idx in zip(no_zoom_axes, no_zoom_idx):
                image_slice[axis] = idx
            # process single image
            res_ref[image_slice] = f_single(x_val[image_slice])

        # compare with reference output
        utt.assert_allclose(res, res_ref)

        if len(res) > 0 and theano.config.mode != 'FAST_COMPILE':
            # First-order gradient
            def fn(x_):
                return zoom(x_.dimshuffle(0, 1, 2, 3, 'x'), zoom=zoom_ar,
                            order=2, axes=axes).dimshuffle(0, 1, 2, 3)
            utt.verify_grad(fn, [x_val])

    def test_zoom_single_axis(self):
        x = T.matrix()
        zoom_ar = [1.5]
        out = zoom(x, zoom_ar, axes=1)
        self.assertEqual(out.owner.op.axes, [1])

    def test_no_zoom_on_broadcastable_axis(self):
        x = T.vector()
        x_bc = x.dimshuffle(['x', 0])
        self.assertRaises(ValueError, T.scipy_ndimage.zoom, x_bc, [2, 2])
        self.assertRaises(ValueError, T.scipy_ndimage.zoom, x_bc, [2], axes=[0])

    def test_zoom_infer_shape(self):
        x = T.matrix()
        y = T.matrix()
        x_val = np.random.uniform(size=(4, 3)).astype(theano.config.floatX)
        for zoom_ar in ([1, 1], [2, 2], [0.5, 0.3]):
            # test shape of forward op
            self._compile_and_check([x],
                                    [zoom(x, zoom=zoom_ar, order=0)],
                                    [x_val], ZoomShift)

            if theano.config.mode != 'FAST_COMPILE':
                # compute the output shape of the forward op
                y_val_shape = zoom(x, zoom=zoom_ar, order=0).shape.eval({x: x_val})
                y_val = np.random.uniform(size=y_val_shape).astype(theano.config.floatX)

                # test shape of gradient
                zoom_ar_in_op = self._compute_zoom_for_op(x_val.shape, y_val.shape)
                self._compile_and_check([y],
                                        [ZoomShiftGrad(order=0)(y, x_val.shape,
                                                                zoom_ar_in_op, None)],
                                        [y_val], ZoomShiftGrad)

    def test_shift(self):
        test_cases = (([1, 1], 'constant', 2.0, True),
                      ([-0.5, 2], 'constant', 2.0, True),
                      ([-0.3, 1], 'nearest', 0.0, True),
                      ([1, -2.3], 'reflect', 0.0, False),
                      (2, 'mirror', 0.0, False),
                      (2, 'wrap', 0.0, False))

        x = T.matrix()
        for shape in ((4, 3), (10, 15), (1, 1)):
            x_val = np.random.uniform(size=shape).astype(theano.config.floatX)
            for (shift_ar, mode, cval, prefilter) in test_cases:
                for order in range(5):
                    # Theano implementation
                    f = theano.function([x], shift(x, shift=shift_ar, order=order, mode=mode,
                                                   cval=cval, prefilter=prefilter))
                    res = f(x_val)

                    if imported_scipy:
                        # Compare with SciPy function
                        res_ref = scipy.ndimage.shift(x_val, shift=shift_ar,
                                                      order=order, mode=mode,
                                                      cval=cval, prefilter=prefilter)
                        utt.assert_allclose(res, res_ref)

                    if len(res) > 0 and theano.config.mode != 'FAST_COMPILE':
                        # First-order gradient
                        def fn(x_):
                            # verify_grad makes the any axis with length == 1 broadcastable
                            x_ = T.patternbroadcast(x_, (False,) * x_.ndim)
                            return shift(x_, shift=shift_ar, order=order, mode=mode,
                                         cval=cval, prefilter=prefilter)
                        utt.verify_grad(fn, [x_val])

                        # The ops internally use negated values for shift_ar.
                        # This is usually handled by the shift(...) helper,
                        # but we compute it here so we can call ZoomShiftGrad directly.
                        if isinstance(shift_ar, list):
                            shift_ar_in_op = -np.array(shift_ar)
                        else:
                            shift_ar_in_op = -np.array([shift_ar] * x_val.ndim)

                        # Second-order gradient
                        def fn_grad(y_):
                            # verify_grad makes the any axis with length == 1 broadcastable
                            y_ = T.patternbroadcast(y_, (False,) * y_.ndim)
                            return ZoomShiftGrad(order=order, mode=mode)(
                                y_, x_val.shape, None, shift_ar_in_op, cval=cval)
                        utt.verify_grad(fn_grad, [res])

    def test_shift_axis(self):
        x = T.tensor4()
        shape = (4, 3, 2, 5)
        shift_ar = [2, 3]
        axes = [1, 3]
        x_val = np.random.uniform(size=shape).astype(theano.config.floatX)

        # compute result
        y = shift(x.dimshuffle(0, 1, 2, 3, 'x'), shift=shift_ar, order=2, axes=axes)
        f = theano.function([x], y.dimshuffle(0, 1, 2, 3))
        res = f(x_val)

        # reference: loop over all images
        no_shift_axes = [axis for axis in range(len(shape)) if axis not in axes]

        img = T.TensorType(theano.config.floatX, ((False,) * len(no_shift_axes)))()
        f_single = theano.function([img], shift(img, shift=shift_ar, order=2))

        res_ref = np.zeros(x_val.shape)
        # iterate over the images
        for no_shift_idx in np.ndindex(*[shape[axis] for axis in no_shift_axes]):
            # find out where this image lives
            image_slice = [slice(None)] * len(shape)
            for axis, idx in zip(no_shift_axes, no_shift_idx):
                image_slice[axis] = idx
            # process single image
            res_ref[image_slice] = f_single(x_val[image_slice])

        # compare with reference output
        utt.assert_allclose(res, res_ref)

        if len(res) > 0 and theano.config.mode != 'FAST_COMPILE':
            # First-order gradient
            def fn(x_):
                return shift(x_.dimshuffle(0, 1, 2, 3, 'x'), shift=shift_ar,
                             order=2, axes=axes).dimshuffle(0, 1, 2, 3)
            utt.verify_grad(fn, [x_val])

    def test_shift_single_axis(self):
        x = T.matrix()
        shift_ar = [1.5]
        out = shift(x, shift_ar, axes=1)
        self.assertEqual(out.owner.op.axes, [1])

    def test_shift_infer_shape(self):
        x = T.matrix()
        y = T.matrix()
        x_val = np.random.uniform(size=(4, 3)).astype(theano.config.floatX)
        shift_ar = [-0.5, 1]
        # test shape of forward op
        self._compile_and_check([x],
                                [shift(x, shift=shift_ar, order=0)],
                                [x_val], ZoomShift)

        if theano.config.mode != 'FAST_COMPILE':
            # compute the output shape of the forward op
            y_val_shape = shift(x, shift=shift_ar, order=0).shape.eval({x: x_val})
            y_val = np.random.uniform(size=y_val_shape).astype(theano.config.floatX)

            # test shape of gradient
            shift_ar_in_op = [-s for s in shift_ar]
            self._compile_and_check([y],
                                    [ZoomShiftGrad(order=0)(y, x_val.shape,
                                                            None, shift_ar_in_op)],
                                    [y_val], ZoomShiftGrad)
