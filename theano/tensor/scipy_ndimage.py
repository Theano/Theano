from __future__ import absolute_import, print_function, division

import os.path

import numpy as np

import theano
from theano.tensor import basic as T

try:
    import scipy.ndimage
    imported_scipy = True
except ImportError:
    # some tests won't work
    imported_scipy = False


def scipy_ndimage_helper_inc_dir():
    return os.path.join(os.path.dirname(__file__), 'c_code/scipy_ndimage')


def _normalize_zoomshift_axes_list(axes, ndim=None):
    if axes is None:
        return []
    if not isinstance(axes, (tuple, list)):
        raise ValueError('axes should be a tuple or list of ints')
    axes = [int(axis) for axis in axes]
    if sorted(set(axes)) != axes:
        raise ValueError('axes should be given in ascending order')
    if ndim is not None:
        if len(axes) > ndim:
            raise ValueError('axes is longer than the number of input dimensions')
        for axis in axes:
            if axis < 0 or axis > ndim:
                raise ValueError('invalid axis: %d' % axis)
    return axes


ZoomShiftMode = theano.gof.EnumList(('NI_NEAREST', 'nearest'),    # 0
                                    ('NI_WRAP', 'wrap'),          # 1
                                    ('NI_REFLECT', 'reflect'),    # 2
                                    ('NI_MIRROR', 'mirror'),      # 3
                                    ('NI_CONSTANT', 'constant'))  # 4


class ZoomShift(theano.gof.COp):
    """
    Uses spline interpolation to zoom and shift an array.

    Wrapper for SciPy's ndimage.interpolation.zoomshift function.
    See `zoom` for more information.

    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'mode')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        mode=ZoomShiftMode,
                                        axes=T.lvector)
    c_func_file = 'c_code/scipy_ndimage_zoomshift.c'
    c_func_name = 'cpu_zoomshift'

    def __init__(self, order=0, mode='constant', axes=[]):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        assert mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant')
        self.order = order
        self.mode = mode
        self.axes = _normalize_zoomshift_axes_list(axes)
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        return (2,)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input, zoom_output_shape, zoom_ar, shift_ar, cval=0.):
        input = T.as_tensor_variable(input)
        zoom_output_shape = T.as_tensor_variable(zoom_output_shape).astype('int64')
        axes = _normalize_zoomshift_axes_list(self.axes, input.ndim)
        naxes = input.ndim if axes == [] else len(axes)
        if zoom_ar is None:
            zoom_ar = T.zeros((naxes,), 'float64')
        else:
            zoom_ar = T.as_tensor_variable(zoom_ar).astype('float64')
        if shift_ar is None:
            shift_ar = T.zeros((naxes,), 'float64')
        else:
            shift_ar = T.as_tensor_variable(shift_ar).astype('float64')
        cval = T.as_tensor_variable(cval).astype('float64')
        assert zoom_output_shape.ndim == 1
        assert zoom_ar.ndim == 1
        assert shift_ar.ndim == 1
        assert cval.ndim == 0

        broadcastable = [False if axes == [] or axis in axes else input.broadcastable[axis]
                         for axis in range(input.ndim)]
        return theano.gof.Apply(self, [input, zoom_output_shape, zoom_ar, shift_ar, cval],
                                [T.TensorType(dtype=input.type.dtype,
                                              broadcastable=broadcastable)()])

    def infer_shape(self, node, shapes):
        input, zoom_output_shape = node.inputs[:2]
        input_shape = shapes[0]
        axes = self.axes if len(self.axes) > 0 else range(input.ndim)
        output_shape = [zoom_output_shape[axis] if axis in axes else input_shape[axis]
                        for axis in range(input.ndim)]
        return [output_shape]

    def connection_pattern(self, node):
        return [[True], [False], [False], [False], [True]]

    def grad(self, inputs, output_grads):
        input, zoom_output_shape, zoom_ar, shift_ar, cval = inputs
        axes = self.axes if len(self.axes) > 0 else range(input.ndim)
        zoom_bottom_shape = [input.shape[axis] for axis in axes]
        grad = ZoomShiftGrad(order=self.order, mode=self.mode, axes=self.axes)(
            output_grads[0], zoom_bottom_shape, zoom_ar, shift_ar, cval)
        return [grad,
                theano.gradient.DisconnectedType()(),
                theano.gradient.DisconnectedType()(),
                theano.gradient.DisconnectedType()(),
                theano.gradient.grad_not_implemented(self, 4, cval)]

    def perform(self, node, inputs, out, params):
        assert imported_scipy, (
            "SciPy ndimage not available. Scipy is needed for ZoomShift.perform")

        input, zoom_output_shape, zoom_ar, shift_ar, cval = inputs
        if len(self.axes) in (0, input.ndim):
            # simple: zoom over all axes
            img_shape = [input.shape[axis] for axis in range(input.ndim)]
            zoom = [(ii / jj) for ii, jj in zip(zoom_output_shape, img_shape)]
            out[0][0] = scipy.ndimage.zoom(input, zoom, order=params.order,
                                           mode=self.mode, cval=cval,
                                           prefilter=False,
                                           output=input.dtype)
        else:
            # zoom over specific axes only
            img_shape = [input.shape[axis] for axis in self.axes]
            zoom = [(ii / jj) for ii, jj in zip(zoom_output_shape, img_shape)]
            output_shape = [zoom_output_shape[self.axes.index(axis)] if axis in self.axes else input.shape[axis]
                            for axis in range(input.ndim)]
            no_zoom_axes = [axis for axis in range(input.ndim) if axis not in self.axes]
            res = np.zeros(shape=output_shape, dtype=input.dtype)
            # iterate over the images
            for no_zoom_idx in np.ndindex(*[input.shape[axis] for axis in no_zoom_axes]):
                # find out where this image lives
                image_slice = [slice(None)] * input.ndim
                for axis, idx in zip(no_zoom_axes, no_zoom_idx):
                    image_slice[axis] = idx
                # process single image
                scipy.ndimage.zoom(input[image_slice], zoom, order=params.order,
                                   mode=self.mode, cval=cval,
                                   prefilter=False,
                                   output=res[image_slice])
            out[0][0] = res


class ZoomShiftGrad(theano.gof.COp):
    """
    Gradient for ZoomShift.

    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'mode')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        mode=ZoomShiftMode,
                                        axes=T.lvector)
    c_func_file = 'c_code/scipy_ndimage_zoomshift.c'
    c_func_name = 'cpu_zoomshift_grad'

    def __init__(self, order=0, mode='constant', axes=[]):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        assert mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant')
        self.order = order
        self.mode = mode
        self.axes = _normalize_zoomshift_axes_list(axes)
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        return (2,)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input, bottom_shape, zoom_ar, shift_ar, cval=0.):
        input = T.as_tensor_variable(input)
        bottom_shape = T.as_tensor_variable(bottom_shape).astype('int64')
        axes = _normalize_zoomshift_axes_list(self.axes, input.ndim)
        naxes = input.ndim if axes == [] else len(axes)
        if zoom_ar is None:
            zoom_ar = T.zeros((naxes,), 'float64')
        else:
            zoom_ar = T.as_tensor_variable(zoom_ar).astype('float64')
        if shift_ar is None:
            shift_ar = T.zeros((naxes,), 'float64')
        else:
            shift_ar = T.as_tensor_variable(shift_ar).astype('float64')
        cval = T.as_tensor_variable(cval).astype('float64')
        assert bottom_shape.ndim == 1
        assert zoom_ar.ndim == 1
        assert shift_ar.ndim == 1
        assert cval.ndim == 0

        broadcastable = [False if axes == [] or axis in axes else input.broadcastable[axis]
                         for axis in range(input.ndim)]
        return theano.gof.Apply(self, [input, bottom_shape, zoom_ar, shift_ar, cval],
                                [T.TensorType(dtype=input.type.dtype,
                                              broadcastable=broadcastable)()])

    def infer_shape(self, node, shapes):
        input, zoom_bottom_shape = node.inputs[:2]
        input_shape = shapes[0]
        axes = self.axes if len(self.axes) > 0 else range(input.ndim)
        bottom_shape = [zoom_bottom_shape[axis] if axis in axes else input_shape[axis]
                        for axis in range(input.ndim)]
        return [bottom_shape]

    def connection_pattern(self, node):
        return [[True], [False], [False], [False], [False]]

    def grad(self, inputs, output_grads):
        assert imported_scipy, (
            "SciPy ndimage not available. Scipy is needed for ZoomShiftGrad.perform")

        input, bottom_shape, zoom_ar, shift_ar, cval = inputs
        axes = self.axes if len(self.axes) > 0 else range(input.ndim)
        zoom_output_shape = [input.shape[axis] for axis in axes]
        grad = ZoomShift(order=self.order, mode=self.mode, axes=self.axes)(
            output_grads[0], zoom_output_shape, zoom_ar, shift_ar, 0.0)
        return [grad] + [theano.gradient.DisconnectedType()() for i in range(4)]


def zoom(input, zoom, order=3, mode='constant', cval=0.0,
         prefilter=True, axes=None):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    This function is equivalent to `scipy.ndimage.interpolation.zoom`.

    Parameters
    ----------
    input : tensor
        The input array.
    zoom : scalar or vector
        The zoom factor along the axes. If a scalar, `zoom` is the same for each
        axis. If a vector, `zoom` should contain one value for each axis.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.
    axes : tuple of int, optional
        Specifies which axes of the input are zoomed. The zoom operation will
        loop over the axes not in this list. If `axes` is None (the default)
        the zoom operation will work on all axes. `axes` can only be given
        in ascending order and should be matched with corresponding zoom
        factors in `zoom`.

    Returns
    -------
    zoom : tensor
        The zoomed input. For the zooming axes in `axes`, the output dimension
        is computed as `round(input_shape * zoom)`. The non-zooming axes have
        the same length as the input.

    Notes
    -----
    The SciPy function `scipy.ndimage.interpolation.zoom` uses a different
    rounding method to compute the output shape in Python 2.7 and Python 3.
    For some combinations of input shapes and zoom factors, this can lead
    to one-pixel differences in the output shape. This Theano function
    always uses the same (Python 3) rounding mode, so with Python 2.7 the
    output shape of `theano.tensor.scipy_ndimage.zoom` might be one
    pixel smaller than that of `scipy.ndimage.interpolation.zoom`.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    if input.ndim < 1:
        raise RuntimeError('input rank must be > 0')
    if mode not in ('nearest', 'wrap', 'reflect', 'mirror', 'constant'):
        raise RuntimeError('invalid mode')
    input = T.as_tensor_variable(input)

    axes = _normalize_zoomshift_axes_list(axes, input.ndim)
    if axes == []:
        axes = list(range(input.ndim))

    zoom = T.as_tensor_variable(zoom).astype('float64')
    if zoom.ndim == 0:
        zoom = theano.tensor.extra_ops.repeat(zoom, len(axes))
    if zoom.ndim != 1:
        raise ValueError('zoom should be a scalar or vector')

    # scipy.ndimage.zoom uses Python's round() to compute the output shape,
    # this gives different results on Python 3.
    img_shape = input.shape[axes]
    zoom_output_shape = T.iround(img_shape * zoom, mode='half_to_even')

    # Zooming to non-finite values is unpredictable, so just choose
    # zoom factor 1 instead
    a = T.switch(T.le(zoom_output_shape, 1),
                 1, img_shape - 1)
    b = T.switch(T.le(zoom_output_shape, 1),
                 1, zoom_output_shape - 1)
    zoom = a.astype('float64') / b.astype('float64')

    if prefilter and order > 1:
        filtered = input
        for axis in axes:
            filtered = spline_filter1d(filtered, order, axis=axis)
    else:
        filtered = input

    return ZoomShift(order, mode, axes=axes)(filtered, zoom_output_shape, zoom, None, cval)


class SplineFilter1D(theano.gof.COp):
    """
    Calculates a one-dimensional spline filter along the given axis.

    Wrapper for SciPy's ndimage.interpolation.spline_filter1d function.
    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'axis', 'inplace')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        axis=theano.scalar.int32,
                                        inplace=theano.scalar.bool)
    c_func_file = 'c_code/scipy_ndimage_splinefilter1d.c'
    c_func_name = 'cpu_splinefilter1d'

    def __init__(self, order=0, axis=-1, inplace=False):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        if order < 2:
            raise ValueError('spline filter with order < 2 does nothing')
        self.order = int(order)
        self.axis = int(axis)
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [0]}
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        return (2,)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input):
        input = T.as_tensor_variable(input)
        if input.ndim < 1:
            raise ValueError('SplineFilter1D does not work for scalars.')
        if self.axis != -1 and self.axis < 0 or self.axis >= input.ndim:
            raise ValueError('Invalid value axis=%d for an input '
                             'with %d dimensions.' % (self.axis, input.ndim))
        return theano.gof.Apply(self, [input], [input.type()])

    def infer_shape(self, node, in_shapes):
        return in_shapes

    def grad(self, inputs, output_grads):
        return SplineFilter1DGrad(order=self.order, axis=self.axis)(output_grads[0]),

    def perform(self, node, inputs, out, params):
        assert imported_scipy, (
            "SciPy ndimage not available. Scipy is needed for SplineFilter1D.perform")

        input, = inputs
        out[0][0] = scipy.ndimage.spline_filter1d(input, output=(input if self.inplace else input.dtype),
                                                  order=params.order, axis=params.axis)
        if self.inplace:
            out[0][0] = input


class SplineFilter1DGrad(theano.gof.COp):
    """
    Gradient for SplineFilter1D.
    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'axis', 'inplace')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        axis=theano.scalar.int32,
                                        inplace=theano.scalar.bool)
    c_func_file = 'c_code/scipy_ndimage_splinefilter1d.c'
    c_func_name = 'cpu_splinefilter1d_grad'

    def __init__(self, order=0, axis=-1, inplace=False):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        if order < 2:
            raise ValueError('spline filter with order < 2 does nothing')
        self.order = int(order)
        self.axis = int(axis)
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [0]}
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        return (2,)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input):
        input = T.as_tensor_variable(input)
        if input.ndim < 1:
            raise ValueError('SplineFilter1DGrad does not work for scalars.')
        return theano.gof.Apply(self, [input], [input.type()])

    def infer_shape(self, node, in_shapes):
        return in_shapes

    def grad(self, inputs, output_grads):
        return SplineFilter1D(order=self.order, axis=self.axis)(output_grads[0]),


def spline_filter1d(input, order=3, axis=-1):
    """
    Calculates a one-dimensional spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    This function is equivalent to `scipy.ndimage.interpolation.spline_filter1d`.

    Parameters
    ----------
    input : tensor
        The input array.
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.

    Returns
    -------
    spline_filter1d : tensor
        The filtered input.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    if order in [0, 1]:
        return input
    else:
        return SplineFilter1D(order, axis)(input)


def spline_filter(input, order=3):
    """
    Multi-dimensional spline filter.

    For more details, see `spline_filter1d`.

    See Also
    --------
    spline_filter1d

    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    if order in [0, 1]:
        return input
    for axis in range(input.ndim):
        input = spline_filter1d(input, order, axis)
    return input


def register_inplace(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        theano.compile.optdb.register(
            name, theano.gof.TopoOptimizer(
                local_opt, failure_callback=theano.gof.TopoOptimizer.warn_inplace),
            60, 'fast_run', 'inplace', *tags)
        return local_opt
    return f


@register_inplace()
@theano.gof.local_optimizer([SplineFilter1D], inplace=True)
def local_spline_filter1d_inplace(node):
    if isinstance(node.op, SplineFilter1D) and not node.op.inplace:
        return [SplineFilter1D(order=node.op.order, axis=node.op.axis, inplace=True)(*node.inputs)]


@register_inplace()
@theano.gof.local_optimizer([SplineFilter1DGrad], inplace=True)
def local_spline_filter1d_grad_inplace(node):
    if isinstance(node.op, SplineFilter1DGrad) and not node.op.inplace:
        return [SplineFilter1DGrad(order=node.op.order, axis=node.op.axis, inplace=True)(*node.inputs)]
