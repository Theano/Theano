from __future__ import absolute_import, print_function, division

import os.path

import numpy as np

import theano
from theano.gradient import DisconnectedType


def scipy_ndimage_helper_inc_dir():
    return os.path.join(os.path.dirname(__file__), 'c_code/scipy_ndimage')


class ZoomShift(theano.gof.COp):
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'mode')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        mode=theano.gof.EnumList(('NI_NEAREST', 'nearest'),  # 0
                                                                 ('NI_WRAP', 'wrap'),        # 1
                                                                 ('NI_REFLECT', 'reflect'),  # 2
                                                                 ('NI_MIRROR', 'mirror'),    # 3),
                                                                 ('NI_CONSTANT', 'constant')))
    c_func_file = 'c_code/scipy_ndimage_zoomshift.c'
    c_func_name = 'cpu_zoomshift'

    def __init__(self, order=0, mode='constant'):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        assert mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant')
        self.order = order
        self.mode = mode
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        import time
        return (time.time(),)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input, output_shape, zoom_ar, shift_ar, cval=0.):
        input = theano.tensor.as_tensor_variable(input)
        output_shape = theano.tensor.as_tensor_variable(output_shape).astype('int64')
        zoom_ar = theano.tensor.as_tensor_variable(zoom_ar).astype('float64')
        shift_ar = theano.tensor.as_tensor_variable(shift_ar).astype('float64')
        cval = theano.tensor.as_tensor_variable(cval).astype('float64')
        assert output_shape.ndim == 1  # TODO dtype is int
        assert zoom_ar.ndim == 1
        assert shift_ar.ndim == 1
        assert cval.ndim == 0

        # TODO broadcastable?
        return theano.gof.Apply(self, [input, output_shape, zoom_ar, shift_ar, cval],
                                [theano.tensor.TensorType(dtype=input.type.dtype,
                                                          broadcastable=(False,) * input.ndim)()])

    def infer_shape(self, node, shapes):
        return node.inputs[1],

    def grad(self, inputs, output_grads):
        input, output_shape, zoom_ar, shift_ar, cval = inputs
        grad = ZoomShiftGrad(order=self.order, mode=self.mode)(output_grads[0], input.shape, zoom_ar, shift_ar, cval)
        return [grad] + [theano.gradient.DisconnectedType()() for i in range(4)]


class ZoomShiftGrad(theano.gof.COp):
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'mode')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        mode=theano.gof.EnumList(('NI_NEAREST', 'nearest'),  # 0
                                                                 ('NI_WRAP', 'wrap'),        # 1
                                                                 ('NI_REFLECT', 'reflect'),  # 2
                                                                 ('NI_MIRROR', 'mirror'),    # 3),
                                                                 ('NI_CONSTANT', 'constant')))
    c_func_file = 'c_code/scipy_ndimage_zoomshift.c'
    c_func_name = 'cpu_zoomshift_grad'

    def __init__(self, order=0, mode='constant'):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        assert mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant')
        self.order = order
        self.mode = mode
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        import time
        return (time.time(),)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input, bottom_shape, zoom_ar, shift_ar, cval=0.):
        input = theano.tensor.as_tensor_variable(input)
        bottom_shape = theano.tensor.as_tensor_variable(bottom_shape).astype('int64')
        zoom_ar = theano.tensor.as_tensor_variable(zoom_ar).astype('float64')
        shift_ar = theano.tensor.as_tensor_variable(shift_ar).astype('float64')
        cval = theano.tensor.as_tensor_variable(cval).astype('float64')
        assert bottom_shape.ndim == 1  # TODO dtype is int
        assert zoom_ar.ndim == 1
        assert shift_ar.ndim == 1
        assert cval.ndim == 0

        # TODO broadcastable?
        return theano.gof.Apply(self, [input, bottom_shape, zoom_ar, shift_ar, cval],
                                [theano.tensor.TensorType(dtype=input.type.dtype,
                                                          broadcastable=(False,) * input.ndim)()])

    def infer_shape(self, node, shapes):
        return node.inputs[1],


class SplineFilter1D(theano.gof.COp):
    """
    Calculates a one-dimensional spline filter along the given axis.

    Wrapper for SciPy's ndimage.interpolation.spline_filter1d function.
    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'axis')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        axis=theano.scalar.int32)
    c_func_file = 'c_code/scipy_ndimage_splinefilter1d.c'
    c_func_name = 'cpu_splinefilter1d'

    def __init__(self, order=0, axis=-1):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        if order < 2:
            raise ValueError('spline filter with order < 2 does nothing')
        self.order = order
        self.axis = axis
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        # TODO
        import time
        return (time.time(),)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input):
        input = theano.tensor.as_tensor_variable(input)

        if input.ndim < 1:
            raise ValueError('SplineFilter1D does not work for scalars.')
        if self.axis != -1 and self.axis < 0 or self.axis >= input.ndim:
            raise ValueError('Invalid value axis=%d for an input '
                             'with %d dimensions.' % (self.axis, input.ndim))

         # TODO broadcastable?
        return theano.gof.Apply(self, [input],
                                [theano.tensor.TensorType(dtype=input.type.dtype,
                                                          broadcastable=(False,) * input.ndim)()])

    def infer_shape(self, node, in_shapes):
        return in_shapes

    def grad(self, inputs, output_grads):
        return SplineFilter1DGrad(order=self.order, axis=self.axis)(output_grads[0]),


class SplineFilter1DGrad(theano.gof.COp):
    """
    Gradient for SplineFilter1D.
    """
    # TODO _f16_ok and check_input ?
    __props__ = ('order', 'axis')
    params_type = theano.gof.ParamsType(order=theano.scalar.int32,
                                        axis=theano.scalar.int32)
    c_func_file = 'c_code/scipy_ndimage_splinefilter1d.c'
    c_func_name = 'cpu_splinefilter1d_grad'

    def __init__(self, order=0, axis=-1):
        if order < 0 or order > 5:
            raise ValueError('spline order %d not supported' % order)
        if order < 2:
            raise ValueError('spline filter with order < 2 does nothing')
        self.order = order
        self.axis = axis
        theano.gof.COp.__init__(self, [self.c_func_file], self.c_func_name)

    def c_code_cache_version(self):
        # TODO
        import time
        return (time.time(),)

    def c_headers(self):
        return ['<stdlib.h>', '<math.h>', 'ni_support.h', 'ni_support.c', 'ni_interpolation.c']

    def c_header_dirs(self):
        return [scipy_ndimage_helper_inc_dir()]

    def make_node(self, input):
        input = theano.tensor.as_tensor_variable(input)

        if input.ndim < 1:
            raise ValueError('SplineFilter1DGrad does not work for scalars.')
        if self.axis != -1 and self.axis < 0 or self.axis >= input.ndim:
            raise ValueError('Invalid value axis=%d for an input '
                             'with %d dimensions.' % (self.axis, input.ndim))

        return theano.gof.Apply(self, [input],
                                [theano.tensor.TensorType(dtype=input.type.dtype,
                                                          broadcastable=(False,) * input.ndim)()])

    def infer_shape(self, node, in_shapes):
        return in_shapes



