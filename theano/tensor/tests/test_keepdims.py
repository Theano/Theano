from __future__ import absolute_import, print_function, division
import unittest

import numpy
from six import integer_types

import theano
from theano import tensor, function
from theano.tests.unittest_tools import attr


# this tests other ops to ensure they keep the dimensions of their
# inputs correctly
class TestKeepDims(unittest.TestCase):

    def makeKeepDims_local(self, x, y, axis):
        if axis is None:
            newaxis = list(range(x.ndim))
        elif isinstance(axis, integer_types):
            if axis < 0:
                newaxis = [axis + x.type.ndim]
            else:
                newaxis = [axis]
        else:
            newaxis = []
            for a in axis:
                if a < 0:
                    a += x.type.ndim
                newaxis.append(a)
        i = 0
        new_dims = []
        for j, _ in enumerate(x.shape):
            if j in newaxis:
                new_dims.append('x')
            else:
                new_dims.append(i)
                i += 1

        return tensor.DimShuffle(y.type.broadcastable, new_dims)(y)

    @attr('slow')
    def test_keepdims(self):

        x = tensor.dtensor3()
        a = numpy.random.rand(3, 2, 4)
        # We don't need to test all opt and C code, as this is tested
        # by the ops tests.
        mode = theano.compile.Mode(optimizer="fast_compile", linker="py")

        # 'max_and_argmax' has two outputs and can be specified with either
        # a single or every axis:
        for axis in [0, 1, 2, [0], [1], [2], None, [0, 1, 2],
                     [-1], [-2], [-3], [-1, -2, -3], [0, -1, -2],
                     [-2, -3, 2]]:

            op = tensor.max_and_argmax
            f = function([x], [op(x, axis=axis, keepdims=True)[0],
                               self.makeKeepDims_local(
                                   x, op(x, axis=axis, keepdims=False)[0],
                                   axis)],
                         mode=mode)
            ans1, ans2 = f(a)
            assert numpy.allclose(ans1, ans2)
            assert ans1.shape == ans2.shape

            f = function([x], [op(x, axis=axis, keepdims=True)[1],
                               self.makeKeepDims_local(
                                   x, op(x, axis=axis, keepdims=False)[1],
                                   axis)],
                         mode=mode)
            ans1, ans2 = f(a)
            assert numpy.allclose(ans1, ans2)
            assert ans1.shape == ans2.shape

        # the following ops can be specified with either a single axis or every
        # axis:
        for op in ([tensor.argmax, tensor.argmin]):
            for axis in [0, 1, 2, [0], [1], [2], None, [0, 1, 2],
                         [-1], [-2], [-3], [-1, -2, -3], [0, -2, 2]]:

                f = function([x], [op(x, axis=axis, keepdims=True),
                                   self.makeKeepDims_local(
                                       x, op(x, axis=axis, keepdims=False),
                                       axis)],
                             mode=mode)
                ans1, ans2 = f(a)
                assert numpy.allclose(ans1, ans2)
                assert ans1.shape == ans2.shape

        # the following ops can be specified with a freely specified axis
        # parameter
        for op in ([tensor.sum, tensor.prod, tensor.mean, tensor.var,
                    tensor.std, tensor.all, tensor.any,
                    tensor.max, tensor.min]):
            for axis in [0, 1, 2, [0], [1], [2], None,
                         [0, 1], [1, 2], [0, 1, 2],
                         [-1], [-2], [-3], [-1, -2], [-1, -2, -3], [0, -2, 2]]:

                f = function([x], [op(x, axis=axis, keepdims=True),
                                   self.makeKeepDims_local(
                                       x, op(x, axis=axis, keepdims=False),
                                       axis)],
                             mode=mode)

                ans1, ans2 = f(a)
                assert numpy.allclose(ans1, ans2)
                assert ans1.shape == ans2.shape
