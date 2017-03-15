from __future__ import absolute_import, print_function, division

import numpy
import unittest
from nose.plugins.skip import SkipTest

import theano
import theano.tensor as T
from theano import function
from theano.tests import unittest_tools as utt
from theano.sandbox import mkl
from theano.sandbox.mkl.mkl_concatenate import Concatenate
from theano.sandbox.mkl.basic_ops import (U2IConcatenate, I2U)

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


class TestMKLConcatenate(unittest.TestCase):
    def mkl_concatenate_func(*inputs):
        # _, axis, tensors = inputs
        axis = inputs[1]
        tensors = inputs[2:]

        tensors_internal = [U2IConcatenate()(x) for x in tensors]
        new_inputs = [axis] + tensors_internal
        out = Concatenate()(*new_inputs)
        output = I2U()(out)
        return output

    def test_concatenate(self):
        def ref(*inputs):
            axis = inputs[0]
            tensors = inputs[1:]

            return numpy.concatenate(tensors, axis)

        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n, c = 4, 2

        axis = 1

        image = T.dtensor4('image')
        image1 = T.dtensor4('image1')
        for imgsize in imgsize_list:
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            output_ref = ref(axis, imval, imval)

            Opout = self.mkl_concatenate_func(axis, image, image1)
            f = function([image, image1], [Opout, ])
            output_mkl = f(imval, imval)

            utt.assert_allclose(output_mkl, output_ref)

    def test_relu_grad(self):
        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n, c = 4, 2

        axis = 1

        image = T.dtensor4('image')
        image1 = T.dtensor4('image1')
        for imgsize in imgsize_list:
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            out = T.concatenate([image, image1], axis)
            sum_ref = T.sum(out)
            gx_ref = T.grad(sum_ref, [image, image1])
            f_ref = theano.function([image, image1], outputs=gx_ref, mode=mode_without_mkl)
            output_ref = f_ref(imval, imval)

            out_mkl = self.mkl_concatenate_func(axis, image, image1)
            sum_mkl = T.sum(out_mkl)
            gx_mkl = T.grad(sum_mkl, [image, image1])
            f_mkl = theano.function([image, image1], outputs=gx_mkl)
            output_mkl = f_mkl(imval, imval)

            utt.assert_allclose(output_mkl, output_ref)


if __name__ == "__main__":
    unittest.main()
