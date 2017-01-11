from __future__ import absolute_import, print_function, division

import numpy
import unittest
from itertools import product

import theano.tensor as T
from theano import function
from theano.tests import unittest_tools as utt
from theano.sandbox.mkl.mkl_relu import Relu
from theano.sandbox.mkl.basic_ops import (U2IRelu, I2U)

uniq_id = 0


class TestMKLRelu(unittest.TestCase):

    def mkl_relu_func(*inputs):
        global uniq_id
        uniq_id += 1

        if len(inputs) == 2:
            # self, image
            _, x, = inputs

            x_internal = U2IRelu(uniq_id=uniq_id)(x)
            reluOut = Relu(uniq_id=uniq_id)(x_internal)
            output = I2U(uniq_id=uniq_id)(reluOut)
        elif len(inputs) == 3:
            # self, image, slope
            _, x, slope, = inputs

            x_internal = U2IRelu(slope=slope, uniq_id=uniq_id)(x)
            reluOut = Relu(slope=slope, uniq_id=uniq_id)(x_internal)
            output = I2U(uniq_id=uniq_id)(reluOut)
        else:
            raise ValueError("incorrect inputs list, should be 2 ~ 3 parameters!")

        return output

    def test_relu(self):
        def ref(input):
            return numpy.maximum(input, 0)

        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n, c = 4, 2

        image = T.dtensor4('image')
        for imgsize in imgsize_list:
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            output_ref = ref(imval)

            output = self.mkl_relu_func(image)

            f = function([image, ], [output, ])
            output_val = f(imval)

        utt.assert_allclose(output_val, output_ref)

    def test_relu_slope(self):
        def ref(input, slope):
            return numpy.where(input > 0, input, slope * input)

        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        slope_list = (0, 0.3, 1, 2, -0.3, -1, -2)
        n, c = 4, 2

        image = T.dtensor4('image')
        for imgsize, slope in product(imgsize_list, slope_list):
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            output_ref = ref(imval, slope)

            output = self.mkl_relu_func(image, slope)

            f = function([image, ], [output, ])
            output_val = f(imval)

        utt.assert_allclose(output_val, output_ref)

    def test_relu_grad(self):
        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        n, c = 4, 2

        for imgsize in imgsize_list:
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            def mp(input):
                return self.mkl_relu_func(input)

            utt.verify_grad(mp, [imval], rng=rng)

    def test_relu_slope_grad(self):
        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)

        imgsize_list = ((5, 5), (6, 6), (6, 6), (8, 8))
        slope_list = (0, 0.3, 1, 2, -0.3, -1, -2)
        n, c = 4, 2

        for imgsize, slope in product(imgsize_list, slope_list):
            imval = rng.rand(n, c, imgsize[0], imgsize[1])

            def mp(input):
                return self.mkl_relu_func(input, slope)

            utt.verify_grad(mp, [imval], rng=rng)


if __name__ == "__main__":
    test_relu()
