import theano
import unittest
import numpy
from nose.plugins.skip import SkipTest
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.sandbox import mkl
from theano.sandbox.mkl.basic_ops import U2IConv, I2U
from theano.sandbox.mkl.mkl_conv import Conv2D

numpy.random.seed(123)

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


class test_mkl_conv_forward(unittest.TestCase):
    def test_conv_U2I(self):
        images = T.ftensor4('inputs')
        a_internal = U2IConv(imshp=(12, 3, 256, 256),
                             kshp=(12, 3, 3, 3))(images)
        out = I2U()(a_internal)

        fopt = theano.function([images], out, mode=mode_with_mkl)
        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float32)
        assert numpy.allclose(fopt(ival), ival)

    def test_conv_no_bias(self):
        images = T.ftensor4('inputs')
        weights = T.ftensor4('weights')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)
        convOut_internal = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights)
        convOut_user = I2U()(convOut_internal)

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float32)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float32)

        fopt = theano.function(inputs=[images, weights], outputs=convOut_user, mode=mode_with_mkl)
        new_out = fopt(ival, wval)

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        fori = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_without_mkl)
        old_out = fori(ival, wval)

        assert numpy.allclose(old_out, new_out)

    def test_conv_with_bias(self):
        images = T.ftensor4('inputs')
        weights = T.ftensor4('weights')
        bias = T.vector('bias')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)
        convOutBias_internal = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights, bias)
        convOutBias_user = I2U()(convOutBias_internal)

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float32)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float32)
        bval = numpy.random.rand(12).astype(numpy.float32)

        fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBias_user, mode=mode_with_mkl)
        new_old = fopt(ival, wval, bval)

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutBias = convOut + bias.dimshuffle('x', 0, 'x', 'x')
        fori = theano.function(inputs=[images, weights, bias], outputs=convOutBias, mode=mode_without_mkl)
        old_out = fori(ival, wval, bval)

        assert numpy.allclose(old_out, new_old)

    def test_no_shape(self):
        images = T.ftensor4('inputs')
        weights = T.ftensor4('weights')
        convOut = conv2d(images, weights, filter_shape=(12, 3, 3, 3), filter_flip=False)
        ival = numpy.random.rand(24, 3, 256, 256).astype(numpy.float32)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float32)

        fopt = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_with_mkl)
        new_out = fopt(ival, wval)

        fori = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_without_mkl)
        old_out = fori(ival, wval)

        assert numpy.allclose(old_out, new_out)


class test_mkl_conv_backward(unittest.TestCase):
    def test_conv_no_bias(self):
        images = T.ftensor4('input_conv')
        weights = T.ftensor4('weights')

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutSum = T.sum(convOut)
        conv_op_di = T.grad(convOutSum, images)
        conv_op_dk = T.grad(convOutSum, weights)
        convOutBack = [conv_op_di, conv_op_dk]

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float32)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float32)

        fopt = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_with_mkl)
        new_out = fopt(ival, wval)

        fori = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_without_mkl)
        old_out = fori(ival, wval)

        assert numpy.allclose(old_out[0], new_out[0])
        # weightsGrad Layout is different.
        # assert numpy.allclose(old_out[1], new_out[1])

    def test_conv_with_bias(self):
        images = T.ftensor4('input_conv')
        weights = T.ftensor4('weights')
        bias = T.vector('bias')

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutSum = T.sum(convOut + bias.dimshuffle('x', 0, 'x', 'x'))
        conv_op_di = theano.grad(convOutSum, images)
        conv_op_dk = theano.grad(convOutSum, weights)
        conv_op_db = theano.grad(convOutSum, bias)

        convOutBack = [conv_op_di, conv_op_dk, conv_op_db]

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float32)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float32)
        bval = numpy.random.rand(12).astype(numpy.float32)

        fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_with_mkl)
        new_out = fopt(ival, wval, bval)

        fori = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_without_mkl)
        old_out = fori(ival, wval, bval)
        assert numpy.allclose(old_out[0], new_out[0])
        # assert numpy.allclose(old_out[1], new_out[1])
        # assert numpy.allclose(old_out[2], new_out[2])


if __name__ == '__main__':
    unittest.main()
