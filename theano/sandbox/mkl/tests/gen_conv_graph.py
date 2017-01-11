import theano
import unittest
import numpy
from nose.plugins.skip import SkipTest
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.sandbox import mkl

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
    def test_conv_no_bias(self):
        images = T.ftensor4('inputs')
        weights = T.ftensor4('weights')
        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)

        theano.printing.pydotprint(convOut, outfile="Conv_before_opt.png", var_with_name_simple=True)
        fopt = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_with_mkl)
        theano.printing.pydotprint(fopt, outfile="Conv_OPT_after_opt.png", var_with_name_simple=True)

        fori = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_without_mkl)
        theano.printing.pydotprint(fori, outfile="Conv_Original_after_opt.png", var_with_name_simple=True)

    def test_conv_with_bias(self):
        images = T.ftensor4('inputs')
        weights = T.ftensor4('weights')
        bias = T.vector('bias')

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutBias = convOut + bias.dimshuffle('x', 0, 'x', 'x')

        theano.printing.pydotprint(convOutBias, outfile="ConvBias_before_opt.png", var_with_name_simple=True)
        fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBias, mode=mode_with_mkl)
        theano.printing.pydotprint(fopt, outfile="ConvBias_OPT_after_opt.png", var_with_name_simple=True)

        fori = theano.function(inputs=[images, weights, bias], outputs=convOutBias, mode=mode_without_mkl)
        theano.printing.pydotprint(fori, outfile="ConvBias_Original_after_opt.png", var_with_name_simple=True)


class test_mkl_conv_backward(unittest.TestCase):
    def test_conv_no_bias(self):
        images = T.ftensor4('input_conv')
        weights = T.ftensor4('weights')

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutSum = T.sum(convOut)
        conv_op_di = T.grad(convOutSum, images)
        conv_op_dk = T.grad(convOutSum, weights)
        convOutBack = [conv_op_di, conv_op_dk]

        theano.printing.pydotprint(convOutBack, outfile="ConvBack_before_opt.png", var_with_name_simple=True)
        fopt = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_with_mkl)
        theano.printing.pydotprint(fopt, outfile="ConvBack_OPT_after_opt.png", var_with_name_simple=True)

        fori = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_without_mkl)
        theano.printing.pydotprint(fori, outfile="ConvBack_Original_after_opt.png", var_with_name_simple=True)

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

        theano.printing.pydotprint(convOutBack, outfile="ConvBiasBack_before_opt.png", var_with_name_simple=True)
        fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_with_mkl)
        theano.printing.pydotprint(fopt, outfile="ConvBiasBack_OPT_after_opt.png", var_with_name_simple=True)

        fori = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_without_mkl)
        theano.printing.pydotprint(fori, outfile="ConvBiasBack_Original_after_opt.png", var_with_name_simple=True)


if __name__ == '__main__':
    unittest.main()
