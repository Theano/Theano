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
        images = T.dtensor4('inputs')
        a_internal = U2IConv(imshp=(12, 3, 256, 256),
                             kshp=(12, 3, 3, 3))(images)
        out = I2U()(a_internal)

        fopt = theano.function([images], out, mode=mode_with_mkl)
        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float64)
        assert numpy.allclose(fopt(ival), ival)

    def test_conv_no_bias(self):
        images = T.dtensor4('inputs')
        weights = T.dtensor4('weights')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)
        convOut_internal = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights)
        convOut_user = I2U()(convOut_internal)

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float64)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float64)

        fopt = theano.function(inputs=[images, weights], outputs=convOut_user, mode=mode_with_mkl)
        new_out = fopt(ival, wval)

        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        fori = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_without_mkl)
        old_out = fori(ival, wval)
        
        assert str(fopt.maker.fgraph.toposort()) != str(fori.maker.fgraph.toposort())
        assert numpy.allclose(old_out, new_out)

    def test_conv_with_bias(self):
        images = T.dtensor4('inputs')
        weights = T.dtensor4('weights')
        bias = T.dvector('bias')

        ishape = [(8, 3, 256, 256), (16, 3, 256, 256), (32, 3, 256, 256), (64, 3, 256, 256)]
        wshape = [(8, 3, 3, 3), (16, 3, 3, 3), (32, 3, 3, 3), (64, 3, 3, 3)]

        for i, ish in enumerate(ishape):
            wsh = wshape[i]
            images_internal = U2IConv(imshp=ish, kshp=wsh)(images)
            convOutBias_internal = Conv2D(imshp=ish, kshp=wsh, filter_flip=False)(images_internal, weights, bias)
            convOutBias_user = I2U()(convOutBias_internal)

            ival = numpy.random.rand(*ish).astype(numpy.float64)
            wval = numpy.random.rand(*wsh).astype(numpy.float64)
            bval = numpy.random.rand(wsh[0]).astype(numpy.float64)

            fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBias_user, mode=mode_with_mkl)
            new_old = fopt(ival, wval, bval)

            convOut = conv2d(images, weights, input_shape=ish, filter_shape=wsh, filter_flip=False)
            convOutBias = convOut + bias.dimshuffle('x', 0, 'x', 'x')
            fori = theano.function(inputs=[images, weights, bias], outputs=convOutBias, mode=mode_without_mkl)
            old_out = fori(ival, wval, bval)

            assert str(fopt.maker.fgraph.toposort()) != str(fori.maker.fgraph.toposort())
            assert numpy.allclose(old_out, new_old)

    def test_no_shape(self):
        images = T.dtensor4('inputs')
        weights = T.dtensor4('weights')

        convOut = conv2d(images, weights, filter_shape=(12, 3, 3, 3), filter_flip=False)

        fopt = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_with_mkl)

        fori = theano.function(inputs=[images, weights], outputs=convOut, mode=mode_without_mkl)
        
        # No optimization for the case image shape is None
        assert all([not isinstance(n, (Conv2D, U2IConv, I2U)) for n in fopt.maker.fgraph.toposort()])
        assert str(fopt.maker.fgraph.toposort()) == str(fori.maker.fgraph.toposort())


class test_mkl_conv_backward(unittest.TestCase):
    def test_conv_no_bias(self):
        images = T.dtensor4('input_conv')
        weights = T.dtensor4('weights')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)

        convOut = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights)
        convOut_user = I2U()(convOut)
        convOutLoss = T.mean(convOut_user)
        conv_op_di = T.grad(convOutLoss, images)
        conv_op_dk = T.grad(convOutLoss, weights)
        convOutBack = [conv_op_di, conv_op_dk]

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float64)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float64)

        fopt = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_with_mkl)
        new_out = fopt(ival, wval)
        
        convOut = conv2d(images, weights, input_shape=(12, 3, 256, 256), filter_shape=(12, 3, 3, 3), filter_flip=False)
        convOutLoss = T.mean(convOut)
        conv_op_di = T.grad(convOutLoss, images)
        conv_op_dk = T.grad(convOutLoss, weights)
        convOutBack = [conv_op_di, conv_op_dk]
        
        fori = theano.function(inputs=[images, weights], outputs=convOutBack, mode=mode_without_mkl)
        old_out = fori(ival, wval)
        
        assert len(fopt.maker.fgraph.toposort()) != len(fori.maker.fgraph.toposort())
        assert numpy.allclose(old_out[0], new_out[0])
        assert new_out[0].dtype == 'float64'
        # weightsGrad Layout is different.
        # assert numpy.allclose(old_out[1], new_out[1])

    def test_conv_with_bias(self):
        images = T.dtensor4('input_conv')
        weights = T.dtensor4('weights')
        bias = T.dvector('bias')

        ishape = [(8, 3, 256, 256), (16, 3, 256, 256), (32, 3, 256, 256), (64, 3, 256, 256)]
        wshape = [(8, 3, 3, 3), (16, 3, 3, 3), (32, 3, 3, 3), (64, 3, 3, 3)]

        for i, ish in enumerate(ishape):
            wsh = wshape[i]

            images_internal = U2IConv(imshp=ish, kshp=wsh)(images)
            convOut = Conv2D(imshp=ish, kshp=wsh, filter_flip=False)(images_internal, weights, bias)
            convOut_user = I2U()(convOut)
            convOutLoss = T.mean(convOut_user)
            conv_op_di = theano.grad(convOutLoss, images)
            conv_op_dk = theano.grad(convOutLoss, weights)
            conv_op_db = theano.grad(convOutLoss, bias)

            convOutBack = [conv_op_di, conv_op_dk, conv_op_db]

            ival = numpy.random.rand(*ish).astype(numpy.float64)
            wval = numpy.random.rand(*wsh).astype(numpy.float64)
            bval = numpy.random.rand(wsh[0]).astype(numpy.float64) - numpy.random.rand(wsh[0]).astype(numpy.float64)

            fopt = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_with_mkl)
            new_out = fopt(ival, wval, bval)

            convOut = conv2d(images, weights, input_shape=ish, filter_shape=wsh, filter_flip=False)
            convOutLoss = T.mean(convOut + bias.dimshuffle('x', 0, 'x', 'x'))
            conv_op_di = theano.grad(convOutLoss, images)
            conv_op_dk = theano.grad(convOutLoss, weights)
            conv_op_db = theano.grad(convOutLoss, bias)

            convOutBack = [conv_op_di, conv_op_dk, conv_op_db]

            fori = theano.function(inputs=[images, weights, bias], outputs=convOutBack, mode=mode_without_mkl)
            old_out = fori(ival, wval, bval)
            assert len(fopt.maker.fgraph.toposort()) != len(fori.maker.fgraph.toposort())
            assert numpy.allclose(old_out[0], new_out[0])
            # assert numpy.allclose(old_out[1], new_out[1])
            assert numpy.allclose(old_out[2], new_out[2])
            assert new_out[0].dtype == 'float64'
            assert new_out[2].dtype == 'float64'

if __name__ == '__main__':
    unittest.main()
