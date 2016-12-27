import theano
import unittest
import numpy
from nose.plugins.skip import SkipTest
from theano import tensor as T
from theano.tensor.nnet.bn import batch_normalization
from theano.sandbox import mkl
from theano.sandbox.mkl.basic_ops import U2IBatchNormalization, I2U
from theano.sandbox.mkl import mkl_bn

numpy.random.seed(123)

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


class test_mkl_bn_forward(unittest.TestCase):
    def test_bn_U2I(self):
        x = T.ftensor4('x')
        x_internal = U2IBatchNormalization(eps=1e-5, uniq_id=1)(x)
        x_out = I2U(uniq_id=2)(x_internal)

        fopt = theano.function([x], x_out, mode=with_mkl)
        ival = numpy.random.rand(64, 5, 128, 128).astype(numpy.float32)
        assert numpy.allclose(fopt(ival), ival)

    def test_bn_U2I_wrong_dim(self):
        x = T.fmatrix('x')
        try:
            U2IBatchNormalization(eps=1e-5, uniq_id=1)(x)
            raise Exception('No exception when ndim is 2.')
        except TypeError:
            pass
        except Exception as e:
            raise Exception('test_bn_U2I_wrong_dim: ' + str(e))

    def test_bn_value(self):
        X = T.ftensor4('x')
        Scale = T.vector('scale')
        Shift = T.vector('shift')

        x_internal = U2IBatchNormalization(eps=0, uniq_id=1)(X)
        z_bn = mkl_bn.BatchNormalization(eps=0, bias=1, term=1, uniq_id=2)(x_internal, Scale, Shift)
        z_out = I2U(uniq_id=3)(z_bn)
        f = theano.function([X, Scale, Shift], z_out, mode=with_mkl)

        ival = numpy.random.rand(16, 3, 4, 4).astype(numpy.float32)
        sval = numpy.random.rand(3).astype(numpy.float32)
        tval = numpy.random.rand(3).astype(numpy.float32)
        new_out = f(ival, sval, tval)

        def bn_ref(x, G, B, M, V):
            n = (x - M) / V
            return G * n + B

        bn_ref_op = bn_ref(X,
                           Scale.dimshuffle('x', 0, 'x', 'x'),
                           Shift.dimshuffle('x', 0, 'x', 'x'),
                           X.mean(axis=1, keepdims=True),
                           X.std(axis=1, keepdims=True))
        f_ref = theano.function([X, Scale, Shift], bn_ref_op, mode=without_mkl)
        ref_out = f_ref(ival, sval, tval)
        assert numpy.allclose(new_out, ref_out)


class test_mkl_bn_backward(unittest.TestCase):
    def test_bn_value(self):
        X = T.ftensor4('x')
        Scale = T.vector('scale')
        Shift = T.vector('shift')

        x_internal = U2IBatchNormalization(eps=1e-5, uniq_id=1)(X)
        z_bn = mkl_bn.BatchNormalization(eps=1e-5, bias=1, term=1, uniq_id=2)(x_internal, Scale, Shift)
        z_out = I2U(uniq_id=3)(z_bn)
        z_sum = T.sum(z_out)
        z_grad = T.grad(z_sum, [X])

        fgrad = theano.function([X, Scale, Shift], z_grad, mode=with_mkl)

        ival = numpy.random.rand(64, 5, 128, 128).astype(numpy.float32)
        sval = numpy.random.rand(5).astype(numpy.float32)
        tval = numpy.random.rand(5).astype(numpy.float32)
        fgrad(ival, sval, tval)

if __name__ == '__main__':
    unittest.main()
