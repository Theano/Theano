from nose.plugins.skip import SkipTest
import numpy
import unittest

import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_pooling():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    
    x = T.tensor4()

    out1 = cuda.dnn.dnn_pool(x, ws=(5,5), stride=(5, 5))
    out2 = T.signal.downsample.max_pool_2d(x, (5, 5))

    f1 = theano.function([x], out1)
    f2 = theano.function([x], out2)

    data = numpy.random.normal(0, 1, (1, 10, 100, 100)).astype("float32")
    a = f1(data).__array__()

    b = f2(data).__array__()

    assert numpy.allclose(a, b)

    gf1 = theano.function([x], theano.grad(out1.sum(), x))
    gf2 = theano.function([x], theano.grad(out2.sum(), x))

    ga = gf1(data).__array__()

    gb = gf2(data).__array__()

    assert numpy.allclose(ga, gb)
