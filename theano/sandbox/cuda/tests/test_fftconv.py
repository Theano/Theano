import theano
from theano.tensor import ftensor4
from theano.tensor.nnet.tests.test_conv import TestConv2D

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

class GTestConv2D(TestConv2D):
    mode = mode_with_gpu
    dtype = 'float32'

    def setUp(self):
        super(GTestConv2D, self).setUp()
        self.input = ftensor4('input')
        self.filters = ftensor4('filters')

