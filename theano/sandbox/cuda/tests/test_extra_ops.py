# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.tensor.tests.test_extra_ops
from theano.sandbox.cuda.extra_ops import GpuCumsum

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestGpuCumsum(theano.tensor.tests.test_extra_ops.TestCumsumOp):
    mode = mode_with_gpu
    op = GpuCumsum
    dtypes = ['float32']
