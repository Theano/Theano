from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest

import theano.tensor
import theano.gpuarray

if theano.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")


init_error = None
if (not theano.gpuarray.pygpu_activated and
        not theano.config.force_device):
    try:
        theano.gpuarray.init_dev('cuda')
    except Exception as e:
        init_error = e

if not theano.gpuarray.pygpu_activated:
    if init_error:
        raise SkipTest(init_error)
    else:
        raise SkipTest("pygpu disabled")

test_ctx_name = None

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')
    mode_without_gpu.check_py_code = False


# If using float16, cast reference input to float32
def ref_cast(x):
    if x.type.dtype == 'float16':
        x = theano.tensor.cast(x, 'float32')
    return x
