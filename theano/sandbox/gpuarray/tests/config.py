from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest

import theano.sandbox.gpuarray

if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")

if (not theano.sandbox.gpuarray.pygpu_activated and
        not theano.config.init_gpu_device.startswith('gpu')):
    theano.sandbox.gpuarray.init_dev('cuda')

if not theano.sandbox.gpuarray.pygpu_activated:
    raise SkipTest("pygpu disabled")

test_ctx_name = None

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')
