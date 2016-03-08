from nose.plugins.skip import SkipTest

import theano.sandbox.gpuarray

if theano.sandbox.gpuarray.pygpu is None:
    raise SkipTest("pygpu not installed")

if not theano.sandbox.gpuarray.pygpu_activated:
    import theano.sandbox.cuda as cuda_ndarray
    if cuda_ndarray.cuda_available:
        cuda_ndarray.use('gpu', default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False)
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
