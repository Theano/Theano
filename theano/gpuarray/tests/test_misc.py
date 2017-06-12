# Test that normaly could be outside gpuarray, to have all gpuarray
# tests in the same directory, we put them here.

import theano
from theano import tensor
from theano.compile.nanguardmode import NanGuardMode

from .config import mode_with_gpu


def test_nan_guard_mode():
    x = tensor.vector(dtype='int64')
    y = x + 1
    mode = NanGuardMode(nan_is_error=True, optimizer=mode_with_gpu.optimizer)
    f = theano.function([x], y, mode=mode)
    theano.printing.debugprint(f)
    f([23, 7])
