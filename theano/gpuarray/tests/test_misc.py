# Test that normaly could be outside gpuarray, to have all gpuarray
# tests in the same directory, we put them here.
from __future__ import absolute_import, print_function, division

import numpy as np

import theano
from theano import tensor
from theano.compile.nanguardmode import NanGuardMode

from .config import mode_with_gpu


def test_nan_guard_mode():
    # Also test that abs uint* and bool have c code.
    for dtype in ['uint8', 'int64', 'bool']:
        x = tensor.vector(dtype=dtype)
        y = x + 1
        mode = NanGuardMode(nan_is_error=True,
                            optimizer=mode_with_gpu.optimizer)
        f = theano.function([x], y, mode=mode)
        d = np.asarray([23, 7]).astype(dtype)
        assert np.allclose(f(d), d + 1)
