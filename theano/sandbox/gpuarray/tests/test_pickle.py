"""Some pickle test when pygpu isn't there. The test when pygpu is
available are in test_type.py.

This is needed as we skip all the test file when pygpu isn't there in
regular test file.

"""
from __future__ import absolute_import, print_function, division
import os
import sys
from six import reraise

from nose.plugins.skip import SkipTest
from nose.tools import assert_raises
import numpy

import theano.sandbox.gpuarray
from theano.compat import PY3
from theano import config
from theano.misc.pkl_utils import CompatUnpickler

if not theano.sandbox.gpuarray.pygpu_activated:
    try:
        import pygpu
    except ImportError:
        pygpu = None
    import theano.sandbox.cuda as cuda_ndarray
    if pygpu and cuda_ndarray.cuda_available:
        cuda_ndarray.use('gpu', default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False)
        theano.sandbox.gpuarray.init_dev('cuda')

from .. import pygpu_activated  # noqa


def test_unpickle_gpuarray_as_numpy_ndarray_flag1():
    """Only test when pygpu isn't
    available. test_unpickle_gpuarray_as_numpy_ndarray_flag0 in
    test_type.py test it when pygpu is there.

    """
    if pygpu_activated:
        raise SkipTest("pygpu disabled")
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = False

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'GpuArray.pkl'

        with open(os.path.join(testfile_dir, fname), 'rb') as fp:
            if PY3:
                u = CompatUnpickler(fp, encoding="latin1")
            else:
                u = CompatUnpickler(fp)
            assert_raises(ImportError, u.load)
    finally:
        config.experimental.unpickle_gpu_on_cpu = oldflag


def test_unpickle_gpuarray_as_numpy_ndarray_flag2():
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = True

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'GpuArray.pkl'

        with open(os.path.join(testfile_dir, fname), 'rb') as fp:
            if PY3:
                u = CompatUnpickler(fp, encoding="latin1")
            else:
                u = CompatUnpickler(fp)
            try:
                mat = u.load()
            except ImportError:
                # Windows sometimes fail with nonsensical errors like:
                #   ImportError: No module named type
                #   ImportError: No module named copy_reg
                # when "type" and "copy_reg" are builtin modules.
                if sys.platform == 'win32':
                    exc_type, exc_value, exc_trace = sys.exc_info()
                    reraise(SkipTest, exc_value, exc_trace)
                raise

        assert isinstance(mat, numpy.ndarray)
        assert mat[0] == -42.0

    finally:
        config.experimental.unpickle_gpu_on_cpu = oldflag
