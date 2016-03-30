from __future__ import absolute_import, print_function, division
import os.path
import sys
from six import reraise

from nose.plugins.skip import SkipTest
from nose.tools import assert_raises
import numpy

from theano import config
from theano.compat import PY3
from theano.misc.pkl_utils import CompatUnpickler
from theano.sandbox.cuda import cuda_available

if not cuda_available:
    raise SkipTest('Optional package cuda disabled')
else:
    from theano.sandbox.cuda import CudaNdarray

# testfile created on cuda enabled machine using
# >>> with open('CudaNdarray.pkl', 'wb') as fp:
# >>> cPickle.dump(theano.sandbox.cuda.CudaNdarray(np.array([-42.0], dtype=np.float32)), fp)


def test_unpickle_flag_is_false_by_default():
    assert not config.experimental.unpickle_gpu_on_cpu, (
        "Config flag experimental.unpickle_gpu_on_cpu is "
        "set to true. Make sure the default value stays false "
        "and that you have not set the flag manually.")


def test_unpickle_cudandarray_as_numpy_ndarray_flag0():
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = False

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'CudaNdarray.pkl'

        with open(os.path.join(testfile_dir, fname), 'rb') as fp:
            if PY3:
                u = CompatUnpickler(fp, encoding="latin1")
            else:
                u = CompatUnpickler(fp)
            if cuda_available:
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

                assert isinstance(mat, CudaNdarray)
                assert numpy.asarray(mat)[0] == -42.0
            else:
                assert_raises(ImportError, u.load)
    finally:
        config.experimental.unpickle_gpu_on_cpu = oldflag


def test_unpickle_cudandarray_as_numpy_ndarray_flag1():
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = True

    try:
        testfile_dir = os.path.dirname(os.path.realpath(__file__))
        fname = 'CudaNdarray.pkl'

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
