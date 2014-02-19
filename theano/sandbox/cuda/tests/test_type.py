import cPickle
from nose.tools import assert_raises
import numpy
import os.path
from theano import config
from theano.sandbox.cuda import cuda_available

if cuda_available:
    from theano.sandbox.cuda import CudaNdarray

# testfile created on cuda enabled machine using
# >>> with open('CudaNdarray.pkl', 'wb') as fp:
# >>> cPickle.dump(theano.sandbox.cuda.CudaNdarray(np.array([-42.0], dtype=np.float32)), fp)

def test_unpickle_flag_is_false_by_default():
    assert not config.experimental.unpickle_gpu_on_cpu

def test_unpickle_cudandarray_as_numpy_ndarray_flag0():
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = False

    testfile_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(testfile_dir, 'CudaNdarray.pkl')) as fp:
        if cuda_available:
            mat = cPickle.load(fp)
        else:
            assert_raises(ImportError, cPickle.load, fp)

    if cuda_available:
        assert isinstance(mat, CudaNdarray)
        assert numpy.asarray(mat)[0] == -42.0

    config.experimental.unpickle_gpu_on_cpu = oldflag


def test_unpickle_cudandarray_as_numpy_ndarray_flag1():
    oldflag = config.experimental.unpickle_gpu_on_cpu
    config.experimental.unpickle_gpu_on_cpu = True

    testfile_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(testfile_dir, 'CudaNdarray.pkl')) as fp:
        mat = cPickle.load(fp)

    assert isinstance(mat, numpy.ndarray)
    assert mat[0] == -42.0

    config.experimental.unpickle_gpu_on_cpu = oldflag
