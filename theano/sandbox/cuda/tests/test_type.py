import cPickle
import numpy
import os.path
from theano.sandbox.cuda import cuda_available


if cuda_available:
    from theano.sandbox.cuda import CudaNdarray

def test_unpickle_cudandarray_as_numpy_ndarray():
    # testfile created on cuda enabled machine using
    # >>> with open('CudaNdarray.pkl', 'wb') as fp:
    # >>> cPickle.dump(theano.sandbox.cuda.CudaNdarray(np.array([-42.0], dtype=np.float32)), fp)

    testfile_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(testfile_dir, 'CudaNdarray.pkl')) as fp:
        mat = cPickle.load(fp)

        if cuda_available:
            assert isinstance(mat, CudaNdarray)
        else:
            assert isinstance(mat, numpy.ndarray)

        assert mat[0] == -42.0