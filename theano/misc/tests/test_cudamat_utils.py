from __future__ import absolute_import, print_function, division
import numpy
import theano
from theano.misc.cudamat_utils import cudamat_available

if not cudamat_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest("gnumpy not installed. Skip test of theano op with pycuda "
                   "code.")

from theano.misc.cudamat_utils import (cudandarray_to_cudamat,
                                       cudamat_to_cudandarray)


def test(shape=(3, 4)):
    """
    Make sure that the cudamat conversion is exact.
    """
    gpu = theano.sandbox.cuda.basic_ops.gpu_from_host
    U = gpu(theano.tensor.fmatrix('U'))
    ii = theano.function([U], gpu(U + 1))

    A_cpu = numpy.asarray(numpy.random.rand(*shape), dtype="float32")
    A_cnd = theano.sandbox.cuda.CudaNdarray(A_cpu)
    A_cmat = cudandarray_to_cudamat(A_cnd)

    B_cnd = cudamat_to_cudandarray(A_cmat)
    B_cnd = ii(A_cnd)

    u = A_cnd.copy()
    u += theano.sandbox.cuda.CudaNdarray(numpy.asarray([[1]], dtype='float32'))
    u = numpy.asarray(u)
    v = numpy.asarray(B_cnd)
    w = A_cmat.add(1).asarray()

    assert abs(u - v).max() == 0
    assert abs(u - w.T.reshape(u.shape)).max() == 0
