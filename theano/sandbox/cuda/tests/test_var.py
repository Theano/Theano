import numpy
from nose.plugins.skip import SkipTest

from theano.sandbox.cuda.var import float32_shared_constructor as f32sc
from theano.sandbox.cuda import CudaNdarrayType, cuda_available

# Skip test if cuda_ndarray is not available.
if cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

def test_float32_shared_constructor():

    npy_row = numpy.zeros((1,10), dtype='float32')

    def eq(a,b):
        return a==b

    # test that we can create a CudaNdarray
    assert (f32sc(npy_row).type == CudaNdarrayType((False, False)))

    # test that broadcastable arg is accepted, and that they
    # don't strictly have to be tuples
    assert eq(
            f32sc(npy_row, broadcastable=(True, False)).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=[True, False]).type,
            CudaNdarrayType((True, False)))
    assert eq(
            f32sc(npy_row, broadcastable=numpy.array([True, False])).type,
            CudaNdarrayType([True, False]))

    # test that we can make non-matrix shared vars
    assert eq(
            f32sc(numpy.zeros((2,3,4,5), dtype='float32')).type,
            CudaNdarrayType((False,)*4))
