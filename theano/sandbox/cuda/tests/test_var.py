import numpy
from nose.plugins.skip import SkipTest

import theano
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

def test_givens():
    # Test that you can use a TensorType expression to replace a
    # CudaNdarrayType in the givens dictionary.
    # This test case uses code mentionned in #757
    data = numpy.float32([1,2,3,4])
    x = f32sc(data)
    y = x**2
    f = theano.function([x], y, givens={x:x+1})

def test_updates():
    # Test that you can use a TensorType expression to update a
    # CudaNdarrayType in the updates dictionary.
    data = numpy.float32([1,2,3,4])
    x = f32sc(data)
    y = x**2
    f = theano.function([], y, updates={x:x+1})

def test_updates2():
    # Test that you can use a TensorType expression to update a
    # CudaNdarrayType in the updates dictionary.
    # This test case uses code mentionned in #698
    data = numpy.random.rand(10,10).astype('float32')
    output_var = f32sc(name="output",
            value=numpy.zeros((10,10), 'float32'))

    x = theano.tensor.fmatrix('x')
    output_updates = {output_var:x**2}
    output_givens = {x:data}
    output_func = theano.function(inputs=[], outputs=[],
            updates=output_updates, givens=output_givens)
    output_func()
