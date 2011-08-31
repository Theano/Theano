import numpy
from nose.plugins.skip import SkipTest

from theano.sandbox.cuda.var import float32_shared_constructor as f32sc
from theano.sandbox.cuda import CudaNdarrayType, cuda_available

import theano

# Skip test if cuda_ndarray is not available.
if cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

def test_shared_pickle():
    import pickle
    picklestring = "ctheano.tensor.sharedvar\nload_shared_variable\np0\n(cnumpy.core.multiarray\n_reconstruct\np1\n(cnumpy\nndarray\np2\n(I0\ntp3\nS'b'\np4\ntp5\nRp6\n(I1\n(I2\ntp7\ncnumpy\ndtype\np8\n(S'f4'\np9\nI0\nI1\ntp10\nRp11\n(I3\nS'<'\np12\nNNNI-1\nI-1\nI0\ntp13\nbI00\nS'\\x00\\x00\\x80?\\x00\\x00\\x00@'\np14\ntp15\nbtp16\nRp17\n."
    g = pickle.loads(picklestring)
    v = numpy.array([1.0, 2.0], dtype='float32')

    # This test will always be on the GPU
    assert isinstance(g, theano.tensor.basic.TensorVariable)
    assert isinstance(g.owner, theano.gof.graph.Apply)
    assert isinstance(g.owner.op, theano.sandbox.cuda.HostFromGpu)
    assert isinstance(g.owner.inputs[0], CudaNdarrayType.SharedVariable)
    assert (g.owner.inputs[0].get_value() == v).all()

    # Make sure it saves the same way (so that the tests before are not bogus)
    s = theano.tensor.as_tensor_variable(theano.shared(v))
    assert pickle.dumps(s) == picklestring

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
