from __future__ import absolute_import, print_function, division
import copy
import unittest
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

import numpy
from six.moves import xrange

import theano
import theano.sandbox.cuda as cuda_ndarray
from theano.tensor.basic import _allclose
from theano.tests import unittest_tools as utt

if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda disabled')


def advantage(cpu_dt, gpu_dt):
    """
    Return ratio of cpu_dt / gpu_dt, which must be non-negative numbers.

    If both arguments are zero, return NaN.
    If only gpu_dt is zero, return Inf.
    """
    assert gpu_dt >= 0 and cpu_dt >= 0
    if gpu_dt == 0 and cpu_dt == 0:
        return numpy.nan
    elif gpu_dt == 0:
        return numpy.inf
    else:
        return cpu_dt / gpu_dt


def test_host_to_device():
    # print >>sys.stdout, 'starting test_host_to_dev'
    for shape in ((), (3,), (2, 3), (3, 4, 5, 6)):
        a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        c = numpy.asarray(b)
        assert numpy.all(a == c)

        # test with float32 dtype
        d = numpy.asarray(b, dtype='float32')
        assert numpy.all(a == d)

        # test with not float32 dtype
        try:
            numpy.asarray(b, dtype='int8')
            assert False
        except TypeError:
            pass


def test_add_iadd_idiv():
    for shapes in ([(5, 5), (5, 1)],
                   [(5, 5), (1, 5)],
                   (), (0,), (3,), (2, 3),
                   (1, 10000000), (10000, 1000), (1000000, 10),
                   (4100, 33, 34), (33, 4100, 34), (33, 34, 4100),
                   (4100, 33, 3, 6), (33, 4100, 3, 6), (33, 3, 4100, 6), (33, 3, 6, 4100),
                   (4100, 3, 34, 6), (3, 4100, 34, 6), (3, 34, 4100, 6), (3, 34, 6, 4100),
                   (4100, 3, 4, 36), (3, 4100, 4, 36), (3, 4, 4100, 36), (3, 4, 36, 4100),
                   (0, 0, 0, 0, 0),
                   (3, 34, 35, 36, 37),
                   (33, 34, 3, 36, 37),
                   (33, 34, 35, 36, 3),
                   (0, 0, 0, 0, 0, 0),
                   (3, 34, 35, 36, 37, 2),
                   (33, 34, 3, 36, 37, 2),
                   (33, 34, 35, 36, 3, 2),
                   (3, 4, 5, 6, 7, 1025),
                   (3, 4, 5, 6, 1025, 7),
                   (3, 4, 5, 1025, 6, 7),
                   (3, 4, 1025, 5, 6, 7),
                   (3, 1025, 4, 5, 6, 7),
                   (1025, 3, 4, 5, 6, 7),
                   ):
        if isinstance(shapes, tuple):
            shape = shapes
            shape2 = shapes
            a0 = theano._asarray(numpy.random.rand(*shape), dtype='float32')
            a0_orig = a0.copy()
            a1 = a0.copy()
            assert numpy.allclose(a0, a1)
        else:
            shape = shapes[0]
            shape2 = shapes[1]

            a0 = theano._asarray(numpy.random.rand(*shape), dtype='float32')
            a0_orig = a0.copy()
            a1 = theano._asarray(numpy.random.rand(*shape2), dtype='float32')

        b0 = cuda_ndarray.CudaNdarray(a0)
        b1 = cuda_ndarray.CudaNdarray(a1)
        assert numpy.allclose(a0, numpy.asarray(b0))
        assert numpy.allclose(a1, numpy.asarray(b1))

        # add don't support stride
        if shape == shape2:
            bsum = b0 + b1
            bsum = b0 + b1
            asum = a0 + a1
            asum = a0 + a1
            # print shape, 'adding ', a0.size, 'cpu', cpu_dt, 'advantage', advantage(cpu_dt, gpu_dt)
            assert numpy.allclose(asum, numpy.asarray(bsum))

        # test not contiguous version.
        # should raise not implemented.
        a0 = a0_orig.copy()
        b0 = cuda_ndarray.CudaNdarray(a0)
        if len(shape) == 0:
            continue
        elif len(shape) == 1:
            _b = b1[::-1]
        elif len(shape) == 2:
            _b = b1[::, ::-1]
        elif len(shape) == 3:
            _b = b1[::, ::, ::-1]
        elif len(shape) == 4:
            _b = b1[::, ::, ::, ::-1]
        elif len(shape) == 5:
            _b = b1[::, ::, ::, ::, ::-1]
        elif len(shape) == 6:
            _b = b1[::, ::, ::, ::, ::, ::-1]
        else:
            raise Exception("You need to modify this case!")
        # TODO: b0[...,::-1] don't work

        # test inplace version
        b0 += b1
        a0 += a1
        # print shape, 'adding inplace', a0.size, 'cpu', cpu_dt, 'advantage', advantage(cpu_dt, gpu_dt)
        assert numpy.allclose(a0, numpy.asarray(b0))
        assert numpy.allclose(a0, a0_orig + a1)

        b0 /= b1
        a0 /= a1
        assert numpy.allclose(a0, numpy.asarray(b0))
        assert numpy.allclose(a0, (a0_orig + a1) / a1)

        # test inplace version
        # for not contiguous input
        b0 += _b
        a0 += a1[..., ::-1]
        assert numpy.allclose(a0, numpy.asarray(b0))
        assert numpy.allclose(a0, (a0_orig + a1) / a1 + a1[..., ::-1])

        b0 /= _b
        a0 /= a1[..., ::-1]
        assert numpy.allclose(a0, numpy.asarray(b0))
        assert numpy.allclose(a0, ((a0_orig + a1) / a1 +
                                   a1[..., ::-1]) / a1[..., ::-1])


def test_exp():
    # print >>sys.stdout, 'starting test_exp'
    for shape in ((), (3,), (2, 3),
                  (1, 10000000), (10, 1000000),
                  (100, 100000), (1000, 10000), (10000, 1000)):
        a0 = theano._asarray(numpy.random.rand(*shape), dtype='float32')
        a1 = a0.copy()
        b0 = cuda_ndarray.CudaNdarray(a0)
        cuda_ndarray.CudaNdarray(a1)
        bsum = b0.exp()
        asum = numpy.exp(a1)
        # print shape, 'adding ', a0.size, 'cpu', cpu_dt, 'advantage', advantage(cpu_dt, gpu_dt)
        # c = numpy.asarray(b0+b1)
        if asum.shape:
            assert numpy.allclose(asum, numpy.asarray(bsum))


def test_copy():
    # print >>sys.stdout, 'starting test_copy'
    shape = (500, 499)
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')

    # print >>sys.stdout, '.. creating device object'
    b = cuda_ndarray.CudaNdarray(a)

    # print >>sys.stdout, '.. copy'
    c = copy.copy(b)
    # print >>sys.stdout, '.. deepcopy'
    d = copy.deepcopy(b)

    # print >>sys.stdout, '.. comparisons'
    assert numpy.allclose(a, numpy.asarray(b))
    assert numpy.allclose(a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))
    b += b
    assert numpy.allclose(a + a, numpy.asarray(b))
    assert numpy.allclose(a + a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))


def test_nvcc_bug():
    """
    The fct k_elemwise_unary_rowmajor_copy(used by cuda.copy()) in cuda_ndarray.cu
    is not well compiled with nvcc 3.0 and 3.1 beta. We found a workaround, so it
    sould work correctly. Without the workaround, this test fail.
    """
    shape = (5, 4)
    aa = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a = aa[::, ::-1]

    b = cuda_ndarray.CudaNdarray(aa)[::, ::-1]
    c = copy.copy(b)
    d = copy.deepcopy(b)

    assert numpy.allclose(a, numpy.asarray(b))
    assert numpy.allclose(a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))
    b += b
    assert numpy.allclose(a + a, numpy.asarray(b))
    assert numpy.allclose(a + a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))


class test_DimShuffle(unittest.TestCase):
    def test_dimshuffle(self):
        utt.seed_rng()
        rng = numpy.random.RandomState(utt.fetch_seed())

        # 2d -> 0d
        a = theano._asarray(rng.randn(1, 1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(numpy.transpose(a),
                              cuda_ndarray.dimshuffle(b, ()))

        # Test when we drop a axis that don't have shape 1
        a = theano._asarray(rng.randn(2, 1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        self.assertRaises(ValueError, cuda_ndarray.dimshuffle, b, ())

        # Test that we can't take a dimensions multiple time
        a = theano._asarray(rng.randn(2, 1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        self.assertRaises(ValueError, cuda_ndarray.dimshuffle, b, (1, 1))

        # 1d
        a = theano._asarray(rng.randn(3,), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(numpy.transpose(a),
                              cuda_ndarray.dimshuffle(b, (0,)))
        assert numpy.allclose(a[None, :, None],
                              cuda_ndarray.dimshuffle(b, (-1, 0, -1)))

        # 2d
        a = theano._asarray(rng.randn(3, 11), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(numpy.transpose(a),
                              cuda_ndarray.dimshuffle(b, (1, 0)))
        assert numpy.allclose(numpy.transpose(a)[None, :, None, :, None],
                              cuda_ndarray.dimshuffle(b, (-1, 1, -1, 0, -1)))

        # 2d -> 1d
        a = theano._asarray(rng.randn(1, 11), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(a[:],
                              cuda_ndarray.dimshuffle(b, (1,)))
        a = theano._asarray(rng.randn(11, 1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(a.reshape((11,)),
                              cuda_ndarray.dimshuffle(b, (0,)))

        # 3d
        a = theano._asarray(rng.randn(3, 4, 5), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(a, cuda_ndarray.dimshuffle(b, (0, 1, 2)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 1),
                              cuda_ndarray.dimshuffle(b, (1, 0, 2)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 2),
                              cuda_ndarray.dimshuffle(b, (2, 1, 0)))
        assert numpy.allclose(numpy.swapaxes(a, 1, 2),
                              cuda_ndarray.dimshuffle(b, (0, 2, 1)))
        assert numpy.allclose(numpy.swapaxes(a, 1, 2)[None, :, None, :, :, None],
                              cuda_ndarray.dimshuffle(b, (-1, 0, -1, 2, 1, -1)))

        # 4d
        a = theano._asarray(rng.randn(3, 11, 4, 5), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert numpy.allclose(numpy.swapaxes(a, 0, 1),
                              cuda_ndarray.dimshuffle(b, (1, 0, 2, 3)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 2),
                              cuda_ndarray.dimshuffle(b, (2, 1, 0, 3)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 3),
                              cuda_ndarray.dimshuffle(b, (3, 1, 2, 0)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 3),
                              cuda_ndarray.dimshuffle(b, (3, 1, 2, 0)))
        assert numpy.allclose(numpy.swapaxes(a, 0, 3)[None, :, None, :, :, :],
                              cuda_ndarray.dimshuffle(b, (-1, 3, -1, 1, 2, 0)))


def test_dot():
    # print >>sys.stdout, 'starting test_dot'

    utt.seed_rng()
    rng = numpy.random.RandomState(utt.fetch_seed())

    a0 = theano._asarray(rng.randn(4, 7), dtype='float32')
    a1 = theano._asarray(rng.randn(7, 6), dtype='float32')

    b0 = cuda_ndarray.CudaNdarray(a0)
    b1 = cuda_ndarray.CudaNdarray(a1)

    assert _allclose(numpy.dot(a0, a1), cuda_ndarray.dot(b0, b1))

    a1 = theano._asarray(rng.randn(6, 7), dtype='float32')
    b1 = cuda_ndarray.CudaNdarray(a1)

    numpy_version = numpy.dot(a0, a1.T)
    transposed = cuda_ndarray.dimshuffle(b1, (1, 0))
    cuda_version = cuda_ndarray.dot(b0, transposed)

    assert _allclose(numpy_version, cuda_version)

    a1 = theano._asarray(rng.randn(7, 6), dtype='float32')
    b1 = cuda_ndarray.CudaNdarray(a1)

    a0 = theano._asarray(rng.randn(7, 4), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)

    assert _allclose(numpy.dot(a0.T, a1),
                     cuda_ndarray.dot(
                         cuda_ndarray.dimshuffle(b0, (1, 0)), b1))

    a1 = theano._asarray(rng.randn(6, 7), dtype='float32')
    b1 = cuda_ndarray.CudaNdarray(a1)

    assert _allclose(
        numpy.dot(a0.T, a1.T),
        cuda_ndarray.dot(cuda_ndarray.dimshuffle(b0, (1, 0)),
                         cuda_ndarray.dimshuffle(b1, (1, 0))))


def test_sum():
    shape = (2, 3)
    a0 = theano._asarray(numpy.arange(shape[0] * shape[1]).reshape(shape),
                         dtype='float32')

    b0 = cuda_ndarray.CudaNdarray(a0)

    assert numpy.allclose(a0.sum(),
                          numpy.asarray(b0.reduce_sum([1, 1])))

    a0.sum(axis=0)
    b0.reduce_sum([1, 0])

    # print 'asum\n',a0sum
    # print 'bsum\n',numpy.asarray(b0sum)

    assert numpy.allclose(a0.sum(axis=0),
                          numpy.asarray(b0.reduce_sum([1, 0])))
    assert numpy.allclose(a0.sum(axis=1),
                          numpy.asarray(b0.reduce_sum([0, 1])))
    assert numpy.allclose(a0, numpy.asarray(b0.reduce_sum([0, 0])))

    shape = (3, 4, 5, 6, 7, 8)
    a0 = theano._asarray(numpy.arange(3 * 4 * 5 * 6 * 7 * 8).reshape(shape),
                         dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=5).sum(axis=3).sum(axis=0),
                          numpy.asarray(b0.reduce_sum([1, 0, 0, 1, 0, 1])))

    shape = (16, 2048)
    a0 = theano._asarray(numpy.arange(16 * 2048).reshape(shape),
                         dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=0), numpy.asarray(b0.reduce_sum([1, 0])))

    shape = (16, 10)
    a0 = theano._asarray(numpy.arange(160).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(), numpy.asarray(b0.reduce_sum([1, 1])))


def test_reshape():
    shapelist = [((1, 2, 3), (1, 2, 3)),
                 ((1,), (1,)),
                 ((1, 2, 3), (3, 2, 1)),
                 ((1, 2, 3), (6,)),
                 ((1, 2, 3, 2), (6, 2)),
                 ((2, 3, 2), (6, 2)),
                 ((2, 3, 2), (12,))
                 ]

    bad_shapelist = [
        ((1, 2, 3), (1, 2, 4)),
        ((1,), (2,)),
        ((1, 2, 3), (2, 2, 1)),
        ((1, 2, 3), (5,)),
        ((1, 2, 3, 2), (6, 3)),
        ((2, 3, 2), (5, 2)),
        ((2, 3, 2), (11,))
        ]

    utt.seed_rng()
    rng = numpy.random.RandomState(utt.fetch_seed())

    def subtest(shape_1, shape_2, rng):
        # print >> sys.stdout, "INFO: shapes", shape_1, shape_2
        a = theano._asarray(rng.randn(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)

        aa = a.reshape(shape_2)
        bb = b.reshape(shape_2)

        n_bb = numpy.asarray(bb)

        # print n_bb

        assert numpy.all(aa == n_bb)
        assert aa.shape == n_bb.shape

        # Test the not contiguous case
        shape_1_2x = (shape_1[0] * 2,) + shape_1[1:]
        a = theano._asarray(rng.randn(*shape_1_2x), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        a = a[::2]
        b = b[::2]

        aa = a.reshape(shape_2)
        bb = b.reshape(shape_2)

        n_bb = numpy.asarray(bb)

        # print n_bb

        assert numpy.all(aa == n_bb)
        assert aa.shape == n_bb.shape

    def bad_subtest(shape_1, shape_2, rng):
        a = theano._asarray(rng.randn(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)

        try:
            b.reshape(shape_2)
        except Exception:
            return
        assert False

    # test working shapes
    for shape_1, shape_2 in shapelist:
        subtest(shape_1, shape_2, rng)
        subtest(shape_2, shape_1, rng)

    # test shape combinations that should give error
    for shape_1, shape_2 in bad_shapelist:
        bad_subtest(shape_1, shape_2, rng)
        bad_subtest(shape_2, shape_1, rng)


def test_getshape():
    shapelist = [
        ((1, 2, 3), (1, 2, 3)),
        ((1,), (1,)),
        ((1, 2, 3), (3, 2, 1)),
        ((1, 2, 3), (6,)),
        ((1, 2, 3, 2), (6, 2)),
        ((2, 3, 2), (6, 2))
        ]

    def subtest(shape):
        a = theano._asarray(numpy.random.rand(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert b.shape == a.shape

    for shape_1, shape_2 in shapelist:
        subtest(shape_1)
        subtest(shape_2)


def test_stride_manipulation():

    a = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype='float32')
    b = cuda_ndarray.CudaNdarray(a)
    v = b.view()
    v._dev_data += 0
    c = numpy.asarray(v)
    assert numpy.all(a == c)

    sizeof_float = 4
    offset = 0

    b_strides = b._strides
    for i in xrange(len(b.shape)):
        offset += (b.shape[i] - 1) * b_strides[i]
        v._set_stride(i, -b_strides[i])

    v._dev_data += offset * sizeof_float
    c = numpy.asarray(v)

    assert numpy.all(c == [[5, 4, 3], [2, 1, 0]])


def test_subtensor_broadcastable():
    a = numpy.zeros((2, 7), dtype='float32')
    cuda_a = cuda_ndarray.CudaNdarray(a)
    # Will have shape (1, 7), so the stride in the first dim should be 0
    sub_a = cuda_a[1:]
    assert sub_a.shape == (1, 7)
    assert sub_a._strides[0] == 0


def test_copy_subtensor0():
    sizeof_float = 4
    a = theano._asarray(numpy.random.rand(30, 20, 5, 5), dtype='float32')
    cuda_a = cuda_ndarray.CudaNdarray(a)
    a_view = cuda_a.view()
    a_view_strides = a_view._strides
    a_view._set_stride(2, -a_view_strides[2])
    a_view._set_stride(3, -a_view_strides[3])
    a_view._dev_data += 24 * sizeof_float

    a_view_copy = copy.deepcopy(a_view)

    assert numpy.all(a[:, :, ::-1, ::-1] == numpy.asarray(a_view_copy))


def test_mapping_getitem_ellipsis():
    a = theano._asarray(numpy.random.rand(5, 4, 3, 2), dtype='float32')
    a = cuda_ndarray.CudaNdarray(a)

    b = a[...]
    assert b._dev_data == a._dev_data
    assert b._strides == a._strides
    assert b.shape == a.shape


def test_mapping_getitem_reverse_some_dims():
    dim = (5, 4, 3, 2)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    _b = _a[:, :, ::-1, ::-1]

    b = numpy.asarray(_b)
    assert numpy.all(b == a[:, :, ::-1, ::-1])


def test_mapping_getitem_w_int():
    def _cmp(x, y):
        assert x.shape == y.shape
        if not numpy.all(x == y):
            print(x)
            print(y)
        assert numpy.all(x == y)

    def _cmpf(x, *y):
        try:
            x.__getitem__(y)
        except IndexError:
            pass
        else:
            raise Exception("Did not generate out or bound error")

    def _cmpfV(x, *y):
        try:
            if len(y) == 1:
                x.__getitem__(*y)
            else:
                x.__getitem__(y)
        except ValueError:
            pass
        else:
            raise Exception("Did not generate out or bound error")

    dim = (2,)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[-1]), a[-1])
    _cmp(numpy.asarray(_a[0]), a[0])
    _cmp(numpy.asarray(_a[::1]), a[::1])
    _cmp(numpy.asarray(_a[::-1]), a[::-1])
    _cmp(numpy.asarray(_a[...]), a[...])
    _cmpf(_a, 2)

    dim = ()
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[...]), a[...])
    _cmpf(_a, 0)
    _cmpfV(_a, slice(1))

    dim = (5, 4, 3, 2)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    _cmpf(_a, slice(-1), slice(-1), 10, -10)
    _cmpf(_a, slice(-1), slice(-1), -10, slice(-1))
    _cmpf(_a, 0, slice(0, -1, -20), -10)
    _cmpf(_a, 10)
    _cmpf(_a, (10, 0, 0, 0))
    _cmpf(_a, -10)

    # test with integer
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[-1]), a[-1])
    _cmp(numpy.asarray(_a[numpy.int64(1)]), a[numpy.int64(1)])
    _cmp(numpy.asarray(_a[numpy.int64(-1)]), a[numpy.int64(-1)])

    # test with slice
    _cmp(numpy.asarray(_a[1:]), a[1:])
    _cmp(numpy.asarray(_a[1:2]), a[1:2])
    _cmp(numpy.asarray(_a[-1:1]), a[-1:1])

    # test with tuple (mix slice, integer, numpy.int64)
    _cmp(numpy.asarray(_a[:, :, ::numpy.int64(-1), ::-1]), a[:, :, ::-1, ::-1])
    _cmp(numpy.asarray(_a[:, :, numpy.int64(1), -1]), a[:, :, 1, -1])
    _cmp(numpy.asarray(_a[:, :, ::-1, ::-1]), a[:, :, ::-1, ::-1])
    _cmp(numpy.asarray(_a[:, :, ::-10, ::-10]), a[:, :, ::-10, ::-10])
    _cmp(numpy.asarray(_a[:, :, 1, -1]), a[:, :, 1, -1])
    _cmp(numpy.asarray(_a[:, :, -1, :]), a[:, :, -1, :])
    _cmp(numpy.asarray(_a[:, ::-2, -1, :]), a[:, ::-2, -1, :])
    _cmp(numpy.asarray(_a[:, ::-20, -1, :]), a[:, ::-20, -1, :])
    _cmp(numpy.asarray(_a[:, ::-2, -1]), a[:, ::-2, -1])
    _cmp(numpy.asarray(_a[0, ::-2, -1]), a[0, ::-2, -1])

    _cmp(numpy.asarray(_a[-1, -1, -1, -2]), a[-1, -1, -1, -2])
    _cmp(numpy.asarray(_a[...]), a[...])


def test_gemm_vector_vector():
    a = theano._asarray(numpy.random.rand(5, 1), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    b = theano._asarray(numpy.random.rand(1, 5), dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    _c = cuda_ndarray.dot(_a, _b)
    assert _c.shape == (5, 5)
    assert numpy.allclose(_c, numpy.dot(a, b))

    _c = cuda_ndarray.dot(_b, _a)
    assert _c.shape == (1, 1)
    assert numpy.allclose(_c, numpy.dot(b, a))

# ---------------------------------------------------------------------


def test_setitem_matrixscalar0():
    a = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray(8, dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # set an element to 8
    _a[1, 1] = _b
    a[1, 1] = b
    assert numpy.allclose(a, numpy.asarray(_a))

    # test direct transfert from numpy
    _a[1, 1] = theano._asarray(888, dtype='float32')
    a[1, 1] = theano._asarray(888, dtype='float32')
    assert numpy.allclose(a, numpy.asarray(_a))

    # broadcast a 0
    _a[1, 1] = 0
    _a[0:2] = 0
    _a[1:] = 0


def test_setitem_matrixvector1():
    a = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([8, 9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # set second column to 8,9
    _a[:, 1] = _b
    a[:, 1] = b
    assert numpy.allclose(a, numpy.asarray(_a))

    # test direct transfert from numpy
    _a[:, 1] = b * 100
    a[:, 1] = b * 100
    assert numpy.allclose(a, numpy.asarray(_a))

    row = theano._asarray([777, 888, 999], dtype='float32')
    _a[1, :] = row
    a[1, :] = row
    assert numpy.allclose(a, numpy.asarray(_a))


def test_setitem_matrix_tensor3():
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8, 9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # set middle row through cube to 7,8,9
    _a[:, 1, 1] = _b

    a[:, 1, 1] = b
    assert numpy.allclose(a, numpy.asarray(_a))

    # test direct transfert from numpy
    _a[:, 1, 1] = b * 100
    a[:, 1, 1] = b * 100
    assert numpy.allclose(a, numpy.asarray(_a))

    row = theano._asarray([777, 888, 999], dtype='float32')
    _a[1, 1, :] = row
    a[1, 1, :] = row
    assert numpy.allclose(a, numpy.asarray(_a))


def test_setitem_from_numpy_error():
    pass


def test_setitem_matrix_bad_shape():
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    try:
        # attempt to assign the ndarray b with setitem
        _a[:, 1, 1] = _b
        assert False
    except ValueError:
        # print e
        assert True

    # test direct transfert from numpy
    try:
        # attempt to assign the ndarray b with setitem
        _a[1, 1, :] = b
        assert False
    except ValueError:
        # print e
        assert True


def test_setitem_matrix_bad_ndim():
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    try:
        # attempt to assign the ndarray b with setitem
        _a[:, :, 1] = _b
        assert False
    except ValueError:
        # print e
        assert True

    # test direct transfert from numpy
    try:
        # attempt to assign the ndarray b with setitem
        _a[1, :, :] = b
        assert False
    except ValueError:
        # print e
        assert True


def test_setitem_matrix_bad_type():
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8], dtype='float64')

    # test direct transfert from numpy
    try:
        # attempt to assign the ndarray b with setitem
        _a[1, :, :] = b
        assert False
    except TypeError:
        # print e
        assert True


def test_setitem_assign_to_slice():
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8, 9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # first get a slice of a
    _c = _a[:, :, 1]

    # set middle row through cube to 7,8,9
    # (this corresponds to middle row of matrix _c)
    _c[:, 1] = _b

    a[:, :, 1][:, 1] = b
    assert numpy.allclose(a, numpy.asarray(_a))

    # test direct transfert from numpy
    _d = _a[1, :, :]
    _d[1, :] = b * 10
    a[1, :, :][1, :] = b * 10
    assert numpy.allclose(a, numpy.asarray(_a))


def test_setitem_broadcast():
    # test scalar to vector without stride
    a = numpy.arange(3)
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray(9, dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)
    _a[:] = _b.reshape((1,))
    a[:] = b.reshape((1,))
    assert numpy.allclose(numpy.asarray(_a), a)

    # test vector to matrice without stride
    a = numpy.arange(9)
    a.resize((3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8, 9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)
    _a[:, :] = _b.reshape((1, 3))
    a[:, :] = b.reshape((1, 3))
    assert numpy.allclose(numpy.asarray(_a), a)

    # test vector to matrice with stride
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([[7, 8, 9], [10, 11, 12]], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)[0]
    b = b[0]
    _a[:, :, 1] = _b.reshape((1, 3))
    a[:, :, 1] = b.reshape((1, 3))
    assert numpy.allclose(numpy.asarray(_a), a)


def test_setitem_broadcast_numpy():
    # test scalar to vector without stride
    a = numpy.arange(3)
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray(9, dtype='float32')
    _a[:] = b.reshape((1,))
    a[:] = b.reshape((1,))
    assert numpy.allclose(numpy.asarray(_a), a)

    # test vector to matrice without stride
    a = numpy.arange(9)
    a.resize((3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8, 9], dtype='float32')
    _a[:, :] = b.reshape((1, 3))
    a[:, :] = b.reshape((1, 3))
    assert numpy.allclose(numpy.asarray(_a), a)

    # test vector to matrice with stride
    a = numpy.arange(27)
    a.resize((3, 3, 3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([[7, 8, 9], [10, 11, 12]], dtype='float32')
    b = b[0]
    _a[1, :, :] = b.reshape((1, 3))
    a[1, :, :] = b.reshape((1, 3))
    assert numpy.allclose(numpy.asarray(_a), a)


# this also fails for the moment
def test_setitem_rightvalue_ndarray_fails():
    """
    Now we don't automatically add dimensions to broadcast
    """
    a = numpy.arange(3 * 4 * 5)
    a.resize((3, 4, 5))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7, 8, 9, 10], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)
    b5 = theano._asarray([7, 8, 9, 10, 11], dtype='float32')
    cuda_ndarray.CudaNdarray(b)

    # attempt to assign the ndarray b with setitem
    _a[:, :, 1] = _b
    a[:, :, 1] = b
    assert numpy.allclose(numpy.asarray(_a), a)

    # test direct transfert from numpy to contiguous region
    # attempt to assign the ndarray b with setitem
    # same number of dim
    mat = numpy.random.rand(4, 5).astype('float32')
    _a[2, :, :] = mat
    a[2, :, :] = mat
    assert numpy.allclose(numpy.asarray(_a), a)

    # without same number of dim
    try:
        _a[0, :, :] = mat
        # a[0, :, :] = mat
        # assert numpy.allclose(numpy.asarray(_a), a)
    except ValueError:
        pass

    # test direct transfert from numpy with broadcast
    _a[0, :, :] = b5
    a[0, :, :] = b5
    assert numpy.allclose(numpy.asarray(_a), a)

    # test direct transfert from numpy to not contiguous region
    # attempt to assign the ndarray b with setitem
    _a[:, :, 2] = b
    a[:, :, 2] = b
    assert numpy.allclose(numpy.asarray(_a), a)


def test_zeros_basic():
    for shp in [(3, 4, 5), (300,), (), (0, 7)]:
        _a = cuda_ndarray.CudaNdarray.zeros(shp)
        _n = numpy.zeros(shp, dtype="float32")
        assert numpy.allclose(numpy.asarray(_a), _n)
        assert _a.shape == _n.shape
        assert all(_a._strides == numpy.asarray(_n.strides) / 4)

    # TODO:The following don't have the same stride!
    #      This should be fixed with the new GpuNdArray.
    for shp in [(3, 0), (4, 1, 5)]:
        _a = cuda_ndarray.CudaNdarray.zeros(shp)
        _n = numpy.zeros(shp, dtype="float32")
        assert numpy.allclose(numpy.asarray(_a), _n)
        assert _a.shape == _n.shape

    try:
        _n = numpy.zeros()
    except TypeError:
        pass
    else:
        raise Exception("An error was expected!")
    try:
        _a = cuda_ndarray.CudaNdarray.zeros()
    except TypeError:
        pass
    else:
        raise Exception("An error was expected!")


def test_base():
    # Test that the 'base' attribute of a CudaNdarray is the one
    # built initially, not an intermediate one.
    a = cuda_ndarray.CudaNdarray.zeros((3, 4, 5))
    for i in xrange(5):
        b = a[:]
    assert b.base is a

    c = a[0]
    d = c[:, 0]
    # print d.shape
    assert c.base is a
    assert d.base is a

    e = b.reshape((5, 2, 2, 3))
    assert e.base is a


def test_set_strides():
    a = cuda_ndarray.CudaNdarray.zeros((5, 5))

    # Test with tuple
    new_strides = (a.strides[1], a.strides[0])
    a.strides = new_strides
    assert a.strides == new_strides

    # Test with list
    new_strides = (a.strides[1], a.strides[0])
    a.strides = [a.strides[1], a.strides[0]]
    assert a.strides == new_strides

    try:
        a.strides = (a.strides[1],)
        assert False
    except ValueError:
        pass

    try:
        a.strides = (1, 1, 1)
        assert False
    except ValueError:
        pass


def test_is_c_contiguous():
    a = cuda_ndarray.CudaNdarray.zeros((3, 4, 5))
    assert a.is_c_contiguous()
    assert a[1].is_c_contiguous()
    assert not a[::2].is_c_contiguous()

if __name__ == '__main__':
    test_setitem_matrixvector1()
    test_setitem_matrix_tensor3()
    test_setitem_assign_to_slice()
    test_setitem_rightvalue_ndarray_fails()
