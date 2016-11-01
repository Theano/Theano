from unittest import TestCase
from theano.tests import unittest_tools as utt
import numpy as np
import theano
import theano.tensor as T


def randomTensor(*shapes):
    dimlist = shapes
    size = 1
    for dimsize in dimlist:
        size *= dimsize
    return np.random.normal(size=size).astype(np.float32).reshape(dimlist)


def numpyMaxAndArgmax(X, axis=None):
    if axis is None:
        axis = range(X.ndim)
    elif not isinstance(axis, (tuple, list)):
        axis = [axis]
    axis = list(set(axis))  # remove duplicated values.
    axis.sort()
    axis = tuple(axis)
    ref_max = np.max(X, axis=axis)
    # Numpy does not support multiple axes for argmax. Work around
    # Code copied from MaxAndArgmax.perform()
    keep_axes = np.array([i for i in range(X.ndim) if i not in axis], dtype='int64')
    # Not-reduced axes in front
    transposed_x = np.transpose(X, np.concatenate((keep_axes, axis)))
    kept_shape = transposed_x.shape[:len(keep_axes)]
    reduced_shape = transposed_x.shape[len(keep_axes):]
    new_shape = kept_shape + (np.prod(reduced_shape),)
    reshaped_x = transposed_x.reshape(new_shape)
    return (ref_max, np.argmax(reshaped_x, axis=-1))


class TestGpuMaxAndArgmax(TestCase):
    # We run all tests with 5-D tensors of 10 000 000 elements.
    # NB: In each test, any first call of theano function should be ignored
    # with Theano config flag profiling.ignore_first_call=True.
    # To just check if GpuMaxAndArgmax is called:
    # $ theano-cache purge && THEANO_FLAGS=floatX=float32,device=cuda,profile=True,profiling.ignore_first_call=True \
    # nosetests --verbose theano/gpuarray/tests/test_GpuMaxAndArgmax.py:TestGpuMaxAndArgmax.test_none

    def _basic_test_tensor5(self, axis=None):
        M = T.tensor5()
        max_M = T.max(M, axis=axis)
        argmax_M = T.argmax(M, axis=axis)
        f = theano.function([M], [max_M, argmax_M])
        test_matrix = randomTensor(1000, 100, 10, 5, 2)
        f(test_matrix)
        theano_max, theano_argmax = f(test_matrix)
        ref_max, ref_argmax = numpyMaxAndArgmax(test_matrix, axis=axis)
        utt.assert_allclose(ref_max, theano_max)
        utt.assert_allclose(ref_argmax, theano_argmax)

    def _basic_test_assert_equals(self, axis1, axis2):
        M1 = T.tensor5()
        M2 = T.tensor5()
        f1 = theano.function([M1], [T.max(M1, axis=axis1), T.argmax(M1, axis=axis1)])
        f2 = theano.function([M2], [T.max(M2, axis=axis2), T.argmax(M2, axis=axis2)])
        test_matrix = randomTensor(1000, 100, 10, 5, 2)
        f1(test_matrix)
        f2(test_matrix)
        theano1 = f1(test_matrix)
        theano2 = f2(test_matrix)
        ref1 = numpyMaxAndArgmax(test_matrix, axis1)
        ref2 = numpyMaxAndArgmax(test_matrix, axis2)
        utt.assert_allclose(ref1, ref2)
        utt.assert_allclose(theano1, theano2)
        utt.assert_allclose(ref1, theano1)

    def test_none(self):
        self._basic_test_tensor5(None)

    def test_all_axes(self):
        self._basic_test_tensor5((0, 1, 2, 3, 4))

    def test_1_axe(self):
        self._basic_test_tensor5(3)

    def test_2_axes(self):
        self._basic_test_tensor5((0, 3))

    def test_3_axes(self):
        self._basic_test_tensor5((0, 3, 4))

    def test_4_axes(self):
        self._basic_test_tensor5((0, 1, 2, 4))

    def test_simple(self):
        self._basic_test_tensor5(None)
        self._basic_test_tensor5((0, 1, 2, 3, 4))
        self._basic_test_tensor5((4, 1, 3, 2))

    def test_assert_equals(self):
        self._basic_test_assert_equals(None, (0, 1, 2, 3, 4))
        self._basic_test_assert_equals(0, (0, 0))
        self._basic_test_assert_equals((4, 1, 3, 2), (1, 2, 3, 4))
        self._basic_test_assert_equals((4, 3, 2, 1, 0), None)
        self._basic_test_assert_equals((1, 3, 4), (1, 4, 4, 1, 3, 1, 3, 4, 3, 1, 1, 3, 1, 4, 1, 4))

    def test_simple_1_axis(self):
        self._basic_test_tensor5(0)
        self._basic_test_tensor5(1)
        self._basic_test_tensor5(2)
        self._basic_test_tensor5(3)
        self._basic_test_tensor5(4)

    def test_simple_2_axis(self):
        self._basic_test_tensor5((0, 0))
        self._basic_test_tensor5((0, 1))
        self._basic_test_tensor5((0, 2))
        self._basic_test_tensor5((0, 3))
        self._basic_test_tensor5((0, 4))

        self._basic_test_tensor5((1, 0))
        self._basic_test_tensor5((1, 1))
        self._basic_test_tensor5((1, 2))
        self._basic_test_tensor5((1, 3))
        self._basic_test_tensor5((1, 4))

        self._basic_test_tensor5((2, 0))
        self._basic_test_tensor5((2, 1))
        self._basic_test_tensor5((2, 2))
        self._basic_test_tensor5((2, 3))
        self._basic_test_tensor5((2, 4))

        self._basic_test_tensor5((3, 0))
        self._basic_test_tensor5((3, 1))
        self._basic_test_tensor5((3, 2))
        self._basic_test_tensor5((3, 3))
        self._basic_test_tensor5((3, 4))

        self._basic_test_tensor5((4, 0))
        self._basic_test_tensor5((4, 1))
        self._basic_test_tensor5((4, 2))
        self._basic_test_tensor5((4, 3))
        self._basic_test_tensor5((4, 4))
