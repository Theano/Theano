from __future__ import print_function, absolute_import, division
from unittest import TestCase
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tests.unittest_tools import SkipTest

from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import rand_gpuarray
from .. import GpuArrayType

import math

# Number of values to be used in test tensors (except with 0-D tensors!).
test_size = 10000

# NB: This order of "unsorted axes" is arbitrary and is here
# just to have the same informations on profile output
# from one test to another.
unsorted_axes = (2, 4, 0, 3, 1)

np.random.seed()


def numpy_random_array(shapes):
    size = 1
    for dimsize in shapes:
        size *= dimsize
    return np.random.normal(size=size).astype(theano.config.floatX).reshape(shapes)


def numpy_maxandargmax(X, axis=None):
    if axis is None:
        axis = list(range(X.ndim))
    elif not isinstance(axis, (tuple, list)):
        axis = [int(axis)]
    axis = list(set(axis))  # remove duplicated values.
    axis.sort()
    axis = tuple(axis)
    ref_max = np.max(X, axis=axis)
    # Following code is copied from MaxAndArgmax.perform():
    # Numpy does not support multiple axes for argmax. Work around.
    keep_axes = np.array([i for i in range(X.ndim) if i not in axis], dtype='int64')
    # Not-reduced axes in front
    transposed_x = np.transpose(X, np.concatenate((keep_axes, axis)))
    kept_shape = transposed_x.shape[:len(keep_axes)]
    reduced_shape = transposed_x.shape[len(keep_axes):]
    new_shape = kept_shape + (np.prod(reduced_shape),)
    new_shape = tuple(int(i) for i in new_shape)
    reshaped_x = transposed_x.reshape(new_shape)
    return (ref_max, np.argmax(reshaped_x, axis=-1))


def check_if_gpu_maxandargmax_in_graph(theano_function):
    assert len([node for node in theano_function.maker.fgraph.apply_nodes
                if isinstance(node.op, theano.gpuarray.reduction.GpuMaxAndArgmax)]) > 0


def check_if_gpu_maxandargmax_not_in_graph(theano_function):
    assert len([node for node in theano_function.maker.fgraph.apply_nodes
                if isinstance(node.op, theano.gpuarray.reduction.GpuMaxAndArgmax)]) == 0


class BaseTest:
    # This attribute must be set in subclasses.
    tensor_size = None
    shape = None

    dtype = theano.config.floatX

    def get_shape(self):
        if self.tensor_size == 0:
            return []
        return [int(math.ceil(math.pow(test_size, 1 / self.tensor_size)))] * self.tensor_size

    def setUp(self):
        if not isinstance(self.tensor_size, int):
            raise SkipTest("No tensor ndim defined.")
        if self.tensor_size < 0 or self.tensor_size > 5:
            raise SkipTest("We allow from 0 (included) to 5 (inclued) dimensons for these tests.")
        if self.shape is None:
            self.shape = self.get_shape()

    def get_host_tensor(self):
        broadcastable = (False,) * self.tensor_size
        return T.tensor(self.dtype, broadcastable)

    def get_gpu_tensor(self):
        broadcastable = (False,) * self.tensor_size
        return GpuArrayType(self.dtype, broadcastable)()

    def get_host_value(self):
        return numpy_random_array(self.shape)

    def get_gpu_value(self):
        return rand_gpuarray(*self.shape)

    # NB: In compute_host() and compute_gpu(),
    # the first call of the theano function should be ignored in profiling,
    # with Theano config flag profiling.ignore_first_call=True.

    def compute_host(self, test_tensor, axis):
        M = self.get_host_tensor()
        f = theano.function([M], [T.max(M, axis=axis), T.argmax(M, axis=axis)],
                            name='shape:' + str(test_tensor.shape) + '/axis:' + str(axis) + '/HOST', mode=mode_without_gpu)
        check_if_gpu_maxandargmax_not_in_graph(f)
        f(test_tensor)
        theano_max, theano_argmax = f(test_tensor)
        ref_max, ref_argmax = numpy_maxandargmax(test_tensor, axis=axis)
        utt.assert_allclose(ref_max, theano_max)
        utt.assert_allclose(ref_argmax, theano_argmax)

    def compute_gpu(self, test_gpu_tensor, test_host_tensor, axis):
        M = self.get_gpu_tensor()
        f = theano.function([M], [T.max(M, axis=axis), T.argmax(M, axis=axis)],
                            name='shape:' + str(test_gpu_tensor.shape) + '/axis:' + str(axis) + '/GPU', mode=mode_with_gpu)
        check_if_gpu_maxandargmax_in_graph(f)
        f(test_gpu_tensor)
        theano_max, theano_argmax = f(test_gpu_tensor)
        ref_max, ref_argmax = numpy_maxandargmax(test_host_tensor, axis=axis)
        utt.assert_allclose(ref_max, theano_max)
        utt.assert_allclose(ref_argmax, theano_argmax)

    def compute(self, axis=None):
        # We want to run CPU op and GPU op on the same tensor randomly generated.
        test_gpu_tensor = self.get_gpu_value()
        test_host_tensor = np.asarray(test_gpu_tensor)
        self.compute_host(test_host_tensor, axis)
        self.compute_gpu(test_gpu_tensor, test_host_tensor, axis)

    def compute_axis(self, pos):
        if self.tensor_size != 1 and 0 <= pos < self.tensor_size:
            self.compute(pos)

    def compute_some_axes(self, count):
        if 0 <= count < self.tensor_size:
            self.compute([i for i in unsorted_axes if i < self.tensor_size][:count])

    # Equivalent to test reduction on all axes.
    def test_none(self):
        self.compute(None)

    def test_axis_1(self):
        self.compute_axis(0)

    def test_axis_2(self):
        self.compute_axis(1)

    def test_axis_3(self):
        self.compute_axis(2)

    def test_axis_4(self):
        self.compute_axis(3)

    def test_axis_5(self):
        self.compute_axis(4)

    # For the tests below, we expect CPU op to run with Python implementation.

    def test_2_axes(self):
        self.compute_some_axes(2)

    def test_3_axes(self):
        self.compute_some_axes(3)

    def test_4_axes(self):
        self.compute_some_axes(4)


class TestScalar(BaseTest, TestCase):
    tensor_size = 0


class TestVector(BaseTest, TestCase):
    tensor_size = 1


# Special case
class TestRow(BaseTest, TestCase):
    tensor_size = 2
    shape = [1, test_size]


# Special case
class TestColumn(BaseTest, TestCase):
    tensor_size = 2
    shape = [test_size, 1]


class TestMatrix(BaseTest, TestCase):
    tensor_size = 2


class TestTensor5(BaseTest, TestCase):
    tensor_size = 5
