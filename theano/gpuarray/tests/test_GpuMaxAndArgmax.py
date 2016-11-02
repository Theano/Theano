from theano.gpuarray import GpuArrayType
from theano.tests import unittest_tools as utt
import numpy as np
import theano
import theano.tensor as T

from .config import mode_with_gpu, mode_without_gpu
from .test_basic_ops import rand_gpuarray

test_shape = (1000, 100, 10, 5, 2)


def numpy_random_array(*shapes):
    dimlist = shapes
    size = 1
    for dimsize in dimlist:
        size *= dimsize
    return np.random.normal(size=size).astype(theano.config.floatX).reshape(dimlist)


def numpy_maxandargmax(X, axis=None):
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

# We run all tests with 5-D tensors of 10 000 000 elements.
# NB: In each test, any first call of theano function should be ignored
# with Theano config flag profiling.ignore_first_call=True.


def check_if_gpu_maxandargmax_in_graph(theano_function):
    assert len([node for node in theano_function.maker.fgraph.apply_nodes
                if isinstance(node.op, theano.gpuarray.reduction.GpuMaxAndArgmax)]) > 0


def check_if_gpu_maxandargmax_not_in_graph(theano_function):
    assert len([node for node in theano_function.maker.fgraph.apply_nodes
                if isinstance(node.op, theano.gpuarray.reduction.GpuMaxAndArgmax)]) == 0


def run_gpu_tensor5(test_matrix=None, axis=None):
    M = GpuArrayType(dtype=theano.config.floatX, broadcastable=(False,) * 5)()
    f = theano.function([M], [T.max(M, axis=axis), T.argmax(M, axis=axis)], name='GPU-function', mode=mode_with_gpu)
    check_if_gpu_maxandargmax_in_graph(f)
    if test_matrix is None:
        test_matrix = rand_gpuarray(*test_shape)
    f(test_matrix)
    theano_max, theano_argmax = f(test_matrix)
    ref_max, ref_argmax = numpy_maxandargmax(np.asarray(test_matrix), axis=axis)
    utt.assert_allclose(ref_max, theano_max)
    utt.assert_allclose(ref_argmax, theano_argmax)


def run_cpu_tensor5(test_matrix=None, axis=None):
    M = T.tensor5()
    f = theano.function([M], [T.max(M, axis=axis), T.argmax(M, axis=axis)], name='cpu-function', mode=mode_without_gpu)
    check_if_gpu_maxandargmax_not_in_graph(f)
    if test_matrix is None:
        test_matrix = numpy_random_array(*test_shape)
    f(test_matrix)
    theano_max, theano_argmax = f(test_matrix)
    ref_max, ref_argmax = numpy_maxandargmax(test_matrix, axis=axis)
    utt.assert_allclose(ref_max, theano_max)
    utt.assert_allclose(ref_argmax, theano_argmax)


def run_tensor5(axis=None):
    test_cpu_matrix = numpy_random_array(*test_shape)
    test_gpu_matrix = rand_gpuarray(*test_shape)
    run_cpu_tensor5(test_cpu_matrix, axis)
    run_gpu_tensor5(test_gpu_matrix, axis)


def test_none():
    run_tensor5(None)


def test_all_axes():
    run_tensor5((0, 1, 2, 3, 4))


def test_all_axes_unsorted():
    run_tensor5((4, 1, 3, 0, 2))


def test_axis_1():
    run_tensor5(0)


def test_axis_2():
    run_tensor5(1)


def test_axis_3():
    run_tensor5(2)


def test_axis_4():
    run_tensor5(3)


def test_axis_5():
    run_tensor5(4)


def test_2_axes():
    run_tensor5((0, 3))


def test_3_axes():
    run_tensor5((0, 3, 4))


def test_4_axes():
    run_tensor5((0, 1, 2, 4))
