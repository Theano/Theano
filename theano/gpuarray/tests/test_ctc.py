from __future__ import (division, absolute_import, print_function)

import unittest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
import theano.gpuarray
from theano.gpuarray.ctc import (gpu_ctc, GpuConnectionistTemporalClassification)
from theano.tensor.nnet.ctc import (ctc, ctc_available, ConnectionistTemporalClassification)
from .config import (mode_with_gpu, mode_without_gpu)


class TestCTC(unittest.TestCase):
    def setUp(self):
        if not ctc_available():
            self.skipTest('Optional library warp-ctc not available')

    def check_ctc(self, activations, labels, input_length, expected_costs, expected_grads):
        # Create symbolic variables
        t_activations = theano.shared(activations, name="activations")
        t_activation_times = theano.shared(input_length, name="activation_times")
        t_labels = theano.shared(labels, name="labels")

        inputs = [t_activations, t_labels, t_activation_times]

        # Execute several tests for each test case
        self.check_expected_values(t_activations, t_labels, t_activation_times, expected_costs, expected_grads)
        self.compare_gpu_and_cpu_values(*inputs)
        self.check_grads_disabled(*inputs)
        self.run_gpu_optimization_with_grad(*inputs)
        self.run_gpu_optimization_no_grad(*inputs)

    def setup_cpu_op(self, activations, labels, input_length, compute_grad=True, mode=mode_without_gpu):
        cpu_ctc_cost = ctc(activations, labels, input_length)
        outputs = [cpu_ctc_cost]
        if compute_grad:
            # Symbolic gradient of CTC cost
            cpu_ctc_grad = T.grad(T.mean(cpu_ctc_cost), activations)
            outputs += [cpu_ctc_grad]
        return theano.function([], outputs, mode=mode)

    def setup_gpu_op(self, activations, labels, input_length, compute_grad=True):
        gpu_ctc_cost = gpu_ctc(activations, labels, input_length)
        outputs = [gpu_ctc_cost]
        if compute_grad:
            # Symbolic gradient of CTC cost
            gpu_ctc_grad = T.grad(T.mean(gpu_ctc_cost), activations)
            outputs += [gpu_ctc_grad]
        return theano.function([], outputs)

    def check_expected_values(self, activations, labels, input_length, expected_costs, expected_grads):
        gpu_train = self.setup_gpu_op(activations, labels, input_length)
        gpu_cost, gpu_grad = gpu_train()
        # Transfer costs from GPU memory to host
        cost_from_gpu = np.asarray(gpu_cost)
        # Transfer gradients from GPU memory to host
        grad_from_gpu = np.asarray(gpu_grad)
        # Check that results are in conformance with expected values
        utt.assert_allclose(expected_grads / cost_from_gpu.shape[0], grad_from_gpu)
        utt.assert_allclose(expected_costs, cost_from_gpu)

    def compare_gpu_and_cpu_values(self, activations, labels, input_length):
        cpu_train = self.setup_cpu_op(activations, labels, input_length)
        cpu_cost, cpu_grad = cpu_train()

        gpu_train = self.setup_gpu_op(activations, labels, input_length)
        gpu_cost, gpu_grad = gpu_train()
        # Transfer costs from GPU memory to host
        cost_from_gpu = np.asarray(gpu_cost)
        # Transfer gradients from GPU memory to host
        grad_from_gpu = np.asarray(gpu_grad)
        # Check that results are in conformance with expected values
        utt.assert_allclose(cpu_grad, grad_from_gpu)
        utt.assert_allclose(cpu_cost, cost_from_gpu)

    def check_grads_disabled(self, activations, labels, input_length):
        """
        Check if optimization to disable gradients is working
        """
        gpu_ctc_cost = gpu_ctc(activations, labels, input_length)
        gpu_ctc_function = theano.function([], [gpu_ctc_cost])
        for node in gpu_ctc_function.maker.fgraph.apply_nodes:
            if isinstance(node.op, GpuConnectionistTemporalClassification):
                assert (node.op.compute_grad is False)

    def run_gpu_optimization_with_grad(self, activations, labels, input_length):
        # Compile CPU function with optimization
        cpu_lifted_train = self.setup_cpu_op(activations, labels, input_length, mode=mode_with_gpu)
        # Check whether Op is lifted to the GPU
        assert self.has_only_gpu_op(cpu_lifted_train)

    def run_gpu_optimization_no_grad(self, activations, labels, input_length):
        cpu_train = self.setup_cpu_op(activations, labels, input_length, compute_grad=False)
        cpu_cost = cpu_train()
        # Compile CPU function with optimization
        cpu_lifted_test = self.setup_cpu_op(activations, labels, input_length, compute_grad=False, mode=mode_with_gpu)
        # Check whether Op is lifted to the GPU
        assert self.has_only_gpu_op(cpu_lifted_test)
        gpu_cost = cpu_lifted_test()
        # Transfer costs from GPU memory to host
        cost_from_gpu = np.asarray(gpu_cost)
        # Compare values from CPU and GPU Ops
        utt.assert_allclose(cpu_cost, cost_from_gpu)

    def has_only_gpu_op(self, function):
        has_cpu_instance = False
        has_gpu_instance = False
        for node in function.maker.fgraph.apply_nodes:
            if isinstance(node.op, ConnectionistTemporalClassification):
                has_cpu_instance = True

            if isinstance(node.op, GpuConnectionistTemporalClassification):
                has_gpu_instance = True
        return has_gpu_instance and (not has_cpu_instance)

    # Test obtained from Torch tutorial at:
    # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
    def test_torch_case(self):
        # Layout, from slowest to fastest changing dimension, is (time, batchSize, inputLayerSize)
        activations = np.asarray([[[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1]],
                                  [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10], [-10, -9, -8, -7, -6]],
                                  [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15], [-15, -14, -13, -12, -11]]],
                                 dtype=np.float32)
        # Duration of each sequence
        activation_times = np.asarray([1, 3, 3], dtype=np.int32)
        # Labels for each sequence
        labels = np.asarray([[1, -1],
                             [3, 3],
                             [2, 3]], dtype=np.int32)

        expected_costs = np.asarray([1.609437943, 7.355742931, 4.938849926],
                                    dtype=np.float32)

        grads = [[[0.2, -0.8, 0.2, 0.2, 0.2],
                  [0.01165623125, 0.03168492019, 0.08612854034, -0.7658783197, 0.636408627],
                  [-0.02115798369, 0.03168492019, -0.8810571432, 0.2341216654, 0.636408627]],
                 [[0, 0, 0, 0, 0],
                  [-0.9883437753, 0.03168492019, 0.08612854034, 0.2341216654, 0.636408627],
                  [-0.02115798369, 0.03168492019, -0.1891518533, -0.4577836394, 0.636408627]],
                 [[0, 0, 0, 0, 0],
                  [0.01165623125, 0.03168492019, 0.08612854034, -0.7658783197, 0.636408627],
                  [-0.02115798369, 0.03168492019, 0.08612854034, -0.7330639958, 0.636408627]]]
        expected_gradients = np.asarray(grads, dtype=np.float32)

        self.check_ctc(activations, labels, activation_times, expected_costs, expected_gradients)

    def test_ctc(self):
        activations = np.asarray([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
                                  [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]],
                                 dtype=np.float32)

        activation_times = np.asarray([2, 2], dtype=np.int32)

        labels = np.asarray([[1, 2], [1, 2]], dtype=np.int32)

        expected_costs = np.asarray([2.962858438, 3.053659201], dtype=np.float32)

        grads = [[[0.177031219, -0.7081246376, 0.177031219, 0.177031219, 0.177031219],
                  [0.177031219, -0.8229685426, 0.291875124, 0.177031219, 0.177031219]],
                 [[0.291875124, 0.177031219, -0.8229685426, 0.177031219, 0.177031219],
                  [0.1786672771, 0.1786672771, -0.7334594727, 0.1974578798, 0.1786672771]]]

        expected_gradients = np.asarray(grads, dtype=np.float32)

        self.check_ctc(activations, labels, activation_times, expected_costs, expected_gradients)

    def test_verify_grad(self):
        def ctc_op_functor(labels, in_lengths):
            def wrapper(acts):
                # Create auxiliary symbolic variables
                t_activation_times = theano.shared(in_lengths, name="activation_times")
                t_labels = theano.shared(labels, name="labels")
                return gpu_ctc(acts, t_labels, t_activation_times)
            return wrapper

        activations = np.asarray([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
                                  [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]],
                                 dtype=np.float32)

        activation_times = np.asarray([2, 2], dtype=np.int32)

        labels = np.asarray([[1, 2], [1, 2]], dtype=np.int32)

        ctc_op = ctc_op_functor(labels, activation_times)

        utt.verify_grad(ctc_op, [activations])
