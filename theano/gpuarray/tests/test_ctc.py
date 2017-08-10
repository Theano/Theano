from __future__ import (division, absolute_import, print_function)

import unittest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
import theano.gpuarray
from theano.gpuarray.ctc import (gpu_ctc, GpuConnectionistTemporalClassification)
from theano.tensor.nnet.ctc import (ctc, ctc_available, ConnectionistTemporalClassification)
from theano.tensor.nnet.tests.test_ctc import (setup_torch_case, setup_ctc_case, setup_grad_case)
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
        return theano.function([], outputs, mode=mode_with_gpu)

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
        activations, labels, activation_times, expected_costs, expected_grads = setup_torch_case()
        self.check_ctc(activations, labels, activation_times, expected_costs, expected_grads)

    def test_ctc(self):
        activations, labels, input_length, expected_costs, expected_grads = setup_ctc_case()
        self.check_ctc(activations, labels, input_length, expected_costs, expected_grads)

    def test_verify_grad(self):
        def ctc_op_functor(labels, in_lengths):
            def wrapper(acts):
                # Create auxiliary symbolic variables
                t_activation_times = theano.shared(in_lengths, name="activation_times")
                t_labels = theano.shared(labels, name="labels")
                return gpu_ctc(acts, t_labels, t_activation_times)
            return wrapper

        activations, labels, activation_times = setup_grad_case()

        ctc_op = ctc_op_functor(labels, activation_times)

        utt.verify_grad(ctc_op, [activations], mode=mode_with_gpu)
