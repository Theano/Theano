from __future__ import (division, absolute_import, print_function)

import unittest
from nose.plugins.skip import SkipTest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.ctc import (ctc_available, ctc, ConnectionistTemporalClassification)


def setup_torch_case():
    # Test obtained from Torch tutorial at:
    # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md

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

    return [activations, labels, activation_times, expected_costs, expected_gradients]


def setup_ctc_case():
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

    return [activations, labels, activation_times, expected_costs, expected_gradients]


def setup_grad_case():
    activations = np.asarray([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
                              [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]],
                             dtype=np.float32)

    activation_times = np.asarray([2, 2], dtype=np.int32)

    labels = np.asarray([[1, 2], [1, 2]], dtype=np.int32)

    return [activations, labels, activation_times]


class TestCTC(unittest.TestCase):
    """
    Test Baidu CTC wrapper implementation.

    Expected values for costs and gradients are obtained through an external
    C implementation, that uses the library directly.
    """

    def setUp(self):
        if theano.config.mode == "FAST_COMPILE" or theano.config.cxx == "":
            raise SkipTest("We need a c compiler")

        if not ctc_available():
            self.skipTest('Optional library warp-ctc not available')

    def run_ctc(self, activations, labels, input_length, expected_costs, expected_grads):
        # Create symbolic variables
        t_activations = theano.shared(activations, name="activations")
        t_activation_times = theano.shared(input_length, name="activation_times")
        t_labels = theano.shared(labels, name="labels")

        t_cost = ctc(t_activations, t_labels, t_activation_times)
        # Symbolic gradient of CTC cost
        t_grad = T.grad(T.mean(t_cost), t_activations)
        # Compile symbolic functions
        train = theano.function([], [t_cost, t_grad])

        cost, grad = train()

        utt.assert_allclose(expected_grads / cost.shape[0], grad)
        utt.assert_allclose(expected_costs, cost)

        self.check_grads_disabled(t_activations, t_labels, t_activation_times)

    def check_grads_disabled(self, activations, labels, input_length):
        """
        Check if optimization to disable gradients is working
        """
        ctc_cost = ctc(activations, labels, input_length)
        ctc_function = theano.function([], [ctc_cost])
        for node in ctc_function.maker.fgraph.apply_nodes:
            if isinstance(node.op, ConnectionistTemporalClassification):
                assert (node.op.compute_grad is False)

    def test_torch_case(self):
        activations, labels, input_length, expected_costs, expected_grads = setup_torch_case()
        self.run_ctc(activations, labels, input_length, expected_costs, expected_grads)

    def test_ctc(self):
        activations, labels, input_length, expected_costs, expected_grads = setup_ctc_case()
        self.run_ctc(activations, labels, input_length, expected_costs, expected_grads)

    def test_verify_grad(self):
        def ctc_op_functor(labels, in_lengths):
            def wrapper(acts):
                # Create auxiliary symbolic variables
                t_activation_times = theano.shared(in_lengths, name="activation_times")
                t_labels = theano.shared(labels, name="labels")
                return ctc(acts, t_labels, t_activation_times)
            return wrapper

        activations, labels, activation_times = setup_grad_case()

        ctc_op = ctc_op_functor(labels, activation_times)

        utt.verify_grad(ctc_op, [activations])
