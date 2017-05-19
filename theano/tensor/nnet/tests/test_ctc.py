from __future__ import print_function

import unittest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.ctc import (ctc_enabled, ctc)


class TestCTC(unittest.TestCase):
    def setUp(self):
        if not ctc_enabled:
            self.skipTest('Optional library warp-ctc not available')

    def run_ctc(self, activations, labels, input_length, expected_costs):
        # Check if softmax probabilites are approximately equal to the gradients
        # of the activations, using utt.assert_allclose(a, b)

        # Create symbolic variables
        t_activations = theano.shared(activations, name="activations")
        t_activation_times = theano.shared(input_length, name="activation_times")
        t_labels = theano.shared(labels, name="labels")

        t_cost = ctc(t_activations, t_labels, t_activation_times)
        # Symbolic gradient of CTC cost
        t_grad = T.grad(T.mean(t_cost), t_activations)
        # Compile symbolic functions
        train = theano.function([], [t_cost, t_grad])
        test = theano.function([], [t_cost])

        cost, grad = train()
        cost, = test()

        utt.assert_allclose(cost, expected_costs)

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

        expected_costs = np.asarray([3.03655, 7.35574, 4.93884],
                                    dtype=np.float32)

        self.run_ctc(activations, labels, activation_times, expected_costs)

