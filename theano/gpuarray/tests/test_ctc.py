from __future__ import (division, absolute_import, print_function)

import unittest
import numpy as np

import theano
import theano.tensor as T
from theano import config
from theano.tests import unittest_tools as utt
import theano.gpuarray
from theano.gpuarray.ctc import (ctc_enabled, ctc)

class TestCTC(unittest.TestCase):
    def setUp(self):
        if not ctc_enabled:
            self.skipTest('Optional library warp-ctc not available')

    def run_ctc(self, activations, labels, input_length, expected_costs, expected_grads):
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
        test_cost, = test()

        #utt.assert_allclose(expected_grads, grad)
        #utt.assert_allclose(expected_costs, cost)

    def simple_test(self):
        activations = np.asarray([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
                                  [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]],
                                 dtype='float32')

        activation_times = np.asarray([2, 2], dtype='int32')

        labels = np.asarray([[1, 2], [1, 2]], dtype='int32')

        expected_costs = np.asarray([2.962858438, 3.053659201], dtype='float32')

        grads = [[[0.177031219, -0.7081246376, 0.177031219, 0.177031219, 0.177031219],
                  [0.177031219, -0.8229685426, 0.291875124, 0.177031219, 0.177031219]],
                 [[0.291875124, 0.177031219, -0.8229685426, 0.177031219, 0.177031219],
                  [0.1786672771, 0.1786672771, -0.7334594727, 0.1974578798, 0.1786672771]]]

        expected_gradients = np.asarray(grads, dtype=np.float32)

        self.run_ctc(activations, labels, activation_times, expected_costs, expected_gradients)
