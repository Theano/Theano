from __future__ import print_function

import unittest
import numpy as np

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.ctc import (ctc_enabled, ctc)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    broadcastShape = x.shape[0:2] + (1,)
    e_x = np.exp(x - np.max(x, axis=2).reshape(broadcastShape))
    return e_x / e_x.sum(axis=2).reshape(broadcastShape)

class TestCTC(unittest.TestCase):
    def setUp(self):
        if not ctc_enabled:
            self.skipTest('Optional library warp-ctc not available')

    def run_ctc(self, activations, labels, input_length):
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

        utt.assert_allclose(grad, softmax(activations))

    # Test obtained from Torch tutorial at:
    # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
    def test_torch_case(self):
        # Layout, from slowest to fastest changing dimension, is (time, batchSize, inputLayerSize)
        inputs = np.asarray([[[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, -6]],
                             [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, -11]],
                             [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, -16]]],
                            dtype=np.float32)

        weights = np.asarray([[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1]], dtype=np.float32)

        activations = np.dot(inputs, weights)
        # Duration of each sequence
        activation_times = np.asarray([1, 3, 3], dtype=np.int32)
        # Labels for each sequence
        labels = np.asarray([[1, -1],
                             [3, 3],
                             [2, 3]], dtype=np.int32)

        self.run_ctc(activations, labels, activation_times)
