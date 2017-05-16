from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet.ctc import ctc

def softmax():
    """ Compute softmax values of each set of scores in x. """
    pass

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

print("Activations:\n{0}".format(activations))
##print("Softmax outputs: {0}".format(softmax(activations)))

# Labels for each sequence
labels = np.asarray([[1, -1],
                     [3,  3],
                     [2,  3]], dtype=np.int32)

# Create symbolic variables
t_inputs = theano.shared(inputs, name="inputs")
t_weights = theano.shared(weights, name="weights")
t_activations = T.dot(t_inputs, t_weights)
t_activaction_times = theano.shared(activation_times, "activation_times")
t_labels = theano.shared(labels, "labels")

# Symbolic CTC cost
t_cost = ctc(t_activations, t_labels, t_activaction_times)
# Symbolic gradient of CTC cost
t_grad = T.grad(T.mean(t_cost), t_weights)

# Compile symbolic functions
ctc_func = theano.function([], [t_cost, t_grad])
cost, grad = ctc_func()

print("CTC costs:\n{0}".format(cost))
print("Gradient of avg. CTC cost w.r.t. weights:\n{0}".format(np.asarray(grad)))
