from __future__ import absolute_import, print_function, division

import theano
import theano.tensor as T
import numpy


class Model(object):
    def __init__(self, name=""):
        self.name = name
        self.layers = []
        self.params = []
        self.other_updates = {}

    def add_layer(self, layer):
        self.layers.append(layer)
        for p in layer.params:
            self.params.append(p)

        if hasattr(layer, 'other_updates'):
            for y in layer.other_updates:
                self.other_updates[y[0]] = y[1]

    def get_params(self):
        return self.params


def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)


def linear_transform_weights(input_dim, output_dim,
                             param_list=None, name=""):
    "theano shared variable given input and output dimension"
    weight_inialization = uniform(numpy.sqrt(2.0 / input_dim),
                                  (input_dim, output_dim))
    W = theano.shared(weight_inialization, name=name)

    assert(param_list is not None)

    param_list.append(W)
    return W


def bias_weights(length, param_list=None, name=""):
    "theano shared variable for bias unit, given length"
    bias_initialization = numpy.zeros(length).astype(theano.config.floatX)

    bias = theano.shared(
        bias_initialization,
        name=name
        )

    if param_list is not None:
        param_list.append(bias)

    return bias


class Layer(object):
    '''Generic Layer Template which all layers should inherit'''
    def __init__(self, name=""):
        self.name = name
        self.params = []

    def get_params(self):
        return self.params


class GRU(Layer):
    def __init__(self, input_dim, output_dim, input_layer, s0=None, batch_normalize=False, name=""):
        '''Layers information'''
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.input_layer = input_layer
        self.X = input_layer.output().dimshuffle(1, 0, 2)
        self.s0 = s0
        self.params = []

        '''Layers weights'''

        '''self.params is passed so that any paramters could be appended to it'''
        self.W_r = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_r")
        self.b_wr = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wr")

        self.W_i = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_i")
        self.b_wi = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wi")

        self.W_h = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_h")
        self.b_wh = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wh")

        self.R_r = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_r")
        self.b_rr = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rr")

        self.R_i = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_i")
        self.b_ru = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ru")

        self.R_h = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_h")
        self.b_rh = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rh")

        '''step through processed input to create output'''
        def step(inp, s_prev):
            i_t = T.nnet.sigmoid(
                T.dot(inp, self.W_i) + T.dot(s_prev, self.R_i) + self.b_wi + self.b_ru)
            r_t = T.nnet.sigmoid(
                T.dot(inp, self.W_r) + T.dot(s_prev, self.R_r) + self.b_wr + self.b_rr)

            h_hat_t = T.tanh(
                T.dot(inp, self.W_h) + (r_t * (T.dot(s_prev, self.R_h) + self.b_rh)) + self.b_wh)

            s_curr = ((1.0 - i_t) * h_hat_t) + (i_t * s_prev)

            return s_curr

        outputs_info = self.s0

        states, updates = theano.scan(
            fn=step,
            sequences=[self.X],
            outputs_info=outputs_info
            )

        self.Y = states.dimshuffle(1, 0, 2)

    def output(self):
        return self.Y


class FC(Layer):
    def __init__(self, input_dim, output_dim, input_layer, name=""):
        self.input_layer = input_layer
        self.name = name
        self.params = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = self.input_layer.output()

        self.W = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W")
        self.b = bias_weights((output_dim,), param_list=self.params, name=name + ".b")

    def output(self):
        return T.dot(self.X, self.W) + self.b


class WrapperLayer(Layer):
    def __init__(self, X, name=""):
        self.params = []
        self.name = name
        self.X = X

    def output(self):
        return self.X
