from __future__ import absolute_import, print_function, division

import theano
import theano.tensor as T
import numpy as np


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
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype(theano.config.floatX)


def linear_transform_weights(input_dim, output_dim,
                             param_list=None, name=""):
    "theano shared variable given input and output dimension"
    weight_inialization = uniform(np.sqrt(2.0 / input_dim),
                                  (input_dim, output_dim))
    W = theano.shared(weight_inialization, name=name)

    assert(param_list is not None)

    param_list.append(W)
    return W


def bias_weights(length, param_list=None, name=""):
    "theano shared variable for bias unit, given length"
    bias_initialization = np.zeros(length).astype(theano.config.floatX)

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
    def __init__(self, input_dim, output_dim, input_layer, s0=None, name=""):
        '''Layers information'''
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.input_layer = input_layer
        self.X = input_layer.output()
        self.s0 = s0
        self.params = []

        '''Layers weights'''

        '''self.params is passed so that any parameters could be appended to it'''
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

        self.Y = states

    def output(self):
        return self.Y


class LSTM(Layer):
    def __init__(self, input_dim, output_dim, input_layer, s0=None, c0=None,
                 name=""):
        '''Layers information'''
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.input_layer = input_layer
        self.X = input_layer.output()
        self.s0 = s0
        self.c0 = c0
        self.params = []

        '''Layers weights'''

        '''self.params is passed so that any parameters could be appended to it'''
        self.W_i = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_i")
        self.b_wi = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wi")

        self.W_f = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_f")
        self.b_wf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wf")

        self.W_c = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_c")
        self.b_wc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wc")

        self.W_o = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name + ".W_o")
        self.b_wo = bias_weights((output_dim,), param_list=self.params, name=name + ".b_wo")

        self.R_i = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_i")
        self.b_ri = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ri")

        self.R_f = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_f")
        self.b_rf = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rf")

        self.R_c = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_c")
        self.b_rc = bias_weights((output_dim,), param_list=self.params, name=name + ".b_rc")

        self.R_o = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name + ".R_o")
        self.b_ro = bias_weights((output_dim,), param_list=self.params, name=name + ".b_ro")

        '''step through processed input to create output'''
        def step(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_i) + T.dot(h_tm1, self.R_i) + self.b_wi + self.b_ri)
            f_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_f) + T.dot(h_tm1, self.R_f) + self.b_wf + self.b_rf)
            o_t = T.nnet.sigmoid(
                T.dot(x_t, self.W_o) + T.dot(h_tm1, self.R_o) + self.b_ro + self.b_wo)

            c_hat_t = T.tanh(
                T.dot(x_t, self.W_c) + T.dot(h_tm1, self.R_c) + self.b_wc + self.b_rc)
            c_t = f_t * c_tm1 + i_t * c_hat_t
            h_t = o_t * T.tanh(c_t)

            return h_t, c_t

        outputs_info = [self.s0, self.c0]

        states, updates = theano.scan(
            fn=step,
            sequences=[self.X],
            outputs_info=outputs_info
            )

        self.Y = states[0]
        self.C = states[1]

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
