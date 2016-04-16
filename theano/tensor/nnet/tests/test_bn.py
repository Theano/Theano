from __future__ import absolute_import, print_function, division
import theano
from theano.tests import unittest_tools as utt
import numpy

from theano.tensor.nnet.bn import batch_normalization


def test_bn():

    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    numpy.random.seed(1234)
    X = 1 + numpy.random.random([10, 20]).astype('float32')
    B = 1 + numpy.random.random([20]).astype('float32')
    G = 1 + numpy.random.random([20]).astype('float32')
    M = 1 + numpy.random.random([20]).astype('float32')
    V = 1 + numpy.random.random([20]).astype('float32')

    x = theano.tensor.matrix('x')
    b = theano.tensor.vector('b')
    g = theano.tensor.vector('g')
    m = theano.tensor.vector('m')
    v = theano.tensor.vector('v')

    bn_ref_op = bn_ref(x, g, b, m, v)
    f_ref = theano.function([x, b, g, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)
    for mode in ['low_mem', 'high_mem']:
        bn_op = batch_normalization(x, g, b, m, v, mode=mode)
        f = theano.function([x, b, g, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def bn(inputs, gamma, beta, mean, std):
            return batch_normalization(inputs, gamma, beta, mean, std, mode=mode)
        utt.verify_grad(bn, [X, G, B, M, V])

    bn_ref_op = bn_ref(x, g, b, x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True))
    f_ref = theano.function([x, b, g], [bn_ref_op])
    res_ref = f_ref(X, G, B)
    for mode in ['low_mem', 'high_mem']:
        bn_op = batch_normalization(x, g, b, x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True), mode=mode)
        f = theano.function([x, b, g], [bn_op])
        res = f(X, G, B)
        utt.assert_allclose(res_ref, res)

        def bn(inputs, gamma, beta, mean, std):
            return batch_normalization(inputs, gamma, beta, mean, std, mode=mode)
        utt.verify_grad(batch_normalization, [X, G, B,
                                              X.mean(axis=0)[numpy.newaxis], X.std(axis=0)[numpy.newaxis]])


def test_bn_feature_maps():

    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    numpy.random.seed(1234)
    X = 1 + numpy.random.random([10, 20, 4, 4]).astype('float32')
    B = 1 + numpy.random.random([20]).astype('float32')
    G = 1 + numpy.random.random([20]).astype('float32')
    M = 1 + numpy.random.random([20]).astype('float32')
    V = 1 + numpy.random.random([20]).astype('float32')

    x = theano.tensor.tensor4('x')
    b = theano.tensor.vector('b')
    g = theano.tensor.vector('g')
    m = theano.tensor.vector('m')
    v = theano.tensor.vector('v')

    bn_ref_op = bn_ref(x,
                       g.dimshuffle('x', 0, 'x', 'x'),
                       b.dimshuffle('x', 0, 'x', 'x'),
                       m.dimshuffle('x', 0, 'x', 'x'),
                       v.dimshuffle('x', 0, 'x', 'x'))
    f_ref = theano.function([x, b, g, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)

    for mode in ['low_mem', 'high_mem']:
        bn_op = batch_normalization(x,
                                    g.dimshuffle('x', 0, 'x', 'x'),
                                    b.dimshuffle('x', 0, 'x', 'x'),
                                    m.dimshuffle('x', 0, 'x', 'x'),
                                    v.dimshuffle('x', 0, 'x', 'x'),
                                    mode=mode)
        f = theano.function([x, b, g, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def conv_bn(inputs, gamma, beta, mean, std):
            return batch_normalization(inputs,
                                       gamma.dimshuffle('x', 0, 'x', 'x'),
                                       beta.dimshuffle('x', 0, 'x', 'x'),
                                       mean.dimshuffle('x', 0, 'x', 'x'),
                                       std.dimshuffle('x', 0, 'x', 'x'),
                                       mode=mode)
        utt.verify_grad(conv_bn, [X, G, B, M, V])
