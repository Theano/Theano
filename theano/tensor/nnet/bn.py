from __future__ import absolute_import, print_function, division
import theano
from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):
    init_param = ('dtype',)

    def __init__(self, dtype):
        self.dtype = dtype
        x = theano.scalar.Scalar(dtype=dtype).make_variable()
        mean = theano.scalar.Scalar(dtype=dtype).make_variable()
        std = theano.scalar.Scalar(dtype=dtype).make_variable()
        gamma = theano.scalar.Scalar(dtype=dtype).make_variable()
        beta = theano.scalar.Scalar(dtype=dtype).make_variable()
        o = add(mul(true_div(sub(x, mean), std), gamma), beta)
        inputs = [x, mean, std, gamma, beta]
        outputs = [o]
        super(BNComposite, self).__init__(inputs, outputs)

    def grad(self, inps, grads):
        x, mean, std, gamma, beta = inps
        top, = grads
        dx = (top * gamma) / std
        dmean = -(top * gamma) / std
        dstd = -(top * gamma * (x - mean)) / (std * std)
        dgamma = top * (x - mean) / std
        return [dx, dmean, dstd, dgamma, top]


def batch_normalization(inputs, gamma, beta, mean, std,
                        mode='low_mem'):
    """
    This function will build the symbolic graph for applying batch normalization
    to a set of activations.
    Also works on GPUs

    .. versionadded:: 0.7.1

    Parameters
    ----------
    inputs : symbolic tensor
        Mini-batch of activations
    gamma: symbolic tensor
        BN scale parameter, must be of same dimensionality as
        inputs and broadcastable against it
    beta: symbolic tensor
        BN shift parameter, must be of same dimensionality as
        inputs and broadcastable against it
    mean: symbolic tensor
        inputs means, must be of same dimensionality as
        inputs and broadcastable against it
    std: symbolic tensor
        inputs standard deviation, must be of same dimensionality as
        inputs and broadcastable against it
    mode: 'low_mem' or 'high_mem'
        Specify which batch_normalization implementation that will be
        used.
        As no intermediate representations are stored for the back-propagation,
        'low_mem' implementation lower the memory usage, however,
        it is 5-10% slower than 'high_mem' implementation. Note that 5-10% computation
        time difference compare the batch_normalization operation only, time difference
        between implementation is likely to be less important on the full model fprop/bprop.
    """
    if mode == 'low_mem':
        elm_bn = theano.tensor.elemwise.Elemwise(scalar_op=BNComposite(dtype=inputs.dtype))
        rval = elm_bn(inputs, mean, std, gamma, beta)
    elif mode == 'high_mem':
        rval = (inputs - mean) * (gamma / std) + beta
    else:
        raise ValueError(
            'mode must be either "low_mem", "high_mem"')
    return rval
