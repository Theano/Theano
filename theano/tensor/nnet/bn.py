import theano
from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):

    def __init__(self, dtype):
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


def batch_normalization(inputs, gamma, beta, mean, std):
    """
    This function will build the symbolic graph for applying batch normalization
    to a set of activations. As no intermediate representations are stored for the
    back-propagation, this implementation lower the memory usage, however,
    it is 5-10% slower than a naive theano implementation, as it redo
    some foward computations for the backprop.

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
    """
    elm_bn = theano.tensor.elemwise.Elemwise(scalar_op=BNComposite(dtype=inputs.dtype))
    rval = elm_bn(inputs, mean, std, gamma, beta)
    return rval
