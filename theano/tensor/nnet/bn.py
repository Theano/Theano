import theano
from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):

    def __init__(self, dtype):
        x = theano.scalar.Scalar(dtype=dtype).make_variable()
        mean = theano.scalar.Scalar(dtype=dtype).make_variable()
        var = theano.scalar.Scalar(dtype=dtype).make_variable()
        gamma = theano.scalar.Scalar(dtype=dtype).make_variable()
        beta = theano.scalar.Scalar(dtype=dtype).make_variable()
        o = add(mul(true_div(sub(x, mean),  var), gamma), beta)
        inputs = [x, mean, var, gamma, beta]
        outputs= [o]
        super(BNComposite, self).__init__(inputs, outputs)

    def grad(self, inps, grads):
        x, mean, var, gamma, beta = inps
        top, = grads
        dx = (top*gamma) / var
        dmean = -(top*gamma) / var
        dvar = -(top * gamma * (x - mean)) / (var*var)
        dgamma = top*(x - mean) / var
        return [dx, dmean, dvar, dgamma, top]


def batch_normalization(inputs, gamma, beta, mean, variance, axis=0):
    """
    This function will build the symbolic graph for applying batch normalization
    to a set of activations.

    Parameters
    ----------
    inputs : symbolic tensor
        Mini-batch of examples
    gamma: symbolic vector
        BN scale parameter, must be of same dimension that
        the number of inputs channel
    beta: symbolic vector
        BN shift parameter, must be of same dimension that
        the number of inputs channel
    mean: symbolic tensor
        inputs means
    variance: symbolic tensor
        inputs variance
    axis: int
        channel axis
    """
    elm_bn = theano.tensor.elemwise.Elemwise(scalar_op=BNComposite(dtype=inputs.dtype))
    rval = elm_bn(inputs, mean, variance, gamma, beta)
    return rval



