from __future__ import absolute_import, print_function, division
import theano
from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):
    init_param = ('dtype',)

    @theano.configparser.change_flags(compute_test_value='off')
    def __init__(self, dtype, fused_grad=False):
        self.dtype = dtype
        self.fused_grad = fused_grad
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
        if self.fused_grad:
            assert top.dtype == x.dtype
            return BNCompositeGrad(top.dtype)(x, mean, std, gamma, top) + [top]

        top_gamma = top * gamma
        x_mean = x - mean
        dx = top_gamma / std
        dmean = -dx
        dstd = -(top_gamma * x_mean) / (std * std)
        dgamma = top * x_mean / std
        return [dx, dmean, dstd, dgamma, top]


class BNCompositeGrad(Composite):
    init_param = ('dtype',)

    @theano.configparser.change_flags(compute_test_value='off')
    def __init__(self, dtype):
        self.dtype = dtype
        x = theano.scalar.Scalar(dtype=dtype).make_variable()
        mean = theano.scalar.Scalar(dtype=dtype).make_variable()
        std = theano.scalar.Scalar(dtype=dtype).make_variable()
        gamma = theano.scalar.Scalar(dtype=dtype).make_variable()
        # beta isn't needed for the output
        # beta = theano.scalar.Scalar(dtype=dtype).make_variable()
        top = theano.scalar.Scalar(dtype=dtype).make_variable()

        top_gamma = top * gamma
        x_mean = x - mean
        dx = top_gamma / std
        dmean = -dx
        dstd = -(top_gamma * x_mean) / (std * std)
        dgamma = top * x_mean / std

        inputs = [x, mean, std, gamma, top]
        outputs = [dx, dmean, dstd, dgamma]
        super(BNCompositeGrad, self).__init__(inputs, outputs)


def batch_normalization(inputs, gamma, beta, mean, std,
                        mode='low_mem'):
    """
    This function will build the symbolic graph for applying batch
    normalization to a set of activations.

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
    mode: 'low_mem', 'high_mem' or 'low_mem_fast_opt'
        Specify which batch_normalization implementation that will be
        used.
        As no intermediate representations are stored for the back-propagation,
        'low_mem' implementation lower the memory usage, however,
        it is 5-10% slower than 'high_mem' implementation. Note that 5-10% computation
        time difference compare the batch_normalization operation only, time difference
        between implementation is likely to be less important on the full model fprop/bprop.
        low_mem_fast_opt is like low mem, but it build a pre fused gradient. So compilation
        should be faster.

    """
    if mode == 'low_mem':
        elm_bn = theano.tensor.elemwise.Elemwise(
            scalar_op=BNComposite(dtype=inputs.dtype))
        rval = elm_bn(inputs, mean, std, gamma, beta)
    elif mode == 'low_mem_fast_opt':
        elm_bn = theano.tensor.elemwise.Elemwise(
            scalar_op=BNComposite(dtype=inputs.dtype, fused_grad=True))
        rval = elm_bn(inputs, mean, std, gamma, beta)
    elif mode == 'high_mem':
        rval = (inputs - mean) * (gamma / std) + beta
    else:
        raise ValueError(
            'mode must be either "low_mem", "high_mem"')
    return rval
