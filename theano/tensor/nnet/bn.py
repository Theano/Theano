from __future__ import absolute_import, print_function, division
import numpy
import theano
from theano import Apply, Op
from theano.gof import local_optimizer
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor import basic as T
from theano.tensor.opt import register_specialize_device
from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):
    init_param = ('dtype',)

    @theano.configparser.change_flags(compute_test_value='off')
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
        top_gamma = top * gamma
        x_mean = x - mean
        dx = top_gamma / std
        dmean = -dx
        dstd = -(top_gamma * x_mean) / (std * std)
        dgamma = top * x_mean / std
        return [dx, dmean, dstd, dgamma, top]


def batch_normalization(inputs, gamma, beta, mean, std,
                        mode='low_mem'):
    """
    This function will build the symbolic graph for applying batch normalization
    to a set of activations.
    Also works on GPUs, but is not optimized using cuDNN.

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


def batch_normalization_train(inputs, gamma, beta, axes='per-activation',
                              epsilon=1e-4):
    """
    Performs batch normalization of the given inputs, using the mean and
    variance of the inputs.

    Parameters
    ----------
    axes : 'per-activation', 'spatial' or a tuple of ints
        The axes along which the input should be normalized. ``'per-activation'``
        normalizes per activation and is equal to ``axes=(0,)``.
        ``'spatial'`` shares normalization factors across spatial dimensions
        (i.e., all dimensions past the second), which for 4D inputs would be
        equal to ``axes=(0, 2, 3)``.
    gamma : tensor
        Learnable scale factors. Must match the dimensionality of `inputs`,
        but have sizes of `1` for all axes normalized over (i.e., in the first
        dimension for ``axes='per-activation'``, and additionally in all
        dimensions past the second for ``axes='spatial'``).
    beta : tensor
        Learnable biases. Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).

    Returns
    -------
    out : tensor
        Batch-normalized inputs.
    mean : tensor
        Means of `inputs` across the normalization axes.
    stdinv : tensor
        Inverse standard deviations of `inputs` across the normalization axes.

    Notes
    -----
    For 5d and lower-dimensional inputs, and only if per-activation or spatial
    normalization is selected, this operation will use the cuDNN implementation.
    (This requires cuDNN 5 or newer.)

    The returned values are equivalent to:

    .. code-block:: python

        # for per-activation normalization
        axes = (0,)
        # for spatial normalization
        axes = (0,) + tuple(range(2, inputs.ndim))
        mean = inputs.mean(axes, keepdims=True)
        stdinv = T.inv(T.sqrt(inputs.var(axes, keepdims=True) + epsilon))
        out = (inputs - mean) * gamma * stdinv + beta
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError("gamma and beta must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, ndim))
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %f" % epsilon)

    if axes == 'per-activation':
        axes = (0,)
    elif axes == 'spatial':
        axes = (0,) + tuple(range(2, inputs.ndim))
    elif isinstance(axes, (tuple, list, numpy.ndarray)):
        axes = tuple(int(a) for a in axes)
    else:
        raise ValueError('invalid axes: %s', str(axes))
    if len(axes) == 0:
        raise ValueError('there should be at least one normalization axis')
    if min(axes) < 0 or max(axes) >= ndim:
        raise ValueError('axes should be less than ndim (<%d), but %s given' % (ndim, str(axes)))

    inputs = as_tensor_variable(inputs)
    gamma = as_tensor_variable(gamma)
    beta = as_tensor_variable(beta)

    gamma = T.addbroadcast(gamma, *axes)
    beta = T.addbroadcast(beta, *axes)

    batchnorm_op = AbstractBatchNormTrain(axes=axes)
    return tuple(batchnorm_op(inputs, gamma, beta, epsilon=epsilon))


def batch_normalization_test(inputs, gamma, beta, mean, var,
                             axes='per-activation', epsilon=1e-4):
    """
    Performs batch normalization of the given inputs, using the given mean and
    variance.

    Parameters
    ----------
    axes : 'per-activation', 'spatial' or a tuple of ints
        The axes along which the input should be normalized. ``'per-activation'``
        normalizes per activation and is equal to ``axes=(0,)``.
        ``'spatial'`` shares normalization factors across spatial dimensions
        (i.e., all dimensions past the second), which for 4D inputs would be
        equal to ``axes=(0, 2, 3)``.
    gamma : tensor
        Scale factors. Must match the dimensionality of `inputs`, but have
        sizes of `1` for all axes normalized over (i.e., in the first dimension
        for ``axes='per-activation'``, and additionally in all dimensions past
        the second for ``axes='spatial'``).
    beta : tensor
        Biases. Must match the tensor layout of `gamma`.
    mean : tensor
        Means. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    var : tensor
        Variances. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).

    Returns
    -------
    out : tensor
        Batch-normalized inputs.

    Notes
    -----
    For 5d and lower-dimensional inputs, and only if per-activation or spatial
    normalization is selected, this operation will use the cuDNN implementation.
    (This requires cuDNN 5 or newer.)

    The returned value is equivalent to:

    .. code-block:: python

        # for per-activation normalization
        axes = (0,)
        # for spatial normalization
        axes = (0,) + tuple(range(2, inputs.ndim))
        gamma, beta, mean, var = (T.addbroadcast(t, *axes)
                                  for t in (gamma, beta, mean, var))
        out = (inputs - mean) * gamma / T.sqrt(var + epsilon) + beta
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError("gamma and beta must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, ndim))
    if mean.ndim != ndim or var.ndim != ndim:
        raise ValueError("mean and var must be of the same dimensionality "
                         "as inputs; got %d and %d instead of %d" %
                         (mean.ndim, var.ndim, ndim))
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %f" % epsilon)

    if axes == 'per-activation':
        axes = (0,)
    elif axes == 'spatial':
        axes = (0,) + tuple(range(2, inputs.ndim))
    elif isinstance(axes, (tuple, list, numpy.ndarray)):
        axes = tuple(int(a) for a in axes)
    else:
        raise ValueError('invalid axes: %s', str(axes))
    if len(axes) == 0:
        raise ValueError('there should be at least one normalization axis')
    if min(axes) < 0 or max(axes) >= ndim:
        raise ValueError('axes should be less than ndim (<%d), but %s given' % (ndim, str(axes)))

    gamma = as_tensor_variable(gamma)
    beta = as_tensor_variable(beta)
    mean = as_tensor_variable(mean)
    var = as_tensor_variable(var)

    gamma = T.addbroadcast(gamma, *axes)
    beta = T.addbroadcast(beta, *axes)
    mean = T.addbroadcast(mean, *axes)
    var = T.addbroadcast(var, *axes)

    batchnorm_op = AbstractBatchNormInference(axes=axes)
    return batchnorm_op(inputs, gamma, beta, mean, var, epsilon=epsilon)


class AbstractBatchNormTrain(Op):
    """
    Abstract Op for Batch Normalization.

    Parameters
    ----------
    axes : a tuple of ints
        The axes along which the input should be normalized.
    x : tensor
        The input to be normalized along `axes`.
    scale : tensor
        `scale` should have the same number of dimensions as `x`.
        All dimensions listed in `axes` should have length 1.
    bias : tensor
        `bias` should have the same number of dimensions as `x`.
        All dimensions listed in `axes` should have length 1.
    epsilon
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    """

    __props__ = ('axes',)

    def __init__(self, axes=(0,)):
        assert isinstance(axes, (tuple, list))
        assert len(axes) > 0
        axes = tuple(int(a) for a in axes)
        self.axes = axes

    def infer_shape(self, node, shape):
        return [shape[0], shape[1], shape[1]]

    def make_node(self, x, scale, bias, epsilon=1e-4):
        assert x.ndim == scale.ndim == bias.ndim
        if not isinstance(epsilon, theano.Variable):
            epsilon = as_tensor_variable(epsilon)
        return Apply(self, [x, scale, bias, epsilon], [x.type(), scale.type(), scale.type()])

    def grad(self, inputs, grads):
        x, scale, bias, epsilon = inputs
        dy = grads[0]
        _, x_mean, x_invstd = self(x, scale, bias, epsilon)
        return AbstractBatchNormTrainGrad(self.axes)(
            x, dy, scale, x_mean, x_invstd, epsilon) + [theano.gradient.DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that epsilon is not connected to outputs.
        return [[True, True, True], [True, True, True], [True, True, True],
                [False, False, False]]

    def perform(self, node, inputs, output_storage):
        x, scale, bias, epsilon = inputs
        axes = self.axes
        if min(axes) < 0 or max(axes) >= x.ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (x.ndim, str(axes)))

        mean = x.mean(axes, keepdims=True)
        stdinv = 1.0 / numpy.sqrt(x.var(axes, keepdims=True) + epsilon)
        out = (x - mean) * (scale * stdinv) + bias

        output_storage[0][0] = out
        output_storage[1][0] = mean
        output_storage[2][0] = stdinv


class AbstractBatchNormInference(Op):
    """
    Abstract Op for Batch Normalization.

    Parameters
    ----------
    axes : a tuple of ints
        The axes along which the input is normalized.
    epsilon
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    """

    __props__ = ('axes',)

    def __init__(self, axes=(0,)):
        assert isinstance(axes, (tuple, list))
        assert len(axes) > 0
        axes = tuple(int(a) for a in axes)
        self.axes = axes

    def infer_shape(self, node, shape):
        return [shape[0]]

    def make_node(self, x, scale, bias, estimated_mean, estimated_variance, epsilon=1e-4):
        assert x.ndim == scale.ndim == bias.ndim == estimated_mean.ndim == estimated_variance.ndim
        if not isinstance(epsilon, theano.Variable):
            epsilon = as_tensor_variable(epsilon)
        return Apply(self, [x, scale, bias, estimated_mean, estimated_variance, epsilon], [x.type()])

    def grad(self, inputs, grads):
        x, scale, bias, est_mean, est_var, epsilon = inputs
        dy = grads[0]
        axes = self.axes
        if min(axes) < 0 or max(axes) >= x.ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (x.ndim, str(axes)))

        scale, bias, est_mean, est_var = (theano.tensor.addbroadcast(t, *axes)
                                          for t in (scale, bias, est_mean, est_var))

        # define helper expressions
        est_var_eps = est_var + epsilon
        est_std = theano.tensor.sqrt(est_var_eps)
        two = theano.tensor.constant(2.)

        # define and return gradients
        dx = dy * (scale / est_std)
        dscale = (dy * (x - est_mean)).sum(axes, keepdims=True) / est_std
        dbias = dy.sum(axes, keepdims=True)
        dmean = -dy.sum(axes, keepdims=True) * (scale / est_std)
        dvar = -(dy * (x - est_mean)).sum(axes, keepdims=True) * (scale / (two * est_var_eps * est_std))
        return [dx, dscale, dbias, dmean, dvar, theano.gradient.DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that epsilon is not connected to outputs.
        return [[True], [True], [True], [True], [True], [False]]

    def perform(self, node, inputs, output_storage):
        x, scale, bias, estimated_mean, estimated_variance, epsilon = inputs
        out = (x - estimated_mean) * (scale / numpy.sqrt(estimated_variance + epsilon)) + bias
        output_storage[0][0] = out


class AbstractBatchNormTrainGrad(Op):
    __props__ = ('axes',)

    def __init__(self, axes=(0,)):
        assert isinstance(axes, (tuple, list))
        assert len(axes) > 0
        axes = tuple(int(a) for a in axes)
        self.axes = axes

    def make_node(self, x, dy, scale, x_mean, x_invstd, epsilon=1e-4):
        assert x.ndim == dy.ndim == scale.ndim == x_mean.ndim == x_invstd.ndim
        if not isinstance(epsilon, theano.Variable):
            epsilon = as_tensor_variable(epsilon)
        return Apply(self, [x, dy, scale, x_mean, x_invstd, epsilon],
                     [x.type(), scale.type(), scale.type()])

    def infer_shape(self, node, shape):
        return [shape[0], shape[2], shape[2]]

    def perform(self, node, inputs, output_storage):
        x, dy, scale, x_mean, x_invstd, epsilon = inputs
        axes = self.axes
        if min(axes) < 0 or max(axes) >= x.ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (x.ndim, str(axes)))

        x_diff = x - x_mean
        mean_dy_x_diff = numpy.mean(dy * x_diff, axis=axes, keepdims=True)
        c = (dy * x_invstd) - (x_diff * mean_dy_x_diff * (x_invstd ** 3))

        g_wrt_inputs = scale * (c - numpy.mean(c, axis=axes, keepdims=True))
        g_wrt_scale = numpy.sum(dy * x_invstd * x_diff, axis=axes, keepdims=True)
        g_wrt_bias = numpy.sum(dy, axis=axes, keepdims=True)

        output_storage[0][0] = g_wrt_inputs
        output_storage[1][0] = g_wrt_scale
        output_storage[2][0] = g_wrt_bias


@local_optimizer([AbstractBatchNormTrain])
def local_abstract_batch_norm_train(node):
    if not isinstance(node.op, AbstractBatchNormTrain):
        return None

    x, scale, bias, epsilon = node.inputs
    axes = node.op.axes
    if min(axes) < 0 or max(axes) > x.ndim:
        return None
    if not isinstance(x.type, TensorType) or \
       not isinstance(scale.type, TensorType) or \
       not isinstance(bias.type, TensorType) or \
       not isinstance(epsilon.type, TensorType):
        return None

    mean = x.mean(axes, keepdims=True)
    stdinv = T.inv(T.sqrt(x.var(axes, keepdims=True) + epsilon))
    out = (x - mean) * (scale * stdinv) + bias
    # TODO copy_stack_trace?
    return [out, mean, stdinv]


@local_optimizer([AbstractBatchNormTrainGrad])
def local_abstract_batch_norm_train_grad(node):
    if not isinstance(node.op, AbstractBatchNormTrainGrad):
        return None

    x, dy, scale, x_mean, x_invstd, epsilon = node.inputs
    axes = node.op.axes
    if min(axes) < 0 or max(axes) > x.ndim:
        return None
    if not isinstance(x.type, TensorType) or \
       not isinstance(dy.type, TensorType) or \
       not isinstance(scale.type, TensorType) or \
       not isinstance(x_mean.type, TensorType) or \
       not isinstance(x_invstd.type, TensorType) or \
       not isinstance(epsilon.type, TensorType):
        return None

    x_diff = x - x_mean
    mean_dy_x_diff = T.mean(dy * x_diff, axis=axes, keepdims=True)
    c = (dy * x_invstd) - x_diff * (mean_dy_x_diff * (x_invstd ** 3))

    g_wrt_inputs = scale * (c - T.mean(c, axis=axes, keepdims=True))
    g_wrt_scale = T.sum(dy * x_invstd * x_diff, axis=axes, keepdims=True)
    g_wrt_bias = T.sum(dy, axis=axes, keepdims=True)
    # TODO copy_stack_trace?
    return [g_wrt_inputs, g_wrt_scale, g_wrt_bias]


@local_optimizer([AbstractBatchNormInference])
def local_abstract_batch_norm_inference(node):
    if not isinstance(node.op, AbstractBatchNormInference):
        return None

    x, scale, bias, estimated_mean, estimated_variance, epsilon = node.inputs

    if not isinstance(x.type, TensorType) or \
       not isinstance(scale.type, TensorType) or \
       not isinstance(bias.type, TensorType) or \
       not isinstance(estimated_mean.type, TensorType) or \
       not isinstance(estimated_variance.type, TensorType) or \
       not isinstance(epsilon.type, TensorType):
        return None

    # TODO copy_stack_trace?
    return [(x - estimated_mean) * (scale / T.sqrt(estimated_variance + epsilon)) + bias]


# Register Cpu Optmization
bn_groupopt = theano.gof.optdb.LocalGroupDB()
bn_groupopt.__name__ = 'batchnorm_opts'
register_specialize_device(bn_groupopt, 'fast_compile', 'fast_run')

bn_groupopt.register('local_abstract_batch_norm_train',
                     local_abstract_batch_norm_train, 30,
                     'fast_compile', 'fast_run')
bn_groupopt.register('local_abstract_batch_norm_train_grad',
                     local_abstract_batch_norm_train_grad, 30,
                     'fast_compile', 'fast_run')
bn_groupopt.register('local_abstract_batch_norm_inference',
                     local_abstract_batch_norm_inference, 30,
                     'fast_compile', 'fast_run')
