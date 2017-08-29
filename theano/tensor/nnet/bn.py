from __future__ import absolute_import, print_function, division
import numpy as np
import theano
from theano import Apply, Op
from theano.gof import local_optimizer
from theano.gof.opt import copy_stack_trace
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor import basic as T
from theano.tensor.opt import register_specialize_device
from theano.scalar import Composite, as_common_dtype
from theano.scalar import add, sub, true_div, mul


class BNComposite(Composite):
    init_param = ('dtype',)

    @theano.change_flags(compute_test_value='off')
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


def _prepare_batch_normalization_axes(axes, ndim):
    if axes == 'per-activation':
        axes = (0,)
    elif axes == 'spatial':
        axes = (0,) + tuple(range(2, ndim))
    elif isinstance(axes, (tuple, list, np.ndarray)):
        axes = tuple(int(a) for a in axes)
    else:
        raise ValueError('invalid axes: %s', str(axes))
    axes = tuple(sorted(axes))
    if len(axes) == 0:
        raise ValueError('there should be at least one normalization axis')
    if min(axes) < 0 or max(axes) >= ndim:
        raise ValueError('axes should be less than ndim (<%d), but %s given' % (ndim, str(axes)))
    non_bc_axes = tuple(i for i in range(ndim) if i not in axes)
    return axes, non_bc_axes


def batch_normalization_train(inputs, gamma, beta, axes='per-activation',
                              epsilon=1e-4, running_average_factor=0.1,
                              running_mean=None, running_var=None):
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
        Learnable scale factors. The shape must match the shape of `inputs`,
        except for the axes in `axes`. These axes should be set to 1 or be
        skipped altogether (such that `gamma.ndim == inputs.ndim - len(axes)`).
    beta : tensor
        Learnable biases. Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - r_a_factor) + batch mean * r_a_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None. The shape should match that of `gamma` and `beta`.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - r_a_factor) + (m / (m - 1)) * batch var * r_a_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None. The shape should match that of `gamma` and `beta`.

    Returns
    -------
    out : tensor
        Batch-normalized inputs.
    mean : tensor
        Means of `inputs` across the normalization axes.
    invstd : tensor
        Inverse standard deviations of `inputs` across the normalization axes.
    new_running_mean : tensor
        New value of the running mean (only if both `running_mean` and
        `running_var` were given).
    new_running_var : tensor
        New value of the running variance (only if both `running_var` and
        `running_mean` were given).

    Notes
    -----
    If per-activation or spatial normalization is selected, this operation
    will use the cuDNN implementation. (This requires cuDNN 5 or newer.)

    The returned values are equivalent to:

    .. code-block:: python

        # for per-activation normalization
        axes = (0,)
        # for spatial normalization
        axes = (0,) + tuple(range(2, inputs.ndim))
        mean = inputs.mean(axes, keepdims=True)
        var = inputs.var(axes, keepdims=True)
        invstd = T.inv(T.sqrt(var + epsilon))
        out = (inputs - mean) * gamma * invstd + beta

        m = T.cast(T.prod(inputs.shape) / T.prod(mean.shape), 'float32')
        running_mean = running_mean * (1 - running_average_factor) + \\
                       mean * running_average_factor
        running_var = running_var * (1 - running_average_factor) + \\
                      (m / (m - 1)) * var * running_average_factor
    """
    ndim = inputs.ndim
    axes, non_bc_axes = _prepare_batch_normalization_axes(axes, ndim)

    # have the parameter tensors been broadcasted yet?
    if gamma.ndim == ndim:
        params_ndim = ndim
    else:
        params_ndim = len(non_bc_axes)
        params_dimshuffle_pattern = ['x'] * ndim
        for i, axis in enumerate(non_bc_axes):
            params_dimshuffle_pattern[axis] = i

    if gamma.ndim != params_ndim or beta.ndim != params_ndim:
        raise ValueError("gamma and beta dimensionality must match the "
                         "number of non-normalized axes, or have the "
                         "same number of dimensions as the inputs; "
                         "got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, params_ndim))
    if (running_mean is None) != (running_var is None):
        raise ValueError("running_mean and running_var must either both be "
                         "given or both be None")
    if running_mean is not None and running_mean.ndim != params_ndim:
        raise ValueError("running_mean must be of the same dimensionality "
                         "as gamma and beta; got %d instead of %d" %
                         (running_mean.ndim, params_ndim))
    if running_var is not None and running_var.ndim != params_ndim:
        raise ValueError("running_var must be of the same dimensionality "
                         "as gamma and beta; got %d instead of %d" %
                         (running_var.ndim, params_ndim))

    # epsilon will be converted to floatX later. we need to check
    # for rounding errors now, since numpy.float32(1e-5) < 1e-5.
    epsilon = np.cast[theano.config.floatX](epsilon)
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %s" % str(epsilon))

    inputs = as_tensor_variable(inputs)
    gamma = as_tensor_variable(gamma)
    beta = as_tensor_variable(beta)

    if params_ndim != ndim:
        gamma = gamma.dimshuffle(params_dimshuffle_pattern)
        beta = beta.dimshuffle(params_dimshuffle_pattern)
    else:
        gamma = T.addbroadcast(gamma, *axes)
        beta = T.addbroadcast(beta, *axes)

    batchnorm_op = AbstractBatchNormTrain(axes=axes)

    if running_mean is not None and running_var is not None:
        running_mean = as_tensor_variable(running_mean)
        running_var = as_tensor_variable(running_var)
        if params_ndim != ndim:
            running_mean = running_mean.dimshuffle(params_dimshuffle_pattern)
            running_var = running_var.dimshuffle(params_dimshuffle_pattern)
        else:
            running_mean = T.addbroadcast(running_mean, *axes)
            running_var = T.addbroadcast(running_var, *axes)
        out, mean, invstd, new_running_mean, new_running_var = batchnorm_op(
            inputs, gamma, beta, epsilon=epsilon,
            running_average_factor=running_average_factor,
            running_mean=running_mean, running_var=running_var)
        if new_running_mean.broadcastable != running_mean.broadcastable:
            new_running_mean = T.patternbroadcast(new_running_mean, running_mean.broadcastable)
        if new_running_var.broadcastable != running_var.broadcastable:
            new_running_var = T.patternbroadcast(new_running_var, running_var.broadcastable)
        results = (out, mean, invstd, new_running_mean, new_running_var)
    else:
        results = batchnorm_op(inputs, gamma, beta, epsilon=epsilon)

    if params_ndim != ndim:
        # remove the broadcasted dimensions (except from the output)
        results = ([results[0]] +
                   [r.dimshuffle(non_bc_axes) for r in results[1:]])
    return tuple(results)


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
        Scale factors. The shape must match the shape of `inputs`,
        except for the axes in `axes`. These axes should be set to 1 or be
        skipped altogether (such that `gamma.ndim == inputs.ndim - len(axes)`).
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
    If per-activation or spatial normalization is selected, this operation
    will use the cuDNN implementation. (This requires cuDNN 5 or newer.)

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
    axes, non_bc_axes = _prepare_batch_normalization_axes(axes, ndim)

    # have the parameter tensors been broadcasted yet?
    if gamma.ndim == ndim:
        params_ndim = ndim
    else:
        params_ndim = len(non_bc_axes)
        params_dimshuffle_pattern = ['x'] * ndim
        for i, axis in enumerate(non_bc_axes):
            params_dimshuffle_pattern[axis] = i

    if gamma.ndim != params_ndim or beta.ndim != params_ndim:
        raise ValueError("gamma and beta dimensionality must match the "
                         "number of non-normalized axes, or have the "
                         "same number of dimensions as the inputs; "
                         "got %d and %d instead of %d" %
                         (gamma.ndim, beta.ndim, params_ndim))
    if mean.ndim != params_ndim or var.ndim != params_ndim:
        raise ValueError("mean and var must be of the same dimensionality "
                         "as gamma and beta; got %d and %d instead of %d" %
                         (mean.ndim, var.ndim, params_ndim))

    # epsilon will be converted to floatX later. we need to check
    # for rounding errors now, since numpy.float32(1e-5) < 1e-5.
    epsilon = np.cast[theano.config.floatX](epsilon)
    if epsilon < 1e-5:
        raise ValueError("epsilon must be at least 1e-5, got %s" % str(epsilon))

    gamma = as_tensor_variable(gamma)
    beta = as_tensor_variable(beta)
    mean = as_tensor_variable(mean)
    var = as_tensor_variable(var)

    if params_ndim != ndim:
        gamma = gamma.dimshuffle(params_dimshuffle_pattern)
        beta = beta.dimshuffle(params_dimshuffle_pattern)
        mean = mean.dimshuffle(params_dimshuffle_pattern)
        var = var.dimshuffle(params_dimshuffle_pattern)
    else:
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
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - running_average_factor) + batch mean * running_average_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - running_average_factor) + (m / (m - 1)) * batch var * running_average_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None.
    """

    __props__ = ('axes',)

    def __init__(self, axes=(0,)):
        assert isinstance(axes, (tuple, list))
        assert len(axes) > 0
        axes = tuple(int(a) for a in axes)
        self.axes = axes

    def infer_shape(self, node, shape):
        return [shape[0]] + [shape[1]] * (len(node.outputs) - 1)

    def make_node(self, x, scale, bias, epsilon=1e-4,
                  running_average_factor=0.1,
                  running_mean=None, running_var=None):
        x = as_tensor_variable(x)
        scale = as_tensor_variable(scale)
        bias = as_tensor_variable(bias)
        epsilon = as_tensor_variable(epsilon)
        running_average_factor = as_tensor_variable(running_average_factor)
        if running_mean is not None:
            running_mean = as_tensor_variable(running_mean)
        if running_var is not None:
            running_var = as_tensor_variable(running_var)
        assert x.ndim == scale.ndim == bias.ndim
        assert ((running_mean is None and running_var is None) or
                (running_mean is not None and running_var is not None))
        assert (running_mean is None or running_mean.ndim == x.ndim)
        assert (running_var is None or running_var.ndim == x.ndim)
        # Upcast to common dtype on the non-scalar
        # Keep as is dtype of scalar (epsilon and running_average_factor)
        if running_mean:
            x, scale, bias, running_mean, running_var = as_common_dtype(
                x, scale, bias, running_mean, running_var)
        else:
            x, scale, bias = as_common_dtype(x, scale, bias)
        inputs = [x, scale, bias, epsilon, running_average_factor]
        output_types = [x.type(), scale.type(), scale.type()]
        if running_mean is not None and running_var is not None:
            inputs.append(running_mean)
            inputs.append(running_var)
            output_types.append(scale.type())
            output_types.append(scale.type())
        return Apply(self, inputs, output_types)

    def L_op(self, inputs, outputs, grads):
        x, scale, bias, epsilon, running_average_factor = inputs[:5]
        dy = grads[0]
        _, x_mean, x_invstd = outputs[:3]
        disconnected_outputs = [
            theano.gradient.DisconnectedType()(),  # epsilon
            theano.gradient.DisconnectedType()()]  # running_average_factor
        # Optional running_mean and running_var.
        for i in range(5, len(inputs)):
            disconnected_outputs.append(theano.gradient.DisconnectedType()())
        return AbstractBatchNormTrainGrad(self.axes)(
            x, dy, scale, x_mean, x_invstd, epsilon) + disconnected_outputs

    def connection_pattern(self, node):
        # Specificy that epsilon and running_average_factor are not connected to outputs.
        patterns = [[True, True, True],     # x
                    [True, True, True],     # scale
                    [True, True, True],     # bias
                    [False, False, False],  # epsilon
                    [False, False, False]]  # running_average_factor
        # Optional running_mean and running_var are only
        # connected to their new values.
        for i in range(5, len(node.inputs)):
            patterns[0].append(True)
            for pattern in patterns[1:]:
                pattern.append(False)
            patterns.append([False] * (3 + i - 5) + [True])
        return patterns

    def perform(self, node, inputs, output_storage):
        x, scale, bias, epsilon, running_average_factor = inputs[:5]
        axes = self.axes
        if min(axes) < 0 or max(axes) >= x.ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (x.ndim, str(axes)))

        mean = x.mean(axes, keepdims=True)
        var = x.var(axes, keepdims=True)
        invstd = 1.0 / np.sqrt(var + epsilon)
        out = (x - mean) * (scale * invstd) + bias

        output_storage[0][0] = out
        output_storage[1][0] = mean
        output_storage[2][0] = invstd

        if len(inputs) > 5:
            running_mean = inputs[5]
            running_mean = running_mean * (1.0 - running_average_factor) + \
                mean * running_average_factor
            output_storage[3][0] = running_mean
        if len(inputs) > 6:
            m = float(np.prod(x.shape) / np.prod(scale.shape))
            running_var = inputs[6]
            running_var = running_var * (1.0 - running_average_factor) + \
                (m / (m - 1)) * var * running_average_factor
            output_storage[4][0] = running_var


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
        x = as_tensor_variable(x)
        scale = as_tensor_variable(scale)
        bias = as_tensor_variable(bias)
        estimated_mean = as_tensor_variable(estimated_mean)
        estimated_variance = as_tensor_variable(estimated_variance)
        epsilon = as_tensor_variable(epsilon)
        # Upcast to common dtype on the non-scalar
        # Keep as is dtype of scalar (epsilon)
        x, scale, bias, estimated_mean, estimated_variance = as_common_dtype(
            x, scale, bias, estimated_mean, estimated_variance)
        assert x.ndim == scale.ndim == bias.ndim == estimated_mean.ndim == estimated_variance.ndim

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
        out = (x - estimated_mean) * (scale / np.sqrt(estimated_variance + epsilon)) + bias
        output_storage[0][0] = out


class AbstractBatchNormTrainGrad(Op):
    __props__ = ('axes',)

    def __init__(self, axes=(0,)):
        assert isinstance(axes, (tuple, list))
        assert len(axes) > 0
        axes = tuple(int(a) for a in axes)
        self.axes = axes

    def make_node(self, x, dy, scale, x_mean, x_invstd, epsilon=1e-4):
        x = as_tensor_variable(x)
        dy = as_tensor_variable(dy)
        scale = as_tensor_variable(scale)
        x_mean = as_tensor_variable(x_mean)
        x_invstd = as_tensor_variable(x_invstd)
        epsilon = as_tensor_variable(epsilon)

        # Upcast to common dtype on the non-scalar
        # Keep as is dtype of scalar (epsilon)
        x, dy, scale, x_mean, x_invstd = as_common_dtype(
            x, dy, scale, x_mean, x_invstd)
        assert x.ndim == dy.ndim == scale.ndim == x_mean.ndim == x_invstd.ndim
        return Apply(self, [x, dy, scale, x_mean, x_invstd, epsilon],
                     [x.type(), scale.type(), scale.type()])

    def grad(self, inp, grads):
        x, dy, scale, x_mean, x_invstd, epsilon = inp
        ddinputs, ddscale, ddbias = grads

        x_diff = x - x_mean
        mean_dy_x_diff = T.mean(dy * x_diff, axis=self.axes, keepdims=True)

        # compute gradients given each of the output gradients
        g_wrt_x = 0
        g_wrt_dy = 0
        g_wrt_scale = 0
        g_wrt_x_mean = 0
        g_wrt_x_invstd = 0

        if not isinstance(ddinputs.type, theano.gradient.DisconnectedType):
            ccc = scale * (ddinputs - T.mean(ddinputs, axis=self.axes, keepdims=True))
            ddd = (x_invstd ** 3) * (ccc * T.mean(dy * x_diff, axis=self.axes, keepdims=True) +
                                     dy * T.mean(ccc * x_diff, axis=self.axes, keepdims=True))

            g_wrt_x = g_wrt_x - ddd
            g_wrt_dy = g_wrt_dy + ((ccc * x_invstd) -
                                   ((x_invstd ** 3) * x_diff *
                                    T.mean(ccc * x_diff, axis=self.axes, keepdims=True)))

            eee = (dy * x_invstd) - ((x_invstd ** 3) * x_diff * mean_dy_x_diff)
            g_wrt_scale = g_wrt_scale + T.sum(ddinputs * (eee - T.mean(eee, axis=self.axes, keepdims=True)),
                                              axis=self.axes, keepdims=True)

            g_wrt_x_mean = g_wrt_x_mean + T.sum(ddd, axis=self.axes, keepdims=True)
            g_wrt_x_invstd = g_wrt_x_invstd + T.sum(ccc * (dy - 3 * (x_invstd ** 2) * x_diff * mean_dy_x_diff),
                                                    axis=self.axes, keepdims=True)

        if not isinstance(ddscale.type, theano.gradient.DisconnectedType):
            g_wrt_x = g_wrt_x + (x_invstd * ddscale * dy)
            g_wrt_dy = g_wrt_dy + (x_invstd * ddscale * x_diff)
            g_wrt_x_mean = g_wrt_x_mean - (x_invstd * ddscale * T.sum(dy, axis=self.axes, keepdims=True))
            g_wrt_x_invstd = g_wrt_x_invstd + (ddscale * T.sum(dy * x_diff, axis=self.axes, keepdims=True))

        if not isinstance(ddbias.type, theano.gradient.DisconnectedType):
            g_wrt_dy = g_wrt_dy + T.fill(dy, ddbias)

        # depending on which output gradients are given,
        # some inputs should be disconnected
        results = [g_wrt_x, g_wrt_dy, g_wrt_scale, g_wrt_x_mean, g_wrt_x_invstd,
                   theano.gradient.DisconnectedType()()]
        return [theano.gradient.DisconnectedType()() if r is 0 else r
                for r in results]

    def connection_pattern(self, node):
        return [[True, True, False],    # x
                [True, True, True],     # dy
                [True, False, False],   # scale
                [True, True, False],    # x_mean
                [True, True, False],    # x_invstd
                [False, False, False]]  # epsilon

    def infer_shape(self, node, shape):
        return [shape[0], shape[2], shape[2]]

    def perform(self, node, inputs, output_storage):
        x, dy, scale, x_mean, x_invstd, epsilon = inputs
        axes = self.axes
        if min(axes) < 0 or max(axes) >= x.ndim:
            raise ValueError('axes should be less than ndim (<%d), but %s given' % (x.ndim, str(axes)))

        x_diff = x - x_mean
        mean_dy_x_diff = np.mean(dy * x_diff, axis=axes, keepdims=True)
        c = (dy * x_invstd) - (x_diff * mean_dy_x_diff * (x_invstd ** 3))

        g_wrt_inputs = scale * (c - np.mean(c, axis=axes, keepdims=True))
        g_wrt_scale = np.sum(dy * x_invstd * x_diff, axis=axes, keepdims=True)
        g_wrt_bias = np.sum(dy, axis=axes, keepdims=True)

        output_storage[0][0] = g_wrt_inputs
        output_storage[1][0] = g_wrt_scale
        output_storage[2][0] = g_wrt_bias


@local_optimizer([AbstractBatchNormTrain])
def local_abstract_batch_norm_train(node):
    if not isinstance(node.op, AbstractBatchNormTrain):
        return None

    x, scale, bias, epsilon, running_average_factor = node.inputs[:5]
    axes = node.op.axes
    if min(axes) < 0 or max(axes) > x.ndim:
        return None
    if not isinstance(x.type, TensorType) or \
       not isinstance(scale.type, TensorType) or \
       not isinstance(bias.type, TensorType) or \
       not isinstance(epsilon.type, TensorType) or \
       not isinstance(running_average_factor.type, TensorType):
        return None
    # optional running_mean and running_var
    if len(node.inputs) > 5 and not isinstance(node.inputs[5].type, TensorType):
        return None
    if len(node.inputs) > 6 and not isinstance(node.inputs[6].type, TensorType):
        return None

    mean = x.mean(axes, keepdims=True)
    var = x.var(axes, keepdims=True)
    # The epsilon should not upcast the dtype.
    if var.dtype == 'float32' and epsilon.dtype == 'float64':
        epsilon = epsilon.astype('float32')
    invstd = T.inv(T.sqrt(var + epsilon))
    out = (x - mean) * (scale * invstd) + bias
    results = [out, mean, invstd]

    if len(node.inputs) > 5:
        running_mean = node.inputs[5]
        running_mean = running_mean * (1.0 - running_average_factor) + \
            mean * running_average_factor
        results.append(running_mean)
    if len(node.inputs) > 6:
        m = T.cast(T.prod(x.shape) / T.prod(scale.shape), theano.config.floatX)
        running_var = node.inputs[6]
        running_var = running_var * (1.0 - running_average_factor) + \
            (m / (m - 1)) * var * running_average_factor
        results.append(running_var)

    results = [T.patternbroadcast(r, r_orig.broadcastable)
               for (r, r_orig) in zip(results, node.outputs)]

    for var in theano.gof.graph.variables(node.inputs, results):
        if var not in node.inputs:
            copy_stack_trace(node.outputs[0], var)
    return results


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
    results = [g_wrt_inputs, g_wrt_scale, g_wrt_bias]

    results = [T.patternbroadcast(r, r_orig.broadcastable)
               for (r, r_orig) in zip(results, node.outputs)]

    for var in theano.gof.graph.variables(node.inputs, results):
        if var not in node.inputs:
            copy_stack_trace(node.outputs[0], var)
    return results


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

    # The epsilon should not upcast the dtype.
    if estimated_variance.dtype == 'float32' and epsilon.dtype == 'float64':
        epsilon = epsilon.astype('float32')

    result = (x - estimated_mean) * (scale / T.sqrt(estimated_variance + epsilon)) + bias
    result = T.patternbroadcast(result, node.outputs[0].broadcastable)

    for var in theano.gof.graph.variables(node.inputs, [result]):
        if var not in node.inputs:
            copy_stack_trace(node.outputs[0], var)
    return [result]


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
