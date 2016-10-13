from __future__ import absolute_import, print_function, division
from functools import wraps

import numpy

from theano import tensor, scalar as scal, Constant
from theano.gof import local_optimizer
from theano.tensor import (DimShuffle, get_scalar_constant_value,
                           NotScalarConstantError)

from theano.sandbox.cuda.basic_ops import (
    GpuFromHost, HostFromGpu, host_from_gpu, GpuDimShuffle, GpuElemwise, GpuReshape)

_one = scal.constant(numpy.asarray(1.0, dtype='float32'))


def grab_cpu_scalar(v, nd):
    if v.owner is not None:
        n = v.owner
        if (isinstance(n.op, GpuDimShuffle) and
                n.op.new_order == ('x',) * nd):
            return host_from_gpu(n.inputs[0])
        elif (isinstance(n.op, DimShuffle) and
              n.op.new_order == ('x',) * nd):
            return n.inputs[0]
        elif isinstance(n.op, GpuFromHost):
            return grab_cpu_scalar(n.inputs[0], nd=nd)
        else:
            return None
    else:
        if (isinstance(v, Constant) and
                v.broadcastable == (True,) * nd):
            return v.dimshuffle(())


def find_node(v, cls, ignore_clients=False):
    # This digs through possibly redundant transfers to for the node
    # that has the op class specified.
    if v.owner is not None and (ignore_clients or len(v.clients) == 1):
        if isinstance(v.owner.op, cls):
            return v.owner
        elif (isinstance(v.owner.op, GpuFromHost) and
              v.owner.inputs[0].owner is not None and
              (ignore_clients or len(v.owner.inputs[0].clients) == 1) and
              isinstance(v.owner.inputs[0].owner.op, HostFromGpu)):
            return find_node(v.owner.inputs[0].owner.inputs[0], cls)
        else:
            return None


def is_equal(var, val):
    # Returns True if var is always equal to val (python value), False
    # otherwise (including if var is not constant)
    try:
        v = get_scalar_constant_value(var)
        return v == val
    except NotScalarConstantError:
        return False


def alpha_merge(cls, alpha_in, beta_in):
    def wrapper(maker):
        @local_optimizer([GpuElemwise])
        @wraps(maker)
        def opt(node):
            if (isinstance(node.op, GpuElemwise) and
                    node.op.scalar_op == scal.mul and
                    node.nin == 2):
                targ = find_node(node.inputs[0], cls)
                if targ is None:
                    targ = find_node(node.inputs[1], cls)
                    if targ is None:
                        return
                    lr = grab_cpu_scalar(node.inputs[0],
                                         nd=targ.outputs[0].ndim)
                else:
                    lr = grab_cpu_scalar(node.inputs[1],
                                         nd=targ.outputs[0].ndim)
                if lr is None or targ is None:
                    return None
                inputs = list(targ.inputs)
                try:
                    c = get_scalar_constant_value(lr)
                    if c == 0:
                        inputs[alpha_in] = lr
                        inputs[beta_in] = lr
                    elif c == 1:
                        inputs[alpha_in] = targ.inputs[alpha_in]
                        inputs[beta_in] = targ.inputs[beta_in]
                    else:
                        inputs[alpha_in] = lr * targ.inputs[alpha_in]
                        inputs[beta_in] = lr * targ.inputs[beta_in]
                except NotScalarConstantError:
                    inputs[alpha_in] = lr * targ.inputs[alpha_in]
                    inputs[beta_in] = lr * targ.inputs[beta_in]
                return maker(targ, *inputs)
        return opt
    return wrapper


def output_merge(cls, alpha_in, beta_in, out_in):
    def wrapper(maker):
        @local_optimizer([GpuElemwise])
        @wraps(maker)
        def opt(node):
            if (isinstance(node.op, GpuElemwise) and
                    node.op.scalar_op == scal.add and
                    node.nin == 2):
                targ = find_node(node.inputs[0], cls)
                W = node.inputs[1]
                if targ is None:
                    targ = find_node(node.inputs[1], cls)
                    W = node.inputs[0]
                if targ is None:
                    return None
                if not is_equal(targ.inputs[beta_in], 0.0):
                    # other cases are too complex for now
                    return None
                if W.broadcastable != targ.inputs[out_in].broadcastable:
                    # May change later to do the broadcast, but it's
                    # under discussion.
                    return None
                inputs = list(targ.inputs)
                inputs[out_in] = W
                inputs[beta_in] = _one.clone()
                return maker(targ, *inputs)
        return opt
    return wrapper


def pad_dims(input, leftdims, rightdims):
    """Reshapes the input to a (leftdims + rightdims) tensor

    This helper function is used to convert pooling inputs with arbitrary
    non-pooling dimensions to the correct number of dimensions for the
    GPU pooling ops.

    This reduces or expands the number of dimensions of the input to
    exactly `leftdims`, by adding extra dimensions on the left or by
    combining some existing dimensions on the left of the input.

    Use `unpad_dims` to reshape back to the original dimensions.

    Examples
    --------
    Given input of shape (3, 5, 7), ``pad_dims(input, 2, 2)``
    adds a singleton dimension and reshapes to (3, 1, 5, 7).
    Given that output from pad_dims, ``unpad_dims(output, input, 2, 2)``
    reshapes back to (3, 5, 7).

    Given input of shape (3, 5, 7, 9), ``pad_dims(input, 2, 2)``
    does not reshape and returns output with shape (3, 5, 7, 9).

    Given input of shape (3, 5, 7, 9, 11), ``pad_dims(input, 2, 2)``
    combines the first two dimensions and reshapes to (8, 7, 9, 11).

    Given input of shape (3, 5, 7, 9), ``pad_dims(input, 2, 3)``
    adds a singleton dimension and reshapes to (3, 1, 5, 7, 9).
    """
    assert input.ndim >= rightdims

    if input.ndim == (leftdims + rightdims):
        return input

    # extract image dimensions
    img_shape = input.shape[-rightdims:]

    non_pool_ndim = input.ndim - rightdims
    if non_pool_ndim < leftdims:
        # too few dimensions, pad on the left
        dummy_dims = tensor.as_tensor([1] * (leftdims - non_pool_ndim))
        new_shape = tensor.join(0, dummy_dims,
                                input.shape[:non_pool_ndim],
                                img_shape)
    else:
        # too many dimensions, combine the leading dimensions
        batched_ndim = non_pool_ndim - leftdims + 1
        batch_size = tensor.prod(input.shape[:batched_ndim])
        # convert to a vector for tensor.join
        batch_size = tensor.shape_padright(batch_size, 1)
        new_shape = tensor.join(0, batch_size,
                                input.shape[batched_ndim:non_pool_ndim],
                                img_shape)

    # store in the required shape
    new_shape = tensor.cast(new_shape, 'int64')
    input_ND = GpuReshape(leftdims + rightdims)(input, new_shape)
    return input_ND


def unpad_dims(output, input, leftdims, rightdims):
    """Reshapes the output after pad_dims.

    This reverts the padding by `pad_dims`.
    """
    if output.ndim == input.ndim:
        return output

    # restore the output to the original shape
    outshp = tensor.join(0, input.shape[:-rightdims], output.shape[-rightdims:])
    return GpuReshape(input.ndim)(output, outshp)
