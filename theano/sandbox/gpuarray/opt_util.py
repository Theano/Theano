from functools import wraps

import numpy

from theano import scalar as scal, Constant
from theano.gof import local_optimizer
from theano.tensor import (DimShuffle, get_scalar_constant_value,
                           NotScalarConstantError)

from .basic_ops import GpuFromHost, HostFromGpu, GpuAllocEmpty
from .elemwise import GpuDimShuffle, GpuElemwise

_one = scal.constant(numpy.asarray(1.0, dtype='float64'))


def grab_cpu_scalar(v, nd):
    """
    Get a scalar variable value from the tree at `v`.

    This function will dig through transfers and dimshuffles to get
    the constant value. If no such constant is found, it returns None.

    Parameters
    ----------
    v
        Theano variable to extract the constant value from.
    nd : int
        Expected number of dimensions for the variable (for
        broadcasted constants).

    """
    if v.owner is not None:
        n = v.owner
        if (isinstance(n.op, (GpuDimShuffle, DimShuffle)) and
                n.op.new_order == ('x',) * nd):
            return grab_cpu_scalar(n.inputs[0], n.inputs[0].ndim)
        elif isinstance(n.op, (GpuFromHost, HostFromGpu)):
            return grab_cpu_scalar(n.inputs[0], nd)
        else:
            return None
    else:
        if (isinstance(v, Constant) and
                v.broadcastable == (True,) * nd):
            return v.dimshuffle(())


def find_node(v, cls, ignore_clients=False):
    """
    Find the node that has an op of of type `cls` in `v`.

    This digs through possibly redundant transfers to for the node
    that has the type `cls`. If `ignore_clients` is False (the
    default) it will only dig through nodes that have a single client
    to avoid duplicating computations.

    Parameters
    ----------
    v
        The variable to dig through
    cls : Op class
        The type of the node we are looking for
    ignore_clients : bool, optional
        Whether to ignore multiple clients or not.

    """
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
    """
    Returns True if `var` is always equal to `val`.

    This will only return True if the variable will always be equal to
    the value.  If it might not be true in some cases then it returns False.

    Parameters
    ----------
    var
        Variable to compare
    val
        Python value

    """
    try:
        v = get_scalar_constant_value(var)
        return v == val
    except NotScalarConstantError:
        return False


def alpha_merge(cls, alpha_in, beta_in):
    """
    Decorator to merge multiplication by a scalar on the output.

    This will find a pattern of `scal * <yourop>(some, params, alpha,
    beta)` and update it so that the scalar multiplication happens as
    part of your op.

    The op needs to accept an alpha and a beta scalar which act this way::

       out = Op() * alpha + out_like * beta

    Where out_like is a buffer that has the same size as the output
    and gets added to the "real" output of the operation.  An example
    of an operation that respects this pattern is GEMM from blas.

    The decorated function must have this signature::

        maker(node, *inputs)

    The `node` argument you recieve is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    The `*inputs` parameters contains the new inputs for your op.  You
    MUST use those inputs instead of the ones on `node`.  Note that
    this function can be as simple as::

        def maker(node, *inputs):
            return node.op(*inputs)

    Parameters
    ----------
    cls : op class
        The class of the op you want to merge
    alpha_in : int
        The input index for the alpha scalar for your op (in node.inputs).
    beta_in : int
        The input index for the beta scalar for your op (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered local optimizer that has the same name as the
        decorated function.

    Notes
    -----
    This was factored out since the code to deal with intervening
    transfers and correctness in the presence of different values of
    alpha and beta scaling factors is not trivial.

    """
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
                if lr is None or lr.dtype != targ.outputs[0].dtype:
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
    """
    Decorator to merge addition by a value on the output.

    This will find a pattern of `val * <yourop>(some, params, alpha,
    beta, out_like)` and update it so that the addtition happens as
    part of your op.

    The op needs to accept an alpha and a beta scalar which act this way::

       out = Op() * alpha + out_like * beta

    Where out_like is a buffer that has the same size as the output
    and gets added to the "real" output of the operation.  An example
    of an operation that respects this pattern is GEMM from blas.

    The decorated function must have this signature::

        maker(node, *inputs)

    The `node` argument you recieve is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    The `*inputs` parameters contains the new inputs for your op.  You
    MUST use those inputs instead of the ones on `node`.  Note that
    this function can be as simple as::

        def maker(node, *inputs):
            return node.op(*inputs)

    Parameters
    ----------
    cls : op class
        The class of the op you want to merge
    alpha_in : int
        The input index for the alpha scalar for your op (in node.inputs).
    beta_in : int
        The input index for the beta scalar for your op (in node.inputs).
    out_in : int
        The input index for the out_like input for your op (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered local optimizer that has the same name as the
        decorated function.

    Notes
    -----
    This was factored out since the code to deal with intervening
    transfers and correctness in the presence of different values of
    alpha and beta scaling factors is not trivial.

    This also correctly handles the case where the added value is
    broadcasted (by not performing the replacement).

    """
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
                if W.dtype != targ.outputs[0].dtype:
                    return None
                if not is_equal(targ.inputs[beta_in], 0.0):
                    # other cases are too complex for now
                    return None
                if W.broadcastable != targ.inputs[out_in].broadcastable:
                    # Would need to explicitly tile the output to fill
                    # the full shape here.  Disable for now.
                    return None
                inputs = list(targ.inputs)
                inputs[out_in] = W
                inputs[beta_in] = _one.clone()
                return maker(targ, *inputs)
        return opt
    return wrapper


def inplace_allocempty(op, idx):
    """
    Wrapper to make an inplace optimization that deals with AllocEmpty

    This will duplicate the alloc input if it has more than one client
    to allow the op to work on it inplace.

    The decorated function must have this signature::

        maker(node, inputs)

    The `node` argument you recieve is the original apply node that
    contains your op.  You should use it to grab relevant properties
    for your op so that the new version performs the same computation.
    You should also switch the op to work inplace.  The `*inputs`
    parameters contains the new inputs for your op.  You MUST use
    those inputs instead of the ones on `node`.  Note that this
    function can be as simple as::

        def maker(node, inputs):
            return [node.op.__class__(inplace=True)(*inputs)]

    Parameters
    ----------
    op : op class
        The op class to look for to make inplace
    idx : int
        The index of the (possibly) AllocEmpty input (in node.inputs).

    Returns
    -------
    local optimizer
        an unregistered inplace local optimizer that has the same name
        as the decorated function.

    """
    def wrapper(maker):
        @local_optimizer([op], inplace=True)
        @wraps(maker)
        def opt(node):
            if type(node.op) != op or node.op.inplace:
                return
            inputs = list(node.inputs)
            alloc = inputs[idx]
            if (alloc.owner and
                    isinstance(alloc.owner.op, GpuAllocEmpty) and
                    len(alloc.clients) > 1):
                alloc_op = GpuAllocEmpty(alloc.owner.op.dtype,
                                         alloc.owner.op.context_name)
                inputs[idx] = alloc_op(*alloc.owner.inputs)
            return maker(node, inputs)
        return opt
    return wrapper
