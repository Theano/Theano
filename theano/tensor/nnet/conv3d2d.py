from __future__ import absolute_import, print_function, division
import theano
from theano.gradient import DisconnectedType
from theano.gof import Op, Apply, TopoOptimizer
from theano.gof.opt import copy_stack_trace
from theano import tensor
import theano.sandbox.cuda as cuda


def get_diagonal_subtensor_view(x, i0, i1):
    """
    Helper function for DiagonalSubtensor and IncDiagonalSubtensor.

    Notes
    -----
    It returns a partial view of x, not a partial copy.

    """
    # We have to cast i0 and i0 to int because python 2.4 (and maybe later)
    # do not support indexing with 0-dim, 'int*' ndarrays.
    i0 = int(i0)
    i1 = int(i1)
    if x.shape[i0] < x.shape[i1]:
        raise NotImplementedError('is this allowed?')
    idx = [slice(None)] * x.ndim
    idx[i0] = slice(x.shape[i1] - 1, None, None)
    xview = x.__getitem__(tuple(idx))
    strides = list(xview.strides)
    if x.shape[i1] != 1:
        strides[i1] -= strides[i0]
        xview.strides = strides
    return xview


class DiagonalSubtensor(Op):
    """
    Return a form a nd diagonal subtensor.

    Parameters
    ----------
    x
        n-d tensor
    i0
        Axis index in x
    i1
        Axis index in x

    Notes
    -----
    Work on the GPU.

    Extended summary
    ----------------
    ``x`` is some n-dimensional tensor, but this Op only deals with a
    matrix-shaped slice, using axes i0 and i1. Without loss of
    generality, suppose that ``i0`` picks out our ``row`` dimension,
    and i1 the ``column`` dimension.

    So the relevant part of ``x`` is some matrix ``u``. Suppose it has 7 rows
    and 4 columns::

        [ 0 0 0 0 ]
        [ 0 0 0 0 ]
        [ 0 0 0 0 ]
        [ 0 0 0 0 ]
        [ 0 0 0 0 ]
        [ 0 0 0 0 ]

    The view returned by this function is also a matrix. It's a thick,
    diagonal ``stripe`` across u that discards the lower left triangle
    and the upper right triangle:

        [ x 0 0 0 ]
        [ x x 0 0 ]
        [ x x x 0 ]
        [ 0 x x x ]
        [ 0 0 x x ]
        [ 0 0 0 x ]

    In this case the return value would be this view of shape 3x4. The
    returned view has the same number of dimensions as the input
    ``x``, and the only difference is that the shape along dimension
    ``i0`` has been reduced by ``shape[i1] - 1`` because of the
    triangles that got chopped out.

    The NotImplementedError is meant to catch the case where shape[i0]
    is too small for the stripe to reach across the matrix, in which
    case it's not clear what this function should do. Maybe always
    raise an error. I'd look back to the call site in the Conv3D to
    see what's necessary at that point.

    """

    __props__ = ("inplace",)

    def __str__(self):
        if self.inplace:
            return "%s{inplace}" % self.__class__.__name__
        return "%s" % self.__class__.__name__

    def __init__(self, inplace=False):
        self.inplace = inplace
        if inplace:
            self.view_map = {0: [0]}

    def make_node(self, x, i0, i1):
        _i0 = tensor.as_tensor_variable(i0)
        _i1 = tensor.as_tensor_variable(i1)
        return Apply(self, [x, _i0, _i1], [x.type()])

    def perform(self, node, inputs, output_storage):
        xview = get_diagonal_subtensor_view(*inputs)
        if self.inplace:
            output_storage[0][0] = xview
        else:
            output_storage[0][0] = xview.copy()

    def grad(self, inputs, g_outputs):
        z = tensor.zeros_like(inputs[0])
        gx = inc_diagonal_subtensor(z, inputs[1], inputs[2], g_outputs[0])
        return [gx, DisconnectedType()(), DisconnectedType()()]

    def connection_pattern(self, node):
        rval = [[True], [False], [False]]
        return rval

diagonal_subtensor = DiagonalSubtensor(False)


class IncDiagonalSubtensor(Op):
    """
    The gradient of DiagonalSubtensor.

    """

    __props__ = ("inplace",)

    def __str__(self):
        if self.inplace:
            return "%s{inplace}" % self.__class__.__name__
        return "%s" % self.__class__.__name__

    def __init__(self, inplace=False):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, x, i0, i1, amt):
        _i0 = tensor.as_tensor_variable(i0)
        _i1 = tensor.as_tensor_variable(i1)
        return Apply(self, [x, _i0, _i1, amt], [x.type()])

    def perform(self, node, inputs, output_storage):
        x, i0, i1, amt = inputs
        if not self.inplace:
            x = x.copy()
        xview = get_diagonal_subtensor_view(x, i0, i1)
        xview += amt
        output_storage[0][0] = x

    def grad(self, inputs, g_outputs):
        x, i0, i1, amt = inputs
        gy = g_outputs[0]
        return [gy, DisconnectedType()(), DisconnectedType()(),
                diagonal_subtensor(gy, i0, i1)]

    def connection_pattern(self, node):
        rval = [[True], [False], [False], [True]]
        return rval
inc_diagonal_subtensor = IncDiagonalSubtensor(False)


def conv3d(signals, filters,
           signals_shape=None, filters_shape=None,
           border_mode='valid'):
    """
    Convolve spatio-temporal filters with a movie.

    It flips the filters.

    Parameters
    ----------
    signals
        Timeseries of images whose pixels have color channels.
        Shape: [Ns, Ts, C, Hs, Ws].
    filters
        Spatio-temporal filters.
        Shape: [Nf, Tf, C, Hf, Wf].
    signals_shape
        None or a tuple/list with the shape of signals.
    filters_shape
        None or a tuple/list with the shape of filters.
    border_mode
        One of 'valid', 'full' or 'half'.

    Notes
    -----
    Another way to define signals: (batch,  time, in channel, row, column)
    Another way to define filters: (out channel,time,in channel, row, column)

    For the GPU, you can use this implementation or
    :func:`conv3d_fft <theano.sandbox.cuda.fftconv.conv3d_fft>`.

    See Also
    --------
    Someone made a script that shows how to swap the axes between
    both 3d convolution implementations in Theano. See the last
    `attachment <https://groups.google.com/d/msg/theano-users/1S9_bZgHxVw/0cQR9a4riFUJ>`_

    """

    if isinstance(border_mode, str):
        border_mode = (border_mode, border_mode, border_mode)

    if signals_shape is None:
        _signals_shape_5d = signals.shape
    else:
        _signals_shape_5d = signals_shape

    if filters_shape is None:
        _filters_shape_5d = filters.shape
    else:
        _filters_shape_5d = filters_shape

    Ns, Ts, C, Hs, Ws = _signals_shape_5d
    Nf, Tf, C, Hf, Wf = _filters_shape_5d

    _signals_shape_4d = (Ns * Ts, C, Hs, Ws)
    _filters_shape_4d = (Nf * Tf, C, Hf, Wf)

    if border_mode[1] != border_mode[2]:
        raise NotImplementedError('height and width bordermodes must match')
    conv2d_signal_shape = _signals_shape_4d
    conv2d_filter_shape = _filters_shape_4d
    if signals_shape is None:
        conv2d_signal_shape = None
    if filters_shape is None:
        conv2d_filter_shape = None

    out_4d = tensor.nnet.conv2d(
        signals.reshape(_signals_shape_4d),
        filters.reshape(_filters_shape_4d),
        input_shape=conv2d_signal_shape,
        filter_shape=conv2d_filter_shape,
        border_mode=border_mode[1])  # ignoring border_mode[2]

    # compute the intended output size
    if border_mode[1] == 'valid':
        Hout = Hs - Hf + 1
        Wout = Ws - Wf + 1
    elif border_mode[1] == 'full':
        Hout = Hs + Hf - 1
        Wout = Ws + Wf - 1
    elif border_mode[1] == 'half':
        Hout = Hs - (Hf % 2) + 1
        Wout = Ws - (Wf % 2) + 1
    elif border_mode[1] == 'same':
        raise NotImplementedError()
    else:
        raise ValueError('invalid border mode', border_mode[1])

    # reshape the temporary output to restore its original size
    out_tmp = out_4d.reshape((Ns, Ts, Nf, Tf, Hout, Wout))

    # now sum out along the Tf to get the output
    # but we have to sum on a diagonal through the Tf and Ts submatrix.
    if Tf == 1:
        # for Tf==1, no sum along Tf, the Ts-axis of the output is unchanged!
        out_5d = out_tmp.reshape((Ns, Ts, Nf, Hout, Wout))
    else:
        # for some types of convolution, pad out_tmp with zeros
        if border_mode[0] == 'valid':
            Tpad = 0
        elif border_mode[0] == 'full':
            Tpad = Tf - 1
        elif border_mode[0] == 'half':
            Tpad = Tf // 2
        elif border_mode[0] == 'same':
            raise NotImplementedError()
        else:
            raise ValueError('invalid border mode', border_mode[0])

        if Tpad == 0:
            out_5d = diagonal_subtensor(out_tmp, 1, 3).sum(axis=3)
        else:
            # pad out_tmp with zeros before summing over the diagonal
            out_tmp_padded = tensor.zeros(dtype=out_tmp.dtype, shape=(
                Ns, Ts + 2 * Tpad, Nf, Tf, Hout, Wout
            ))
            out_tmp_padded = tensor.set_subtensor(
                out_tmp_padded[:, Tpad:(Ts + Tpad), :, :, :, :],
                out_tmp)
            out_5d = diagonal_subtensor(out_tmp_padded, 1, 3).sum(axis=3)

    return out_5d


def make_gpu_optimizer(op, to_gpu):
    """
    This function create optimizer that move some inputs to the GPU
    for op that work on both CPU and GPU.

    The op object is created by calling op(), so good default value
    are needed.

    We suppose the same op work with CPU and GPU inputs.

    Parameters
    ----------
    op
        The op that support GPU inputs.
    to_gpu
        A list of op inputs that are moved to the GPU.

    """
    @theano.gof.local_optimizer([op, cuda.gpu_from_host])
    def local_to_gpu(node):
        """
        op(host_from_gpu()) -> host_from_gpu(op)
        gpu_from_host(op) -> op(gpu_from_host)

        """
        if isinstance(node.op, op):
            # op(host_from_gpu()) -> host_from_gpu(op)
            # If any of the input that go on the GPU are on the GPU,
            # move the op to the gpu.
            if any(node.inputs[idx].owner and
                   isinstance(node.inputs[idx].owner.op, cuda.HostFromGpu)
                   for idx in to_gpu):
                new_inp = list(node.inputs)
                for idx in to_gpu:
                    new_inp[idx] = cuda.gpu_from_host(new_inp[idx])
                result_node = op()(*new_inp)
                copy_stack_trace(node.outputs[0], result_node)
                transfer_node = cuda.host_from_gpu(result_node)
                copy_stack_trace(node.outputs[0], transfer_node)
                return [transfer_node]
        if node.op == cuda.gpu_from_host:
            # gpu_from_host(op) -> op(gpu_from_host)
            host_input = node.inputs[0]
            if host_input.owner and isinstance(host_input.owner.op,
                                               op):
                op_node = host_input.owner
                new_inp = list(op_node.inputs)
                for idx in to_gpu:
                    new_inp[idx] = cuda.gpu_from_host(new_inp[idx])
                new_node = op()(*new_inp)
                copy_stack_trace(host_input, new_node)
                return [new_node]
        return False
    local_to_gpu.__name__ = "local_to_gpu_" + op.__name__
    cuda.opt.register_opt()(local_to_gpu)

if cuda.cuda_available:
    make_gpu_optimizer(DiagonalSubtensor, [0])
    make_gpu_optimizer(IncDiagonalSubtensor, [0, 3])


@theano.gof.local_optimizer([DiagonalSubtensor, IncDiagonalSubtensor])
def local_inplace_DiagonalSubtensor(node):
    """Also work for IncDiagonalSubtensor."""
    if (isinstance(node.op, (DiagonalSubtensor, IncDiagonalSubtensor)) and
            not node.op.inplace):
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs[0], new_node)
        return [new_node]
    return False
theano.compile.optdb.register(
    'local_inplace_DiagonalSubtensor',
    TopoOptimizer(
        local_inplace_DiagonalSubtensor,
        failure_callback=TopoOptimizer.warn_inplace),
    60, 'fast_run', 'inplace')
