import theano
from theano.gradient import DisconnectedType
from theano.gof import Op, Apply
from theano import tensor
import theano.sandbox.cuda as cuda


def get_diagonal_subtensor_view(x, i0, i1):
    if x.shape[i0] < x.shape[i1]:
        raise NotImplementedError('is this allowed?')
    idx = [slice(None)] * x.ndim
    idx[i0] = slice(x.shape[i1] - 1, None, None)
    xview = x.__getitem__(tuple(idx))
    strides = list(xview.strides)
    strides[i1] -= strides[i0]
    xview.strides = strides
    return xview


class DiagonalSubtensor(Op):
    """
    Work on the GPU.
    """
    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return hash((type(self), self.inplace))

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
    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return hash((type(self), self.inplace))

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
           border_mode='valid', subsample=(1, 1, 1), **kwargs):
    """
    Convolve spatio-temporal filters with a movie.

    signals - timeseries of images whose pixels have color channels.
            shape: [Ns, Ts, C, Hs, Ws]
    filters - spatio-temporal filters
            shape: [Nf, Tf, C, Hf, Wf]

    border_mode - tuple of string mode names (or just a mode name, which means a
            homogenous tuple). A mode name can be one of 'full', 'valid', and 'same'.
    """

    if isinstance(border_mode, str):
        border_mode = (border_mode, border_mode, border_mode)

    #TODO: support variables in the shape
    if signals_shape is None or filters_shape is None:
        raise NotImplementedError('need shapes for now')
    _signals_shape_5d = signals.shape if signals_shape is None else signals_shape
    _filters_shape_5d = filters.shape if filters_shape is None else filters_shape

    _signals_shape_4d = (
        _signals_shape_5d[0] * _signals_shape_5d[1],
        _signals_shape_5d[2],
        _signals_shape_5d[3],
        _signals_shape_5d[4],
        )
    _filters_shape_4d = (
        _filters_shape_5d[0] * _filters_shape_5d[1],
        _filters_shape_5d[2],
        _filters_shape_5d[3],
        _filters_shape_5d[4],
        )

    if border_mode[1] != border_mode[2]:
        raise NotImplementedError('height and width bordermodes must match')

    out_4d = tensor.nnet.conv2d(
        signals.reshape(_signals_shape_4d),
        filters.reshape(_filters_shape_4d),
        image_shape=_signals_shape_4d,
        filter_shape=_filters_shape_4d,
        border_mode = border_mode[1])  # ignoring border_mode[2]

    # reshape the output to restore its original size
    # shape = Ns, Ts, Nf, Tf, W-Wf+1, H-Hf+1
    if border_mode[1] == 'valid':
        out_tmp = out_4d.reshape((
            _signals_shape_5d[0],  # Ns
            _signals_shape_5d[1],  # Ts
            _filters_shape_5d[0],  # Nf
            _filters_shape_5d[1],  # Tf
            _signals_shape_5d[3] - _filters_shape_5d[3] + 1,
            _signals_shape_5d[4] - _filters_shape_5d[4] + 1,
            ))
    elif border_mode[1] == 'full':
        out_tmp = out_4d.reshape((
            _signals_shape_5d[0],  # Ns
            _signals_shape_5d[1],  # Ts
            _filters_shape_5d[0],  # Nf
            _filters_shape_5d[1],  # Tf
            _signals_shape_5d[3] + _filters_shape_5d[3] - 1,
            _signals_shape_5d[4] + _filters_shape_5d[4] - 1,
            ))
    elif border_mode[1] == 'same':
        raise NotImplementedError()
    else:
        raise ValueError('invalid border mode', border_mode[1])

    # now sum out along the Tf to get the output
    # but we have to sum on a diagonal through the Tf and Ts submatrix.
    if border_mode[0] == 'valid':
        out_5d = diagonal_subtensor(out_tmp, 1, 3).sum(axis=3)
    elif border_mode[0] in ('full', 'same'):
        raise NotImplementedError('sequence border mode', border_mode[0])
    else:
        raise ValueError('invalid border mode', border_mode[1])
    return out_5d


@cuda.opt.register_opt()
@theano.gof.local_optimizer([])
def local_gpu_diagonal_subtensor(node):
    """
    diagonal_subtensor(host_from_gpu()) -> host_from_gpu(diagonal_subtensor)
    gpu_from_host(diagonal_subtensor) -> diagonal_subtensor(gpu_from_host)
    """
    if isinstance(node.op, DiagonalSubtensor):
        input = node.inputs[0]
        if input.owner and isinstance(input.owner.op, cuda.HostFromGpu):
            return [cuda.host_from_gpu(diagonal_subtensor(cuda.gpu_from_host(input),
                                                     *node.inputs[1:]))]
    if node.op == cuda.gpu_from_host:
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op,
                                           DiagonalSubtensor):
            diag_node = host_input.owner
            return [tensor.diagonal_subtensor(
                cuda.gpu_from_host(diag_node.inputs[0]),
                *diag_node.inputs[1:])]
    return False
