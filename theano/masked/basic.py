import theano
from theano.gof import Variable, Op, Apply
from theano.masked.basic import MaskedTensorType


class _masked_tensor_py_operators(object):
    ndim = property(lambda self: self.type.ndim)
    broadcastable = property(lambda self: self.type.broadcastable)
    dtype = property(lambda self: self.type.dtype)
    ttype = property(lambda self: self.type.ttype)
    mask_ttype = property(lambda self: self.type.mask_ttype)

    data = property(lambda self: GetData()(self))
    mask = property(lambda self: GetMask()(self))

    def __add__(self, other):
        return self.add(self, other)

    def add(self, other, strict=True):
        data = GetData()(self)
        mask = GetMask()(self)
        if isinstance(other, MaskedTensorVariable):
            other_data = GetData()(other)
            other_mask = GetMask()(other)
            if strict:
                new_mask = mask | other_mask
            else:
                new_mask = mask & other_mask
        else:
            other_data = other
            new_mask = mask
        new_data = theano.tensor.basic.add(data, other_data)
        masked_data = MakeMask()(new_data, new_mask)
        return masked_data


class MaskedTensorVariable(_masked_tensor_py_operators, Variable):
    pass

MaskedTensorType.Variable = MaskedTensorVariable


class MakeMask(Op):
    view_map = {0: [0]}

    def make_node(self, data, mask):
        return Apply(self, [data, mask],
                     [MaskedTensorVariable(data.type)])

    def perform(self, node, (data, mask), (out,)):
        out[0] = np.ma.array(data, mask=mask)


class GetMask(Op):
    view_map = {0: [0]}

    def make_node(self, ma):
        return Apply(self, [ma],
                     [ma.mask_ttype.make_variable()])

    def perform(self, node, (ma,), (out,)):
        out[0] = ma.mask.view('int8')


class GetData(Op):
    view_map = {0: [0]}

    def make_node(self, ma):
        return Apply(self, [ma], [ma.ttype.make_variable()])

    def perform(self, node, (ma,), (out,)):
        out[0] = ma.data
