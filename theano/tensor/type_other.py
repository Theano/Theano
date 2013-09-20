#
# Slice type and Op. None Type and NoneConst.
#
import theano
from theano.gof import Apply, Constant, Op, Type
from theano.gradient import DisconnectedType


def as_int_none_variable(x):
    if x is None:
        return NoneConst
    x = theano.tensor.as_tensor_variable(x, ndim=0)
    if x.type.dtype[:3] not in ('int', 'uin'):
        raise TypeError('index must be integers')
    return x


class MakeSlice(Op):
    def make_node(self, slc):
        return Apply(self,
                     map(as_int_none_variable,
                         [slc.start, slc.stop, slc.step]),
                     [slicetype()])

    def perform(self, node, inp, out_):
        out, = out_
        out[0] = slice(*inp)

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

make_slice = MakeSlice()


class SliceType(Type):

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, slice):
            return x
        else:
            raise TypeError('Expected a slice!')

    def __str__(self):
        return "slice"

slicetype = SliceType()


class NoneTypeT(Type):

    def filter(self, x, strict=False, allow_downcast=None):
        if x is None:
            return x
        else:
            raise TypeError('Expected None!')

    def __str__(self):
        return "None"

NoneConst = Constant(NoneTypeT(), None, name='None')
