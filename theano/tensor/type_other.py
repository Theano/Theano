from __future__ import absolute_import, print_function, division
#
# Slice type and Op. None Type and NoneConst.
#

import numpy as np

import theano
from theano.gof import Apply, Constant, Generic, Op, Type, hashtype
from theano.gradient import DisconnectedType


def as_int_none_variable(x):
    if x is None:
        return NoneConst
    elif NoneConst.equals(x):
        return x
    x = theano.tensor.as_tensor_variable(x, ndim=0)
    if x.type.dtype not in theano.tensor.integer_dtypes:
        raise TypeError('index must be integers')
    return x


class MakeSlice(Op):

    __props__ = ()

    def make_node(self, slc, stop=None, step=None):
        # We need to accept and handle in make_node inputs the node
        # inputs to allow redoing a new op elsewhere in the graph by
        # optimization.
        if isinstance(slc, slice):
            assert stop is None
            assert step is None
            inp = [slc.start, slc.stop, slc.step]
        else:
            inp = [slc, stop, step]
        return Apply(self,
                     list(map(as_int_none_variable, inp)),
                     [slicetype()])

    def perform(self, node, inp, out_):
        out, = out_
        out[0] = slice(*inp)

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

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hashtype(self)

    @staticmethod
    def may_share_memory(a, b):
        # Slices never shared memory between object
        return isinstance(a, slice) and a is b

slicetype = SliceType()


class SliceConstant(Constant):
    def __init__(self, type, data, name=None):
        assert isinstance(data, slice)
        # Numpy ndarray aren't hashable, so get rid of them.
        if isinstance(data.start, np.ndarray):
            assert data.start.ndim == 0
            assert str(data.start.dtype) in theano.tensor.integer_dtypes
            data = slice(int(data.start), data.stop, data.step)
        elif isinstance(data.stop, np.ndarray):
            assert data.stop.ndim == 0
            assert str(data.stop.dtype) in theano.tensor.integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        elif isinstance(data.step, np.ndarray):
            assert data.step.ndim == 0
            assert str(data.step.dtype) in theano.tensor.integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        Constant.__init__(self, type, data, name)

    def signature(self):
        return (SliceConstant, self.data.start, self.data.stop, self.data.step)

    def __str__(self):
        return "%s{%s, %s, %s}" % (self.__class__.__name__,
                                   self.data.start,
                                   self.data.stop,
                                   self.data.step)
SliceType.Constant = SliceConstant


class NoneTypeT(Generic):
    """
    Inherit from Generic to have c code working.

    """

    def filter(self, x, strict=False, allow_downcast=None):
        if x is None:
            return x
        else:
            raise TypeError('Expected None!')

    @staticmethod
    def may_share_memory(a, b):
        # None never share memory between object, in the sence of DebugMode.
        # Python None are singleton
        return False

none_type_t = NoneTypeT()

# This is a variable instance. It can be used only once per fgraph.
# So use NoneConst.clone() before using it in a Theano graph.
# Use NoneConst.equals(x) to check if two variable are NoneConst.
NoneConst = Constant(none_type_t, None, name='NoneConst')
