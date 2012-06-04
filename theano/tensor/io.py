import numpy
import theano
from theano import gof
from theano.gof import Apply, Constant, Generic, Op, Type, Value, Variable
from basic import tensor
##########################
# Disk Access
##########################

class LoadFromDisk(Op):
    """
    An operation to load an array from disk

    See Also
        load

    @note: Non-differentiable.
    """
    def __init__(self, dtype, broadcastable):
        self.dtype = dtype
        self.broadcastable = broadcastable

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.broadcastable == other.broadcastable and
                self.dtype == other.dtype)

    def __hash__(self):
        return hash((type(self), self.dtype, self.broadcastable))

    def make_node(self, path):
        if isinstance(path, str):
            path = Constant(Generic(), path)
        return gof.Apply(self, [path], [tensor(self.dtype,
                                        broadcastable=self.broadcastable)])

    def perform(self, node, inp, out):
        path = inp[0]
        d = numpy.load(path)
        out[0][0] = d[d.keys()[0]].astype(self.dtype)

    def __str__(self):
        return "Load: %s, %s"%(self.dtype, self.broadcastable)

def load(path, dtype, broadcastable):
    """
    Load an array from a .npz file

    >>> from theano import *
    >>> path = Variable(Generic())
    >>> x = tensor.load(path, 'int64', (False,))
    >>> y = x*2
    >>> fn = function([path], y)
    >>> fn("stored-array.npz")
    array([0, 2, 4, 6, 8], dtype=int64)
    """

    return LoadFromDisk(dtype, broadcastable)(path)

