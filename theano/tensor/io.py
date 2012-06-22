import numpy
import theano
from theano import gof
from theano.gof import Apply, Constant, Generic, Op, Type, Variable
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
    def __init__(self, dtype, broadcastable, mmap_mode=None):
        self.dtype = numpy.dtype(dtype) # turn "float64" into numpy.float64
        self.broadcastable = broadcastable
        self.mmap_mode = mmap_mode
        self._info = (dtype, broadcastable, mmap_mode)

    def __eq__(self, other):
        return (type(self) == type(other) and self._info == other._info)

    def __hash__(self):
        return hash(self._info)

    def make_node(self, path):
        if isinstance(path, str):
            path = Constant(Generic(), path)
        return gof.Apply(self, [path], [tensor(self.dtype,
                                        broadcastable=self.broadcastable)])

    def perform(self, node, inp, out):
        path = inp[0]
        if (path.split('.')[-1] == 'npz'):
            raise ValueError("Expected a .npy file, got %s instead"%path)
        result = numpy.load(path, mmap_mode=self.mmap_mode)
        if result.dtype != self.dtype:
            raise TypeError("Expected an array of type %s, got %s instead"%
                    (self.dtype, result.dtype))
        out[0][0] = result

    def __str__(self):
        return "Load{dtype:%s, broadcastable:%s, mmep:%s}"%self._info

def load(path, dtype, broadcastable, mmap_mode=None):
    """
    Load an array from an .npy file

    >>> from theano import *
    >>> path = Variable(Generic())
    >>> x = tensor.load(path, 'int64', (False,))
    >>> y = x*2
    >>> fn = function([path], y)
    >>> fn("stored-array.npy")
    array([0, 2, 4, 6, 8], dtype=int64)
    """

    return LoadFromDisk(dtype, broadcastable, mmap_mode)(path)

