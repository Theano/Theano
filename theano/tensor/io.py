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

##########################
# MPI
##########################

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_enabled = True
except:
    mpi_enabled = False


class MPIRecv(Op):
    """
    An operation to asynchronously receive an array to a remote host using MPI

    See Also
       MPIRecv
       MPIWait

    @note: Non-differentiable.
    """

    def __init__(self, rank, tag, dtype, shape):
        self.rank = rank
        self.tag  = tag
        self.shape = shape
        self.dtype = numpy.dtype(dtype) # turn "float64" into numpy.float64
        self.broadcastable = (False,)*len(shape)
        self._info = (rank, tag, dtype, shape)

    def __eq__(self, other):
        return (type(self) == type(other) and self._info == other._info)

    def __hash__(self):
        return hash(self._info)

    def make_node(self):
        return gof.Apply(self, [], [theano.Generic(),
                                    tensor(self.dtype,
                                           broadcastable=self.broadcastable)])
    def perform(self, node, inp, out):

        data = numpy.empty(self.shape, dtype=self.dtype)
        request = comm.Irecv(data, self.rank, self.tag)

        out[0][0] = request
        out[0][1] = data

    def __str__(self):
        return "MPIRecv{source: %d, tag: %d, dtype:%s, shape:%s, :%s}"%self._info

class MPIRecvWait(Op):
    """
    An operation to wait on a previously received array using MPI

    See Also
       MPIRecv

    @note: Non-differentiable.
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(self.type)

    def make_node(self):
        return gof.Apply(self, [theano.Generic(),
                                tensor(self.dtype,
                                       broadcastable=self.broadcastable)],
                               [tensor(self.dtype,
                                       broadcastable=self.broadcastable)])
    def perform(self, node, inp, out):

        request = inp[0][0]
        data    = inp[0][1]

        request.wait()

        out[0][0] = data

    def __str__(self):
        return "MPIRecvWait"

class MPISend(Op):
    """
    An operation to asynchronously Send an array to a remote host using MPI

    See Also
       MPIRecv
       MPISendWait

    @note: Non-differentiable.
    """

    def __init__(self, rank, tag):
        self.rank = rank
        self.tag  = tag
        self._info = (rank, tag)

    def __eq__(self, other):
        return (type(self) == type(other) and self._info == other._info)

    def __hash__(self):
        return hash(self._info)

    def make_node(self):
        return gof.Apply(self, [tensor(self.dtype, broadcastable=self.broadcastable)],
                               [theano.Generic()])

    def perform(self, node, inp, out):

        data = inp[0][0]

        request = comm.Isend(data, self.rank, self.tag)

        out[0][0] = request

    def __str__(self):
        return "MPISend{dest: %d, tag: %d}"%self._info

class MPISendWait(Op):
    """
    An operation to wait on a previously sent array using MPI

    See Also:
       MPISend

    @note: Non-differentiable.
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(self.type)

    def make_node(self):
        return gof.Apply(self, [theano.Generic()], [])

    def perform(self, node, inp, out):
        request = inp[0][0]
        request.wait()

    def __str__(self):
        return "MPISendWait"
