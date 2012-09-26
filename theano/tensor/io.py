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

    See Also:
        recv
        send
        MPIRecvWait

    @note: Non-differentiable.
    """

    def __init__(self, source, tag, shape, dtype):
        self.source = source
        self.tag  = tag
        self.shape = shape
        self.dtype = numpy.dtype(dtype) # turn "float64" into numpy.float64
        self.broadcastable = (False,) * len(shape)
        self._info = (source, tag, shape, dtype)

    def __eq__(self, other):
        return (type(self) == type(other) and self._info == other._info)

    def __hash__(self):
        return hash(self._info)

    def make_node(self):
        return gof.Apply(self, [], [theano.Variable(Generic()),
                                    tensor(self.dtype,
                                           broadcastable=self.broadcastable)])
    def perform(self, node, inp, out):

        data = numpy.empty(self.shape, dtype=self.dtype)
        request = comm.Irecv(data, self.source, self.tag)

        out[0][0] = request
        out[1][0] = data

    def __str__(self):
        return "MPIRecv{source: %d, tag: %d, shape: %s, dtype: %s}"%self._info

    def infer_shape(self, node, shapes):
        return [None, self.shape]

    def do_constant_folding(self, node):
        return False

class MPIRecvWait(Op):
    """
    An operation to wait on a previously received array using MPI

    See Also:
        recv
        send
        MPIRecv

    @note: Non-differentiable.
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, request, data):
        return gof.Apply(self, [request, data],
                               [tensor(data.dtype,
                                       broadcastable=data.broadcastable)])
    def perform(self, node, inp, out):

        request = inp[0]
        data    = inp[1]

        request.wait()

        out[0][0] = data

    def __str__(self):
        return "MPIRecvWait"

    def infer_shape(self, node, shapes):
        return [shapes[1]]

class MPISend(Op):
    """
    An operation to asynchronously send an array to a remote host using MPI

    See Also:
        send
        recv
        MPISendWait

    @note: Non-differentiable.
    """

    def __init__(self, dest, tag):
        self.dest = dest
        self.tag  = tag
        self._info = (dest, tag)

    def __eq__(self, other):
        return (type(self) == type(other) and self._info == other._info)

    def __hash__(self):
        return hash(self._info)

    def make_node(self, data):
        return gof.Apply(self, [data],
                               [theano.Variable(Generic())])

    def perform(self, node, inp, out):

        data = inp[0]

        request = comm.Isend(data, self.dest, self.tag)

        out[0][0] = request

    def __str__(self):
        return "MPISend{dest: %d, tag: %d}"%self._info

class MPISendWait(Op):
    """
    An operation to wait on a previously sent array using MPI

    See Also:
        send
        recv
        MPISend

    @note: Non-differentiable.
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, request):
        return gof.Apply(self, [request],
                               [theano.Variable(Generic())])

    def perform(self, node, inp, out):
        request = inp[0]
        request.wait()
        out[0][0] = True

    def __str__(self):
        return "MPISendWait"

def isend(var, dest, tag):
    """ Asynchronous version of send

    See Also:
        send
        irecv
    """
    return MPISend(dest, tag)(var)

def send(var, dest, tag):
    """Send a variable to a remote host using MPI

    inputs:
        var  - variable to send                   :: A Theano Tensor
        dest - rank of the destination host       :: int
        tag  - unique identifier of this transfer :: int

    outputs  - a boolean value True that the send succeeded
             - You must include this value in the outputs of your function

    >>> x = theano.tensor.matrix('x')
    >>> finished = send(x, 1, 123) # send x to machine with rank 1
    >>> f = theano.function([x], finished)

    See Also:
        recv
    """
    return MPISendWait()(isend(var, dest, tag))

def irecv(shape, dtype, source, tag):
    """ Asynchronous version of recv

    See Also:
        recv
        isend
    """
    return MPIRecv(source, tag, shape, dtype)()

def recv(shape, dtype, source, tag):
    """Receive a variable from a remote host using MPI

    You must know the shape and dtype of the incoming variable

    inputs:
        shape   - shape of the received variable        :: (int, int)
        dtype   - dtype of the received variable        :: dtype
        source  - rank of the source host               :: int
        tag     - unique identifier of this transfer    :: int

    outputs  - a theano Tensor received from the remote host

    >>> x = recv((10, 10), 'float32', 0, 123)
    >>> f = theano.function([], x)

    See Also:
        send
    """
    return MPIRecvWait()(*irecv(shape, dtype, source, tag))
