"""
Classes for handling sparse matrices.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""

import copy #for __copy__
import numpy
from scipy import sparse

import gof.op, gof.result
import tensor


# Wrapper type

def assparse(sp, **kwargs):
    """
    Wrapper around SparseR constructor.
    @param sp:  A sparse matrix. assparse reads dtype and format properties
                out of this sparse matrix.
    @return:    SparseR version of sp.

    @todo Verify that sp is sufficiently sparse, and raise a warning if it is not
    """
    if isinstance(sp, SparseR):
        return sp
    else:
        # @todo Verify that sp is sufficiently sparse, and raise a
        # warning if it is not
        rval = SparseR(str(sp.dtype), sp.format, **kwargs)
        rval.data = sp
        return rval

class SparseR(gof.result.Result):
    """
    Attribute:
    format - a string identifying the type of sparsity

    Properties:
    T - read-only: return a transpose of self

    Methods:

    Notes:

    """
    format_cls = {
            'csr' : sparse.csr_matrix,
            'csc' : sparse.csc_matrix
            }
    dtype_set = set(['int', 'int32', 'int64', 'float32', 'float64'])

    def __init__(self, dtype, format, **kwargs):
        """
        Fundamental way to do create a sparse node.
        @param dtype:   Type of numbers in the matrix.
        @param format:  The sparse storage strategy.
        @return         An empty SparseR instance.
        """

        gof.Result.__init__(self, **kwargs)
        if dtype in SparseR.dtype_set:
            self._dtype = dtype
        assert isinstance(format, str)

        #print format, type(format), SparseR.format_cls.keys(), format in SparseR.format_cls
        if format in SparseR.format_cls:
            self._format = format
        else:
            raise NotImplementedError('unsupported format "%s" not in list' % format, SparseR.format_cls.keys())

    def filter(self, value):
        if isinstance(value, SparseR.format_cls[self.format])\
                and value.dtype == self.dtype:
                    return value
        #print 'pass-through failed', type(value)
        sp = SparseR.format_cls[self.format](value)
        if str(sp.dtype) != self.dtype:
            raise NotImplementedError()
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    def __copy__(self):
        if self.name is not None:
            rval = SparseR(self._dtype, self._format, name=self.name)
        else:
            rval = SparseR(self._dtype, self._format)
        rval.data = copy.copy(self.data)
        return rval


    dtype = property(lambda self: self._dtype)
    format = property(lambda self: self._format)
    T = property(lambda self: transpose(self), doc = "Return aliased transpose")


    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)


#
# Conversion
#

# convert a sparse matrix to an ndarray
class DenseFromSparse(gof.op.Op):
    def __init__(self, x, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        self.inputs = [assparse(x)]
        self.outputs = [tensor.Tensor(x.dtype,[0,0])]
    def impl(self, x):
        return numpy.asarray(x.todense())
    def grad(self, x, gz): 
        return sparse_from_dense(gz, x.format)
dense_from_sparse = gof.op.constructor(DenseFromSparse)

class SparseFromDense(gof.op.Op):
    def __init__(self, x, format, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        if isinstance(format, gof.result.Result):
            self.inputs = [tensor.astensor(x), format]
        else:
            self.inputs =  [tensor.astensor(x), gof.result.PythonResult()]
            self.inputs[1].data = format
        self.outputs = [SparseR(x.dtype, self.inputs[1].data)]
    def impl(self, x, fmt):
        # this would actually happen anyway when we try to assign to
        # self.outputs[0].data, but that seems hackish -JB
        return SparseR.format_cls[fmt](x)
    def grad(self, (x, fmt), gz):
        return dense_from_sparse(gz)
sparse_from_dense = gof.op.constructor(SparseFromDense)

# Linear Algebra

class Transpose(gof.op.Op):
    format_map = {
            'csr' : 'csc',
            'csc' : 'csr'}
    def __init__(self, x, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x = assparse(x)
        self.inputs = [x]
        self.outputs = [SparseR(x.dtype, Transpose.format_map[x.format])]
    def impl(self, x):
        return x.transpose() 
    def grad(self, x, gz): 
        return transpose(gz)
transpose = gof.op.constructor(Transpose)

class AddSS(gof.op.Op): #add two sparse matrices
    def __init__(self, x, y, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x, y = [assparse(x), assparse(y)]
        self.inputs = [x, y]
        if x.dtype != y.dtype:
            raise NotImplementedError()
        if x.format != y.format:
            raise NotImplementedError()
        self.outputs = [SparseR(x.dtype, x.format)]
    def impl(self, x,y): 
        return x + y
    def grad(self, (x, y), gz):
        return gz, gz
add_s_s = gof.op.constructor(AddSS)

class Dot(gof.op.Op):
    """
    Attributes:
    grad_preserves_dense - a boolean flags [default: True].
    grad_preserves_dense controls whether gradients with respect to inputs
    are converted to dense matrices when the corresponding input y is
    dense (not in a L{SparseR} wrapper). This is generally a good idea
    when L{Dot} is in the middle of a larger graph, because the types
    of gy will match that of y. This conversion might be inefficient if
    the gradients are graph outputs though, hence this mask.
    """
    def __init__(self, x, y, grad_preserves_dense=True):
        """
        Because of trickiness of implementing, we assume that the left argument x is SparseR (not dense)
        """
        if x.dtype != y.dtype:
            raise NotImplementedError()

        # These are the conversions performed by scipy.sparse.dot
        if x.format == "csc" or x.format == "coo":
            myformat = "csc"
        elif x.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        self.inputs = [x, y]    # Need to convert? e.g. assparse
        self.outputs = [SparseR(x.dtype, myformat)]
        self.grad_preserves_dense = grad_preserves_dense
    def perform(self):
        """
        @todo: Verify that output is sufficiently sparse, and raise a warning if it is not
        @todo: Also determine that we are storing the output in the best storage format?
        """
        self.outputs[0].data = self.inputs[0].data.dot(self.inputs[1].data)
    def grad(self, (x, y), (gz,)):
        rval = [dot(gz, y.T), dot(x.T, gz)]
        assert isinstance(self.inputs[0], SparseR)
        if not isinstance(self.inputs[1], SparseR):
            if self.grad_preserves_dense:
                rval[1] = dense_from_sparse(rval[1])
        return rval
    def __copy__(self):
        return self.__class__(self.inputs[0], self.inputs[1], self.grad_preserves_dense)
    def clone_with_new_inputs(self, *new_inputs):
        return self.__class__(new_inputs[0], new_inputs[1], self.grad_preserves_dense)
def dot(x, y, grad_preserves_dense=True):
    """
    @todo: Maybe the triple-transposition formulation (when x is dense)
    is slow. See if there is a direct way to do this.
    """
    if hasattr(x, 'getnnz'): x = assparse(x)
    if hasattr(y, 'getnnz'): y = assparse(y)

    x_is_sparse = isinstance(x, SparseR)
    y_is_sparse = isinstance(y, SparseR)
    if not x_is_sparse and not y_is_sparse:
        raise TypeError()
    if x_is_sparse:
        return Dot(x,y,grad_preserves_dense).outputs[0]
    else:
        return transpose(Dot(transpose(y), transpose(x), grad_preserves_dense).outputs[0])
