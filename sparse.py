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


""" Types of sparse matrices to use for testing """
_mtypes = [sparse.csc_matrix, sparse.csr_matrix]
#_mtypes = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix, sparse.coo_matrix]
_mtype_to_str = {sparse.csc_matrix: "csc", sparse.csr_matrix: "csr"}


## Type checking

def _is_sparse_result(x):
    """
    @rtype: boolean
    @return: True iff x is a L{SparseResult} (and not a L{tensor.Tensor})
    """
    if not isinstance(x, SparseResult) and not isinstance(x, tensor.Tensor):
        raise NotImplementedError("_is_sparse should only be called on sparse.SparseResult or tensor.Tensor, not,", x)
    return isinstance(x, SparseResult)
def _is_dense_result(x):
    """
    @rtype: boolean
    @return: True unless x is a L{SparseResult} (and not a L{tensor.Tensor})
    """
    if not isinstance(x, SparseResult) and not isinstance(x, tensor.Tensor):
        raise NotImplementedError("_is_sparse should only be called on sparse.SparseResult or tensor.Tensor, not,", x)
    return isinstance(x, tensor.Tensor)

def _is_sparse(x):
    """
    @rtype: boolean
    @return: True iff x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("_is_sparse should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, sparse.spmatrix)
def _is_dense(x):
    """
    @rtype: boolean
    @return: True unless x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("_is_sparse should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, numpy.ndarray)



# Wrapper type

def assparse(sp, **kwargs):
    """
    Wrapper around SparseResult constructor.
    @param sp:  A sparse matrix. assparse reads dtype and format properties
                out of this sparse matrix.
    @return:    SparseResult version of sp.

    @todo Verify that sp is sufficiently sparse, and raise a warning if it is not
    """
    if isinstance(sp, SparseResult):
        rval = sp
    else:
        # @todo Verify that sp is sufficiently sparse, and raise a
        # warning if it is not
        rval = SparseResult(str(sp.dtype), sp.format, **kwargs)
        rval.data = sp
    assert _is_sparse_result(rval)
    return rval

class SparseResult(gof.result.Result):
    """
    @type _dtype: numpy dtype string such as 'int64' or 'float64' (among others)
    @type _format: string
    @ivar _format: The sparse storage strategy.

    @note As far as I can tell, L{scipy.sparse} objects must be matrices, i.e. have dimension 2.
    """
    format_cls = {
            'csr' : sparse.csr_matrix,
            'csc' : sparse.csc_matrix
            }
    dtype_set = set(['int', 'int32', 'int64', 'float32', 'float64'])

    def __init__(self, dtype, format, **kwargs):
        """
        Fundamental way to create a sparse node.
        @param dtype:   Type of numbers in the matrix.
        @param format:  The sparse storage strategy.
        @return         An empty SparseResult instance.
        """

        gof.Result.__init__(self, **kwargs)
        if dtype in SparseResult.dtype_set:
            self._dtype = dtype
        assert isinstance(format, str)

        #print format, type(format), SparseResult.format_cls.keys(), format in SparseResult.format_cls
        if format in SparseResult.format_cls:
            self._format = format
        else:
            raise NotImplementedError('unsupported format "%s" not in list' % format, SparseResult.format_cls.keys())

    def filter(self, value):
        if isinstance(value, SparseResult.format_cls[self.format])\
                and value.dtype == self.dtype:
                    return value
        #print 'pass-through failed', type(value)
        sp = SparseResult.format_cls[self.format](value)
        if str(sp.dtype) != self.dtype:
            raise NotImplementedError()
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    def __copy__(self):
        if self.name is not None:
            rval = SparseResult(self._dtype, self._format, name=self.name)
        else:
            rval = SparseResult(self._dtype, self._format)
        rval.data = copy.copy(self.data)
        return rval


    dtype = property(lambda self: self._dtype)
    format = property(lambda self: self._format)
    T = property(lambda self: transpose(self), doc = "Return aliased transpose of self (read-only)")


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
        assert _is_sparse(x)
        return numpy.asarray(x.todense())
    def grad(self, (x,), (gz,)):
        assert _is_sparse_result(x) and _is_dense_result(gz)
        return sparse_from_dense(gz, x.format),
dense_from_sparse = gof.op.constructor(DenseFromSparse)

class SparseFromDense(gof.op.Op):
    def __init__(self, x, format, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        if isinstance(format, gof.result.Result):
            self.inputs = [tensor.astensor(x), format]
        else:
            self.inputs =  [tensor.astensor(x), gof.result.PythonResult()]
            self.inputs[1].data = format
        self.outputs = [SparseResult(x.dtype, self.inputs[1].data)]
    def impl(self, x, fmt):
        # this would actually happen anyway when we try to assign to
        # self.outputs[0].data, but that seems hackish -JB
        assert _is_dense(x)
        return SparseResult.format_cls[fmt](x)
    def grad(self, (x, fmt), (gz,)):
        assert _is_dense_result(x) and _is_sparse_result(gz)
        return dense_from_sparse(gz), None
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
        self.outputs = [SparseResult(x.dtype, Transpose.format_map[x.format])]
    def impl(self, x):
        assert _is_sparse(x)
        return x.transpose() 
    def grad(self, (x,), (gz,)):
        assert _is_sparse_result(x) and _is_sparse_result(gz)
        return transpose(gz),
transpose = gof.op.constructor(Transpose)

class AddSS(gof.op.Op):
    ''' Add two sparse matrices '''
    def __init__(self, x, y, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x, y = [assparse(x), assparse(y)]
        self.inputs = [x, y]
        if x.dtype != y.dtype:
            raise NotImplementedError()
        if x.format != y.format:
            raise NotImplementedError()
        self.outputs = [SparseResult(x.dtype, x.format)]
    def impl(self, x,y): 
        assert _is_sparse(x) and _is_sparse(y)
        return x + y
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(x) and _is_sparse_result(y)
        assert _is_sparse_result(gz)
        return gz, gz
add_s_s = gof.op.constructor(AddSS)
class AddSD(gof.op.Op):
    ''' Add a sparse and a dense matrix '''
    def __init__(self, x, y, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x, y = [assparse(x), tensor.astensor(y)]
        self.inputs = [x, y]
        if x.dtype != y.dtype:
            raise NotImplementedError()
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert len(y.broadcastable) == 2
        self.outputs = [tensor.Tensor(y.dtype, y.broadcastable)]
    def impl(self, x,y): 
        assert _is_sparse(x) and _is_dense(y)
        return x + y
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(x) and _is_dense_result(y)
        assert _is_dense_result(gz)
        return SparseFromDense(gz), gz
add_s_d = gof.op.constructor(AddSD)
def add(x,y):
    """
    Add two matrices, at least one of which is sparse.
    """
    if hasattr(x, 'getnnz'): x = assparse(x)
    if hasattr(y, 'getnnz'): y = assparse(y)
    
    x_is_sparse_result = _is_sparse_result(x)
    y_is_sparse_result = _is_sparse_result(y)

    assert x_is_sparse_result or y_is_sparse_result
    if x_is_sparse_result and y_is_sparse_result: return add_s_s(x,y)
    elif x_is_sparse_result and not y_is_sparse_result: return add_s_d(x,y)
    elif y_is_sparse_result and not x_is_sparse_result: return add_s_d(y,x)
    else: raise NotImplementedError()


class Dot(gof.op.Op):
    """
    Attributes:
    grad_preserves_dense - a boolean flags [default: True].
    grad_preserves_dense controls whether gradients with respect to inputs
    are converted to dense matrices when the corresponding input y is
    dense (not in a L{SparseResult} wrapper). This is generally a good idea
    when L{Dot} is in the middle of a larger graph, because the types
    of gy will match that of y. This conversion might be inefficient if
    the gradients are graph outputs though, hence this mask.

    @todo: Simplify code by splitting into DotSS and DotSD.
    """
    def __init__(self, x, y, grad_preserves_dense=True):
        """
        Because of trickiness of implementing, we assume that the left argument x is SparseResult (not dense)
        """
        if x.dtype != y.dtype:
            raise NotImplementedError()

        assert _is_sparse_result(x)
        # These are the conversions performed by scipy.sparse.dot
        if x.format == "csc" or x.format == "coo":
            myformat = "csc"
        elif x.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        self.inputs = [x, y]    # Need to convert? e.g. assparse
        self.outputs = [SparseResult(x.dtype, myformat)]
        self.grad_preserves_dense = grad_preserves_dense
    def perform(self):
        """
        @todo: Verify that output is sufficiently sparse, and raise a warning if it is not
        @todo: Also determine that we are storing the output in the best storage format?
        """
        self.outputs[0].data = self.inputs[0].data.dot(self.inputs[1].data)
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(gz)
        rval = [dot(gz, y.T), dot(x.T, gz)]
        assert _is_sparse_result(x)
        if _is_dense_result(y):
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

    x_is_sparse_result = _is_sparse_result(x)
    y_is_sparse_result = _is_sparse_result(y)
    if not x_is_sparse_result and not y_is_sparse_result:
        raise TypeError()
    if x_is_sparse_result:
        return Dot(x, y, grad_preserves_dense).outputs[0]
    else:
        assert y_is_sparse_result
        return transpose(Dot(y.T, x.T, grad_preserves_dense).outputs[0])

