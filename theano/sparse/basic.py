"""
Classes for handling sparse matrices.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""

import numpy
from scipy import sparse

from .. import gof
from .. import tensor


""" Types of sparse matrices to use for testing """
_mtypes = [sparse.csc_matrix, sparse.csr_matrix]
#_mtypes = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix, sparse.coo_matrix]
_mtype_to_str = {sparse.csc_matrix: "csc", sparse.csr_matrix: "csr"}


def _is_sparse_result(x):
    """
    @rtype: boolean
    @return: True iff x is a L{SparseResult} (and not a L{tensor.Tensor})
    """
    if not isinstance(x.type, Sparse) and not isinstance(x.type, tensor.Tensor):
        raise NotImplementedError("this function should only be called on results of type sparse.Sparse or tensor.Tensor, not,", x)
    return isinstance(x.type, Sparse)
def _is_dense_result(x):
    """
    @rtype: boolean
    @return: True unless x is a L{SparseResult} (and not a L{tensor.Tensor})
    """
    if not isinstance(x.type, Sparse) and not isinstance(x.type, tensor.Tensor):
        raise NotImplementedError("this function should only be called on results of type sparse.Sparse or tensor.Tensor, not,", x)
    return isinstance(x.type, tensor.Tensor)

def _is_sparse(x):
    """
    @rtype: boolean
    @return: True iff x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("this function should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, sparse.spmatrix)
def _is_dense(x):
    """
    @rtype: boolean
    @return: True unless x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("this function should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, numpy.ndarray)



# Wrapper type

def as_sparse(x):
    """
    Wrapper around SparseResult constructor.
    @param x:  A sparse matrix. as_sparse reads dtype and format properties
               out of this sparse matrix.
    @return:   SparseResult version of sp.

    @todo Verify that sp is sufficiently sparse, and raise a warning if it is not
    """
    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, gof.Result):
        if not isinstance(x.type, Sparse):
            raise TypeError("Result type field must be a Sparse.", x, x.type)
        return x
    try:
        return constant(x)
    except TypeError:
        raise TypeError("Cannot convert %s to Sparse" % x, type(x))

def constant(x):
    if not isinstance(x, sparse.spmatrix):
        raise TypeError("sparse.constant must be called on a scipy.sparse.spmatrix")
    try:
        return SparseConstant(Sparse(format = x.format,
                                     dtype = x.dtype), x)
    except TypeError:
        raise TypeError("Could not convert %s to Sparse" % x, type(x))

def value(x):
    if not isinstance(x, sparse.spmatrix):
        raise TypeError("sparse.value must be called on a scipy.sparse.spmatrix")
    try:
        return SparseValue(Sparse(format = x.format,
                                  dtype = x.dtype), x)
    except TypeError:
        raise TypeError("Could not convert %s to Sparse" % x, type(x))



class Sparse(gof.Type):
    """
    @type dtype: numpy dtype string such as 'int64' or 'float64' (among others)
    @type format: string
    @ivar format: The sparse storage strategy.

    @note As far as I can tell, L{scipy.sparse} objects must be matrices, i.e. have dimension 2.
    """
    format_cls = {
            'csr' : sparse.csr_matrix,
            'csc' : sparse.csc_matrix
            }
    dtype_set = set(['int', 'int32', 'int64', 'float32', 'float64'])
    ndim = 2

    def __init__(self, format, dtype = 'float64'):
        """
        Fundamental way to create a sparse node.
        @param dtype:   Type of numbers in the matrix.
        @param format:  The sparse storage strategy.
        @return         An empty SparseResult instance.
        """

        dtype = str(dtype)
        if dtype in self.dtype_set:
            self.dtype = dtype
        else:
            raise NotImplementedError('unsupported dtype "%s" not in list' % dtype, list(self.dtype_set))
        
        assert isinstance(format, str)
        if format in self.format_cls:
            self.format = format
        else:
            raise NotImplementedError('unsupported format "%s" not in list' % format, self.format_cls.keys())

    def filter(self, value, strict = False):
        if isinstance(value, self.format_cls[self.format])\
                and value.dtype == self.dtype:
            return value
        if strict:
            raise TypeError("%s is not sparse" % value)
        sp = self.format_cls[self.format](value)
        if str(sp.dtype) != self.dtype:
            raise NotImplementedError()
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    def make_result(self, name = None):
        return SparseResult(self, name = name)

    def __eq__(self, other):
        return type(self) == type(other) and other.dtype == self.dtype and other.format == self.format

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.format)

    def __str__(self):
        return "Sparse[%s, %s]" % (str(self.dtype), str(self.format))

    def __repr__(self):
        return "Sparse[%s, %s]" % (str(self.dtype), str(self.format))


class _sparse_py_operators:
    T = property(lambda self: transpose(self), doc = "Return aliased transpose of self (read-only)")
    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)


class SparseResult(gof.Result, _sparse_py_operators):
    pass

class SparseConstant(gof.Constant, _sparse_py_operators):
    pass

class SparseValue(gof.Value, _sparse_py_operators):
    pass

    
    
#
# Conversion
#
# convert a sparse matrix to an ndarray
class DenseFromSparse(gof.op.Op):
    def make_node(self, x):
        x = as_sparse(x)
        return gof.Apply(self,
                         [x],
                         [tensor.Tensor(dtype = x.type.dtype,
                                        broadcastable = (False, False)).make_result()])
    def perform(self, node, (x, ), (out, )):
        out[0] = numpy.asarray(x.todense())
    def grad(self, (x, ), (gz, )):
        return SparseFromDense(x.type.format)(gz),
dense_from_sparse = DenseFromSparse()

class SparseFromDense(gof.op.Op):
    def __init__(self, format):
        self.format = format
    def make_node(self, x):
        x = tensor.as_tensor(x)
        return gof.Apply(self,
                         [x],
                         [Sparse(dtype = x.type.dtype,
                                 format = self.format).make_result()])
    def perform(self, node, (x, ), (out, )):
        out[0] = Sparse.format_cls[self.format](x)
    def grad(self, (x, ), (gz, )):
        return dense_from_sparse(gz),
    def __eq__(self, other):
        return type(self) == type(other) and self.format == other.format
    def __hash__(self):
        return hash(self.format) ^ hash(DenseFromSparse)
csr_from_dense = SparseFromDense('csr')
csc_from_dense = SparseFromDense('csc')



# Linear Algebra

class Transpose(gof.op.Op):
    format_map = {'csr' : 'csc',
                  'csc' : 'csr'}
    def make_node(self, x):
        x = as_sparse(x)
        return gof.Apply(self,
                         [x],
                         [Sparse(dtype = x.type.dtype,
                                 format = self.format_map[x.type.format]).make_result()])
    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = x.transpose()
    def grad(self, (x,), (gz,)):
        assert _is_sparse_result(x) and _is_sparse_result(gz)
        return transpose(gz),
transpose = Transpose()

class AddSS(gof.op.Op):
    ''' Add two sparse matrices '''
    def make_node(self, x, y):
        x, y = map(as_sparse, [x, y])
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        if x.type.format != y.type.format:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [Sparse(dtype = x.type.dtype,
                                 format = x.type.format).make_result()])
    def perform(self, node, (x, y), (out, )): 
        assert _is_sparse(x) and _is_sparse(y)
        out[0] = x + y
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(x) and _is_sparse_result(y)
        assert _is_sparse_result(gz)
        return gz, gz
add_s_s = AddSS()
class AddSD(gof.op.Op):
    ''' Add a sparse and a dense matrix '''
    def make_node(self, x, y):
        x, y = as_sparse(x), tensor.as_tensor(y)
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        return gof.Apply(self,
                         [x, y],
                         [tensor.Tensor(dtype = y.type.dtype,
                                        broadcastable = y.type.broadcastable).make_result()])
    def perform(self, node, (x, y), (out, )): 
        assert _is_sparse(x) and _is_dense(y)
        out[0] = x + y
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(x) and _is_dense_result(y)
        assert _is_dense_result(gz)
        return SparseFromDense(x.type.format)(gz), gz
add_s_d = AddSD()
def add(x,y):
    """
    Add two matrices, at least one of which is sparse.
    """
    if hasattr(x, 'getnnz'): x = as_sparse(x)
    if hasattr(y, 'getnnz'): y = as_sparse(y)
    
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
    def __init__(self, grad_preserves_dense=True):
        self.grad_preserves_dense = grad_preserves_dense
    def make_node(self, x, y):
        """
        Because of trickiness of implementing, we assume that the left argument x is SparseResult (not dense)
        """
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()

        assert _is_sparse_result(x)
        # These are the conversions performed by scipy.sparse.dot
        if x.type.format == "csc" or x.type.format == "coo":
            myformat = "csc"
        elif x.type.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        inputs = [x, y]    # Need to convert? e.g. assparse
        outputs = [Sparse(dtype = x.type.dtype, format = myformat).make_result()]
        return gof.Apply(self, inputs, outputs)
    def perform(self, node, (x, y), (out, )):
        """
        @todo: Verify that output is sufficiently sparse, and raise a warning if it is not
        @todo: Also determine that we are storing the output in the best storage format?
        """
        out[0] = x.dot(y)
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(gz)
        assert _is_sparse_result(x)
        rval = [dot(gz, y.T), dot(x.T, gz)]
        if _is_dense_result(y):
            if self.grad_preserves_dense:
                rval[1] = dense_from_sparse(rval[1])
        return rval
    def __eq__(self, other):
        return type(self) == type(other) and self.grad_preserves_dense == other.grad_preserves_dense
    def __hash__(self):
        return hash(self.grad_preserves_dense)
    
def dot(x, y, grad_preserves_dense=True):
    """
    @todo: Maybe the triple-transposition formulation (when x is dense)
    is slow. See if there is a direct way to do this.
    """
    if hasattr(x, 'getnnz'): x = as_sparse(x)
    if hasattr(y, 'getnnz'): y = as_sparse(y)

    x_is_sparse_result = _is_sparse_result(x)
    y_is_sparse_result = _is_sparse_result(y)
    if not x_is_sparse_result and not y_is_sparse_result:
        raise TypeError()
    if x_is_sparse_result:
        return Dot(grad_preserves_dense)(x, y)
    else:
        assert y_is_sparse_result
        return transpose(Dot(grad_preserves_dense)(y.T, x.T))
