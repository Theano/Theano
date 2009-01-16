"""
Classes for handling sparse matrices.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""

import sys, operator
import numpy
from scipy import sparse

from .. import gof
from .. import tensor
from .. import compile

#TODO: move this decorator to the compile submodule
def register_specialize(lopt, *tags, **kwargs):
    compile.optdb['specialize'].register((kwargs and kwargs.pop('name')) or lopt.__name__, lopt, 'fast_run', *tags)


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

def sp_ones_like(x):
    data, indices, indptr, shape = csm_properties(x) #TODO: don't restrict to CSM formats
    return CSM(format=x.format)(tensor.ones_like(data), indices, indptr, shape)


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

csc_matrix = Sparse(format='csc')
csr_matrix = Sparse(format='csr')

class _sparse_py_operators:
    T = property(lambda self: transpose(self), doc = "Return aliased transpose of self (read-only)")
    def __neg__(self): return neg(self)
    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)
    def __sub__(left, right): return sub(left, right)
    def __rsub__(right, left): return sub(left, right)
    def __mul__(left, right): return mul(left, right)
    def __rmul__(left, right): return mul(left, right)

    #extra pseudo-operator symbols
    def __dot__(left, right): return structured_dot(left, right)
    def __rdot__(right, left): return structured_dot(left, right)


class SparseResult(gof.Result, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

class SparseConstant(gof.Constant, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

class SparseValue(gof.Value, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

    
# CONSTRUCTION
class CSMProperties(gof.Op):
    """Extract all of .data .indices and .indptr"""
    view_map = {0:[0],1:[0],2:[0],3:[0]}

    def __init__(self, map=None):
        self.map = map

    def make_node(self, csm):
        csm = as_sparse(csm)
        data = tensor.Tensor(dtype=csm.type.dtype, broadcastable = (False,)).make_result()
        return gof.Apply(self, [csm], 
                [data, tensor.ivector(), tensor.ivector(), tensor.ivector()])

    def perform(self, node, (csm,), out):
        out[0][0] = csm.data if self.map is None else csm.data[self.map]
        out[1][0] = numpy.asarray(csm.indices, dtype='int32')
        out[2][0] = numpy.asarray(csm.indptr, dtype='int32')
        out[3][0] = numpy.asarray(csm.shape, dtype='int32')

    # TODO FIX THIS
    def grad(self, (csm,), g):
        assert [gg is None for gg in g[1:]]
        data, indices, indptr, shape = csm_properties(csm)
        if csm.format == 'csc':
            return [CSM('csc')(g_data, indices, indptr, shape)]
        else:
            return [CSR('csm')(g_data, indices, indptr, shape)]

def csm_properties(csm): return CSMProperties()(csm)
def csm_data(csm): return csm_properties(csm)[0]
def csm_indices(csm): return csm_properties(csm)[1]
def csm_indptr(csm): return csm_properties(csm)[2]
def csm_shape(csm): return csm_properties(csm)[3]

class CSM(gof.Op):
    """Construct a CSC or CSR matrix from the internal representation """
    view_map = {0:[0]} #should view the other inputs too, but viewing multiple inputs is not
    #currently supported by the destroyhandler

    def __init__(self, format, map=None):
        if format not in ('csr', 'csc'):
            raise ValueError("format must be one of: 'csr', 'csc'", format)
        self.format = format
       
        # for efficiency, if remap does nothing, then do not apply it
        if map is not None and all(map==numpy.arange(numpy.size(map))):
            map = None

        self.map = map

    def __eq__(self, other):
        return type(other) is CSM \
                and other.format == self.format and other.map==self.map

    def __hash__(self):
        return hash(CSM) ^ hash(self.format) ^ hash(numpy.str(self.map))

    def make_node(self, data, indices, indptr, shape): 
        """Build a SparseResult from the internal parametrization
        
        :param data: 
        :param indices:
        :param indptr:
        :type data: 1-d tensor
        :type indices: 1-d tensor of ints
        :type indptr: 1-d tensor of ints

        """
        data = tensor.as_tensor(data)
        indices = tensor.as_tensor(indices)
        indptr = tensor.as_tensor(indptr)
        shape = tensor.as_tensor(shape)

        if data.type.ndim != 1:
            raise TypeError('data argument must be a vector', data.type)
        if indices.type not in tensor.int_vector_types:
            raise TypeError('indices must be vector of integers')
        if indptr.type not in tensor.int_vector_types:
            raise TypeError('indices must be vector of integers')
        if shape.type not in tensor.int_vector_types:
            raise TypeError('n_rows must be integer type')

        return gof.Apply(self,
                         [data, indices, indptr, shape],
                         [Sparse(dtype = data.type.dtype,
                                 format = self.format).make_result()])

    def perform(self, node, (data, indices, indptr, shape), (out,)):
        """Build a csc_matrix"""
        #assert len(data.flatten()) == len(indices.flatten())
        data = data[self.map] if self.map!=None else data

        if len(shape) != 2:
            raise ValueError('Shape should be an array of length 2')
        if data.shape != indices.shape:
            raise ValueError('data indices shape mismatch', (data.shape, indices.shape))
        if self.format == 'csc':
            out[0] = sparse.csc_matrix((data, indices.copy(), indptr.copy()), 
                    numpy.asarray(shape),
                    copy = False #1000*len(data.flatten())
                    )
        else:
            assert self.format == 'csr'
            out[0] = sparse.csr_matrix((data, indices.copy(), indptr.copy()), 
                    shape.copy(),
                    copy = False #1000*len(data.flatten())
                    )

    def grad(self, (data, indices, indptr, shape), (g_out,)):
        """Return a gradient on the data vector"""
        #unpack the data vector and wrap it as a 1d Tensor
        g_data = csm_grad(self.map)(data, csm_data(g_out),csm_indices(g_out))
        return [g_data, None, None, None]

CSC = CSM('csc')
CSR = CSM('csr')

class CSMGrad(gof.op.Op):
    def __init__(self, map=None):
        self.map = map

    def make_node(self, data, gout_data, gout_indices):
        g_data = data.type()
        return gof.Apply(self, [data, gout_data, gout_indices], [g_data])

    def perform(self, node, (data, gout_data, gout_indices), (g_data,)):
        if self.map is None:
            g_data[0] = gout_data
        else:
            grad = numpy.zeros_like(data)
            grad[self.map] = gout_data
            g_data[0] = grad
csm_grad = CSMGrad

@gof.local_optimizer([csm_properties])
def skip_pack_csc01(node):
    """if we find csm_properties(CSM(*args)), then we can replace that with the *args
    directly"""
    if node.op == csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            return csm.owner.inputs
    return False
register_specialize(skip_pack_csc01)


    
#
# Conversion
#
# convert a sparse matrix to an ndarray
class DenseFromSparse(gof.op.Op):
    sparse_grad = True
    def make_node(self, x):
        x = as_sparse(x)
        return gof.Apply(self,
                         [x],
                         [tensor.Tensor(dtype = x.type.dtype,
                                        broadcastable = (False, False)).make_result()])
    def perform(self, node, (x, ), (out, )):
        out[0] = x.toarray()
    def grad(self, (x, ), (gz, )):
        if self.sparse_grad:
            return [sp_ones_like(x) * gz]
        else:
            return [SparseFromDense(x.type.format)(gz)]
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

class Neg(gof.op.Op):
    def make_node(self, x):
        x = as_sparse(x)
        return gof.Apply(self, [x], [x.type()])
    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = -x
    def grad(self, (x,), (gz,)):
        assert _is_sparse_result(x) and _is_sparse_result(gz)
        return -gz,
neg = Neg()

class AddSS(gof.op.Op):
    '''Add two sparse matrices '''
    def make_node(self, x, y):
        x, y = map(as_sparse, [x, y])
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        if x.type.format != y.type.format:
            print x.type.format, y.type.format
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
        return sp_one_like(x) * gz, gz
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
def sub(x,y):
    return x + (-y)



class MulSS(gof.op.Op):
    ''' Elementwise multiply a sparse and a ndarray '''
    def make_node(self, x, y):
        x, y = as_sparse(x), as_sparse(y)
        if x.type != y.type:
            raise NotImplementedError()
        return gof.Apply(self, [x, y], [x.type()])
    def perform(self, node, (x, y), (out, )): 
        assert _is_sparse(x) and _is_sparse(y)
        assert len(x.shape) == 2
        assert y.shape == x.shape
        if (numpy.all(y.indptr == x.indptr) and numpy.all(y.indices == x.indices)):
            out[0] = y.copy()
            out[0].data *= x.data
        else:
            raise NotImplementedError() #RowScale / ColScale
    def grad(self, (x, y), (gz,)):
        return y * gz, x * gz
mul_s_s = MulSS()
class MulSD(gof.op.Op):
    ''' Elementwise multiply a sparse and a ndarray '''
    def make_node(self, x, y):
        x, y = as_sparse(x), tensor.as_tensor(y)
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        # Broadcasting of the sparse matrix is not supported.
        assert y.type.ndim <= 2
        return gof.Apply(self, [x, y], [x.type()])
    def perform(self, node, (x, y), (out, )): 
        assert _is_sparse(x) and _is_dense(y)
        if len(y.shape) == 0:
            out[0] = x.copy()
            out[0].data *= y
        elif len(y.shape) == 1:
            raise NotImplementedError() #RowScale / ColScale
        elif len(y.shape) == 2:
            #if we have enough memory to fit y, maybe we can fit x.asarray() too?
            #TODO: change runtime from O(M*N) to O(nonzeros)
            M, N = x.shape
            assert x.shape == y.shape

            if x.format == 'csc':
                x_data = x.data
                indices = x.indices
                indptr = x.indptr
                z = x.copy()
                z_data = z.data

                for j in xrange(0, N):
                    for i_idx in xrange(indptr[j], indptr[j+1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i,j]
                out[0] = z
            elif x.format == 'csr':
                x_data = x.data
                indices = x.indices
                indptr = x.indptr
                z = x.copy()
                z_data = z.data

                for i in xrange(0, M):
                    for j_idx in xrange(indptr[i], indptr[i+1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i,j]
                out[0] = z
            else:
                print >> sys.stderr, "WARNING: crappy implementation of MulSD", x.format
                out[0] = type(x)(x.toarray() * y)

    def grad(self, (x, y), (gz,)):
        assert _is_sparse_result(x) and _is_dense_result(y)
        assert _is_sparse_result(gz)
        return y * gz, x * gz
mul_s_d = MulSD()
def mul(x,y):
    """
    Multiply (elementwise) two matrices, at least one of which is sparse.
    """
    if hasattr(x, 'getnnz'): x = as_sparse(x)
    if hasattr(y, 'getnnz'): y = as_sparse(y)
    
    x_is_sparse_result = _is_sparse_result(x)
    y_is_sparse_result = _is_sparse_result(y)

    assert x_is_sparse_result or y_is_sparse_result
    if x_is_sparse_result and y_is_sparse_result: return mul_s_s(x,y)
    elif x_is_sparse_result and not y_is_sparse_result: return mul_s_d(x,y)
    elif y_is_sparse_result and not x_is_sparse_result: return mul_s_d(y,x)
    else: raise NotImplementedError()

###############
#
# TrueDot
#
class TrueDot(gof.op.Op):
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
    
def true_dot(x, y, grad_preserves_dense=True):
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


###############
#
# StructuredDot
#
class StructuredDot(gof.Op):
    """Structured Dot is like dot, except that only the gradient wrt non-zero elements of the
    sparse matrix A are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a Tensor instance.
    """
    def make_node(self, a, b):
        assert a.type.dtype == b.type.dtype
        if type(a) is not SparseResult:
            raise TypeError('First argument must be of type SparseResult');

        return gof.Apply(self, [a,b], [tensor.tensor(a.type.dtype, (False, False))])

    def perform(self, node, (a,b), (out,)):
        if a.shape[1] != b.shape[0]:
            raise ValueError('shape mismatch in StructuredDot.perform', (a.shape, b.shape))
        if b.shape[0] == 1:
            raise NotImplemented('ERROR: scipy.csc_matrix dot has bug with singleton dimensions')

        result = a.dot(b)

        # sparse dot generates sparse matrix, unless output has single dimension
        if sparse.issparse(result):
            result = result.toarray()
        assert isinstance(result, numpy.ndarray)

        # dot of an NxM sparse matrix, with a Mx1 dense matrix, returns vector not matrix
        if result.ndim == 1:
            result = numpy.expand_dims(result,1)
        elif result.ndim != 2:
            raise Exception('Output of structured dot should be a matrix (ndim=2)')

        assert result.ndim == 2

        ## Commenting this out because result should be a numpy.ndarray since the assert above
        ## (JB 20090109)
        #out[0] = numpy.asarray(result)  #TODO: fix this really bad implementation
        #
        out[0] = result

    def grad(self, (a,b), (g_out,)):
        #a is sparse, b is dense, g_out is dense
        #ga = g_out x b.T
        #gb = a.T x g_out
        return structured_dot_grad(a, b, g_out), structured_dot(a.T,g_out)
_structured_dot = StructuredDot()
def structured_dot(x, y):
    """
    @todo: Maybe the triple-transposition formulation (when x is dense)
    is slow. See if there is a direct way to do this.
    """
    if hasattr(x, 'getnnz'): x = as_sparse(x)
    if hasattr(y, 'getnnz'): y = as_sparse(y)

    x_is_sparse_result = _is_sparse_result(x)
    y_is_sparse_result = _is_sparse_result(y)
    if not x_is_sparse_result and not y_is_sparse_result:
        raise TypeError('structured_dot requires at least one sparse argument')
    if x_is_sparse_result:
        return _structured_dot(x, y)
    else:
        assert y_is_sparse_result
        return _structured_dot(y.T, x.T).T

class StructuredDotCSC(gof.Op):
    def make_node(self, a_val, a_ind, a_ptr, a_nrows, b):
        assert a_val.type.dtype == b.type.dtype
        r = gof.Apply(self, [a_val, a_ind, a_ptr, a_nrows, b], 
                [tensor.tensor(a_val.type.dtype, (False, False))])
        return r

    def perform(self, node, (a_val, a_ind, a_ptr, a_nrows, b), (out,)):
        a = sparse.csc_matrix((a_val, a_ind, a_ptr), 
                (a_nrows, b.shape[0]),
                copy = False)
        out[0] = numpy.asarray(a.dot(b).todense())

    def c_code(self, node, name, (a_val, a_ind, a_ptr, a_nrows, b), (z,), sub):
        return """
        if (%(a_val)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (%(a_ind)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (%(a_ptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (%(a_nrows)s->nd != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); %(fail)s;}
        if (%(b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (%(a_val)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "a_val dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(a_ind)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (%(a_ptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (%(a_nrows)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_nrows dtype not INT32"); %(fail)s;}

        if (%(b)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "b's dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(a_val)s->dimensions[0] != %(a_ind)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if (%(a_ptr)s->dimensions[0] != %(b)s->dimensions[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "a's number of columns doesn't match b's rows"); %(fail)s;}

        if ((!%(z)s)
            || (%(z)s->dimensions[0] != ((npy_int32 *)%(a_nrows)s->data)[0])
            || (%(z)s->dimensions[1] != %(b)s->dimensions[1])
            )
        {
            if (%(z)s) Py_DECREF(%(z)s);
            npy_intp dims[] = {0,0};
            dims[0] = ((npy_int32 *)%(a_nrows)s->data)[0];
            dims[1] = %(b)s->dimensions[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(b)s->descr->type_num);
        }

        {
            //the output array has size M x N
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(b)s->dimensions[0];
            npy_intp Szm = %(z)s->strides[0] / %(z)s->descr->elsize;
            npy_intp Szn = %(z)s->strides[1] / %(z)s->descr->elsize;
            //npy_intp Sbm = %(b)s->strides[0] / %(b)s->descr->elsize;
            npy_intp Sbn = %(b)s->strides[1] / %(b)s->descr->elsize;
            npy_intp Sval = %(a_val)s->strides[0] / %(a_val)s->descr->elsize;
            npy_intp Sind = %(a_ind)s->strides[0] / %(a_ind)s->descr->elsize;
            npy_intp Sptr = %(a_ptr)s->strides[0] / %(a_ptr)s->descr->elsize;

            npy_double * __restrict__ Dz = (npy_double*)%(z)s->data;
            //const npy_double * __restrict__ Db = (npy_double*)%(b)s->data;
            const npy_double * __restrict__ Dval = (npy_double*)%(a_val)s->data;
            const npy_int32 * __restrict__ Dind = (npy_int32*)%(a_ind)s->data;
            const npy_int32 * __restrict__ Dptr = (npy_int32*)%(a_ptr)s->data;

            //npy_intp nnz = %(a_ind)s->dimensions[0];

            //clear the output array
            for (npy_intp m = 0; m < M; ++m)
            {
                for (npy_intp n = 0; n < N; ++n)
                {
                    Dz[m*Szm + n*Szn] = 0.0;
                }
            }

            //iterate over the sparse array, making the most of an entry wherever we find it.
            //
            // Normal matrix matrix multiply:
            // for m
            //   for n
            //     for k
            //        z[m,n] += a[m,k] * b[k,n]
            // Here instead:
            // for k
            //   for m (sparse)
            //     for n
            //        z[m,n] += a[m,k] * b[k,n]

            for (npy_int32 k = 0; k < K; ++k)
            {
                const npy_double * __restrict__ bk = (double *)(%(b)s->data + %(b)s->strides[0] * k);

                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1) * Sptr]; ++m_idx)
                {
                    npy_int32 m = Dind[m_idx * Sind];
                    const double Amk = Dval[m_idx * Sval];

                    npy_double * __restrict__ zm = (npy_double *)(%(z)s->data + %(z)s->strides[0] * m);

                    if (m >= %(z)s->dimensions[0]) 
                    {PyErr_SetString(PyExc_NotImplementedError, "illegal row index in a"); %(fail)s;}

                    for(npy_int32 n = 0; n < N; ++n)
                    {
                        zm[n*Szn] += Amk * bk[n*Sbn];
                    }
                }
            }
        }
        """% dict(locals(), **sub)
sd_csc = StructuredDotCSC()

#TODO: register a specialization to replace StructuredDot -> StructuredDotCSC

class StructuredDotGrad(gof.Op):
    def make_node(self, a, b, g_ab):
        return gof.Apply(self, [a, b, g_ab], [a.type()])
    def perform(self, node, (a, b, g_ab), (out,)):
        g_a_data = a.data.copy()
        if a.format == 'csc':
            for j in xrange(len(a.indptr)-1):
                ind0 = a.indptr[j]
                ind1 = a.indptr[j+1]
                for i_idx in xrange(ind0, ind1):
                    i = a.indices[i_idx]
                    #v = a.data[i_idx]
                    #print (i, j, v)
                    g_a_data[i_idx] = numpy.dot(g_ab[i], b[j])
            out[0] = sparse.csc_matrix((g_a_data, a.indices.copy(), a.indptr.copy()),
                    a.shape, copy=False)
        elif a.format == 'csr':
            raise NotImplementedError()
        else:
            raise TypeError()
_structured_dot_grad = StructuredDotGrad()

class StructureDotGradCSC(gof.Op):
    def make_node(self, a_indices, a_indptr, b, g_ab):
        return gof.Apply(self, [a_indices, a_indptr, b, g_ab], [tensor.tensor(b.dtype, (False,))])
    def perform(self, node, (a_indices, a_indptr, b, g_ab), (out,)):
        g_a_data = numpy.zeros(a_indices.shape, dtype=g_ab.dtype)
        for j in xrange(len(a_indptr)-1):
            ind0 = a_indptr[j]
            ind1 = a_indptr[j+1]
            for i_idx in xrange(ind0, ind1):
                i = a_indices[i_idx]
                g_a_data[i_idx] = numpy.dot(g_ab[i], b[j])
        out[0] = g_a_data
    def c_code(self, node, name, (_indices, _indptr, _d, _g), (_zout, ), sub):
        return """
        if (%(_d)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); %(fail)s;}
        if (%(_g)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if( %(_d)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "d's dtype not NPY_DOUBLE"); %(fail)s;}

        if( %(_g)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "g's dtype not NPY_DOUBLE"); %(fail)s;}

        if( %(_d)s->dimensions[1] != %(_g)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "d and g have different numbers of columns"); %(fail)s;}

        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, %(_indices)s->dimensions, %(_g)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError, "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }
        
        {   //makes it compile even though labels jump over variable definitions.
            npy_intp nnz = %(_indices)s->dimensions[0];
            npy_intp N =  %(_indptr)s->dimensions[0]-1; //TODO: error checking with this

            npy_intp Sindices = %(_indices)s->strides[0]/%(_indices)s->descr->elsize;
            npy_intp Sindptr = %(_indptr)s->strides[0]/%(_indptr)s->descr->elsize;

            const npy_intp Sd1 = %(_d)s->strides[1]/%(_d)s->descr->elsize;
            const npy_intp Sg1 = %(_g)s->strides[1]/%(_g)s->descr->elsize;

            const npy_intp K = %(_d)s->dimensions[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            for (npy_int32 j = 0; j < N; ++j)
            {
                const npy_double * __restrict__ d_row = (double *)(%(_d)s->data + %(_d)s->strides[0] * j);
                if(j >= %(_d)s->dimensions[0]) {PyErr_SetString(PyExc_NotImplementedError, "G"); %(fail)s;}

                for (npy_int32 i_idx = indptr[j * Sindptr]; i_idx < indptr[(j+1) * Sindptr]; ++i_idx)
                {
                    npy_int32 i = indices[i_idx * Sindices];
                    const npy_double * __restrict__ g_row = (npy_double *)(%(_g)s->data + %(_g)s->strides[0] * i);
                    double ip = 0.0;

                    if (i >= %(_g)s->dimensions[0]) 
                    {PyErr_SetString(PyExc_NotImplementedError, "H"); %(fail)s;}

                    for(int k = 0; k < K; ++k)
                    {
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }
                    ((double * __restrict__)(%(_zout)s->data + i_idx * %(_zout)s->strides[0]))[0] = ip;
                }
            }
        }

        """% dict(locals(), **sub)
_sdgcsc = StructureDotGradCSC()

def structured_dot_grad(sparse_A, dense_B, ga):
    #TODO: 1. move this switch to be a specialization of structuredDotGrad
    #      2. implement StructuredDotGrad.grad()
    if 0:
        return _structured_dot_grad(sparse_A, dense_B, ga)
    else:
        if sparse_A.type.format == 'csc':
            g_A_data = _sdgcsc(csm_indices(sparse_A),\
                               csm_indptr(sparse_A), dense_B, ga)
            return CSC(g_A_data, csm_indices(sparse_A),\
                                     csm_indptr(sparse_A),\
                                     csm_shape(sparse_A))
        else:
            raise NotImplementedError()

