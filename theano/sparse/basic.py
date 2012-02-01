"""
Classes for handling sparse matrices.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""
from itertools import izip
import sys

import numpy, theano
import scipy.sparse

from theano import gof
from theano import tensor
from theano import compile
from theano import scalar
from theano import config
from theano.gof.python25 import all, any
from theano.tensor import blas

sparse_formats = ['csc', 'csr']

#TODO: move this decorator to the compile submodule
def register_specialize(lopt, *tags, **kwargs):
    compile.optdb['specialize'].register((kwargs and kwargs.pop('name')) or lopt.__name__, lopt, 'fast_run', *tags)


""" Types of sparse matrices to use for testing """
_mtypes = [scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
#_mtypes = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix, sparse.coo_matrix]
#* new class ``dia_matrix`` : the sparse DIAgonal format
#* new class ``bsr_matrix`` : the Block CSR format
_mtype_to_str = {scipy.sparse.csc_matrix: "csc", scipy.sparse.csr_matrix: "csr"}

def _is_sparse_variable(x):
    """
    @rtype: boolean
    @return: True iff x is a L{SparseVariable} (and not a L{tensor.TensorType})
    """
    if not isinstance(x.type, SparseType) and not isinstance(x.type, tensor.TensorType):
        raise NotImplementedError("this function should only be called on *variables* (of type sparse.SparseType or tensor.TensorType), not,", x)
    return isinstance(x.type, SparseType)
def _is_dense_variable(x):
    """
    @rtype: boolean
    @return: True unless x is a L{SparseVariable} (and not a L{tensor.TensorType})
    """
    if not isinstance(x.type, SparseType) and not isinstance(x.type, tensor.TensorType):
        raise NotImplementedError("this function should only be called on *variables* (of type sparse.SparseType or tensor.TensorType), not,", x)
    return isinstance(x.type, tensor.TensorType)

def _is_sparse(x):
    """
    @rtype: boolean
    @return: True iff x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, scipy.sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("this function should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, scipy.sparse.spmatrix)
def _is_dense(x):
    """
    @rtype: boolean
    @return: True unless x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray})
    """
    if not isinstance(x, scipy.sparse.spmatrix) and not isinstance(x, numpy.ndarray):
        raise NotImplementedError("this function should only be called on sparse.scipy.sparse.spmatrix or numpy.ndarray, not,", x)
    return isinstance(x, numpy.ndarray)

def _kmap_eq(a, b):
    if a is None and b is None:
        return True
    return numpy.all(a == b)

def _kmap_hash(a):
    if a is None: return 12345
    return hash(numpy.str(a))


# Wrapper type

def as_sparse_variable(x, name=None):
    """
    Wrapper around SparseVariable constructor.
    @param x: A sparse matrix. as_sparse_variable reads dtype and format
    properties out of this sparse matrix.
    @return: SparseVariable version of sp.

    @todo Verify that sp is sufficiently sparse, and raise a warning if it is not
    """
    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, gof.Variable):
        if not isinstance(x.type, SparseType):
            raise TypeError("Variable type field must be a SparseType.", x, x.type)
        return x
    try:
        return constant(x, name=name)
    except TypeError:
        raise TypeError("Cannot convert %s to SparseType" % x, type(x))


as_sparse = as_sparse_variable
def as_sparse_or_tensor_variable(x, name=None):
    """
    If we can't make a sparse variable, we try to make a tensor variable.
    """
    try:
        return as_sparse_variable(x,name)
    except (ValueError, TypeError):
        return theano.tensor.as_tensor_variable(x,name)


def constant(x, name=None):
    if not isinstance(x, scipy.sparse.spmatrix):
        raise TypeError("sparse.constant must be called on a scipy.sparse.spmatrix")
    try:
        return SparseConstant(SparseType(format = x.format,
                                     dtype = x.dtype), x.copy(),name=name)
    except TypeError:
        raise TypeError("Could not convert %s to SparseType" % x, type(x))

if 0:
    def value(x):
        if not isinstance(x, scipy.sparse.spmatrix):
            raise TypeError("sparse.value must be called on a scipy.sparse.spmatrix")
        try:
            return SparseValue(SparseType(format = x.format,
                                      dtype = x.dtype), x)
        except TypeError:
            raise TypeError("Could not convert %s to SparseType" % x, type(x))

def sp_ones_like(x):
    data, indices, indptr, shape = csm_properties(x) #TODO: don't restrict to CSM formats
    return CSM(format=x.format)(tensor.ones_like(data), indices, indptr, shape)


def sp_zeros_like(x):
    #TODO: don't restrict to CSM formats
    _, _, indptr, shape = csm_properties(x)
    return CSM(format=x.format)(numpy.array([], dtype=x.type.dtype),
                                numpy.array([]), tensor.zeros_like(indptr),
                                shape)


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

    #N.B. THIS IS COMMENTED OUT ON PURPOSE!!!
    #     Discussion with Fred & James (at least, and maybe others before)
    #     we decided that casting from a sparse to dense should be explicit
    #     because it's usually something you just want to be pretty careful about,
    #     and not to do by accident.
    #def _as_TensorVariable(self):
    #    return dense_from_sparse(self)

    shape = property(lambda self: tensor.shape(dense_from_sparse(self))) # don't worry!
    # ... the plan is that the ShapeFeature in tensor.opt will do shape propagation
    # ... and remove the dense_from_sparse from the graph.  This will *NOT* actually expand
    # ... your sparse matrix just to get the shape.
    ndim = property(lambda self: self.type.ndim)
    dtype = property(lambda self: self.type.dtype)

    # Note that the `size` attribute of sparse matrices behaves differently
    # from dense matrices: it is the number of elements stored in the matrix
    # rather than the total number of elements that may be stored. Note also
    # that stored zeros *do* count in the size.
    size = property(lambda self: csm_data(self).size)

    def zeros_like(model):
        return sp_zeros_like(model)

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,

        if len(args) == 2:
            scalar_arg_1 = (numpy.isscalar(args[0]) or
                            getattr(args[0], 'type', None) == tensor.iscalar)
            scalar_arg_2 = (numpy.isscalar(args[1]) or
                            getattr(args[1], 'type', None) == tensor.iscalar)
            if scalar_arg_1 and scalar_arg_2:
                ret = get_item_scalar(self, args)
            else:
                ret = get_item_2d(self, args)
        else:
            ret = get_item_2d(self, args)
        return ret


class SparseVariable(gof.Variable, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)
    def __str__(self):
        return '%s{%s,%s}'%(
                self.__class__.__name__,
                self.format,
                self.dtype)
    def __repr__(self):
        return str(self)

class SparseConstantSignature(tuple):
    def __eq__(self, other):
        (a, b), (x,y) = self, other
        return a == x \
                and (b.dtype == y.dtype)\
                and (type(b) == type(y))\
                and (b.shape == y.shape)\
                and (abs(b-y).sum() < 1e-6 * b.nnz)
    def __hash__(self):
        (a,b) = self
        return hash(type(self)) ^ hash(a) ^ hash(type(b))

class SparseConstant(gof.Constant, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

    def signature(self):
        assert self.data is not None
        return SparseConstantSignature((self.type, self.data))
    def __str__(self):
        return '%s{%s,%s,shape=%s,nnz=%s}'%(
                self.__class__.__name__,
                self.format,
                self.dtype,
                self.data.shape,
                self.data.nnz)
    def __repr__(self):
        return str(self)

class SparseValue(gof.Value, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

class SparseType(gof.Type):
    """
    @type dtype: numpy dtype string such as 'int64' or 'float64' (among others)
    @type format: string
    @ivar format: The sparse storage strategy.

    @note As far as I can tell, L{scipy.sparse} objects must be matrices, i.e. have dimension 2.
    """
    format_cls = {
            'csr' : scipy.sparse.csr_matrix,
            'csc' : scipy.sparse.csc_matrix
            }
    dtype_set = set(['int', 'int8', 'int16','int32', 'int64', 'float32', 'float64', 'complex64','complex128'])
    ndim = 2

    Variable = SparseVariable
    Constant = SparseConstant

    def __init__(self, format, dtype):
        """
        Fundamental way to create a sparse node.
        @param dtype:   Type of numbers in the matrix.
        @param format:  The sparse storage strategy.
        @return         An empty SparseVariable instance.
        """
        dtype = str(dtype)
        if dtype in self.dtype_set:
            self.dtype = dtype
        else:
            raise NotImplementedError('unsupported dtype "%s" not in list' % dtype, list(self.dtype_set))

        assert isinstance(format, basestring)
        if format in self.format_cls:
            self.format = format
        else:
            raise NotImplementedError('unsupported format "%s" not in list' % format, self.format_cls.keys())

    def filter(self, value, strict=False, allow_downcast=None):
        if isinstance(value, self.format_cls[self.format])\
                and value.dtype == self.dtype:
            return value
        if strict:
            raise TypeError("%s is not sparse, or not the right dtype (is %s, expected %s)"
                    % (value, value.dtype, self.dtype))
        #The input format could be converted here
        if allow_downcast:
            sp = self.format_cls[self.format](value, dtype=self.dtype)
        else:
            sp = self.format_cls[self.format](value)
            if str(sp.dtype) != self.dtype:
                raise NotImplementedError("Expected %s dtype but got %s"%(self.dtype,str(sp.dtype)))
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    @staticmethod
    def may_share_memory(a,b):
        # This is Fred suggestion for a quick and dirty way of checking
        # aliasing .. this can potentially be further refined (ticket #374)
        if _is_sparse(a) and _is_sparse(b):
            return a is b
        if _is_sparse(b) and isinstance(a, numpy.ndarray):
            a,b=b,a
        if _is_sparse(a) and isinstance(b, numpy.ndarray):
            if (numpy.may_share_memory(a.data,b) or
                numpy.may_share_memory(a.indices,b) or
                numpy.may_share_memory(a.indptr,b)):
                    #currently we can't share memory with a.shape as it is a tuple
                return True
        return False


    def make_variable(self, name = None):
        return SparseVariable(self, name = name)

    def __eq__(self, other):
        return type(self) == type(other) and other.dtype == self.dtype and other.format == self.format

    def __hash__(self):
        return hash(self.dtype) ^ hash(self.format)

    def __str__(self):
        return "Sparse[%s, %s]" % (str(self.dtype), str(self.format))

    def __repr__(self):
        return "Sparse[%s, %s]" % (str(self.dtype), str(self.format))

    def values_eq_approx(self, a, b, eps=1e-6):
        #WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        if not scipy.sparse.issparse(a) or not scipy.sparse.issparse(b):
            return False
        diff = abs(a-b)
        if diff.nnz == 0:
            return True
        return max(diff)<eps

    def values_eq(self, a, b):
        #WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        return scipy.sparse.issparse(a) \
                and scipy.sparse.issparse(b) \
                and abs(a-b).sum() == 0.0

    def is_valid_value(self, a):
        return scipy.sparse.issparse(a) and (a.format == self.format)

# Register CudaNdarrayType to the OutputGuard list of known types
# to have OutputGuard generate C code for this type.
theano.compile.mode.register_OutputGuard_c_code(SparseType)


# for more dtypes, call SparseType(format, dtype)
def matrix(format, name=None, dtype=None):
    if dtype is None:
        dtype = config.floatX
    type = SparseType(format=format, dtype=dtype)
    return type(name)
def csc_matrix(name=None, dtype=None):
    return matrix('csc', name, dtype)
def csr_matrix(name=None, dtype=None):
    return matrix('csr', name, dtype)
# for more dtypes, call SparseType(format, dtype)
csc_matrix = SparseType(format='csc', dtype=config.floatX)
csr_matrix = SparseType(format='csr', dtype=config.floatX)
csc_dmatrix = SparseType(format='csc', dtype='float64')
csr_dmatrix = SparseType(format='csr', dtype='float64')
csc_fmatrix = SparseType(format='csc', dtype='float32')
csr_fmatrix = SparseType(format='csr', dtype='float32')

# CONSTRUCTION
class CSMProperties(gof.Op):
    """Extract all of .data .indices and .indptr"""

    #we don't return a view of the shape, we create a new ndarray from the shape tuple.
    view_map = {0:[0],1:[0],2:[0]}

    kmap = None
    """ WRITEME """

    def __init__(self, kmap=None):
        self.kmap = kmap

    def __eq__(self, other):
        return type(self) == type(other) and _kmap_eq(self.kmap, other.kmap)

    def __ne__(self, other): return not (self == other)

    def __hash__(self):
        return 8234 ^ hash(type(self)) ^ _kmap_hash(self.kmap)

    def make_node(self, csm):
        csm = as_sparse_variable(csm)
        data = tensor.TensorType(dtype=csm.type.dtype, broadcastable = (False,)).make_variable()
        return gof.Apply(self, [csm],
                [data, tensor.ivector(), tensor.ivector(), tensor.ivector()])

    def perform(self, node, (csm,), out):
        if self.kmap is None:
            out[0][0] = csm.data
        else:
            out[0][0] = csm.data[self.kmap]
        if str(csm.data.dtype) == 'int32':
            out[0][0] = theano._asarray(out[0][0], dtype='int32')
        #backport
        #out[0][0] = csm.data if self.kmap is None else csm.data[self.kmap]
        out[1][0] = theano._asarray(csm.indices, dtype='int32')
        out[2][0] = theano._asarray(csm.indptr, dtype='int32')
        out[3][0] = theano._asarray(csm.shape, dtype='int32')

    # TODO FIX THIS
    def grad(self, (csm,), g):
        assert [gg is None for gg in g[1:]]
        data, indices, indptr, shape = csm_properties(csm)
        if csm.format == 'csc':
            return [CSM('csc')(g_data, indices, indptr, shape)]
        else:
            return [CSR('csm')(g_data, indices, indptr, shape)]
csm_properties = CSMProperties() #don't make this a function or it breaks some optimizations below
def csm_data(csm): return csm_properties(csm)[0]
def csm_indices(csm): return csm_properties(csm)[1]
def csm_indptr(csm): return csm_properties(csm)[2]
def csm_shape(csm): return csm_properties(csm)[3]

class CSM(gof.Op):
    """Construct a CSC or CSR matrix from the internal representation """
    view_map = {0:[0]} #should view the other inputs too, but viewing multiple inputs is not
    #currently supported by the destroyhandler

    format = None
    """WRITEME"""

    kmap = None
    """WRITEME"""

    _hashval = None
    """Pre-computed hash value, defined by __init__"""

    def __init__(self, format, kmap=None):
        if format not in ('csr', 'csc'):
            raise ValueError("format must be one of: 'csr', 'csc'", format)
        self.format = format

        # for efficiency, if remap does nothing, then do not apply it
        if kmap is not None and all(kmap==numpy.arange(numpy.size(kmap))):
            kmap = None

        self.kmap = kmap

        self._hashval = hash(type(self)) ^ hash(self.format) ^ _kmap_hash(self.kmap)

    def __eq__(self, other):
        return type(other) is CSM \
                and other.format == self.format and _kmap_eq(self.kmap, other.kmap)

    def __hash__(self):
        return self._hashval

    def make_node(self, data, indices, indptr, shape):
        """Build a SparseVariable from the internal parametrization

        :param data:
        :param indices:
        :param indptr:
        :type data: 1-d tensor
        :type indices: 1-d tensor of ints
        :type indptr: 1-d tensor of ints

        """
        data = tensor.as_tensor_variable(data)

        if not isinstance(indices, tensor.TensorVariable):
            indices = theano._asarray(indices, dtype='int32')
        if not isinstance(indptr, tensor.TensorVariable):
            indptr = theano._asarray(indptr, dtype='int32')
        if not isinstance(shape, tensor.TensorVariable):
            shape = theano._asarray(shape, dtype='int32')
        indices = tensor.as_tensor_variable(indices)
        indptr = tensor.as_tensor_variable(indptr)
        shape = tensor.as_tensor_variable(shape)

        if data.type.ndim != 1:
            raise TypeError('data argument must be a vector', data.type, data.type.ndim)
        if indices.type.ndim != 1 or indices.type.dtype != 'int32':
            raise TypeError('indices must be vector of integers', indices, indices.type)
        if indptr.type.ndim != 1 or indptr.type.dtype != 'int32':
            raise TypeError('indices must be vector of integers', indptr, indptr.type)
        if shape.type.ndim != 1 or shape.type.dtype != 'int32':
            raise TypeError('n_rows must be integer type', shape, shape.type)

        return gof.Apply(self,
                         [data, indices, indptr, shape],
                         [SparseType(dtype = data.type.dtype,
                                 format = self.format).make_variable()])

    def perform(self, node, (data, indices, indptr, shape), (out,)):
        """Build a csc_matrix"""
        # for efficiency, if remap does nothing, then do not apply it
        if self.kmap is not None:
            data = data[self.kmap]

        if len(shape) != 2:
            raise ValueError('Shape should be an array of length 2')
        if data.shape != indices.shape and numpy.size(data) != numpy.size(self.kmap):
            errmsg = 'Data (shape '+`data.shape`+' must have the same number of elements '+\
                     'as indices (shape'+`indices.shape`+') or elements as kmap ('+`numpy.size(self.kmap)`+')'
            raise ValueError(errmsg)
        if self.format == 'csc':
            out[0] = scipy.sparse.csc_matrix((data, indices.copy(), indptr.copy()),
                    numpy.asarray(shape),
                    copy = False #1000*len(data.flatten())
                    )
        else:
            assert self.format == 'csr'
            out[0] = scipy.sparse.csr_matrix((data, indices.copy(), indptr.copy()),
                    shape.copy(),
                    copy = False #1000*len(data.flatten())
                    )

    def grad(self, (data, indices, indptr, shape), (g_out,)):
        """Return a gradient on the data vector"""
        #unpack the data vector and wrap it as a 1d TensorType
        g_data = csm_grad(self.kmap)(data, csm_data(g_out),csm_indices(g_out))
        return [g_data, None, None, None]

CSC = CSM('csc')
CSR = CSM('csr')

class CSMGrad(gof.op.Op):
    def __init__(self, kmap=None):
        self.kmap = kmap
        if self.kmap is None:
            self.view_map = {0 : [1]}

    def __eq__(self, other):
        return type(self) == type(other) and _kmap_eq(self.kmap, other.kmap)

    def __ne__(self, other): return not (self == other)

    def __hash__(self):
        return 82345 ^ hash(type(self)) ^ _kmap_hash(self.kmap)

    def make_node(self, data, gout_data, gout_indices):
        g_data = gout_data.type()
        return gof.Apply(self, [data, gout_data, gout_indices], [g_data])

    def perform(self, node, (data, gout_data, gout_indices), (g_data,)):
        if self.kmap is None:
            g_data[0] = gout_data
        else:
            grad = numpy.zeros_like(data)
            grad[self.kmap] = gout_data
            g_data[0] = grad
csm_grad = CSMGrad

@gof.local_optimizer([csm_properties])
def skip_pack_csc01(node):
    """if we find csm_properties(CSM(*args)), then we can replace that with the *args
    directly"""
    if node.op == csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            # csm.owner.inputs could be broadcastable. In that case, we have
            # to adjust the broadcasting flag here.
            ret_var = [tensor.patternbroadcast(i, o.broadcastable)
                    for i, o in izip(csm.owner.inputs, node.outputs)]
            return ret_var

    return False
register_specialize(skip_pack_csc01)



#
# Conversion
#
class DenseFromSparse(gof.op.Op):
    """
    Convert a sparse matrix to an `ndarray`.
    """
    sparse_grad = True
    """WRITEME"""
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self,
                         [x],
                         [tensor.TensorType(dtype = x.type.dtype,
                                        broadcastable = (False, False)).make_variable()])
    def perform(self, node, (x, ), (out, )):
        if _is_dense(x):
            print >> sys.stderr, "WARNING: You just called DenseFromSparse on a dense matrix."
            out[0] = x
        else:
            out[0] = x.toarray()
        assert _is_dense(out[0])
    def grad(self, (x, ), (gz, )):
        if self.sparse_grad:
            return [sp_ones_like(x) * gz]
        else:
            return [SparseFromDense(x.type.format)(gz)]
    def infer_shape(self, node, (ishape,)):
        return [ishape]
dense_from_sparse = DenseFromSparse()

class SparseFromDense(gof.op.Op):
    def __init__(self, format):
        self.format = format
    def __eq__(self, other):
        return type(self) == type(other) and self.format == other.format
    def __ne__(self, other):
        return not (self == other)
    def __hash__(self):
        return 982374 ^ hash(self.format) ^ hash(DenseFromSparse)

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self,
                         [x],
                         [SparseType(dtype = x.type.dtype,
                                 format = self.format).make_variable()])
    def perform(self, node, (x, ), (out, )):
        out[0] = SparseType.format_cls[self.format](x)
    def grad(self, (x, ), (gz, )):
        return dense_from_sparse(gz),
    def infer_shape(self, node, (ishape,)):
        return [ishape]
csr_from_dense = SparseFromDense('csr')
csc_from_dense = SparseFromDense('csc')


# Indexing
class GetItem2d(gof.op.Op):
    """
    Implement a subtensor of sparse variable and that return a sparse matrix.

    If you want to take only one element of a sparse matrix see the
    class GetItemScalar that return a tensor scalar.

    :note: that subtensor selection always returns a matrix so
    indexing with [a:b, c:d] is forced.  If one index is a scalar,
    e.g. x[a:b, c] and x[a, b:c], generate an error. Use instead
    x[a:b, c:c+1] and x[a:a+1, b:c].

    The above indexing methods are not supported because the rval
    would be a sparse matrix rather than a sparse vector, which is a
    deviation from numpy indexing rule.  This decision is made largely
    for keeping the consistency between numpy and theano.  Subjected
    to modification when sparse vector is supported.
    """

    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

# Fred:Too complicated for now. If you need it, look at
#      the Subtensor.infer_shape.
#    def infer_shape(self, node, i0_shapes):
#        return i0_shapes

    def make_node(self, x, index):
        x = as_sparse_variable(x)
        assert len(index) in [1, 2]

        input_op = [x]
        generic_None = theano.gof.Constant(theano.gof.generic, None)

        for ind in index:
            if isinstance(ind, slice):
                # in case of slice is written in theano variable
                start = ind.start
                stop = ind.stop
                if ind.step is not None:
                    raise ValueError((
                        "Using a slice with non-default step when "
                        "indexing into a sparse matrix is not supported. "),
                        ind, ind.step)

                # If start or stop are None, make them a Generic constant
                # Else, they should be converted to Tensor Variables of
                # dimension 1 and int/uint dtype.
                if start is None:
                    start = generic_None
                else:
                    if not isinstance(start, gof.Variable):
                        start = tensor.as_tensor_variable(start)
                    if not (start.ndim == 0 and start.dtype in tensor.discrete_dtypes):
                        raise ValueError((
                            "Impossible to index into a sparse matrix with "
                            "slice where start=%s" % start),
                            start.ndim, start.dtype)

                if stop is None:
                    stop = generic_None
                else:
                    if not isinstance(stop, gof.Variable):
                        stop = tensor.as_tensor_variable(stop)
                    if not (stop.ndim == 0 and stop.dtype in tensor.discrete_dtypes):
                        raise ValueError((
                            "Impossible to index into a sparse matrix with "
                            "slice where stop=%s" % stop),
                            stop.ndim, stop.dtype)

            elif ((isinstance(ind, gof.Variable) and
                        getattr(ind, 'ndim', -1) == 0)
                        or numpy.isscalar(ind)):
                raise NotImplementedError(
                    'Theano has no sparse vector' +
                    'Use X[a:b,c:d], X[a:b,c:c+1] or X[a:b] instead.')
            else:
                raise ValueError((
                    'Advanced indexing is not implemented for sparse '
                    'matrices. Argument not supported: %s' % ind))
            input_op += [start, stop]
        if len(index) == 1:
            input_op += [generic_None, generic_None]

        return gof.Apply(self, input_op, [x.type()])

    def perform(self, node, (x, start1, stop1, start2, stop2), (out, )):
        assert _is_sparse(x)
        out[0] = x[start1:stop1, start2:stop2]

    def __str__(self):
        return self.__class__.__name__

get_item_2d = GetItem2d()


class GetItemScalar(gof.op.Op):
    """
    Implement a subtensor of a sparse variable that take two scalar as
    index and return a scalar

    :see: GetItem2d to return more then one element.
    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def infer_shape(self, node, i0_shapes):
        return [()]

    def make_node(self, x, index):
        x = as_sparse_variable(x)
        assert len(index) == 2

        input_op = [x]

        for ind in index:

            if isinstance(ind, slice):
                raise Exception("GetItemScalar called with a slice as index!")

            #in case of indexing using int instead of theano variable
            elif isinstance(ind, int):
                ind = theano.tensor.constant(ind)
                input_op += [ind]

            # in case of indexing using theano variable
            elif ind.ndim == 0:
                input_op += [ind]
            else:
                raise NotImplemented()

        return gof.Apply(self, input_op, [tensor.scalar(dtype=x.dtype)])

    def perform(self, node, (x, ind1, ind2), (out, )):
        assert _is_sparse(x)
        out[0] = theano._asarray(x[ind1, ind2], x.dtype)

    def __str__(self):
        return self.__class__.__name__

get_item_scalar = GetItemScalar()

# Linear Algebra


class Transpose(gof.op.Op):
    format_map = {'csr' : 'csc',
                  'csc' : 'csr'}
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self,
                         [x],
                         [SparseType(dtype = x.type.dtype,
                                 format = self.format_map[x.type.format]).make_variable()])
    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = x.transpose()

    def grad(self, (x,), (gz,)):
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return transpose(gz),
transpose = Transpose()

class Neg(gof.op.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self, [x], [x.type()])
    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = -x
    def grad(self, (x,), (gz,)):
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return -gz,
neg = Neg()

class AddSS(gof.op.Op):
    '''Add two sparse matrices '''
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x, y):
        x, y = map(as_sparse_variable, [x, y])
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        if x.type.format != y.type.format:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype = x.type.dtype,
                                 format = x.type.format).make_variable()])
    def perform(self, node, (x, y), (out, )):
        assert _is_sparse(x) and _is_sparse(y)
        assert x.shape == y.shape
        out[0] = x + y
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_variable(x) and _is_sparse_variable(y)
        assert _is_sparse_variable(gz)
        return gz, gz
add_s_s = AddSS()
class AddSD(gof.op.Op):
    ''' Add a sparse and a dense matrix '''
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x, y):
        x, y = as_sparse_variable(x), tensor.as_tensor_variable(y)
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        return gof.Apply(self,
                         [x, y],
                         [tensor.TensorType(dtype = y.type.dtype,
                                        broadcastable = y.type.broadcastable).make_variable()])
    def perform(self, node, (x, y), (out, )):
        assert _is_sparse(x) and _is_dense(y)
        # The asarray is needed as in some case, this return a
        # numpy.matrixlib.defmatrix.matrix object and not an ndarray.
        out[0] = theano._asarray(x + y, dtype=node.outputs[0].type.dtype)
    def grad(self, (x, y), (gz,)):
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_dense_variable(gz)
        return sp_ones_like(x) * gz, gz
add_s_d = AddSD()
def add(x,y):
    """
    Add two matrices, at least one of which is sparse.
    """
    if hasattr(x, 'getnnz'): x = as_sparse_variable(x)
    if hasattr(y, 'getnnz'): y = as_sparse_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable: return add_s_s(x,y)
    elif x_is_sparse_variable and not y_is_sparse_variable: return add_s_d(x,y)
    elif y_is_sparse_variable and not x_is_sparse_variable: return add_s_d(y,x)
    else: raise NotImplementedError()
def sub(x,y):
    return x + (-y)



class MulSS(gof.op.Op):
    ''' Elementwise multiply a sparse and a sparse '''
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x, y):
        x, y = as_sparse_variable(x), as_sparse_variable(y)
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
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x, y):
        x, y = as_sparse_variable(x), tensor.as_tensor_variable(y)

        #upcast the tensor. Is the cast of sparse done implemented?
        dtype = scalar.upcast(x.type.dtype, y.type.dtype)
        if y.type.dtype != dtype:
            y = tensor.cast(y,dtype)

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
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_sparse_variable(gz)
        return y * gz, x * gz
mul_s_d = MulSD()
def mul(x,y):
    """
    Multiply (elementwise) two matrices, at least one of which is sparse.
    """
    x = as_sparse_or_tensor_variable(x)
    y = as_sparse_or_tensor_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable: return mul_s_s(x,y)
    elif x_is_sparse_variable and not y_is_sparse_variable: return mul_s_d(x,y)
    elif y_is_sparse_variable and not x_is_sparse_variable: return mul_s_d(y,x)
    else: raise NotImplementedError()

###############
#
# StructuredDot
#
class StructuredDot(gof.Op):
    """Structured Dot is like dot, except that only the gradient wrt non-zero elements of the
    sparse matrix A are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a TensorType instance.
    """
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a, b):
        if not _is_sparse_variable(a):
            raise TypeError('First argument must be of type SparseVariable or SparseConstant');
        dtype_out = scalar.upcast(a.type.dtype, b.type.dtype)
        if b.type.ndim != 2:
            raise NotImplementedError('non-matrix b')

        if _is_sparse_variable(b):
            return gof.Apply(self, [a,b], [SparseType(a.type.format,dtype_out)()])
        else:
            return gof.Apply(self, [a,b], [tensor.tensor(dtype_out, (False, b.type.broadcastable[1]))])

    def perform(self, node, (a,b), (out,)):
        if a.shape[1] != b.shape[0]:
            raise ValueError('shape mismatch in StructuredDot.perform', (a.shape, b.shape))

        #variable = a.dot(b)  # deprecated
        variable = a * b
        if isinstance(node.outputs[0].type,SparseType):
            assert _is_sparse(variable)
            out[0] = variable
            return

        assert _is_dense(variable) # scipy 0.7 automatically converts to dense

        # dot of an NxM sparse matrix, with a Mx1 dense matrix, returns vector not matrix
        if variable.ndim == 1:
            variable = numpy.expand_dims(variable,1)
        elif variable.ndim != 2:
            raise Exception('Output of structured dot should be a matrix (ndim=2)')

        assert variable.ndim == 2

        if variable.shape != (a.shape[0], b.shape[1]):
            if b.shape[0] == 1:
                raise Exception("a.shape=%s, b.shape=%s, variable.shape=%s ??? This is probably because scipy.csc_matrix dot has a bug with singleton dimensions (i.e. b.shape[0]=1), for scipy 0.6. Use scipy 0.7. NB you have scipy version %s" % (a.shape, b.shape, variable.shape, scipy.__version__))
            else:
                raise Exception("a.shape=%s, b.shape=%s, variable.shape=%s ??? I have no idea why")

        #The cast is needed as otherwise we hit the bug mentioned into
        #theano._asarray function documentation.
        out[0] = theano._asarray(variable, str(variable.dtype))

    def grad(self, (a,b), (g_out,)):
        # a is sparse, b is dense, g_out is dense
        # ga = g_out x b.T
        # gb = a.T x g_out
        return [structured_dot_grad(a, b, g_out), structured_dot(a.T,g_out)]

_structured_dot = StructuredDot()

def structured_dot(x, y):
    """
    @todo: Maybe the triple-transposition formulation (when x is dense)
    is slow. See if there is a direct way to do this.
    (JB 20090528: Transposing tensors and sparse matrices is constant-time, inplace, and fast.)
    """
    if hasattr(x, 'getnnz'): x = as_sparse_variable(x)
    if hasattr(y, 'getnnz'): y = as_sparse_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)
    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError('structured_dot requires at least one sparse argument')

    if x_is_sparse_variable:
        return _structured_dot(x, y)
    else:
        assert y_is_sparse_variable
        return _structured_dot(y.T, x.T).T

class StructuredDotCSC(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a_val, a_ind, a_ptr, a_nrows, b):
        dtype_out = scalar.upcast(a_val.type.dtype, b.type.dtype)
        r = gof.Apply(self, [a_val, a_ind, a_ptr, a_nrows, b],
                [tensor.tensor(dtype_out, (False, b.type.broadcastable[1]))])
        return r

    def perform(self, node, (a_val, a_ind, a_ptr, a_nrows, b), (out,)):
        a = scipy.sparse.csc_matrix((a_val, a_ind, a_ptr),
                (a_nrows, b.shape[0]),
                copy = False)
        #out[0] = a.dot(b)
        out[0] = theano._asarray(a * b, dtype=node.outputs[0].type.dtype)
        assert _is_dense(out[0]) # scipy 0.7 automatically converts to dense

    def c_code(self, node, name, (a_val, a_ind, a_ptr, a_nrows, b), (z,), sub):
        """
        C-implementation of the dot product of the sparse matrix A and matrix B.
        @param a_val: non-zero values of the sparse matrix
        @param a_ind: column indices of the non-null values (.indices of a scipy.csc_matrix)
        @param a_ptr: a_ptr indicates col indices for col. i are in the range a_ptr[i]:a_ptr[i+1]
        @param n_rows: number of rows of sparse matrix
        @param b: dense matrix to perform dot product with, as in dot(a,b)
        @param z: return value
        @param sub: TODO, not too sure, something to do with weave probably
        """

        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a_val')
        if node.inputs[4].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        typenum_z = node.outputs[0].type.dtype_specs()[-1] # retrieve dtype number
        typenum_a_val = node.inputs[0].type.dtype_specs()[-1] # retrieve dtype number
        typenum_b = node.inputs[4].type.dtype_specs()[-1] # retrieve dtype number

        rval = """

        if (%(a_val)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (%(a_ind)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (%(a_ptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (%(a_nrows)s->nd != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); %(fail)s;}
        if (%(b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (%(a_val)s->descr->type_num != %(typenum_a_val)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for a_val"); %(fail)s;}

        if (%(b)s->descr->type_num != %(typenum_b)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for b"); %(fail)s;}

        if (%(a_ind)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (%(a_ptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (%(a_nrows)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_nrows dtype not INT32"); %(fail)s;}

        if (%(a_val)s->dimensions[0] != %(a_ind)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if (%(a_ptr)s->dimensions[0] != %(b)s->dimensions[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "a's number of columns doesn't match b's rows"); %(fail)s;}

        if ((!%(z)s)
            || (%(z)s->dimensions[0] != ((npy_int32 *)%(a_nrows)s->data)[0])
            || (%(z)s->dimensions[1] != %(b)s->dimensions[1])
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0,0};
            dims[0] = ((npy_int32 *)%(a_nrows)s->data)[0];
            dims[1] = %(b)s->dimensions[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(b)s->dimensions[0];

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = %(z)s->strides[0] / %(z)s->descr->elsize;
            npy_intp Szn = %(z)s->strides[1] / %(z)s->descr->elsize;
            //npy_intp Sbm = %(b)s->strides[0] / %(b)s->descr->elsize;
            npy_intp Sbn = %(b)s->strides[1] / %(b)s->descr->elsize;
            npy_intp Sval = %(a_val)s->strides[0] / %(a_val)s->descr->elsize;
            npy_intp Sind = %(a_ind)s->strides[0] / %(a_ind)s->descr->elsize;
            npy_intp Sptr = %(a_ptr)s->strides[0] / %(a_ptr)s->descr->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s*     __restrict__ Dz   = (dtype_%(z)s*)%(z)s->data;
            const dtype_%(a_val)s* __restrict__ Dval = (dtype_%(a_val)s*)%(a_val)s->data;
            const npy_int32 * __restrict__ Dind = (npy_int32*)%(a_ind)s->data;
            const npy_int32 * __restrict__ Dptr = (npy_int32*)%(a_ptr)s->data;

            //npy_intp nnz = %(a_ind)s->dimensions[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            //
            // Normal matrix matrix multiply: A MxK, B KxN =>  Z = AB
            // for m
            //   for n
            //     for k
            //        z[m,n] += a[m,k] * b[k,n]
            // Here instead: Z =
            // for k
            //   for m (sparse)
            //     for n
            //        z[m,n] += a[m,k] * b[k,n]

            // loop over inner dimension
            for (npy_int32 k = 0; k < K; ++k)
            {
                // get pointer to k-th row of dense matrix
                const dtype_%(b)s* __restrict__ bk = (dtype_%(b)s*)(%(b)s->data + %(b)s->strides[0] * k);

                // loop over sparse column indices through index pointer array
                // (amounts to looping over rows M of sparse matrix)

                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1) * Sptr]; ++m_idx)
                {
                    npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K
                    const dtype_%(a_val)s Amk = Dval[m_idx * Sval]; // actual value at that location

                    // pointer to m-th row of the output matrix Z
                    dtype_%(z)s* __restrict__ zm = (dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * m);

                    //RESOLVE: a.shape[0] equals z.shape[0], why is this not an equality constraint?
                    if (m >= %(z)s->dimensions[0])
                    {PyErr_SetString(PyExc_NotImplementedError, "illegal row index in a"); %(fail)s;}

                    // loop over final dimension (cols of dense matrix) and perform dot product
                    if ((Szn == 1) && (Sbn == 1)) {
                        for(npy_int32 n = 0; n < N; ++n)
                        {
                            zm[n] += Amk * bk[n];
                        }
                    }
                    else
                    {
                        for(npy_int32 n = 0; n < N; ++n)
                        {
                            zm[n*Szn] += Amk * bk[n*Sbn];
                        }
                    }
                }
            }
        }
        """% dict(locals(), **sub)

        return rval

    def c_code_cache_version(self):
        return (2,)
sd_csc = StructuredDotCSC()

class StructuredDotCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a_val, a_ind, a_ptr, b):
        self.dtype_out = scalar.upcast(a_val.type.dtype, b.type.dtype)
        r = gof.Apply(self, [a_val, a_ind, a_ptr, b],
                [tensor.tensor(self.dtype_out, (False, b.type.broadcastable[1]))])
        return r

    def perform(self, node, (a_val, a_ind, a_ptr, b), (out,)):
        a = scipy.sparse.csr_matrix((a_val, a_ind, a_ptr),
                (len(a_ptr)-1, b.shape[0]),
                copy = True) #use view_map before setting this to False
        #out[0] = a.dot(b)
        out[0] = a * b
        assert _is_dense(out[0]) # scipy 0.7 automatically converts to dense, but not .6 sometimes

    def c_code(self, node, name, (a_val, a_ind, a_ptr, b), (z,), sub):
        """
        C-implementation of the dot product of the sparse matrix A and matrix B.
        @param a_val: non-zero values of the sparse matrix
        @param a_ind: column indices of the non-null values (.indices of a scipy.csc_matrix)
        @param a_ptr: a_ptr indicates col indices for col. i are in the range a_ptr[i]:a_ptr[i+1]
        @param n_cols: number of columns of sparse matrix
        @param b: dense matrix to perform dot product with, as in dot(a,b)
        @param z: return value
        @param sub: TODO, not too sure, something to do with weave probably
        """
        typenum_z = tensor.TensorType(self.dtype_out, []).dtype_specs()[-1] # retrieve dtype number
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a_val')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(a_val)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (%(a_ind)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (%(a_ptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (%(b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (%(a_ind)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (%(a_ptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (%(a_val)s->dimensions[0] != %(a_ind)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if ((!%(z)s)
            || (%(z)s->dimensions[0] != %(a_ptr)s->dimensions[0]-1) //a's rows
            || (%(z)s->dimensions[1] != %(b)s->dimensions[1])       //b's columns
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0,0};
            dims[0] = %(a_ptr)s->dimensions[0]-1;
            dims[1] = %(b)s->dimensions[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(b)s->dimensions[0];

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = %(z)s->strides[0] / %(z)s->descr->elsize;
            npy_intp Szn = %(z)s->strides[1] / %(z)s->descr->elsize;
            npy_intp Sbm = %(b)s->strides[0] / %(b)s->descr->elsize;
            npy_intp Sbn = %(b)s->strides[1] / %(b)s->descr->elsize;
            npy_intp Sval = %(a_val)s->strides[0] / %(a_val)s->descr->elsize;
            npy_intp Sind = %(a_ind)s->strides[0] / %(a_ind)s->descr->elsize;
            npy_intp Sptr = %(a_ptr)s->strides[0] / %(a_ptr)s->descr->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz = (dtype_%(z)s*)%(z)s->data;
            const dtype_%(a_val)s* __restrict__ Dval = (dtype_%(a_val)s*)%(a_val)s->data;
            const npy_int32 * __restrict__ Dind = (npy_int32*)%(a_ind)s->data;
            const npy_int32 * __restrict__ Dptr = (npy_int32*)%(a_ptr)s->data;

            //npy_intp nnz = %(a_ind)s->dimensions[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            // Normal matrix matrix multiply:
            // for m
            //   for n
            //     for k
            //        z[m,n] += a[m,k] * b[k,n]
            // Here instead:
            // for m
            //   for k (sparse)
            //     for n
            //        z[m,n] += a[m,k] * b[k,n]

            // loop over inner dimension
            for (npy_int64 m = 0; m < M; ++m)
            {
                // pointer to m-th row of the output matrix Z
                dtype_%(z)s* __restrict__ zm = (dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * m);

                // loop over sparse rows indices through index pointer array
                // (amounts to looping over cols k of sparse matrix)
                for (npy_int32 k_idx = Dptr[m * Sptr]; k_idx < Dptr[(m+1) * Sptr]; ++k_idx)
                {
                    npy_int32 k = Dind[k_idx * Sind]; // col index of non-null value for row m
                    const dtype_%(a_val)s Amk = Dval[k_idx * Sval]; // actual value at that location

                    // get pointer to k-th row of dense matrix
                    const dtype_%(b)s* __restrict__ bk = (dtype_%(b)s*)(%(b)s->data + %(b)s->strides[0] * k);

                    // loop over final dimension (cols of dense matrix) and perform dot product
                    for(npy_int32 n = 0; n < N; ++n)
                    {
                        zm[n*Szn] += Amk * bk[n*Sbn];
                    }
                }
            }
        }

        """% dict(locals(), **sub)

    def c_code_cache_version(self):
        return (1,)
sd_csr = StructuredDotCSR()

# register a specialization to replace StructuredDot -> StructuredDotCSx
@gof.local_optimizer([_structured_dot])
def local_structured_dot(node):
    if node.op == _structured_dot:
        a, b = node.inputs
        if a.type.format == 'csc':
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            a_nsparse = a_shape[0]
            return [sd_csc(a_val, a_ind, a_ptr, a_nsparse, b)]
        if a.type.format == 'csr':
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            return [sd_csr(a_val, a_ind, a_ptr, b)]
    return False

# Commented out because
# a) it is only slightly faster than scipy these days, and sometimes a little slower, and
# b) the resulting graphs make it very difficult for an op to do size checking on the matrices
#    involved.  dimension mismatches are hard to detect sensibly.
#register_specialize(local_structured_dot)

def structured_dot_grad(sparse_A, dense_B, ga):
    if sparse_A.type.format in ('csc','csr'):

        if sparse_A.type.format == 'csc':
            sdgcsx = sdg_csc
        else:
            sdgcsx = sdg_csr
        #backport
        #sdgcsx = sdg_csc if sparse_A.type.format == 'csc' else sdg_csr

        if sparse_A.type.format == 'csc':
            CSx = CSC
        else:
            CSx = CSR
        #backport
        #CSx = CSC if sparse_A.type.format == 'csc' else CSR

        g_A_data = sdgcsx(csm_indices(sparse_A),\
                          csm_indptr(sparse_A), dense_B, ga)
        return CSx(g_A_data, csm_indices(sparse_A),\
                                 csm_indptr(sparse_A),\
                                 csm_shape(sparse_A))
    else:
        raise NotImplementedError()


class StructuredDotGradCSC(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a_indices, a_indptr, b, g_ab):
        return gof.Apply(self, [a_indices, a_indptr, b, g_ab],
                               [tensor.tensor(g_ab.dtype, (False,))])
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

        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for g_ab')

        return """
        if (%(_d)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); %(fail)s;}
        if (%(_g)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

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

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {
                // extract j-th row of dense matrix
                const dtype_%(_d)s* __restrict__ d_row = (dtype_%(_d)s*)(%(_d)s->data + %(_d)s->strides[0] * j);
                if(j >= %(_d)s->dimensions[0]) {PyErr_SetString(PyExc_NotImplementedError, "G"); %(fail)s;}

                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j * Sindptr]; i_idx < indptr[(j+1) * Sindptr]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx * Sindices];

                    // extract corresponding row in gradient
                    const dtype_%(_g)s* __restrict__ g_row = (dtype_%(_g)s*)(%(_g)s->data + %(_g)s->strides[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= %(_g)s->dimensions[0])
                    {PyErr_SetString(PyExc_NotImplementedError, "H"); %(fail)s;}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }

                    // write resulting gradient to sparse output
                    ((dtype_%(_zout)s* __restrict__)(%(_zout)s->data + i_idx * %(_zout)s->strides[0]))[0] = ip;
                }
            }
        }

        """% dict(locals(), **sub)
sdg_csc = StructuredDotGradCSC()


class StructuredDotGradCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_indices, a_indptr, b, g_ab):
        return gof.Apply(self, [a_indices, a_indptr, b, g_ab], [tensor.tensor(b.dtype, (False,))])

    def perform(self, node, (a_indices, a_indptr, b, g_ab), (out,)):
        g_a_data = numpy.zeros(a_indices.shape, dtype=g_ab.dtype)
        for i in xrange(len(a_indptr)-1): # loop over rows
            ind0 = a_indptr[i]
            ind1 = a_indptr[i+1]
            for j_idx in xrange(ind0, ind1): # loop over values in that row (columns)
                j = a_indices[j_idx]
                # grad is dot product of i-th row of gradient with j-th row of b
                g_a_data[j_idx] = numpy.dot(g_ab[i], b[j])
        out[0] = g_a_data

    def c_code(self, node, name, (_indices, _indptr, _d, _g), (_zout, ), sub):

        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for g_ab')

        return """
        if (%(_d)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); %(fail)s;}
        if (%(_g)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

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
            // extract number of rows
            npy_intp N =  %(_indptr)s->dimensions[0]-1; //TODO: error checking with this

            npy_intp Sindices = %(_indices)s->strides[0]/%(_indices)s->descr->elsize;
            npy_intp Sindptr = %(_indptr)s->strides[0]/%(_indptr)s->descr->elsize;

            const npy_intp Sd1 = %(_d)s->strides[1]/%(_d)s->descr->elsize;
            const npy_intp Sg1 = %(_g)s->strides[1]/%(_g)s->descr->elsize;

            const npy_intp K = %(_d)s->dimensions[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            // loop over columns of sparse matrix
            for (npy_int32 i = 0; i < N; ++i)
            {
                // for each non-null value in the sparse row
                for (npy_int32 j_idx = indptr[i * Sindptr]; j_idx < indptr[(i+1) * Sindptr]; ++j_idx)
                {
                    // extract column index of non-null value
                    npy_int32 j = indices[j_idx * Sindices];

                    // extract j-th row of dense matrix
                    const dtype_%(_d)s* __restrict__ d_row = (dtype_%(_d)s*)(%(_d)s->data + %(_d)s->strides[0] * j);
                    if(j >= %(_d)s->dimensions[0]) {PyErr_SetString(PyExc_NotImplementedError, "G"); %(fail)s;}

                    // extract corresponding row in gradient
                    const dtype_%(_g)s* __restrict__ g_row = (dtype_%(_g)s*)(%(_g)s->data + %(_g)s->strides[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= %(_g)s->dimensions[0])
                    {PyErr_SetString(PyExc_NotImplementedError, "H"); %(fail)s;}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }

                    // write resulting gradient to sparse output
                    ((dtype_%(_zout)s* __restrict__)(%(_zout)s->data + j_idx * %(_zout)s->strides[0]))[0] = ip;
                }
            }
        }

        """% dict(locals(), **sub)


sdg_csr = StructuredDotGradCSR()


class Dot(gof.op.Op):
    """
    Operation for efficiently calculating the dot product when
    one or all operands is sparse. Supported format are CSC and CSR.
    The output of the operation is dense.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __ne__(self, other):
        return not (self == other)

    def infer_shape(self, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs
        if x.ndim == 2 and y.ndim == 2:
            return [(xshp[0], yshp[1])]
        if x.ndim == 1 and y.ndim == 2:
            return [(yshp[1],)]
        if x.ndim == 2 and y.ndim == 1:
            return [(xshp[0],)]
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        raise NotImplementedError()

    def make_node(self, x, y):
        dtype_out = scalar.upcast(x.type.dtype, y.type.dtype)

        if not _is_sparse_variable(x) and not _is_sparse_variable(y):
            raise TypeError(x)

        return gof.Apply(self, [x, y], [tensor.tensor(dtype=dtype_out,
                         broadcastable=(False, False))])

    def perform(self, node, inputs, out):
        x, y = inputs
        out = out[0]
        x_is_sparse = _is_sparse(x)
        y_is_sparse = _is_sparse(y)

        if not x_is_sparse and not y_is_sparse:
            raise TypeError(x)

        rval = x * y

        if x_is_sparse and y_is_sparse:
            rval = rval.toarray()

        out[0] = rval

    def grad(self, (x, y), (gz,)):
        assert _is_sparse_variable(x) or _is_sparse_variable(y)
        rval = []

        if _is_dense_variable(y):
            rval.append(tensor.dot(gz, y.T))
        else:
            rval.append(dot(gz, y.T))
        if _is_dense_variable(x):
            rval.append(tensor.dot(x.T, gz))
        else:
            rval.append(dot(x.T, gz))

        return rval
_dot = Dot()


def dot(x, y):
    """
    Operation for efficiently calculating the dot product when
    one or all operands is sparse. Supported format are CSC and CSR.
    The output of the operation is dense.
    """
    if hasattr(x, 'getnnz'):
        x = as_sparse_variable(x)
    if hasattr(y, 'getnnz'):
        y = as_sparse_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()

    return _dot(x, y)


class Usmm(gof.op.Op):
    """
    Performs the expression is alpha * x y + z

    x or y are sparse matrix(the other can be sparse or dense)
    z is a dense matrix
    alpha is a scalar

    :note: We don't implement the infer_shape as it is inserted by optimization only
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return 'Usmm{no_inplace}'

    def make_node(self, alpha, x, y, z):
        if not _is_sparse_variable(x) and not _is_sparse_variable(y):
            # If x and y are tensor, we don't want to use this class
            # We should use Dot22 and Gemm in that case.
            raise TypeError(x)

        dtype_out = scalar.upcast(alpha.type.dtype, x.type.dtype,
                                  y.type.dtype, z.type.dtype)
        alpha = tensor.as_tensor_variable(alpha)
        z = tensor.as_tensor_variable(z)

        assert z.ndim == 2
        assert alpha.type.broadcastable == (True,) * alpha.ndim
        if not _is_sparse_variable(x):
            x = tensor.as_tensor_variable(x)
            assert x.ndim == 2
        if not _is_sparse_variable(y):
            y = tensor.as_tensor_variable(y)
            assert y.ndim == 2

        return gof.Apply(self, [alpha, x, y, z],
                         [tensor.tensor(dtype=dtype_out,
                                        broadcastable=(False, False))])

    def perform(self, node, (alpha, x, y, z), (out, )):
        x_is_sparse = _is_sparse(x)
        y_is_sparse = _is_sparse(y)

        if not x_is_sparse and not y_is_sparse:
            raise TypeError(x)

        rval = x * y
        if isinstance(rval, scipy.sparse.spmatrix):
            rval = rval.toarray()
        if rval.dtype == alpha.dtype:
            rval *= alpha  # Faster because operation is inplace
        else:
            rval = rval * alpha
        if rval.dtype == z.dtype:
            rval += z   # Faster because operation is inplace
        else:
            rval = rval + z

        out[0] = rval
usmm = Usmm()


class UsmmCscDense(gof.Op):
    """
    Performs the expression is alpha * x y + z
    This is an optimized operation for the case when x is in CSC format.

    x are sparse matrix
    y, z is a dense matrix
    alpha is a scalar

    :note: We don't implement the infer_shape as it is inserted by optimization only
    """
    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [6]}

    def __str__(self):
        if self.inplace:
            return 'UsmmCscDense{inplace}'
        else:
            return 'UsmmCscDense{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)) and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ self.inplace

    def make_node(self, alpha, x_val, x_ind, x_ptr, x_nrows, y, z):
        alpha = tensor.as_tensor_variable(alpha)
        x_val = tensor.as_tensor_variable(x_val)
        x_ind = tensor.as_tensor_variable(x_ind)
        x_ptr = tensor.as_tensor_variable(x_ptr)
        x_nrows = tensor.as_tensor_variable(x_nrows)
        y = tensor.as_tensor_variable(y)
        z = tensor.as_tensor_variable(z)
        assert x_ind.dtype == 'int32'
        assert x_ptr.dtype == 'int32'
        assert x_nrows.dtype == 'int32'
        assert alpha.ndim == 2 and alpha.type.broadcastable == (True, True)
        assert x_val.ndim == 1
        assert y.ndim == 2
        assert z.ndim == 2

        dtype_out = scalar.upcast(alpha.type.dtype, x_val.type.dtype,
            y.type.dtype, z.type.dtype)
        
        if dtype_out not in ('float32', 'float64'):
            raise NotImplementedError('only float types are supported in operands')
        
        if self.inplace:
            assert z.type.dtype == dtype_out

        # axpy work only with the same dtype, so we should upcast the input
        if dtype_out != alpha.type.dtype:
            alpha = tensor.cast(alpha, dtype_out)
        if dtype_out != x_val.type.dtype:
            x_val = tensor.cast(x_val, dtype_out)
        if dtype_out != y.type.dtype:
            y = tensor.cast(y, dtype_out)
        if dtype_out != z.type.dtype:
            z = tensor.cast(z, dtype_out)
        
        r = gof.Apply(self, [alpha, x_val, x_ind, x_ptr, x_nrows, y, z],
                [tensor.tensor(dtype_out, (False, y.type.broadcastable[1]))])
        return r

    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, inputs, outputs, sub):
        alpha, x_val, x_ind, x_ptr, x_nrows, y, z = inputs
        zn = outputs[0]
        if node.inputs[1].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for '
                                      'x_val')
        if node.inputs[5].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for y')
        if node.inputs[6].type.dtype != node.outputs[0].type.dtype:
            raise NotImplementedError('z and output must have same type')

        if node.inputs[1].type.dtype == "float32":
            conv_type = "float"
            axpy = "saxpy_"
        else:
            conv_type = "double"
            axpy = "daxpy_"
        # retrieve dtype numbers
        typenum_alpha = node.inputs[0].type.dtype_specs()[-1]
        typenum_x_val = node.inputs[1].type.dtype_specs()[-1]
        typenum_y = node.inputs[5].type.dtype_specs()[-1]
        typenum_z = node.inputs[6].type.dtype_specs()[-1]
        typenum_zn = node.outputs[0].type.dtype_specs()[-1]

        inplace = int(self.inplace)

        rval = """
        if (%(x_val)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_val) != 1"); %(fail)s;}
        if (%(x_ind)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_ind) != 1"); %(fail)s;}
        if (%(x_ptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_ptr) != 1"); %(fail)s;}
        if (%(x_nrows)s->nd != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_nrows) != 0"); %(fail)s;}
        if (%(y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}

        if (%(x_val)s->descr->type_num != %(typenum_x_val)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x_val"); %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}

        if (%(z)s->descr->type_num != %(typenum_z)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for z"); %(fail)s;}

        if (%(alpha)s->descr->type_num != %(typenum_alpha)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for alpha"); %(fail)s;}

        if (%(x_ind)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "x_ind dtype not INT32"); %(fail)s;}

        if (%(x_ptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "x_ptr dtype not INT32"); %(fail)s;}

        if (%(x_nrows)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "x_nrows dtype not INT32"); %(fail)s;}

        if (%(x_val)s->dimensions[0] != %(x_ind)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "x_val and x_ind have different lengths"); %(fail)s;}

        if (%(x_ptr)s->dimensions[0] != %(y)s->dimensions[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); %(fail)s;}

        if (%(z)s->dimensions[0] != ((npy_int32 *)%(x_nrows)s->data)[0] || %(z)s->dimensions[1] != %(y)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of the allocated output doesn't match the correct output size."); %(fail)s;}

        if (PyArray_SIZE(%(alpha)s) != 1)
        {PyErr_SetString(PyExc_NotImplementedError, "The number of element in alpha must be 1"); %(fail)s;}

        if (%(alpha)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of alpha must be 2"); %(fail)s;}

        if (%(x_val)s->nd != 1)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of x_val must be 1"); %(fail)s;}

        if (%(y)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of y must be 2"); %(fail)s;}

        if (%(z)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of z must be 2"); %(fail)s;}

        if (%(inplace)s)
        {
            if (%(typenum_zn)s != %(typenum_z)s) {
            PyErr_SetString(PyExc_NotImplementedError, "When inplace the output dtype must be the same as the input"); %(fail)s;}

            Py_XDECREF(%(zn)s);
            %(zn)s = %(z)s;
            Py_INCREF(%(zn)s);
        }
        else if (!%(zn)s
            || (%(zn)s->dimensions[0] != ((npy_int32 *)%(x_nrows)s->data)[0])
            || (%(zn)s->dimensions[1] != %(y)s->dimensions[1])
            )
        {
            {Py_XDECREF(%(zn)s);}
            npy_intp dims[] = {0,0};
            dims[0] = ((npy_int32 *)%(x_nrows)s->data)[0];
            dims[1] = %(y)s->dimensions[1];
            %(zn)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_zn)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(zn)s->dimensions[0];
            npy_intp N = %(zn)s->dimensions[1];
            npy_intp K = %(y)s->dimensions[0];

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz   = (dtype_%(z)s*)%(z)s->data;
            dtype_%(zn)s* __restrict__ Dzn   = (dtype_%(zn)s*)%(zn)s->data;
            const dtype_%(x_val)s* __restrict__ Dval = (dtype_%(x_val)s*)%(x_val)s->data;
            const npy_int32 * __restrict__ Dind = (npy_int32*)%(x_ind)s->data;
            const npy_int32 * __restrict__ Dptr = (npy_int32*)%(x_ptr)s->data;
            const dtype_%(alpha)s alpha = ((dtype_%(alpha)s*)%(alpha)s->data)[0];

            npy_intp Sz = %(z)s->strides[1] / %(z)s->descr->elsize;
            npy_intp Szn = %(zn)s->strides[1] / %(zn)s->descr->elsize;
            npy_intp Sval = %(x_val)s->strides[0] / %(x_val)s->descr->elsize;
            npy_intp Sind = %(x_ind)s->strides[0] / %(x_ind)s->descr->elsize;
            npy_intp Sptr = %(x_ptr)s->strides[0] / %(x_ptr)s->descr->elsize;
            npy_intp Sy = %(y)s->strides[1] / %(y)s->descr->elsize;


            if (!(%(inplace)s))
            {
                memcpy(Dzn, Dz, M*N*sizeof(dtype_%(zn)s));
            }

            for (npy_int32 k = 0; k < K; ++k)
            {
                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1)*Sptr]; ++m_idx)
                {
                    const npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K

                    const dtype_%(x_val)s Amk = alpha * Dval[m_idx * Sval]; // actual value at that location

                    const dtype_%(y)s* y_row = (dtype_%(y)s*)(%(y)s->data + %(y)s->strides[0] * k);

                    const dtype_%(zn)s* z_row = (dtype_%(zn)s*)(%(zn)s->data + %(zn)s->strides[0] * m);

                    %(axpy)s((int*)&N, (%(conv_type)s*)&Amk, (%(conv_type)s*)y_row, (int*)&Sy, (%(conv_type)s*)z_row, (int*)&Szn);
                }
            }
        }
        """ % dict(locals(), **sub)

        return rval


usmm_csc_dense = UsmmCscDense(inplace=False)
usmm_csc_dense_inplace = UsmmCscDense(inplace=True)


local_usmm = gof.opt.PatternSub(
    (tensor.sub, 'z',
     (tensor.mul,
      {'pattern': 'alpha',
       'constraint': lambda expr: numpy.all(expr.type.broadcastable)},
    (_dot, 'x', 'y'))),
    (usmm, (tensor.neg, 'alpha'), 'x', 'y', 'z'))


register_specialize(local_usmm, name="local_usmm")


@gof.local_optimizer([usmm])
def local_usmm_csx(node):
    """ usmm -> usmm_csc_dense """
    if node.op == usmm:
        alpha, x, y, z = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable and not y_is_sparse_variable:
            if x.type.format == 'csc':
                x_val, x_ind, x_ptr, x_shape = csm_properties(x)
                x_nsparse = x_shape[0]
                dtype_out = scalar.upcast(alpha.type.dtype, x.type.dtype,
                                          y.type.dtype, z.type.dtype)
                if dtype_out not in ('float32', 'float64'):
                    return False
                # Sparse cast is not implemented.
                if y.type.dtype != dtype_out:
                    return False

                return [usmm_csc_dense(alpha, x_val, x_ind, x_ptr,
                                       x_nsparse, y, z)]
    return False
register_specialize(local_usmm_csx)


@gof.local_optimizer([usmm_csc_dense])
def local_usmm_csc_dense_inplace(node):
    if node.op == usmm_csc_dense:
        return [usmm_csc_dense_inplace(*node.inputs)]
register_specialize(local_usmm_csc_dense_inplace, 'inplace')
