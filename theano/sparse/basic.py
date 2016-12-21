"""
Classes for handling sparse matrices.

To read about different sparse formats, see
http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps

"""
from __future__ import absolute_import, print_function, division

# TODO
# Automatic methods for determining best sparse format?

import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided
from six import integer_types
from six.moves import xrange
import scipy.sparse

import theano
from theano import gof, tensor, scalar, config
from theano.gradient import DisconnectedType
from theano.sparse.utils import hash_from_sparse
from theano.gradient import grad_not_implemented, grad_undefined
from theano.sparse.type import SparseType, _is_sparse

sparse_formats = ['csc', 'csr']


"""
Types of sparse matrices to use for testing.

"""
_mtypes = [scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
# _mtypes = [sparse.csc_matrix, sparse.csr_matrix, sparse.dok_matrix,
# sparse.lil_matrix, sparse.coo_matrix]
# * new class ``dia_matrix`` : the sparse DIAgonal format
# * new class ``bsr_matrix`` : the Block CSR format
_mtype_to_str = {scipy.sparse.csc_matrix: "csc",
                 scipy.sparse.csr_matrix: "csr"}


def _is_sparse_variable(x):
    """

    Returns
    -------
    boolean
        True iff x is a L{SparseVariable} (and not a L{tensor.TensorType},
        for instance).

    """
    if not isinstance(x, gof.Variable):
        raise NotImplementedError("this function should only be called on "
                                  "*variables* (of type sparse.SparseType "
                                  "or tensor.TensorType, for instance), not ",
                                  x)
    return isinstance(x.type, SparseType)


def _is_dense_variable(x):
    """

    Returns
    -------
    boolean
        True if x is a L{tensor.TensorType} (and not a L{SparseVariable},
        for instance).

    """
    if not isinstance(x, gof.Variable):
        raise NotImplementedError("this function should only be called on "
                                  "*variables* (of type sparse.SparseType or "
                                  "tensor.TensorType, for instance), not ", x)
    return isinstance(x.type, tensor.TensorType)


def _is_dense(x):
    """

    Returns
    -------
    boolean
        True unless x is a L{scipy.sparse.spmatrix} (and not a
        L{numpy.ndarray}).

    """
    if not isinstance(x, (scipy.sparse.spmatrix, np.ndarray)):
        raise NotImplementedError("this function should only be called on "
                                  "sparse.scipy.sparse.spmatrix or "
                                  "numpy.ndarray, not,", x)
    return isinstance(x, np.ndarray)


# Wrapper type
def as_sparse_variable(x, name=None):
    """
    Wrapper around SparseVariable constructor to construct
    a Variable with a sparse matrix with the same dtype and
    format.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    object
        SparseVariable version of `x`.

    """

    # TODO
    # Verify that sp is sufficiently sparse, and raise a
    # warning if it is not

    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a "
                             "multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, gof.Variable):
        if not isinstance(x.type, SparseType):
            raise TypeError("Variable type field must be a SparseType.", x,
                            x.type)
        return x
    try:
        return constant(x, name=name)
    except TypeError:
        raise TypeError("Cannot convert %s to SparseType" % x, type(x))
as_sparse = as_sparse_variable


def as_sparse_or_tensor_variable(x, name=None):
    """
    Same as `as_sparse_variable` but if we can't make a
    sparse variable, we try to make a tensor variable.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    SparseVariable or TensorVariable version of `x`

    """

    try:
        return as_sparse_variable(x, name)
    except (ValueError, TypeError):
        return theano.tensor.as_tensor_variable(x, name)


def constant(x, name=None):
    if not isinstance(x, scipy.sparse.spmatrix):
        raise TypeError("sparse.constant must be called on a "
                        "scipy.sparse.spmatrix")
    try:
        return SparseConstant(SparseType(format=x.format,
                                         dtype=x.dtype), x.copy(), name=name)
    except TypeError:
        raise TypeError("Could not convert %s to SparseType" % x, type(x))


def sp_ones_like(x):
    """
    Construct a sparse matrix of ones with the same sparsity pattern.

    Parameters
    ----------
    x
        Sparse matrix to take the sparsity pattern.

    Returns
    -------
    A sparse matrix
        The same as `x` with data changed for ones.

    """
    # TODO: don't restrict to CSM formats
    data, indices, indptr, shape = csm_properties(x)
    return CSM(format=x.format)(tensor.ones_like(data), indices, indptr, shape)


def sp_zeros_like(x):
    """
    Construct a sparse matrix of zeros.

    Parameters
    ----------
    x
        Sparse matrix to take the shape.

    Returns
    -------
    A sparse matrix
        The same as `x` with zero entries for all element.

    """

    # TODO: don't restrict to CSM formats
    _, _, indptr, shape = csm_properties(x)
    return CSM(format=x.format)(data=np.array([], dtype=x.type.dtype),
                                indices=np.array([], dtype='int32'),
                                indptr=tensor.zeros_like(indptr),
                                shape=shape)


class _sparse_py_operators:
    T = property(lambda self: transpose(self),
                 doc="Return aliased transpose of self (read-only)")

    def astype(self, dtype):
        return cast(self, dtype)

    def __neg__(self):
        return neg(self)

    def __add__(left, right):
        return add(left, right)

    def __radd__(right, left):
        return add(left, right)

    def __sub__(left, right):
        return sub(left, right)

    def __rsub__(right, left):
        return sub(left, right)

    def __mul__(left, right):
        return mul(left, right)

    def __rmul__(left, right):
        return mul(left, right)

    # comparison operators

    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    # extra pseudo-operator symbols

    def __dot__(left, right):
        return structured_dot(left, right)

    def __rdot__(right, left):
        return structured_dot(left, right)

    # N.B. THIS IS COMMENTED OUT ON PURPOSE!!!
    #     Discussion with Fred & James (at least, and maybe others before)
    #     we decided that casting from a sparse to dense should be explicit
    #     because it's usually something you just want to be pretty careful
    #     about, and not to do by accident.
    # def _as_TensorVariable(self):
    #    return dense_from_sparse(self)

    def toarray(self):
        return dense_from_sparse(self)
    shape = property(lambda self: tensor.shape(dense_from_sparse(self)))
    # don't worry!
    # the plan is that the ShapeFeature in tensor.opt will do shape propagation
    # and remove the dense_from_sparse from the graph.  This will *NOT*
    # actually expand your sparse matrix just to get the shape.
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
            scalar_arg_1 = (np.isscalar(args[0]) or
                            getattr(args[0], 'type', None) == tensor.iscalar)
            scalar_arg_2 = (np.isscalar(args[1]) or
                            getattr(args[1], 'type', None) == tensor.iscalar)
            if scalar_arg_1 and scalar_arg_2:
                ret = get_item_scalar(self, args)
            elif isinstance(args[0], list):
                ret = get_item_2lists(self, args[0], args[1])
            else:
                ret = get_item_2d(self, args)
        elif isinstance(args[0], list):
            ret = get_item_list(self, args[0])
        else:
            ret = get_item_2d(self, args)
        return ret


class SparseVariable(_sparse_py_operators, gof.Variable):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

    def __str__(self):
        return '%s{%s,%s}' % (
            self.__class__.__name__,
            self.format,
            self.dtype)

    def __repr__(self):
        return str(self)


class SparseConstantSignature(tuple):
    def __eq__(self, other):
        (a, b), (x, y) = self, other
        return (a == x and
                (b.dtype == y.dtype) and
                (type(b) == type(y)) and
                (b.shape == y.shape) and
                (abs(b - y).sum() < 1e-6 * b.nnz))

    def __hash__(self):
        (a, b) = self
        return hash(type(self)) ^ hash(a) ^ hash(type(b))

    def theano_hash(self):
        (_, d) = self
        return hash_from_sparse(d)


class SparseConstant(gof.Constant, _sparse_py_operators):
    dtype = property(lambda self: self.type.dtype)
    format = property(lambda self: self.type.format)

    def signature(self):
        assert self.data is not None
        return SparseConstantSignature((self.type, self.data))

    def __str__(self):
        return '%s{%s,%s,shape=%s,nnz=%s}' % (
            self.__class__.__name__,
            self.format,
            self.dtype,
            self.data.shape,
            self.data.nnz)

    def __repr__(self):
        return str(self)


SparseType.Variable = SparseVariable
SparseType.Constant = SparseConstant


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


def bsr_matrix(name=None, dtype=None):
    return matrix('bsr', name, dtype)


# for more dtypes, call SparseType(format, dtype)
csc_dmatrix = SparseType(format='csc', dtype='float64')
csr_dmatrix = SparseType(format='csr', dtype='float64')
bsr_dmatrix = SparseType(format='bsr', dtype='float64')
csc_fmatrix = SparseType(format='csc', dtype='float32')
csr_fmatrix = SparseType(format='csr', dtype='float32')
bsr_fmatrix = SparseType(format='bsr', dtype='float32')

all_dtypes = SparseType.dtype_set
complex_dtypes = [t for t in all_dtypes if t[:7] == 'complex']
float_dtypes = [t for t in all_dtypes if t[:5] == 'float']
int_dtypes = [t for t in all_dtypes if t[:3] == 'int']
uint_dtypes = [t for t in all_dtypes if t[:4] == 'uint']
integer_dtypes = int_dtypes + uint_dtypes

continuous_dtypes = complex_dtypes + float_dtypes
discrete_dtypes = int_dtypes + uint_dtypes


# CONSTRUCTION
class CSMProperties(gof.Op):
    # See doc in instance of this Op or function after this class definition.
    # NOTE
    # We won't implement infer_shape for this op now. This will
    # ask that we implement an GetNNZ op, and this op will keep
    # the dependence on the input of this op. So this won't help
    # to remove computations in the graph. To remove computation,
    # we will need to make an infer_sparse_pattern feature to
    # remove computations. Doing this is trickier then the
    # infer_shape feature. For example, how do we handle the case
    # when some op create some 0 values? So there is dependence
    # on the values themselves. We could write an infer_shape for
    # the last output that is the shape, but I dough this will
    # get used.

    # we don't return a view of the shape, we create a new ndarray from the
    # shape tuple.
    __props__ = ()
    view_map = {0: [0], 1: [0], 2: [0]}

    """
    Indexing to speficied what part of the data parameter
    should be use to construct the sparse matrix.

    """

    def __init__(self, kmap=None):
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")

    def make_node(self, csm):
        csm = as_sparse_variable(csm)
        assert csm.format in ["csr", "csc"]
        data = tensor.TensorType(dtype=csm.type.dtype,
                                 broadcastable=(False,))()
        return gof.Apply(self, [csm],
                         [data, tensor.ivector(),
                          tensor.ivector(), tensor.ivector()])

    def perform(self, node, inputs, out):
        (csm,) = inputs
        out[0][0] = csm.data
        if str(csm.data.dtype) == 'int32':
            out[0][0] = theano._asarray(out[0][0], dtype='int32')
        # backport
        out[1][0] = theano._asarray(csm.indices, dtype='int32')
        out[2][0] = theano._asarray(csm.indptr, dtype='int32')
        out[3][0] = theano._asarray(csm.shape, dtype='int32')

    def grad(self, inputs, g):

        # g[1:] is all integers, so their Jacobian in this op
        # is 0. We thus don't need to worry about what their values
        # are.

        # if g[0] is disconnected, then this op doesn't contribute
        # any gradient anywhere. but we know that at least one of
        # g[1:] is connected, or this grad method wouldn't have been
        # called, so we should report zeros
        (csm,) = inputs
        if isinstance(g[0].type, DisconnectedType):
            return [csm.zeros_like()]

        data, indices, indptr, shape = csm_properties(csm)
        return [CSM(csm.format)(g[0], indices, indptr, shape)]

# don't make this a function or it breaks some optimizations below
csm_properties = CSMProperties()
"""
Extract all of .data, .indices, .indptr and .shape field.

For specific field, `csm_data`, `csm_indices`, `csm_indptr`
and `csm_shape` are provided.

Parameters
----------
csm
    Sparse matrix in CSR or CSC format.

Returns
    (data, indices, indptr, shape), the properties of `csm`.

Notes
-----
The grad implemented is regular, i.e. not structured.
`infer_shape` method is not available for this op.

"""


def csm_data(csm):
    """
    Return the data field of the sparse variable.

    """
    return csm_properties(csm)[0]


def csm_indices(csm):
    """
    Return the indices field of the sparse variable.

    """
    return csm_properties(csm)[1]


def csm_indptr(csm):
    """
    Return the indptr field of the sparse variable.

    """
    return csm_properties(csm)[2]


def csm_shape(csm):
    """
    Return the shape field of the sparse variable.

    """
    return csm_properties(csm)[3]


class CSM(gof.Op):
    # See doc in instance of this Op or function after this class definition.
    """
    Indexing to speficied what part of the data parameter
    should be used to construct the sparse matrix.

    """
    __props__ = ('format',)
    """
    Pre-computed hash value, defined by __init__.

    """

    def __init__(self, format, kmap=None):
        if format not in ('csr', 'csc'):
            raise ValueError("format must be one of: 'csr', 'csc'", format)
        self.format = format
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")
        # should view the other inputs too, but viewing multiple
        # inputs is not currently supported by the destroyhandler
        self.view_map = {0: [0]}

    def make_node(self, data, indices, indptr, shape):
        data = tensor.as_tensor_variable(data)

        if not isinstance(indices, gof.Variable):
            indices_ = np.asarray(indices)
            indices_32 = theano._asarray(indices, dtype='int32')
            assert (indices_ == indices_32).all()
            indices = indices_32
        if not isinstance(indptr, gof.Variable):
            indptr_ = np.asarray(indptr)
            indptr_32 = theano._asarray(indptr, dtype='int32')
            assert (indptr_ == indptr_32).all()
            indptr = indptr_32
        if not isinstance(shape, gof.Variable):
            shape_ = np.asarray(shape)
            shape_32 = theano._asarray(shape, dtype='int32')
            assert (shape_ == shape_32).all()
            shape = shape_32

        indices = tensor.as_tensor_variable(indices)
        indptr = tensor.as_tensor_variable(indptr)
        shape = tensor.as_tensor_variable(shape)

        if data.type.ndim != 1:
            raise TypeError('data argument must be a vector', data.type,
                            data.type.ndim)
        if indices.type.ndim != 1 or indices.type.dtype not in discrete_dtypes:
            raise TypeError('indices must be vector of integers', indices,
                            indices.type)
        if indptr.type.ndim != 1 or indptr.type.dtype not in discrete_dtypes:
            raise TypeError('indices must be vector of integers', indptr,
                            indptr.type)
        if shape.type.ndim != 1 or shape.type.dtype not in discrete_dtypes:
            raise TypeError('n_rows must be integer type', shape, shape.type)

        return gof.Apply(self,
                         [data, indices, indptr, shape],
                         [SparseType(dtype=data.type.dtype,
                                     format=self.format)()])

    def perform(self, node, inputs, outputs):
        # for efficiency, if remap does nothing, then do not apply it
        (data, indices, indptr, shape) = inputs
        (out,) = outputs

        if len(shape) != 2:
            raise ValueError('Shape should be an array of length 2')
        if data.shape != indices.shape:
            errmsg = ('Data (shape ' + repr(data.shape) +
                      ' must have the same number of elements ' +
                      'as indices (shape' + repr(indices.shape) +
                      ')')
            raise ValueError(errmsg)
        if self.format == 'csc':
            out[0] = scipy.sparse.csc_matrix((data, indices.copy(),
                                              indptr.copy()),
                                             np.asarray(shape), copy=False)
        else:
            assert self.format == 'csr'
            out[0] = scipy.sparse.csr_matrix((data, indices.copy(),
                                              indptr.copy()), shape.copy(),
                                             copy=False)

    def connection_pattern(self, node):
        return [[True], [False], [False], [False]]

    def grad(self, inputs, gout):
        (x_data, x_indices, x_indptr, x_shape) = inputs
        (g_out,) = gout
        g_data, g_indices, g_indptr, g_shape = csm_properties(g_out)
        # unpack the data vector and wrap it as a 1d TensorType
        g_data = csm_grad()(x_data, x_indices, x_indptr, x_shape,
                            g_data, g_indices, g_indptr, g_shape)
        return [g_data, DisconnectedType()(), DisconnectedType()(), DisconnectedType()()]

    def infer_shape(self, node, shapes):
        # node.inputs[3] is of lenght as we only support sparse matrix.
        return [(node.inputs[3][0], node.inputs[3][1])]

CSC = CSM('csc')
"""
Construct a CSC matrix from the internal representation.

Parameters
----------
data
    One dimensional tensor representing the data of the sparse matrix to
    construct.
indices
    One dimensional tensor of integers representing the indices of the sparse
    matrix to construct.
indptr
    One dimensional tensor of integers representing the indice pointer for
    the sparse matrix to construct.
shape
    One dimensional tensor of integers representing the shape of the sparse
    matrix to construct.

Returns
-------
sparse matrix
    A sparse matrix having the properties specified by the inputs.

Notes
-----
The grad method returns a dense vector, so it provides a regular grad.

"""

CSR = CSM('csr')
"""
Construct a CSR matrix from the internal representation.

Parameters
----------
data
    One dimensional tensor representing the data of the sparse matrix to
    construct.
indices
    One dimensional tensor of integers representing the indices of the sparse
    matrix to construct.
indptr
    One dimensional tensor of integers representing the indice pointer for
    the sparse matrix to construct.
shape
    One dimensional tensor of integers representing the shape of the sparse
    matrix to construct.

Returns
-------
sparse matrix
    A sparse matrix having the properties specified by the inputs.

Notes
-----
The grad method returns a dense vector, so it provides a regular grad.

"""


class CSMGrad(gof.op.Op):
    # Note
    # This Op computes the gradient of the CSM Op. CSM creates a matrix from
    # data, indices, and indptr vectors; it's gradient is the gradient of
    # the data vector only. There are two complexities to calculate this
    # gradient:
    # 1. The gradient may be sparser than the input matrix defined by (data,
    # indices, indptr). In this case, the data vector of the gradient will have
    # less elements than the data vector of the input because sparse formats
    # remove 0s. Since we are only returning the gradient of the data vector,
    # the relevant 0s need to be added back.
    # 2. The elements in the sparse dimension are not guaranteed to be sorted.
    # Therefore, the input data vector may have a different order than the
    # gradient data vector.
    __props__ = ()

    def __init__(self, kmap=None):
        if kmap is not None:
            raise Exception("Do not use kmap, it is removed")
        # This class always allocate a new output.
        # I keep this here to help GD understand what this kmap think is.
        # if self.kmap is None:
        #    self.view_map = {0: [1]}

    def make_node(self, x_data, x_indices, x_indptr, x_shape,
                  g_data, g_indices, g_indptr, g_shape):
        gout_data = g_data.type()
        return gof.Apply(self, [x_data, x_indices, x_indptr, x_shape,
                         g_data, g_indices, g_indptr, g_shape], [gout_data])

    def perform(self, node, inputs, outputs):
        (x_data, x_indices, x_indptr, x_shape,
         g_data, g_indices, g_indptr, g_shape) = inputs
        (g_out,) = outputs
        if len(x_indptr) - 1 == x_shape[0]:
            sp_dim = x_shape[1]
        else:
            sp_dim = x_shape[0]

        g_row = np.zeros(sp_dim, dtype=g_data.dtype)
        gout_data = np.zeros(x_data.shape, dtype=node.outputs[0].dtype)

        for i in range(len(x_indptr) - 1):
            for j_ptr in range(g_indptr[i], g_indptr[i + 1]):
                g_row[g_indices[j_ptr]] += g_data[j_ptr]

            for j_ptr in range(x_indptr[i], x_indptr[i + 1]):
                gout_data[j_ptr] = g_row[x_indices[j_ptr]]

            for j_ptr in range(g_indptr[i], g_indptr[i + 1]):
                g_row[g_indices[j_ptr]] = 0

        g_out[0] = gout_data

    def infer_shape(self, node, shapes):
        return [shapes[1]]

csm_grad = CSMGrad


class Cast(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ("out_type",)

    def __init__(self, out_type):
        self.out_type = out_type

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(
            self, [x],
            [SparseType(dtype=self.out_type, format=x.format)()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x.astype(self.out_type)

    def grad(self, inputs, outputs_gradients):
        gz = outputs_gradients[0]

        if gz.dtype in complex_dtypes:
            raise NotImplementedError("grad not implemented for complex types")
        if inputs[0].dtype in complex_dtypes:
            raise NotImplementedError("grad not implemented for complex types")

        if gz.dtype in discrete_dtypes:
            if inputs[0].dtype in discrete_dtypes:
                return [inputs[0].zeros_like(dtype=theano.config.floatX)]
            else:
                return [inputs[0].zeros_like()]
        else:
            if inputs[0].dtype in discrete_dtypes:
                return [gz]
            else:
                return [Cast(inputs[0].dtype)(gz)]

    def infer_shape(self, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.out_type)

bcast = Cast('int8')
wcast = Cast('int16')
icast = Cast('int32')
lcast = Cast('int64')
fcast = Cast('float32')
dcast = Cast('float64')
ccast = Cast('complex64')
zcast = Cast('complex128')


def cast(variable, dtype):
    """
    Cast sparse variable to the desired dtype.

    Parameters
    ----------
    variable
        Sparse matrix.
    dtype
        The dtype wanted.

    Returns
    -------
    Same as `x` but having `dtype` as dtype.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """
    return Cast(dtype)(variable)

#
# Conversion
#


class DenseFromSparse(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()  # We don't put sparse_grad in the props.

    def __init__(self, structured=True):
        self.sparse_grad = structured

    def __str__(self):
        return "%s{structured_grad=%s}" % (
            self.__class__.__name__,
            self.sparse_grad)

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self,
                         [x],
                         [tensor.TensorType(dtype=x.type.dtype,
                                            broadcastable=(False, False))()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        if _is_dense(x):
            print((
                "WARNING: You just called DenseFromSparse on a dense matrix."
            ), file=sys.stderr)
            out[0] = x
        else:
            out[0] = x.toarray()
        assert _is_dense(out[0])

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self.sparse_grad:
            left = sp_ones_like(x)
            right = gz

            # Do upcasting if necessary to avoid an unimplemented case
            # of mul

            if right.dtype == 'float64' and left.dtype == 'float32':
                left = left.astype('float64')

            if right.dtype == 'float32' and left.dtype == 'float64':
                right = right.astype('float64')

            return [left * right]
        else:
            return [SparseFromDense(x.type.format)(gz)]

    def infer_shape(self, node, shapes):
        return [shapes[0]]

dense_from_sparse = DenseFromSparse()
"""
Convert a sparse matrix to a dense one.

Parameters
----------
x
    A sparse matrix.

Returns
-------
theano.tensor.matrix
    A dense matrix, the same as `x`.

Notes
-----
The grad implementation can be controlled through the constructor via the
`structured` parameter. `True` will provide a structured grad while `False`
will provide a regular grad. By default, the grad is structured.

"""


class SparseFromDense(gof.op.Op):

    __props__ = ()

    def __init__(self, format):
        self.format = format

    def __str__(self):
        return "%s{%s}" % (
            self.__class__.__name__,
            self.format)

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.ndim > 2:
            raise TypeError(
                "Theano does not have sparse tensor types with more "
                "than 2 dimensions, but %s.ndim = %i" % (x, x.ndim))
        elif x.ndim == 1:
            x = x.dimshuffle('x', 0)
        elif x.ndim == 0:
            x = x.dimshuffle('x', 'x')
        else:
            assert x.ndim == 2

        return gof.Apply(self,
                         [x],
                         [SparseType(dtype=x.type.dtype,
                                     format=self.format)()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        out[0] = SparseType.format_cls[self.format](x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        gx = dense_from_sparse(gz)
        gx = tensor.patternbroadcast(gx, x.broadcastable)
        return gx,

    def infer_shape(self, node, shapes):
        return [shapes[0]]

csr_from_dense = SparseFromDense('csr')
"""
Convert a dense matrix to a sparse csr matrix.

Parameters
----------
x
    A dense matrix.

Returns
-------
sparse matrix
    The same as `x` in a sparse csr matrix format.

"""

csc_from_dense = SparseFromDense('csc')
"""
Convert a dense matrix to a sparse csc matrix.

Parameters
----------
x
    A dense matrix.

Returns
-------
sparse matrix
    The same as `x` in a sparse csc matrix format.

"""


# Indexing
class GetItemList(gof.op.Op):

    __props__ = ()

    def infer_shape(self, node, shapes):
        return [(shapes[1][0], shapes[0][1])]

    def make_node(self, x, index):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]

        ind = tensor.as_tensor_variable(index)
        assert ind.ndim == 1
        assert ind.dtype in integer_dtypes

        return gof.Apply(self, [x, ind], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        indices = inp[1]
        assert _is_sparse(x)
        out[0] = x[indices]

    def grad(self, inputs, g_outputs):
        x, indices = inputs
        gout, = g_outputs
        return [get_item_list_grad(x, indices, gout),
                grad_undefined(self, 1, indices, "No gradient for this input")]

get_item_list = GetItemList()
"""
Select row of sparse matrix, returning them as a new sparse matrix.

Parameters
----------
x
    Sparse matrix.
index
    List of rows.

Returns
-------
sparse matrix
    The corresponding rows in `x`.

"""


class GetItemListGrad(gof.op.Op):

    __props__ = ()

    def infer_shape(self, node, shapes):
        return [(shapes[0])]

    def make_node(self, x, index, gz):
        x = as_sparse_variable(x)
        gz = as_sparse_variable(gz)

        assert x.format in ["csr", "csc"]
        assert gz.format in ["csr", "csc"]

        ind = tensor.as_tensor_variable(index)
        assert ind.ndim == 1
        assert ind.dtype in integer_dtypes

        scipy_ver = [int(n) for n in scipy.__version__.split('.')[:2]]

        if not scipy_ver >= [0, 13]:
            raise NotImplementedError("Scipy version is to old")

        return gof.Apply(self, [x, ind, gz], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        indices = inp[1]
        gz = inp[2]

        if x.format in ["csr"]:
            y = scipy.sparse.csr_matrix((x.shape[0], x.shape[1]))
        else:
            y = scipy.sparse.csc_matrix((x.shape[0], x.shape[1]))
        for a in range(0, len(indices)):
                y[indices[a]] = gz[a]

        out[0] = y

get_item_list_grad = GetItemListGrad()


class GetItem2Lists(gof.op.Op):

    __props__ = ()

    def make_node(self, x, ind1, ind2):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        ind1 = tensor.as_tensor_variable(ind1)
        ind2 = tensor.as_tensor_variable(ind2)
        assert ind1.dtype in integer_dtypes
        assert ind2.dtype in integer_dtypes

        return gof.Apply(self, [x, ind1, ind2],
                         [theano.tensor.vector()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        ind1 = inp[1]
        ind2 = inp[2]
        out[0] = np.asarray(x[ind1, ind2]).flatten()
        """
        Here scipy returns the corresponding elements in a matrix which isn't
        what we are aiming for. Using asarray and flatten, out[0] becomes an
        array.

        """
    def grad(self, inputs, g_outputs):
        x, ind1, ind2 = inputs
        gout, = g_outputs
        return [get_item_2lists_grad(x, ind1, ind2, gout),
                grad_undefined(self, 1, ind1, "No gradient for this input"),
                grad_undefined(self, 1, ind2, "No gradient for this input")]

get_item_2lists = GetItem2Lists()
"""
Select elements of sparse matrix, returning them in a vector.

Parameters
----------
x
    Sparse matrix.
index
    List of two lists, first list indicating the row of each element and second
    list indicating its column.

Returns
-------
theano.tensor.vector
    The corresponding elements in `x`.

"""


class GetItem2ListsGrad(gof.op.Op):

    __props__ = ()

    def infer_shape(self, node, shapes):
        return [(shapes[0])]

    def make_node(self, x, ind1, ind2, gz):
        x = as_sparse_variable(x)

        assert x.format in ["csr", "csc"]

        ind1 = tensor.as_tensor_variable(ind1)
        ind2 = tensor.as_tensor_variable(ind2)
        assert ind1.ndim == 1
        assert ind2.ndim == 1
        assert ind1.dtype in integer_dtypes
        assert ind2.dtype in integer_dtypes

        return gof.Apply(self, [x, ind1, ind2, gz], [x.type()])

    def perform(self, node, inp, outputs):
        (out,) = outputs
        x = inp[0]
        ind1 = inp[1]
        ind2 = inp[2]
        gz = inp[3]

        if x.format in ["csr"]:
            y = scipy.sparse.csr_matrix((x.shape[0], x.shape[1]))
        else:
            y = scipy.sparse.csc_matrix((x.shape[0], x.shape[1]))
        z = 0
        for z in range(0, len(ind1)):
            y[(ind1[z], ind2[z])] = gz[z]

        out[0] = y

get_item_2lists_grad = GetItem2ListsGrad()


class GetItem2d(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.

    __props__ = ()

# Fred:Too complicated for now. If you need it, look at
# the Subtensor.infer_shape.
#    def infer_shape(self, node, i0_shapes):
#        return i0_shapes
    def make_node(self, x, index):
        scipy_ver = [int(n) for n in scipy.__version__.split('.')[:2]]
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        assert len(index) in [1, 2]

        input_op = [x]
        generic_None = theano.gof.Constant(theano.gof.generic, None)

        for ind in index:
            if isinstance(ind, slice):
                # in case of slice is written in theano variable
                start = ind.start
                stop = ind.stop
                step = ind.step
                # If start or stop or step are None, make them a Generic
                # constant. Else, they should be converted to Tensor Variables
                # of dimension 1 and int/uint dtype.
                if scipy_ver < [0, 14] and ind.step is not None:
                    raise ValueError(
                        'Slice with step is not support with current'
                        ' version of Scipy.')
                if ind.step is None or ind.step == 1:
                    step = generic_None
                else:
                    if not isinstance(step, gof.Variable):
                        step = tensor.as_tensor_variable(step)
                    if not (step.ndim == 0 and step.dtype in
                            tensor.discrete_dtypes):
                        raise ValueError((
                            "Impossible to index into a sparse matrix with "
                            "slice where step=%s" % step),
                            step.ndim, step.dtype)

                if start is None:
                    start = generic_None
                else:
                    if not isinstance(start, gof.Variable):
                        start = tensor.as_tensor_variable(start)
                    if not (start.ndim == 0 and start.dtype in
                            tensor.discrete_dtypes):
                        raise ValueError((
                            "Impossible to index into a sparse matrix with "
                            "slice where start=%s" % start),
                            start.ndim, start.dtype)

                if stop is None:
                    stop = generic_None
                else:
                    if not isinstance(stop, gof.Variable):
                        stop = tensor.as_tensor_variable(stop)
                    if not (stop.ndim == 0 and stop.dtype in
                            tensor.discrete_dtypes):
                        raise ValueError((
                            "Impossible to index into a sparse matrix with "
                            "slice where stop=%s" % stop),
                            stop.ndim, stop.dtype)

            elif ((isinstance(ind, gof.Variable) and
                    getattr(ind, 'ndim', -1) == 0) or
                    np.isscalar(ind)):
                raise NotImplementedError(
                    'Theano has no sparse vector' +
                    'Use X[a:b, c:d], X[a:b, c:c+1] or X[a:b] instead.')
            else:
                raise ValueError((
                    'Advanced indexing is not implemented for sparse '
                    'matrices. Argument not supported: %s' % ind))
            input_op += [start, stop, step]
        if len(index) == 1:
            input_op += [generic_None, generic_None, generic_None]

        return gof.Apply(self, input_op, [x.type()])

    def perform(self, node, inputs, outputs):
        (x, start1, stop1, step1, start2, stop2, step2) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x[start1:stop1:step1, start2:stop2:step2]

get_item_2d = GetItem2d()
"""
Implement a subtensor of sparse variable, returning a sparse matrix.

If you want to take only one element of a sparse matrix see
`GetItemScalar` that returns a tensor scalar.

Parameters
----------
x
    Sparse matrix.
index
    Tuple of slice object.

Returns
-------
sparse matrix
    The corresponding slice in `x`.


Notes
-----
Subtensor selection always returns a matrix, so indexing with [a:b, c:d]
is forced. If one index is a scalar, for instance, x[a:b, c] or x[a, b:c],
an error will be raised. Use instead x[a:b, c:c+1] or x[a:a+1, b:c].

The above indexing methods are not supported because the return value
would be a sparse matrix rather than a sparse vector, which is a
deviation from numpy indexing rule. This decision is made largely
to preserve consistency between numpy and theano. This may be revised
when sparse vectors are supported.

The grad is not implemented for this op.

"""


class GetItemScalar(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def infer_shape(self, node, shapes):
        return [()]

    def make_node(self, x, index):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        assert len(index) == 2

        input_op = [x]

        for ind in index:

            if isinstance(ind, slice):
                raise Exception("GetItemScalar called with a slice as index!")

            # in case of indexing using int instead of theano variable
            elif isinstance(ind, integer_types):
                ind = theano.tensor.constant(ind)
                input_op += [ind]

            # in case of indexing using theano variable
            elif ind.ndim == 0:
                input_op += [ind]
            else:
                raise NotImplemented()

        return gof.Apply(self, input_op, [tensor.scalar(dtype=x.dtype)])

    def perform(self, node, inputs, outputs):
        (x, ind1, ind2) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = theano._asarray(x[ind1, ind2], x.dtype)

get_item_scalar = GetItemScalar()
"""
Implement a subtensor of a sparse variable that takes two scalars as index and
returns a scalar.

If you want to take a slice of a sparse matrix see `GetItem2d` that returns a
sparse matrix.

Parameters
----------
x
    Sparse matrix.
index
    Tuple of scalars.

Returns
-------
theano.tensor.scalar
    The corresponding item in `x`.

Notes
-----
The grad is not implemented for this op.

"""


# Linear Algebra
class Transpose(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    view_map = {0: [0]}

    format_map = {'csr': 'csc',
                  'csc': 'csr'}
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self,
                         [x],
                         [SparseType(dtype=x.type.dtype,
                                     format=self.format_map[x.type.format])()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = x.transpose()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return transpose(gz),

    def infer_shape(self, node, shapes):
        return [shapes[0][::-1]]
transpose = Transpose()
"""
Return the transpose of the sparse matrix.

Parameters
----------
x
    Sparse matrix.

Returns
-------
sparse matrix
    `x` transposed.

Notes
-----
The returned matrix will not be in the same format. `csc` matrix will be changed
in `csr` matrix and `csr` matrix in `csc` matrix.

The grad is regular, i.e. not structured.

"""


class Neg(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.

    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        out[0] = -x

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_sparse_variable(gz)
        return -gz,

    def infer_shape(self, node, shapes):
        return [shapes[0]]
neg = Neg()
"""
Return the negation of the sparse matrix.

Parameters
----------
x
    Sparse matrix.

Returns
-------
sparse matrix
    -`x`.

Notes
-----
The grad is regular, i.e. not structured.

"""


class ColScaleCSC(gof.op.Op):
    # Scale each columns of a sparse matrix by the corresponding
    # element of a dense vector

    # :param x: A sparse matrix.
    # :param s: A dense vector with length equal to the number
    #           of columns of `x`.

    # :return: A sparse matrix in the same format as `x` which
    #          each column had been multiply by the corresponding
    #          element of `s`.

    # :note: The grad implemented is structured.

    __props__ = ()

    def make_node(self, x, s):
        if x.format != 'csc':
            raise ValueError('x was not a csc matrix')
        return gof.Apply(self, [x, s], [x.type()])

    def perform(self, node, inputs, outputs):
        (x, s) = inputs
        (z,) = outputs
        M, N = x.shape
        assert x.format == 'csc'
        assert s.shape == (N, )

        y = x.copy()

        for j in xrange(0, N):
            y.data[y.indptr[j]: y.indptr[j + 1]] *= s[j]

        z[0] = y

    def grad(self, inputs, gout):
        (x, s) = inputs
        (gz,) = gout
        return [col_scale(gz, s), sp_sum(x * gz, axis=0)]

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]


class RowScaleCSC(gof.op.Op):
    # Scale each row of a sparse matrix by the corresponding element of
    # a dense vector

    # :param x: A sparse matrix.
    # :param s: A dense vector with length equal to the number
    #           of rows of `x`.

    # :return: A sparse matrix in the same format as `x` which
    #          each row had been multiply by the corresponding
    #          element of `s`.

    # :note: The grad implemented is structured.

    view_map = {0: [0]}
    __props__ = ()

    def make_node(self, x, s):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self, [x, s], [x.type()])

    def perform(self, node, inputs, outputs):
        (x, s) = inputs
        (z,) = outputs
        M, N = x.shape
        assert x.format == 'csc'
        assert s.shape == (M,)

        indices = x.indices
        indptr = x.indptr

        y_data = x.data.copy()

        for j in xrange(0, N):
            for i_idx in xrange(indptr[j], indptr[j + 1]):
                y_data[i_idx] *= s[indices[i_idx]]

        z[0] = scipy.sparse.csc_matrix((y_data, indices, indptr), (M, N))

    def grad(self, inputs, gout):
        (x, s) = inputs
        (gz,) = gout
        return [row_scale(gz, s), sp_sum(x * gz, axis=1)]

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]


def col_scale(x, s):
    """
    Scale each columns of a sparse matrix by the corresponding element of a
    dense vector.

    Parameters
    ----------
    x
        A sparse matrix.
    s
        A dense vector with length equal to the number of columns of `x`.

    Returns
    -------
    A sparse matrix in the same format as `x` which each column had been
    multiply by the corresponding element of `s`.

    Notes
    -----
    The grad implemented is structured.

    """

    if x.format == 'csc':
        return ColScaleCSC()(x, s)
    elif x.format == 'csr':
        return RowScaleCSC()(x.T, s).T
    else:
        raise NotImplementedError()


def row_scale(x, s):
    """
    Scale each row of a sparse matrix by the corresponding element of
    a dense vector.

    Parameters
    ----------
    x
        A sparse matrix.
    s
        A dense vector with length equal to the number of rows of `x`.

    Returns
    -------
    A sparse matrix
        A sparse matrix in the same format as `x` whose each row has been
        multiplied by the corresponding element of `s`.

    Notes
    -----
    The grad implemented is structured.

    """
    return col_scale(x.T, s).T


class SpSum(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.

    __props__ = ("axis",)
    # WARNING: judgement call...
    # We are not using the structured in the comparison or hashing
    # because it doesn't change the perform method therefore, we
    # *do* want Sums with different structured values to be merged
    # by the merge optimization and this requires them to compare equal.

    def __init__(self, axis=None, sparse_grad=True):
        super(SpSum, self).__init__()
        self.axis = axis
        self.structured = sparse_grad
        if self.axis not in (None, 0, 1):
            raise ValueError('Illegal value for self.axis.')

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        b = ()
        if self.axis is not None:
            b = (False,)

        z = tensor.TensorType(broadcastable=b, dtype=x.dtype)()
        return gof.Apply(self, [x], [z])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.axis is None:
            z[0] = np.asarray(x.sum())
        else:
            z[0] = np.asarray(x.sum(self.axis)).ravel()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.dtype not in continuous_dtypes:
            return [x.zeros_like(dtype=theano.config.floatX)]
        if self.structured:
            if self.axis is None:
                r = gz * theano.sparse.sp_ones_like(x)
            elif self.axis == 0:
                r = col_scale(theano.sparse.sp_ones_like(x), gz)
            elif self.axis == 1:
                r = row_scale(theano.sparse.sp_ones_like(x), gz)
            else:
                raise ValueError('Illegal value for self.axis.')
        else:
            o_format = x.format
            x = dense_from_sparse(x)
            if _is_sparse_variable(gz):
                gz = dense_from_sparse(gz)
            if self.axis is None:
                r = tensor.second(x, gz)
            else:
                ones = tensor.ones_like(x)
                if self.axis == 0:
                    r = tensor.addbroadcast(gz.dimshuffle('x', 0), 0) * ones
                elif self.axis == 1:
                    r = tensor.addbroadcast(gz.dimshuffle(0, 'x'), 1) * ones
                else:
                    raise ValueError('Illegal value for self.axis.')
            r = SparseFromDense(o_format)(r)
        return [r]

    def infer_shape(self, node, shapes):
        r = None
        if self.axis is None:
            r = [()]
        elif self.axis == 0:
            r = [(shapes[0][1],)]
        else:
            r = [(shapes[0][0],)]
        return r

    def __str__(self):
        return self.__class__.__name__ + "{axis=%s}" % str(self.axis)


def sp_sum(x, axis=None, sparse_grad=False):
    """
    Calculate the sum of a sparse matrix along the specified axis.

    It operates a reduction along the specified axis. When `axis` is `None`,
    it is applied along all axes.

    Parameters
    ----------
    x
        Sparse matrix.
    axis
        Axis along which the sum is applied. Integer or `None`.
    sparse_grad : bool
        `True` to have a structured grad.

    Returns
    -------
    object
        The sum of `x` in a dense format.

    Notes
    -----
    The grad implementation is controlled with the `sparse_grad` parameter.
    `True` will provide a structured grad and `False` will provide a regular
    grad. For both choices, the grad returns a sparse matrix having the same
    format as `x`.

    This op does not return a sparse matrix, but a dense tensor matrix.

    """

    return SpSum(axis, sparse_grad)(x)


class Diag(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self, [x], [tensor.tensor(broadcastable=(False,),
                                                   dtype=x.dtype)])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        N, M = x.shape
        if N != M:
            raise ValueError('Diag only apply on square matrix')
        z[0] = x.diagonal()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [square_diagonal(gz)]

    def infer_shape(self, nodes, shapes):
        return [(tensor.minimum(*shapes[0]), )]

diag = Diag()
"""
Extract the diagonal of a square sparse matrix as a dense vector.

Parameters
----------
x
    A square sparse matrix in csc format.

Returns
-------
theano.tensor.vector
    A dense vector representing the diagonal elements.

Notes
-----
The grad implemented is regular, i.e. not structured, since the output is a
dense vector.

"""


class SquareDiagonal(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.

    __props__ = ()

    def make_node(self, diag):
        diag = tensor.as_tensor_variable(diag)
        if diag.type.ndim != 1:
            raise TypeError('data argument must be a vector', diag.type)

        return gof.Apply(self, [diag],
                         [SparseType(dtype=diag.dtype, format='csc')()])

    def perform(self, node, inputs, outputs):
        (z,) = outputs
        diag = inputs[0]

        N = len(diag)
        data = diag[:N]
        indices = list(range(N))
        indptr = list(range(N + 1))
        tup = (data, indices, indptr)

        z[0] = scipy.sparse.csc_matrix(tup, copy=True)

    def grad(self, inputs, gout):
        (gz,) = gout
        return [diag(gz)]

    def infer_shape(self, nodes, shapes):
        return [(shapes[0][0], shapes[0][0])]

square_diagonal = SquareDiagonal()
"""
Return a square sparse (csc) matrix whose diagonal is given by the dense vector
argument.

Parameters
----------
x
    Dense vector for the diagonal.

Returns
-------
sparse matrix
    A sparse matrix having `x` as diagonal.

Notes
-----
The grad implemented is regular, i.e. not structured.

"""


class EnsureSortedIndices(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.view_map = {0: [0]}

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.inplace:
            z[0] = x.sort_indices()
        else:
            z[0] = x.sorted_indices()

    def grad(self, inputs, output_grad):
        return [output_grad[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def __str__(self):
        if self.inplace:
            return self.__class__.__name__ + "{inplace}"
        else:
            return self.__class__.__name__ + "{no_inplace}"
ensure_sorted_indices = EnsureSortedIndices(inplace=False)
"""
Re-sort indices of a sparse matrix.

CSR column indices are not necessarily sorted. Likewise
for CSC row indices. Use `ensure_sorted_indices` when sorted
indices are required (e.g. when passing data to other
libraries).

Parameters
----------
x
    A sparse matrix.

Returns
-------
sparse matrix
    The same as `x` with indices sorted.

Notes
-----
The grad implemented is regular, i.e. not structured.

"""


def clean(x):
    """
    Remove explicit zeros from a sparse matrix, and re-sort indices.

    CSR column indices are not necessarily sorted. Likewise
    for CSC row indices. Use `clean` when sorted
    indices are required (e.g. when passing data to other
    libraries) and to ensure there are no zeros in the data.

    Parameters
    ----------
    x
        A sparse matrix.

    Returns
    -------
    A sparse matrix
        The same as `x` with indices sorted and zeros
        removed.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    """
    return ensure_sorted_indices(remove0(x))


class AddSS(gof.op.Op):
    # add(sparse, sparse).
    # see the doc of add() for more detail.
    __props__ = ()

    def make_node(self, x, y):
        x, y = map(as_sparse_variable, [x, y])
        assert x.format in ["csr", "csc"]
        assert y.format in ["csr", "csc"]
        out_dtype = scalar.upcast(x.type.dtype, y.type.dtype)
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=out_dtype,
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and _is_sparse(y)
        assert x.shape == y.shape
        out[0] = x + y

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_sparse_variable(y)
        assert _is_sparse_variable(gz)
        return gz, gz

    def infer_shape(self, node, shapes):
        return [shapes[0]]

add_s_s = AddSS()


class AddSSData(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def make_node(self, x, y):
        x, y = map(as_sparse_variable, [x, y])
        assert x.format in ["csr", "csc"]
        assert y.format in ["csr", "csc"]
        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        if x.type.format != y.type.format:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=x.type.dtype,
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and _is_sparse(y)
        assert x.shape == y.shape
        assert x.data.shape == y.data.shape
        out[0] = x.copy()
        out[0].data += y.data

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [(i.dtype in continuous_dtypes)
                         for i in inputs]
        derivative = {True: gz, False: None}
        return [derivative[b] for b in is_continuous]

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]

add_s_s_data = AddSSData()
"""
Add two sparse matrices assuming they have the same sparsity pattern.

Parameters
----------
x
    Sparse matrix.
y
    Sparse matrix.

Returns
-------
A sparse matrix
    The sum of the two sparse matrices element wise.

Notes
-----
`x` and `y` are assumed to have the same sparsity pattern.

The grad implemented is structured.

"""


class AddSD(gof.op.Op):
    # add(sparse, sparse).
    # see the doc of add() for more detail.
    __props__ = ()

    def make_node(self, x, y):
        x, y = as_sparse_variable(x), tensor.as_tensor_variable(y)
        assert x.format in ["csr", "csc"]
        out_dtype = scalar.upcast(x.type.dtype, y.type.dtype)

        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        return gof.Apply(self,
                         [x, y],
                         [tensor.TensorType(dtype=out_dtype,
                                            broadcastable=y.type.broadcastable
                                            )()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_dense(y)

        # The asarray is needed as in some case, this return a
        # numpy.matrixlib.defmatrix.matrix object and not an ndarray.
        out[0] = theano._asarray(x + y, dtype=node.outputs[0].type.dtype)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_dense_variable(gz)
        return sp_ones_like(x) * gz, gz

    def infer_shape(self, node, shapes):
        return [shapes[1]]

add_s_d = AddSD()


class StructuredAddSV(gof.op.Op):

    __props__ = ()

    def make_node(self, x, y):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        y = tensor.as_tensor_variable(y)

        assert y.type.ndim == 1

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=x.type.dtype,
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and not _is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x + (x.toarray() != 0) * y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and not _is_sparse_variable(y)
        assert _is_sparse_variable(gz)
        return gz, sp_sum(gz, axis=0, sparse_grad=True)

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]

structured_add_s_v = StructuredAddSV()
"""
Structured addition of a sparse matrix and a dense vector.
The elements of the vector are only added to the corresponding
non-zero elements of the sparse matrix. Therefore, this operation
outputs another sparse matrix.

Parameters
----------
x
    Sparse matrix.
y
    Tensor type vector.

Returns
-------
A sparse matrix
    A sparse matrix containing the addition of the vector to
    the data of the sparse matrix.

Notes
-----
The grad implemented is structured since the op is structured.

"""


def add(x, y):
    """
    Add two matrices, at least one of which is sparse.

    This method will provide the right op according
    to the inputs.

    Parameters
    ----------
    x
        A matrix variable.
    y
        A matrix variable.

    Returns
    -------
    A sparse matrix
        `x` + `y`

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    The grad will be structured only when one of the variable will be a dense
    matrix.

    """

    if hasattr(x, 'getnnz'):
        x = as_sparse_variable(x)
    if hasattr(y, 'getnnz'):
        y = as_sparse_variable(y)
    if not isinstance(x, theano.Variable):
        x = theano.tensor.as_tensor_variable(x)
    if not isinstance(y, theano.Variable):
        y = theano.tensor.as_tensor_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable:
        return add_s_s(x, y)
    elif x_is_sparse_variable and not y_is_sparse_variable:
        return add_s_d(x, y)
    elif y_is_sparse_variable and not x_is_sparse_variable:
        return add_s_d(y, x)
    else:
        raise NotImplementedError()


def sub(x, y):
    """
    Subtract two matrices, at least one of which is sparse.

    This method will provide the right op according
    to the inputs.

    Parameters
    ----------
    x
        A matrix variable.
    y
        A matrix variable.

    Returns
    -------
    A sparse matrix
        `x` - `y`

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    The grad will be structured only when one of the variable will be a dense
    matrix.

    """
    return x + (-y)


class MulSS(gof.op.Op):
    # mul(sparse, sparse)
    # See the doc of mul() for more detail
    __props__ = ()

    def make_node(self, x, y):
        x, y = as_sparse_variable(x), as_sparse_variable(y)
        assert x.format in ["csr", "csc"]
        assert y.format in ["csr", "csc"]
        out_dtype = scalar.upcast(x.type.dtype, y.type.dtype)
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=out_dtype,
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and _is_sparse(y)
        assert len(x.shape) == 2
        assert y.shape == x.shape
        # This calls the element-wise multiple
        # x * y calls dot...
        out[0] = x.multiply(y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        return y * gz, x * gz

    def infer_shape(self, node, shapes):
        return [shapes[0]]

mul_s_s = MulSS()


class MulSD(gof.op.Op):
    # mul(sparse, dense)
    # See the doc of mul() for more detail
    __props__ = ()

    def make_node(self, x, y):
        x, y = as_sparse_variable(x), tensor.as_tensor_variable(y)

        assert x.format in ["csr", "csc"]

        # upcast the tensor. Is the cast of sparse done implemented?
        dtype = scalar.upcast(x.type.dtype, y.type.dtype)

        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        # Broadcasting of the sparse matrix is not supported.
        # We support nd == 0 used by grad of SpSum()
        assert y.type.ndim in [0, 2]
        out = SparseType(dtype=dtype,
                         format=x.type.format)()
        return gof.Apply(self, [x, y], [out])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and _is_dense(y)
        if len(y.shape) == 0:
            out_dtype = node.outputs[0].dtype
            if x.dtype == out_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)
            out[0] = z
            out[0].data *= y
        elif len(y.shape) == 1:
            raise NotImplementedError()  # RowScale / ColScale
        elif len(y.shape) == 2:
            # if we have enough memory to fit y, maybe we can fit x.asarray()
            # too?
            # TODO: change runtime from O(M*N) to O(nonzeros)
            M, N = x.shape
            assert x.shape == y.shape
            out_dtype = node.outputs[0].dtype

            if x.format == 'csc':
                indices = x.indices
                indptr = x.indptr
                if x.dtype == out_dtype:
                    z = x.copy()
                else:
                    z = x.astype(out_dtype)
                z_data = z.data

                for j in xrange(0, N):
                    for i_idx in xrange(indptr[j], indptr[j + 1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i, j]
                out[0] = z
            elif x.format == 'csr':
                indices = x.indices
                indptr = x.indptr
                if x.dtype == out_dtype:
                    z = x.copy()
                else:
                    z = x.astype(out_dtype)
                z_data = z.data

                for i in xrange(0, M):
                    for j_idx in xrange(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i, j]
                out[0] = z
            else:
                print((
                    "WARNING: crappy implementation of MulSD"
                ), x.format, file=sys.stderr)
                out[0] = type(x)(x.toarray() * y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_sparse_variable(gz)
        return y * gz, dense_from_sparse(x * gz)

    def infer_shape(self, node, shapes):
        return [shapes[0]]
mul_s_d = MulSD()


class MulSV(gof.op.Op):

    __props__ = ()

    def make_node(self, x, y):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        y = tensor.as_tensor_variable(y)

        assert y.type.ndim == 1

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError(
                "MulSV not implemented for differing dtypes."
                "Got %s and %s." % (str(x.type.dtype), str(y.type.dtype)))
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=x.type.dtype,
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and not _is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x.toarray() * y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_sparse_variable(gz)

        # mul_s_v is not implemented if the types vary

        if gz.dtype == 'float64' and y.dtype == 'float32':
            y = y.astype('float64')

        if gz.dtype == 'float32' and y.dtype == 'float64':
            gz = gz.astype('float64')

        return mul_s_v(gz, y), sp_sum(x * gz, axis=0, sparse_grad=True)

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]

mul_s_v = MulSV()
"""
Multiplication of sparse matrix by a broadcasted dense vector element wise.

Parameters
----------
x
    Sparse matrix to multiply.
y
    Tensor broadcastable vector.

Returns
-------
A sparse matrix
    The product x * y element wise.

Notes
-----
The grad implemented is regular, i.e. not structured.

"""


def mul(x, y):
    """
    Multiply elementwise two matrices, at least one of which is sparse.

    This method will provide the right op according to the inputs.

    Parameters
    ----------
    x
        A matrix variable.
    y
        A matrix variable.

    Returns
    -------
    A sparse matrix
        `x` + `y`

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.
    The grad is regular, i.e. not structured.

    """

    x = as_sparse_or_tensor_variable(x)
    y = as_sparse_or_tensor_variable(y)

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    assert x_is_sparse_variable or y_is_sparse_variable
    if x_is_sparse_variable and y_is_sparse_variable:

        # mul_s_s is not implemented if the types differ
        if y.dtype == 'float64' and x.dtype == 'float32':
            x = x.astype('float64')

        return mul_s_s(x, y)
    elif x_is_sparse_variable and not y_is_sparse_variable:

        # mul is unimplemented if the dtypes differ
        if y.dtype == 'float64' and x.dtype == 'float32':
            x = x.astype('float64')

        return mul_s_d(x, y)
    elif y_is_sparse_variable and not x_is_sparse_variable:
        return mul_s_d(y, x)
    else:
        raise NotImplementedError()


class __ComparisonOpSS(gof.op.Op):
    """
    Used as a superclass for all comparisons between two sparses matrices.

    Parameters
    ----------
    x
        First compared sparse matrix.
    y
        Second compared sparse matrix

    Returns
    -------
    object
        Comparison(x,y)

    """

    __props__ = ()

    # Function to override
    def comparison(self, x, y):
        raise NotImplementedError()

    def make_node(self, x, y):
        x = as_sparse_variable(x)
        y = as_sparse_variable(y)

        if x.type.format != y.type.format:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype='uint8',
                                     format=x.type.format)()])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x) and _is_sparse(y)
        assert x.shape == y.shape
        out[0] = self.comparison(x, y).astype('uint8')

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]


class __ComparisonOpSD(gof.op.Op):
    """
    Used as a superclass for all comparisons between sparse and dense matrix.

    Parameters
    ----------
    x
        Sparse matrix.
    y
        Dense matrix.

    Returns
    -------
    object
        Comparison(x,y)

    """

    __props__ = ()

    # Function to override
    def comparison(self, x, y):
        raise NotImplementedError()

    def make_node(self, x, y):
        x, y = as_sparse_variable(x), tensor.as_tensor_variable(y)

        assert y.type.ndim == 2
        out = tensor.TensorType(dtype='uint8', broadcastable=(False, False))()
        return gof.Apply(self,
                         [x, y],
                         [out])

    def perform(self, node, inputs, outputs):
        (x, y) = inputs
        (out,) = outputs
        assert _is_sparse(x)
        assert x.shape == y.shape
        assert _is_dense(y)
        o = self.comparison(x, y).astype('uint8')
        o = np.asarray(o)
        out[0] = o

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]


def __ComparisonSwitch(SS, SD, DS):
    """

    Parameters
    ----------
    SS
        Function to apply between two sparses matrices.
    SD
        Function to apply between a sparse and a dense matrix.
    DS
        Function to apply between a dense and a sparse matrix.

    Returns
    -------
    function
        Switch function taking two matrices as input.

    Notes
    -----
    At least one of `x` and `y` must be a sparse matrix.

    DS swap input as a dense matrix cannot be a left operand.

    """

    def helper(x, y):

        scipy_ver = [int(n) for n in scipy.__version__.split('.')[:2]]

        assert scipy_ver >= [0, 13]

        if hasattr(x, 'getnnz'):
            x = as_sparse_variable(x)
        if hasattr(y, 'getnnz'):
            y = as_sparse_variable(y)
        if not isinstance(x, theano.Variable):
            x = theano.tensor.as_tensor_variable(x)
        if not isinstance(y, theano.Variable):
            y = theano.tensor.as_tensor_variable(y)

        x_is_sparse_variable = _is_sparse_variable(x)
        y_is_sparse_variable = _is_sparse_variable(y)

        assert x_is_sparse_variable or y_is_sparse_variable
        if x_is_sparse_variable and y_is_sparse_variable:
            return SS(x, y)
        elif x_is_sparse_variable and not y_is_sparse_variable:
            return SD(x, y)
        elif y_is_sparse_variable and not x_is_sparse_variable:
            return DS(y, x)
        else:
            raise NotImplementedError()

    return helper


class EqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x == y


equal_s_s = EqualSS()


class EqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x == y

equal_s_d = EqualSD()


class NotEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x != y

not_equal_s_s = NotEqualSS()


class NotEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x != y

not_equal_s_d = NotEqualSD()


class LessThanSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x < y

less_than_s_s = LessThanSS()


class LessThanSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x < y

less_than_s_d = LessThanSD()


class GreaterThanSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x > y

greater_than_s_s = GreaterThanSS()


class GreaterThanSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x > y

greater_than_s_d = GreaterThanSD()


class LessEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x <= y

less_equal_s_s = LessEqualSS()


class LessEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x <= y

less_equal_s_d = LessEqualSD()


class GreaterEqualSS(__ComparisonOpSS):
    def comparison(self, x, y):
        return x >= y

greater_equal_s_s = GreaterEqualSS()


class GreaterEqualSD(__ComparisonOpSD):
    def comparison(self, x, y):
        return x >= y

greater_equal_s_d = GreaterEqualSD()


eq = __ComparisonSwitch(equal_s_s, equal_s_d, equal_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` == `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""


neq = __ComparisonSwitch(not_equal_s_s, not_equal_s_d, not_equal_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` != `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""


lt = __ComparisonSwitch(less_than_s_s, less_than_s_d, greater_than_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` < `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""


gt = __ComparisonSwitch(greater_than_s_s, greater_than_s_d, less_than_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` > `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""

le = __ComparisonSwitch(less_equal_s_s, less_equal_s_d, greater_equal_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` <= `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""

ge = __ComparisonSwitch(greater_equal_s_s, greater_equal_s_d,
                        less_equal_s_d)
"""
Parameters
----------
x
    A matrix variable.
y
    A matrix variable.

Returns
-------
matrix variable
    `x` >= `y`

Notes
-----
At least one of `x` and `y` must be a sparse matrix.

"""


class HStack(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ("format", "dtype")

    def __init__(self, format=None, dtype=None):
        if format is None:
            self.format = 'csc'
        else:
            self.format = format

        if dtype is None:
            raise ValueError('The output dtype must be specified.')
        self.dtype = dtype

    def make_node(self, *mat):
        if not mat:
            raise ValueError('Cannot join an empty list of sparses.')
        var = [as_sparse_variable(x) for x in mat]

        for x in var:
            assert x.format in ["csr", "csc"]

        return gof.Apply(self,
                         var,
                         [SparseType(dtype=self.dtype,
                                     format=self.format)()])

    def perform(self, node, block, outputs):
        (out,) = outputs
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.hstack(block, format=self.format,
                                     dtype=self.dtype)
        # Some version of scipy (at least 0.14.0.dev-c4314b0)
        # Do not cast to the wanted dtype.
        if out[0].dtype != self.dtype:
            out[0] = out[0].astype(self.dtype)

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [(inputs[i].dtype in tensor.continuous_dtypes)
                         for i in range(len(inputs))]

        if _is_sparse_variable(gz):
            gz = dense_from_sparse(gz)

        split = tensor.Split(len(inputs))(gz, 1,
                                          tensor.stack(
                                              [x.shape[1]
                                               for x in inputs]))
        if not isinstance(split, list):
            split = [split]

        derivative = [SparseFromDense(self.format)(s) for s in split]

        def choose(continuous, derivative):
            if continuous:
                return derivative
            else:
                return None
        return [choose(c, d) for c, d in zip(is_continuous, derivative)]

    def infer_shape(self, node, ins_shapes):
        def _get(l):
            return l[1]
        d = sum(map(_get, ins_shapes))
        return [(ins_shapes[0][0], d)]

    def __str__(self):
        return "%s(%s,%s)" % (self.__class__.__name__, self.format, self.dtype)


def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally (column wise).

    This wrap the method hstack from scipy.

    Parameters
    ----------
    blocks
        List of sparse array of compatible shape.
    format
        String representing the output format. Default is csc.
    dtype
        Output dtype.

    Returns
    -------
    array
        The concatenation of the sparse array column wise.

    Notes
    -----
    The number of line of the sparse matrix must agree.

    The grad implemented is regular, i.e. not structured.

    """

    blocks = [as_sparse_variable(i) for i in blocks]
    if dtype is None:
        dtype = theano.scalar.upcast(*[i.dtype for i in blocks])
    return HStack(format=format, dtype=dtype)(*blocks)


class VStack(HStack):
    # See doc in instance of this Op or function after this class definition.
    def perform(self, node, block, outputs):
        (out,) = outputs
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.vstack(block, format=self.format,
                                     dtype=self.dtype)
        # Some version of scipy (at least 0.14.0.dev-c4314b0)
        # Do not cast to the wanted dtype.
        if out[0].dtype != self.dtype:
            out[0] = out[0].astype(self.dtype)

    def grad(self, inputs, gout):
        (gz,) = gout
        is_continuous = [(inputs[i].dtype in tensor.continuous_dtypes)
                         for i in range(len(inputs))]

        if _is_sparse_variable(gz):
            gz = dense_from_sparse(gz)

        split = tensor.Split(len(inputs))(gz, 0,
                                          tensor.stack(
                                              [x.shape[0]
                                               for x in inputs]))
        if not isinstance(split, list):
            split = [split]

        derivative = [SparseFromDense(self.format)(s) for s in split]

        def choose(continuous, derivative):
            if continuous:
                return derivative
            else:
                return None
        return [choose(c, d) for c, d in zip(is_continuous, derivative)]

    def infer_shape(self, node, ins_shapes):
        def _get(l):
            return l[0]
        d = sum(map(_get, ins_shapes))
        return [(d, ins_shapes[0][1])]


def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices vertically (row wise).

    This wrap the method vstack from scipy.

    Parameters
    ----------
    blocks
        List of sparse array of compatible shape.
    format
        String representing the output format. Default is csc.
    dtype
        Output dtype.

    Returns
    -------
    array
        The concatenation of the sparse array row wise.

    Notes
    -----
    The number of column of the sparse matrix must agree.

    The grad implemented is regular, i.e. not structured.

    """

    blocks = [as_sparse_variable(i) for i in blocks]
    if dtype is None:
        dtype = theano.scalar.upcast(*[i.dtype for i in blocks])
    return VStack(format=format, dtype=dtype)(*blocks)


class Remove0(gof.Op):
    # See doc in instance of this Op or a function after the class definition.
    __props__ = ("inplace",)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        l = []
        if self.inplace:
            l.append('inplace')
        return self.__class__.__name__ + '{%s}' % ', '.join(l)

    def make_node(self, x):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        if self.inplace:
            c = x
        else:
            c = x.copy()
        c.eliminate_zeros()
        z[0] = c

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [gz]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes
remove0 = Remove0()
"""
Remove explicit zeros from a sparse matrix.

Parameters
----------
x
    Sparse matrix.

Returns
-------
sparse matrix
    Exactly `x` but with a data attribute exempt of zeros.

Notes
-----
The grad implemented is regular, i.e. not structured.

"""


# Structured monoid
def structured_monoid(tensor_op):
    # Generic operation to perform many kinds of monoid element-wise
    # operations on the non-zeros of a sparse matrix.

    # The first parameter must always be a sparse matrix. The other parameters
    # must be scalars which will be passed as argument to the tensor_op.

    def decorator(f):
        def wrapper(*args):
            x = as_sparse_variable(args[0])
            assert x.format in ["csr", "csc"]

            xs = [scalar.as_scalar(arg) for arg in args[1:]]

            data, ind, ptr, shape = csm_properties(x)

            data = tensor_op(data, *xs)

            return CSM(x.format)(data, ind, ptr, shape)
        wrapper.__name__ = str(tensor_op.scalar_op)
        return wrapper
    return decorator


@structured_monoid(tensor.nnet.sigmoid)
def structured_sigmoid(x):
    """
    Structured elemwise sigmoid.

    """
    # see decorator for function body


@structured_monoid(tensor.exp)
def structured_exp(x):
    """
    Structured elemwise exponential.

    """
    # see decorator for function body


@structured_monoid(tensor.log)
def structured_log(x):
    """
    Structured elemwise logarithm.

    """
    # see decorator for function body


@structured_monoid(tensor.pow)
def structured_pow(x, y):
    """
    Structured elemwise power of sparse matrix x by scalar y.

    """
    # see decorator for function body


@structured_monoid(tensor.minimum)
def structured_minimum(x, y):
    """
    Structured elemwise minimum of sparse matrix x by scalar y.

    """
    # see decorator for function body


@structured_monoid(tensor.maximum)
def structured_maximum(x, y):
    """
    Structured elemwise maximum of sparse matrix x by scalar y.

    """
    # see decorator for function body


@structured_monoid(tensor.add)
def structured_add(x):
    """
    Structured addition of sparse matrix x and scalar y.

    """
    # see decorator for function body


# Sparse operation (map 0 to 0)
@structured_monoid(tensor.sin)
def sin(x):
    """
    Elemwise sinus of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.tan)
def tan(x):
    """
    Elemwise tan of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.arcsin)
def arcsin(x):
    """
    Elemwise arcsinus of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.arctan)
def arctan(x):
    """
    Elemwise arctan of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.sinh)
def sinh(x):
    """
    Elemwise sinh of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.arcsinh)
def arcsinh(x):
    """
    Elemwise arcsinh of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.tanh)
def tanh(x):
    """
    Elemwise tanh of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.arctanh)
def arctanh(x):
    """
    Elemwise arctanh of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.round_half_to_even)
def rint(x):
    """
    Elemwise round half to even of `x`.

   """
    # see decorator for function body

# Give it a simple name instead of the complex one that would automatically
# be derived from `tensor.round_half_to_even`.
rint.__name__ = 'rint'


@structured_monoid(tensor.sgn)
def sgn(x):
    """
    Elemwise signe of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.ceil)
def ceil(x):
    """
    Elemwise ceiling of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.floor)
def floor(x):
    """
    Elemwise floor of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.log1p)
def log1p(x):
    """
    Elemwise log(1 + `x`).

    """
    # see decorator for function body


@structured_monoid(tensor.expm1)
def expm1(x):
    """
    Elemwise e^`x` - 1.

    """
    # see decorator for function body


@structured_monoid(tensor.deg2rad)
def deg2rad(x):
    """
    Elemwise degree to radian.

    """
    # see decorator for function body


@structured_monoid(tensor.rad2deg)
def rad2deg(x):
    """
    Elemwise radian to degree.

    """
    # see decorator for function body


@structured_monoid(tensor.trunc)
def trunc(x):
    """
    Elemwise truncature.

    """
    # see decorator for function body


@structured_monoid(tensor.sqr)
def sqr(x):
    """
    Elemwise `x` * `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.sqrt)
def sqrt(x):
    """
    Elemwise square root of `x`.

    """
    # see decorator for function body


@structured_monoid(tensor.conj)
def conj(x):
    """
    Elemwise complex conjugate of `x`.

    """
    # see decorator for function body


class TrueDot(gof.op.Op):

    # TODO
    # Simplify code by splitting into DotSS and DotSD.

    __props__ = ()
    # The grad_preserves_dense attribute doesn't change the
    # execution behavior.  To let the optimizer merge nodes with
    # different values of this attribute we shouldn't compare it
    # here.

    def __init__(self, grad_preserves_dense=True):
        self.grad_preserves_dense = grad_preserves_dense

    def make_node(self, x, y):
        # NOTE
        # Because of trickiness of implementing,
        # we assume that the left argument x is a
        # SparseVariable (not dense)

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()

        if not _is_sparse_variable(x):
            raise TypeError(x)

        # These are the conversions performed by scipy.sparse.dot
        if x.type.format == "csc" or x.type.format == "coo":
            myformat = "csc"
        elif x.type.format == "csr":
            myformat = "csr"
        else:
            raise NotImplementedError()

        inputs = [x, y]  # Need to convert? e.g. assparse
        outputs = [SparseType(dtype=x.type.dtype, format=myformat)()]
        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inp, out_):
        # TODO
        # -Verify that output is sufficiently sparse,
        #  and raise a warning if it is not.
        # -Also determine that we are storing the
        #  output in the best storage format?

        x, y = inp
        out, = out_
        rval = x.dot(y)
        if not scipy.sparse.issparse(rval):
            rval = getattr(scipy.sparse, x.format + '_matrix')(rval)
        # x.dot call tocsr() that will "upcast" to ['int8', 'uint8', 'short',
        # 'ushort', 'intc', 'uintc', 'longlong', 'ulonglong', 'single',
        # 'double', 'longdouble', 'csingle', 'cdouble', 'clongdouble']
        # But ulonglong is uint64 on x86-64, but with a different typenum!
        if rval.dtype.num != np.dtype(str(rval.dtype)).num:
            assert str(rval.dtype) == node.outputs[0].dtype
            # Create a view with the expected typenum.
            format = node.outputs[0].type.format
            data = rval.data.view(dtype=node.outputs[0].dtype)
            indices = rval.indices
            indptr = rval.indptr
            shape = rval.shape
            # No need to copy indices and indptr as in CSM.perform(),
            # as there is only one user of them.
            if format == 'csc':
                rval = scipy.sparse.csc_matrix((data, indices, indptr),
                                               shape, copy=False)
            else:
                assert format == 'csr'
                rval = scipy.sparse.csr_matrix((data, indices, indptr),
                                               shape, copy=False)
        out[0] = rval

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        assert _is_sparse_variable(gz)
        assert _is_sparse_variable(x)

        rval = [true_dot(gz, y.T), true_dot(x.T, gz)]
        if _is_dense_variable(y):
            if self.grad_preserves_dense:
                rval[1] = dense_from_sparse(rval[1])
        return rval

    def infer_shape(self, node, shapes):
        return [(shapes[0][0], shapes[1][1])]


def true_dot(x, y, grad_preserves_dense=True):
    """
    Operation for efficiently calculating the dot product when
    one or all operands are sparse. Supported formats are CSC and CSR.
    The output of the operation is sparse.

    Parameters
    ----------
    x
        Sparse matrix.
    y
        Sparse matrix or 2d tensor variable.
    grad_preserves_dense : bool
        If True (default), makes the grad of dense inputs dense.
        Otherwise the grad is always sparse.

    Returns
    -------
    The dot product `x`.`y` in a sparse format.

    Notex
    -----
    The grad implemented is regular, i.e. not structured.

    """
    # TODO
    # Maybe the triple-transposition formulation
    # (when x is dense) is slow. See if there is a
    # direct way to do this.

    if hasattr(x, 'getnnz'):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
    if hasattr(y, 'getnnz'):
        y = as_sparse_variable(y)
        assert y.format in ["csr", "csc"]

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)

    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError()
    if x_is_sparse_variable:
        return TrueDot(grad_preserves_dense)(x, y)
    else:
        assert y_is_sparse_variable
        return transpose(TrueDot(grad_preserves_dense)(y.T, x.T))


# Dot
class StructuredDot(gof.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def make_node(self, a, b):

        a = as_sparse_variable(a)
        assert a.format in ["csr", "csc", "bsr"]

        if not _is_sparse_variable(a):
            raise TypeError('First argument must be of type SparseVariable '
                            'or SparseConstant')
        dtype_out = scalar.upcast(a.type.dtype, b.type.dtype)
        if b.type.ndim != 2:
            raise NotImplementedError('non-matrix b')

        if _is_sparse_variable(b):
            return gof.Apply(self, [a, b],
                             [SparseType(a.type.format, dtype_out)()])
        else:
            return gof.Apply(self, [a, b],
                             [tensor.tensor(dtype_out,
                                            (False, b.type.broadcastable[1]))])

    def perform(self, node, inputs, outputs):
        (a, b) = inputs
        (out,) = outputs
        if a.shape[1] != b.shape[0]:
            raise ValueError('shape mismatch in StructuredDot.perform',
                             (a.shape, b.shape))

        variable = a * b
        if isinstance(node.outputs[0].type, SparseType):
            assert _is_sparse(variable)
            out[0] = variable
            return

        assert _is_dense(variable)  # scipy 0.7 automatically converts to dense

        # dot of an NxM sparse matrix, with a Mx1 dense matrix, returns vector
        # not matrix
        if variable.ndim == 1:
            variable = np.expand_dims(variable, 1)
        elif variable.ndim != 2:
            raise Exception('Output of structured dot should be a matrix '
                            '(ndim=2)')

        assert variable.ndim == 2

        if variable.shape != (a.shape[0], b.shape[1]):
            if b.shape[0] == 1:
                raise Exception("a.shape=%s, b.shape=%s, "
                                "variable.shape=%s ??? This is probably "
                                "because scipy.csc_matrix dot has a bug "
                                "with singleton dimensions (i.e. "
                                "b.shape[0]=1), for scipy 0.6. Use scipy "
                                "0.7. NB you have scipy version %s" %
                                (a.shape, b.shape, variable.shape,
                                 scipy.__version__))
            else:
                raise Exception("a.shape=%s, b.shape=%s, variable.shape=%s "
                                " ??? I have no idea why")

        # The cast is needed as otherwise we hit the bug mentioned into
        # theano._asarray function documentation.
        out[0] = theano._asarray(variable, str(variable.dtype))

    def grad(self, inputs, gout):
        # a is sparse, b is dense, g_out is dense
        # ga = g_out x b.T
        # gb = a.T x g_out
        (a, b) = inputs
        (g_out,) = gout
        return [structured_dot_grad(a, b, g_out), structured_dot(a.T, g_out)]

    def infer_shape(self, node, shapes):
        return [(shapes[0][0], shapes[1][1])]

_structured_dot = StructuredDot()


def structured_dot(x, y):
    """
    Structured Dot is like dot, except that only the
    gradient wrt non-zero elements of the sparse matrix
    `a` are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a
    TensorType instance.

    Parameters
    ----------
    a
        A sparse matrix.
    b
        A sparse or dense matrix.

    Returns
    -------
    A sparse matrix
        The dot product of `a` and `b`.

    Notes
    -----
    The grad implemented is structured.

    """

    # @todo: Maybe the triple-transposition formulation (when x is dense)
    # is slow. See if there is a direct way to do this.
    # (JB 20090528: Transposing tensors and sparse matrices is constant-time,
    # inplace, and fast.)

    if hasattr(x, 'getnnz'):
        x = as_sparse_variable(x)
        assert x.format in ["csr", "csc"]
    if hasattr(y, 'getnnz'):
        y = as_sparse_variable(y)
        assert y.format in ["csr", "csc"]

    x_is_sparse_variable = _is_sparse_variable(x)
    y_is_sparse_variable = _is_sparse_variable(y)
    if not x_is_sparse_variable and not y_is_sparse_variable:
        raise TypeError('structured_dot requires at least one sparse argument')

    if x_is_sparse_variable:
        return _structured_dot(x, y)
    else:
        assert y_is_sparse_variable
        return _structured_dot(y.T, x.T).T


class StructuredDotGradCSC(gof.Op):
    # Op that produces the grad of StructuredDot.

    # :param a_indices: Matrix indicies
    # :param a_indptr: Matrix indptr
    # :param b: Right operand
    # :param g_ab: Accumulated gradient.

    # :return: The grad of `a`.`b` for `a` accumulated
    #          with g_ab.

    # :note: The grad implemented is structured.
    # :note: a_* are the corresponding properties of a sparse
    #        matrix in csc format.
    __props__ = ()

    def make_node(self, a_indices, a_indptr, b, g_ab):
        return gof.Apply(self, [a_indices, a_indptr, b, g_ab],
                               [tensor.tensor(g_ab.dtype, (False,))])

    def perform(self, node, inputs, outputs):
        (a_indices, a_indptr, b, g_ab) = inputs
        (out,) = outputs
        g_a_data = np.zeros(a_indices.shape, dtype=g_ab.dtype)
        for j in xrange(len(a_indptr) - 1):
            ind0 = a_indptr[j]
            ind1 = a_indptr[j + 1]
            for i_idx in xrange(ind0, ind1):
                i = a_indices[i_idx]
                # Depending on the type of g_ab and b (sparse or dense),
                # the following dot product can result in a scalar or
                # a (1, 1) sparse matrix.
                dot_val = np.dot(g_ab[i], b[j].T)
                if isinstance(dot_val, scipy.sparse.spmatrix):
                    dot_val = dot_val[0, 0]
                g_a_data[i_idx] = dot_val
        out[0] = g_a_data

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):

        (_indices, _indptr, _d, _g) = inputs
        (_zout,) = outputs
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for '
                                      'g_ab')

        return """
        if (PyArray_NDIM(%(_d)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_g)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_indices)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(_indptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if( PyArray_DIMS(%(_d)s)[1] != PyArray_DIMS(%(_g)s)[1])
        {PyErr_SetString(PyExc_NotImplementedError, "d and g have different numbers of columns"); %(fail)s;}

        if (!%(_zout)s
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_g)s));
        }

        {   //makes it compile even though labels jump over variable definitions.
            npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1; //TODO: error checking with this

            npy_intp Sindices = PyArray_STRIDES(%(_indices)s)[0]/PyArray_DESCR(%(_indices)s)->elsize;
            npy_intp Sindptr = PyArray_STRIDES(%(_indptr)s)[0]/PyArray_DESCR(%(_indptr)s)->elsize;

            const npy_intp Sd1 = PyArray_STRIDES(%(_d)s)[1]/PyArray_DESCR(%(_d)s)->elsize;
            const npy_intp Sg1 = PyArray_STRIDES(%(_g)s)[1]/PyArray_DESCR(%(_g)s)->elsize;

            const npy_intp K = PyArray_DIMS(%(_d)s)[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {
                // extract j-th row of dense matrix
                const dtype_%(_d)s* __restrict__ d_row = (dtype_%(_d)s*)(PyArray_BYTES(%(_d)s) + PyArray_STRIDES(%(_d)s)[0] * j);
                if(j >= PyArray_DIMS(%(_d)s)[0]) {PyErr_SetString(PyExc_NotImplementedError, "G"); %(fail)s;}

                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j * Sindptr]; i_idx < indptr[(j+1) * Sindptr]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx * Sindices];

                    // extract corresponding row in gradient
                    const dtype_%(_g)s* __restrict__ g_row = (dtype_%(_g)s*)(PyArray_BYTES(%(_g)s) + PyArray_STRIDES(%(_g)s)[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= PyArray_DIMS(%(_g)s)[0])
                    {PyErr_SetString(PyExc_NotImplementedError, "H"); %(fail)s;}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }

                    // write resulting gradient to sparse output
                    ((dtype_%(_zout)s* __restrict__)(PyArray_BYTES(%(_zout)s) + i_idx * PyArray_STRIDES(%(_zout)s)[0]))[0] = ip;
                }
            }
        }

        """ % dict(locals(), **sub)

    def infer_shape(self, node, shapes):
        return [shapes[0]]
sdg_csc = StructuredDotGradCSC()


class StructuredDotGradCSR(gof.Op):
    # Op that produces the grad of StructuredDot.

    # :param a_indices: Matrix indicies
    # :param a_indptr: Matrix indptr
    # :param b: Right operand
    # :param g_ab: Accumulated gradient.

    # :return: The grad of `a`.`b` for `a` accumulated
    #          with g_ab.

    # :note: The grad implemented is structured.
    # :note: a_* are the corresponding properties of a sparse
    #        matrix in csr format.
    __props__ = ()

    def make_node(self, a_indices, a_indptr, b, g_ab):
        return gof.Apply(self, [a_indices, a_indptr, b, g_ab],
                         [tensor.tensor(b.dtype, (False,))])

    def perform(self, node, inputs, outputs):
        (a_indices, a_indptr, b, g_ab) = inputs
        (out,) = outputs
        g_a_data = np.zeros(a_indices.shape, dtype=g_ab.dtype)
        for i in xrange(len(a_indptr) - 1):  # loop over rows
            ind0 = a_indptr[i]
            ind1 = a_indptr[i + 1]
            # loop over values in that row (columns)
            for j_idx in xrange(ind0, ind1):
                j = a_indices[j_idx]
                # grad is dot product of i-th row of gradient with j-th row of b
                # Depending on the type of g_ab and b (sparse or dense),
                # the following dot product can result in a scalar or
                # a (1, 1) sparse matrix.
                dot_val = np.dot(g_ab[i], b[j].T)
                if isinstance(dot_val, scipy.sparse.spmatrix):
                    dot_val = dot_val[0, 0]
                g_a_data[j_idx] = dot_val
        out[0] = g_a_data

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):

        (_indices, _indptr, _d, _g) = inputs
        (_zout,) = outputs
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for '
                                      'g_ab')

        return """
        if (PyArray_NDIM(%(_d)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(d) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_g)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(g) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_indices)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(_indptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if( PyArray_DIMS(%(_d)s)[1] != PyArray_DIMS(%(_g)s)[1])
        {PyErr_SetString(PyExc_NotImplementedError, "d and g have different numbers of columns"); %(fail)s;}

        if (!%(_zout)s
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_g)s));
        }

        {   //makes it compile even though labels jump over variable definitions.
            npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            // extract number of rows
            npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1; //TODO: error checking with this

            npy_intp Sindices = PyArray_STRIDES(%(_indices)s)[0]/PyArray_DESCR(%(_indices)s)->elsize;
            npy_intp Sindptr = PyArray_STRIDES(%(_indptr)s)[0]/PyArray_DESCR(%(_indptr)s)->elsize;

            const npy_intp Sd1 = PyArray_STRIDES(%(_d)s)[1]/PyArray_DESCR(%(_d)s)->elsize;
            const npy_intp Sg1 = PyArray_STRIDES(%(_g)s)[1]/PyArray_DESCR(%(_g)s)->elsize;

            const npy_intp K = PyArray_DIMS(%(_d)s)[1];

            const npy_int32 * __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            // loop over columns of sparse matrix
            for (npy_int32 i = 0; i < N; ++i)
            {
                // for each non-null value in the sparse row
                for (npy_int32 j_idx = indptr[i * Sindptr]; j_idx < indptr[(i+1) * Sindptr]; ++j_idx)
                {
                    // extract column index of non-null value
                    npy_int32 j = indices[j_idx * Sindices];

                    // extract j-th row of dense matrix
                    const dtype_%(_d)s* __restrict__ d_row = (dtype_%(_d)s*)(PyArray_BYTES(%(_d)s) + PyArray_STRIDES(%(_d)s)[0] * j);
                    if(j >= PyArray_DIMS(%(_d)s)[0]) {PyErr_SetString(PyExc_NotImplementedError, "G"); %(fail)s;}

                    // extract corresponding row in gradient
                    const dtype_%(_g)s* __restrict__ g_row = (dtype_%(_g)s*)(PyArray_BYTES(%(_g)s) + PyArray_STRIDES(%(_g)s)[0] * i);
                    double ip = 0.0;

                    // make sure that row index is not bigger than actual number of rows
                    // Note: wouldn't the above operation fail if that were the case ?
                    //       when would this ever be true anyway ?
                    if (i >= PyArray_DIMS(%(_g)s)[0])
                    {PyErr_SetString(PyExc_NotImplementedError, "H"); %(fail)s;}

                    // perform dot product of dense and sparse rows
                    for(int k = 0; k < K; ++k)
                    {
                        ip += d_row[k * Sd1] * g_row[k*Sg1];
                    }

                    // write resulting gradient to sparse output
                    ((dtype_%(_zout)s* __restrict__)(PyArray_BYTES(%(_zout)s) + j_idx * PyArray_STRIDES(%(_zout)s)[0]))[0] = ip;
                }
            }
        }

        """ % dict(locals(), **sub)

    def infer_shape(self, node, shapes):
        return [shapes[0]]
sdg_csr = StructuredDotGradCSR()


def structured_dot_grad(sparse_A, dense_B, ga):
    if sparse_A.type.format in ('csc', 'csr'):

        if sparse_A.type.format == 'csc':
            sdgcsx = sdg_csc
            CSx = CSC
        else:
            sdgcsx = sdg_csr
            CSx = CSR

        g_A_data = sdgcsx(csm_indices(sparse_A),
                          csm_indptr(sparse_A), dense_B, ga)
        return CSx(g_A_data, csm_indices(sparse_A),
                   csm_indptr(sparse_A), csm_shape(sparse_A))
    else:
        raise NotImplementedError()


class SamplingDot(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def make_node(self, x, y, p):
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        p = as_sparse_variable(p)
        assert p.format in ["csr", "csc"]

        if not _is_sparse_variable(p):
            raise TypeError(p)

        # TODO: use it.
        dtype_out = scalar.upcast(x.type.dtype, y.type.dtype, p.type.dtype)  # noqa

        return gof.Apply(self, [x, y, p], [p.type()])

    def perform(self, node, inputs, outputs):
        (x, y, p) = inputs
        (out,) = outputs
        if _is_sparse(x):
            raise TypeError(x)

        if _is_sparse(y):
            raise TypeError(y)

        if not _is_sparse(p):
            raise TypeError(p)

        out[0] = p.__class__(p.multiply(np.dot(x, y.T)))

    def grad(self, inputs, gout):
        (x, y, p) = inputs
        (gz,) = gout
        rval = [
            dot(p * gz, y),
            dot((p * gz).T, x),
            grad_not_implemented(self, 2, p)
        ]

        return rval

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[2]]

sampling_dot = SamplingDot()
"""
Operand for calculating the dot product dot(`x`, `y`.T) = `z` when you
only want to calculate a subset of `z`.

It is equivalent to `p` o (`x` . `y`.T) where o is the element-wise
product, `x` and `y` operands of the dot product and `p` is a matrix that
contains 1 when the corresponding element of `z` should be calculated
and 0 when it shouldn't. Note that SamplingDot has a different interface
than `dot` because SamplingDot requires `x` to be a `m`x`k` matrix while
`y` is a `n`x`k` matrix instead of the usual `k`x`n` matrix.

Notes
-----
It will work if the pattern is not binary value, but if the
pattern doesn't have a high sparsity proportion it will be slower
then a more optimized dot followed by a normal elemwise
multiplication.

The grad implemented is regular, i.e. not structured.

Parameters
----------
x
    Tensor matrix.
y
    Tensor matrix.
p
    Sparse matrix in csr format.

Returns
-------
sparse matrix
    A dense matrix containing the dot product of `x` by `y`.T only
    where `p` is 1.

"""


class Dot(gof.op.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def __str__(self):
        return "Sparse" + self.__class__.__name__

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
        dtype_out = scalar.upcast(x.dtype, y.dtype)

        # Sparse dot product should have at least one sparse variable
        # as input. If the other one is not sparse, it has to be converted
        # into a tensor.
        if isinstance(x, scipy.sparse.spmatrix):
            x = as_sparse_variable(x)
        if isinstance(y, scipy.sparse.spmatrix):
            y = as_sparse_variable(y)
        x_is_sparse_var = _is_sparse_variable(x)
        y_is_sparse_var = _is_sparse_variable(y)

        if not x_is_sparse_var and not y_is_sparse_var:
            raise TypeError(
                "Sparse dot product should have at least one "
                "sparse variable as inputs, but the inputs are "
                "%s (%s) and %s (%s)." % (x, x.type, y, y.type))

        if not x_is_sparse_var:
            x = tensor.as_tensor_variable(x)
            assert y.format in ["csr", "csc"]
            if x.ndim not in (1, 2):
                raise TypeError(
                    'theano.sparse.Dot: input 0 (0-indexed) must have ndim of '
                    '1 or 2, %d given.' % x.ndim)

        if not y_is_sparse_var:
            y = tensor.as_tensor_variable(y)
            assert x.format in ["csr", "csc"]
            if y.ndim not in (1, 2):
                raise TypeError(
                    'theano.sparse.Dot: input 1 (1-indexed) must have ndim of '
                    '1 or 2, %d given.' % y.ndim)

        if y.ndim == 1 or x.ndim == 1:
            bz = (False,)
        else:
            bz = (False, False)
        return gof.Apply(self, [x, y], [tensor.tensor(dtype=dtype_out,
                                                      broadcastable=bz)])

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

        out[0] = theano._asarray(rval, dtype=node.outputs[0].dtype)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
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

    Parameters
    ----------
    x
        Sparse or dense matrix variable.
    y
        Sparse or dense matrix variable.

    Returns
    -------
    The dot product `x`.`y` in a dense format.

    Notes
    -----
    The grad implemented is regular, i.e. not structured.

    At least one of `x` or `y` must be a sparse matrix.

    When the operation has the form dot(csr_matrix, dense)
    the gradient of this operation can be performed inplace
    by UsmmCscDense. This leads to significant speed-ups.

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
    # See doc in instance of this Op or function after this class definition.
    # We don't implement the infer_shape as it is
    # inserted by optimization only.
    __props__ = ()

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
            assert y.format in ["csr", "csc"]
            assert x.ndim == 2
        if not _is_sparse_variable(y):
            y = tensor.as_tensor_variable(y)
            assert x.format in ["csr", "csc"]
            assert y.ndim == 2

        return gof.Apply(self, [alpha, x, y, z],
                         [tensor.tensor(dtype=dtype_out,
                                        broadcastable=(False, False))])

    def perform(self, node, inputs, outputs):
        (alpha, x, y, z) = inputs
        (out,) = outputs
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
"""
Performs the expression `alpha` * `x` `y` + `z`.

Parameters
----------
x
    Matrix variable.
y
    Matrix variable.
z
    Dense matrix.
alpha
    A tensor scalar.

Returns
-------
The dense matrix resulting from `alpha` * `x` `y` + `z`.

Notes
-----
The grad is not implemented for this op.
At least one of `x` or `y` must be a sparse matrix.

"""


class ConstructSparseFromList(gof.Op):
    # See doc in instance of this Op or function after this class definition.
    __props__ = ()

    def make_node(self, x, values, ilist):
        """

        Parameters
        ----------
        x
            A dense matrix that specify the output shape.
        values
            A dense matrix with the values to use for output.
        ilist
            A dense vector with the same length as the number of rows of values.
            It specify where in the output to put the corresponding rows.

        This create a sparse matrix with the same shape as `x`. Its
        values are the rows of `values` moved. Pseudo-code::

            output = csc_matrix.zeros_like(x, dtype=values.dtype)
            for in_idx, out_idx in enumerate(ilist):
                output[out_idx] = values[in_idx]

        """
        x_ = theano.tensor.as_tensor_variable(x)
        values_ = theano.tensor.as_tensor_variable(values)
        ilist_ = theano.tensor.as_tensor_variable(ilist)

        if ilist_.type.dtype not in integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim != 2:
            raise TypeError(
                'cannot create a sparse matrix with %d dimensions' %
                x_.type.ndim)
        if values_.type.ndim != 2:
            raise TypeError(
                'cannot create a sparse matrix from values with %d ndim' %
                values_.type.ndim)

        # We only need the shape of `x` in the perform
        # If we keep in the graph the x variable as input of the Apply node,
        # this can rise the memory usage. That is why the Apply node
        # take `x_.shape` as input and not `x`.
        return gof.Apply(self, [x_.shape, values_, ilist_],
                         [csc_matrix(dtype=x.dtype)])

    def perform(self, node, inp, out_):
        out_shape, values, ilist = inp
        out, = out_
        rows, cols = values.shape
        assert rows == len(ilist)
        indptr = np.arange(cols + 1) * rows
        indices = as_strided(ilist,
                             strides=(0, ilist.strides[0]),
                             shape=(cols, ilist.shape[0])).flatten()
        data = values.T.flatten()
        out[0] = scipy.sparse.csc_matrix((data, indices, indptr),
                                         shape=out_shape,
                                         dtype=values.dtype)

    def infer_shape(self, node, ishapes):
        x = node.inputs[0]
        return [[x[0], x[1]]]

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1],
                              *inputs[2:]).outputs

    def connection_pattern(self, node):

        rval = [[True], [True], [False]]
        return rval

    def grad(self, inputs, grads):
        g_output, = grads
        x, y = inputs[:2]
        idx_list = inputs[2:]

        gx = g_output
        gy = theano.tensor.advanced_subtensor1(g_output, *idx_list)

        return [gx, gy] + [DisconnectedType()()] * len(idx_list)

construct_sparse_from_list = ConstructSparseFromList()
"""
Constructs a sparse matrix out of a list of 2-D matrix rows.

Notes
-----
The grad implemented is regular, i.e. not structured.

"""
