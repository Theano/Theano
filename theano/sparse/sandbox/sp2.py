import theano
import numpy
import scipy.sparse

from theano import gof, tensor, scalar, sparse
from theano.tensor import blas

from theano.sparse.basic import (
    as_sparse_variable, SparseType, add_s_s, neg,
    mul_s_s, mul_s_d, dot,
    CSMProperties, CSM, register_specialize,
    _is_sparse_variable, _is_dense_variable, CSC, CSR,
    csm_properties, csm_data, csm_indices, csm_indptr, csm_shape,
    _is_sparse, Remove0, remove0)
from theano.sparse.sandbox.sp import sp_sum


EliminateZeros = Remove0
eliminate_zeros = remove0


class Cast(gof.op.Op):
    """Cast sparse variable to the desired dtype.

    This wrap the method astype from scipy.
    """
    # It returns a new matrix, not a view.

    def __init__(self, out_type):
        self.out_type = out_type

    def __eq__(self, other):
        return (type(self) == type(other)) and self.out_type == other.out_type

    def __hash__(self):
        return hash(type(self)) ^ hash(self.out_type)

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(
            self, [x],
            [SparseType(dtype=self.out_type, format=x.format).make_variable()])

    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = x.astype(self.out_type)

    def grad(self, inputs, outputs_gradients):
        if inputs[0].dtype in tensor.continuous_dtypes:
            gz = outputs_gradients[0]
            return [Cast(inputs[0].dtype)(gz)]
        else:
            return [None]

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


class HStack(gof.op.Op):
    """Stack sparse matrices horizontally (column wise).

    This wrap the method hstack from scipy.

    :Parameters:
    - `blocks`: Sequence of sparse array of compatible shape
    - `format`: String representing the output format
    - `dtype`: Output dtype

    :return: the concatenation of the sparse arrays column wise.

    The number of line of the sparse matrix must agree.
    """

    def __init__(self, format=None, dtype=None):
        if format is None:
            self.format = 'csc'
        else:
            self.format = format
        if dtype is None:
            self.dtype = theano.config.floatX
        else:
            self.dtype = dtype

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.format == other.format and
                self.dtype == other.dtype)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.format) ^ hash(self.dtype)

    def make_node(self, *mat):
        if not mat:
            raise ValueError('Cannot join an empty list of sparses.')
        var = [as_sparse_variable(x) for x in mat]
        return gof.Apply(
            self, var,
            [SparseType(dtype=self.dtype, format=self.format).make_variable()])

    def perform(self, node, block, (out, )):
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.hstack(block, format=self.format,
                                     dtype=self.dtype)

    def grad(self, inputs, (gz, )):
        is_continuous = [(inputs[i].dtype in tensor.continuous_dtypes)
                        for i in range(len(inputs))]

        if all(is_continuous):
            if _is_sparse_variable(gz):
                gz = sparse.DenseFromSparse()(gz)

            split = tensor.Split(len(inputs))(gz, 1,
                                              tensor.stack(
                                                  *[x.shape[1]
                                                    for x in inputs]))
            if not isinstance(split, list):
                split = [split]
            return [sparse.SparseFromDense(self.format)(s) for s in split]
        else:
            return [None] * len(inputs)

    def infer_shape(self, node, ins_shapes):
        def _get(l):
            return l[1]
        d = sum(map(_get, ins_shapes))
        return [(ins_shapes[0][0], d)]

    def __str__(self):
        return "%s(%s,%s)" % (self.__class__.__name__, self.format, self.dtype)


def hstack(blocks, format=None, dtype=None):
    """Stack sparse matrices horizontally (column wise).

    This wrap the method hstack from scipy.

    :Parameters:
    - `blocks`: Sequence of sparse array of compatible shape
    - `format`: String representing the output format
    - `dtype`: Output dtype

    :return: the concatenation of the sparse array column wise.

    The number of line of the sparse matrix must agree.
    """
    return HStack(format=format, dtype=dtype)(*blocks)


class VStack(HStack):
    """Stack sparse matrices vertically (row wise).

    This wrap the method vstack from scipy.

    :Parameters:
    - `blocks`: Sequence of sparse array of compatible shape
    - `format`: String representing the output format
    - `dtype`: Output dtype

    :return: the concatenation of the sparse arrays row wise.

    The number of column of the sparse matrix must agree.
    """
    def perform(self, node, block, (out, )):
        for b in block:
            assert _is_sparse(b)
        out[0] = scipy.sparse.vstack(block, format=self.format,
                                     dtype=self.dtype)

    def grad(self, inputs, (gz, )):
        is_continuous = [(inputs[i].dtype in tensor.continuous_dtypes)
                        for i in range(len(inputs))]

        if all(is_continuous):
            if _is_sparse_variable(gz):
                gz = sparse.DenseFromSparse()(gz)

            split = tensor.Split(len(inputs))(gz, 0,
                                              tensor.stack(
                                                  *[x.shape[0]
                                                    for x in inputs]))
            if not isinstance(split, list):
                split = [split]
            return [sparse.SparseFromDense(self.format)(s) for s in split]
        else:
            return [None] * len(inputs)

    def infer_shape(self, node, ins_shapes):
        def _get(l):
            return l[0]
        d = sum(map(_get, ins_shapes))
        return [(d, ins_shapes[0][1])]


def hstack(blocks, format=None, dtype=None):
    """Stack sparse matrices vertically (row wise).

    This wrap the method vstack from scipy.

    :Parameters:
    - `blocks`: Sequence of sparse array of compatible shape
    - `format`: String representing the output format
    - `dtype`: Output dtype

    :return: the concatenation of the sparse array row wise.

    The number of column of the sparse matrix must agree.
    """
    return VStack(format=format, dtype=dtype)(*blocks)


class AddSSData(gof.op.Op):
    """Add two sparse matrices assuming they have the same sparsity
    pattern.

    :Parameters:
    - `x`: Sparse matrix.
    - `y`: Sparse matrix.

    :return: The sum of the two sparse matrix element wise.

    :note: `x` and `y` are assumed to have the same sparsity pattern.
           The grad implemented is structured.

    """
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
                         [SparseType(dtype=x.type.dtype,
                                 format=x.type.format).make_variable()])

    def perform(self, node, (x, y), (out, )):
        assert _is_sparse(x) and _is_sparse(y)
        assert x.shape == y.shape
        assert x.data.shape == y.data.shape
        out[0] = x.copy()
        out[0].data += y.data

    def grad(self, inputs, (gz, )):
        is_continuous = [(i.dtype in sparse.continuous_dtypes)
                         for i in inputs]

        if all(is_continuous):
            return [gz, gz]
        else:
            return [None] * len(inputs)

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[0]]

    def __str__(self):
        return self.__class__.__name__
add_s_s_data = AddSSData()


# register a specialization to replace MulSD -> MulSDCSX
@gof.local_optimizer([mul_s_d])
def local_mul_s_d(node):
    if node.op == mul_s_d:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        # y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 2:
            return False
        if svar.type.format == 'csc':
            CSx = CSC
            mul_s_d_csx = mul_s_d_csc
        elif svar.type.format == 'csr':
            CSx = CSR
            mul_s_d_csx = mul_s_d_csr
        else:
            raise NotImplemented()

        c_data = mul_s_d_csx(csm_data(svar), csm_indices(svar),
                             csm_indptr(svar), dvar)

        return [CSx(c_data, csm_indices(svar), csm_indptr(svar),
                    csm_shape(svar))]

    return False
register_specialize(local_mul_s_d)


class MulSDCSC(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 2
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (1,)

    #def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplementedError()
    def c_code(self, node, name, (_data, _indices, _indptr, _b,),
               (_zout, ), sub):

        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 2) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            %(fail)s;}
        if (%(_data)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;}
        if (%(_indices)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;}
        if (%(_indptr)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                  %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError,
    "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            //TODO: error checking with this
            const npy_intp N =  %(_indptr)s->dimensions[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)%(_data)s->data;
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * const __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)%(_zout)s->data;

            const npy_intp Sb = %(_b)s->strides[0];

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // extract i-th row of dense matrix
                    const dtype_%(_b)s* __restrict__ b_row = (dtype_%(_b)s*)(%(_b)s->data + Sb * i);

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] * b_row[j];
                }
            }
        }

        """ % dict(locals(), **sub)
mul_s_d_csc = MulSDCSC()


class MulSDCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 2
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (1,)

    #def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplemented()
    def c_code(self, node, name, (_data, _indices, _indptr, _b,),
               (_zout, ), sub):

        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 2) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            %(fail)s;}
        if (%(_data)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;}
        if (%(_indices)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;}
        if (%(_indptr)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError,
    "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            //TODO: error checking with this
            const npy_intp N =  %(_indptr)s->dimensions[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)%(_data)s->data;
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * const __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)%(_zout)s->data;

            const npy_intp Sb = %(_b)s->strides[0];

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {
                // extract i-th row of dense matrix
                const dtype_%(_b)s* __restrict__ b_row = (dtype_%(_b)s*)(%(_b)s->data + Sb * j);

                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] * b_row[i];
                }
            }
        }

        """ % dict(locals(), **sub)
mul_s_d_csr = MulSDCSR()


class Poisson(gof.op.Op):
    """Return a sparse having random values from a poisson density
    with mean from the input.

    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = x.copy()
        out[0].data = numpy.asarray(numpy.random.poisson(out[0].data),
                                    dtype=x.dtype)
        out[0].eliminate_zeros()

    def grad(self, inputs, outputs_gradients):
        return [None]

    def infer_shape(self, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return self.__class__.__name__
poisson = Poisson()


class Multinomial(gof.op.Op):
    """Return a sparse matrix having random values from a multinomial
    density having number of experiment `n` and probability of succes
    `p`.

    :Parameters:
    - `n`: Number of experiment.
    - `p`: Sparse probability of each of the different outcomes.

    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, n, p):
        n = tensor.as_tensor_variable(n)
        p = as_sparse_variable(p)

        return gof.Apply(self, [n, p], [p.type()])

    def perform(self, node, (n, p), (out, )):
        assert _is_sparse(p)

        if p.format != 'csr':
            raise NotImplemented()

        out[0] = p.copy()
        for i in xrange(p.shape[0]):
            k, l = p.indptr[i], p.indptr[i + 1]
            out[0].data[k:l] = numpy.random.multinomial(n[i], p.data[k:l])

    def grad(self, inputs, outputs_gradients):
        return [None, None]

    def infer_shape(self, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return self.__class__.__name__
multinomial = Multinomial()


class Binomial(gof.op.Op):
    def __init__(self, format, dtype):
        self.format = format
        self.dtype = dtype

    def __eq__(self, other):
        return ((type(self) == type(other)) and
                self.format == other.format and
                self.dtype == other.dtype)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.format) ^ hash(self.dtype)

    def make_node(self, n, p, shape):
        n = tensor.as_tensor_variable(n)
        p = tensor.as_tensor_variable(p)
        shape = tensor.as_tensor_variable(shape)
        return gof.Apply(self, [n, p, shape], [SparseType(dtype=self.dtype,
                                 format=self.format).make_variable()])

    def perform(self, node, (n, p, shape, ), (out, )):
        N = n * p * shape[0] * shape[1]
        data = numpy.ones(N, dtype=self.dtype)
        row = numpy.random.randint(0, shape[0], N)
        col = numpy.random.randint(0, shape[1], N)

        res = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

        out[0] = getattr(res, 'to' + self.format)()
        out[0].data = numpy.ones_like(out[0].data)

    def grad(self, (n, p, shape, ), (gz,)):
        return None, None, None

    def infer_shape(self, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return self.__class__.__name__
csr_fbinomial = Binomial('csr', 'float32')
csc_fbinomial = Binomial('csc', 'float32')
csr_dbinomial = Binomial('csr', 'float64')
csc_dbinomial = Binomial('csc', 'float64')


def structured_monoid(tensor_op):
    """
    Generic operation to perform many kinds of monoid element-wise
    operations on the non-zeros of a sparse matrix.

    The first parameter must always be a sparse matrix. The other parameters
    must be scalars which will be passed as argument to the tensor_op.
    """
    def decorator(f):
        def wrapper(*args):
            x = as_sparse_variable(args[0])

            xs = [scalar.as_scalar(arg) for arg in args[1:]]

            data, ind, ptr, shape = csm_properties(x)

            data = tensor_op(data, *xs)

            return CSM(x.format)(data, ind, ptr, shape)
        return wrapper
    return decorator


@structured_monoid(tensor.nnet.sigmoid)
def structured_sigmoid(x):
    """structured elemwise sigmoid.
    """
    # see decorator for function body


@structured_monoid(tensor.exp)
def structured_exp(x):
    """structured elemwise exponential.
    """
    # see decorator for function body


@structured_monoid(tensor.log)
def structured_log(x):
    """structured elemwise logarithm.
    """
    # see decorator for function body


@structured_monoid(tensor.pow)
def structured_pow(x, y):
    """structured elemwise power of sparse matrix
    x by scalar y.
    """
    # see decorator for function body


@structured_monoid(tensor.minimum)
def structured_minimum(x, y):
    """structured elemwise minimum of sparse matrix
    x by scalar y.
    """
    # see decorator for function body


@structured_monoid(tensor.maximum)
def structured_maximum(x, y):
    """structured elemwise maximum of sparse matrix
    x by scalar y.
    """
    # see decorator for function body


@structured_monoid(tensor.add)
def structured_add(x):
    """structured addition of sparse matrix
    x and scalar y.
    """
    # see decorator for function body


class MulSV(gof.op.Op):
    '''Multiplication of sparse matrix by a broadcasted dense vector.'''
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        x = as_sparse_variable(x)
        y = tensor.as_tensor_variable(y)

        assert y.type.ndim == 1

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=x.type.dtype,
                                 format=x.type.format).make_variable()])

    def perform(self, node, (x, y), (out, )):
        assert _is_sparse(x) and not _is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x.toarray() * y)

    def grad(self, (x, y), (gz,)):
        assert _is_sparse_variable(x) and _is_dense_variable(y)
        assert _is_sparse_variable(gz)
        return mul_s_v(gz, y), sp_sum(x * gz, axis=0, sparse_grad=True)
mul_s_v = MulSV()


class MulSVCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 1
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        _data, _indices, _indptr, _b, = inputs
        _zout, = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            %(fail)s;
        }
        if (%(_data)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;
        }
        if (%(_indices)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;
        }
        if (%(_indptr)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;
        }

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s
            || %(_zout)s->dimensions[0] != %(_indices)s->dimensions[0]
            || !PyArray_ISCONTIGUOUS(%(_zout)s))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            //TODO: error checking with this
            const npy_intp N =  %(_indptr)s->dimensions[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)%(_data)s->data;
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * const __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            const dtype_%(_b)s* __restrict__ Db = (dtype_%(_b)s*)%(_b)s->data;

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)%(_zout)s->data;

            const npy_intp Sb = %(_b)s->strides[0] / %(_b)s->descr->elsize;

            // loop over rows
            for (npy_int32 j = 0; j < N; ++j)
            {
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    zout[i_idx] = data[i_idx] * Db[i * Sb];
                }
            }
        }

        """ % dict(locals(), **sub)
mul_s_v_csr = MulSVCSR()


@gof.local_optimizer([mul_s_v])
def local_mul_s_v(node):
    if node.op == mul_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == 'csr':
            CSx = CSR
            mul_s_v_csx = mul_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = csm_properties(svar)

        c_data = mul_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_mul_s_v)


class StructuredAddSV(gof.op.Op):
    '''Structured addition of a sparse matrix and a dense vector.
    The elements of the vector are are only added to the corresponding
    non-zero elements. Therefore, this operation outputs another sparse
    matrix.'''
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        x = as_sparse_variable(x)
        y = tensor.as_tensor_variable(y)

        assert y.type.ndim == 1

        if x.type.dtype != y.type.dtype:
            raise NotImplementedError()
        return gof.Apply(self,
                         [x, y],
                         [SparseType(dtype=x.type.dtype,
                                 format=x.type.format).make_variable()])

    def perform(self, node, (x, y), (out, )):
        assert _is_sparse(x) and not _is_sparse(y)
        assert x.shape[1] == y.shape[0]
        out[0] = x.__class__(x + (x.toarray() != 0) * y)

    def grad(self, (x, y), (gz,)):
        assert _is_sparse_variable(x) and not _is_sparse_variable(y)
        assert _is_sparse_variable(gz)
        return gz, sp_sum(gz, axis=0, sparse_grad=True)
structured_add_s_v = StructuredAddSV()


class StrucutedAddSVCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_data, a_indices, a_indptr, b):
        b = tensor.as_tensor_variable(b)
        a_data = tensor.as_tensor_variable(a_data)
        a_indices = tensor.as_tensor_variable(a_indices)
        a_indptr = tensor.as_tensor_variable(a_indptr)
        assert a_data.type.ndim == 1
        assert a_indices.type.ndim == 1
        assert a_indptr.type.ndim == 1
        assert b.type.ndim == 1
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        _data, _indices, _indptr, _b, = inputs
        _zout, = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            %(fail)s;
        }
        if (%(_data)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;
        }
        if (%(_indices)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;
        }
        if (%(_indptr)s->nd != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;
        }

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError,
     "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            //TODO: error checking with this
            const npy_intp N =  %(_indptr)s->dimensions[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)%(_data)s->data;
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)%(_indptr)s->data;
            const npy_int32 * const __restrict__ indices = (npy_int32 *)%(_indices)s->data;

            const dtype_%(_b)s* __restrict__ Db = (dtype_%(_b)s*)%(_b)s->data;

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)%(_zout)s->data;

            const npy_intp Sb = %(_b)s->strides[0] / %(_b)s->descr->elsize;

            // loop over columns
            for (npy_int32 j = 0; j < N; ++j)
            {
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] + Db[i * Sb];
                }
            }
        }

        """ % dict(locals(), **sub)
structured_add_s_v_csr = StrucutedAddSVCSR()


@gof.local_optimizer([structured_add_s_v])
def local_structured_add_s_v(node):
    if node.op == structured_add_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        #y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == 'csr':
            CSx = CSR
            structured_add_s_v_csx = structured_add_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = csm_properties(svar)

        c_data = structured_add_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_structured_add_s_v)


class SamplingDot(gof.op.Op):
    """
    Operand for calculating the dot product DOT(X, Y) = Z when you
    only want to calculate a subset of Z. It is equivalent to P o (X
    . Y) where o is the element-wise product, X and Y operands of the
    dot product and P is a matrix that contains 1 when the
    corresponding element of Z should be calculated and 0 when it
    shouldn't. Note that SamplingDot has a different interface than
    DOT because SamplingDot requires X to be a MxK matrix while Y is a
    NxK matrix instead of the usual KxN matrix.

    It will work if the pattern is not binary value, but if the
    pattern doesn't have a high sparsity proportion it will be slower
    then a more optimized dot followed by a normal elemwise
    multiplication.

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDot'

    def make_node(self, x, y, p):
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)

        if not _is_sparse_variable(p):
            raise TypeError(p)

        #TODO: use it.
        dtype_out = scalar.upcast(x.type.dtype, y.type.dtype, p.type.dtype)

        return gof.Apply(self, [x, y, p], [p.type()])

    def perform(self, node, (x, y, p), (out,)):
        if _is_sparse_variable(x):
            raise TypeError(x)

        if _is_sparse_variable(y):
            raise TypeError(y)

        if not _is_sparse(p):
            raise TypeError(p)

        rval = p.__class__(p.multiply(numpy.dot(x, y.T)))

        out[0] = rval

    def grad(self, (x, y, p), (gz,)):
        rval = [
            dot(p * gz, y),
            dot(p.T * gz.T, x),
            None
        ]

        return rval
sampling_dot = SamplingDot()


class SamplingDotCsr(gof.Op):
    """
    Optimized SamplingDot when the pattern P is a CSR matrix.

    If we have the input of mixed dtype, we insert cast elemwise in the graph
    to be able to call blas function as they don't allow mixed dtype.

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDot{Csr}'

    def make_node(self, x, y, p_data, p_ind, p_ptr, p_ncols):
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        p_data = tensor.as_tensor_variable(p_data)
        p_ind = tensor.as_tensor_variable(p_ind)
        p_ptr = tensor.as_tensor_variable(p_ptr)
        p_ncols = tensor.as_tensor_variable(p_ncols)

        assert p_ncols.dtype == 'int32'

        dtype_out = scalar.upcast(x.type.dtype, y.type.dtype,
                                  p_data.type.dtype)
        dot_out = scalar.upcast(x.type.dtype, y.type.dtype)

        # We call blas ?dot function that take only param of the same type
        x = tensor.cast(x, dot_out)
        y = tensor.cast(y, dot_out)

        return gof.Apply(self, [x, y, p_data, p_ind, p_ptr, p_ncols], [
            tensor.tensor(dtype=dtype_out, broadcastable=(False,)),
            tensor.tensor(dtype=p_ind.type.dtype, broadcastable=(False,)),
            tensor.tensor(dtype=p_ptr.type.dtype, broadcastable=(False,))
        ])

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
        x, y, p_data, p_ind, p_ptr, p_ncols = inputs
        z_data, z_ind, z_ptr = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for x')
        if node.inputs[1].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for y')
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError(
                'Complex types are not supported for pattern')

        dot_out = scalar.upcast(node.inputs[0].type.dtype,
                                node.inputs[1].type.dtype)

        if dot_out == "float32":
            conv_type = "float"
            cdot = "sdot_"
        else:
            conv_type = "double"
            cdot = "ddot_"

        # retrieve dtype number
        typenum_x = node.inputs[0].type.dtype_specs()[-1]
        typenum_y = node.inputs[1].type.dtype_specs()[-1]
        typenum_p = node.inputs[2].type.dtype_specs()[-1]
        typenum_zd = tensor.TensorType(node.outputs[0].dtype,
                                       []).dtype_specs()[-1]
        typenum_zi = tensor.TensorType(node.outputs[1].dtype,
                                       []).dtype_specs()[-1]
        typenum_zp = tensor.TensorType(node.outputs[2].dtype,
                                       []).dtype_specs()[-1]

        rval = """
        if (%(x)s->nd != 2) {
PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(y)s->nd != 2) {
PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}

        if (%(x)s->descr->type_num != %(typenum_x)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for x");
            %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for y");
            %(fail)s;}

        if (%(p_data)s->descr->type_num != %(typenum_p)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for pattern");
            %(fail)s;}

        if (%(x)s->dimensions[1] != %(y)s->dimensions[1]) {
            PyErr_SetString(PyExc_NotImplementedError,
              "x's number of columns doesn't match y's rows! Note: sampling_dot is different from dot because y is assumed to be transposed.");
            %(fail)s;}

        if (%(y)s->dimensions[0] != ((npy_int32 *)%(p_ncols)s->data)[0] ||
            %(x)s->dimensions[0] != (%(p_ptr)s->dimensions[0] - 1))
        {PyErr_SetString(PyExc_NotImplementedError,
        "The dimension of the pattern and the output must match"); %(fail)s;}

        // Allocate output
        if (!%(z_data)s
            || (%(z_data)s->dimensions[0] != %(p_data)s->dimensions[0])
            || (%(z_data)s->descr->type_num != %(typenum_zd)s)) {
            {Py_XDECREF(%(z_data)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_data)s->dimensions[0];
            %(z_data)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                            %(typenum_zd)s);
        }
        if (!%(z_ind)s
            || (%(z_ind)s->dimensions[0] != %(p_ind)s->dimensions[0])
            || (%(z_ind)s->descr->type_num != %(typenum_zi)s)) {
            {Py_XDECREF(%(z_ind)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_ind)s->dimensions[0];
            %(z_ind)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           %(typenum_zi)s);
        }
        if (!%(z_ptr)s
            || (%(z_ptr)s->dimensions[0] != %(p_ptr)s->dimensions[0])
            || (%(z_ptr)s->descr->type_num != %(typenum_zp)s)) {
            {Py_XDECREF(%(z_ptr)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_ptr)s->dimensions[0];
            %(z_ptr)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           %(typenum_zp)s);
        }

        {
            // Product of MxK and NxK, output MxN
            npy_intp M = %(x)s->dimensions[0];
            npy_intp N = %(y)s->dimensions[0];
            npy_intp K = %(y)s->dimensions[1];

            // pointers to access actual data in the arrays passed as params.
            const dtype_%(x)s* __restrict__ Dx = (dtype_%(x)s*)%(x)s->data;
            const dtype_%(y)s* __restrict__ Dy = (dtype_%(y)s*)%(y)s->data;
            const dtype_%(p_data)s* __restrict__ Dpd = (dtype_%(p_data)s*)%(p_data)s->data;
            const dtype_%(p_ind)s* __restrict__ Dpi = (dtype_%(p_ind)s*)%(p_ind)s->data;
            const dtype_%(p_ptr)s* __restrict__ Dpp = (dtype_%(p_ptr)s*)%(p_ptr)s->data;
            dtype_%(z_data)s* __restrict__ Dzd = (dtype_%(z_data)s*)%(z_data)s->data;
            dtype_%(z_ind)s* __restrict__ Dzi = (dtype_%(z_ind)s*)%(z_ind)s->data;
            dtype_%(z_ptr)s* __restrict__ Dzp = (dtype_%(z_ptr)s*)%(z_ptr)s->data;
            
            const npy_intp Sdx = %(x)s->strides[1]/%(x)s->descr->elsize;
            const npy_intp Sdy = %(y)s->strides[1]/%(y)s->descr->elsize;
            const npy_intp Sdpd = %(p_data)s->strides[0] / %(p_data)s->descr->elsize;
            const npy_intp Sdpi = %(p_ind)s->strides[0] / %(p_ind)s->descr->elsize;
            const npy_intp Sdpp = %(p_ptr)s->strides[0] / %(p_ptr)s->descr->elsize;
            const npy_intp Sdzd = %(z_data)s->strides[0] / %(z_data)s->descr->elsize;
            const npy_intp Sdzi = %(z_ind)s->strides[0] / %(z_ind)s->descr->elsize;
            const npy_intp Sdzp = %(z_ptr)s->strides[0] / %(z_ptr)s->descr->elsize;
            
            memcpy(Dzi, Dpi, %(p_ind)s->dimensions[0]*sizeof(dtype_%(p_ind)s));
            memcpy(Dzp, Dpp, %(p_ptr)s->dimensions[0]*sizeof(dtype_%(p_ptr)s));
            
            for (npy_int32 m = 0; m < M; ++m) {
                for (npy_int32 n_idx = Dpp[m * Sdpp]; n_idx < Dpp[(m+1)*Sdpp]; ++n_idx) {
                    const npy_int32 n = Dpi[n_idx * Sdpi]; // row index of non-null value for column K
                    
                    const dtype_%(x)s* x_row = (dtype_%(x)s*)(%(x)s->data + %(x)s->strides[0] * m);
                    
                    const dtype_%(y)s* y_col = (dtype_%(y)s*)(%(y)s->data + %(y)s->strides[0] * n);
                    
                    Dzd[n_idx * Sdzd] = Dpd[n_idx * Sdpd] * %(cdot)s((int*)&K, (const %(conv_type)s*)x_row, (int*)&Sdx, (const %(conv_type)s*)y_col, (int*)&Sdy);
                }
            }
        }
        """ % dict(locals(), **sub)

        return rval
sampling_dot_csr = SamplingDotCsr()


# register a specialization to replace SamplingDot -> SamplingDotCsr
@gof.local_optimizer([sampling_dot])
def local_sampling_dot_csr(node):
    if node.op == sampling_dot:
        x, y, p = node.inputs
        if p.type.format == 'csr':
            p_data, p_ind, p_ptr, p_shape = csm_properties(p)

            z_data, z_ind, z_ptr = sampling_dot_csr(x, y, p_data,
                p_ind, p_ptr, p_shape[1])

            return [CSR(z_data, z_ind, z_ptr, p_shape)]
    return False
register_specialize(local_sampling_dot_csr, name='local_sampling_dot_csr')
