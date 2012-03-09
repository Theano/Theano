from theano.sparse.basic import * # To facilitate later merge into sparse module
from theano.sparse.basic import (
    _is_sparse, _is_sparse_variable, _is_dense_variable,
    _is_sparse, _is_dense, _kmap_eq, _kmap_hash)


class Cast(gof.op.Op):
    def __init__(self, out_type):
        self.out_type = out_type

    def __eq__(self, other):
        return (type(self) == type(other)) and self.out_type == other.out_type

    def __hash__(self):
        return hash(type(self)) ^ hash(self.out_type)

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self, [x],
            [SparseType(dtype=self.out_type, format=x.format).make_variable()])

    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        out[0] = x
        out[0].data = numpy.asarray(out[0].data, dtype=self.out_type)
fcast = Cast('float32')
dcast = Cast('float64')


# register a specialization to replace AddSS -> AddSSData
@gof.local_optimizer([add_s_s])
def local_add_s_s(node):
    """
    If two matrices are known to have the same sparsity pattern,
    optimize  the addition by only adding their data vector.

    Very special case optimization. Activate when for add(x, y),
    y is an expression like sp_ones_like(x) * another_matrix.

    This is useful for sparse weight updates.

    Work also for add(x, neg(y)) in the same case.

    As of this writting sub is only implemented as x + neg(y) for
    sparse matrix.
    """
    if node.op == add_s_s:
        x, y = node.inputs

        # In case addition was transformed to subtraction
        if hasattr(y.owner, 'op') and y.owner.op == neg:
            y_ = y.owner.inputs[0]
        else:
            y_ = y
        if y_.owner is None:
            return False
        if hasattr(y_.owner, 'op') and y_.owner.op not in [mul_s_s, mul_s_d]:
            return False

        def same_pattern(node):
            """Check node has same sparsity as x."""
            # In case the sparse matrix is multiplied by a scalar (ex:
            # learning rate)
            if hasattr(node.owner, 'op') and node.owner.op == mul_scalar:
                node = node.owner.inputs[1]

            # Check node creates a matrix
            if not hasattr(node.owner, 'op') or not isinstance(node.owner.op,
                                                               CSM):
                return False

            # Check matrix is creates from CSMProperties
            if filter(lambda i: not hasattr(i.owner, 'op') or
                      not isinstance(i.owner.op, CSMProperties),
                      node.owner.inputs[1:]):
                return False

            # Verify indices, indptr and shape are the same as x
            if filter(lambda i: i.owner.inputs[0] != x, node.owner.inputs[1:]):
                return False

            return True

        if filter(same_pattern, y_.owner.inputs):
            return [add_s_s_data(x, y)]
    return False
register_specialize(local_add_s_s)


class AddSSData(gof.op.Op):
    '''Add two sparse matrices assuming they have the same sparsity
    pattern. '''
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
        out[0] = x.copy()
        out[0].data += y.data
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

    #def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplementedError()
    def c_code(self, node, name, (_data, _indices, _indptr, _b,),
               (_zout, ), sub):

        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}
        if (%(_data)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}
        
        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError, "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }
        
        {   //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            const npy_intp N =  %(_indptr)s->dimensions[0]-1; //TODO: error checking with this
            
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

    #def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplemented()
    def c_code(self, node, name, (_data, _indices, _indptr, _b,),
               (_zout, ), sub):

        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}
        if (%(_data)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}
        
        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError, "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }
        
        {   //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            const npy_intp N =  %(_indptr)s->dimensions[0]-1; //TODO: error checking with this
            
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
poisson = Poisson()


class Multinomial(gof.op.Op):
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
multinomial = Multinomial()


class EliminateZeros(gof.op.Op):
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
        out[0].eliminate_zeros()
eliminate_zeros = EliminateZeros()


class Sum(gof.op.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, a):
        x = as_sparse_variable(x)
        a = tensor.as_tensor_variable(a)
        return gof.Apply(self, [x, a], [tensor.TensorType(dtype=x.type.dtype,
                        broadcastable=(False,)).make_variable()])

    def perform(self, node, (x, a), (out, )):
        assert _is_sparse(x)
        out[0] = numpy.asarray(x.sum(a), dtype=x.dtype).flatten()
sum = Sum()


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
csr_fbinomial = Binomial('csr', 'float32')
csc_fbinomial = Binomial('csc', 'float32')
csr_dbinomial = Binomial('csr', 'float64')
csc_dbinomial = Binomial('csc', 'float64')


def structured_sigmoid(x):
    """
    Element-wise sigmoid function only to the non-zero elements.
    """
    x = as_sparse_variable(x)

    x_data, x_ind, x_ptr, x_shape = csm_properties(x)

    x_data = tensor.nnet.sigmoid(x_data)

    return CSR(x_data, x_ind, x_ptr, x_shape)


def structured_exp(x):
    """
    Element-wise exponential function to the non-zero elements.
    """
    x = as_sparse_variable(x)

    x_data, x_ind, x_ptr, x_shape = csm_properties(x)

    x_data = tensor.exp(x_data)

    return CSR(x_data, x_ind, x_ptr, x_shape)


def structured_pow(x, y):
    """
    Element-wise power function only to non-zero elements.
    """
    x = as_sparse_variable(x)

    y = tensor.as_tensor_variable(y)

    x_data, x_ind, x_ptr, x_shape = csm_properties(x)

    x_data = tensor.pow(x_data, y)

    return CSR(x_data, x_ind, x_ptr, x_shape)


def structured_minimum(x, y):
    """
    Element-wise minimum function only to non-zero elements.
    """
    x = as_sparse_variable(x)

    y = tensor.as_tensor_variable(y)

    x_data, x_ind, x_ptr, x_shape = csm_properties(x)

    x_data = tensor.minimum(x_data, y)

    return CSR(x_data, x_ind, x_ptr, x_shape)


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
        assert _is_sparse_variable(x) and _is_sparse_variable(y)
        assert _is_sparse_variable(gz)
        return gz, gz
structured_add_s_v = StructuredAddSV()


class StrucutedAddSVCSR(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 1
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code(self, node, name, inputs, outputs, sub):
        _data, _indices, _indptr, _b, = inputs
        _zout, = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (%(_b)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1"); %(fail)s;}
        if (%(_data)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1"); %(fail)s;}
        if (%(_indices)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1"); %(fail)s;}
        if (%(_indptr)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1"); %(fail)s;}

        if( %(_indices)s->descr->type_num != PyArray_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( %(_indptr)s->descr->type_num != PyArray_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}
        
        if (!%(_zout)s)
        {
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1, %(_indices)s->dimensions, %(_b)s->descr->type_num);
        }

        if (%(_zout)s->dimensions[0] != %(_indices)s->dimensions[0])
        {
            PyErr_SetString(PyExc_NotImplementedError, "somehow _zout got the wrong size.. and I don't know how to resize it.");
            %(fail)s;
        }
        
        {   //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = %(_indices)s->dimensions[0];
            const npy_intp N =  %(_indptr)s->dimensions[0]-1; //TODO: error checking with this
            
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
            raise NotImplemented()

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
            dot(gz, y),
            dot(gz.T, x),
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
        import pdb; pdb.set_trace()
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

        # TODO: why 2 times the same inputs?
        dot_out = scalar.upcast(node.inputs[0].type.dtype,
                                node.inputs[0].type.dtype)

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
        if (%(x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}

        if (%(x)s->descr->type_num != %(typenum_x)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x"); %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}
        
        if (%(p_data)s->descr->type_num != %(typenum_p)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for pattern"); %(fail)s;}
        
        if (%(x)s->dimensions[1] != %(y)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows! Note: sampling_dot is different from dot because y is assumed to be transposed."); %(fail)s;}
        
        if (%(y)s->dimensions[0] != ((npy_int32 *)%(p_ncols)s->data)[0] || %(x)s->dimensions[0] != (%(p_ptr)s->dimensions[0] - 1))
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of the pattern and the output must match"); %(fail)s;}
        
        // Allocate output
        if (!%(z_data)s
            || (%(z_data)s->dimensions[0] != %(p_data)s->dimensions[0])
            || (%(z_data)s->descr->type_num != %(typenum_zd)s)) {
            {Py_XDECREF(%(z_data)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_data)s->dimensions[0];
            %(z_data)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, %(typenum_zd)s);
        }
        if (!%(z_ind)s
            || (%(z_ind)s->dimensions[0] != %(p_ind)s->dimensions[0])
            || (%(z_ind)s->descr->type_num != %(typenum_zi)s)) {
            {Py_XDECREF(%(z_ind)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_ind)s->dimensions[0];
            %(z_ind)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, %(typenum_zi)s);
        }
        if (!%(z_ptr)s
            || (%(z_ptr)s->dimensions[0] != %(p_ptr)s->dimensions[0])
            || (%(z_ptr)s->descr->type_num != %(typenum_zp)s)) {
            {Py_XDECREF(%(z_ptr)s);}
            npy_intp dims[] = {0};
            dims[0] = %(p_ptr)s->dimensions[0];
            %(z_ptr)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, %(typenum_zp)s);
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
