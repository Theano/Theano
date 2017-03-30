from __future__ import absolute_import, print_function, division
import numpy as np
import scipy

import theano
from theano import gof, scalar, tensor
from theano.compat import izip
from theano.tensor import blas
from theano.tensor.opt import register_specialize, register_canonicalize
from theano.sparse import (CSC, CSR, csm_properties,
                           csm_grad, usmm, csm_indices, csm_indptr,
                           csm_data)
from theano.sparse import basic as sparse

_is_sparse_variable = sparse._is_sparse_variable
_is_dense = sparse._is_dense

# This is tested in tests/test_opt.py:test_local_csm_properties_csm


@gof.local_optimizer([csm_properties])
def local_csm_properties_csm(node):
    """
    If we find csm_properties(CSM(*args)), then we can replace that with the
    *args directly.

    """
    if node.op == csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            # csm.owner.inputs could be broadcastable. In that case, we have
            # to adjust the broadcasting flag here.
            ret_var = [theano.tensor.patternbroadcast(i, o.broadcastable)
                       for i, o in izip(csm.owner.inputs, node.outputs)]
            return ret_var

    return False
register_specialize(local_csm_properties_csm)


# This is tested in tests/test_basic.py:test_remove0
@gof.local_optimizer([sparse.Remove0])
def local_inplace_remove0(node):
    """
    Optimization to insert inplace versions of Remove0.

    """
    # If inplace is not enabled, enable it and replace that op with a
    # new op which has inplace enabled
    if isinstance(node.op, sparse.Remove0) and not node.op.inplace:
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False

theano.compile.optdb.register(
    'local_inplace_remove0',
    gof.TopoOptimizer(local_inplace_remove0,
                      failure_callback=gof.TopoOptimizer.warn_inplace),
    60, 'fast_run', 'inplace')


class AddSD_ccode(gof.op.Op):
    """
    Add a sparse and a dense matrix.

    Parameters
    ----------
    x
        A sparse matrix.
    y
        A dense matrix

    Returns
    -------
    matrix
        `x`+`y`

    Notes
    -----
    The grad implemented is structured on `x`.

    """

    __props__ = ("format", "inplace")

    def __init__(self, format, inplace=False, *args, **kwargs):
        gof.Op.__init__(self, *args, **kwargs)
        # Should we do inplace addition or not ?
        self.inplace = inplace
        self.format = format
        if self.inplace:
            self.destroy_map = {0: [3]}

    def __str__(self):
        inp = ''
        if self.inplace:
            inp = ',inplace'
        return "%s{%s%s}" % (self.__class__.__name__,
                             self.format, inp)

    def make_node(self, x, y):
        x, y = sparse.as_sparse_variable(x), tensor.as_tensor_variable(y)
        out_dtype = scalar.upcast(x.type.dtype, y.type.dtype)
        if self.inplace:
            assert out_dtype == y.dtype

        indices, indptr, data = csm_indices(x), csm_indptr(x), csm_data(x)
        # We either use CSC or CSR depending on the format of input
        assert self.format == x.type.format
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        out = tensor.TensorType(dtype=out_dtype,
                                broadcastable=y.type.broadcastable)()
        return gof.Apply(self,
                         [data, indices, indptr, y],
                         [out])

    def c_code(self, node, name, inputs, outputs, sub):
        (_data, _indices, _indptr, y) = inputs
        (z,) = outputs
        inplace = int(self.inplace)
        format = {'csc': 0, 'csr': 1}[self.format]
        out_typenum = node.outputs[0].type.dtype_specs()[2]
        code = """
                Py_XDECREF(%(z)s);
                if (!%(inplace)s){
                    if(PyArray_TYPE(%(y)s) != %(out_typenum)s){
                        %(z)s = (PyArrayObject *) PyArray_FromArray(%(y)s,  PyArray_DescrFromType(%(out_typenum)s), 0);
                    }else{
                        %(z)s = (PyArrayObject *) PyArray_NewCopy(%(y)s, NPY_CORDER);
                    }
                }else{
                  %(z)s = %(y)s;
                  Py_XINCREF(%(z)s);
                }

                npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1;

                const dtype_%(_indptr)s* __restrict__ indptr = (dtype_%(_indptr)s*)PyArray_DATA(%(_indptr)s);
                const dtype_%(_indices)s* __restrict__ indices = (dtype_%(_indices)s*)PyArray_DATA(%(_indices)s);
                const dtype_%(_data)s* __restrict__ data = (dtype_%(_data)s*)PyArray_DATA(%(_data)s);

                dtype_%(y)s* ydata = (dtype_%(y)s*)PyArray_DATA(%(y)s);
                dtype_%(z)s* zdata = (dtype_%(z)s*)PyArray_DATA(%(z)s);
                npy_intp Yi = PyArray_STRIDES(%(y)s)[0]/PyArray_DESCR(%(y)s)->elsize;
                npy_intp Yj = PyArray_STRIDES(%(y)s)[1]/PyArray_DESCR(%(y)s)->elsize;

                npy_intp pos;
                if (%(format)s == 0){
                for (npy_intp col = 0; col < N; ++col){
                  for (dtype_%(_indptr)s ind = indptr[col]; ind < indptr[col+1]; ++ind){
                    npy_intp row = indices[ind];
                    pos = row * Yi + col * Yj;
                    zdata[pos] = ydata[pos] + data[ind];
                  }
                }
                }else{
                for (npy_intp row = 0; row < N; ++row){
                  for (dtype_%(_indptr)s ind = indptr[row]; ind < indptr[row+1]; ++ind){
                    npy_intp col = indices[ind];
                    pos = row * Yi + col * Yj;
                    zdata[pos] = ydata[pos] + data[ind];
                  }
                 }
                }
             """ % dict(locals(), **sub)
        return code

    def infer_shape(self, node, shapes):
        return [shapes[3]]

    def c_code_cache_version(self):
        return (2,)


@gof.local_optimizer([sparse.AddSD])
def local_inplace_addsd_ccode(node):
    """
    Optimization to insert inplace versions of AddSD.

    """
    if isinstance(node.op, sparse.AddSD) and theano.config.cxx:
        out_dtype = scalar.upcast(*node.inputs)
        if out_dtype != node.inputs[1].dtype:
            return
        new_node = AddSD_ccode(format=node.inputs[0].type.format,
                               inplace=True)(*node.inputs)
        return [new_node]
    return False
theano.compile.optdb.register(
    'local_inplace_addsd_ccode',
    gof.TopoOptimizer(local_inplace_addsd_ccode,
                      failure_callback=gof.TopoOptimizer.warn_inplace),
    60, 'fast_run', 'inplace')


@register_canonicalize("fast_compile")
@register_specialize
@gof.local_optimizer([sparse.DenseFromSparse])
def local_dense_from_sparse_sparse_from_dense(node):
    if isinstance(node.op, sparse.DenseFromSparse):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, sparse.SparseFromDense):
            return inp.owner.inputs


@gof.local_optimizer([sparse.AddSD])
def local_addsd_ccode(node):
    """
    Convert AddSD to faster AddSD_ccode.

    """
    if isinstance(node.op, sparse.AddSD) and theano.config.cxx:
        new_node = AddSD_ccode(format=node.inputs[0].type.format)(*node.inputs)
        return [new_node]
    return False
theano.compile.optdb.register('local_addsd_ccode',
                              gof.TopoOptimizer(local_addsd_ccode),
                              # Must be after local_inplace_addsd_ccode at 60
                              61, 'fast_run')


class StructuredDotCSC(gof.Op):
    """
    Structured Dot CSC is like dot, except that only the gradient wrt non-zero
    elements of the sparse matrix `a` are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a
    TensorType instance.

    Parameters
    ----------
    a
        A sparse matrix in csc format.
    b
        A sparse or dense matrix.

    Returns
    -------
    The dot product of `a` and `b`.

    Notes
    -----
    The grad implemented is structured.
    This op is used as an optimization for StructuredDot.

    """

    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, a_nrows, b):
        dtype_out = scalar.upcast(a_val.type.dtype, b.type.dtype)
        r = gof.Apply(self, [a_val, a_ind, a_ptr, a_nrows, b],
                      [tensor.tensor(dtype_out,
                                     (False, b.type.broadcastable[1]))])
        return r

    def perform(self, node, inputs, outputs):
        (a_val, a_ind, a_ptr, a_nrows, b) = inputs
        (out,) = outputs
        a = scipy.sparse.csc_matrix((a_val, a_ind, a_ptr),
                                    (a_nrows, b.shape[0]),
                                    copy=False)
        # out[0] = a.dot(b)
        out[0] = theano._asarray(a * b, dtype=node.outputs[0].type.dtype)
        assert _is_dense(out[0])  # scipy 0.7 automatically converts to dense

    def c_code(self, node, name, inputs, outputs, sub):
        # C-implementation of the dot product of the sparse matrix A and matrix
        # B.
        # @param a_val: non-zero values of the sparse matrix
        # @param a_ind: column indices of the non-null values (.indices of a
        # scipy.csc_matrix)
        # @param a_ptr: a_ptr indicates col indices for col. i are in the range
        # a_ptr[i]:a_ptr[i+1]
        # @param n_rows: number of rows of sparse matrix
        # @param b: dense matrix to perform dot product with, as in dot(a, b)
        # @param z: return value
        # @param sub: TODO, not too sure, something to do with weave probably

        (a_val, a_ind, a_ptr, a_nrows, b) = inputs
        (z,) = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a_val')
        if node.inputs[4].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        typenum_z = node.outputs[0].type.dtype_specs()[2]  # retrieve dtype number
        typenum_a_val = node.inputs[0].type.dtype_specs()[2]  # retrieve dtype number
        typenum_b = node.inputs[4].type.dtype_specs()[2]  # retrieve dtype number

        rval = """

        if (PyArray_NDIM(%(a_val)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ind)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_nrows)s) != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); %(fail)s;}
        if (PyArray_NDIM(%(b)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (PyArray_TYPE(%(a_val)s) != %(typenum_a_val)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for a_val"); %(fail)s;}

        if (PyArray_TYPE(%(b)s) != %(typenum_b)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for b"); %(fail)s;}

        if (PyArray_TYPE(%(a_ind)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(a_ptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(a_nrows)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_nrows dtype not INT32"); %(fail)s;}

        if (PyArray_DIMS(%(a_val)s)[0] != PyArray_DIMS(%(a_ind)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if (PyArray_DIMS(%(a_ptr)s)[0] != PyArray_DIMS(%(b)s)[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "a's number of columns doesn't match b's rows"); %(fail)s;}

        if ((!%(z)s)
            || (PyArray_DIMS(%(z)s)[0] != ((npy_int32 *)PyArray_DATA(%(a_nrows)s))[0])
            || (PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(b)s)[1])
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0, 0};
            dims[0] = ((npy_int32 *)PyArray_DATA(%(a_nrows)s))[0];
            dims[1] = PyArray_DIMS(%(b)s)[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS(%(z)s)[0];
            npy_intp N = PyArray_DIMS(%(z)s)[1];
            npy_intp K = PyArray_DIMS(%(b)s)[0];
            if (N > 0x7fffffffL)
            {PyErr_SetString(PyExc_NotImplementedError, "array too big (overflows int32 index)"); %(fail)s;}

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = PyArray_STRIDES(%(z)s)[0] / PyArray_DESCR(%(z)s)->elsize;
            npy_intp Szn = PyArray_STRIDES(%(z)s)[1] / PyArray_DESCR(%(z)s)->elsize;
            //npy_intp Sbm = PyArray_STRIDES(%(b)s)[0] / PyArray_DESCR(%(b)s)->elsize;
            npy_intp Sbn = PyArray_STRIDES(%(b)s)[1] / PyArray_DESCR(%(b)s)->elsize;
            npy_intp Sval = PyArray_STRIDES(%(a_val)s)[0] / PyArray_DESCR(%(a_val)s)->elsize;
            npy_intp Sind = PyArray_STRIDES(%(a_ind)s)[0] / PyArray_DESCR(%(a_ind)s)->elsize;
            npy_intp Sptr = PyArray_STRIDES(%(a_ptr)s)[0] / PyArray_DESCR(%(a_ptr)s)->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s*     __restrict__ Dz   = (dtype_%(z)s*)PyArray_DATA(%(z)s);
            const dtype_%(a_val)s* __restrict__ Dval = (dtype_%(a_val)s*)PyArray_DATA(%(a_val)s);
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA(%(a_ind)s);
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA(%(a_ptr)s);

            //npy_intp nnz = PyArray_DIMS(%(a_ind)s)[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            //
            // Normal matrix matrix multiply: A MxK, B KxN =>  Z = AB
            // for m
            //   for n
            //     for k
            //        z[m, n] += a[m, k] * b[k, n]
            // Here instead: Z =
            // for k
            //   for m (sparse)
            //     for n
            //        z[m, n] += a[m, k] * b[k, n]

            // loop over inner dimension
            for (npy_int32 k = 0; k < K; ++k)
            {
                // get pointer to k-th row of dense matrix
                const dtype_%(b)s* __restrict__ bk = (dtype_%(b)s*)(PyArray_BYTES(%(b)s) + PyArray_STRIDES(%(b)s)[0] * k);

                // loop over sparse column indices through index pointer array
                // (amounts to looping over rows M of sparse matrix)

                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1) * Sptr]; ++m_idx)
                {
                    npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K
                    const dtype_%(a_val)s Amk = Dval[m_idx * Sval]; // actual value at that location

                    // pointer to m-th row of the output matrix Z
                    dtype_%(z)s* __restrict__ zm = (dtype_%(z)s*)(PyArray_BYTES(%(z)s) + PyArray_STRIDES(%(z)s)[0] * m);

                    //RESOLVE: a.shape[0] equals z.shape[0], why is this not an equality constraint?
                    if (m >= PyArray_DIMS(%(z)s)[0])
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
        """ % dict(locals(), **sub)

        return rval

    def c_code_cache_version(self):
        return (3,)
sd_csc = StructuredDotCSC()


class StructuredDotCSR(gof.Op):
    """
    Structured Dot CSR is like dot, except that only the
    gradient wrt non-zero elements of the sparse matrix
    `a` are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a
    TensorType instance.

    Parameters
    ----------
    a
        A sparse matrix in csr format.
    b
        A sparse or dense matrix.

    Returns
    -------
    matrix
        The dot product of `a` and `b`.

    Notes
    -----
    The grad implemented is structured.
    This op is used as an optimization for StructuredDot.

    """
    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, b):
        self.dtype_out = scalar.upcast(a_val.type.dtype, b.type.dtype)
        r = gof.Apply(self, [a_val, a_ind, a_ptr, b],
                      [tensor.tensor(self.dtype_out,
                                     (False, b.type.broadcastable[1]))])
        return r

    def perform(self, node, inputs, outputs):
        (a_val, a_ind, a_ptr, b) = inputs
        (out,) = outputs
        a = scipy.sparse.csr_matrix(
            (a_val, a_ind, a_ptr),
            (len(a_ptr) - 1, b.shape[0]),
            copy=True)  # use view_map before setting this to False
        # out[0] = a.dot(b)
        out[0] = a * b
        # scipy 0.7 automatically converts to dense, but not .6 sometimes
        assert _is_dense(out[0])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        C-implementation of the dot product of the sparse matrix A and matrix B.

        Parameters
        ----------
        a_val
            Non-zero values of the sparse matrix.
        a_ind
            Column indices of the non-null values (.indices of a
            scipy.csc_matrix).
        a_ptr
            Indicates col indices for col. i are in the range
            a_ptr[i]:a_ptr[i+1].
        n_cols
            Number of columns of sparse matrix.
        b
            Dense matrix to perform dot product with, as in dot(a, b).
        z
            Return value.
        sub
            TODO, not too sure, something to do with weave probably.

        """
        (a_val, a_ind, a_ptr, b) = inputs
        (z,) = outputs
        typenum_z = tensor.TensorType(self.dtype_out, []).dtype_specs()[2]
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a_val')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (PyArray_NDIM(%(a_val)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ind)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(b)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (PyArray_TYPE(%(a_ind)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(a_ptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (PyArray_DIMS(%(a_val)s)[0] != PyArray_DIMS(%(a_ind)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if ((!%(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(a_ptr)s)[0]-1) //a's rows
            || (PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(b)s)[1])       //b's columns
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0, 0};
            dims[0] = PyArray_DIMS(%(a_ptr)s)[0]-1;
            dims[1] = PyArray_DIMS(%(b)s)[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS(%(z)s)[0];
            npy_intp N = PyArray_DIMS(%(z)s)[1];
            npy_intp K = PyArray_DIMS(%(b)s)[0];
            if (N > 0x7fffffffL)
            {PyErr_SetString(PyExc_NotImplementedError, "array too big (overflows int32 index)"); %(fail)s;}

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = PyArray_STRIDES(%(z)s)[0] / PyArray_DESCR(%(z)s)->elsize;
            npy_intp Szn = PyArray_STRIDES(%(z)s)[1] / PyArray_DESCR(%(z)s)->elsize;
            npy_intp Sbm = PyArray_STRIDES(%(b)s)[0] / PyArray_DESCR(%(b)s)->elsize;
            npy_intp Sbn = PyArray_STRIDES(%(b)s)[1] / PyArray_DESCR(%(b)s)->elsize;
            npy_intp Sval = PyArray_STRIDES(%(a_val)s)[0] / PyArray_DESCR(%(a_val)s)->elsize;
            npy_intp Sind = PyArray_STRIDES(%(a_ind)s)[0] / PyArray_DESCR(%(a_ind)s)->elsize;
            npy_intp Sptr = PyArray_STRIDES(%(a_ptr)s)[0] / PyArray_DESCR(%(a_ptr)s)->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz = (dtype_%(z)s*)PyArray_DATA(%(z)s);
            const dtype_%(a_val)s* __restrict__ Dval = (dtype_%(a_val)s*)PyArray_DATA(%(a_val)s);
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA(%(a_ind)s);
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA(%(a_ptr)s);

            //npy_intp nnz = PyArray_DIMS(%(a_ind)s)[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            // Normal matrix matrix multiply:
            // for m
            //   for n
            //     for k
            //        z[m, n] += a[m, k] * b[k, n]
            // Here instead:
            // for m
            //   for k (sparse)
            //     for n
            //        z[m, n] += a[m, k] * b[k, n]

            // loop over inner dimension
            for (npy_int64 m = 0; m < M; ++m)
            {
                // pointer to m-th row of the output matrix Z
                dtype_%(z)s* __restrict__ zm = (dtype_%(z)s*)(PyArray_BYTES(%(z)s) + PyArray_STRIDES(%(z)s)[0] * m);

                // loop over sparse rows indices through index pointer array
                // (amounts to looping over cols k of sparse matrix)
                for (npy_int32 k_idx = Dptr[m * Sptr]; k_idx < Dptr[(m+1) * Sptr]; ++k_idx)
                {
                    npy_int32 k = Dind[k_idx * Sind]; // col index of non-null value for row m
                    const dtype_%(a_val)s Amk = Dval[k_idx * Sval]; // actual value at that location

                    // get pointer to k-th row of dense matrix
                    const dtype_%(b)s* __restrict__ bk = (dtype_%(b)s*)(PyArray_BYTES(%(b)s) + PyArray_STRIDES(%(b)s)[0] * k);

                    // loop over final dimension (cols of dense matrix) and perform dot product
                    for(npy_int32 n = 0; n < N; ++n)
                    {
                        zm[n*Szn] += Amk * bk[n*Sbn];
                    }
                }
            }
        }

        """ % dict(locals(), **sub)

    def c_code_cache_version(self):
        return (2,)
sd_csr = StructuredDotCSR()


# register a specialization to replace StructuredDot -> StructuredDotCSx
# This is tested in tests/test_basic.py:792
@gof.local_optimizer([sparse._structured_dot])
def local_structured_dot(node):
    if node.op == sparse._structured_dot:
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
# a) it is only slightly faster than scipy these days, and sometimes a little
# slower, and
# b) the resulting graphs make it very difficult for an op to do size checking
# on the matrices involved.  dimension mismatches are hard to detect sensibly.
# register_specialize(local_structured_dot)


class UsmmCscDense(gof.Op):
    """
    Performs the expression is `alpha` * `x` `y` + `z`.

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
    Optimized version os Usmm when `x` is in csc format and `y` is dense.
    """

    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [6]}

    def __str__(self):
        if self.inplace:
            return 'UsmmCscDense{inplace}'
        else:
            return 'UsmmCscDense{no_inplace}'

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
            raise NotImplementedError('only float types are supported in '
                                      'operands')

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

        r = gof.Apply(
            self, [alpha, x_val, x_ind, x_ptr, x_nrows, y, z],
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
        typenum_alpha = node.inputs[0].type.dtype_specs()[2]
        typenum_x_val = node.inputs[1].type.dtype_specs()[2]
        typenum_y = node.inputs[5].type.dtype_specs()[2]
        typenum_z = node.inputs[6].type.dtype_specs()[2]
        typenum_zn = node.outputs[0].type.dtype_specs()[2]

        inplace = int(self.inplace)

        rval = """

        if (PyArray_NDIM(%(x_val)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_val) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(x_ind)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_ind) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(x_ptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(x_ptr) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(x_nrows)s) != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); %(fail)s;}
        if (PyArray_NDIM(%(y)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}

        if (PyArray_TYPE(%(x_val)s) != %(typenum_x_val)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x_val"); %(fail)s;}

        if (PyArray_TYPE(%(y)s) != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}

        if (PyArray_TYPE(%(z)s) != %(typenum_z)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for z"); %(fail)s;}

        if (PyArray_TYPE(%(alpha)s) != %(typenum_alpha)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for alpha"); %(fail)s;}

        if (PyArray_TYPE(%(x_ind)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "x_ind dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(x_ptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "x_ptr dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(x_nrows)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "x_nrows dtype not INT32"); %(fail)s;}

        if (PyArray_DIMS(%(x_val)s)[0] != PyArray_DIMS(%(x_ind)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "x_val and x_ind have different lengths"); %(fail)s;}

        if (PyArray_DIMS(%(x_ptr)s)[0] != PyArray_DIMS(%(y)s)[0]+1)
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); %(fail)s;}

        if (PyArray_DIMS(%(z)s)[0] != ((npy_int32 *)PyArray_DATA(%(x_nrows)s))[0] || PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(y)s)[1])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of the allocated output doesn't match the correct output size."); %(fail)s;}

        if (PyArray_SIZE(%(alpha)s) != 1)
        {PyErr_SetString(PyExc_NotImplementedError, "The number of element in alpha must be 1"); %(fail)s;}

        if (PyArray_NDIM(%(alpha)s) != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of alpha must be 2"); %(fail)s;}

        if (PyArray_NDIM(%(x_val)s) != 1)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of x_val must be 1"); %(fail)s;}

        if (PyArray_NDIM(%(y)s) != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "The number dimension of y must be 2"); %(fail)s;}

        if (PyArray_NDIM(%(z)s) != 2)
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
            || (PyArray_DIMS(%(zn)s)[0] != ((npy_int32 *)PyArray_DATA(%(x_nrows)s))[0])
            || (PyArray_DIMS(%(zn)s)[1] != PyArray_DIMS(%(y)s)[1])
            )
        {
            {Py_XDECREF(%(zn)s);}
            npy_intp dims[] = {0, 0};
            dims[0] = ((npy_int32 *)PyArray_DATA(%(x_nrows)s))[0];
            dims[1] = PyArray_DIMS(%(y)s)[1];
            %(zn)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_zn)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS(%(zn)s)[0];
            npy_intp N = PyArray_DIMS(%(zn)s)[1];
            npy_intp K = PyArray_DIMS(%(y)s)[0];

            // pointers to access actual data in the arrays passed as params.
            const dtype_%(x_val)s* __restrict__ Dval = (dtype_%(x_val)s*)PyArray_DATA(%(x_val)s);
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA(%(x_ind)s);
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA(%(x_ptr)s);
            const dtype_%(alpha)s alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];

            npy_intp Sz = PyArray_STRIDES(%(z)s)[1] / PyArray_DESCR(%(z)s)->elsize;
            npy_intp Szn = PyArray_STRIDES(%(zn)s)[1] / PyArray_DESCR(%(zn)s)->elsize;
            npy_intp Sval = PyArray_STRIDES(%(x_val)s)[0] / PyArray_DESCR(%(x_val)s)->elsize;
            npy_intp Sind = PyArray_STRIDES(%(x_ind)s)[0] / PyArray_DESCR(%(x_ind)s)->elsize;
            npy_intp Sptr = PyArray_STRIDES(%(x_ptr)s)[0] / PyArray_DESCR(%(x_ptr)s)->elsize;
            npy_intp Sy = PyArray_STRIDES(%(y)s)[1] / PyArray_DESCR(%(y)s)->elsize;

            // blas expects ints; convert here (rather than just making N etc ints) to avoid potential overflow in the negative-stride correction
            if ((N > 0x7fffffffL)||(Sy > 0x7fffffffL)||(Szn > 0x7fffffffL)||(Sy < -0x7fffffffL)||(Szn < -0x7fffffffL))
            {PyErr_SetString(PyExc_NotImplementedError, "array too big for BLAS (overflows int32 index)"); %(fail)s;}
            int N32 = N;
            int Sy32 = Sy;
            int Szn32 = Szn;

            if (!(%(inplace)s))
            {
                if (PyArray_CopyInto(%(zn)s, %(z)s))
                {
                    Py_XDECREF(%(zn)s);
                    %(fail)s;
                }
            }

            for (npy_intp k = 0; k < K; ++k)
            {
                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1)*Sptr]; ++m_idx)
                {
                    const npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K

                    const dtype_%(x_val)s Amk = alpha * Dval[m_idx * Sval]; // actual value at that location

                    dtype_%(y)s* y_row = (dtype_%(y)s*)(PyArray_BYTES(%(y)s) + PyArray_STRIDES(%(y)s)[0] * k);
                    // axpy expects pointer to the beginning of memory arrays,
                    // so when the stride is negative, we need to get the
                    // last element
                    if (Sy < 0)
                        y_row += (K - 1) * Sy;

                    dtype_%(zn)s* z_row = (dtype_%(zn)s*)(PyArray_BYTES(%(zn)s) + PyArray_STRIDES(%(zn)s)[0] * m);
                    if (Szn < 0)
                        z_row += (N - 1) * Szn;

                    %(axpy)s(&N32, (%(conv_type)s*)&Amk, (%(conv_type)s*)y_row, &Sy32, (%(conv_type)s*)z_row, &Szn32);
                }
            }
        }
        """ % dict(locals(), **sub)

        return rval

    def c_code_cache_version(self):
        return (3, blas.blas_header_version())
usmm_csc_dense = UsmmCscDense(inplace=False)
usmm_csc_dense_inplace = UsmmCscDense(inplace=True)


# This is tested in tests/test_basic.py:UsmmTests
local_usmm = gof.opt.PatternSub(
    (theano.tensor.sub, 'z',
     (theano.tensor.mul,
      {'pattern': 'alpha',
       'constraint': lambda expr: (np.all(expr.type.broadcastable) and
                                   theano.config.blas.ldflags)},
      (sparse._dot, 'x', 'y'))),
    (usmm, (theano.tensor.neg, 'alpha'), 'x', 'y', 'z'))
register_specialize(local_usmm, name="local_usmm")


# register a specialization to replace usmm_csc_dense -> usmm_csc_dense_inplace
# This is tested in tests/test_basic.py:UsmmTests
@gof.local_optimizer([usmm_csc_dense])
def local_usmm_csc_dense_inplace(node):
    if node.op == usmm_csc_dense:
        return [usmm_csc_dense_inplace(*node.inputs)]
register_specialize(local_usmm_csc_dense_inplace, 'cxx_only', 'inplace')


# This is tested in tests/test_basic.py:UsmmTests
@gof.local_optimizer([usmm])
def local_usmm_csx(node):
    """
    usmm -> usmm_csc_dense

    """
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
register_specialize(local_usmm_csx, 'cxx_only')


class CSMGradC(gof.Op):

    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, a_dim,
                  b_val, b_ind, b_ptr, b_dim):
        return gof.Apply(self, [a_val, a_ind, a_ptr, a_dim,
                         b_val, b_ind, b_ptr, b_dim], [b_val.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        # retrieve dtype number
        (a_val, a_ind, a_ptr, a_dim,
         b_val, b_ind, b_ptr, b_dim) = inputs
        (z,) = outputs
        typenum_z = node.outputs[0].type.dtype_specs()[2]
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a_val')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b_val')

        return """
        if (PyArray_NDIM(%(a_val)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ind)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(a_ptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(b_val)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b_val) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(b_ind)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b_ind) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(b_ptr)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b_ptr) != 1"); %(fail)s;}

        if (PyArray_TYPE(%(a_ind)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(a_ptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(b_ind)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "b_ind dtype not INT32"); %(fail)s;}

        if (PyArray_TYPE(%(b_ptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "b_ptr dtype not INT32"); %(fail)s;}

        if (PyArray_DIMS(%(a_val)s)[0] != PyArray_DIMS(%(a_ind)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); %(fail)s;}

        if (PyArray_DIMS(%(b_val)s)[0] != PyArray_DIMS(%(b_ind)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "b_val and b_ind have different lengths"); %(fail)s;}

        if (PyArray_DIMS(%(a_ptr)s)[0] != PyArray_DIMS(%(b_ptr)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a_ptr and b_ptr have different lengths"); %(fail)s;}

        if ((!%(z)s) || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(a_val)s)[0]))
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0};
            dims[0] = PyArray_DIMS(%(a_val)s)[0];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS(%(a_ptr)s)[0] - 1;
            npy_intp a_dim_0 = ((npy_int32 *)PyArray_DATA(%(a_dim)s))[0];
            npy_intp a_dim_1 = ((npy_int32 *)PyArray_DATA(%(a_dim)s))[1];

            npy_intp sp_dim = (M == a_dim_0)?a_dim_1:a_dim_0;

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Sz = PyArray_STRIDES(%(z)s)[0] / PyArray_DESCR(%(z)s)->elsize;
            npy_intp Sa_val = PyArray_STRIDES(%(a_val)s)[0] / PyArray_DESCR(%(a_val)s)->elsize;
            npy_intp Sa_ind = PyArray_STRIDES(%(a_ind)s)[0] / PyArray_DESCR(%(a_ind)s)->elsize;
            npy_intp Sa_ptr = PyArray_STRIDES(%(a_ptr)s)[0] / PyArray_DESCR(%(a_ptr)s)->elsize;
            npy_intp Sb_val = PyArray_STRIDES(%(b_val)s)[0] / PyArray_DESCR(%(b_val)s)->elsize;
            npy_intp Sb_ind = PyArray_STRIDES(%(b_ind)s)[0] / PyArray_DESCR(%(b_ind)s)->elsize;
            npy_intp Sb_ptr = PyArray_STRIDES(%(b_ptr)s)[0] / PyArray_DESCR(%(b_ptr)s)->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz = (dtype_%(z)s*)PyArray_DATA(%(z)s);
            const dtype_%(a_val)s* __restrict__ Da_val = (dtype_%(a_val)s*)PyArray_DATA(%(a_val)s);
            const npy_int32 * __restrict__ Da_ind = (npy_int32*)PyArray_DATA(%(a_ind)s);
            const npy_int32 * __restrict__ Da_ptr = (npy_int32*)PyArray_DATA(%(a_ptr)s);
            const dtype_%(b_val)s* __restrict__ Db_val = (dtype_%(b_val)s*)PyArray_DATA(%(b_val)s);
            const npy_int32 * __restrict__ Db_ind = (npy_int32*)PyArray_DATA(%(b_ind)s);
            const npy_int32 * __restrict__ Db_ptr = (npy_int32*)PyArray_DATA(%(b_ptr)s);

            npy_intp nnz = PyArray_DIMS(%(a_ind)s)[0];

            dtype_%(b_val)s b_row[sp_dim];

            //clear the output array
            for (npy_int64 i = 0; i < nnz; ++i)
            {
                Dz[i*Sz] = 0;
            }
            memset(b_row, 0, sp_dim*sizeof(dtype_%(b_val)s));

            // loop over inner dimension
            for (npy_int64 m = 0; m < M; ++m)
            {
                for (npy_int32 j_ptr = Db_ptr[m * Sb_ptr];
                    j_ptr < Db_ptr[(m + 1) * Sb_ptr]; j_ptr++) {
                    b_row[Db_ind[j_ptr * Sb_ind]] += Db_val[j_ptr*Sb_val];
                }

                for (npy_int32 j_ptr = Da_ptr[m * Sa_ptr];
                    j_ptr < Da_ptr[(m + 1) * Sa_ptr]; j_ptr++) {
                    Dz[j_ptr*Sz] = b_row[Da_ind[j_ptr * Sa_ind]];
                }

                for (npy_int32 j_ptr = Db_ptr[m * Sb_ptr];
                    j_ptr < Db_ptr[(m + 1) * Sb_ptr]; j_ptr++) {
                    b_row[Db_ind[j_ptr * Sb_ind]] = 0;
                }
            }
        }

        """ % dict(locals(), **sub)

    def c_code_cache_version(self):
        return (3,)
csm_grad_c = CSMGradC()


# register a specialization to replace csm_grad -> csm_grad_c
# This is tested in tests/test_opt.py:test_local_csm_grad_c
@gof.local_optimizer([csm_grad(None)])
def local_csm_grad_c(node):
    """
    csm_grad(None) -> csm_grad_c

    """
    if node.op == csm_grad(None):
        return [csm_grad_c(*node.inputs)]
    return False
# DISABLED AS IT IS BROKEN FOR UNSORTED INDICES!
# register_specialize(local_csm_grad_c, 'cxx_only')


class MulSDCSC(gof.Op):
    """
    Multiplication of sparse matrix by a broadcasted dense vector
    element wise.

    Parameters
    ----------
    a_data
        Sparse matrix data.
    a_indices
        Sparse matrix indices.
    a_indptr
        Sparse matrix indptr.
    b
        Tensor type matrix.

    Returns
    -------
    The multiplication of the two matrices element-wise.

    Notes
    -----
    `a_data`, `a_indices` and `a_indptr` must be the properties of a sparse
    matrix in csc format.

    The dtype of `a_data`, i.e. the dtype of the sparse matrix, cannot be a
    complex type.

    This op is used as an optimization of mul_s_d.

    """

    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 2
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (3,)

    # def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplementedError()

    def c_code(self, node, name, inputs, outputs, sub):

        (_data, _indices, _indptr, _b,) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (PyArray_NDIM(%(_b)s) != 2) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            %(fail)s;}
        if (PyArray_NDIM(%(_data)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;}
        if (PyArray_NDIM(%(_indices)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;}
        if (PyArray_NDIM(%(_indptr)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;}

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s ||
            (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0]) ||
            !(PyArray_ISCONTIGUOUS(%(_zout)s)))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                  PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_b)s));
            if (!%(_zout)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                %(fail)s;
            }
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)PyArray_DATA(%(_data)s);
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)PyArray_DATA(%(_zout)s);

            const npy_intp Sb = PyArray_STRIDES(%(_b)s)[0];

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
            {
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // extract i-th row of dense matrix
                    const dtype_%(_b)s* __restrict__ b_row = (dtype_%(_b)s*)(PyArray_BYTES(%(_b)s) + Sb * i);

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] * b_row[j];
                }
            }
        }

        """ % dict(locals(), **sub)

    def __str__(self):
        return self.__class__.__name__
mul_s_d_csc = MulSDCSC()


class MulSDCSR(gof.Op):
    """
    Multiplication of sparse matrix by a broadcasted dense vector
    element wise.

    Parameters
    ----------
    a_data
        Sparse matrix data.
    a_indices
        Sparse matrix indices.
    a_indptr
        Sparse matrix indptr.
    b
        Tensor type matrix.

    Returns
    -------
    The multiplication of the two matrix element wise.

    Notes
    -----
    `a_data`, `a_indices` and `a_indptr` must be the properties
    of a sparse matrix in csr format.

    The dtype of `a_data`, i.e. the dtype of the sparse matrix,
    cannot be a complex type.

    This op is used as an optimization of mul_s_d.

    """
    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 2
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (3,)

    # def perform(self, node, (a_data, a_indices, a_indptr, b), (out,)):
    #    return NotImplemented()

    def c_code(self, node, name, inputs, outputs, sub):

        (_data, _indices, _indptr, _b,) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (PyArray_NDIM(%(_b)s) != 2) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            %(fail)s;}
        if (PyArray_NDIM(%(_data)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;}
        if (PyArray_NDIM(%(_indices)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;}
        if (PyArray_NDIM(%(_indptr)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;}

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s ||
            (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0]) ||
            !(PyArray_ISCONTIGUOUS(%(_zout)s)))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_b)s));
            if (!%(_zout)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                %(fail)s;
            }
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)PyArray_DATA(%(_data)s);
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)PyArray_DATA(%(_zout)s);

            const npy_intp Sb = PyArray_STRIDES(%(_b)s)[0];

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
            {
                // extract i-th row of dense matrix
                const dtype_%(_b)s* __restrict__ b_row = (dtype_%(_b)s*)(PyArray_BYTES(%(_b)s) + Sb * j);

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

    def __str__(self):
        return self.__class__.__name__
mul_s_d_csr = MulSDCSR()


# register a specialization to replace MulSD -> MulSDCSX
@gof.local_optimizer([sparse.mul_s_d])
def local_mul_s_d(node):
    if node.op == sparse.mul_s_d:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 2:
            return False
        if svar.type.format == 'csc':
            CSx = sparse.CSC
            mul_s_d_csx = mul_s_d_csc
        elif svar.type.format == 'csr':
            CSx = sparse.CSR
            mul_s_d_csx = mul_s_d_csr
        else:
            raise NotImplemented()
        if x.dtype != y.dtype:
            # mul_s_d_csx don't support that case
            return

        c_data = mul_s_d_csx(sparse.csm_data(svar),
                             sparse.csm_indices(svar),
                             sparse.csm_indptr(svar), dvar)

        return [CSx(c_data,
                    sparse.csm_indices(svar),
                    sparse.csm_indptr(svar),
                    sparse.csm_shape(svar))]

    return False
register_specialize(local_mul_s_d, 'cxx_only')


class MulSVCSR(gof.Op):
    """
    Multiplication of sparse matrix by a broadcasted dense vector
    element wise.

    Parameters
    ----------
    a_data
        Sparse matrix data.
    a_indices
        Sparse matrix indices.
    a_indptr
        Sparse matrix indptr.
    b
        Tensor type matrix.

    Returns
    -------
    The multiplication of the two matrix element wise.

    Notes
    -----
    `a_data`, `a_indices` and `a_indptr` must be the properties
    of a sparse matrix in csr format.

    The dtype of `a_data`, i.e. the dtype of the sparse matrix,
    cannot be a complex type.

    This op is used as an optimization of MulSV.

    """
    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        assert b.type.ndim == 1
        return gof.Apply(self, [a_data, a_indices, a_indptr, b],
                               [tensor.tensor(b.dtype, (False,))])

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        _data, _indices, _indptr, _b, = inputs
        _zout, = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (PyArray_NDIM(%(_b)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_data)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_indices)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_indptr)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;
        }

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s
            || PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0]
            || !PyArray_ISCONTIGUOUS(%(_zout)s))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_b)s));
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)PyArray_DATA(%(_data)s);
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            const dtype_%(_b)s* __restrict__ Db = (dtype_%(_b)s*)PyArray_DATA(%(_b)s);

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)PyArray_DATA(%(_zout)s);

            const npy_intp Sb = PyArray_STRIDES(%(_b)s)[0] / PyArray_DESCR(%(_b)s)->elsize;

            // loop over rows
            for (npy_intp j = 0; j < N; ++j)
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

    def __str__(self):
        return self.__class__.__name__
mul_s_v_csr = MulSVCSR()


# register a specialization to replace MulSV -> MulSVCSR
@gof.local_optimizer([sparse.mul_s_v])
def local_mul_s_v(node):
    if node.op == sparse.mul_s_v:
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
            CSx = sparse.CSR
            mul_s_v_csx = mul_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = mul_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_mul_s_v, 'cxx_only')


class StructuredAddSVCSR(gof.Op):
    """
    Structured addition of a sparse matrix and a dense vector.
    The elements of the vector are are only added to the corresponding
    non-zero elements. Therefore, this operation outputs another sparse
    matrix.

    Parameters
    ----------
    a_data
        Sparse matrix data.
    a_indices
        Sparse matrix indices.
    a_indptr
        Sparse matrix indptr.
    b
        Tensor type vector.

    Returns
    -------
    A sparse matrix containing the addition of the vector to the data of the
    sparse matrix.

    Notes
    -----
    The a_* are the properties of a sparse matrix in csr format.

    This op is used as an optimization for StructuredAddSV.

    """

    __props__ = ()

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
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        _data, _indices, _indptr, _b, = inputs
        _zout, = outputs
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for b')

        return """
        if (PyArray_NDIM(%(_b)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_data)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_indices)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            %(fail)s;
        }
        if (PyArray_NDIM(%(_indptr)s) != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            %(fail)s;
        }

        if( PyArray_TYPE(%(_indices)s) != NPY_INT32) {
        PyErr_SetString(PyExc_NotImplementedError, "C"); %(fail)s;}

        if( PyArray_TYPE(%(_indptr)s) != NPY_INT32)
        {PyErr_SetString(PyExc_NotImplementedError, "D"); %(fail)s;}

        if (!%(_zout)s
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_indices)s)[0])
            || !(PyArray_ISCONTIGUOUS(%(_zout)s)))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS(%(_indices)s), PyArray_TYPE(%(_b)s));
            if (!%(_zout)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                %(fail)s;
            }
        }

        { //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS(%(_indices)s)[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS(%(_indptr)s)[0]-1;

            const dtype_%(_data)s * const __restrict__ data = (dtype_%(_data)s*)PyArray_DATA(%(_data)s);
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA(%(_indptr)s);
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA(%(_indices)s);

            const dtype_%(_b)s* __restrict__ Db = (dtype_%(_b)s*)PyArray_DATA(%(_b)s);

            dtype_%(_zout)s * const __restrict__ zout = (dtype_%(_zout)s*)PyArray_DATA(%(_zout)s);

            const npy_intp Sb = PyArray_STRIDES(%(_b)s)[0] / PyArray_DESCR(%(_b)s)->elsize;

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
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

    def __str__(self):
        return self.__class__.__name__
structured_add_s_v_csr = StructuredAddSVCSR()


# register a specialization to replace
# structured_add_s_v -> structured_add_s_v_csr
@gof.local_optimizer([sparse.structured_add_s_v])
def local_structured_add_s_v(node):
    if node.op == sparse.structured_add_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        # y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == 'csr':
            CSx = sparse.CSR
            structured_add_s_v_csx = structured_add_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = structured_add_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_structured_add_s_v, 'cxx_only')


class SamplingDotCSR(gof.Op):
    """
    Operand optimized for calculating the dot product dot(`x`, `y`.T) = `z`
    when you only want to calculate a subset of `z`.

    It is equivalent to `p` o (`x` . `y`.T) where o is the element-wise
    product, `x` and `y` operands of the dot product and `p` is a matrix
    that contains 1 when the corresponding element of `z` should be
    calculated and 0 when it shouldn't. Note that SamplingDot has a different
    interface than `dot` because SamplingDot requires `x` to be a `m`x`k`
    matrix while `y` is a `n`x`k` matrix instead of the usual `k`x`n` matrix.

    Parameters
    ----------
    x
        Tensor matrix.
    y
        Tensor matrix.
    p_data
        Sparse matrix data.
    p_ind
        Sparse matrix indices.
    p_ptr
        Sparse matric indptr.
    p_ncols
        Sparse matrix number of columns.

    Returns
    -------
    A dense matrix containing the dot product of `x` by `y`.T only
    where `p` is 1.

    Notes
    -----
    It will work if the pattern is not binary value, but if the
    pattern doesn't have a high sparsity proportion it will be slower
    then a more optimized dot followed by a normal elemwise
    multiplication.

    If we have the input of mixed dtype, we insert cast elemwise
    in the graph to be able to call blas function as they don't
    allow mixed dtype.

    This op is used as an optimization for SamplingDot.

    """

    __props__ = ()

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

    def c_code_cache_version(self):
        return (4, blas.blas_header_version())

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
        typenum_x = node.inputs[0].type.dtype_specs()[2]
        typenum_y = node.inputs[1].type.dtype_specs()[2]
        typenum_p = node.inputs[2].type.dtype_specs()[2]
        typenum_zd = tensor.TensorType(node.outputs[0].dtype,
                                       []).dtype_specs()[2]
        typenum_zi = tensor.TensorType(node.outputs[1].dtype,
                                       []).dtype_specs()[2]
        typenum_zp = tensor.TensorType(node.outputs[2].dtype,
                                       []).dtype_specs()[2]

        rval = """
        if (PyArray_NDIM(%(x)s) != 2) {
PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(y)s) != 2) {
PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}

        if (PyArray_TYPE(%(x)s) != %(typenum_x)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for x");
            %(fail)s;}

        if (PyArray_TYPE(%(y)s) != %(typenum_y)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for y");
            %(fail)s;}

        if (PyArray_TYPE(%(p_data)s) != %(typenum_p)s) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for pattern");
            %(fail)s;}

        if (PyArray_DIMS(%(x)s)[1] != PyArray_DIMS(%(y)s)[1]) {
            PyErr_SetString(PyExc_NotImplementedError,
              "x's number of columns doesn't match y's rows! Note: sampling_dot is different from dot because y is assumed to be transposed.");
            %(fail)s;}

        if (PyArray_DIMS(%(y)s)[0] != ((npy_int32 *)PyArray_DATA(%(p_ncols)s))[0] ||
            PyArray_DIMS(%(x)s)[0] != (PyArray_DIMS(%(p_ptr)s)[0] - 1))
        {PyErr_SetString(PyExc_NotImplementedError,
        "The dimension of the pattern and the output must match"); %(fail)s;}

        // Allocate output
        if (!%(z_data)s
            || (PyArray_DIMS(%(z_data)s)[0] != PyArray_DIMS(%(p_data)s)[0])
            || (PyArray_TYPE(%(z_data)s) != %(typenum_zd)s)
            || !(PyArray_ISCONTIGUOUS(%(z_data)s)))
         {
            {Py_XDECREF(%(z_data)s);}
            npy_intp dims[] = {0};
            dims[0] = PyArray_DIMS(%(p_data)s)[0];
            %(z_data)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                            %(typenum_zd)s);
        }
        if (!%(z_ind)s
            || (PyArray_DIMS(%(z_ind)s)[0] != PyArray_DIMS(%(p_ind)s)[0])
            || (PyArray_TYPE(%(z_ind)s) != %(typenum_zi)s)
            || !(PyArray_ISCONTIGUOUS(%(z_ind)s)))
        {
            {Py_XDECREF(%(z_ind)s);}
            npy_intp dims[] = {0};
            dims[0] = PyArray_DIMS(%(p_ind)s)[0];
            %(z_ind)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           %(typenum_zi)s);
        }
        if (!%(z_ptr)s
            || (PyArray_DIMS(%(z_ptr)s)[0] != PyArray_DIMS(%(p_ptr)s)[0])
            || (PyArray_TYPE(%(z_ptr)s) != %(typenum_zp)s)
            || !(PyArray_ISCONTIGUOUS(%(z_ptr)s)))
        {
            {Py_XDECREF(%(z_ptr)s);}
            npy_intp dims[] = {0};
            dims[0] = PyArray_DIMS(%(p_ptr)s)[0];
            %(z_ptr)s = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           %(typenum_zp)s);
        }

        {
            // Product of MxK and NxK, output MxN
            npy_intp M = PyArray_DIMS(%(x)s)[0];
            npy_intp N = PyArray_DIMS(%(y)s)[0];
            npy_intp K = PyArray_DIMS(%(y)s)[1];

            // pointers to access actual data in the arrays passed as params.
            const dtype_%(x)s* __restrict__ Dx = (dtype_%(x)s*)PyArray_DATA(%(x)s);
            const dtype_%(y)s* __restrict__ Dy = (dtype_%(y)s*)PyArray_DATA(%(y)s);
            const dtype_%(p_data)s* __restrict__ Dpd = (dtype_%(p_data)s*)PyArray_DATA(%(p_data)s);
            const dtype_%(p_ind)s* __restrict__ Dpi = (dtype_%(p_ind)s*)PyArray_DATA(%(p_ind)s);
            const dtype_%(p_ptr)s* __restrict__ Dpp = (dtype_%(p_ptr)s*)PyArray_DATA(%(p_ptr)s);
            dtype_%(z_data)s* __restrict__ Dzd = (dtype_%(z_data)s*)PyArray_DATA(%(z_data)s);
            dtype_%(z_ind)s* __restrict__ Dzi = (dtype_%(z_ind)s*)PyArray_DATA(%(z_ind)s);
            dtype_%(z_ptr)s* __restrict__ Dzp = (dtype_%(z_ptr)s*)PyArray_DATA(%(z_ptr)s);

            const npy_intp Sdx = PyArray_STRIDES(%(x)s)[1]/PyArray_DESCR(%(x)s)->elsize;
            const npy_intp Sdy = PyArray_STRIDES(%(y)s)[1]/PyArray_DESCR(%(y)s)->elsize;
            const npy_intp Sdpd = PyArray_STRIDES(%(p_data)s)[0] / PyArray_DESCR(%(p_data)s)->elsize;
            const npy_intp Sdpi = PyArray_STRIDES(%(p_ind)s)[0] / PyArray_DESCR(%(p_ind)s)->elsize;
            const npy_intp Sdpp = PyArray_STRIDES(%(p_ptr)s)[0] / PyArray_DESCR(%(p_ptr)s)->elsize;
            const npy_intp Sdzd = PyArray_STRIDES(%(z_data)s)[0] / PyArray_DESCR(%(z_data)s)->elsize;
            const npy_intp Sdzi = PyArray_STRIDES(%(z_ind)s)[0] / PyArray_DESCR(%(z_ind)s)->elsize;
            const npy_intp Sdzp = PyArray_STRIDES(%(z_ptr)s)[0] / PyArray_DESCR(%(z_ptr)s)->elsize;

            memcpy(Dzi, Dpi, PyArray_DIMS(%(p_ind)s)[0]*sizeof(dtype_%(p_ind)s));
            memcpy(Dzp, Dpp, PyArray_DIMS(%(p_ptr)s)[0]*sizeof(dtype_%(p_ptr)s));

            // blas expects ints; convert here (rather than just making K etc ints) to avoid potential overflow in the negative-stride correction
            if ((K > 0x7fffffffL)||(Sdx > 0x7fffffffL)||(Sdy > 0x7fffffffL)||(Sdx < -0x7fffffffL)||(Sdy < -0x7fffffffL))
            {PyErr_SetString(PyExc_NotImplementedError, "array too big for BLAS (overflows int32 index)"); %(fail)s;}
            int K32 = K;
            int Sdx32 = Sdx;
            int Sdy32 = Sdy;

            for (npy_intp m = 0; m < M; ++m) {
                for (npy_int32 n_idx = Dpp[m * Sdpp]; n_idx < Dpp[(m+1)*Sdpp]; ++n_idx) {
                    const npy_int32 n = Dpi[n_idx * Sdpi]; // row index of non-null value for column K

                    const dtype_%(x)s* x_row = (dtype_%(x)s*)(PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * m);

                    const dtype_%(y)s* y_col = (dtype_%(y)s*)(PyArray_BYTES(%(y)s) + PyArray_STRIDES(%(y)s)[0] * n);
                    // dot expects pointer to the beginning of memory arrays,
                    // so when the stride is negative, we need to get the
                    // last element
                    if (Sdx < 0)
                        x_row += (K - 1) * Sdx;
                    if (Sdy < 0)
                        y_col += (K - 1) * Sdy;

                    Dzd[n_idx * Sdzd] = Dpd[n_idx * Sdpd] * %(cdot)s(&K32, (const %(conv_type)s*)x_row, &Sdx32, (const %(conv_type)s*)y_col, &Sdy32);
                }
            }
        }
        """ % dict(locals(), **sub)

        return rval
sampling_dot_csr = SamplingDotCSR()


# register a specialization to replace SamplingDot -> SamplingDotCsr
@gof.local_optimizer([sparse.sampling_dot])
def local_sampling_dot_csr(node):
    if not theano.config.blas.ldflags:
        # The C implementation of SamplingDotCsr relies on BLAS routines
        return
    if node.op == sparse.sampling_dot:
        x, y, p = node.inputs
        if p.type.format == 'csr':
            p_data, p_ind, p_ptr, p_shape = sparse.csm_properties(p)

            z_data, z_ind, z_ptr = sampling_dot_csr(x, y, p_data,
                                                    p_ind, p_ptr, p_shape[1])

            return [sparse.CSR(z_data, z_ind, z_ptr, p_shape)]
    return False

register_specialize(local_sampling_dot_csr,
                    'cxx_only',
                    name='local_sampling_dot_csr')
