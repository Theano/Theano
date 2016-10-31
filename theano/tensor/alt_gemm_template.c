/** %(name)s **/

/* Scalar*Matrix function.
 * Computes: matrix = scalar*matrix. */
void alt_numpy_scale_matrix_inplace_%(float_type)s(const %(float_type)s* scalar, PyArrayObject* matrix) {
    NpyIter* iterator = NpyIter_New(matrix, 
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK, 
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix "
                        "for a scalar * matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((%(float_type)s*)data) *= *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}
/* Matrix+Matrix function.
 * Computes: matrix2 = (scalar1 * matrix1) + (scalar2 * matrix2) */
void alt_numpy_matrix_extended_sum_inplace_%(float_type)s(
        const %(float_type)s* scalar1, PyArrayObject* matrix1,
        const %(float_type)s* scalar2, PyArrayObject* matrix2
) {
    PyArrayObject* op[2]       = {matrix1, matrix2};
    npy_uint32     op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_READWRITE};
    npy_uint32     flags       = 0;
    NpyIter*       iterators   = NpyIter_MultiNew(
            2, op, flags, NPY_CORDER, NPY_NO_CASTING, op_flags, NULL);
    if(iterators == NULL)
        alt_fatal_error("Unable to iterate over some matrices "
                        "for matrix + matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterators, NULL);
    char** data_ptr_array = NpyIter_GetDataPtrArray(iterators);
    do {
        %(float_type)s* from_matrix1 = (%(float_type)s*)data_ptr_array[0];
        %(float_type)s* from_matrix2 = (%(float_type)s*)data_ptr_array[1];
        *from_matrix2 = (*scalar1)*(*from_matrix1) + (*scalar2)*(*from_matrix2);
    } while(get_next(iterators));
    NpyIter_Deallocate(iterators);
}
/* NumPy Wrapping function. Wraps a data into a NumPy's PyArrayObject.
 * By default, data is considered as Fortran-style array (column by column).
 * If to_transpose, data will be considered as C-style array (row by row)
 * with dimensions reversed. */
PyObject* alt_op_%(float_type)s(int to_transpose, %(float_type)s* M, int nrow, int ncol, int LDM) {
    npy_intp dims[2];
    npy_intp strides[2];
    if(to_transpose) {
        dims[0] = ncol;
        dims[1] = nrow;
        strides[0] = LDM * %(float_size)d;
        strides[1] = %(float_size)d;
    } else {
        dims[0] = nrow;
        dims[1] = ncol;
        strides[0] = %(float_size)d;
        strides[1] = LDM * %(float_size)d;
    }
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, M, 0, 0, NULL);
}
/* Special wrapping case used for matrix C in gemm implementation. */
inline PyObject* alt_wrap_fortran_writeable_matrix_%(float_type)s(
    %(float_type)s* matrix, const int* nrow, const int* ncol, const int* LD
) {
    npy_intp dims[2] = {*nrow, *ncol};
    npy_intp strides[2] = {%(float_size)d, (*LD) * %(float_size)d};
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, matrix, 0, NPY_ARRAY_WRITEABLE, NULL);
}
/* %(name)s template code */
void %(name)s(
    char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
    const %(float_type)s* ALPHA, %(float_type)s* A, const int* LDA, 
    %(float_type)s* B, const int* LDB, const %(float_type)s* BETA, 
    %(float_type)s* C, const int* LDC
) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        alt_fatal_error("The integer arguments passed to %(name)s must all be at least 0.");
    /* If M or N is null, there is nothing to do with C,
     * as C should contain M*N == 0 items. */
    if(*M == 0 || *N == 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    int to_transpose_A = alt_trans_to_bool(TRANSA);
    int to_transpose_B = alt_trans_to_bool(TRANSB);
    if(to_transpose_A) {
        nrowa = *K;
        ncola = *M;
    } else {
        nrowa = *M;
        ncola = *K;
    }
    if(to_transpose_B) {
        nrowb = *N;
        ncolb = *K;
    } else {
        nrowb = *K;
        ncolb = *N;
    }
    int computation_flags;
    void* computation_pointer;
    npy_intp* computation_strides;
    npy_intp computation_dims[2] = {*N, *M};
    npy_intp default_computation_strides[2] = {(*LDC) * %(float_size)d, %(float_size)d};
    if(*BETA == 0 && *LDC == *M) {
        /* BETA == 0, so C is never read.
         * LDC == M, so C is contiguous in memory
         * (that condition is needed for dot operation, se below).
         * Then we can compute ALPHA*op(A)*op(B) directly in C. */
        computation_flags = NPY_ARRAY_WRITEABLE;
        computation_pointer = C;
        computation_strides = default_computation_strides;
    } else {
        /* Either BETA != 0 (C will be read)
         * or LDC != M (C is not read but is not contiguous in memory).
         * Then in both cases, we need to allocate a new memory
         * to compute ALPHA*op(A)*op(B). */
        computation_flags = 0;
        computation_pointer = NULL;
        computation_strides = NULL;
    }
    /* The memory buffer used to compute op(A)*op(B) (either C or
     * new allocated buffer) will be considered as C-contiguous because
     * the 3rd parameter of PyArray_MatrixProduct2 (used below)
     * expects a C-contiguous array.
     * Also, to avoid some memory copy, transposition conditions
     * for A and B will be reversed, so that the buffer will contain
     * C-contiguous opB_transposed * opA_transposed (N*M matrix).
     * After that, the code that uses the buffer (either the code calling
     * this function, or this function if BETA != 0) just has to 
     * consider the buffer as a F-contiguous M*N matrix, so that
     * it will get the transposed of op_B_transposed * op_A_transposed,
     * that is op_A * op_B (M*N matrix) as expected. */
    PyObject* opA_transposed = alt_op_%(float_type)s(!to_transpose_A, A, nrowa, ncola, *LDA);
    PyObject* opB_transposed = alt_op_%(float_type)s(!to_transpose_B, B, nrowb, ncolb, *LDB);
    PyObject* opB_trans_dot_opA_trans = PyArray_New(&PyArray_Type, 2, computation_dims, %(npy_float)s, computation_strides, computation_pointer, 0, computation_flags, NULL);
    PyArray_MatrixProduct2(opB_transposed, opA_transposed, (PyArrayObject*)opB_trans_dot_opA_trans);
    if(*BETA == 0) {
        if(*ALPHA != 1.0)
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opB_trans_dot_opA_trans);
        if(*LDC != *M) {
            /* A buffer has been created to compute ALPHA*op(A)*op(B),
             * so we must copy it to the real output, that is C. */
            PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
            PyObject* alpha_opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrix_C, (PyArrayObject*)alpha_opA_dot_opB))
                alt_fatal_error("NumPy %(name)s implementation: unable to copy ALPHA*op(A)*op(B) into C when BETA == 0.");
            Py_XDECREF(alpha_opA_dot_opB);
            Py_XDECREF(matrix_C);
        }
    } else {
        /* C is read, so we must consider it as Fortran-style matrix. */
        PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
        PyObject* opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
        alt_numpy_matrix_extended_sum_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opA_dot_opB, BETA, (PyArrayObject*)matrix_C);
        Py_XDECREF(opA_dot_opB);
        Py_XDECREF(matrix_C);
    }
    Py_XDECREF(opB_trans_dot_opA_trans);
    Py_XDECREF(opB_transposed);
    Py_XDECREF(opA_transposed);
}
