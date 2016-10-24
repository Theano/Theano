/** %(name)s **/
void alt_numpy_scalar_matrix_product_in_place_%(float_type)s(%(float_type)s scalar, PyArrayObject* matrix) {
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
            *((%(float_type)s*)data) *= scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}
/*Matrix+Matrix function. Compute (coeffA * matrixA) + (coeffB * matrixB)
 * Remark: This function actually sums a C-contiguous matrix (alpha*op(A)*op(B)) with a F-contiguous matrix (beta*C)
 * (see gemm implementation at next function for more details) */
void alt_numpy_matrix_extended_sum_in_place_%(float_type)s(
        const %(float_type)s* ALPHA, PyArrayObject* A,
        const %(float_type)s* BETA, PyArrayObject* B
) {
    PyArrayObject* op[2]       = {A, B};
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
        %(float_type)s* from_A = (%(float_type)s*)data_ptr_array[0];
        %(float_type)s* from_B = (%(float_type)s*)data_ptr_array[1];
        *from_B = (*ALPHA)*(*from_A) + (*BETA)*(*from_B);
    } while(get_next(iterators));
    NpyIter_Deallocate(iterators);
}
PyObject* alt_op_without_copy_%(float_type)s(int transposable, %(float_type)s* M, int nrow, int ncol, int LDM) {
    // By default, M is considered as a nrow*ncol F-contiguous matrix with LDM as stride indicator for the columns.
    npy_intp dims[2];
    npy_intp strides[2];
    int flags;
    if(transposable) {
        dims[0] = ncol;
        dims[1] = nrow;
        strides[0] = dims[1] * %(float_size)d;
        strides[1] = %(float_size)d;
        flags = NPY_ARRAY_C_CONTIGUOUS;
    } else {
        dims[0] = nrow;
        dims[1] = ncol;
        strides[0] = %(float_size)d;
        strides[1] = LDM * %(float_size)d;
        flags = NPY_ARRAY_F_CONTIGUOUS;
    }
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, M, 0, flags, NULL);
}
/* %(name)s template code */
void %(name)s(
    char* TRANSA, char* TRANSB, 
    const int* M, const int* N, const int* K,
    const %(float_type)s* ALPHA, %(float_type)s* A, const int* LDA, 
    %(float_type)s* B, const int* LDB, const %(float_type)s* BETA, 
    %(float_type)s* C, const int* LDC) {
    /* NB: it seems that matrix+matrix and scalar*matrix functions
     * defined above do not allocate iterator for a matrix with 0
     * content, that is a matrix whose nrow*ncol == 0. As these
     * functions actually work with M*N matrices (op(A)*op(B) and/or C),
     * I think that we could just return if M or N is null. */
    if(*M < 1 || *N < 1 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    int is_A_transposable = alt_trans_to_bool(TRANSA);
    int is_B_transposable = alt_trans_to_bool(TRANSB);
    if(is_A_transposable) {
        nrowa = *K; ncola = *M;
    } else {
        nrowa = *M; ncola = *K;
    }
    if(is_B_transposable) {
        nrowb = *N; ncolb = *K;
    } else {
        nrowb = *K; ncolb = *N;
    }
    if(*BETA == 0) {
        /*C is never read, just written.
         * matrix_C will be created as C-contiguous because
         * the 3rd parameter of PyArray_MatrixProduct2 (used below)
         * expects a C-contiguous array.
         * Also, to avoid some memory copy, transposition conditions
         * for A and B will be reversed, so that C will contain
         * C-contiguous op_B_transposed * op_A_transposed (N*M matrix).
         * As the calling code will consider C as F-contiguous M*N matrix,
         * it will get the transposed of op_B_transposed * op_A_transposed,
         * that is op_A * op_B (M*N matrix) as expected. */
        npy_intp dims_C[2] = {*N, *M};
        npy_intp strides_C[2] = {(*M) * %(float_size)d, %(float_size)d};
        PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, %(npy_float)s, strides_C, C, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);
        PyObject* op_A_transposed = alt_op_without_copy_%(float_type)s(!is_A_transposable, A, nrowa, ncola, *LDA);
        PyObject* op_B_transposed = alt_op_without_copy_%(float_type)s(!is_B_transposable, B, nrowb, ncolb, *LDB);
        PyArray_MatrixProduct2(op_B_transposed, op_A_transposed, (PyArrayObject*)matrix_C);
        if(*ALPHA != 1.0)
            alt_numpy_scalar_matrix_product_in_place_%(float_type)s(*ALPHA, (PyArrayObject*)matrix_C);
        Py_XDECREF(op_B_transposed);
        Py_XDECREF(op_A_transposed);
        Py_XDECREF(matrix_C);
    } else {
        /*C is read, so we must consider it as F-contiguous, as we do for A and B.*/
        npy_intp dims_C[2] = {*M, *N};
        npy_intp strides_C[2] = {%(float_size)d, (*LDC) * %(float_size)d};
        PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, %(npy_float)s, strides_C, C, 0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);
        PyObject* op_A = alt_op_without_copy_%(float_type)s(is_A_transposable, A, nrowa, ncola, *LDA);
        PyObject* op_B = alt_op_without_copy_%(float_type)s(is_B_transposable, B, nrowb, ncolb, *LDB);
        PyArrayObject* op_A_times_op_B = (PyArrayObject*)PyArray_MatrixProduct(op_A, op_B);
        alt_numpy_matrix_extended_sum_in_place_%(float_type)s(ALPHA, op_A_times_op_B, BETA, (PyArrayObject*)matrix_C);
        /*C is already F-contiguous, thus no conversion needed for output.*/
        Py_XDECREF(op_A_times_op_B);
        Py_XDECREF(op_B);
        Py_XDECREF(op_A);
        Py_XDECREF(matrix_C);
    }
}
