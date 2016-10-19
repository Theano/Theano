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
/*Matrix+Matrix function.
 * Remark: This function actually sums a C-contiguous matrix (alpha*op(A)*op(B)) with a F-contiguous matrix (beta*C)
 * (see gemm implementation at next function for more details) */
void alt_numpy_matrix_sum_in_place_%(float_type)s(PyArrayObject* A, PyArrayObject* B) {
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
        char* from_A = data_ptr_array[0];
        char* from_B = data_ptr_array[1];
        *((%(float_type)s*)from_B) += *((%(float_type)s*)from_A);
    } while(get_next(iterators));
    NpyIter_Deallocate(iterators);
}
/* %(name)s template code */
void %(name)s(
    char* TRANSA, char* TRANSB, 
    const int* M, const int* N, const int* K,
    const %(float_type)s* ALPHA, %(float_type)s* A, const int* LDA, 
    %(float_type)s* B, const int* LDB, const %(float_type)s* BETA, 
    %(float_type)s* C, const int* LDC) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    if(*TRANSA == 'N' || *TRANSA == 'n') {
        nrowa = *M; ncola = *K;
    } else {
        nrowa = *K; ncola = *M;
    }
    if(*TRANSB == 'N' || *TRANSB == 'n') {
        nrowb = *K; ncolb = *N;
    } else {
        nrowb = *N; ncolb = *K;
    }
    npy_intp dims_A[2] = {nrowa, ncola};
    npy_intp dims_B[2] = {nrowb, ncolb};
    npy_intp dims_C[2] = {*M, *N};
    npy_intp strides_A[2] = {%(float_size)d, (*LDA) * %(float_size)d};
    npy_intp strides_B[2] = {%(float_size)d, (*LDB) * %(float_size)d};
    PyObject* matrix_A = PyArray_New(&PyArray_Type, 2, dims_A, %(npy_float)s, strides_A, A, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
    PyObject* matrix_B = PyArray_New(&PyArray_Type, 2, dims_B, %(npy_float)s, strides_B, B, 0, NPY_ARRAY_F_CONTIGUOUS, NULL);
    PyObject* op_A = alt_op(TRANSA, (PyArrayObject*)matrix_A);
    PyObject* op_B = alt_op(TRANSB, (PyArrayObject*)matrix_B);
    if(*BETA == 0) {
        /*C is never red, just written.*/
        npy_intp strides_C[2] = {(*N) * %(float_size)d, %(float_size)d};
        /*matrix_C is created as C-contiguous because the 3rd parameter of PyArray_MatrixProduct2 (below) expects a C-contiguous array.*/
        PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, %(npy_float)s, strides_C, C, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);
        PyArray_MatrixProduct2(op_A, op_B, (PyArrayObject*)matrix_C);
        if(*ALPHA != 1.0)
            alt_numpy_scalar_matrix_product_in_place_%(float_type)s(*ALPHA, (PyArrayObject*)matrix_C);
        /*But it seems Python|NumPy expects C to be F-contiguous at output, so the convert it.*/
        PyObject* matrix_C_as_f_contiguous = PyArray_FromAny(matrix_C, PyArray_DESCR((PyArrayObject*)matrix_C), 2, 2, NPY_ARRAY_F_CONTIGUOUS, NULL);
        if(matrix_C_as_f_contiguous != matrix_C) {
            memcpy(C, PyArray_DATA((PyArrayObject*)matrix_C_as_f_contiguous), (*M)*(*N)*sizeof(%(float_type)s));
            Py_XDECREF(matrix_C_as_f_contiguous);
        }
        Py_XDECREF(matrix_C);
    } else {
        /*C is read, so we must consider it as F-contiguous, as we do for A and B.*/
        npy_intp strides_C[2] = {%(float_size)d, (*LDC) * %(float_size)d};
        PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, %(npy_float)s, strides_C, C, 0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);
        PyArrayObject* op_A_times_op_B = (PyArrayObject*)PyArray_MatrixProduct(op_A, op_B);
        if(*ALPHA != 1.0)
            alt_numpy_scalar_matrix_product_in_place_%(float_type)s(*ALPHA, op_A_times_op_B);
        if(*BETA != 1.0)
            alt_numpy_scalar_matrix_product_in_place_%(float_type)s(*BETA, (PyArrayObject*)matrix_C);
        alt_numpy_matrix_sum_in_place_%(float_type)s(op_A_times_op_B, (PyArrayObject*)matrix_C);
        /*C is already F-contiguous, thus no conversion needed for output.*/
        Py_XDECREF(op_A_times_op_B);
        Py_XDECREF(matrix_C);
    }
    if(op_B != matrix_B) Py_XDECREF(op_B);
    if(op_A != matrix_A) Py_XDECREF(op_A);
    Py_XDECREF(matrix_B);
    Py_XDECREF(matrix_A);
}
