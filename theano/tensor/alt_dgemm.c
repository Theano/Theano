/** C Implementation of dgemm_ based on NumPy
 * Used instead of blas when Theano config flag blas.ldflags is empty.
 * PS: For further comments, see equivalent functions in alt_sgemm.c.
**/
void alt_numpy_double_scalar_matrix_product_in_place(double scalar, PyArrayObject* matrix) {
    NpyIter* iterator = NpyIter_New(matrix, 
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK, 
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix for a scalar * matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            double new_value = scalar * (*((double*)data));
            memcpy(data, &new_value, sizeof(double));
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}
void alt_numpy_double_matrix_sum(PyArrayObject* A, PyArrayObject* B, PyArrayObject* out) {
    PyArrayObject* op[3]       = {A, B, out};
    npy_uint32     op_flags[3] = {NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_WRITEONLY};
    npy_uint32     flags       = NPY_ITER_EXTERNAL_LOOP;
    NpyIter*       iterators   = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, NULL);
    if(iterators == NULL)
        alt_fatal_error("Unable to iterate over some matrices for matrix + matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterators, NULL);
    npy_intp innerstride = NpyIter_GetInnerStrideArray(iterators)[0];
    npy_intp *innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterators);
    char** data_ptr_array = NpyIter_GetDataPtrArray(iterators);
    do {
        char* from_A = data_ptr_array[0];
        char* from_B = data_ptr_array[1];
        char* from_out = data_ptr_array[2];
        npy_intp size = *innersize_ptr;
        for(npy_intp i = 0; i < size; ++i, from_A += innerstride, from_B += innerstride, from_out += innerstride) {
            double sum = *((double*)from_A);
            sum += *((double*)from_B);
            memcpy(from_out, &sum, sizeof(double));
        }
    } while(get_next(iterators));
    NpyIter_Deallocate(iterators);
}
/* dgemm */
void dgemm_(char* TRANSA, char* TRANSB, 
            const int* M, const int* N, const int* K,
            const double* ALPHA, double* A, const int* LDA, 
            double* B, const int* LDB, const double* BETA, 
            double* C, const int* LDC) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    if(*TRANSA == 'N' || *TRANSA == 'n') {
        nrowa = *M;
        ncola = *K;
    } else {
        nrowa = *K;
        ncola = *M;
    }
    if(*TRANSB == 'N' || *TRANSB == 'n') {
        nrowb = *K;
        ncolb = *N;
    } else {
        nrowb = *N;
        ncolb = *K;
    }
    npy_intp dims_A[2] = {nrowa, ncola};
    npy_intp dims_B[2] = {nrowb, ncolb};
    npy_intp dims_C[2] = {*M, *N};
    npy_intp strides_A[2] = {ncola*8, 8};
    npy_intp strides_B[2] = {ncolb*8, 8};
    npy_intp strides_C[2] =  {(*N)*8, 8};
    PyObject* matrix_A = PyArray_New(&PyArray_Type, 2, dims_A, NPY_FLOAT64, strides_A, A, 0, 0, NULL);
    PyObject* matrix_B = PyArray_New(&PyArray_Type, 2, dims_B, NPY_FLOAT64, strides_B, B, 0, 0, NULL);
    PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, NPY_FLOAT64, strides_C, C, 0, NPY_ARRAY_WRITEABLE, NULL);
    PyObject* op_A = alt_op(TRANSA, (PyArrayObject*)matrix_A);
    PyObject* op_B = alt_op(TRANSB, (PyArrayObject*)matrix_B);
    if(*BETA == 0) {
        PyArray_MatrixProduct2(op_A, op_B, (PyArrayObject*)matrix_C);
        alt_numpy_double_scalar_matrix_product_in_place(*ALPHA, (PyArrayObject*)matrix_C);
    } else {
        PyArrayObject* op_A_times_op_B = (PyArrayObject*)PyArray_MatrixProduct(op_A, op_B);
        alt_numpy_double_scalar_matrix_product_in_place(*ALPHA, op_A_times_op_B);
        alt_numpy_double_scalar_matrix_product_in_place(*BETA, (PyArrayObject*)matrix_C);
        alt_numpy_double_matrix_sum(op_A_times_op_B, (PyArrayObject*)matrix_C, (PyArrayObject*)matrix_C);
        Py_XDECREF(op_A_times_op_B);
    }
    if(op_B != matrix_B) Py_XDECREF(op_B);
    if(op_A != matrix_A) Py_XDECREF(op_A);
    Py_XDECREF(matrix_C);
    Py_XDECREF(matrix_B);
    Py_XDECREF(matrix_A);
}
