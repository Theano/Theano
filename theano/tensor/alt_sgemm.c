/** C Implementation of sgemm_ based on NumPy
 * Used instead of blas when Theano config flag blas.ldflags is empty.
 * PS: Comments are the same for equivalent functions in alt_dgemm.c.
**/
void alt_fatal_error(const char* message) {
    if(message != NULL) puts(message);
    exit(-1);
}
void alt_numpy_scalar_matrix_product_in_place(float scalar, PyArrayObject* matrix) {
    // Get an iterator on matrix.
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
            float new_value = scalar * (*((float*)data));
            memcpy(data, &new_value, sizeof(float));
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}
void alt_numpy_matrix_sum(PyArrayObject* A, PyArrayObject* B, PyArrayObject* out) {
    /* NB: It may be better to check if A,B and out have the same dimensions.
     * But for now, we made this assumption. */
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
            float sum = *((float*)from_A);
            sum += *((float*)from_B);
            memcpy(from_out, &sum, sizeof(float));
        }
    } while(get_next(iterators));
    NpyIter_Deallocate(iterators);
}
inline PyObject* alt_op(char* trans, PyArrayObject* matrix) {
    return (*trans == 'N' || *trans == 'n') ? (PyObject*)matrix : PyArray_Transpose(matrix, NULL);
}
/* sgemm
 * Recall: operation performed:
 * C = ALPHA * op(TRANSA, A) * op(TRANSB, B) + BETA * C
 * NB: We assume that none of the 13 pointers passed as arguments is null.
 * NB: We can more optimize this function (for example, when ALPHA == 0).
 * */
void sgemm_(char* TRANSA, char* TRANSB, 
            const int* M, const int* N, const int* K,
            const float* ALPHA, float* A, const int* LDA, 
            float* B, const int* LDB, const float* BETA, 
            float* C, const int* LDC) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        return;
    /* Recall:
     * op(A) is a m by k matrix.
     * op(B) is a k by n matrix.
     *    C  is a m by n matrix.
     */
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
    /*NB: It seems that A,B and C are always row-major matrices, thus
     * the stride for the 1st dimension (from a row to the next row) only depends on the number of elements in a row
     * (that is, the size of the 2nd dimension, which is ncola for A, ncolb for B, and *N for C),
     * and the stride for the 2nd dimension (from a column to the next column) is always the size of one element
     * (that is, 4 bytes for a float32 matrix, 8 bytes for a float64 matrix).
     * Then, LDA, LDB and LDC seems totally useless in the strides calcuulations.
     * For LDA,LDB,LDC to be taken account, we need to have column-major matrices,
     * which seems never to happen. */
    npy_intp strides_A[2] = {ncola*4, 4};
    npy_intp strides_B[2] = {ncolb*4, 4};
    npy_intp strides_C[2] =  {(*N)*4, 4};
    /*NB: in fact, we could replace the strides with NULL as argument in the 3 following lines.*/
    PyObject* matrix_A = PyArray_New(&PyArray_Type, 2, dims_A, NPY_FLOAT32, strides_A, A, 0, 0, NULL);
    PyObject* matrix_B = PyArray_New(&PyArray_Type, 2, dims_B, NPY_FLOAT32, strides_B, B, 0, 0, NULL);
    PyObject* matrix_C = PyArray_New(&PyArray_Type, 2, dims_C, NPY_FLOAT32, strides_C, C, 0, NPY_ARRAY_WRITEABLE, NULL);
    PyObject* op_A = alt_op(TRANSA, (PyArrayObject*)matrix_A);
    PyObject* op_B = alt_op(TRANSB, (PyArrayObject*)matrix_B);
    if(*BETA == 0) {
        PyArray_MatrixProduct2(op_A, op_B, (PyArrayObject*)matrix_C);
        alt_numpy_scalar_matrix_product_in_place(*ALPHA, (PyArrayObject*)matrix_C);
    } else {
        PyArrayObject* op_A_times_op_B = (PyArrayObject*)PyArray_MatrixProduct(op_A, op_B);
        alt_numpy_scalar_matrix_product_in_place(*ALPHA, op_A_times_op_B);
        alt_numpy_scalar_matrix_product_in_place(*BETA, (PyArrayObject*)matrix_C);
        alt_numpy_matrix_sum(op_A_times_op_B, (PyArrayObject*)matrix_C, (PyArrayObject*)matrix_C);
        Py_XDECREF(op_A_times_op_B);
    }
    if(op_B != matrix_B) Py_XDECREF(op_B);
    if(op_A != matrix_A) Py_XDECREF(op_A);
    Py_XDECREF(matrix_C);
    Py_XDECREF(matrix_B);
    Py_XDECREF(matrix_A);
}
