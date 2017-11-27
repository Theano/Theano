/** Alternative template NumPy-based implementation of BLAS functions used in Theano. **/

/* Compute matrix[i][j] = scalar for every position (i, j) in matrix. */
void alt_numpy_memset_inplace_%(float_type)s(PyArrayObject* matrix, const %(float_type)s* scalar) {
    if (PyArray_IS_C_CONTIGUOUS(matrix) && *scalar == (char)(*scalar)) {
        // This will use memset.
        PyArray_FILLWBYTE(matrix, (char)(*scalar));
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix for a memory assignation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((%(float_type)s*)data) = *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Scalar * Matrix function.
 * Computes: matrix = scalar * matrix. */
void alt_numpy_scale_matrix_inplace_%(float_type)s(const %(float_type)s* scalar, PyArrayObject* matrix) {
    if (*scalar == 1)
        return;
    if (*scalar == 0) {
        alt_numpy_memset_inplace_%(float_type)s(matrix, scalar);
        return;
    }
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

/* Matrix + Matrix function.
 * Computes: matrix2 = (scalar1 * matrix1) + (scalar2 * matrix2) */
void alt_numpy_matrix_extended_sum_inplace_%(float_type)s(
        const %(float_type)s* scalar1, PyArrayObject* matrix1,
        const %(float_type)s* scalar2, PyArrayObject* matrix2
) {
    if (*scalar1 == 0 && *scalar2 == 0) {
        alt_numpy_memset_inplace_%(float_type)s(matrix2, scalar2);
        return;
    }
    if (*scalar1 == 0) {
        alt_numpy_scale_matrix_inplace_%(float_type)s(scalar2, matrix2);
        return;
    }
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
    if (*scalar2 == 0) {
        do {
            %(float_type)s* from_matrix1 = (%(float_type)s*)data_ptr_array[0];
            %(float_type)s* from_matrix2 = (%(float_type)s*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1);
        } while(get_next(iterators));
    } else {
        do {
            %(float_type)s* from_matrix1 = (%(float_type)s*)data_ptr_array[0];
            %(float_type)s* from_matrix2 = (%(float_type)s*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1) + (*scalar2)*(*from_matrix2);
        } while(get_next(iterators));
    }
    NpyIter_Deallocate(iterators);
}

/* NumPy Wrapping function. Wraps a data into a NumPy's PyArrayObject.
 * By default, data is considered as Fortran-style array (column by column).
 * If to_transpose, data will be considered as C-style array (row by row)
 * with dimensions reversed. */
PyObject* alt_op_%(float_type)s(int to_transpose, %(float_type)s* M, int nrow, int ncol, int LDM, int numpyFlags) {
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
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, M, 0, numpyFlags, NULL);
}

/* Special wrapping case used for matrix C in gemm_ implementation. */
inline PyObject* alt_wrap_fortran_writeable_matrix_%(float_type)s(
    %(float_type)s* matrix, const int* nrow, const int* ncol, const int* LD
) {
    npy_intp dims[2] = {*nrow, *ncol};
    npy_intp strides[2] = {%(float_size)d, (*LD) * %(float_size)d};
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, matrix, 0, NPY_ARRAY_WRITEABLE, NULL);
}

/* gemm */
void %(precision)sgemm_(
    char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
    const %(float_type)s* ALPHA, %(float_type)s* A, const int* LDA,
    %(float_type)s* B, const int* LDB, const %(float_type)s* BETA,
    %(float_type)s* C, const int* LDC
) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        alt_fatal_error("The integer arguments passed to %(precision)sgemm_ must all be at least 0.");
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
    PyObject* opA_transposed = alt_op_%(float_type)s(!to_transpose_A, A, nrowa, ncola, *LDA, 0);
    PyObject* opB_transposed = alt_op_%(float_type)s(!to_transpose_B, B, nrowb, ncolb, *LDB, 0);
    PyObject* opB_trans_dot_opA_trans = PyArray_New(&PyArray_Type, 2, computation_dims, %(npy_float)s,
                                                    computation_strides, computation_pointer, 0,
                                                    computation_flags, NULL);
    PyArray_MatrixProduct2(opB_transposed, opA_transposed, (PyArrayObject*)opB_trans_dot_opA_trans);
    /* PyArray_MatrixProduct2 adds a reference to the output array,
     * which we need to remove to avoid a memory leak. */
    Py_XDECREF(opB_trans_dot_opA_trans);
    if(*BETA == 0) {
        if(*ALPHA != 1.0)
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opB_trans_dot_opA_trans);
        if(*LDC != *M) {
            /* A buffer has been created to compute ALPHA*op(A)*op(B),
             * so we must copy it to the real output, that is C. */
            PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
            PyObject* alpha_opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrix_C, (PyArrayObject*)alpha_opA_dot_opB))
                alt_fatal_error("NumPy %(precision)sgemm_ implementation: unable to copy ALPHA*op(A)*op(B) into C when BETA == 0.");
            Py_XDECREF(alpha_opA_dot_opB);
            Py_XDECREF(matrix_C);
        }
    } else {
        /* C is read, so we must consider it as Fortran-style matrix. */
        PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
        PyObject* opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
        alt_numpy_matrix_extended_sum_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opA_dot_opB,
                                                             BETA, (PyArrayObject*)matrix_C);
        Py_XDECREF(opA_dot_opB);
        Py_XDECREF(matrix_C);
    }
    Py_XDECREF(opB_trans_dot_opA_trans);
    Py_XDECREF(opB_transposed);
    Py_XDECREF(opA_transposed);
}

/* gemv */
void %(precision)sgemv_(
    char* TRANS,
    const int* M,
    const int* N,
    const %(float_type)s* ALPHA,
    %(float_type)s* A,
    const int* LDA,
    %(float_type)s* x,
    const int* incx,
    const %(float_type)s* BETA,
    %(float_type)s* y,
    const int* incy
) {
    /**
    If TRANS is 'n' or 'N', computes:
        y = ALPHA * A * x + BETA * y
    Else, computes:
        y = ALPHA * A.T * x + BETA * y
    A is a M*N matrix, A.T is A transposed
    x, y are vectors
    ALPHA, BETA are scalars
    **/

    // If alpha == 0 and beta == 1, we have nothing to do, as alpha*A*x + beta*y == y.
    if (*ALPHA == 0 && *BETA == 1)
        return;
    if (*M < 0 || *N < 0 || *LDA < 0)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: M, N and LDA must be at least 0.");
    if (*incx == 0 || *incy == 0)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: incx and incy must not be 0.");
    int transpose = alt_trans_to_bool(TRANS);
    int size_x = 0, size_y = 0;
    if (transpose) {
        size_x = *M;
        size_y = *N;
    } else {
        size_x = *N;
        size_y = *M;
    }
    if (*M == 0 || *N == 0) {
        /* A contains M * N == 0 values. y should be empty too, and we have nothing to do. */
        if (size_y != 0)
            alt_fatal_error("NumPy %(precision)sgemv_ implementation: the output vector should be empty.");
        return;
    }
    /* Vector pointers points to the beginning of memory (see function `theano.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*incx < 0)
        x += (size_x - 1) * (-*incx);
    if (*incy < 0)
        y += (size_y - 1) * (-*incy);
    PyObject* matrixA = alt_op_%(float_type)s(transpose, A, *M, *N, *LDA, 0);
    PyObject* matrixX = alt_op_%(float_type)s(1, x, 1, size_x, *incx, 0);
    PyObject* matrixY = alt_op_%(float_type)s(1, y, 1, size_y, *incy, NPY_ARRAY_WRITEABLE);
    if (matrixA == NULL || matrixX == NULL || matrixY == NULL)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to wrap A, x or y arrays.")
    if (*ALPHA == 0) {
        // Just BETA * y
        alt_numpy_scale_matrix_inplace_%(float_type)s(BETA, (PyArrayObject*)matrixY);
    } else if (*BETA == 0) {
        // We can directly compute alpha * A * x into y if y is C-contiguous.
        if (PyArray_IS_C_CONTIGUOUS((PyArrayObject*)matrixY)) {
            PyArray_MatrixProduct2(matrixA, matrixX, (PyArrayObject*)matrixY);
            // PyArray_MatrixProduct2 adds an extra reference to the output array.
            Py_XDECREF(matrixY);
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)matrixY);
        } else {
            // If y is not contiguous, we need a temporar workspace.
            PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
            if (tempAX == NULL)
                alt_fatal_error("NumPy %(precision)sgemv_ implementation: Unable to get matrix product.");
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)tempAX);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrixY, (PyArrayObject*)tempAX)) {
                alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to update output.");
            }
            Py_XDECREF(tempAX);
        }
    } else {
        // We must perform full computation.
        PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
        if (tempAX == NULL)
            alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to get matrix product.");
        // ALPHA * (A * x) + BETA * y.
        alt_numpy_matrix_extended_sum_inplace_%(float_type)s(ALPHA, (PyArrayObject*)tempAX,
                                                             BETA, (PyArrayObject*)matrixY);
        Py_XDECREF(tempAX);
    }
    Py_XDECREF(matrixY);
    Py_XDECREF(matrixX);
    Py_XDECREF(matrixA);
}

/* dot */
%(float_type)s %(precision)sdot_(
    const int* N,
    %(float_type)s *SX,
    const int *INCX,
    %(float_type)s *SY,
    const int *INCY
) {
    if (*N < 0)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: N must be at least 0.");
    if (*INCX == 0 || *INCY == 0)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: INCX and INCY must not be 0.");
    %(float_type)s result = 0;
    int one = 1;
    /* Vector pointers points to the beginning of memory (see function `theano.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*INCX < 0)
        SX += (*N - 1) * (-*INCX);
    if (*INCY < 0)
        SY += (*N - 1) * (-*INCY);
    // Create vector_x with shape (1, N)
    PyObject* vector_x = alt_op_%(float_type)s(0, SX, 1, *N, *INCX, 0);
    // Create vector_y with shape (N, 1)
    PyObject* vector_y = alt_op_%(float_type)s(1, SY, 1, *N, *INCY, 0);
    // Create output scalar z with shape (1, 1) to wrap `result`.
    PyArrayObject* dot_product = (PyArrayObject*)alt_wrap_fortran_writeable_matrix_%(float_type)s(&result, &one, &one, &one);

    if (vector_x == NULL || vector_y == NULL || dot_product == NULL)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: unable to wrap x, y or output arrays.");

    // Compute matrix product: (1, N) * (N, 1) => (1, 1)
    PyArray_MatrixProduct2(vector_x, vector_y, dot_product);
    // PyArray_MatrixProduct2 adds an extra reference to the output array.
    Py_XDECREF(dot_product);

    if (PyErr_Occurred())
        alt_fatal_error("NumPy %(precision)sdot_ implementation: unable to compute dot.");

    Py_XDECREF(dot_product);
    Py_XDECREF(vector_y);
    Py_XDECREF(vector_x);
    return result;
}
