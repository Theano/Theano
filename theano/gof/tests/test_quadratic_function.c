#section support_code_apply
int APPLY_SPECIFIC(quadratic_function)(PyArrayObject* tensor, DTYPE_INPUT_0 a, DTYPE_INPUT_0 b, DTYPE_INPUT_0 c) {
    NpyIter* iterator = NpyIter_New(tensor,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to iterate over a tensor for an elemwise operation.");
        return -1;
    }
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            DTYPE_INPUT_0 x = *((DTYPE_INPUT_0*)data);
            *((DTYPE_INPUT_0*)data) = a*x*x + b*x + c;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
    return 0;
}

int APPLY_SPECIFIC(compute_quadratic)(PyArrayObject* X, PyArrayObject** Y, PARAMS_TYPE* coeff) {
    DTYPE_INPUT_0 a = (DTYPE_INPUT_0) (*(DTYPE_PARAM_a*) PyArray_GETPTR1(coeff->a, 0)); // 0-D TensorType.
    DTYPE_INPUT_0 b =                                                    coeff->b;      // Scalar.
    DTYPE_INPUT_0 c =                   (DTYPE_INPUT_0) PyFloat_AsDouble(coeff->c);     // Generic.
    Py_XDECREF(*Y);
    *Y = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(X), PyArray_DIMS(X), TYPENUM_INPUT_0, PyArray_IS_F_CONTIGUOUS(X));
    if (PyArray_CopyInto(*Y, X) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to copy input into output.");
        return 1;
    };
    if (APPLY_SPECIFIC(quadratic_function)(*Y, a, b, c) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to compute quadratic function.");
        return 1;
    }
    return 0;
}
