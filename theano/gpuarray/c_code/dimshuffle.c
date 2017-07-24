#section support_code_apply

int gpu_dimshuffle(PyGpuArrayObject* input, PyGpuArrayObject** out, PARAMS_TYPE* params) {
    PyGpuArrayObject *tmp = NULL;
    npy_intp nd_in = PyArray_SIZE(params->input_broadcastable);
    npy_intp nd_out = PyArray_SIZE(params->_new_order);
    npy_int64* new_order = NULL;
    unsigned int* transposition = NULL;
    size_t* sh = NULL;
    int e;

    if (input->ga.nd != nd_in) {
        PyErr_SetString(PyExc_TypeError, "input nd");
        return 1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(params->_new_order)) {
        PyErr_SetString(PyExc_RuntimeError, "DimShuffle: param _new_order must be C-contiguous.");
        return 1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(params->transposition)) {
        PyErr_SetString(PyExc_RuntimeError, "GpuDimShuffle: param transposition must be C-contiguous.");
        return 1;
    }

    Py_XDECREF(*out);

    /** Do shuffle. **/

    new_order = (npy_int64*) PyArray_DATA(params->_new_order);
    transposition = (unsigned int*) malloc(nd_in * sizeof(unsigned int));
    sh = (size_t*) malloc(nd_out * sizeof(size_t));
    if (transposition == NULL || sh == NULL) {
        PyErr_NoMemory();
        free(transposition);
        free(sh);
        return 1;
    }
    for (npy_intp i = 0; i < nd_in; ++i) {
        transposition[i] = ((npy_int64*) PyArray_DATA(params->transposition))[i];
    }
    tmp = pygpu_transpose(input, transposition);
    if (!tmp) {
        PyErr_SetString(PyExc_RuntimeError, "GpuDimShuffle: unable to transpose input.");
        free(transposition);
        free(sh);
        return 1;
    }
    e = 0;
    for (npy_intp i = 0; i < nd_out; ++i) {
        if (new_order[i] == -1) {
            sh[i] = 1;
        } else {
            sh[i] = tmp->ga.dimensions[e];
            ++e;
        }
    }
    *out = pygpu_reshape(tmp, nd_out, sh, GA_ANY_ORDER, 1, -1);
    Py_DECREF(tmp);
    free(transposition);
    free(sh);

    /** End shuffle. **/

    if (*out == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "GpuDimShuffle: unable to reshape output.");
        return 1;
    }

    if (!params->inplace) {
        tmp = pygpu_copy(*out, GA_ANY_ORDER);
        Py_DECREF(*out);
        if (!tmp) {
            PyErr_SetString(PyExc_RuntimeError, "GpuDimShuffle: unable to copy output.");
            *out = NULL;
            return 1;
        }
        *out = tmp;
    }

    return 0;
}
