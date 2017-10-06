#section support_code_apply

int APPLY_SPECIFIC(gpu_dimshuffle)(PyGpuArrayObject* input, PyGpuArrayObject** out, PARAMS_TYPE* params) {
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
    /* Type of params->transposition (npy_uint32) should be an alias of unsigned int
     * on platforms supported by Theano. */
    transposition = (unsigned int*) PyArray_DATA(params->transposition);
    sh = (size_t*) malloc(nd_out * sizeof(size_t));
    if (sh == NULL) {
        PyErr_NoMemory();
        return 1;
    }
    tmp = pygpu_transpose(input, transposition);
    if (!tmp) {
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
    free(sh);

    if (*out == NULL) {
        return 1;
    }

    /** End shuffle. **/

    if (!params->inplace) {
        tmp = pygpu_copy(*out, GA_ANY_ORDER);
        Py_DECREF(*out);
        if (!tmp) {
            *out = NULL;
            return 1;
        }
        *out = tmp;
    }

    return 0;
}
