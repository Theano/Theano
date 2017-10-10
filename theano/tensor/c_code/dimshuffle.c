#section support_code_apply

int APPLY_SPECIFIC(cpu_dimshuffle)(PyArrayObject* input, PyArrayObject** res, PARAMS_TYPE* params) {
    npy_bool* input_broadcastable;
    npy_int64* new_order;
    npy_intp nd_in;
    npy_intp nd_out;
    PyArrayObject* basename;
    npy_intp* dimensions;
    npy_intp* strides;

    if (!PyArray_IS_C_CONTIGUOUS(params->input_broadcastable)) {
        PyErr_SetString(PyExc_RuntimeError, "DimShuffle: param input_broadcastable must be C-contiguous.");
        return 1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(params->_new_order)) {
        PyErr_SetString(PyExc_RuntimeError, "DimShuffle: param _new_order must be C-contiguous.");
        return 1;
    }
    input_broadcastable = (npy_bool*) PyArray_DATA(params->input_broadcastable);
    new_order = (npy_int64*) PyArray_DATA(params->_new_order);
    nd_in = PyArray_SIZE(params->input_broadcastable);
    nd_out = PyArray_SIZE(params->_new_order);

    /* check_input_nd */
    if (PyArray_NDIM(input) != nd_in) {
        PyErr_SetString(PyExc_NotImplementedError, "input nd");
        return 1;
    }

    /* clear_output */
    if (*res)
        Py_XDECREF(*res);

    /* get_base */
    if (params->inplace) {
        basename = input;
        Py_INCREF((PyObject*)basename);
    } else {
        basename =
            (PyArrayObject*)PyArray_FromAny((PyObject*)input,
                                            NULL, 0, 0, NPY_ARRAY_ALIGNED|NPY_ARRAY_ENSURECOPY, NULL);
    }

    /* shape_statements and strides_statements */
    dimensions = (npy_intp*) malloc(nd_out * sizeof(npy_intp));
    strides = (npy_intp*) malloc(nd_out * sizeof(npy_intp));
    if (dimensions == NULL || strides == NULL) {
        PyErr_NoMemory();
        free(dimensions);
        free(strides);
        return 1;
    };

    for (npy_intp i = 0; i < nd_out; ++i) {
        if (new_order[i] != -1) {
            dimensions[i] = PyArray_DIMS(basename)[new_order[i]];
            strides[i] = PyArray_DIMS(basename)[new_order[i]] == 1 ?
                            0 : PyArray_STRIDES(basename)[new_order[i]];
        } else {
            dimensions[i] = 1;
            strides[i] = 0;
        }
    }

    /* set the strides of the broadcasted dimensions.
     * This algorithm is from numpy: PyArray_Newshape() in
     * cvs/numpy/numpy/core/src/multiarraymodule.c */
    if (nd_out > 0) {
        if (strides[nd_out - 1] == 0)
            strides[nd_out - 1] = PyArray_DESCR(basename)->elsize;
        for (npy_intp i = nd_out - 2; i > -1; --i) {
            if (strides[i] == 0)
                strides[i] = strides[i + 1] * dimensions[i + 1];
        }
    }

    /* close_bracket */
    // create a new array.
    *res = (PyArrayObject*)PyArray_New(&PyArray_Type, nd_out, dimensions,
                                       PyArray_TYPE(basename), strides,
                                       PyArray_DATA(basename), PyArray_ITEMSIZE(basename),
                                       // borrow only the writable flag from the base
                                       // the NPY_OWNDATA flag will default to 0.
                                       (NPY_ARRAY_WRITEABLE * PyArray_ISWRITEABLE(basename)),
                                       NULL);

    if (*res == NULL) {
        free(dimensions);
        free(strides);
        return 1;
    }

    // recalculate flags: CONTIGUOUS, FORTRAN, ALIGNED
    PyArray_UpdateFlags(*res, NPY_ARRAY_UPDATE_ALL);

    // we are making a view in both inplace and non-inplace cases
    PyArray_SetBaseObject(*res, (PyObject*)basename);

    free(strides);
    free(dimensions);

    return 0;
}
