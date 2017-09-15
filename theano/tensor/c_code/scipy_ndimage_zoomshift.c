#section support_code_apply

int cpu_zoomshift(PyArrayObject* input, PyArrayObject* output_shape,
                  PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                  PyArrayObject* cval, PyArrayObject** output,
                  PARAMS_TYPE* params) {

    int order = params->order;
    int mode = params->mode;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    if (mode < 0 || mode > 4) {
        PyErr_SetString(PyExc_RuntimeError, "mode not supported");
        return 1;
    }

    npy_intp ndim = PyArray_NDIM(input);

    if (PyArray_NDIM(output_shape) != 1 || PyArray_DIMS(output_shape)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid output shape");
        return 1;
    }
    if (PyArray_NDIM(zoom_ar) != 1 || PyArray_DIMS(zoom_ar)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid zoom argument");
        return 1;
    }
    if (PyArray_NDIM(shift_ar) != 1 || PyArray_DIMS(shift_ar)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid shift argument");
        return 1;
    }

    npy_intp* out_dims = (npy_intp*)malloc(ndim * sizeof(npy_intp));
    for (int i=0; i<ndim; i++) {
        out_dims[i] = *((npy_intp*)PyArray_GETPTR1(output_shape, i));
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                            PyArray_TYPE(input), 0);
    free(out_dims);
    if (*output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cpu_zoomshift failed to initialize output array");
        return 1;
    }

    double const_val = *((double*)PyArray_GETPTR1(cval, 0));

    zoom_ar = PyArray_GETCONTIGUOUS(zoom_ar);
    shift_ar = PyArray_GETCONTIGUOUS(shift_ar);

    // NI_ZoomShift will set a Python error if necessary
    if (!NI_ZoomShift(input, zoom_ar, shift_ar, *output, order, mode, const_val, false)) {
        return 1;
    }
    return 0;
}

int cpu_zoomshift_grad(PyArrayObject* input, PyArrayObject* bottom_shape,
                       PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                       PyArrayObject* cval, PyArrayObject** output,
                       PARAMS_TYPE* params) {

    int order = params->order;
    int mode = params->mode;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    if (mode < 0 || mode > 4) {
        PyErr_SetString(PyExc_RuntimeError, "mode not supported");
        return 1;
    }

    npy_intp ndim = PyArray_NDIM(input);

    if (PyArray_NDIM(bottom_shape) != 1 || PyArray_DIMS(bottom_shape)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid bottom shape");
        return 1;
    }
    if (PyArray_NDIM(zoom_ar) != 1 || PyArray_DIMS(zoom_ar)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid zoom argument");
        return 1;
    }
    if (PyArray_NDIM(shift_ar) != 1 || PyArray_DIMS(shift_ar)[0] != ndim) {
        PyErr_SetString(PyExc_RuntimeError, "invalid shift argument");
        return 1;
    }

    npy_intp* out_dims = (npy_intp*)malloc(ndim * sizeof(npy_intp));
    for (int i=0; i<ndim; i++) {
        out_dims[i] = *((npy_intp*)PyArray_GETPTR1(bottom_shape, i));
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                            PyArray_TYPE(input), 0);
    free(out_dims);
    if (*output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cpu_zoomshift_grad failed to initialize output array");
        return 1;
    }

    double const_val = *((double*)PyArray_GETPTR1(cval, 0));

    zoom_ar = PyArray_GETCONTIGUOUS(zoom_ar);
    shift_ar = PyArray_GETCONTIGUOUS(shift_ar);

    // NI_ZoomShift will set a Python error if necessary
    if (!NI_ZoomShift(*output, zoom_ar, shift_ar, input, order, mode, const_val, true)) {
        return 1;
    }
    return 0;
}

