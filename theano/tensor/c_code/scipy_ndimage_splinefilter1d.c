#section support_code_apply

int cpu_splinefilter1d(PyArrayObject* input, PyArrayObject** output,
                        PARAMS_TYPE* params) {

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int axis = params->axis;
    if (axis < 0) {
        axis += PyArray_NDIM(input);
    }
    if (axis < 0 || axis >= PyArray_NDIM(input)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid axis");
        return 1;
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(input), PyArray_DIMS(input),
                                            PyArray_TYPE(input), 0);
    if (*output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cpu_splinefilter1d failed to initialize output array");
        return 1;
    }

    // NI_SplineFilter1D will set a Python error if necessary
    if (!NI_SplineFilter1D(input, order, axis, *output)) {
        return 1;
    }
    return 0;
}

int cpu_splinefilter1d_grad(PyArrayObject* input, PyArrayObject** output,
                        PARAMS_TYPE* params) {

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int axis = params->axis;
    if (axis < 0) {
        axis += PyArray_NDIM(input);
    }
    if (axis < 0 || axis >= PyArray_NDIM(input)) {
        PyErr_SetString(PyExc_RuntimeError, "invalid axis");
        return 1;
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(input), PyArray_DIMS(input),
                                            PyArray_TYPE(input), 0);
    if (*output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cpu_splinefilter1d_grad failed to initialize output array");
        return 1;
    }

    // NI_SplineFilter1DGrad will set a Python error if necessary
    if (!NI_SplineFilter1DGrad(input, order, axis, *output)) {
        return 1;
    }
    return 0;
}

