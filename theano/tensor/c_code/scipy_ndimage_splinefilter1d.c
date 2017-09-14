#section support_code_apply

int cpu_splinefilter1d(PyArrayObject* input, PyArrayObject** output,
                        PARAMS_TYPE* params) {

    // TODO check inputs

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int axis = params->axis;

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(input), PyArray_DIMS(input),
                                            PyArray_TYPE(input), 0);
    // TODO check that output is not NULL

    int res = NI_SplineFilter1D(input, order, axis, *output);
    // TODO check that res == 0

    return 0;
}

int cpu_splinefilter1d_grad(PyArrayObject* input, PyArrayObject** output,
                        PARAMS_TYPE* params) {

    // TODO check inputs

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int axis = params->axis;

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(input), PyArray_DIMS(input),
                                            PyArray_TYPE(input), 0);
    // TODO check that output is not NULL

    int res = NI_SplineFilter1DGrad(input, order, axis, *output);
    // TODO check that res == 0

    return 0;
}

