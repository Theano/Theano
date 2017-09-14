#section support_code_apply

int cpu_zoomshift(PyArrayObject* input, PyArrayObject* output_shape,
                  PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                  PyArrayObject* cval, PyArrayObject** output,
                  PARAMS_TYPE* params) {

    // TODO check that PyARRAY_NDIM(input) == PyArray_DIMS(output_shape[0]) etc.
    // TODO check that output shape is int64
    // TODO check that zoom_ar and shift_ar have the right float type

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int mode = params->mode;

    npy_intp ndim = PyArray_NDIM(input);
    npy_intp* out_dims = (npy_intp*)malloc(ndim * sizeof(npy_intp));

    for (int i=0; i<ndim; i++) {
        out_dims[i] = *((npy_intp*)PyArray_GETPTR1(output_shape, i));
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                            PyArray_TYPE(input), 0);
    // TODO check that output is not NULL

    // TODO check type, dim
    double const_val = *((double*)PyArray_GETPTR1(cval, 0));

//  printf("zoom_ar = %d(%d) [%f %f]\n", PyArray_NDIM(zoom_ar), PyArray_DIMS(zoom_ar)[0],
//                                       *((double*)PyArray_GETPTR1(zoom_ar, 0)),
//                                       *((double*)PyArray_GETPTR1(zoom_ar, 1)));
//  printf("shift_ar = %d(%d) [%f %f]\n", PyArray_NDIM(shift_ar), PyArray_DIMS(shift_ar)[0],
//                                       *((double*)PyArray_GETPTR1(shift_ar, 0)),
//                                       *((double*)PyArray_GETPTR1(shift_ar, 1)));
//  printf("order = %ld\n", order);
//  printf("mode = %ld\n", mode);
//  printf("const_val = %f\n", const_val);

    int res = NI_ZoomShift(input, zoom_ar, shift_ar, *output, order, mode, const_val, false);
    // TODO check that res == 0

    free(out_dims);

    return 0;
}

int cpu_zoomshift_grad(PyArrayObject* input, PyArrayObject* bottom_shape,
                       PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                       PyArrayObject* cval, PyArrayObject** output,
                       PARAMS_TYPE* params) {

    // TODO check that PyARRAY_NDIM(input) == PyArray_DIMS(output_shape[0]) etc.
    // TODO check that output shape is int64
    // TODO check that zoom_ar and shift_ar have the right float type

    int order = params->order;
    if (order < 0 || order > 5) {
        PyErr_SetString(PyExc_RuntimeError, "spline order not supported");
        return 1;
    }
    int mode = params->mode;

    npy_intp ndim = PyArray_NDIM(input);
    npy_intp* out_dims = (npy_intp*)malloc(ndim * sizeof(npy_intp));

    for (int i=0; i<ndim; i++) {
        out_dims[i] = *((npy_intp*)PyArray_GETPTR1(bottom_shape, i));
    }

    // create output array
    if (*output)
        Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                            PyArray_TYPE(input), 0);
    // TODO check that output is not NULL

    // TODO check type, dim
    double const_val = *((double*)PyArray_GETPTR1(cval, 0));

//  printf("zoom_ar = %d(%d) [%f %f]\n", PyArray_NDIM(zoom_ar), PyArray_DIMS(zoom_ar)[0],
//                                       *((double*)PyArray_GETPTR1(zoom_ar, 0)),
//                                       *((double*)PyArray_GETPTR1(zoom_ar, 1)));
//  printf("shift_ar = %d(%d) [%f %f]\n", PyArray_NDIM(shift_ar), PyArray_DIMS(shift_ar)[0],
//                                       *((double*)PyArray_GETPTR1(shift_ar, 0)),
//                                       *((double*)PyArray_GETPTR1(shift_ar, 1)));
//  printf("order = %ld\n", order);
//  printf("mode = %ld\n", mode);
//  printf("const_val = %f\n", const_val);

    int res = NI_ZoomShift(*output, zoom_ar, shift_ar, input, order, mode, const_val, true);
    // TODO check that res == 0

    free(out_dims);

    return 0;
}

