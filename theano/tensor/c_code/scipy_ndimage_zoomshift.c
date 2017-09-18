#section support_code_apply

int NI_ZoomShift_over_images(PyArrayObject* input,
                             PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                             PyArrayObject* output,
                             int order, int mode, double const_val, int reverse,
                             npy_intp naxes, npy_intp* axes) {

    npy_intp ndim = PyArray_NDIM(input);

    // set true if the axis is for zooming, false for axes we loop over
    int* zoom_axes = (int*)calloc(ndim, sizeof(int));
    for (int i=0; i<naxes; i++) {
        zoom_axes[axes[i]] = true;
    }

    // collect strides and dimensions for images in input and output
    npy_intp nimg = 1;
    npy_intp *img_in_strides = (npy_intp*)malloc((ndim - naxes) * sizeof(npy_intp));
    npy_intp *zoom_in_strides = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    npy_intp *img_out_strides = (npy_intp*)malloc((ndim - naxes) * sizeof(npy_intp));
    npy_intp *zoom_out_strides = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    npy_intp *img_in_dims = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    npy_intp *zoom_in_dims = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    npy_intp *zoom_out_dims = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    int im = 0, zo = 0;
    for (int i=0; i<ndim; i++) {
        if (zoom_axes[i]) {
            zoom_in_strides[zo] = PyArray_STRIDES(input)[i];
            zoom_out_strides[zo] = PyArray_STRIDES(output)[i];
            zoom_in_dims[zo] = PyArray_DIMS(input)[i];
            zoom_out_dims[zo] = PyArray_DIMS(output)[i];
            zo++;
        } else {
            nimg *= PyArray_DIMS(input)[i];
            img_in_dims[im] = PyArray_DIMS(input)[i];
            img_in_strides[im] = PyArray_STRIDES(input)[i];
            img_out_strides[im] = PyArray_STRIDES(output)[i];
            im++;
        }
    }

    int ret = 1;

    for (int img=0; img<nimg; img++) {
        // compute the ofset of the current image in input an output
        npy_intp in_offset = img_in_strides[0] * (img % img_in_dims[0]);
        npy_intp out_offset = img_out_strides[0] * (img % img_in_dims[0]);
        int img2 = img / img_in_dims[0];
        for (int d=1; d<(ndim - naxes); d++) {
            in_offset += img_in_strides[d] * (img2 % img_in_dims[d]);
            out_offset += img_out_strides[d] * (img2 % img_in_dims[d]);
            img2 /= img_in_dims[d];
        }

        // create views for input and output
        PyArrayObject *view_in = (PyArrayObject*)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DESCR(input), naxes, zoom_in_dims, zoom_in_strides,
            PyArray_BYTES(input) + in_offset, PyArray_FLAGS(input), NULL);
        PyArrayObject *view_out = (PyArrayObject*)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DESCR(output), naxes, zoom_out_dims, zoom_out_strides,
            PyArray_BYTES(output) + out_offset, PyArray_FLAGS(output), NULL);
        Py_INCREF(PyArray_DESCR(input));
        Py_INCREF(PyArray_DESCR(output));

        // process the current image
        ret = NI_ZoomShift(view_in, zoom_ar, shift_ar, view_out, order, mode, const_val, reverse);

        Py_XDECREF(view_in);
        Py_XDECREF(view_out);

        if (ret == 0) {
            break;
        }
    }

    free(zoom_axes);
    free(img_in_strides);
    free(zoom_in_strides);
    free(img_out_strides);
    free(zoom_out_strides);
    free(img_in_dims);
    free(zoom_in_dims);
    free(zoom_out_dims);

    return ret;
}


int cpu_zoomshift_or_grad(PyArrayObject** input, PyArrayObject* output_shape,
                          PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                          PyArrayObject* cval, PyArrayObject** output,
                          PARAMS_TYPE* params, int reverse) {

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

    npy_intp ndim = PyArray_NDIM(reverse ? *output : *input);
    npy_intp naxes = PyArray_DIMS(params->axes)[0];
    if (naxes == 0) {
        naxes = ndim;
    }

    if (PyArray_NDIM(output_shape) != 1 || PyArray_DIMS(output_shape)[0] != naxes) {
        PyErr_SetString(PyExc_RuntimeError, "invalid output shape");
        return 1;
    }
    if (PyArray_SIZE(zoom_ar) == 0) {
        zoom_ar = NULL;
    } else if (PyArray_NDIM(zoom_ar) != 1 || PyArray_DIMS(zoom_ar)[0] != naxes) {
        PyErr_SetString(PyExc_RuntimeError, "invalid zoom argument");
        return 1;
    } else {
        zoom_ar = PyArray_GETCONTIGUOUS(zoom_ar);
    }
    if (PyArray_SIZE(shift_ar) == 0) {
        shift_ar = NULL;
    } else if (PyArray_NDIM(shift_ar) != 1 || PyArray_DIMS(shift_ar)[0] != naxes) {
        PyErr_SetString(PyExc_RuntimeError, "invalid shift argument");
        return 1;
    } else {
        shift_ar = PyArray_GETCONTIGUOUS(shift_ar);
    }

    npy_intp* axes = (npy_intp*)malloc(naxes * sizeof(npy_intp));
    if (PyArray_DIMS(params->axes)[0] == 0) {
        for (int i=0; i<naxes; i++) {
            axes[i] = i;
        }
    } else {
        for (int i=0; i<naxes; i++) {
            axes[i] = *((npy_intp*)PyArray_GETPTR1(params->axes, i));
            if (axes[i] < 0 || axes[i] >= ndim || (i > 0 && axes[i - 1] >= axes[i])) {
                PyErr_SetString(PyExc_RuntimeError, "invalid axes");
                free(axes);
                return 1;
            }
        }
    }

    npy_intp* out_dims = (npy_intp*)malloc(ndim * sizeof(npy_intp));
    for (int i=0; i<ndim; i++) {
        out_dims[i] = PyArray_DIMS(reverse ? *output : *input)[i];
    }
    for (int i=0; i<naxes; i++) {
        out_dims[axes[i]] = *((npy_intp*)PyArray_GETPTR1(output_shape, i));
    }

    // create output array
    if (reverse) {
        if (*input)
            Py_XDECREF(*input);
        *input = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                                PyArray_TYPE(*output), 0);
        free(out_dims);
        if (*input == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "cpu_zoomshift_grad failed to initialize output array");
            free(axes);
            return 1;
        }
    } else {
        if (*output)
            Py_XDECREF(*output);
        *output = (PyArrayObject*)PyArray_ZEROS(ndim, out_dims,
                                                PyArray_TYPE(*input), 0);
        free(out_dims);
        if (*output == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "cpu_zoomshift failed to initialize output array");
            free(axes);
            return 1;
        }
    }

    double const_val = *((double*)PyArray_GETPTR1(cval, 0));

    int ret = 1;

    if (naxes == ndim) {
        // NI_ZoomShift will set a Python error if necessary
        ret = NI_ZoomShift(*input, zoom_ar, shift_ar, *output, order, mode, const_val, reverse);
    } else {
        ret = NI_ZoomShift_over_images(*input, zoom_ar, shift_ar, *output, order, mode,
                                       const_val, reverse, naxes, axes);
    }

    free(axes);

    // NI_ZoomShift has set a Python error if necessary
    return ret == 0 ? 1 : 0;
}

int cpu_zoomshift(PyArrayObject* input, PyArrayObject* output_shape,
                  PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                  PyArrayObject* cval, PyArrayObject** output,
                  PARAMS_TYPE* params) {

    return cpu_zoomshift_or_grad(&input, output_shape, zoom_ar, shift_ar,
                                 cval, output, params, false);
}

int cpu_zoomshift_grad(PyArrayObject* input, PyArrayObject* bottom_shape,
                       PyArrayObject* zoom_ar, PyArrayObject* shift_ar,
                       PyArrayObject* cval, PyArrayObject** output,
                       PARAMS_TYPE* params) {

    return cpu_zoomshift_or_grad(output, bottom_shape, zoom_ar, shift_ar,
                                 cval, &input, params, true);
}

