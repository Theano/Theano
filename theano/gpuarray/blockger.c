#section support_code_apply

int APPLY_SPECIFIC(blockger)(PyGpuArrayObject *o, PyGpuArrayObject *x,
                             PyGpuArrayObject *y, PyArrayObject *xIdx,
                             PyArrayObject *yIdx, PyArrayObject *alpha,
                             PyGpuArrayObject **_out,
                             PyGpuContextObject *ctx) {
  PyGpuArrayObject *out = *_out;
  gpudata **o_list = NULL;
  gpudata **x_list = NULL;
  gpudata **y_list = NULL;
  size_t *offOut = NULL;
  size_t *offX = NULL;
  size_t *offY = NULL;
  int err;

  err = gpublas_setup(ctx->ctx);
  if (err != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "Can't setup blas");
    return -1;
  }

#ifdef INPLACE
  Py_XDECREF(out);
  out = o;
  Py_INCREF(out);
#else
  out = theano_try_copy(out, o);
  if (out == NULL)
    return -1;
#endif
  size_t maxi = PyGpuArray_DIMS(x)[1];
  size_t maxj = PyGpuArray_DIMS(y)[1];
  size_t maxb = PyGpuArray_DIMS(x)[0];

  ssize_t x_str_0 = PyGpuArray_STRIDES(x)[0];
  ssize_t x_str_1 = PyGpuArray_STRIDES(x)[1];
  ssize_t y_str_0 = PyGpuArray_STRIDES(y)[0];
  ssize_t y_str_1 = PyGpuArray_STRIDES(y)[1];
  ssize_t o_str_0 = PyGpuArray_STRIDES(out)[0];
  ssize_t o_str_1 = PyGpuArray_STRIDES(out)[1];

  o_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offOut = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  x_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offX = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  y_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offY = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  if (o_list == NULL || offOut == NULL ||
      x_list == NULL || offX == NULL ||
      y_list == NULL || offY == NULL) {
    free(o_list);
    free(offOut);
    free(x_list);
    free(offX);
    free(y_list);
    free(offY);
    PyErr_NoMemory();
    return -1;
  }
  for (size_t i = 0; i < maxi; i++) {
    for (size_t j = 0; j < maxj; j++) {
      for (size_t b = 0; b < maxb; b++) {
        size_t p = i + j * maxi + b * maxi * maxj;
        x_list[p] = x->ga.data;
        offX[p] = b * x_str_0 + i * x_str_1 + x->ga.offset;
        y_list[p] = y->ga.data;
        offY[p] = b * y_str_0 + j * y_str_1 + y->ga.offset;
        o_list[p] = out->ga.data;
        offOut[p] = *(DTYPE_INPUT_3 *)PyArray_GETPTR2(xIdx, b, i) * o_str_0 + *(DTYPE_INPUT_4 *)PyArray_GETPTR2(yIdx, b, j) * o_str_1 + out->ga.offset;
      }
    }
  }

  ssize_t str_y = PyGpuArray_STRIDES(y)[2] / gpuarray_get_elsize(y->ga.typecode);
  ssize_t str_x = PyGpuArray_STRIDES(x)[2] / gpuarray_get_elsize(x->ga.typecode);
  ssize_t str_out = PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode);

  if (out->ga.typecode == GA_FLOAT) {
    err = gpublas_sgerBatch(cb_fortran,
                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
                            *(float *)PyArray_GETPTR1(alpha, 0),
                            y_list, offY, str_y, x_list, offX, str_x,
                            o_list, offOut, str_out,
                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
  } else if (out->ga.typecode == GA_DOUBLE) {
    err = gpublas_dgerBatch(cb_fortran,
                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
                            *(double *)PyArray_GETPTR1(alpha, 0),
                            y_list, offY, str_y, x_list, offX, str_x,
                            o_list, offOut, str_out,
                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
  } else if (out->ga.typecode == GA_HALF) {
    err = gpublas_hgerBatch(cb_fortran,
                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
                            *(float *)PyArray_GETPTR1(alpha, 0),
                            y_list, offY, str_y, x_list, offX, str_x,
                            o_list, offOut, str_out,
                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
  } else {
    err = GA_INVALID_ERROR;
  }
  free(o_list);
  free(offOut);
  free(x_list);
  free(offX);
  free(y_list);
  free(offY);
  if (err != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "gerBatch failed");
    return -1;
  }
  *_out = out;
  return 0;
}


