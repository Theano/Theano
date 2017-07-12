#section support_code_apply

int APPLY_SPECIFIC(blockgemv)(PyGpuArrayObject *o, PyGpuArrayObject *W,
                              PyGpuArrayObject *h, PyArrayObject *inputIdx,
                              PyArrayObject *outputIdx,
                              PyGpuArrayObject **_out,
                              PARAMS_TYPE* params) {
  PyGpuArrayObject *out = *_out;
  if (params->inplace) {
    Py_XDECREF(out);
    out = o;
    Py_INCREF(out);
  } else {
    out = theano_try_copy(out, o);
    if (out == NULL) {
      // Error already set
      return -1;
    }
  }

  gpudata **W_list = NULL;
  gpudata **inp_list = NULL;
  gpudata **out_list = NULL;
  size_t *offW = NULL;
  size_t *offInp = NULL;
  size_t *offOut = NULL;
  int err;

  err = gpublas_setup(params->context->ctx);
  if (err != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "Can't setup blas");
    return -1;
  }

  /* Prepare lists for the batch */
  size_t maxi = PyGpuArray_DIMS(h)[1];
  size_t maxj = PyGpuArray_DIMS(out)[1];
  size_t maxb = PyGpuArray_DIMS(out)[0];
  ssize_t h_str_0 = PyGpuArray_STRIDES(h)[0];
  ssize_t h_str_1 = PyGpuArray_STRIDES(h)[1];
  ssize_t o_str_0 = PyGpuArray_STRIDES(out)[0];
  ssize_t o_str_1 = PyGpuArray_STRIDES(out)[1];
  ssize_t W_str_0 = PyGpuArray_STRIDES(W)[0];
  ssize_t W_str_1 = PyGpuArray_STRIDES(W)[1];

  W_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offW = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  inp_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offInp = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  out_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
  offOut = (size_t *)calloc(sizeof(size_t), maxi * maxj * maxb);
  if (W_list == NULL || offW == NULL ||
      inp_list == NULL || offInp == NULL ||
      out_list == NULL || offOut == NULL) {
    free(W_list);
    free(offW);
    free(inp_list);
    free(offInp);
    free(out_list);
    free(offOut);
    PyErr_NoMemory();
    return -1;
  }

  for (size_t i = 0; i < maxi; i++) {
    for (size_t j = 0; j < maxj; j++) {
      for (size_t b = 0; b < maxb; b++) {
        size_t p = i + j * maxi + b * maxi * maxj;
        inp_list[p] = h->ga.data;
        offInp[p] = b * h_str_0 + i * h_str_1 + h->ga.offset;
        out_list[p] = out->ga.data;
        offOut[p] = b * o_str_0 + j * o_str_1 + out->ga.offset;
        W_list[p] = W->ga.data;
        offW[p] = *(DTYPE_INPUT_3 *)PyArray_GETPTR2(inputIdx, b, i) * W_str_0 +
          *(DTYPE_INPUT_4 *)PyArray_GETPTR2(outputIdx, b, j) * W_str_1 +
          W->ga.offset;
      }
    }
  }

  cb_transpose transA = cb_no_trans;
  size_t lda = PyGpuArray_STRIDES(W)[2] / gpuarray_get_elsize(W->ga.typecode);
  if (lda == 1) {
    transA = cb_trans;
    lda = PyGpuArray_STRIDES(W)[3] / gpuarray_get_elsize(W->ga.typecode);
  }

  if (out->ga.typecode == GA_FLOAT) {
    err = gpublas_sgemvBatch(cb_fortran, transA,
                             PyGpuArray_DIMS(out)[2],
                             PyGpuArray_DIMS(h)[2], 1,
                             W_list, offW, lda,
                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
  } else if (out->ga.typecode == GA_DOUBLE) {
    err = gpublas_dgemvBatch(cb_fortran, transA,
                             PyGpuArray_DIMS(out)[2],
                             PyGpuArray_DIMS(h)[2], 1,
                             W_list, offW, lda,
                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
  } else if (out->ga.typecode == GA_HALF) {
    err = gpublas_sgemvBatch(cb_fortran, transA,
                             PyGpuArray_DIMS(out)[2],
                             PyGpuArray_DIMS(h)[2], 1,
                             W_list, offW, lda,
                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
  } else {
    err = GA_INVALID_ERROR;
  }
  
  free(W_list);
  free(offW);
  free(inp_list);
  free(offInp);
  free(out_list);
  free(offOut);
  if (err != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "gemvBatch failed");
    return -1;
  }
  *_out = out;
  return 0;
}

