#section support_code

int dnn_dropout_fwd(PyGpuArrayObject *x,
                    cudnnDropoutDescriptor_t *desc,
                    PyGpuArrayObject *state,
                    PyGpuArrayObject **y,
                    PyGpuArrayObject **ostate,
                    gpudata **reserve,
                    cudnnHandle_t _handle) {
  PyGpuArrayContext *c = x->context;
  cudnnTensorDescriptor_t xdesc;
  cudnnTensorDescriptor_t ydesc;
  gpudata *res;
  size_t res_sz;
  cudnnStatus_t err;

  if (c_make_tensorNd(x, &xdesc))
    return -1;

  if (theano_prep_output(y, x->ga.nd, x->ga.dimensions, x->ga.typecode,
                         GA_C_ORDER, c)) {
    cudnnDestroyTensorDescriptor(xdesc);
    return -1;
  }

  if (c_make_tensorNd(y, &ydesc)) {
    cudnnDestroyTensorDescriptor(xdesc);
    return -1;
  }

  *ostate = state;
  Py_INCREF((PyObject *)state);

  /* This can't fail according to the docs */
  err = cudnnDropoutGetReserveSpaceSize(desc, &res_sz);
  res = gpudata_alloc(c->ctx, res_zs, NULL, 0, NULL);
  if (res == NULL) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyTensorDescriptor(ydesc);
    PyErr_SetString(PyExc_RuntimeError, "Could not allocate reserve for dropout");
  }
  *reserve = res;

  cuda_enter(c->ctx);
  err = cudnnDropoutForward(_handle, desc, xdesc, PyGpuArray_DEV_DATA(x),
                            ydesc, PyGpuArray_DEV_DATA(y), *(void **)res,
                            res_sz);
  cudnnDestroyTensorDescriptor(xdesc);
  cudnnDestroyTensorDescriptor(ydesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not run dropout: %s",
                 cudnnGetErrorString(err));
    cuda_exit(c->ctx);
    return -1;
  }

  cuda_exit(c->ctx);
  return 0;
}
