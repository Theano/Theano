#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(dy);
cudnnTensorDescriptor_t APPLY_SPECIFIC(sm);
cudnnTensorDescriptor_t APPLY_SPECIFIC(out);

#section init_code_struct

APPLY_SPECIFIC(dy) = NULL;
APPLY_SPECIFIC(sm) = NULL;
APPLY_SPECIFIC(out) = NULL;

{
  cudnnStatus_t err;
  err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(dy));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor: %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(sm));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor: %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(out));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor: %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(dy) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(dy));
if (APPLY_SPECIFIC(sm) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(sm));
if (APPLY_SPECIFIC(out) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(out));

#section support_code_struct

int APPLY_SPECIFIC(softmax_grad)(PyGpuArrayObject *dy,
                                 PyGpuArrayObject *sm,
                                 PyGpuArrayObject **out) {
  cudnnStatus_t err;
  PyGpuContextObject *c = pygpu_default_context();

  if (c_set_tensorNd(dy, APPLY_SPECIFIC(dy)) != 0)
    return 1;
  if (c_set_tensorNd(sm, APPLY_SPECIFIC(sm)) != 0)
    return 1;

  if (theano_prep_output(out, PyGpuArray_NDIM(dy),
                         PyGpuArray_DIMS(dy), dy->ga.typecode,
                         GA_C_ORDER, c) != 0)
    return 1;

  if (c_set_tensorNd(*out, APPLY_SPECIFIC(out)) != 0)
    return 1;

  {
    const float alpha = 1.;
    const float beta = 0.;

    cuda_enter(c->ctx);
    err = cudnnSoftmaxBackward(
      APPLY_SPECIFIC(_handle),
      SOFTMAX_ALGO,
      SOFTMAX_MODE,
      (void *)&alpha,
      APPLY_SPECIFIC(sm),
      PyGpuArray_DEV_DATA(sm),
      APPLY_SPECIFIC(dy),
      PyGpuArray_DEV_DATA(dy),
      (void*) &beta,
      APPLY_SPECIFIC(out),
      PyGpuArray_DEV_DATA(*out)
      );
    cuda_exit(c->ctx);
  }

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error during operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
