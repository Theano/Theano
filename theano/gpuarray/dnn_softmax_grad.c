#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(dy);
cudnnTensorDescriptor_t APPLY_SPECIFIC(sm);
cudnnTensorDescriptor_t APPLY_SPECIFIC(dx);

#section init_code_struct

APPLY_SPECIFIC(dy) = NULL;
APPLY_SPECIFIC(sm) = NULL;
APPLY_SPECIFIC(dx) = NULL;

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
  err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(dx));
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
if (APPLY_SPECIFIC(dx) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(dx));

#section support_code_struct

int APPLY_SPECIFIC(softmax_grad)(PyGpuArrayObject *dy,
                                 PyGpuArrayObject *sm,
                                 PyGpuArrayObject **dx,
                                 PARAMS_TYPE* wrapper) {
  PyGpuContextObject *c = dy->context;
  cudnnStatus_t err;

  if (theano_prep_output(dx, PyGpuArray_NDIM(dy),
                         PyGpuArray_DIMS(dy), dy->ga.typecode,
                         GA_C_ORDER, c) != 0)
    return 1;

  // Directly return the output if any of the dimensions is 0.
  // (cuDNN does not support zero-length dimensions.)
  if (PyGpuArray_SIZE(*dx) == 0)
    return 0;

  if (c_set_tensorNd(dy, APPLY_SPECIFIC(dy)) != 0)
    return 1;
  if (c_set_tensorNd(sm, APPLY_SPECIFIC(sm)) != 0)
    return 1;

  if (c_set_tensorNd(*dx, APPLY_SPECIFIC(dx)) != 0)
    return 1;

  {
    const float alphaf = 1;
    const float betaf = 0;
    const double alphad = 1;
    const double betad = 0;
    void *alpha, *beta;

    switch (sm->ga.typecode) {
    case GA_DOUBLE:
      alpha = (void *)&alphad;
      beta = (void *)&betad;
      break;
    case GA_FLOAT:
    case GA_HALF:
      alpha = (void *)&alphaf;
      beta = (void *)&betaf;
      break;
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported type in softmax gradient");
      return 1;
    }

    cuda_enter(c->ctx);

    cuda_wait(sm->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait(dy->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait((*dx)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    err = cudnnSoftmaxBackward(
      wrapper->handle,
      wrapper->algo,
      wrapper->mode,
      alpha,
      APPLY_SPECIFIC(sm),
      PyGpuArray_DEV_DATA(sm),
      APPLY_SPECIFIC(dy),
      PyGpuArray_DEV_DATA(dy),
      beta,
      APPLY_SPECIFIC(dx),
      PyGpuArray_DEV_DATA(*dx)
      );

    cuda_record(sm->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(dy->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*dx)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    cuda_exit(c->ctx);
  }

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error during operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
