#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;

if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input)); }
if (APPLY_SPECIFIC(output) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output)); }

#section support_code_struct

int APPLY_SPECIFIC(dnn_pool)(PyGpuArrayObject *img,
                             cudnnPoolingDescriptor_t desc,
                             PyGpuArrayObject **out,
                             PyGpuContextObject *c) {
  cudnnStatus_t err;
  size_t dims[5];

  if (!GpuArray_IS_C_CONTIGUOUS(&img->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
    return 1;
  }

  if (c_set_tensorNd(img, APPLY_SPECIFIC(input)) != 0)
    return 1;

  cudnnPoolingMode_t mode;
  int w[3];
  int p[3];
  int s[3];
  int ndims;

  err = cudnnGetPoolingNdDescriptor(desc, 3, &mode, &ndims, w, p, s);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "error doing cudnnGetPoolingDescriptor operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }

  dims[0] = PyGpuArray_DIM(img, 0);
  dims[1] = PyGpuArray_DIM(img, 1);
  dims[2] = (PyGpuArray_DIM(img, 2) + (p[0]*2) - w[0]) / s[0] + 1;
  dims[3] = (PyGpuArray_DIM(img, 3) + (p[1]*2) - w[1]) / s[1] + 1;
  if (ndims == 3)
    dims[4] = (PyGpuArray_DIM(img, 4) + (p[2]*2) - w[2]) / s[2] + 1;

  if (theano_prep_output(out, ndims+2, dims, img->ga.typecode,
                         GA_C_ORDER, c) != 0)
    return 1;

  if (c_set_tensorNd(*out, APPLY_SPECIFIC(output)) != 0)
    return 1;

  {
    const float alphaf = 1;
    const float betaf = 0;
    const double alphad = 1;
    const double betad = 0;
    void *alpha, *beta;

    switch (img->ga.typecode) {
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
      PyErr_SetString(PyExc_TypeError, "Unsupported type in pooling");
      return 1;
    }

    cuda_enter(c->ctx);

    cuda_wait(img->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    err = cudnnPoolingForward(
      APPLY_SPECIFIC(_handle), desc,
      alpha,
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(img),
      beta,
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*out));

    cuda_record(img->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    cuda_exit(c->ctx);
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "GpuDnnPool: error doing cudnnPoolingForward operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
