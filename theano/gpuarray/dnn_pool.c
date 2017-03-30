#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnPoolingDescriptor_t APPLY_SPECIFIC(pool);


#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(pool) = NULL;

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
if ((APPLY_SPECIFIC(err) = cudnnCreatePoolingDescriptor(&APPLY_SPECIFIC(pool))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate pooling descriptor"
                "(pool): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input)); }
if (APPLY_SPECIFIC(output) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output)); }
if (APPLY_SPECIFIC(pool) != NULL) { cudnnDestroyPoolingDescriptor(APPLY_SPECIFIC(pool)); }


#section support_code_struct

int APPLY_SPECIFIC(dnn_pool)(PyGpuArrayObject *img,
                             PyArrayObject *ws,
                             PyArrayObject *stride,
                             PyArrayObject *pad,
                             PyGpuArrayObject **out,
                             cudnnHandle_t _handle) {
  PyGpuContextObject *c = img->context;
  size_t dims[5];
  cudnnStatus_t err;

  if (!GpuArray_IS_C_CONTIGUOUS(&img->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
    return 1;
  }

  cudnnPoolingMode_t mode;
  int w[3];
  int p[3];
  int s[3];
  int ndims = PyArray_DIM(ws, 0);//PyGpuArray_NDIM(img) - 2;

  for(int i = 0; i < ndims; i++) {
     w[i] = *((npy_intp*)PyArray_GETPTR1(ws, i));
  }
  for(int i = 0; i < ndims; i++) {
     p[i] = *((npy_intp*)PyArray_GETPTR1(pad, i));
  }
  for(int i = 0; i < ndims; i++) {
     s[i] = *((npy_intp*)PyArray_GETPTR1(stride, i));
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

  // if input batch is empty, we return the empty output without calling cuDNN
  // (which will fail on zero batch size).
  if (PyGpuArray_DIM(*out, 0) == 0)
    return 0;

  if (c_set_tensorNd(img, APPLY_SPECIFIC(input)) != 0)
    return 1;

  if (c_set_tensorNd(*out, APPLY_SPECIFIC(output)) != 0)
    return 1;

  err = cudnnSetPoolingNdDescriptor(APPLY_SPECIFIC(pool), MODE_FLAG, CUDNN_PROPAGATE_NAN, ndims, w, p, s);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor %s", cudnnGetErrorString(err));
  }

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
      _handle, APPLY_SPECIFIC(pool),
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
