#section support_code

int dnn_rnn_gw(cudnnRNNDescriptor_t desc, npy_uint64 _wsize,
               PyGpuArrayObject *x, PyGpuArrayObject *hx,
               PyGpuArrayObject *y, gpudata *reserve,
               PyGpuArrayObject **dw, cudnnHandle_t _handle) {
  PyGpuContextObject *c = x->context;
  cudnnTensorDescriptor_t xdesc = NULL;
  cudnnTensorDescriptor_t hxdesc = NULL;
  cudnnTensorDescriptor_t ydesc = NULL;
  cudnnFilterDescriptor_t dwdesc = NULL;
  cudnnTensorDescriptor_t *xl = NULL;
  cudnnTensorDescriptor_t *yl = NULL;
  gpudata *workspace = NULL;
  size_t worksize, ressize;

  size_t iters = PyGpuArray_DIM(x, 0);
  size_t wsize = _wsize;
  int dims[3], strs[3];
  cudnnStatus_t err;
  cudnnDataType_t dt;
  int res = -1;

  switch (x->ga.typecode) {
  case GA_FLOAT:
    dt = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    dt = CUDNN_DATA_DOUBLE;
    break;
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Unsupported data type for x");
    return -1;
  }

  // This is early to match the exit() in the fail label.
  cuda_enter(c->ctx);

  err = cudnnCreateTensorDescriptor(&xdesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create xdesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  /* We need to use the last two dimensions for this, this is not a typo */
  dims[0] = PyGpuArray_DIM(x, 1);
  dims[1] = PyGpuArray_DIM(x, 2);
  dims[2] = 1;
  strs[0] = dims[2] * dims[1];
  strs[1] = dims[2];
  strs[2] = 1;

  err = cudnnSetTensorNdDescriptor(xdesc, dt, 3, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not set xdesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  if (c_make_tensorNd(hx, &hxdesc) != 0)
    goto fail;

  err = cudnnCreateTensorDescriptor(&ydesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create ydesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  /* Again not a typo, we need to use the last two dimensions */
  dims[0] = PyGpuArray_DIM(y, 1);
  dims[1] = PyGpuArray_DIM(y, 2);
  dims[2] = 1;

  strs[0] = dims[2] * dims[1];
  strs[1] = dims[2];
  strs[2] = 1;

  err = cudnnSetTensorNdDescriptor(ydesc, dt, 3, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not set ydesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  if (theano_prep_output(dw, 1, &wsize, x->ga.typecode, GA_C_ORDER, c) != 0)
    goto fail;
  GpuArray_memset(&(*dw)->ga, 0);

  if (c_make_filter(*dw, &dwdesc) != 0)
    goto fail;

  xl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), iters);
  if (xl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < iters; i++)
    xl[i] = xdesc;

  yl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), iters);
  if (yl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < iters; i++)
    yl[i] = ydesc;

  err = cudnnGetRNNWorkspaceSize(_handle, desc, (int)iters,
                                 xl, &worksize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not get worksize: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
  if (workspace == NULL) {
    PyErr_Format(PyExc_RuntimeError, "Could not allocate workspace");
    goto fail;
  }

  err = cudnnGetRNNTrainingReserveSize(_handle, desc, (int)iters,
                                       xl, &ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not get reserve size: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  err = cudnnRNNBackwardWeights(_handle, desc, (int)iters,
                                xl, PyGpuArray_DEV_DATA(x),
                                hxdesc, PyGpuArray_DEV_DATA(hx),
                                yl, PyGpuArray_DEV_DATA(y),
                                *(void **)workspace, worksize,
                                dwdesc, PyGpuArray_DEV_DATA(*dw),
                                *(void **)reserve, ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could run RNN grad weights: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  res = 0;
fail:
  if (xdesc != NULL)
    cudnnDestroyTensorDescriptor(xdesc);
  if (hxdesc != NULL)
    cudnnDestroyTensorDescriptor(hxdesc);
  if (ydesc != NULL)
    cudnnDestroyTensorDescriptor(ydesc);
  if (dwdesc != NULL)
    cudnnDestroyFilterDescriptor(dwdesc);
  free(xl);
  free(yl);
  if (workspace != NULL)
    gpudata_release(workspace);
  cuda_exit(c->ctx);
  return res;
}
