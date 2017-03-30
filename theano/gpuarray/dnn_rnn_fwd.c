#section support_code

int dnn_rnn_fwd(cudnnRNNDescriptor_t desc, uint32_t numDirs,
                PyGpuArrayObject *w, PyGpuArrayObject *x,
                PyGpuArrayObject *hx, PyGpuArrayObject *cx,
                gpudata **reserve, PyGpuArrayObject **y,
                PyGpuArrayObject **hy, PyGpuArrayObject **cy,
                cudnnHandle_t _handle) {
  PyGpuContextObject *c = x->context;
  cudnnTensorDescriptor_t xdesc = NULL;
  cudnnTensorDescriptor_t hxdesc = NULL;
  cudnnTensorDescriptor_t cxdesc = NULL;
  cudnnTensorDescriptor_t ydesc = NULL;
  cudnnTensorDescriptor_t hydesc = NULL;
  cudnnTensorDescriptor_t cydesc = NULL;
  cudnnFilterDescriptor_t wdesc = NULL;
  cudnnTensorDescriptor_t *xl = NULL;
  cudnnTensorDescriptor_t *yl = NULL;
  gpudata *workspace = NULL;
  size_t worksize, ressize;

  size_t seqLength = PyGpuArray_DIM(x, 0);
  size_t miniBatch = PyGpuArray_DIM(x, 1);
  size_t inputSize = PyGpuArray_DIM(x, 2);
  size_t hiddenSize = PyGpuArray_DIM(hx, 2);
  size_t shape[3];
  int strs[3], dims[3];
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

  dims[0] = PyGpuArray_DIM(x, 1);
  dims[1] = PyGpuArray_DIM(x, 2);
  dims[2] = 1;

  strs[0] = dims[1] * dims[2];
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

  if (cx != NULL)
    if (c_make_tensorNd(cx, &cxdesc) != 0)
      goto fail;

  if (c_make_filter(w, &wdesc) != 0)
    goto fail;

  shape[0] = seqLength;
  shape[1] = miniBatch;
  shape[2] = hiddenSize * numDirs;
  if (theano_prep_output(y, 3, shape, x->ga.typecode, GA_C_ORDER, c) != 0)
    goto fail;

  err = cudnnCreateTensorDescriptor(&ydesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create ydesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  dims[0] = shape[1];
  dims[1] = shape[2];
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

  if (theano_prep_output(hy, 3, PyGpuArray_DIMS(hx),
                         hx->ga.typecode, GA_C_ORDER, c) != 0)
    goto fail;

  if (c_make_tensorNd(*hy, &hydesc) != 0)
    goto fail;

  if (cy != NULL) {
    if (theano_prep_output(cy, 3, PyGpuArray_DIMS(cx),
                           cx->ga.typecode, GA_C_ORDER, c) != 0)
      goto fail;

    if (c_make_tensorNd(*cy, &cydesc) != 0)
      goto fail;
  }

  xl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), seqLength);
  if (xl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < seqLength; i++)
    xl[i] = xdesc;

  yl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), seqLength);
  if (yl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < seqLength; i++)
    yl[i] = ydesc;

  err = cudnnGetRNNWorkspaceSize(_handle, desc, (int)seqLength,
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

  err = cudnnGetRNNTrainingReserveSize(_handle, desc, (int)seqLength,
                                       xl, &ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not get reserve size: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  *reserve = gpudata_alloc(c->ctx, ressize, NULL, 0, NULL);
  if (*reserve == NULL) {
    PyErr_Format(PyExc_RuntimeError, "Could not allocate reserve");
    goto fail;
  }

  err = cudnnRNNForwardTraining(_handle, desc, (int)seqLength,
                                xl, PyGpuArray_DEV_DATA(x),
                                hxdesc, PyGpuArray_DEV_DATA(hx),
                                cxdesc, cx ? PyGpuArray_DEV_DATA(cx) : NULL,
                                wdesc, PyGpuArray_DEV_DATA(w),
                                yl, PyGpuArray_DEV_DATA(*y),
                                hydesc, PyGpuArray_DEV_DATA(*hy),
                                cydesc, cy ? PyGpuArray_DEV_DATA(*cy) : NULL,
                                *(void **)workspace, worksize,
                                *(void **)(*reserve), ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could run RNN: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  res = 0;
 fail:
  if (xdesc != NULL)
    cudnnDestroyTensorDescriptor(xdesc);
  if (hxdesc != NULL)
    cudnnDestroyTensorDescriptor(hxdesc);
  if (cxdesc != NULL)
    cudnnDestroyTensorDescriptor(cxdesc);
  if (wdesc != NULL)
    cudnnDestroyFilterDescriptor(wdesc);
  if (ydesc != NULL)
    cudnnDestroyTensorDescriptor(ydesc);
  if (hydesc != NULL)
    cudnnDestroyTensorDescriptor(hydesc);
  if (cydesc != NULL)
    cudnnDestroyTensorDescriptor(cydesc);
  free(xl);
  free(yl);
  if (workspace != NULL)
    gpudata_release(workspace);
  cuda_exit(c->ctx);
  return res;
}
