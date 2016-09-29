#section support_code

int dnn_rnn_gi(cudnnRNNDescriptor_t desc, npy_uint64 xshp,
               PyGpuArrayObject *y, PyGpuArrayObject *dy,
               PyGpuArrayObject *w, PyGpuArrayObject *hx,
               gpudata *reserve, PyGpuArrayObject *cx,
               PyGpuArrayObject *dhy, PyGpuArrayObject *dcy,
               gpudata **oreserve, PyGpuArrayObject **dx,
               PyGpuArrayObject **dhx, PyGpuArrayObject **dcx,
               cudnnHandle_t _handle) {
  PyGpuContextObject *c = y->context;
  cudnnTensorDescriptor_t ydesc = NULL;
  cudnnTensorDescriptor_t dhydesc = NULL;
  cudnnTensorDescriptor_t dcydesc = NULL;
  cudnnFilterDescriptor_t wdesc = NULL;
  cudnnTensorDescriptor_t hxdesc = NULL;
  cudnnTensorDescriptor_t cxdesc = NULL;
  cudnnTensorDescriptor_t dxdesc = NULL;
  cudnnTensorDescriptor_t dhxdesc = NULL;
  cudnnTensorDescriptor_t dcxdesc = NULL;
  cudnnTensorDescriptor_t *yl = NULL;
  cudnnTensorDescriptor_t *dxl = NULL;
  gpudata *workspace = NULL;
  size_t worksize, ressize;

  size_t seqLength = PyGpuArray_DIM(y, 0);
  size_t miniBatch = PyGpuArray_DIM(y, 1);
  size_t inputSize = xshp;
  size_t shape[3];
  int dims[3], strs[3];
  cudnnStatus_t err;
  cudnnDataType_t dt;
  int res = -1;

  switch (y->ga.typecode) {
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
    PyErr_SetString(PyExc_TypeError, "Unsupported data type for y");
    return -1;
  }

  cuda_enter(c->ctx);

  err = cudnnCreateTensorDescriptor(&ydesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create ydesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  /* We need to use the last two dimensions for this, this is not a typo */
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

  if (dhy != NULL)
    if (c_make_tensorNd(dhy, &dhydesc) != 0)
      goto fail;

  if (dcy != NULL)
    if (c_make_tensorNd(dcy, &dcydesc) != 0)
      goto fail;
  
  if (c_make_filter(w, &wdesc) != 0)
    goto fail;

  if (c_make_tensorNd(hx, &hxdesc) != 0)
    goto fail;

  if (cx != NULL)
    if (c_make_tensorNd(cx, &cxdesc) != 0)
      goto fail;

  shape[0] = seqLength;
  shape[1] = miniBatch;
  shape[2] = inputSize;
  if (theano_prep_output(dx, 3, shape, y->ga.typecode, GA_C_ORDER, c) != 0)
    goto fail;

  err = cudnnCreateTensorDescriptor(&dxdesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create dxdesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  /* Again not a typo, we need to use the last two dimensions */
  dims[0] = shape[1];
  dims[1] = shape[2];
  dims[2] = 1;
  strs[0] = dims[2] * dims[1];
  strs[1] = dims[2];
  strs[2] = 1;

  err = cudnnSetTensorNdDescriptor(dxdesc, dt, 3, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not set dxdesc: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  if (theano_prep_output(dhx, 3, PyGpuArray_DIMS(hx), hx->ga.typecode,
                         GA_C_ORDER, c) != 0)
    goto fail;

  if (c_make_tensorNd(*dhx, &dhxdesc) != 0)
    goto fail;

  if (cx != NULL) {
    if (theano_prep_output(dcx, 3, PyGpuArray_DIMS(cx), cx->ga.typecode,
                           GA_C_ORDER, c) != 0)
      goto fail;

    if (c_make_tensorNd(*dcx, &dcxdesc) != 0)
      goto fail;
  }

  yl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), seqLength);
  if (yl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < seqLength; i++)
    yl[i] = ydesc;
  
  dxl = (cudnnTensorDescriptor_t *)calloc(sizeof(cudnnTensorDescriptor_t), seqLength);
  if (dxl == NULL) {
    PyErr_NoMemory();
    goto fail;
  }

  for (size_t i = 0; i < seqLength; i++)
    dxl[i] = dxdesc;

  err = cudnnGetRNNWorkspaceSize(_handle, desc, (int)seqLength, dxl, &worksize);
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
                                       dxl, &ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not get reserve size: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  *oreserve = gpudata_alloc(c->ctx, ressize, NULL, 0, NULL);
  if (*oreserve == NULL) {
    PyErr_Format(PyExc_RuntimeError, "Could not allocate reserve");
    goto fail;
  }

  if (gpudata_move(*oreserve, 0, reserve, 0, ressize) != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "could not copy reserve");
    goto fail;
  }

  err = cudnnRNNBackwardData(_handle, desc, (int)seqLength,
                             yl, PyGpuArray_DEV_DATA(y),
                             /* y and dy are the same shape */
                             yl, PyGpuArray_DEV_DATA(dy),
                             dhydesc, dhy ? PyGpuArray_DEV_DATA(dhy) : NULL,
                             dcydesc, dcy ? PyGpuArray_DEV_DATA(dcy) : NULL,
                             wdesc, PyGpuArray_DEV_DATA(w),
                             hxdesc, PyGpuArray_DEV_DATA(hx),
                             cxdesc, cx ? PyGpuArray_DEV_DATA(cx) : NULL,
                             dxl, PyGpuArray_DEV_DATA(*dx),
                             dhxdesc, PyGpuArray_DEV_DATA(*dhx),
                             dcxdesc, dcx ? PyGpuArray_DEV_DATA(*dcx) : NULL,
                             *(void **)workspace, worksize,
                             *(void **)(*oreserve), ressize);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could run RNN grad inputs: %s",
                 cudnnGetErrorString(err));
    goto fail;
  }

  res = 0;
fail:
  if (ydesc != NULL)
    cudnnDestroyTensorDescriptor(ydesc);
  if (dhydesc != NULL)
    cudnnDestroyTensorDescriptor(dhydesc);
  if (dcydesc != NULL)
    cudnnDestroyTensorDescriptor(dcydesc);
  if (wdesc != NULL)
    cudnnDestroyFilterDescriptor(wdesc);
  if (hxdesc != NULL)
    cudnnDestroyTensorDescriptor(hxdesc);
  if (cxdesc != NULL)
    cudnnDestroyTensorDescriptor(cxdesc);
  if (dxdesc != NULL)
    cudnnDestroyTensorDescriptor(dxdesc);
  if (dhxdesc != NULL)
    cudnnDestroyTensorDescriptor(dhxdesc);
  if (dcxdesc != NULL)
    cudnnDestroyTensorDescriptor(dcxdesc);
  free(yl);
  free(dxl);
  if (workspace != NULL)
    gpudata_release(workspace);
  cuda_exit(c->ctx);
  return res;
}
