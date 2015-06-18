#section support_code
static cudnnHandle_t _handle = NULL;


static int
c_set_tensorNd(CudaNdarray *var, int dim, cudnnTensorDescriptor_t desc) {


  int strides[dim];

  for (int i = 0; i < dim; ++i)
  {
    if (CudaNdarray_HOST_STRIDES(var)[i])
      strides[i] = CudaNdarray_HOST_STRIDES(var)[i];
    else
    {
      strides[i] = 1;
      for (int j = i + 1; j < dim; ++j)
        strides[i] *= CudaNdarray_HOST_DIMS(var)[j];
    }
  }

  cudnnStatus_t err = cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, dim,
                                                 CudaNdarray_HOST_DIMS(var),
                                                 strides);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensorNd descriptor: %s"
		 "dim=%d",
		 cudnnGetErrorString(err), dim);
    return -1;
  }
  return 0;
}


static int
c_set_filterNd(CudaNdarray *var, int dim, cudnnFilterDescriptor_t desc) {
  if (!CudaNdarray_is_c_contiguous(var)) {
    PyErr_SetString(PyExc_ValueError,
		    "Only contiguous filters (kernels) are supported.");
    return -1;
  }
  cudnnStatus_t err = cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, dim,
                                                 CudaNdarray_HOST_DIMS(var));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s."
		 " dims= %d",
		 cudnnGetErrorString(err), dim);
    return -1;
  }
  return 0;
}

#section init_code

{
  cudnnStatus_t err;
  if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not create cuDNN handle: %s",
		 cudnnGetErrorString(err));
#if PY_MAJOR_VERSION >= 3
    return NULL;
#else
    return;
#endif
  }
}
