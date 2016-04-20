#section support_code
static cudnnHandle_t _handle = NULL;


static int
c_set_tensorNd(CudaNdarray *var, cudnnTensorDescriptor_t desc) {

  int dim = CudaNdarray_NDIM(var);
  int *strides = (int *)malloc(dim * sizeof(int));
  int default_str = 1;
  int return_value = 0;
  
  if (strides != NULL) {
    for (int i = dim-1; i >= 0; i--)
    {
      if (CudaNdarray_HOST_STRIDES(var)[i])
        strides[i] = CudaNdarray_HOST_STRIDES(var)[i];
      else
        strides[i] = default_str;
      default_str *= CudaNdarray_HOST_DIMS(var)[i];
    }
    
    cudnnStatus_t err = cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, dim,
                                                   CudaNdarray_HOST_DIMS(var),
                                                   strides);
  	 									
    
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
		  "Could not set tensorNd descriptor: %s"
		  "dim=%d",
		  cudnnGetErrorString(err), dim);
		  
	  return_value = -1;
    }
  } else {
    PyErr_Format(PyExc_MemoryError,
		"Could not allocate memory for strides array of size %d.",
		dim);
		
    return_value = -1;  
  }
    
  free(strides);
  return return_value;
}


static int
c_set_filterNd(CudaNdarray *var, cudnnFilterDescriptor_t desc) {
  if (!CudaNdarray_is_c_contiguous(var)) {
    PyErr_SetString(PyExc_ValueError,
		    "Only contiguous filters (kernels) are supported.");
    return -1;
  }
  int dim = CudaNdarray_NDIM(var);
  cudnnStatus_t err = cudnnSetFilterNdDescriptor_v4(desc,
                                                    CUDNN_DATA_FLOAT,
                                                    CUDNN_TENSOR_NCHW,
                                                    dim,
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
