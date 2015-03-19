#section support_code
static cudnnHandle_t _handle = NULL;

static int
c_set_tensor4d(CudaNdarray *var, cudnnTensorDescriptor_t desc) {
  cudnnStatus_t err = cudnnSetTensor4dDescriptorEx(
    desc, CUDNN_DATA_FLOAT,
    CudaNdarray_HOST_DIMS(var)[0],
    CudaNdarray_HOST_DIMS(var)[1],
    CudaNdarray_HOST_DIMS(var)[2],
    CudaNdarray_HOST_DIMS(var)[3],
    CudaNdarray_HOST_STRIDES(var)[0]?CudaNdarray_HOST_STRIDES(var)[0]:CudaNdarray_HOST_DIMS(var)[2]*CudaNdarray_HOST_DIMS(var)[3]*CudaNdarray_HOST_DIMS(var)[1],
    CudaNdarray_HOST_STRIDES(var)[1]?CudaNdarray_HOST_STRIDES(var)[1]:CudaNdarray_HOST_DIMS(var)[2]*CudaNdarray_HOST_DIMS(var)[3],
    CudaNdarray_HOST_STRIDES(var)[2]?CudaNdarray_HOST_STRIDES(var)[2]:CudaNdarray_HOST_DIMS(var)[3],
    CudaNdarray_HOST_STRIDES(var)[3]?CudaNdarray_HOST_STRIDES(var)[3]:1
    );
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensor4d descriptor: %s"
		 "shapes=%d %d %d %d strides=%d %d %d %d",
		 cudnnGetErrorString(err),
		 CudaNdarray_HOST_DIMS(var)[0],
		 CudaNdarray_HOST_DIMS(var)[1],
		 CudaNdarray_HOST_DIMS(var)[2],
		 CudaNdarray_HOST_DIMS(var)[3],
		 CudaNdarray_HOST_STRIDES(var)[0]?CudaNdarray_HOST_STRIDES(var)[0]:CudaNdarray_HOST_DIMS(var)[2]*CudaNdarray_HOST_DIMS(var)[3]*CudaNdarray_HOST_DIMS(var)[1],
		 CudaNdarray_HOST_STRIDES(var)[1]?CudaNdarray_HOST_STRIDES(var)[1]:CudaNdarray_HOST_DIMS(var)[2]*CudaNdarray_HOST_DIMS(var)[3],
		 CudaNdarray_HOST_STRIDES(var)[2]?CudaNdarray_HOST_STRIDES(var)[2]:CudaNdarray_HOST_DIMS(var)[3],
		 CudaNdarray_HOST_STRIDES(var)[3]?CudaNdarray_HOST_STRIDES(var)[3]:1
      );
    return -1;
  }
  return 0;
}

static int
c_set_filter(CudaNdarray *var, cudnnFilterDescriptor_t desc) {
  if (!CudaNdarray_is_c_contiguous(var)) {
    PyErr_SetString(PyExc_ValueError,
		    "Only contiguous filters (kernels) are supported.");
    return -1;
  }
  cudnnStatus_t err = cudnnSetFilter4dDescriptor(
    desc, CUDNN_DATA_FLOAT,
    CudaNdarray_HOST_DIMS(var)[0],
    CudaNdarray_HOST_DIMS(var)[1],
    CudaNdarray_HOST_DIMS(var)[2],
    CudaNdarray_HOST_DIMS(var)[3]
    );
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s."
		 " dims= %d %d %d %d",
		 cudnnGetErrorString(err),
		 CudaNdarray_HOST_DIMS(var)[0],
		 CudaNdarray_HOST_DIMS(var)[1],
		 CudaNdarray_HOST_DIMS(var)[2],
		 CudaNdarray_HOST_DIMS(var)[3]);
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
