#section support_code
static cudnnHandle_t _handle = NULL;

static int
c_set_tensor4d(PyGpuArrayObject *var, cudnnTensorDescriptor_t desc) {
  cudnnDataType_t dt;
  switch (var->ga.typecode) {
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
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_tensor4d");
    return -1;
  }
  cudnnStatus_t err = cudnnSetTensor4dDescriptorEx(
    desc, dt,
    PyGpuArray_DIM(var, 0), PyGpuArray_DIM(var, 1),
    PyGpuArray_DIM(var, 2), PyGpuArray_DIM(var, 3),
    PyGpuArray_STRIDE(var, 0), PyGpuArray_STRIDE(var, 1),
    PyGpuArray_STRIDE(var, 2), PyGpuArray_STRIDE(var, 3));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensor4d descriptor: %s"
		 "shapes=%d %d %d %d strides=%d %d %d %d",
		 cudnnGetErrorString(err),
		 PyGpuArray_DIMS(var)[0],
		 PyGpuArray_DIMS(var)[1],
		 PyGpuArray_DIMS(var)[2],
		 PyGpuArray_DIMS(var)[3],
		 PyGpuArray_STRIDES(var)[0],
		 PyGpuArray_STRIDES(var)[1],
		 PyGpuArray_STRIDES(var)[2],
		 PyGpuArray_STRIDES(var)[3]);
    return -1;
  }
  return 0;
}

static int
c_set_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t desc) {
  cudnnDataType_t dt;
  if (!GpuArray_IS_C_CONTIGUOUS(&var->ga))
    PyErr_SetString(PyExc_ValueError,
		    "Only contiguous filters (kernels) are supported.");
    return -1;
  }
  switch (var->ga.typecode) {
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
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_filter");
    return -1;
  }
  cudnnStatus_t err = cudnnSetFilter4dDescriptor(
    desc, dt,
    PyGpuArray_DIMS(var)[0], PyGpuArray_DIMS(var)[1],
    PyGpuArray_DIMS(var)[2], PyGpuArray_DIMS(var)[3]);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s."
		 " dims= %d %d %d %d",
		 cudnnGetErrorString(err),
		 PyGpuArray_DIMS(var)[0],
		 PyGpuArray_DIMS(var)[1],
		 PyGpuArray_DIMS(var)[2],
		 PyGpuArray_DIMS(var)[3]);
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
