#section support_code

static int
c_set_tensorNd(PyGpuArrayObject *var, cudnnTensorDescriptor_t desc) {
  cudnnDataType_t dt;
  size_t ds;
  switch (var->ga.typecode) {
  case GA_FLOAT:
    dt = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    dt = CUDNN_DATA_DOUBLE;
    break;
#if CUDNN_VERSION > 3000
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
#endif
  default:
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_tensorNd");
    return -1;
  }
  ds = gpuarray_get_elsize(var->ga.typecode);

  int strs[5], dims[5], default_stride = 1;
  unsigned int nd = PyGpuArray_NDIM(var);

  if (nd > 5) {
    PyErr_SetString(PyExc_TypeError, "Tensor of more than 5d");
    return -1;
  }

  for (unsigned int _i = nd; _i > 0; _i--) {
    unsigned int i = _i - 1;
    strs[i] = PyGpuArray_STRIDE(var, i) ?
      PyGpuArray_STRIDE(var, i)/ds : default_stride;
    default_stride *= PyGpuArray_DIM(var, i);
    dims[i] = PyGpuArray_DIM(var, i);
  }

  cudnnStatus_t err = cudnnSetTensorNdDescriptor(desc, dt, nd, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensorNd descriptor: %s",
		 cudnnGetErrorString(err));
    return -1;
  }
  return 0;
}

static int
c_set_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t desc) {
  cudnnDataType_t dt;
  if (!GpuArray_IS_C_CONTIGUOUS(&var->ga)) {
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
#if CUDNN_VERSION > 3000
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
#endif
  default:
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_filter");
    return -1;
  }

  int dims[5];
  unsigned int nd = PyGpuArray_NDIM(var);

  if (nd > 5) {
    PyErr_SetString(PyExc_TypeError, "Tensor of more than 5d");
    return -1;
  }

  for (unsigned int _i = nd; _i > 0; _i--) {
    unsigned int i = _i - 1;
    dims[i] = PyGpuArray_DIM(var, i);
  }

  cudnnStatus_t err = cudnnSetFilterNdDescriptor(desc, dt, nd, dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s.",
		 cudnnGetErrorString(err));
    return -1;
  }
  return 0;
}

#section init_code

setup_ext_cuda();

#section support_code_struct

PyGpuContextObject *ctx;
cudnnHandle_t APPLY_SPECIFIC(_handle);

#section init_code_struct

{
  // We need to keep a reference here to have it available in the destructor.
  ctx = PARAMS;
  Py_INCREF(ctx);

  cuda_enter(PARAMS->ctx);
  cudnnStatus_t err;
  APPLY_SPECIFIC(_handle) = NULL;
  if ((err = cudnnCreate(&APPLY_SPECIFIC(_handle))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not create cuDNN handle: %s",
                 cudnnGetErrorString(err));
    cuda_exit(PARAMS->ctx);
    FAIL;
  }
  if ((err = cudnnSetStream(APPLY_SPECIFIC(_handle),
                            cuda_get_stream(PARAMS->ctx))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not set cudnn stream: %s",
                 cudnnGetErrorString(err));
    cuda_exit(PARAMS->ctx);
    FAIL;
  }
  cuda_exit(PARAMS->ctx);
}

#section cleanup_code_struct

cuda_enter(ctx->ctx);
cudnnDestroy(APPLY_SPECIFIC(_handle));
cuda_exit(ctx->ctx);
Py_DECREF((PyObject *)ctx);
