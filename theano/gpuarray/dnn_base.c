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
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_tensorNd");
    return -1;
  }
  ds = gpuarray_get_elsize(var->ga.typecode);

  int strs[8], dims[8], default_stride = 1;
  unsigned int nd = PyGpuArray_NDIM(var);

  if (nd > 8) {
    PyErr_SetString(PyExc_TypeError, "Tensor of more than 8d");
    return -1;
  }

  for (unsigned int _i = nd; _i > 0; _i--) {
    unsigned int i = _i - 1;
    strs[i] = (PyGpuArray_DIM(var, i) != 1 && PyGpuArray_STRIDE(var, i)) ?
      PyGpuArray_STRIDE(var, i)/ds : default_stride;
    default_stride *= PyGpuArray_DIM(var, i);
    dims[i] = PyGpuArray_DIM(var, i);
  }

  /* Tensors can't be smaller than 3d for cudnn so we pad the
   * descriptor if they are */
  for (unsigned int i = nd; i < 3; i++) {
    strs[i] = 1;
    dims[i] = 1;
  }

  cudnnStatus_t err = cudnnSetTensorNdDescriptor(desc, dt, nd < 3 ? 3 : nd,
                                                 dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensorNd descriptor: %s",
		 cudnnGetErrorString(err));
    return -1;
  }
  return 0;
}

static int c_make_tensorNd(PyGpuArrayObject *var, cudnnTensorDescriptor_t *desc) {
  cudnnStatus_t err;
  err = cudnnCreateTensorDescriptor(desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create tensor descriptor: %s",
                 cudnnGetErrorString(err));
    return -1;
  }
  if (c_set_tensorNd(var, *desc) != 0) {
    cudnnDestroyTensorDescriptor(*desc);
    return -1;
  }
  return 0;
}

static int
c_set_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t desc) {
  cudnnDataType_t dt;
  cudnnStatus_t err;

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
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Non-float datatype in c_set_filter");
    return -1;
  }

  int dims[8];
  unsigned int nd = PyGpuArray_NDIM(var);

  if (nd > 8) {
    PyErr_SetString(PyExc_TypeError, "Tensor of more than 8d");
    return -1;
  }

  for (unsigned int _i = nd; _i > 0; _i--) {
    unsigned int i = _i - 1;
    dims[i] = PyGpuArray_DIM(var, i);
  }

  /* Filters can't be less than 3d so we pad */
  for (unsigned int i = nd; i < 3; i++)
    dims[i] = 1;

  if (nd < 3)
    nd = 3;

    err = cudnnSetFilterNdDescriptor(desc, dt, CUDNN_TENSOR_NCHW, nd, dims);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s.",
		 cudnnGetErrorString(err));
    return -1;
  }
  return 0;
}

static int c_make_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t *desc) {
  cudnnStatus_t err;
  err = cudnnCreateFilterDescriptor(desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "Could not create tensor descriptor: %s",
                 cudnnGetErrorString(err));
    return -1;
  }
  if (c_set_filter(var, *desc) != 0) {
    cudnnDestroyFilterDescriptor(*desc);
    return -1;
  }
  return 0;
}

#section init_code

setup_ext_cuda();
