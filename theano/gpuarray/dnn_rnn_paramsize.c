#section support_code

int dnn_rnn_paramsize(cudnnRNNDescriptor_t desc,
                      PyArrayObject *isize,
                      npy_int32 typecode,
                      npy_uint64 *oparam_size,
                      cudnnHandle_t _handle) {
  cudnnTensorDescriptor_t xdesc;
  size_t param_size;
  cudnnStatus_t err;
  cudnnDataType_t dt;
  int shape[3];
  int strides[3];

  if (PyArray_DIM(isize, 0) != 2) {
    PyErr_SetString(PyExc_ValueError, "input_size should be of length two");
    return -1;
  }

  switch (typecode) {
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
    PyErr_SetString(PyExc_ValueError, "Unsupported data type");
    return -1;
  }

  err = cudnnCreateTensorDescriptor(&xdesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create tensor descriptor");
    return -1;
  }

  shape[0] = *(npy_uint64 *)PyArray_GETPTR1(isize, 0);
  shape[1] = *(npy_uint64 *)PyArray_GETPTR1(isize, 1);
  shape[2] = 1;
  strides[0] = shape[2] * shape[1];
  strides[1] = shape[2];
  strides[2] = 1;

  err = cudnnSetTensorNdDescriptor(xdesc, dt, 3, shape, strides);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not set tensor descriptor: %s",
                 cudnnGetErrorString(err));
    return -1;
  }

  err = cudnnGetRNNParamsSize(_handle, desc, xdesc, &param_size, dt);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Could not get parameter size");
    return -1;
  }

  cudnnDestroyTensorDescriptor(xdesc);
  *oparam_size = param_size;
  return 0;
}
