#section support_code_apply

int APPLY_SPECIFIC(conv_desc)(PyArrayObject *filt_shp, 
                              PyArrayObject* padding,
                              PyArrayObject* subsample, 
                              PyObject* conv_mode, 
                              PyObject* precision,
                              PyArrayObject* bmode,
                              PyObject* nb_dims,
                              cudnnConvolutionDescriptor_t *desc) {
  cudnnStatus_t err;
  int pad[3] = {*(npy_int64 *)PyArray_GETPTR1(padding, 0),
                *(npy_int64 *)PyArray_GETPTR1(padding, 1),
                *(npy_int64 *)PyArray_GETPTR1(padding, 2)};
  int strides[3] = {*(npy_int64 *)PyArray_GETPTR1(subsample, 0),
                    *(npy_int64 *)PyArray_GETPTR1(subsample, 1),
                    *(npy_int64 *)PyArray_GETPTR1(subsample, 2)};
  int upscale[3] = {1, 1, 1};
  long precision_code = PyInt_AsLong(precision);
  cudnnDataType_t PRECISION;
  long conv_mode_code = PyInt_AsLong(conv_mode);
  cudnnConvolutionMode_t CONV_MODE;
  long BORDER_MODE = *(npy_int64 *)PyArray_GETPTR1(bmode, 0);
  long NB_DIMS = PyInt_AsLong(nb_dims);

if (precision_code == 16L)
{
  PRECISION = CUDNN_DATA_HALF;
}
else if (precision_code == 32L)
{
  PRECISION = CUDNN_DATA_FLOAT;
}
else if (precision_code == 64L)
{
  PRECISION = CUDNN_DATA_DOUBLE;
}

if (conv_mode_code == 0L)
{
  CONV_MODE = CUDNN_CONVOLUTION;
}
else if (conv_mode_code == 1L)
{
  CONV_MODE = CUDNN_CROSS_CORRELATION;
}

if (BORDER_MODE == 0L)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1;
if (NB_DIMS > 2L)
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1;
}
else if (BORDER_MODE == 2L)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) / 2;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) / 2;
if (NB_DIMS > 2L)
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) / 2;
}


  if (PyArray_DIM(filt_shp, 0) - 2 != NB_DIMS) {
    PyErr_Format(PyExc_ValueError, "Filter shape has too many dimensions: "
                 "expected %ld, got %lld.", NB_DIMS,
                 (long long)PyArray_DIM(filt_shp, 0));
    return -1;
  }

  err = cudnnCreateConvolutionDescriptor(desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  err = cudnnSetConvolutionNdDescriptor(*desc, NB_DIMS, pad, strides,
                                        upscale, CONV_MODE, PRECISION);
  return 0;
}
