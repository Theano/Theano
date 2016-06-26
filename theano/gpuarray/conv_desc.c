#section support_code_apply

int APPLY_SPECIFIC(conv_desc)(PyArrayObject *filt_shp, PyArrayObject *subsample, PyArrayObject *padding,
                              PyObject *bmode, PyObject *conv_flag, PyObject *precision_flag,
                              cudnnConvolutionDescriptor_t *desc) {
  cudnnStatus_t err;
  int pad[3] = {0,0,0};
  int strides[3] = {0,0,0};
  int upscale[3] = {1, 1, 1};
  char *precision_flag;
  char *conv_mode;

strides[0] = *(npy_int64 *)PyArray_GETPTR1(subsample, 0);
strides[1] = *(npy_int64 *)PyArray_GETPTR1(subsample, 1);
strides[2] = *(npy_int64 *)PyArray_GETPTR1(subsample, 2);

if (PyInt_asLong(bmode) == 0L)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1;
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1;
}
else if (PyInt_asLong(bmode) == 2L)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) / 2;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) / 2;
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) / 2;
}

if (PyInt_asLong(conv_flag) == 1L)
{
  conv_mode = "CUDNN_CONVOLUTION";
}
else if (PyInt_asLong(conv_flag) == 2L)
{
  conv_mode = "CUDNN_CROSS_CORRELATION";
}

if (PyInt_asLong(precision_flag) == 1L)
{
  precision_flag = "CUDNN_DATA_HALF" ;
}
else if (PyInt_asLong(precision_flag) == 2L)
{
  precision_flag = "CUDNN_DATA_FLOAT";
}
else if (PyInt_asLong(precision_flag) == 3L)
{
  precision_flag = "CUDNN_DATA_DOUBLE";
}

  if (PyArray_DIM(filt_shp, 0) - 2 != 3) {
    PyErr_Format(PyExc_ValueError, "Filter shape has too many dimensions: "
                 "expected 3, got %lld.", 
                 (long long)PyArray_DIM(filt_shp, 0));
    return -1;
  }

  err = cudnnCreateConvolutionDescriptor(desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  err = cudnnSetConvolutionNdDescriptor(*desc, 3, pad, strides,
                                        upscale, conv_mode, precision_flag);
  return 0;
}
