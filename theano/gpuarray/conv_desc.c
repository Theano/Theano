#section support_code_apply

int APPLY_SPECIFIC(conv_desc)(PyArrayObject *filt_shp, 
                              PyArrayObject *padding,
                              PyArrayObject *subsample,
                              PyArrayObject *conv_mode,
                              PyArrayObject *precision,
                              PyArrayObject *bmode,
                              PyArrayObject *nb_dims,
                              cudnnConvolutionDescriptor_t *desc) {
  cudnnStatus_t err;
  int pad[3] = {*(npy_int8 *)PyArray_GETPTR1(padding, 0),
                *(npy_int8 *)PyArray_GETPTR1(padding, 1),
                *(npy_int8 *)PyArray_GETPTR1(padding, 2)};
  int strides[3] = {*(npy_int8 *)PyArray_GETPTR1(subsample, 0),
                    *(npy_int8 *)PyArray_GETPTR1(subsample, 1),
                    *(npy_int8 *)PyArray_GETPTR1(subsample, 2)};
  int upscale[3] = {1, 1, 1};
  int precision_code = *(npy_int8 *)PyArray_GETPTR1(precision, 0);
  cudnnDataType_t PRECISION;
  int conv_mode_code = *(npy_int8 *)PyArray_GETPTR1(conv_mode, 0);
  cudnnConvolutionMode_t CONV_MODE;
  int BORDER_MODE = *(npy_int8 *)PyArray_GETPTR1(bmode, 0);
  int NB_DIMS = *(npy_int8 *)PyArray_GETPTR1(nb_dims, 0);

if (precision_code == 16)
{
  PRECISION = CUDNN_DATA_HALF;
}
else if (precision_code == 32)
{
  PRECISION = CUDNN_DATA_FLOAT;
}
else if (precision_code == 64)
{
  PRECISION = CUDNN_DATA_DOUBLE;
}

if (conv_mode_code == 0)
{
  CONV_MODE = CUDNN_CONVOLUTION;
}
else if (conv_mode_code == 1)
{
  CONV_MODE = CUDNN_CROSS_CORRELATION;
}

if (BORDER_MODE == 0)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1;
if (NB_DIMS > 2)
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1;
}
else if (BORDER_MODE == 2)
{
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) / 2;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) / 2;
if (NB_DIMS > 2)
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) / 2;
}


  if (PyArray_DIM(filt_shp, 0) - 2 != NB_DIMS) {
    PyErr_Format(PyExc_ValueError, "Filter shape has too many dimensions: "
                 "expected %d, got %lld.", NB_DIMS,
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
