#section support_code_apply

int APPLY_SPECIFIC(conv_desc)(PyArrayObject *filt_shp,
                              cudnnConvolutionDescriptor_t *desc) {
  cudnnStatus_t err;
  int pad[3] = {PAD_0, PAD_1, PAD_2};
  int strides[3] = {SUB_0, SUB_1, SUB_2};
  int upscale[3] = {1, 1, 1};

#if BORDER_MODE == 0
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1;
#if NB_DIMS > 2
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1;
#endif
#elif BORDER_MODE == 2
  pad[0] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) / 2;
  pad[1] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) / 2;
#if NB_DIMS > 2
  pad[2] = *(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) / 2;
#endif
#endif

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

  err = cudnnSetConvolutionNdDescriptor_v3(*desc, NB_DIMS, pad, strides,
                                           upscale, CONV_MODE, PRECISION);
  return 0;
}
