#section support_code_apply

static int c_set_groups_for_conv(cudnnConvolutionDescriptor_t desc, int groups) {
#if CUDNN_MAJOR >= 7
  cudnnStatus_t err = cudnnSetConvolutionGroupCount(desc, groups);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		   "error setting groups for convolution : %s",
		   cudnnGetErrorString(err));
    return -1;
  }
#endif
  return 0;
}

int APPLY_SPECIFIC(conv_desc)(PyArrayObject *filt_shp,
                              cudnnConvolutionDescriptor_t *desc,
                              PARAMS_TYPE* params) {
  cudnnStatus_t err;
  int pad[3] = {params->pad0, params->pad1, params->pad2};
  int strides[3] = {params->sub0, params->sub1, params->sub2};
  int dilation[3] = {params->dil0, params->dil1, params->dil2};

  if (params->bmode == BORDER_MODE_FULL) {
    pad[0] = (*(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1) * dilation[0];
    pad[1] = (*(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1) * dilation[1];
    if (params->nb_dims > 2) {
      pad[2] = (*(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1) * dilation[2];
    }
  } else if(params->bmode == BORDER_MODE_HALF) {
    pad[0] = ((*(npy_int64 *)PyArray_GETPTR1(filt_shp, 2) - 1) * dilation[0] + 1) / 2;
    pad[1] = ((*(npy_int64 *)PyArray_GETPTR1(filt_shp, 3) - 1) * dilation[1] + 1) / 2;
    if (params->nb_dims > 2) {
      pad[2] = ((*(npy_int64 *)PyArray_GETPTR1(filt_shp, 4) - 1) * dilation[2] + 1) / 2;
    }
  }

  if (PyArray_DIM(filt_shp, 0) - 2 != params->nb_dims) {
    PyErr_Format(PyExc_ValueError, "Filter shape has too many dimensions: "
                 "expected %d, got %lld.", params->nb_dims,
                 (long long)PyArray_DIM(filt_shp, 0));
    return -1;
  }

  err = cudnnCreateConvolutionDescriptor(desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }

  err = cudnnSetConvolutionNdDescriptor(*desc, params->nb_dims, pad, strides,
                                        dilation, params->conv_mode, params->precision);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not set convolution "
                 "descriptor: %s", cudnnGetErrorString(err));
    return -1;
  }
  if (c_set_groups_for_conv(*desc, params->num_groups) == -1)
      return -1;
  return 0;
}
