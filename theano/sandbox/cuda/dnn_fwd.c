#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(CudaNdarray *input, CudaNdarray *kerns,
                         cudnnConvolutionDescriptor_t desc,
                         float alpha, float beta,
                         CudaNdarray **output) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  {
    int out_dims[4];
    err = cudnnGetConvolution2dForwardOutputDim(
      desc,
      APPLY_SPECIFIC(input),
      APPLY_SPECIFIC(kerns),
      &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3]);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
		   "GpuDnnConv: error while computing the output shape: %s",
		   cudnnGetErrorString(err));
      return 1;
    }
    if (CudaNdarray_prep_output(output, 4, out_dims) != 0) {
      return 1;
    }
  }

  if (c_set_tensor4d(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  {
    size_t worksize;
    void *workspace;

    err = cudnnGetConvolutionForwardWorkspaceSize(_handle,
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  CONV_ALGO,
                                                  &worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDnnConv: error getting worksize: %s",
                   cudnnGetErrorString(err));
      return 1;
    }

    workspace = get_work_mem(worksize);
    if (workspace == NULL && worksize != 0)
      return 1;

    err = cudnnConvolutionForward(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
      desc,
      CONV_ALGO,
      workspace, worksize,
      (void *)&beta,
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(*output));
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConv: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
