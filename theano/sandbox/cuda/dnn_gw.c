#section support_code_struct

int 
APPLY_SPECIFIC(conv_gw)(CudaNdarray *input, CudaNdarray *output,
			cudnnConvolutionDescriptor_t desc,
			int h, int w,
			CudaNdarray **kerns) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  {
    int out_dims[4];
    out_dims[0] = CudaNdarray_HOST_DIMS(output)[1];
    out_dims[1] = CudaNdarray_HOST_DIMS(input)[1];
    out_dims[2] = h;
    out_dims[3] = w;
    if (CudaNdarray_prep_output(kerns, 4, out_dims) != 0) {
      return 1;
    }
  }

  if (c_set_filter(*kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  {
    const float alpha = 1;
    const float beta = 0;

    err = cudnnConvolutionBackwardFilter(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
      desc,
      (void *)&beta,
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(*kerns));
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradW: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
