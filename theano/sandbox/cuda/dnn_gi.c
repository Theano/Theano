#section support_code_struct

int
APPLY_SPECIFIC(conv_gi)(CudaNdarray *kerns, CudaNdarray *output,
			cudnnConvolutionDescriptor_t desc,
			int h, int w, float alpha, float beta,
			CudaNdarray **input) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  {
    int out_dims[4];
    out_dims[0] = CudaNdarray_HOST_DIMS(output)[0];
    out_dims[1] = CudaNdarray_HOST_DIMS(kerns)[1];
    out_dims[2] = h;
    out_dims[3] = w;
    if (CudaNdarray_prep_output(input, 4, out_dims) != 0) {
      return 1;
    }
  }

  if (c_set_tensor4d(*input, APPLY_SPECIFIC(input)) == -1)
    return 1;

  {
    err = cudnnConvolutionBackwardData(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
      desc,
      (void *)&beta,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(*input));
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradI: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
