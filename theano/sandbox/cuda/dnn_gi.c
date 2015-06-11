#section support_code_struct

int
APPLY_SPECIFIC(conv_gi)(CudaNdarray *kerns, CudaNdarray *output,
                        CudaNdarray *im, cudnnConvolutionDescriptor_t desc,
                        float alpha, float beta, CudaNdarray **input) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (CudaNdarray_HOST_DIMS(im)[1] != CudaNdarray_HOST_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*input);
  *input = im;
  Py_INCREF(*input);
#else
  if (CudaNdarray_prep_output(input, 4, CudaNdarray_HOST_DIMS(im)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*input, im))
    return 1;
#endif

  if (c_set_tensor4d(*input, APPLY_SPECIFIC(input)) == -1)
    return 1;

  err = cudnnConvolutionBackwardData(
    _handle,
    (void *)&alpha,
    APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
    APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
    desc,
    (void *)&beta,
    APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(*input));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradI: error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
