#section support_code_struct

int 
APPLY_SPECIFIC(conv_gw)(CudaNdarray *input, CudaNdarray *output,
                        CudaNdarray *km, cudnnConvolutionDescriptor_t desc,
                        float alpha, float beta, CudaNdarray **kerns) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (CudaNdarray_HOST_DIMS(input)[1] != CudaNdarray_HOST_DIMS(km)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*kerns);
  *kerns = km;
  Py_INCREF(*kerns);
#else
  if (CudaNdarray_prep_output(kerns, 4, CudaNdarray_HOST_DIMS(km)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*kerns, km))
    return 1;
#endif

  if (c_set_filter(*kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  err = cudnnConvolutionBackwardFilter(
    _handle,
    (void *)&alpha,
    APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
    APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
    desc,
    (void *)&beta,
    APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(*kerns));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradW: error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
