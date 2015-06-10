#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(CudaNdarray *input, CudaNdarray *kerns,
                         CudaNdarray *om, cudnnConvolutionDescriptor_t desc,
                         float alpha, float beta, CudaNdarray **output) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  if (CudaNdarray_HOST_DIMS(input)[1] != CudaNdarray_HOST_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*output);
  *output = om;
  Py_INCREF(*output);
#else
  if (CudaNdarray_prep_output(output, 4, CudaNdarray_HOST_DIMS(om)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*output, om))
    return 1;
#endif

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
