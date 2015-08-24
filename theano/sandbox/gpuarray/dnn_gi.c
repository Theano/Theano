#section support_code_struct

int
APPLY_SPECIFIC(conv_gi)(PyGpuArrayObject *kerns, PyGpuArrayObject *output,
                        PyGpuArrayObject *im,
                        cudnnConvolutionDescriptor_t desc,
                        double alpha, double beta, PyGpuArrayObject **input) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  float af = alpha, bf = beta;
  void *alpha_p;
  void *beta_p;

  if (PyGpuArray_DIMS(im)[1] != PyGpuArray_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size");
    return 1;
  }

  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  switch (im->ga.typecode) {
  case GA_DOUBLE:
    alpha_p = (void *)&alpha;
    beta_p = (void *)&beta;
    break;
  case GA_FLOAT:
    alpha_p = (void *)&af;
    beta_p = (void *)&bf;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "Unsupported type in convolution");
    return 1;
  }

#ifdef CONV_INPLACE
  Py_XDECREF(*input);
  *input = im;
  Py_INCREF(*input);
#else
  if (theano_prep_output(input, PyGpuArray_NDIM(im), PyGpuArray_DIMS(im),
                         im->ga.typecode, GA_C_ORDER,
                         pygpu_default_context()) != 0)
    return 1;
  if (beta != 0.0 && pygpu_move(*input, im))
    return 1;
#endif

  if (c_set_tensor4d(*input, APPLY_SPECIFIC(input)) == -1)
    return 1;

  err = cudnnConvolutionBackwardData(
    _handle,
    alpha_p,
    APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
    APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output),
    desc,
    beta_p,
    APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(*input));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradI: error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
