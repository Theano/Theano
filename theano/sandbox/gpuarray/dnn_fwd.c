#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(PyGpuArrayObject *input, PyGpuArrayObject *kerns,
                         PyGpuArrayObject *om,
                         cudnnConvolutionDescriptor_t desc,
                         double alpha, double beta,
                         PyGpuArrayObject **output) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  float af = alpha, bf = beta;
  void *alpha_p;
  void *beta_p;

  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size");
    return 1;
  }

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  switch (input->ga.typecode) {
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
  Py_XDECREF(*output);
  *output = om;
  Py_INCREF(*output);
#else
  if (theano_prep_output(output, PyGpuArray_NDIM(om), PyGpuArray_DIMS(om),
                         om->ga.typecode, GA_C_ORDER,
                         pygpu_default_context()) != 0)
    return 1;
  if (beta != 0.0 && pygpu_move(*output, om))
    return 1;
#endif

  if (c_set_tensor4d(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  {
    size_t worksize;
    gpudata *workspace;
    PyGpuContextObject *c;

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

    /* 
     * This is less than ideal since we need to free it after (which
     * introduces a synchronization point. But we don't have a module
     * to place a nice get_work_mem() function in.
     */
    if (worksize != 0) {
      c = pygpu_default_context();
      workspace = c->ops->buffer_alloc(c->ctx, worksize, NULL, 0, NULL);
      if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate working memory");
        return 1;
      }
    }

    err = cudnnConvolutionForward(
      _handle,
      alpha_p,
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
      desc, CONV_ALGO,
      worksize == 0 ? NULL : *(void **)workspace, worksize,
      beta_p,
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output));

    if (worksize != 0)
      c->ops->buffer_release(workspace);
  }

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConv: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
