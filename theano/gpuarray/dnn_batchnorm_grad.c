#section init_code_struct

{
  cudnnStatus_t err;

  bn_doutput = NULL;
  if ((err = cudnnCreateTensorDescriptor(&bn_doutput)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
                 "(bn_doutput): %s", cudnnGetErrorString(err));
    FAIL;
  }
}

#section cleanup_code_struct

if (bn_doutput != NULL)
  cudnnDestroyTensorDescriptor(bn_doutput);

#section support_code_struct

cudnnTensorDescriptor_t bn_doutput;

int dnn_batchnorm_grad(PyGpuArrayObject *inp, PyGpuArrayObject *doutp,
                       PyGpuArrayObject *scale, PyGpuArrayObject *x_mean,
                       PyGpuArrayObject *x_invstd, npy_float64 epsilon,
                       PyGpuArrayObject **dinp, PyGpuArrayObject **dscale,
                       PyGpuArrayObject **dbias, PARAMS_TYPE* params) {
  PyGpuContextObject *c = inp->context;

  if (c_set_tensorNd(inp, bn_input) != 0)
    return 1;
  if (c_set_tensorNd(doutp, bn_doutput) != 0)
    return 1;
  if (c_set_tensorNd(scale, bn_params) != 0)
    return 1;

  if (epsilon < 1e-5) {
    PyErr_Format(PyExc_ValueError, "epsilon must be at least 1e-5, got %f", epsilon);
    return 1;
  }

  if (theano_prep_output(dinp, inp->ga.nd, inp->ga.dimensions, inp->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(dscale, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(dbias, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;

  if (c_set_tensorNd(*dinp, bn_output) != 0)
    return 1;

  {
    const float falpha = 1.;
    const float fbeta = 0.;
    const double dalpha = 1.;
    const double dbeta = 0.;
    void *alphaData;
    void *betaData;
    void *alphaParam;
    void *betaParam;
    if (inp->ga.typecode == GA_DOUBLE) {
      alphaData = (void *)&dalpha;
      betaData = (void *)&dbeta;
      alphaParam = (void *)&dalpha;
      betaParam = (void *)&dbeta;
    } else {
      alphaData = (void *)&falpha;
      betaData = (void *)&fbeta;
      alphaParam = (void *)&falpha;
      betaParam = (void *)&fbeta;
    }
    cudnnStatus_t err = cudnnBatchNormalizationBackward(
      params->handle,
      params->mode,
      alphaData,
      betaData,
      alphaParam,
      betaParam,
      bn_input,
      PyGpuArray_DEV_DATA(inp),
      bn_doutput,
      PyGpuArray_DEV_DATA(doutp),
      bn_output,
      PyGpuArray_DEV_DATA(*dinp),
      bn_params,
      PyGpuArray_DEV_DATA(scale),
      PyGpuArray_DEV_DATA(*dscale),
      PyGpuArray_DEV_DATA(*dbias),
      epsilon,
      PyGpuArray_DEV_DATA(x_mean),
      PyGpuArray_DEV_DATA(x_invstd)
      );
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "Error during batchnorm: %s\n",
                   cudnnGetErrorString(err));
      return 1;
    }
  }
  return 0;
}
