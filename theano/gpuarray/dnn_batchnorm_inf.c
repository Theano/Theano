#section support_code_struct

int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
                     PyGpuArrayObject *bias, PyGpuArrayObject *est_mean,
                     PyGpuArrayObject *est_var, npy_float64 epsilon, 
                     PyGpuArrayObject **outp, PARAMS_TYPE* params) {
  PyGpuContextObject *c = inp->context;

  if (c_set_tensorNd(inp, bn_input) != 0)
    return 1;
  if (c_set_tensorNd(scale, bn_params) != 0)
    return 1;

  if (epsilon < 1e-5) {
    PyErr_Format(PyExc_ValueError, "epsilon must be at least 1e-5, got %f", epsilon);
    return 1;
  }

  if (params->inplace) {
    Py_XDECREF(*outp);
    *outp = inp;
    Py_INCREF(*outp);
  } else {
    if (theano_prep_output(outp, inp->ga.nd, inp->ga.dimensions, inp->ga.typecode, GA_C_ORDER, c) != 0)
      return 1;
  }

  if (c_set_tensorNd(*outp, bn_output) != 0)
    return 1;

  {
    const float falpha = 1.;
    const float fbeta = 0.;
    const double dalpha = 1.;
    const double dbeta = 0.;
    void *alpha;
    void *beta;
    if (inp->ga.typecode == GA_DOUBLE) {
      alpha = (void *)&dalpha;
      beta = (void *)&dbeta;
    } else {
      alpha = (void *)&falpha;
      beta = (void *)&fbeta;
    }
    cudnnStatus_t err = cudnnBatchNormalizationForwardInference(
      params->handle,
      params->mode,
      alpha,
      beta,
      bn_input,
      PyGpuArray_DEV_DATA(inp),
      bn_output,
      PyGpuArray_DEV_DATA(*outp),
      bn_params,
      PyGpuArray_DEV_DATA(scale),
      PyGpuArray_DEV_DATA(bias),
      PyGpuArray_DEV_DATA(est_mean),
      PyGpuArray_DEV_DATA(est_var),
      epsilon
      );
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "Error during batchnorm: %s\n",
                   cudnnGetErrorString(err));
      return 1;
    }
  }
  return 0;
}
