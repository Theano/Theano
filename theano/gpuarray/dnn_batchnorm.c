#section support_code_struct

int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
                     PyGpuArrayObject *bias, npy_float64 epsilon,
                     PyGpuArrayObject **outp, PyGpuArrayObject **x_mean,
                     PyGpuArrayObject **x_invstd, cudnnHandle_t _handle) {
  PyGpuContextObject *c = inp->context;

  if (c_set_tensorNd(inp, bn_input) != 0)
    return 1;
  if (c_set_tensorNd(scale, bn_params) != 0)
    return 1;

  if (epsilon < 1e-5)
    return 1;

  if (theano_prep_output(outp, inp->ga.nd, inp->ga.dimensions, inp->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(x_mean, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(x_invstd, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;

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
    cudnnStatus_t err = cudnnBatchNormalizationForwardTraining(
      _handle,
      MODE,
      alpha,
      beta,
      bn_input,
      PyGpuArray_DEV_DATA(inp),
      bn_output,
      PyGpuArray_DEV_DATA(*outp),
      bn_params,
      PyGpuArray_DEV_DATA(scale),
      PyGpuArray_DEV_DATA(bias),
      0,
      NULL,  // running mean, deliberately unused
      NULL,  // running var, deliberately unused
      epsilon,
      PyGpuArray_DEV_DATA(*x_mean),
      PyGpuArray_DEV_DATA(*x_invstd)
      );
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "Error during batchnorm: %s\n",
                   cudnnGetErrorString(err));
      return 1;
    }
  }
  return 0;
}
