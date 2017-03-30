#section support_code_struct

int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
                     PyGpuArrayObject *bias, npy_float64 epsilon,
                     npy_float64 running_average_factor,
#ifdef RUNNING_AVERAGES
                     PyGpuArrayObject *in_running_mean,
                     PyGpuArrayObject *in_running_var,
#endif
                     PyGpuArrayObject **outp,
                     PyGpuArrayObject **x_mean,
                     PyGpuArrayObject **x_invstd,
#ifdef RUNNING_AVERAGES
                     PyGpuArrayObject **out_running_mean,
                     PyGpuArrayObject **out_running_var,
#endif
                     cudnnHandle_t _handle) {
  PyGpuContextObject *c = inp->context;

  if (c_set_tensorNd(inp, bn_input) != 0)
    return 1;
  if (c_set_tensorNd(scale, bn_params) != 0)
    return 1;

  if (epsilon < 1e-5) {
    PyErr_Format(PyExc_ValueError, "epsilon must be at least 1e-5, got %f", epsilon);
    return 1;
  }

#ifdef INPLACE_OUTPUT
  Py_XDECREF(*outp);
  *outp = inp;
  Py_INCREF(*outp);
#else
  if (theano_prep_output(outp, inp->ga.nd, inp->ga.dimensions, inp->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
#endif
  if (theano_prep_output(x_mean, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(x_invstd, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;

  if (c_set_tensorNd(*outp, bn_output) != 0)
    return 1;

#ifdef RUNNING_AVERAGES
#ifdef INPLACE_RUNNING_MEAN
  Py_XDECREF(*out_running_mean);
  PyGpuArrayObject *running_mean = in_running_mean;
  Py_INCREF(running_mean);
#else
  PyGpuArrayObject *running_mean = *out_running_mean;
  running_mean = theano_try_copy(running_mean, in_running_mean);
  if (running_mean == NULL) {
    return 1;
  }
#endif
#ifdef INPLACE_RUNNING_VAR
  Py_XDECREF(*out_running_var);
  PyGpuArrayObject *running_var = in_running_var;
  Py_INCREF(running_var);
#else
  PyGpuArrayObject *running_var = *out_running_var;
  running_var = theano_try_copy(running_var, in_running_var);
  if (running_var == NULL) {
    return 1;
  }
#endif
#endif

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
#ifdef RUNNING_AVERAGES
      running_average_factor,
      PyGpuArray_DEV_DATA(running_mean),
      PyGpuArray_DEV_DATA(running_var),
#else
      0,
      NULL,  // running mean, deliberately unused
      NULL,  // running var, deliberately unused
#endif
      epsilon,
      PyGpuArray_DEV_DATA(*x_mean),
      PyGpuArray_DEV_DATA(*x_invstd)
      );
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "Error during batchnorm: %s\n",
                   cudnnGetErrorString(err));
      return 1;
    }
#ifdef RUNNING_AVERAGES
    *out_running_mean = running_mean;
    *out_running_var = running_var;
#endif
  }
  return 0;
}
