#section support_code_struct

int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
                     PyGpuArrayObject *bias, npy_float64 epsilon,
                     npy_float64 running_average_factor,
                     PyGpuArrayObject *in_running_mean, // may be NULL
                     PyGpuArrayObject *in_running_var, // may be NULL
                     PyGpuArrayObject **outp,
                     PyGpuArrayObject **x_mean,
                     PyGpuArrayObject **x_invstd,
                     PyGpuArrayObject **out_running_mean, // may be NULL
                     PyGpuArrayObject **out_running_var, // may be NULL
                     PARAMS_TYPE* params) {
  /* Note: based on Python code, in_running_mean, in_running_var, out_running_mean and out_running_var
  are together NULL (or not NULL) at same time, so we just need to check only one of them. */
  bool running_averages = (in_running_mean != NULL);
  PyGpuContextObject *c = inp->context;

  if (c_set_tensorNd(inp, bn_input) != 0)
    return 1;
  if (c_set_tensorNd(scale, bn_params) != 0)
    return 1;

  if (epsilon < 1e-5) {
    PyErr_Format(PyExc_ValueError, "epsilon must be at least 1e-5, got %f", epsilon);
    return 1;
  }

  if (params->inplace_output) {
    Py_XDECREF(*outp);
    *outp = inp;
    Py_INCREF(*outp);
  } else if (theano_prep_output(outp, inp->ga.nd, inp->ga.dimensions, inp->ga.typecode, GA_C_ORDER, c) != 0) {
    return 1;
  }

  if (theano_prep_output(x_mean, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (theano_prep_output(x_invstd, scale->ga.nd, scale->ga.dimensions, scale->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;

  if (c_set_tensorNd(*outp, bn_output) != 0)
    return 1;

  PyGpuArrayObject *running_mean = NULL;
  PyGpuArrayObject *running_var = NULL;
  if (running_averages) {
    if (params->inplace_running_mean) {
      Py_XDECREF(*out_running_mean);
      running_mean = in_running_mean;
      Py_INCREF(running_mean);
    } else {
      running_mean = *out_running_mean;
      running_mean = theano_try_copy(running_mean, in_running_mean);
      if (running_mean == NULL) {
        return 1;
      }
    }
    if (params->inplace_running_var) {
      Py_XDECREF(*out_running_var);
      running_var = in_running_var;
      Py_INCREF(running_var);
    } else {
      running_var = *out_running_var;
      running_var = theano_try_copy(running_var, in_running_var);
      if (running_var == NULL) {
        return 1;
      }
    }
  }

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
      running_averages ? running_average_factor : 0,
      running_averages ? PyGpuArray_DEV_DATA(running_mean) : NULL,
      running_averages ? PyGpuArray_DEV_DATA(running_var): NULL,
      epsilon,
      PyGpuArray_DEV_DATA(*x_mean),
      PyGpuArray_DEV_DATA(*x_invstd)
      );
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "Error during batchnorm: %s\n",
                   cudnnGetErrorString(err));
      return 1;
    }
    if (running_averages) {
      *out_running_mean = running_mean;
      *out_running_var = running_var;
    }
  }
  return 0;
}
