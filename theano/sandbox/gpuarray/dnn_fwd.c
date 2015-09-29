#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(PyGpuArrayObject *input, PyGpuArrayObject *kerns,
                         PyGpuArrayObject *om,
                         cudnnConvolutionDescriptor_t desc,
                         double alpha, double beta,
                         PyGpuArrayObject **output,
                         PyGpuContextObject *c) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  float af = alpha, bf = beta;
  void *alpha_p;
  void *beta_p;

  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "images and kernel must have the same stack size");
    return 1;
  }

  if (c_set_tensorNd(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  switch (input->ga.typecode) {
  case GA_DOUBLE:
    alpha_p = (void *)&alpha;
    beta_p = (void *)&beta;
    break;
  case GA_FLOAT:
  case GA_HALF:
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
                         om->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (beta != 0.0 && pygpu_move(*output, om))
    return 1;
#endif

  if (c_set_tensorNd(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  cudnnConvolutionFwdAlgo_t algo = CONV_ALGO;

  cuda_enter(c->ctx);
#ifdef CHOOSE_ALGO
  /* Static variables are only initialized once so this will not
   * reset the previous algo every time */
  static int reuse_algo = 0;
  static cudnnConvolutionFwdAlgo_t prev_algo = CONV_ALGO;

#ifndef CHOOSE_ONCE
  static size_t prev_img_dims[5] = {0};
  static size_t prev_kern_dims[5] = {0};

  reuse_algo = 1;
  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
    reuse_algo = (reuse_algo &&
                  PyGpuArray_DIM(input, i) == prev_img_dims[i]);
    reuse_algo = (reuse_algo &&
                  PyGpuArray_DIM(kerns, i) == prev_kern_dims[i]);
  }
#endif

  if (!reuse_algo) {
#ifdef CHOOSE_TIME
    int count;
    cudnnConvolutionFwdAlgoPerf_t choice;
    err = cudnnFindConvolutionForwardAlgorithm(
      APPLY_SPECIFIC(_handle), APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
      desc, APPLY_SPECIFIC(output), 1, &count, &choice);

    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error selecting convolution algo: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }
    algo = choice.algo;
#else
    size_t free = 0, total = 0;
    cudaError_t err2 = cudaMemGetInfo(&free, &total);
    if (err2 != cudaSuccess) {
      PyErr_Format(PyExc_RuntimeError, "Error when trying to find the "
                   "memory information on the GPU: %s\n",
                   cudaGetErrorString(err2));
      cuda_exit(c->ctx);
      return 1;
    }

    err = cudnnGetConvolutionForwardAlgorithm(
      APPLY_SPECIFIC(_handle), APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
      desc, APPLY_SPECIFIC(output),
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, free, &algo);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error selecting convolution algo: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }
#endif
    prev_algo = algo;
  } else {
    algo = prev_algo;
  }

#ifdef CHOOSE_ONCE
  reuse_algo = 1;
#else
  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
    prev_img_dims[i] = PyGpuArray_DIM(input, i);
    prev_kern_dims[i] = PyGpuArray_DIM(kerns, i);
  }
#endif

#endif

  /* These two algos are not supported for 3d conv */
  if (PyGpuArray_NDIM(input) == 5 &&
      (algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||
       algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM))
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

#if CUDNN_VERSION > 3000
  if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT) {
    int nd;
    int pad[2];
    int stride[2];
    int upscale[2];
    cudnnConvolutionMode_t mode;
    err = cudnnGetConvolutionNdDescriptor(desc, 2, &nd, pad, stride,
                                          upscale, &mode);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting convolution properties: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    if (stride[0] != 1 || stride[1] != 1 ||
        PyGpuArray_DIM(input, 0) > 1024 || PyGpuArray_DIM(input, 1) > 1024 ||
        (PyGpuArray_DIM(kerns, 0) == 1 && PyGpuArray_DIM(kerns, 1) == 1)) {
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
  }
#endif

#if CUDNN_VERSION < 3000
  /* cuDNN before v3 does not support kernels larger than input even
   * if appropriate padding is selected. */
  for (unsigned int i = 2; i < PyGpuArray_NDIM(input); i++) {
    if (PyGpuArray_DIM(kerns, i) > PyGpuArray_DIM(input, i)) {
      PyErr_SetString(PyExc_RuntimeError, "the current version "
                      "of CuDNN does not support kernels larger than the "
                      "inputs in any spatial dimension, even if the inputs "
                      "are padded such that the padded inputs are larger "
                      "than the kernels. Update your installation of CuDNN "
                      "to V3 or more recent to solve the issue.");
      cuda_exit(c->ctx);
      return 1;
    }
  }
#endif

  {
    size_t worksize;
    gpudata *workspace;
    err = cudnnGetConvolutionForwardWorkspaceSize(APPLY_SPECIFIC(_handle),
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  algo,
                                                  &worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting worksize: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    /*
     * This is less than ideal since we need to free it after (which
     * introduces a synchronization point. But we don't have a module
     * to place a nice get_work_mem() function in.
     */
    if (worksize != 0) {
      workspace = c->ops->buffer_alloc(c->ctx, worksize, NULL, 0, NULL);
      if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate working memory");
        cuda_exit(c->ctx);
        return 1;
      }
    }

    err = cudnnConvolutionForward(
      APPLY_SPECIFIC(_handle),
      alpha_p,
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
      desc, algo,
      worksize == 0 ? NULL : *(void **)workspace, worksize,
      beta_p,
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output));

    if (worksize != 0)
      c->ops->buffer_release(workspace);
  }
  cuda_exit(c->ctx);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
