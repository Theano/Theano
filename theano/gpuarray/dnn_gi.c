#section init_code_struct

#ifdef CHOOSE_ALGO
reuse_algo = 0;
prev_algo = CONV_ALGO;
#ifndef CHOOSE_ONCE
memset(prev_kern_dims, 0, sizeof(prev_kern_dims));
memset(prev_top_dims, 0, sizeof(prev_top_dims));
#endif
#endif

#section support_code_struct

#ifdef CHOOSE_ALGO
int reuse_algo = 0;
cudnnConvolutionBwdDataAlgo_t prev_algo = CONV_ALGO;
#ifndef CHOOSE_ONCE
size_t prev_kern_dims[5] = {0};
size_t prev_top_dims[5] = {0};
#endif
#endif

int
APPLY_SPECIFIC(conv_gi)(PyGpuArrayObject *kerns, PyGpuArrayObject *output,
                        PyGpuArrayObject *im,
                        cudnnConvolutionDescriptor_t desc,
                        double alpha, double beta, PyGpuArrayObject **input,
                        cudnnHandle_t _handle) {
  PyGpuContextObject *c = kerns->context;
  void *alpha_p;
  void *beta_p;
  float af = alpha, bf = beta;
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (PyGpuArray_DIMS(im)[1] != PyGpuArray_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError, "images and kernel must have the same "
                    "stack size");
    return 1;
  }

  switch (im->ga.typecode) {
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
  Py_XDECREF(*input);
  *input = im;
  Py_INCREF(*input);
#else
  if (theano_prep_output(input, PyGpuArray_NDIM(im), PyGpuArray_DIMS(im),
                         im->ga.typecode, GA_C_ORDER, c) != 0)
    return 1;
  if (beta != 0.0 && pygpu_move(*input, im))
    return 1;
#endif

  if (PyGpuArray_DIMS(im)[0] == 0 || PyGpuArray_DIMS(kerns)[0] == 0 || PyGpuArray_DIMS(kerns)[1] == 0) {
    int err2 = GpuArray_memset(&(*input)->ga, 0);
    if (err2 != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuDnnConv grad wrt. inputs could not fill the output with zeros: %d", err2);
        return 1;
    }
    return 0;
  }

  if (c_set_tensorNd(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;
  if (c_set_tensorNd(*input, APPLY_SPECIFIC(input)) == -1)
    return 1;

  cudnnConvolutionBwdDataAlgo_t algo = CONV_ALGO;

  cuda_enter(c->ctx);

  int expected_output_dims[5] = {0};
  err = cudnnGetConvolutionNdForwardOutputDim(desc, APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
                                              PyGpuArray_NDIM(im), expected_output_dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error computing convolution output dim: %s",
                 cudnnGetErrorString(err));
    cuda_exit(c->ctx);
    return 1;
  }
  if (PyGpuArray_NDIM(im) == 4) {
    if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
        (PyGpuArray_DIMS(output)[1] != expected_output_dims[1]) ||
        (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
        (PyGpuArray_DIMS(output)[3] != expected_output_dims[3])) {
      PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%ldx%ld"
                                     " but received gradient with shape %ldx%ldx%ldx%ld",
                   expected_output_dims[0], expected_output_dims[1],
                   expected_output_dims[2], expected_output_dims[3],
                   PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                   PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3]);
      cuda_exit(c->ctx);
      return 1;
    }
  } else if (PyGpuArray_NDIM(im) == 5) {
    if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
        (PyGpuArray_DIMS(output)[1] != expected_output_dims[1]) ||
        (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
        (PyGpuArray_DIMS(output)[3] != expected_output_dims[3]) ||
        (PyGpuArray_DIMS(output)[4] != expected_output_dims[4])) {
      PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%ldx%ldx%ld"
                                     " but received gradient with shape %ldx%ldx%ldx%ldx%ld",
                   expected_output_dims[0], expected_output_dims[1],
                   expected_output_dims[2], expected_output_dims[3],
                   expected_output_dims[4],
                   PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                   PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3],
                   PyGpuArray_DIMS(output)[4]);
      cuda_exit(c->ctx);
      return 1;
    }
  }

#ifdef CHOOSE_ALGO
#ifndef CHOOSE_ONCE
  reuse_algo = 1;
  for (unsigned int i = 0; i < PyGpuArray_NDIM(kerns); i++) {
    reuse_algo = (reuse_algo &&
                  PyGpuArray_DIM(kerns, i) == prev_kern_dims[i]);
    reuse_algo = (reuse_algo &&
                  PyGpuArray_DIM(output, i) == prev_top_dims[i]);
  }
#endif

  if (!reuse_algo) {
    size_t free;
    int err2 = gpucontext_property(c->ctx, GA_CTX_PROP_LARGEST_MEMBLOCK, &free);

    if (err2 != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "Error when trying to find the "
                   "memory information on the GPU");
      cuda_exit(c->ctx);
      return 1;
    }

    // Guess 4Mb if the info is not available
    if (free == 0) free = 4 * 1024 * 1024;

#ifdef CHOOSE_TIME
    int count;
    cudnnConvolutionBwdDataAlgoPerf_t choice;
    gpudata *tmpmem;

    tmpmem = gpudata_alloc(c->ctx, free, NULL, 0, NULL);
    if (tmpmem == NULL) {
      PyErr_SetString(PyExc_MemoryError, "Could not allocate working GPU memory");
      return -1;
    }

    err = cudnnFindConvolutionBackwardDataAlgorithmEx(
      _handle, APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output), desc,
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(*input),
      1, &count, &choice, *(void **)tmpmem, free);
    gpudata_release(tmpmem);

    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "error selecting convolution algo: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    algo = choice.algo;
#else
    err = cudnnGetConvolutionBackwardDataAlgorithm(
      _handle, APPLY_SPECIFIC(kerns), APPLY_SPECIFIC(output),
      desc, APPLY_SPECIFIC(input),
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, free, &algo);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "error selecting convolution algo: %s",
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
  for (unsigned int i = 0; i < PyGpuArray_NDIM(kerns); i++) {
    prev_kern_dims[i] = PyGpuArray_DIM(kerns, i);
    prev_top_dims[i] = PyGpuArray_DIM(output, i);
  }
#endif

#endif

  // The FFT implementation does not support strides, 1x1 filters or inputs
  // with a spatial dimension larger than 1024. The tiled-FFT implementation
  // does not support strides.
  // If the chosen implementation is FFT or tiled-FFT, validate that it can
  // be used on the current data and default to a safe implementation if it
  // can't.
  // The following code is 2d-specific but it is fine as FFT and tiled-FFT are
  // defined only for 2d filters
  if ((algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
       algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT) && PyGpuArray_NDIM(kerns) == 4) {

    // Extract the properties of the convolution descriptor
    int nd;
    int pad[2];
    int stride[2];
    int upscale[2];
    cudnnConvolutionMode_t mode;
    cudnnDataType_t data_type;
    err = cudnnGetConvolutionNdDescriptor(desc, 2, &nd, pad, stride,
                                             upscale, &mode, &data_type);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting convolution properties: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    if (algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
    {
      if (stride[0] != 1 || stride[1] != 1 ||
          PyGpuArray_DIM(*input, 2) > 1024 || PyGpuArray_DIM(*input, 3) > 1024 ||
          (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1))
      {
        algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }
    }
    else
    {
      // algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
      if (stride[0] != 1 || stride[1] != 1)
      {
        algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }
    }
  }

  size_t worksize;
  gpudata *workspace;

  err = cudnnGetConvolutionBackwardDataWorkspaceSize(
    _handle, APPLY_SPECIFIC(kerns), APPLY_SPECIFIC(output), desc,
    APPLY_SPECIFIC(input), algo, &worksize);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error getting worksize: %s",
                 cudnnGetErrorString(err));
    cuda_exit(c->ctx);
    return 1;
  }

  if (worksize != 0) {
    workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
    if (workspace == NULL) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Could not allocate working memory");
      cuda_exit(c->ctx);
      return 1;
    }
  }

  cuda_wait(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
  cuda_wait(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
  cuda_wait((*input)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

  err = cudnnConvolutionBackwardData(
    _handle,
    alpha_p,
    APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
    APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output),
    desc, algo, worksize == 0 ? NULL : *(void **)workspace, worksize,
    beta_p,
    APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(*input));

  if (worksize != 0)
    gpudata_release(workspace);

  cuda_record(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
  cuda_record(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
  cuda_record((*input)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

  cuda_exit(c->ctx);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
