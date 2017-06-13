#section init_code_struct

if (PARAMS->choose_algo) {
  reuse_algo = 0;
  prev_algo = PARAMS->conv_algo;
  if (!PARAMS->choose_once) {
    memset(prev_img_dims, 0, sizeof(prev_img_dims));
    memset(prev_kern_dims, 0, sizeof(prev_kern_dims));
  }
}

#section support_code_struct

int reuse_algo;
cudnnConvolutionFwdAlgo_t prev_algo;
size_t prev_img_dims[5];
size_t prev_kern_dims[5];

int
APPLY_SPECIFIC(conv_fwd)(PyGpuArrayObject *input, PyGpuArrayObject *kerns,
                         PyGpuArrayObject *om,
                         cudnnConvolutionDescriptor_t desc,
                         double alpha, double beta,
                         PyGpuArrayObject **output,
                         PARAMS_TYPE* params) {
  PyGpuContextObject *c = input->context;
  void *alpha_p;
  void *beta_p;
  float af = alpha, bf = beta;
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "images and kernel must have the same stack size");
    return 1;
  }

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

  if (params->inplace) {
    Py_XDECREF(*output);
    *output = om;
    Py_INCREF(*output);
  } else {
    if (theano_prep_output(output, PyGpuArray_NDIM(om), PyGpuArray_DIMS(om),
                           om->ga.typecode, GA_C_ORDER, c) != 0)
      return 1;
    if (beta != 0.0 && pygpu_move(*output, om))
      return 1;
  }

  if (PyGpuArray_DIMS(input)[0] == 0 || PyGpuArray_DIMS(kerns)[0] == 0 || PyGpuArray_DIMS(kerns)[1] == 0) {
    int err2 = GpuArray_memset(&(*output)->ga, 0);
    if (err2 != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuDnnConv could not fill the output with zeros: %d", err2);
        return 1;
    }
    return 0;
  }

  if (c_set_tensorNd(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;
  if (c_set_tensorNd(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  cudnnConvolutionFwdAlgo_t algo = params->conv_algo;

  cuda_enter(c->ctx);

  if (params->choose_algo) {
    if (params->choose_once) {
      reuse_algo = 1;
      for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
        reuse_algo = (reuse_algo &&
                      PyGpuArray_DIM(input, i) == prev_img_dims[i]);
        reuse_algo = (reuse_algo &&
                      PyGpuArray_DIM(kerns, i) == prev_kern_dims[i]);
      }
    }

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

      if (params->choose_time) {
        int count;
        cudnnConvolutionFwdAlgoPerf_t choice;
        gpudata *tmpmem;

        tmpmem = gpudata_alloc(c->ctx, free, NULL, 0, NULL);
        if (tmpmem == NULL) {
          PyErr_SetString(PyExc_MemoryError, "Could not allocate working GPU memory");
          return -1;
        }
        // We don't sync the buffer as we don't care about the values.
        err = cudnnFindConvolutionForwardAlgorithmEx(
          params->handle, APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
          APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
          desc, APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output),
          1, &count, &choice, *(void **)tmpmem,
          free);
        gpudata_release(tmpmem);

        if (err != CUDNN_STATUS_SUCCESS) {
          PyErr_Format(PyExc_RuntimeError,
                       "error selecting convolution algo: %s",
                       cudnnGetErrorString(err));
          cuda_exit(c->ctx);
          return 1;
        }
        algo = choice.algo;
      } else {
        err = cudnnGetConvolutionForwardAlgorithm(
          params->handle, APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
          desc, APPLY_SPECIFIC(output),
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, free, &algo);
          if (err != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "error selecting convolution algo: %s",
                         cudnnGetErrorString(err));
            cuda_exit(c->ctx);
            return 1;
          }
      }
      prev_algo = algo;
    } else {
      algo = prev_algo;
    }

    if (params->choose_once) {
      reuse_algo = 1;
    } else {
      for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
        prev_img_dims[i] = PyGpuArray_DIM(input, i);
        prev_kern_dims[i] = PyGpuArray_DIM(kerns, i);
      }
    }
  }

  /* These two algos are not supported for 3d conv */
  if (PyGpuArray_NDIM(input) == 5 &&
      (algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||
       algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM))
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // The FFT implementation does not support strides, 1x1 filters or inputs
  // with a spatial dimension larger than 1024. The tiled-FFT implementation
  // does not support strides.
  // If the chosen implementation is FFT or tiled-FFT, validate that it can
  // be used on the current data and default to a safe implementation if it
  // can't.
  // The following code is 2d-specific but it is fine as FFT and tiled-FFT are
  // defined only for 2d filters
  if ((algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
       algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) && PyGpuArray_NDIM(input) == 4) {

    // Extract the properties of the convolution descriptor
    int nd;
    int pad[2];
    int stride[2];
    int dilation[2];
    cudnnConvolutionMode_t mode;
    cudnnDataType_t data_type;
    err = cudnnGetConvolutionNdDescriptor(desc, 2, &nd, pad, stride,
                                             dilation, &mode, &data_type);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error getting convolution properties: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }

    if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT) {
      if (stride[0] != 1 || stride[1] != 1 ||
          PyGpuArray_DIM(input, 2) > 1024 || PyGpuArray_DIM(input, 3) > 1024 ||
          (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1))
      {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      }
    } else {
      // algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
      if (stride[0] != 1 || stride[1] != 1) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      }
    }
  }

  {
    size_t worksize;
    gpudata *workspace;
    err = cudnnGetConvolutionForwardWorkspaceSize(params->handle,
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  algo,
                                                  &worksize);

    if (err == CUDNN_STATUS_NOT_SUPPORTED) {
      // Fallback to none algo if not supported
      // TODO: Print a warning
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

      err = cudnnGetConvolutionForwardWorkspaceSize(params->handle,
                                                    APPLY_SPECIFIC(input),
                                                    APPLY_SPECIFIC(kerns),
                                                    desc,
                                                    APPLY_SPECIFIC(output),
                                                    algo,
                                                    &worksize);
    }

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
      workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
      if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate working memory");
        cuda_exit(c->ctx);
        return 1;
      }
    }

    cuda_wait(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    err = cudnnConvolutionForward(
      params->handle,
      alpha_p,
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
      desc, algo,
      worksize == 0 ? NULL : *(void **)workspace, worksize,
      beta_p,
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output));

    if (worksize != 0)
      gpudata_release(workspace);

    cuda_record(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
  }
  cuda_exit(c->ctx);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
