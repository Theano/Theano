#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(CudaNdarray *input, CudaNdarray *kerns,
                         CudaNdarray *om, cudnnConvolutionDescriptor_t desc,
                         float alpha, float beta, int nb_dim, CudaNdarray **output) {

  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  if (CudaNdarray_HOST_DIMS(input)[1] != CudaNdarray_HOST_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensorNd(input, nb_dim, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filterNd(kerns, nb_dim, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*output);
  *output = om;
  Py_INCREF(*output);
#else
  if (CudaNdarray_prep_output(output, nb_dim, CudaNdarray_HOST_DIMS(om)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*output, om))
    return 1;
#endif

   if (c_set_tensorNd(*output, nb_dim, APPLY_SPECIFIC(output)) == -1)
     return 1;

  {
    size_t worksize;
    void *workspace;
    cudnnConvolutionFwdAlgo_t chosen_algo;


    if (CHOOSE_ALGO)
    {
      // Check if the input and the kernels have the same shape as they have
      // last time the apply node was executed
      bool same_shapes = true;
      for (int i = 0; (i < nb_dim) && same_shapes; i++)
      {
          same_shapes &= (CudaNdarray_HOST_DIMS(input)[i] ==
                          APPLY_SPECIFIC(previous_input_shape)[i]);
          same_shapes &= (CudaNdarray_HOST_DIMS(kerns)[i] ==
                          APPLY_SPECIFIC(previous_kerns_shape)[i]);
      }

      if (!same_shapes)
      {
        // The shape of the inputs and/or the kernels is different from the
        // last execution. Use the current shapes to infer the implementation
        // to use from now on.

        // Get the amount of available memory
        size_t free = 0, total = 0;
        cudaError_t err2 = cudaMemGetInfo(&free, &total);
        if (err2 != cudaSuccess){
          cudaGetLastError();
          fprintf(stderr,
                  "Error when trying to find the memory information"
                  " on the GPU: %s\n", cudaGetErrorString(err2));
          return 1;
        }


        // Obtain a convolution algorithm appropriate for the input and kernel
        // shapes. Either by choosing one according to heuristics or by making
        // CuDNN time every implementation and choose the best one.
        if (CHOOSE_ALGO_TIME)
        {
          // Time the different implementations to choose the best one
          int requestedCount = 1;
          int count;
          cudnnConvolutionFwdAlgoPerf_t choosen_algo_perf;
          err = cudnnFindConvolutionForwardAlgorithm(_handle,
                                                     APPLY_SPECIFIC(input),
                                                     APPLY_SPECIFIC(kerns),
                                                     desc,
                                                     APPLY_SPECIFIC(output),
                                                     requestedCount,
                                                     &count,
                                                     &choosen_algo_perf);
          if (err != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuDnnConv: error selecting convolution algo: %s",
                         cudnnGetErrorString(err));
            return 1;
          }

          chosen_algo = choosen_algo_perf.algo;
        }
        else
        {
          // Use heuristics to choose the implementation
          err = cudnnGetConvolutionForwardAlgorithm(_handle,
                                                    APPLY_SPECIFIC(input),
                                                    APPLY_SPECIFIC(kerns),
                                                    desc,
                                                    APPLY_SPECIFIC(output),
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                    free,
                                                    &chosen_algo);

          if (err != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuDnnConv: error selecting convolution algo: %s",
                         cudnnGetErrorString(err));
            return 1;
          }
        }

        // Store the shapes of the inputs and kernels as well as the chosen
        // algorithm for future use.
        APPLY_SPECIFIC(previous_algo) = chosen_algo;
        for (int i = 0; i < nb_dim; i++)
        {
            APPLY_SPECIFIC(previous_input_shape)[i] =
                                            CudaNdarray_HOST_DIMS(input)[i];
            APPLY_SPECIFIC(previous_kerns_shape)[i] =
                                            CudaNdarray_HOST_DIMS(kerns)[i];
        }
      }
      else
      {
          // The shapes of the inputs and the kernels are the same as for the
          // last execution. The convolution algorithm used last time can also
          // be used here
          chosen_algo = APPLY_SPECIFIC(previous_algo);
      }
    }
    else
    {
      chosen_algo = CONV_ALGO;
    }

    // The FFT implementation does not support strides, 1x1 filters or
    // inputs with a spatial dimension larger than 1024.
    // If the chosen implementation is FFT, validate that it can be used
    // on the current data and default on a safe implementation if it
    // can't.
    // Following code is 2d-specific, but it is fine as ftt is define only for 2d-filters
    if (chosen_algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT && nb_dim == 4)
    {

      // Extract the properties of the convolution descriptor
      int pad_h, pad_w, stride_v, stride_h, upscale_x, upscale_y;
      cudnnConvolutionMode_t mode;
      err = cudnnGetConvolution2dDescriptor(desc, &pad_h, &pad_w,
                                            &stride_v, &stride_h,
                                            &upscale_x, &upscale_y,
                                            &mode);

      if (err != CUDNN_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuDnnConv: error getting convolution properties: %s",
                     cudnnGetErrorString(err));
        return 1;
      }

      // Extract the spatial size of the filters
      int filter_h = CudaNdarray_HOST_DIMS(kerns)[3];
      int filter_w = CudaNdarray_HOST_DIMS(kerns)[4];

      // Extract the spatial size of the input
      int input_h = CudaNdarray_HOST_DIMS(input)[3];
      int input_w = CudaNdarray_HOST_DIMS(input)[4];

      // Ensure that the selected implementation supports the requested
      // convolution. Fall back to a safe implementation otherwise.
      if (stride_v != 1 || stride_h != 1 || input_h > 1024 ||
          input_w > 1024 || (filter_h == 1 && filter_w == 1))
      {
        chosen_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      }
    }

    err = cudnnGetConvolutionForwardWorkspaceSize(_handle,
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  chosen_algo,
                                                  &worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDnnConv: error getting worksize: %s",
                   cudnnGetErrorString(err));
      return 1;
    }
    workspace = get_work_mem(worksize);
    if (workspace == NULL && worksize != 0)
      return 1;

    err = cudnnConvolutionForward(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
      desc,
      chosen_algo,
      workspace, worksize,
      (void *)&beta,
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(*output));
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConv: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
