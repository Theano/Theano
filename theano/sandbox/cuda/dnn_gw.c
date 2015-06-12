#section support_code_struct

int
APPLY_SPECIFIC(conv_gw)(CudaNdarray *input, CudaNdarray *output,
                        CudaNdarray *km, cudnnConvolutionDescriptor_t desc,
                        float alpha, float beta, CudaNdarray **kerns) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

  if (CudaNdarray_HOST_DIMS(input)[1] != CudaNdarray_HOST_DIMS(km)[1]) {
    PyErr_SetString(PyExc_ValueError,
                   "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_tensor4d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*kerns);
  *kerns = km;
  Py_INCREF(*kerns);
#else
  if (CudaNdarray_prep_output(kerns, 4, CudaNdarray_HOST_DIMS(km)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*kerns, km))
    return 1;
#endif

  if (c_set_filter(*kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  {
    size_t worksize;
    void *workspace;
    cudnnConvolutionBwdFilterAlgo_t chosen_algo;

    if (CHOOSE_ALGO)
    {
      // Check if the input and the output have the same shape as they have
      // last time the apply node was executed
      bool same_shapes = true;
      for (int i = 0; (i < 4) && same_shapes; i++)
      {
          same_shapes &= (CudaNdarray_HOST_DIMS(input)[i] !=
                          APPLY_SPECIFIC(previous_input_shape)[i]);
          same_shapes &= (CudaNdarray_HOST_DIMS(output)[i] !=
                          APPLY_SPECIFIC(previous_output_shape)[i]);
      }

      if (!same_shapes)
      {
        // The shape of the inputs and/or the output is different from the
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

        // Use heuristics to choose the implementation
        err = cudnnGetConvolutionBackwardFilterAlgorithm(_handle,
                                                         APPLY_SPECIFIC(input),
                                                         APPLY_SPECIFIC(output),
                                                         desc,
                                                         APPLY_SPECIFIC(kerns),
                                                         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                         free,
                                                         &chosen_algo);

        if (err != CUDNN_STATUS_SUCCESS) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuDnnConvGradW: error selecting convolution algo: %s",
                       cudnnGetErrorString(err));
          return 1;
        }

        // Store the shapes of the inputs and kernels as well as the chosen
        // algorithm for future use.
        APPLY_SPECIFIC(previous_bwd_f_algo) = chosen_algo;
        for (int i = 0; i < 4; i++)
        {
            APPLY_SPECIFIC(previous_input_shape)[i] =
                                            CudaNdarray_HOST_DIMS(input)[i];
            APPLY_SPECIFIC(previous_output_shape)[i] =
                                            CudaNdarray_HOST_DIMS(output)[i];
        }

      }
      else
      {
          chosen_algo = CONV_ALGO;
      }
    }

    // Infer required workspace size from the chosen implementation
    err = cudnnGetConvolutionBackwardFilterWorkspaceSize(_handle,
                                                         APPLY_SPECIFIC(input),
                                                         APPLY_SPECIFIC(output),
                                                         desc,
                                                         APPLY_SPECIFIC(kerns),
                                                         chosen_algo,
                                                         &worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDnnConvGradW: error getting worksize: %s",
                   cudnnGetErrorString(err));
      return 1;
    }

    // Allocate workspace for the convolution
    workspace = get_work_mem(worksize);
    if (workspace == NULL && worksize != 0)
      return 1;

    // Perform the convolution
    err = cudnnConvolutionBackwardFilter_v3(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
      desc,
      chosen_algo,
      &workspace, worksize,
      (void *)&beta,
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(*kerns));

  }

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnnConvGradW: error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
