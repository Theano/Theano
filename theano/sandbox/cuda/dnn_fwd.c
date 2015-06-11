#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(CudaNdarray *input, CudaNdarray *kerns,
                         CudaNdarray *om, cudnnConvolutionDescriptor_t desc,
                         float alpha, float beta, CudaNdarray **output) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  if (CudaNdarray_HOST_DIMS(input)[1] != CudaNdarray_HOST_DIMS(kerns)[1]) {
    PyErr_SetString(PyExc_ValueError,
		    "GpuDnnConv images and kernel must have the same stack size\n");
    return 1;
  }

  if (c_set_tensor4d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

#ifdef CONV_INPLACE
  Py_XDECREF(*output);
  *output = om;
  Py_INCREF(*output);
#else
  if (CudaNdarray_prep_output(output, 4, CudaNdarray_HOST_DIMS(om)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*output, om))
    return 1;
#endif

  if (c_set_tensor4d(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  {
    size_t worksize;
    void *workspace;
    cudnnConvolutionFwdAlgo_t chosen_algo;

    if (CHOOSE_ALGO){

      // Check if the input and the kernels have the same shape as they have
      // last time the apply node was executed
      bool same_shapes = true;
      for (int i = 0; (i < 4) && same_shapes; i++)
      {
          same_shapes &= (CudaNdarray_HOST_DIMS(input)[i] !=
                          APPLY_SPECIFIC(previous_input_shape)[i]);
          same_shapes &= (CudaNdarray_HOST_DIMS(kerns)[i] !=
                          APPLY_SPECIFIC(previous_kerns_shape)[i]);
      }

      if (same_shapes)
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
        }

        // Obtain a convolution algorithm appropriate for the input and kernel
        // shapes
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

        // Store the shapes of the inputs and kernels as well as the chosen
        // algorithm for future use.
        APPLY_SPECIFIC(previous_algo) = chosen_algo;
        for (int i = 0; i < 4; i++)
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
