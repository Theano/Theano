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

  int nb_dim = CudaNdarray_NDIM(output);

#ifdef CONV_INPLACE
  Py_XDECREF(*kerns);
  *kerns = km;
  Py_INCREF(*kerns);
#else
  if (CudaNdarray_prep_output(kerns, nb_dim, CudaNdarray_HOST_DIMS(km)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*kerns, km))
    return 1;
#endif

  if (CudaNdarray_DIMS(input)[0] == 0 || CudaNdarray_DIMS(km)[0] == 0 || CudaNdarray_DIMS(km)[1] == 0) {
    cudaError_t err2 = cudaMemset((*kerns)->devdata, 0,
                                  CudaNdarray_SIZE(*kerns) * sizeof(real));
    if (err2 != cudaSuccess) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDnnConv grad wrt. weights could not fill the output with zeros: %s",
                   cudaGetErrorString(err2));
      return 1;
    }
    return 0;
  }

  if (c_set_tensorNd(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_tensorNd(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  if (c_set_filterNd(*kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

  int expected_output_dims[5] = {0};
  err = cudnnGetConvolutionNdForwardOutputDim(desc, APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
                                              nb_dim, expected_output_dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error computing convolution output dim: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  if (nb_dim == 4) {
    if ((CudaNdarray_HOST_DIMS(output)[0] != expected_output_dims[0]) ||
        (CudaNdarray_HOST_DIMS(output)[1] != expected_output_dims[1]) ||
        (CudaNdarray_HOST_DIMS(output)[2] != expected_output_dims[2]) ||
        (CudaNdarray_HOST_DIMS(output)[3] != expected_output_dims[3])) {
      PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%dx%ld"
                                     " but received gradient with shape %ldx%ldx%dx%ld",
                   (long int)expected_output_dims[0], (long int)expected_output_dims[1],
                   (long int)expected_output_dims[2], (long int)expected_output_dims[3],
                   (long int)CudaNdarray_HOST_DIMS(output)[0], (long int)CudaNdarray_HOST_DIMS(output)[1],
                   (long int)CudaNdarray_HOST_DIMS(output)[2], (long int)CudaNdarray_HOST_DIMS(output)[3]);
      return 1;
    }
  } else if (nb_dim == 5) {
    if ((CudaNdarray_HOST_DIMS(output)[0] != expected_output_dims[0]) ||
        (CudaNdarray_HOST_DIMS(output)[1] != expected_output_dims[1]) ||
        (CudaNdarray_HOST_DIMS(output)[2] != expected_output_dims[2]) ||
        (CudaNdarray_HOST_DIMS(output)[3] != expected_output_dims[3]) ||
        (CudaNdarray_HOST_DIMS(output)[4] != expected_output_dims[4])) {
      PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%ldx%ldx%ld"
                                     " but received gradient with shape %ldx%ldx%ldx%ldx%ld",
                   (long int)expected_output_dims[0], (long int)expected_output_dims[1],
                   (long int)expected_output_dims[2], (long int)expected_output_dims[3],
                   (long int)expected_output_dims[4],
                   (long int)CudaNdarray_HOST_DIMS(output)[0], (long int)CudaNdarray_HOST_DIMS(output)[1],
                   (long int)CudaNdarray_HOST_DIMS(output)[2], (long int)CudaNdarray_HOST_DIMS(output)[3],
                   (long int)CudaNdarray_HOST_DIMS(output)[4]);
      return 1;
    }
  }

  {
    size_t worksize;
    void *workspace;
    cudnnConvolutionBwdFilterAlgo_t chosen_algo;

    if (CHOOSE_ALGO)
    {

      // A new convolution implementation should be selected, based either on
      // timing or heuristics, if in one of the two following cases :
      // - The implementation should only be chosen during the first execution
      //   of an apply node and this is the first execution of the apply node.
      // - The implementation should be chosen as often as necessary and the
      //   shapes of the inputs differ from the last time an implementation
      //   was chosen.
      bool reuse_previous_algo;
      if (CHOOSE_ALGO_ONCE)
      {
        // Only choose a new implementation of none has been chosen before.
        reuse_previous_algo = APPLY_SPECIFIC(previous_algo_set);
      }
      else
      {
        // Reuse the previous implementation if the the kernels and the outputs
        // have the same shapes as they had when the previous implementation
        // was selected
        bool same_shapes = true;
        for (int i = 0; (i < nb_dim) && same_shapes; i++)
        {
            same_shapes &= (CudaNdarray_HOST_DIMS(input)[i] ==
                            APPLY_SPECIFIC(previous_input_shape)[i]);
            same_shapes &= (CudaNdarray_HOST_DIMS(output)[i] ==
                            APPLY_SPECIFIC(previous_output_shape)[i]);
        }
        reuse_previous_algo = same_shapes;
      }

      // If the previously choosen implementation can't be reused, select a
      // new one based on the shapes of the current inputs
      if (!reuse_previous_algo)
      {
        // Obtain a convolution algorithm appropriate for the input and output
        // shapes. Either by choosing one according to heuristics or by making
        // cuDNN time every implementation and choose the best one.
        if (CHOOSE_ALGO_TIME)
        {
          // Time the different implementations to choose the best one
          int requestedCount = 1;
          int count;
          cudnnConvolutionBwdFilterAlgoPerf_t choosen_algo_perf;
          err = cudnnFindConvolutionBackwardFilterAlgorithm(_handle,
                                                            APPLY_SPECIFIC(input),
                                                            APPLY_SPECIFIC(output),
                                                            desc,
                                                            APPLY_SPECIFIC(kerns),
                                                            requestedCount,
                                                            &count,
                                                            &choosen_algo_perf);
          if (err != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuDnnConvGradW: error selecting convolution algo: "
                         "%s", cudnnGetErrorString(err));
            return 1;
          }

          chosen_algo = choosen_algo_perf.algo;
        }
        else
        {
          // Choose the convolution implementation using heuristics based on the
          // shapes of the inputs and the amount of memory available.

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
                                                           CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                                           free,
                                                           &chosen_algo);

          if (err != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuDnnConvGradW: error selecting convolution algo: %s",
                         cudnnGetErrorString(err));
            return 1;
          }
        }

        // Store the shapes of the inputs and kernels as well as the chosen
        // algorithm for future use.
        APPLY_SPECIFIC(previous_bwd_f_algo) = chosen_algo;
        APPLY_SPECIFIC(previous_algo_set) = true;
        for (int i = 0; i < nb_dim; i++)
        {
            APPLY_SPECIFIC(previous_input_shape)[i] =
                                            CudaNdarray_HOST_DIMS(input)[i];
            APPLY_SPECIFIC(previous_output_shape)[i] =
                                            CudaNdarray_HOST_DIMS(output)[i];
        }

      }
      else
      {
        // Reuse the previously chosen convlution implementation
        chosen_algo = APPLY_SPECIFIC(previous_bwd_f_algo);
      }
    }
    else
    {
        chosen_algo = CONV_ALGO;
    }

    if (0){
      char * a;
      switch(chosen_algo){
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
	a = "algo 0 (0)";
	break;
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
	a = "algo 1 (1)";
	break;
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
	a = "fft (2)";
	break;
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
	a = "algo 3 (3)";
	break;
      }
      printf("GpuDNNConvGW: algo %s\n", a);
    }

    // The FFT implementation (only in v3 and onward) does not support strides,
    // 1x1 filters or inputs with a spatial dimension larger than 1024.
    // If the chosen implementation is FFT, validate that it can be used
    // on the current data and default on a safe implementation if it
    // can't.
    if (chosen_algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT && nb_dim == 4)
    {

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
                     "GpuDnnConvGradW: error getting convolution properties: %s",
                     cudnnGetErrorString(err));
        return 1;
      }

      // Extract the spatial size of the filters
      int filter_h = CudaNdarray_HOST_DIMS(*kerns)[2];
      int filter_w = CudaNdarray_HOST_DIMS(*kerns)[3];

      // Extract the spatial size of the input
      int input_h = CudaNdarray_HOST_DIMS(input)[2];
      int input_w = CudaNdarray_HOST_DIMS(input)[3];

      // Ensure that the selected implementation supports the requested
      // convolution. Fall back to a safe implementation otherwise.
      if (stride[0] != 1 || stride[1] != 1 || input_h > 1024 ||
          input_w > 1024 || (filter_h == 1 && filter_w == 1))
      {
        chosen_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
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
    err = cudnnConvolutionBackwardFilter(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
      desc,
      chosen_algo,
      workspace, worksize,
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
