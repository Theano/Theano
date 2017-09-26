#section init_code_struct
prev_algo.algo = PARAMS->conv_algo;
prev_algo.mathType = CUDNN_DEFAULT_MATH;
reuse_algo = 0;
hash_prefix = std::string("FWD|GPU#");
#ifdef DEBUG_TIMING
total_computation_time = 0;
total_selection_time = 0;
n_computations = 0;
n_selections = 0;
if (PARAMS->choose_algo) {
    if (PARAMS->choose_time) {
        selection_name = "fastest";
    } else {
        selection_name = "best suited";
    }
};
#endif

#section support_code_struct
#line 22 "dnn_fwd.c"
int     reuse_algo;
AlgoRec prev_algo;
std::string hash_prefix;

#define THEANO_DONT_MEMSET_STRUCT

#ifdef DEBUG
char algorithm_name[128];
#endif
#ifdef DEBUG_TIMING
double total_computation_time;
double total_selection_time;
size_t n_computations;
size_t n_selections;
const char* selection_name;
#endif

/** Check given algorithm against inputs and convolution descriptor,
    change algorithm inplace to a fallback algorithm if checkings fail.
    Return 0 on success, non-0 on error. **/
int dnn_conv_fwd_fallback(cudnnConvolutionFwdAlgo_t* _algo,
                          const PyGpuArrayObject* input,
                          const PyGpuArrayObject* kerns,
                          cudnnConvolutionDescriptor_t desc) {
  cudnnConvolutionFwdAlgo_t algo = *_algo;

  /* Only these algos are supported for 3d conv with cuDNN >= V5.1. */
  if (PyGpuArray_NDIM(input) == 5 &&
      !(algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM ||
        algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||
        algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING))
  {
    #ifdef DEBUG
    if (0 != theano_enum_to_string_cudnnConvolutionFwdAlgo_t(algo, algorithm_name))
        return 1;
    fprintf(stderr, "(%s unsupported for 3D: fallback to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)\n", algorithm_name);
    #endif
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }

  // Algo `small` does not work for a batch size > 2^16, with cuDNN >= V5.1.
  // Issue should be resolved for cuDNN > V6.0.
  // NB: In cuDNN V7, issue is resolved for 2D convolutionss only.
  if ((cudnnGetVersion() < 6100 || PyGpuArray_NDIM(input) == 5) &&
      algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM &&
      PyGpuArray_DIM(input, 0) > 65536)
  {
    #ifdef DEBUG
    fprintf(stderr, "(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM "
                    "will fail with batch size > 2^16, fallback to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)\n");
    #endif
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }

  // The FFT implementation does not support strides, 1x1 filters or inputs
  // with a spatial dimension larger than 1024. The tiled-FFT implementation
  // does not support strides.
  // If the chosen implementation is FFT or tiled-FFT, validate that it can
  // be used on the current data and default to a safe implementation if it
  // can't.
  // The following code is 2d-specific but it is fine as FFT and tiled-FFT are
  // defined only for 2d filters
  /* NB:
  TODO: These checkings seems outdated for FFT algorithms with cuDNN >= 5.1.
  New conditions apply and may depend on number of dimensions (2D or 3D)
  e.g. for FFT_TILING.
  TODO: More globally, how to handle CUDNN_STATUS_NOT_SUPPORTED with unsupported algorithms?
  */
  if ((algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
       algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) && PyGpuArray_NDIM(input) == 4) {

    // Extract the properties of the convolution descriptor
    int nd;
    int pad[2];
    int stride[2];
    int dilation[2];
    cudnnConvolutionMode_t mode;
    cudnnDataType_t data_type;
    cudnnStatus_t err = cudnnGetConvolutionNdDescriptor(desc, 2, &nd, pad, stride, dilation, &mode, &data_type);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "error getting convolution properties: %s",
                   cudnnGetErrorString(err));
      return 1;
    }

    if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT) {
      if (stride[0] != 1 || stride[1] != 1 ||
          PyGpuArray_DIM(input, 2) > 1024 || PyGpuArray_DIM(input, 3) > 1024 ||
          (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1))
      {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        #ifdef DEBUG
        fprintf(stderr, "(replacing fwd algo fft with none)\n");
        #endif
      }
    } else {
      // algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
      if (stride[0] != 1 || stride[1] != 1) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        #ifdef DEBUG
        fprintf(stderr, "(replacing fwd algo fft_tiling with none)\n");
        #endif
      }
    }
  }
  *_algo = algo;
  return 0;
}

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
  bool use_cached = 0;
  #ifdef DEBUG
  if (_cppver) fprintf(stderr, "%s\n", _cppver);
  #endif
  #ifdef DEBUG_TIMING
  TheanoTimer timer;
  #endif

  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(kerns)[1] * params->num_groups) {
    PyErr_SetString(PyExc_ValueError,
		    "images and kernel must have the same stack size");
    return 1;
  }
  if ((PyGpuArray_DIMS(kerns)[0] % params->num_groups) != 0) {
    PyErr_SetString(PyExc_ValueError,
		    "Number of filters must be divisible by number of groups");
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

  int groups = c_get_groups_for_conv(desc, params->num_groups);
  if (groups == -1)
    return 1;
  if (c_set_tensor_for_conv(input, APPLY_SPECIFIC(input), groups) == -1)
    return 1;
  if (c_set_filter(kerns, APPLY_SPECIFIC(kerns), groups) == -1)
    return 1;
  if (c_set_tensor_for_conv(*output, APPLY_SPECIFIC(output), groups) == -1)
    return 1;
  size_t input_offset = PyGpuArray_STRIDE(input, 0) / groups;
  size_t kern_offset = PyGpuArray_STRIDE(kerns, 0) * PyGpuArray_DIM(kerns, 0) / groups;
  size_t output_offset = PyGpuArray_STRIDE(*output, 0) / groups;

  cudnnConvolutionFwdAlgo_t algo = params->conv_algo;
  size_t worksize = 0;
  cudnnMathType_t mathtype = CUDNN_DEFAULT_MATH;

  std::string hashkey;


  cuda_enter(c->ctx);

  size_t maxfree = c_get_largest_free_block_size(c);
  if (PyErr_Occurred()) {
    cuda_exit(c->ctx);
    return 1;
  }

  if (params->choose_algo) {

    if (!reuse_algo) {
      char pci_id[16];
      gpucontext_property(c->ctx, GA_CTX_PROP_UNIQUE_ID, pci_id);
      // check out cache
      hashkey = dnn_conv_shape(APPLY_SPECIFIC(input), input, APPLY_SPECIFIC(kerns), kerns, desc, *output, groups);
      if (hashkey.empty()) {
        cuda_exit(c->ctx);
        return 1;
      }
      hashkey = hash_prefix + pci_id + (params->choose_time ? " -t " : " ") + hashkey;
      const AlgoRec* cached = dnn_conv_check_cache(hashkey);
      if (cached) {
        prev_algo = *cached;
        use_cached = 1;
      }
    }

    if (reuse_algo || use_cached) {
      algo = (cudnnConvolutionFwdAlgo_t)prev_algo.algo;
      worksize = prev_algo.wsSize;
      mathtype = prev_algo.mathType;
    }  else {
      if (params->choose_time) {
        int count;
        cudnnConvolutionFwdAlgoPerf_t choice;
        gpudata *tmpmem;

        tmpmem = gpudata_alloc(c->ctx, maxfree, NULL, 0, NULL);
        if (tmpmem == NULL) {
          PyErr_SetString(PyExc_MemoryError, "Could not allocate GPU memory for FindEx");
          cuda_exit(c->ctx);
          return -1;
        }
        // set the 'tensor math ok' flag
        if (input->ga.typecode == GA_HALF)
          c_set_math_type_for_conv(desc, CUDNN_TENSOR_OP_MATH);

        /* cudnnFindConvolutionForwardAlgorithmEx() may write to output.
           We don't want that if output is used in computation (ie. if beta != 0). */
        PyGpuArrayObject* o = *output;
        if (beta != 0) {
            o = pygpu_empty(PyGpuArray_NDIM(*output), PyGpuArray_DIMS(*output), (*output)->ga.typecode, GA_C_ORDER, c, Py_None);
        }

        #ifdef DEBUG_TIMING
        timer.start();
        #endif
        // We don't sync the buffer as we don't care about the values.
        err = cudnnFindConvolutionForwardAlgorithmEx(
          params->handle, APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
          APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
          desc, APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(o),
          1, &count, &choice, *(void **)tmpmem,
          maxfree);
        #ifdef DEBUG_TIMING
        timer.end();
        #endif
        gpudata_release(tmpmem);
        if (beta != 0) {
            Py_XDECREF(o);
        }

        if (err != CUDNN_STATUS_SUCCESS) {
          PyErr_Format(PyExc_RuntimeError,
                       "error selecting convolution algo: %s",
                       cudnnGetErrorString(err));
          cuda_exit(c->ctx);
          return 1;
        }

        #ifdef DEBUG
        if (count == 0) {
            PyErr_SetString(PyExc_RuntimeError, "No best-timed conv fwd algorithm found");
            cuda_exit(c->ctx);
            return 1;
        } else if (choice.status != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "error getting best-timed FWD algo: %s",
                         cudnnGetErrorString(choice.status));
            cuda_exit(c->ctx);
            return 1;
        } // Else, count is necessarly 1 for current implementation.
        #endif

        algo = choice.algo;
        worksize = choice.memory;
#if CUDNN_MAJOR >= 7
        if (input->ga.typecode == GA_HALF)
          mathtype = choice.mathType;
#endif
      } else {
        #ifdef DEBUG_TIMING
        timer.start();
        #endif
        err = cudnnGetConvolutionForwardAlgorithm(
          params->handle, APPLY_SPECIFIC(input), APPLY_SPECIFIC(kerns),
          desc, APPLY_SPECIFIC(output),
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, maxfree, &algo);
        #ifdef DEBUG_TIMING
        timer.end();
        #endif
        if (err != CUDNN_STATUS_SUCCESS) {
          PyErr_Format(PyExc_RuntimeError,
                       "error selecting convolution algo: %s",
                       cudnnGetErrorString(err));
          cuda_exit(c->ctx);
          return 1;
        }
      }
      #ifdef DEBUG_TIMING
      total_selection_time += timer.milliseconds;
      ++n_selections;
      #endif
    }
  }

  if (c_set_math_type_for_conv(desc, mathtype) == -1 ||
      dnn_conv_fwd_fallback(&algo, input, kerns, desc) != 0) {
    cuda_exit(c->ctx);
    return 1;
  }

  // if FindEx was used (choose_time), workspace size is set.
  if (!(reuse_algo || use_cached || params->choose_time))
  {
    err = cudnnGetConvolutionForwardWorkspaceSize(params->handle,
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  algo,
                                                  &worksize);
    if (err == CUDNN_STATUS_NOT_SUPPORTED) {
      // Fallback to none algo if not supported
      #ifdef DEBUG
      if (0 != theano_enum_to_string_cudnnConvolutionFwdAlgo_t(algo, algorithm_name)) {
        cuda_exit(c->ctx);
        return 1;
      }
      fprintf(stderr, "(error getting worksize for %s: failing back to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)\n",
              algorithm_name);
      #endif
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
      PyErr_Format(PyExc_RuntimeError, "error getting worksize: %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }
  }

  if (params->choose_algo) {

#ifdef DEBUG
    if (0 != theano_enum_to_string_cudnnConvolutionFwdAlgo_t(algo, algorithm_name)) {
      cuda_exit(c->ctx);
      return 1;
    }
    fprintf(stderr, "(using %s%s %s%s%s, ws:%ld, hash:%s)\n",
            algorithm_name,
            mathtype == CUDNN_TENSOR_OP_MATH ? "(tensor_op)" : "",
            params->choose_time ? "(timed)": "" ,
            reuse_algo ? "(reused)" : "",
            use_cached ? "(cache)": "",
            worksize,
            hashkey.c_str()
    );
#endif
#ifdef DEBUG_TIMING
    if (!(reuse_algo || use_cached)) {
        // We have selected an algorithm at runtime.
        // `timer` still contains timing about selection step.
        fprintf(stderr, "\t(selected %s fwd algo in %g milliseconds)\n", selection_name, timer.milliseconds);
        if (n_selections > 1) {
            fprintf(stderr, "\t(selected %lu fwd algos in %g milliseconds (average: %g milliseconds per selection))\n",
                    n_selections, total_selection_time, total_selection_time / n_selections);
        }
    }
#endif

    if (!reuse_algo) {
      // save for next time/cache
      prev_algo.algo = algo;
      prev_algo.wsSize = worksize;
      prev_algo.mathType = mathtype;
      // Add to the cache if we choose on shape change, or first time if
      // we choose once.
      if (!use_cached)
        dnn_conv_update_cache(hashkey, prev_algo);

      if (params->choose_once)
        reuse_algo = 1;
    }

  } // params->choose_algo

  {
    gpudata *workspace = 0;
    if (worksize != 0) {
      workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
      if (workspace == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate working memory");
        cuda_exit(c->ctx);
        return 1;
      }
    }

    if (worksize != 0)
      cuda_wait(workspace, GPUARRAY_CUDA_WAIT_WRITE);
    cuda_wait(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    #ifdef DEBUG_TIMING
    GpuArray_sync(&(*output)->ga);
    timer.start();
    #endif

    for ( int g = 0; g < groups; g++) {
      err = cudnnConvolutionForward(
        params->handle,
        alpha_p,
        APPLY_SPECIFIC(input), ((char *)PyGpuArray_DEV_DATA(input)) + input_offset * g,
        APPLY_SPECIFIC(kerns), ((char *)PyGpuArray_DEV_DATA(kerns)) + kern_offset * g,
        desc, algo,
        worksize == 0 ? NULL : *(void **)workspace, worksize,
        beta_p,
        APPLY_SPECIFIC(output), ((char *)PyGpuArray_DEV_DATA(*output)) + output_offset * g);
    }

    if (worksize != 0) {
      cuda_record(workspace, GPUARRAY_CUDA_WAIT_WRITE);
      gpudata_release(workspace);
    }

    cuda_record(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
  }

  #ifdef DEBUG_TIMING
  GpuArray_sync(&(*output)->ga);
  timer.end();
  total_computation_time += timer.milliseconds;
  ++n_computations;
  #endif

  cuda_exit(c->ctx);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error doing cuDNN conv FWD operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  #ifdef DEBUG_TIMING
  fprintf(stderr, "\t(ran fwd algo in %g milliseconds)\n", timer.milliseconds);
  if (n_computations > 1) {
    fprintf(stderr, "\t(ran %lu fwd computations in %g milliseconds (average: %g milliseconds per call))\n",
            n_computations, total_computation_time, total_computation_time / n_computations);
  }
  #endif
  return 0;
}


