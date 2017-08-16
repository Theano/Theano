#section init_code_struct
reuse_algo = 0;
prev_algo.algo = PARAMS->conv_algo;
prev_algo.mathType = CUDNN_DEFAULT_MATH;
prev_algo.dataType = CUDNN_DATA_FLOAT;
hash_prefix = std::string("FW| GPU#");

#section support_code_struct
#line 12 "dnn_fwd.c"
int     reuse_algo;
bool    use_cached;
AlgoRec prev_algo;
std::string hash_prefix;

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
  
  int groups = c_check_groups_for_conv(desc, params->num_groups);
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
  size_t   worksize  = 0;
  cudnnMathType_t mathtype = CUDNN_DEFAULT_MATH;
  
  std::string hashkey;
  
  #ifdef DEBUG
  char algorithm_name[128];
  #endif

  size_t free = c_get_largest_free_block_size(c);
  
  cuda_enter(c->ctx);
  
  if (params->choose_algo) {
    
    if (!reuse_algo) {
      char pci_id[16];
      gpucontext_property(c->ctx, GA_CTX_PROP_PCIBUSID, pci_id);
      hashkey = dnn_conv_shape(APPLY_SPECIFIC(input), input, APPLY_SPECIFIC(kerns), kerns, desc, *output, groups);
      if (hashkey.empty())
	return 1;
      hashkey =  hash_prefix + pci_id + hashkey;
      // check out cache
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

	//	fprintf(stderr, "(cudnnFindConvolutionForwardAlgorithmEx: (err:%d), algo: %d, mem: %ld, free: %ld\n",
	//	err, choice.algo, choice.memory, free);
	
        if (err != CUDNN_STATUS_SUCCESS) {
          PyErr_Format(PyExc_RuntimeError,
                       "error selecting convolution algo: %s",
                       cudnnGetErrorString(err));
          cuda_exit(c->ctx);
          return 1;
        }
        algo = choice.algo;
        prev_algo.algo = (int)algo;
        prev_algo.wsSize = worksize = choice.memory;
#if CUDNN_MAJOR >= 7        
        prev_algo.mathType = mathtype = choice.mathType;
#endif
        // Add to the cache
        dnn_conv_update_cache(hashkey, prev_algo);

        #ifdef DEBUG
        if (count == 0) {
            PyErr_SetString(PyExc_RuntimeError, "No best-timed conv fwd algorithm found");
            return 1;
        } else if (choice.status != CUDNN_STATUS_SUCCESS) {
            PyErr_Format(PyExc_RuntimeError,
                         "error getting best-timed FWD algo: %s",
                         cudnnGetErrorString(choice.status));
            return 1;
        } // Else, count is necessarly 1 for current implementation.
        #endif

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
          prev_algo.algo = algo;
          // no tensor_op returned from Get()
          prev_algo.mathType = mathtype = CUDNN_DEFAULT_MATH;
	  // fprintf(stderr, "(cudnnGetConvolutionForwardAlgorithm: (err:%d), algo: %d\n", err, algo);
      }
    } 
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
      if (0 != theano_enum_to_string_cudnnConvolutionFwdAlgo_t(algo, algorithm_name))
        return 1;
      fprintf(stderr, "(%s error getting worksize: "
                      "fallback to CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)\n", algorithm_name);
      #endif
      algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }

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
  // save worksize for next time/cache
  prev_algo.wsSize = worksize;
  
  // Add to the cache
  if (params->choose_algo)
    dnn_conv_update_cache(hashkey, prev_algo);
  
#ifdef DEBUG  
  if (params->choose_algo) {  
    if (0 != theano_enum_to_string_cudnnConvolutionFwdAlgo_t(algo, algorithm_name))
      return 1;
    fprintf(stderr, "%s%s algo: %d %s%s ws: %ld, tensor: %d hash:%s\n",
            params->choose_algo ? "[A]": "" ,
            params->choose_time ? "[T]": "" ,
            algo, // algorithm_name,
            reuse_algo ? "(reused)" : "",
            use_cached ? "(cache)": "",
            worksize, mathtype, hashkey.c_str()
      );
  }
#endif
  
  if (params->choose_once) {
    reuse_algo = 1;
  }

  {
    gpudata *workspace = 0;  
#if CUDNN_MAJOR >= 7    
    // CUDNN7: need to set math type
    err = cudnnSetConvolutionMathType(desc, prev_algo.mathType);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "error setting math type for convolution : %s",
                   cudnnGetErrorString(err));
      cuda_exit(c->ctx);
      return 1;
    }
#endif
    
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


