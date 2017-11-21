#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnReduceTensorDescriptor_t APPLY_SPECIFIC(red);
GpuElemwise* elemwise;
gpuelemwise_arg arg;

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(red) = NULL;

if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateReduceTensorDescriptor(&APPLY_SPECIFIC(red))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate reduction descriptor"
               "(red): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

elemwise = NULL;

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input)); }
if (APPLY_SPECIFIC(output) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output)); }
if (APPLY_SPECIFIC(red) != NULL) { cudnnDestroyReduceTensorDescriptor(APPLY_SPECIFIC(red)); }

if (elemwise) {
    GpuElemwise_free(elemwise);
    elemwise = NULL;
}

#section support_code_struct

int APPLY_SPECIFIC(dnn_redux)(PyGpuArrayObject *input,
                              PyGpuArrayObject **output,
                              PyGpuArrayObject **indices,
                              PARAMS_TYPE* params) {
  PyGpuContextObject *c = input->context;
  gpudata *workspace = NULL;
  size_t worksize = 0;
  size_t indsize = 0;
  size_t *tdims;
  ssize_t *tstrs;
  size_t dims[8];
  ssize_t strs[8];
  size_t rsz;
  void *alpha;
  void *beta;
  cudnnStatus_t err;
  unsigned int p;
  int e;

  static float falpha = 1.0f;
  static double dalpha = 1.0;
  static float fbeta = 0.0f;
  static double dbeta = 0.0;

  if (!GpuArray_IS_C_CONTIGUOUS(&input->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
    return 1;
  }

  if (c_set_tensorNd(input, APPLY_SPECIFIC(input)) != 0)
    return 1;

  p = 0;
  rsz = 1;
  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
    if (!(params->c_axis & (1U << i))) {
      dims[p] = PyGpuArray_DIM(input, i);
      p++;
    } else {
      rsz *= PyGpuArray_DIM(input, i);
    }
  }

  if (indices != NULL) {
    if (theano_prep_output(indices, p, dims, GA_UINT, GA_C_ORDER, c) != 0)
      return 1;
    indsize = PyGpuArray_SIZE(*indices) * 4;
  }

  if (p == input->ga.nd || rsz == 1) {
    int err;
    Py_XDECREF(*output);
    *output = pygpu_copy(input, GA_C_ORDER);
    if (*output == NULL)
      return 1;
    err = GpuArray_reshape_inplace(&(*output)->ga, p, dims, GA_C_ORDER);
    if (err != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuArray_reshape_inplace: %s", GpuArray_error(&(*output)->ga, err));
      return 1;
    }

    if (rsz == 1) {
      /* We must reduce some dimensions which have all size 1.
       * cuDNN (up to 7004) does not support this case. Let's use GpuElemwise. */
      switch (params->red_op) {
        // Nothing to do for following cases.
        case CUDNN_REDUCE_TENSOR_ADD: break;
        case CUDNN_REDUCE_TENSOR_MUL: break;
        case CUDNN_REDUCE_TENSOR_MIN: break;
        case CUDNN_REDUCE_TENSOR_MAX: break;
        case CUDNN_REDUCE_TENSOR_AVG: break;
        /* Work to do for following cases.
        AMAX (maximum on absolute values) => apply abs(output)
        NORM1 (addition of absolute values) => apply abs(output)
        NORM2 (square root of sum of squares) => sqroot(output^2) => abs(output)
        So, we must apply abs(output) for all following cases.
        */
        case CUDNN_REDUCE_TENSOR_AMAX:
        case CUDNN_REDUCE_TENSOR_NORM1:
        case CUDNN_REDUCE_TENSOR_NORM2:
        {
            if (elemwise == NULL) {
              arg.name = "out";
              arg.typecode = (*output)->ga.typecode;
              arg.flags = GE_READ | GE_WRITE;
              elemwise = GpuElemwise_new(c->ctx, "", "out = (out < 0 ? -out : out)", 1, &arg, p, GE_CONVERT_F16);
              if (!elemwise) {
                  PyErr_SetString(PyExc_RuntimeError, "Unable to create GpuElemwise for output.");
                  return 1;
              }
            }
            void* args[1] = { (void*)&(*output)->ga };
            int err = GpuElemwise_call(elemwise, args, 0);
            if (err != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to call GpuElemwise on output.");
                return 1;
            };
        }
            break;
        default: break;
      }
    }

    if (indices != NULL) {
      // All indices will be 0 since the size of the reduced area is 1.
      err = GpuArray_memset(&(*indices)->ga, 0);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError, "GpuArray_memset: %s", GpuArray_error(&(*indices)->ga, err));
        return 1;
      }
    }
    // This is a shortcut path.
    return 0;
  }

  if (theano_prep_output(output, p, dims, input->ga.typecode,
                         GA_C_ORDER, c) != 0)
    return 1;

  // cuDNN expect that the output has the same number of dimension as
  // the input, but the dimensions to reduce are of size 1 in the output.
  // We have to do some trickery to be able to pass it what it need.
  p = 0;
  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
    if (params->c_axis & (1U << i)) {
      dims[i] = 1;
      strs[i] = 0;
    } else {
      dims[i] = PyGpuArray_DIM(input, i);
      strs[i] = PyGpuArray_STRIDE(*output, p);
      p++;
    }
  }

  // Perform horrible surgery to be able to reuse c_set_tensorNd()
  tdims = (*output)->ga.dimensions;
  tstrs = (*output)->ga.strides;
  (*output)->ga.dimensions = dims;
  (*output)->ga.strides = strs;
  (*output)->ga.nd = input->ga.nd;

  // Delay error checking to avoid exposing a broken object
  e = c_set_tensorNd(*output, APPLY_SPECIFIC(output));

  // Undo our horrible surgery
  (*output)->ga.nd = p;
  (*output)->ga.dimensions = tdims;
  (*output)->ga.strides = tstrs;

  if (e != 0)
    return 1;
  // Back to normal, no more horrible things

  // Note that only CUDNN_32BIT_INDICES is implemented
  err = cudnnSetReduceTensorDescriptor(
    APPLY_SPECIFIC(red), params->red_op,
    params->acc_dtype, CUDNN_PROPAGATE_NAN,
    indices == NULL ? CUDNN_REDUCE_TENSOR_NO_INDICES : CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
    CUDNN_32BIT_INDICES);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set reduce descriptor: %s",
                 cudnnGetErrorString(err));
    return 1;
  }

  switch (input->ga.typecode) {
  case GA_FLOAT:
  case GA_HALF:
    alpha = &falpha;
    beta = &fbeta;
    break;
  case GA_DOUBLE:
    alpha = &dalpha;
    beta = &dbeta;
    break;
  default:
    PyErr_SetString(PyExc_RuntimeError, "Unsupported dtype in dnn reduce");
    return 1;
  }

  err = cudnnGetReductionWorkspaceSize(params->handle,
                                       APPLY_SPECIFIC(red),
                                       APPLY_SPECIFIC(input),
                                       APPLY_SPECIFIC(output),
                                       &worksize);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not get reduce workspace size: %s",
                 cudnnGetErrorString(err));
    return 1;
  }

  if (worksize != 0) {
    workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, &e);
    if (workspace == NULL) {
      PyErr_Format(PyExc_RuntimeError, "gpudata_alloc: %s",
                   gpucontext_error(c->ctx, e));
      return 1;
    }
  }

  err = cudnnReduceTensor(params->handle, APPLY_SPECIFIC(red),
                          indices ? PyGpuArray_DEV_DATA(*indices) : NULL, indsize,
                          worksize ? *((void **)workspace) : NULL, worksize,
                          alpha,
                          APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
                          beta,
                          APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output));

  if (workspace != NULL)
    gpudata_release(workspace);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not run reduction: %s",
                 cudnnGetErrorString(err));
    return 1;
  }

  return 0;
}
