#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(input_grad);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output_grad);
cudnnPoolingDescriptor_t APPLY_SPECIFIC(pool);

#section init_code_struct

APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(input_grad) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(output_grad) = NULL;
APPLY_SPECIFIC(pool) = NULL;

{
  cudnnStatus_t err;
  if ((err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError,
                 "could not allocate tensor descriptor (input): %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input_grad))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError,
                 "could not allocate tensor descriptor (input_grad): %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError,
                 "could not allocate tensor descriptor (output): %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output_grad))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError,
                 "could not allocate tensor descriptor (output_grad): %s",
                 cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreatePoolingDescriptor(&APPLY_SPECIFIC(pool))) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate pooling descriptor"
                "(pool): %s", cudnnGetErrorString(err));  
    FAIL;
  }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input)); }
if (APPLY_SPECIFIC(input_grad) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input_grad)); }
if (APPLY_SPECIFIC(output) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output)); }
if (APPLY_SPECIFIC(output_grad) != NULL) { cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output_grad)); }
if (APPLY_SPECIFIC(pool) != NULL) { cudnnDestroyPoolingDescriptor(APPLY_SPECIFIC(pool)); }

#section support_code_struct

int APPLY_SPECIFIC(dnn_pool_grad)(PyGpuArrayObject *inp,
                                  PyGpuArrayObject *out,
                                  PyGpuArrayObject *out_grad,
                                  PyArrayObject *ws, 
                                  PyArrayObject *stride,
                                  PyArrayObject *pad,
                                  PyGpuArrayObject **inp_grad,
                                  PyGpuContextObject *c) {
  cudnnStatus_t err;

  if (!GpuArray_IS_C_CONTIGUOUS(&inp->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
    return 1;
  }

  if (!GpuArray_IS_C_CONTIGUOUS(&out_grad->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous output gradients are supported.");
    return 1;
  }

  if (!GpuArray_IS_C_CONTIGUOUS(&out->ga)) {
    PyErr_SetString(PyExc_ValueError, "Only contiguous outputs are supported.");
    return 1;
  }

  if (c_set_tensorNd(inp, APPLY_SPECIFIC(input)) != 0)
    return 1;
  if (c_set_tensorNd(out_grad, APPLY_SPECIFIC(output_grad)) != 0)
    return 1;
  if (c_set_tensorNd(out, APPLY_SPECIFIC(output)) != 0)
    return 1;

  if (theano_prep_output(inp_grad, PyGpuArray_NDIM(inp),
                         PyGpuArray_DIMS(inp), inp->ga.typecode,
                         GA_C_ORDER, c) != 0) {
    return 1;
  }

  int w[3];
  int p[3];
  int s[3];
  int ndims = PyArray_DIM(ws, 0);//PyGpuArray_NDIM(img) - 2;

  for(int i = 0; i < ndims; i++) {
     w[i] = *((npy_intp*)PyArray_GETPTR1(ws, i));
  }
  for(int i = 0; i < ndims; i++) {
     p[i] = *((npy_intp*)PyArray_GETPTR1(pad, i));
  }
  for(int i = 0; i < ndims; i++) {
     s[i] = *((npy_intp*)PyArray_GETPTR1(stride, i));
  }
  err = cudnnSetPoolingNdDescriptor(APPLY_SPECIFIC(pool), MODE_FLAG, ndims, w, p, s);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor %s", cudnnGetErrorString(err));
  }

  if (c_set_tensorNd(*inp_grad, APPLY_SPECIFIC(input_grad)) != 0)
    return 1;

  {
    const float alphaf = 1;
    const float betaf = 0;
    const double alphad = 1;
    const double betad = 0;
    void *alpha, *beta;

    switch (inp->ga.typecode) {
    case GA_DOUBLE:
      alpha = (void *)&alphad;
      beta = (void *)&betad;
      break;
    case GA_FLOAT:
    case GA_HALF:
      alpha = (void *)&alphaf;
      beta = (void *)&betaf;
      break;
    default:
      PyErr_SetString(PyExc_TypeError, "Unsupported type in pooling gradient");
      return 1;
    }

    cuda_enter(c->ctx);

    cuda_wait(out->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait(out_grad->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait(inp->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_wait((*inp_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    err = cudnnPoolingBackward(
      APPLY_SPECIFIC(_handle), APPLY_SPECIFIC(pool),
      alpha,
      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(out),
      APPLY_SPECIFIC(output_grad), PyGpuArray_DEV_DATA(out_grad),
      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(inp),
      beta,
      APPLY_SPECIFIC(input_grad), PyGpuArray_DEV_DATA(*inp_grad)
      );

    cuda_record(out->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(out_grad->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(inp->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*inp_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    cuda_exit(c->ctx);
  }

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "error doing operation: %s.",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
