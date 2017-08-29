#section init_code_struct

{
  cudnnStatus_t err;

  bn_input = NULL;
  bn_params = NULL;
  bn_output = NULL;

  if ((err = cudnnCreateTensorDescriptor(&bn_input)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
                 "(bn_input): %s", cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreateTensorDescriptor(&bn_params)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
                 "(bn_params): %s", cudnnGetErrorString(err));
    FAIL;
  }
  if ((err = cudnnCreateTensorDescriptor(&bn_output)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
                 "(bn_output): %s", cudnnGetErrorString(err));
    FAIL;
  }
}

#section cleanup_code_struct

if (bn_input != NULL)
  cudnnDestroyTensorDescriptor(bn_input);
if (bn_params != NULL)
  cudnnDestroyTensorDescriptor(bn_params);
if (bn_output != NULL)
  cudnnDestroyTensorDescriptor(bn_output);

#section support_code_struct

cudnnTensorDescriptor_t bn_input;
cudnnTensorDescriptor_t bn_params;
cudnnTensorDescriptor_t bn_output;
