#section support_code

int dnn_rnn_desc(int hidden_size, int num_layers,
                 cudnnDropoutDescriptor_t ddesc,
                 int input_mode, int direction_mode, int rnn_mode,
                 int dtype, cudnnRNNDescriptor_t *odesc,
                 cudnnHandle_t _handle) {
  cudnnRNNDescriptor_t desc;
  cudnnDataType_t data_type;
  cudnnStatus_t err;

  switch (dtype) {
  case GA_FLOAT:
    data_type = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    data_type = CUDNN_DATA_DOUBLE;
    break;
  case GA_HALF:
    data_type = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "Unsupported data type");
    return -1;
  }

  err = cudnnCreateRNNDescriptor(&desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Can't create RNN descriptor");
    return -1;
  }

  err = cudnnSetRNNDescriptor(desc, hidden_size, num_layers, ddesc,
                              (cudnnRNNInputMode_t)input_mode,
                              (cudnnDirectionMode_t)direction_mode,
                              (cudnnRNNMode_t)rnn_mode, data_type);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyRNNDescriptor(desc);
    PyErr_SetString(PyExc_RuntimeError, "Can't set RNN descriptor");
    return -1;
  }

  *odesc = desc;
  return 0;
}
