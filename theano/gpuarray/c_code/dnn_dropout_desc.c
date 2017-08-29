#section support_code

int dnn_dropout_desc(float dropout, unsigned long long seed,
                     PyGpuContextObject *c,
                     cudnnDropoutDescriptor_t *odesc,
                     PyGpuArrayObject **ostates,
                     cudnnHandle_t _handle) {
  PyGpuArrayObject *states;
  cudnnDropoutDescriptor_t desc;
  size_t states_sz;
  cudnnStatus_t err;

  cuda_enter(c->ctx);
  err = cudnnCreateDropoutDescriptor(&desc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Can't create dropout descriptor");
    cuda_exit(c->ctx);
    return -1;
  }

  /* Can't fail according to docs */
  cudnnDropoutGetStatesSize(_handle, &states_sz);
  
  states = pygpu_empty(1, &states_sz, GA_UBYTE, GA_C_ORDER, c, Py_None);
  if (states == NULL) {
    cudnnDestroyDropoutDescriptor(desc);
    cuda_exit(c->ctx);
    return -1;
  }

  err = cudnnSetDropoutDescriptor(desc, _handle, dropout,
                                  PyGpuArray_DEV_DATA(states),
                                  states_sz, seed);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Can't set dropout descriptor");
    Py_DECREF((PyObject *)states);
    cudnnDestroyDropoutDescriptor(desc);
    cuda_exit(c->ctx);
    return -1;
  }
  cuda_exit(c->ctx);
  *odesc = desc;
  *ostates = states;
  return 0;
}
