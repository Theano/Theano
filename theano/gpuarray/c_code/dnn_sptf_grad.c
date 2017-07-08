#section support_code_struct

int
dnn_sptf_grad(PyGpuArrayObject * input,
              PyGpuArrayObject * theta,
              PyGpuArrayObject * grid,
              PyArrayObject * grid_dims,
              PyGpuArrayObject * dy,
              cudnnSpatialTransformerDescriptor_t desc,
              double alpha, double beta,
              PyGpuArrayObject ** output_grad,
              PyGpuArrayObject ** grid_grad,
              cudnnHandle_t _handle)
{
    PyErr_SetString(PyExc_NotImplementedError, "Gradient for spatial transformer is not yet implemented.");
    return -1;
}
