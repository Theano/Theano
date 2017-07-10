#section support_code_struct

int
dnn_sptf_gt(PyGpuArrayObject * dgrid,
            cudnnSpatialTransformerDescriptor_t desc,
            PyGpuArrayObject ** dtheta,
            cudnnHandle_t _handle)
{
    PyErr_SetString(PyExc_NotImplementedError, "Gradient for spatial transformer is not yet implemented.");
    return -1;
}
