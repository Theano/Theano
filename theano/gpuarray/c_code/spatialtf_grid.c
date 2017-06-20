#section support_code

int spatialtf_grid(cudnnSpatialTransformerDescriptor_t desc,
                   PyGpuArrayObject * theta,
                   PyGpuArrayObject * num_dimensions,
                   PyGpuArrayObject ** grid,
                   cudnnHandle_t _handle)
{
    cudnnDataType_t dt;
    cudnnStatus_t err;

    // Obtain GPU datatype from theta
    switch( theta->ga.typecode )
    {
    case GA_FLOAT:
        dt = CUDNN_DATA_FLOAT;
        break;
    case GA_DOUBLE:
        dt = CUDNN_DATA_DOUBLE;
        break;
    case GA_HALF:
        dt = CUDNN_DATA_HALF;
        break;
    default:
        PyErr_SetString( PyExc_TypeError, "Unsupported data type for theta" );
        return -1;
    }

    switch( num_dimensions->ga.typecode )
    {
    case GA_INT:
        break;
    default:
        PyErr_SetString( PyExc_TypeError, "Unsupported data type for the number of dimensions" );
    }

    return 0;
}
