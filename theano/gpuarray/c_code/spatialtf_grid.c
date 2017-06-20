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

    if ( NULL == desc )
    {
        err = cudnnCreateSpatialTransformerDescriptor( &desc );
        if ( CUDNN_STATUS_SUCCESS != err )
        {
            PyErr_SetString( PyExc_MemoryError,
                "Could not allocate spatial transformer descriptor" );
            return -1;
        }

        err = cudnnSetSpatialTransformerNdDescriptor( desc, CUDNN_SAMPLER_BILINEAR, dt, , );

        if ( CUDNN_STATUS_SUCCESS != err )
        {
            PyErr_Format( PyExc_RuntimeError,
                "Could not set spatial transformer descriptor: %s",
                cudnnGetErrorString(err)) ;
            return -1;
        }
    }

    return 0;
}
