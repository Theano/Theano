#section support_code

int spatialtf_grid(cudnnSpatialTransformerDescriptor_t desc,
                   PyGpuArrayObject * theta,
                   PyArrayObject * dimensions,
                   PyGpuArrayObject ** grid,
                   cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = theta->context;
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

    if ( NULL == *grid )
    {
        // Obtain grid dimensions
        npy_int * dimensions_data = (npy_int *)PyArray_DATA( dimensions );
        const size_t width = dimensions_data[0];
        const size_t height = dimensions_data[1];
        const size_t num_images = dimensions_data[3];
        // Grid of coordinates is of size num_images * height * width * 2 for a 2D transformation
        const size_t grid_dims[4] = { width, height, 2, num_images };

        *grid = pygpu_empty( 4, &(grid_dims[0]), theta->ga.typecode, GA_C_ORDER,
            gpu_ctx, Py_None );

        if ( NULL == *grid )
        {
            PyErr_Format( PyExc_MemoryError,
                          "Could not allocate memory for grid coordinates" );
            return 1;
        }
    }

    const void * theta_data = PyGpuArray_DEV_DATA( theta );
    void * grid_data = PyGpuArray_DEV_DATA( *grid );

    err = cudnnSpatialTfGridGeneratorForward( _handle, desc, theta_data, grid_data );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "Failed to create grid of coordinates: %s",
                      cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}
