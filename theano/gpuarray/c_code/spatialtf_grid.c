#section support_code

int
spatialtf_grid(PyArrayObject * grid_dimensions,
               PyGpuArrayObject * theta,
               cudnnSpatialTransformerDescriptor_t desc,
               PyGpuArrayObject ** grid,
               cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = theta->context;
    cudnnStatus_t err;

    if ( theta->ga.typecode != GA_FLOAT &&
         theta->ga.typecode != GA_DOUBLE &&
         theta->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError, "Unsupported data type for theta" );
        return -1;
    }

    if ( PyGpuArray_NDIM( theta ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "theta must have three dimensions!" );
        return -1;
    }

    if ( PyGpuArray_DIM( theta, 1 ) != 2 && PyGpuArray_DIM( theta, 2 ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "Incorrect dimensions for theta, should be (%d, %d, %d), got (%d, %d, %d)",
                      PyGpuArray_DIMS( theta )[0], 2, 3, PyGpuArray_DIMS( theta )[0],
                      PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
        return -1;
    }

    if ( PyArray_DIM( grid_dimensions, 0 ) < 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "Grid dimensions array must have at least 3 dimensions!" );
        return -1;
    }

    // Obtain grid dimensions
    npy_int * dimensions_data = (npy_int *)PyArray_DATA( grid_dimensions );

    const size_t num_images = dimensions_data[0];
    const size_t height = dimensions_data[1];
    const size_t width = dimensions_data[2];
    // Grid of coordinates is of size num_images * height * width * 2 for a 2D transformation
    const size_t grid_dims[4] = { num_images, height, width, 2 };

    if ( width == 0 || height == 0 || num_images == 0 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "One of the grid dimensions is zero" );
        return -1;
    }

    if ( NULL == *grid ||
         ! theano_size_check( *grid, 4, &(grid_dims[0]), (*grid)->ga.typecode ) )
    {
        Py_XDECREF( *grid );

        *grid = pygpu_zeros( 4, &(grid_dims[0]), theta->ga.typecode, GA_C_ORDER,
            gpu_ctx, Py_None );
        if ( NULL == *grid )
        {
            PyErr_SetString( PyExc_MemoryError,
                             "Could not allocate memory for grid of coordinates" );
            return -1;
        }
    }
    else
    {
        GpuArray_memset( &( (*grid)->ga ), 0 );
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &(theta->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "theta data is not C-contiguous" );
        return -1;
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &((*grid)->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "grid data is not C-contiguous" );
        return -1;
    }

    const void * theta_data = PyGpuArray_DEV_DATA( theta );
    void * grid_data = PyGpuArray_DEV_DATA( *grid );

    err = cudnnSpatialTfGridGeneratorForward( _handle, desc, theta_data, grid_data );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "Failed to create grid of coordinates: %s",
                      cudnnGetErrorString( err ) );
        return -1;
    }

    return 0;
}
