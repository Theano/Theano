#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_grid)(PyGpuArrayObject * theta,
                              PyArrayObject * out_dims,
                              cudnnSpatialTransformerDescriptor_t desc,
                              PyGpuArrayObject ** grid,
                              cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = theta->context;
    size_t grid_dims[4];
    int num_images, num_channels, height, width;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    if ( theta->ga.typecode != GA_FLOAT &&
         theta->ga.typecode != GA_DOUBLE &&
         theta->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformerGrid: unsupported data type for theta in spatial transformer." );
        return 1;
    }
    else if ( PyGpuArray_DIM( theta, 1 ) != 2 || PyGpuArray_DIM( theta, 2 ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerGrid: incorrect dimensions for theta, expected (%d, %d, %d), got (%d, %d, %d)",
            PyGpuArray_DIMS( theta )[0], 2, 3, PyGpuArray_DIMS( theta )[0],
            PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
        return 1;
    }

    if ( PyArray_NDIM( out_dims ) != 1 || PyArray_SIZE( out_dims ) != 4 )
    {
        PyErr_SetString( PyExc_MemoryError,
            "GpuDnnTransformerGrid: out_dims must have 4 elements." );
        return 1;
    }

    // Obtain output dimensions
    num_images = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 0 ) );
    height = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 2 ) );
    width = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 3 ) );
    // Set grid dimensions
    grid_dims[0] = num_images;
    grid_dims[1] = height;
    grid_dims[2] = width;
    grid_dims[3] = 2;

    if ( theano_prep_output( grid, 4, grid_dims, theta->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "GpuDnnTransformerGrid: could not allocate memory for grid of coordinates" );
        return 1;
    }

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfGridGeneratorForward( _handle, desc, PyGpuArray_DEV_DATA( theta ),
        PyGpuArray_DEV_DATA( *grid ) );

    cuda_record( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    cuda_exit( gpu_ctx->ctx );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerGrid: could not create grid of coordinates: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}

