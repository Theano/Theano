#section support_code_struct

cudnnSpatialTransformerDescriptor_t APPLY_SPECIFIC(sptf);

#section init_code_struct

APPLY_SPECIFIC(sptf) = NULL;

{
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    if ((err = cudnnCreateSpatialTransformerDescriptor(&APPLY_SPECIFIC(sptf))) != CUDNN_STATUS_SUCCESS)
    {
        PyErr_Format(PyExc_MemoryError,
            "GpuDnnTransformerGrid: could not allocate spatial transformer descriptor (sptf): %s",
            cudnnGetErrorString(err));
        FAIL;
    }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(sptf) != NULL) { cudnnDestroySpatialTransformerDescriptor(APPLY_SPECIFIC(sptf)); }

#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_grid)(PyGpuArrayObject * theta,
                              PyArrayObject * out_dims,
                              PyGpuArrayObject ** grid,
                              cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = theta->context;
    size_t grid_dims[4];
    int num_images, num_channels, height, width;
    int desc_dims[4];
    cudnnDataType_t dt;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    switch(theta->ga.typecode)
    {
    case GA_DOUBLE:
        dt = CUDNN_DATA_DOUBLE;
        break;
    case GA_FLOAT:
        dt = CUDNN_DATA_FLOAT;
        break;
    case GA_HALF:
        dt = CUDNN_DATA_HALF;
        break;
    default:
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformerGrid: unsupported data type for theta in spatial transformer." );
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
    num_channels = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 1 ) );
    height = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 2 ) );
    width = (int) *( (npy_int64 *) PyArray_GETPTR1( out_dims, 3 ) );

    if ( PyGpuArray_DIM( theta, 0 ) != num_images ||
         PyGpuArray_DIM( theta, 1 ) != 2 || PyGpuArray_DIM( theta, 2 ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerGrid: incorrect dimensions for theta, expected (%d, %d, %d), got (%d, %d, %d)",
            num_images, 2, 3, PyGpuArray_DIMS( theta )[0],
            PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
        return 1;
    }

    // Set transformed output dimensions to setup the descriptor
    desc_dims[0] = num_images;
    desc_dims[1] = num_channels;
    desc_dims[2] = height;
    desc_dims[3] = width;
    // Set sampling grid dimensions
    grid_dims[0] = num_images;
    grid_dims[1] = height;
    grid_dims[2] = width;
    grid_dims[3] = 2;

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so the sampler method is currently not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor(APPLY_SPECIFIC(sptf), CUDNN_SAMPLER_BILINEAR,
        dt, 4, desc_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGrid: could not initialize descriptor (sptf): %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

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

    err = cudnnSpatialTfGridGeneratorForward( _handle, APPLY_SPECIFIC(sptf),
        PyGpuArray_DEV_DATA( theta ), PyGpuArray_DEV_DATA( *grid ) );

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
