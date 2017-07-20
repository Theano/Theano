#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(xdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(ydesc);

#section init_code_struct

APPLY_SPECIFIC(xdesc) = NULL;
APPLY_SPECIFIC(ydesc) = NULL;

{
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(xdesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor xdesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(ydesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor ydesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }
}

#section cleanup_code_struct

if ( APPLY_SPECIFIC(xdesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(xdesc) );

if ( APPLY_SPECIFIC(ydesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(ydesc) );

#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf)(PyGpuArrayObject * input,
                         PyGpuArrayObject * theta,
                         PyArrayObject * grid_dims,
                         cudnnSpatialTransformerDescriptor_t desc,
                         PyGpuArrayObject ** output,
                         PyGpuArrayObject ** grid,
                         cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    double alpha = 1.0, beta = 0.0;
    float af = alpha, bf = beta;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    int num_images, num_channels, height, width;
    size_t gpu_grid_dims[4], out_dims[4];

    switch (input->ga.typecode)
    {
    case GA_DOUBLE:
        alpha_p = (void *)&alpha;
        beta_p = (void *)&beta;
        break;
    case GA_FLOAT:
        alpha_p = (void *)&af;
        beta_p = (void *)&bf;
        break;
    case GA_HALF:
        alpha_p = (void *)&af;
        beta_p = (void *)&bf;
        break;
    default:
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformer: unsupported type for input in spatial transformer." );
        return 1;
    }

    if ( theta->ga.typecode != GA_FLOAT &&
         theta->ga.typecode != GA_DOUBLE &&
         theta->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformer: unsupported data type for theta in spatial transformer." );
        return 1;
    }
    else if ( PyGpuArray_DIM( theta, 1 ) != 2 && PyGpuArray_DIM( theta, 2 ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformer: incorrect dimensions for theta, expected (%d, %d, %d), got (%d, %d, %d)",
            PyGpuArray_DIMS( theta )[0], 2, 3, PyGpuArray_DIMS( theta )[0],
            PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
        return 1;
    }

    if ( PyArray_NDIM( grid_dims ) != 1 || PyArray_SIZE( grid_dims ) != 4 )
    {
        PyErr_SetString( PyExc_MemoryError,
            "GpuDnnTransformer: grid_dims must have 4 elements." );
        return 1;
    }

    // Obtain grid dimensions
    num_images = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 0 ) );
    num_channels = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 1 ) );
    height = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 2 ) );
    width = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 3 ) );

    gpu_grid_dims[0] = num_images;
    gpu_grid_dims[1] = height;
    gpu_grid_dims[2] = width;
    gpu_grid_dims[3] = 2;

    out_dims[0] = num_images;
    out_dims[1] = num_channels;
    out_dims[2] = height;
    out_dims[3] = width;

    if ( width == 0 || height == 0 || num_images == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "GpuDnnTransformer: grid_dims has a dimension with value zero" );
        return 1;
    }

    if ( PyGpuArray_DIM( input, 0 ) != num_images )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformer: expected input to have %d inputs, got %d inputs.",
            num_images, PyGpuArray_DIM( input, 0 ) );
        return 1;
    }
    else if ( PyGpuArray_DIM( input, 1 ) != num_channels )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformer: expected input to have %d channels, got %d channels.",
            num_channels, PyGpuArray_DIM( input, 1 ) );
        return 1;
    }

    if ( theano_prep_output( grid, 4, gpu_grid_dims, input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "GpuDnnTransformer: could not allocate memory for grid of coordinates" );
        return 1;
    }

    if ( theano_prep_output( output, 4, out_dims, input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        PyErr_SetString( PyExc_MemoryError,
            "GpuDnnTransformer: could not allocate memory for grid sampler" );
        return 1;
    }

    if ( c_set_tensorNd( input, APPLY_SPECIFIC(xdesc) ) != 0 )
        return 1;

    if ( c_set_tensorNd( *output, APPLY_SPECIFIC(ydesc) ) != 0 )
        return 1;

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    cuda_wait( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfGridGeneratorForward( _handle, desc, PyGpuArray_DEV_DATA( theta ),
        PyGpuArray_DEV_DATA( *grid ) );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformer: could not create grid of coordinates: %s",
            cudnnGetErrorString( err ) );
        cuda_exit( gpu_ctx->ctx );
        return 1;
    }

    err = cudnnSpatialTfSamplerForward( _handle, desc, alpha_p, APPLY_SPECIFIC(xdesc),
        PyGpuArray_DEV_DATA( input ), PyGpuArray_DEV_DATA( *grid ), beta_p,
        APPLY_SPECIFIC(ydesc), PyGpuArray_DEV_DATA( *output ) );

    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    cuda_record( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    cuda_exit( gpu_ctx->ctx );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformer: could not create grid sampler: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}
