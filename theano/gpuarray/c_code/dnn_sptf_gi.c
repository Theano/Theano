#section support_code_struct

cudnnSpatialTransformerDescriptor_t APPLY_SPECIFIC(sptf);
cudnnTensorDescriptor_t APPLY_SPECIFIC(xdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(dxdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(dydesc);

#section init_code_struct

APPLY_SPECIFIC(sptf) = NULL;
APPLY_SPECIFIC(xdesc) = NULL;
APPLY_SPECIFIC(dxdesc) = NULL;
APPLY_SPECIFIC(dydesc) = NULL;

{
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    
    err = cudnnCreateSpatialTransformerDescriptor(&APPLY_SPECIFIC(sptf));
    if (err != CUDNN_STATUS_SUCCESS)
    {
        PyErr_Format(PyExc_MemoryError,
            "GpuDnnTransformerGradI: could not allocate spatial transformer descriptor (sptf): %s",
            cudnnGetErrorString(err));
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(xdesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor xdesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(dxdesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor dxdesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(dydesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor dydesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(sptf) != NULL)
    cudnnDestroySpatialTransformerDescriptor( APPLY_SPECIFIC(sptf) );

if ( APPLY_SPECIFIC(xdesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(xdesc) );

if ( APPLY_SPECIFIC(dxdesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(dxdesc) );

if ( APPLY_SPECIFIC(dydesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(dydesc) );

#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_gi)(PyGpuArrayObject * input,
                            PyGpuArrayObject * grid,
                            PyGpuArrayObject * dy,
                            PyGpuArrayObject ** input_grad,
                            PyGpuArrayObject ** grid_grad,
                            cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    double alpha = 1.0, beta = 0.0;
    float af = alpha, bf = beta;
    int out_dims[4];
    cudnnDataType_t dt;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    switch (input->ga.typecode)
    {
    case GA_DOUBLE:
        alpha_p = (void *)&alpha;
        beta_p = (void *)&beta;
        dt = CUDNN_DATA_DOUBLE;
        break;
    case GA_FLOAT:
        alpha_p = (void *)&af;
        beta_p = (void *)&bf;
        dt = CUDNN_DATA_FLOAT;
        break;
    case GA_HALF:
        alpha_p = (void *)&af;
        beta_p = (void *)&bf;
        dt = CUDNN_DATA_HALF;
        break;
    default:
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformerGradI: unsupported type for input in spatial transformer gradients" );
        return 1;
    }

    if ( grid->ga.typecode != GA_FLOAT &&
         grid->ga.typecode != GA_DOUBLE &&
         grid->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformerGradI: unsupported data type for grid in spatial transformer gradients." );
        return 1;
    }

    if ( theano_prep_output( input_grad, PyGpuArray_NDIM( input ),
                             PyGpuArray_DIMS( input ), input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

    if ( theano_prep_output( grid_grad, PyGpuArray_NDIM( grid ),
                             PyGpuArray_DIMS( grid ), grid->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

    // Obtain output dimensions to setup descriptor
    out_dims[0] = (int) PyGpuArray_DIM(input, 0); // num_images
    out_dims[1] = (int) PyGpuArray_DIM(input, 1); // num_channels
    out_dims[2] = (int) PyGpuArray_DIM(grid, 1); // grid height
    out_dims[3] = (int) PyGpuArray_DIM(grid, 2); // grid width

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so the sampler method is currently not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor(APPLY_SPECIFIC(sptf), CUDNN_SAMPLER_BILINEAR,
        dt, 4, out_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerGradI: could not initialize descriptor (sptf): %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    if ( c_set_tensorNd( input, APPLY_SPECIFIC(xdesc) ) != 0 )
        return 1;

    if ( c_set_tensorNd( dy, APPLY_SPECIFIC(dydesc) ) != 0 )
        return 1;

    if ( c_set_tensorNd( *input_grad, APPLY_SPECIFIC(dxdesc) ) != 0 )
        return 1;

    // Directly return the outputs if any of the dimensions is 0.
    // (cuDNN does not support zero-length dimensions.)
    if ( PyGpuArray_SIZE( *input_grad ) == 0 || PyGpuArray_SIZE( *grid_grad ) == 0 )
        return 0;

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( dy->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*input_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    cuda_wait( (*grid_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfSamplerBackward( _handle, APPLY_SPECIFIC(sptf), alpha_p,
        APPLY_SPECIFIC(xdesc), PyGpuArray_DEV_DATA( input ), beta_p,
        APPLY_SPECIFIC(dxdesc), PyGpuArray_DEV_DATA( *input_grad ), alpha_p,
        APPLY_SPECIFIC(dydesc), PyGpuArray_DEV_DATA( dy ), PyGpuArray_DEV_DATA( grid ),
        beta_p, PyGpuArray_DEV_DATA( *grid_grad ) );

    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( dy->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*input_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    cuda_record( (*grid_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    cuda_exit( gpu_ctx->ctx );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerGradI: failed to compute gradients of the inputs: %s",
            cudnnGetErrorString( err ) );

        return 1;
    }

    return 0;
}
