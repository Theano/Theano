#section support_code_struct

cudnnTensorDescriptor_t APPLY_SPECIFIC(xdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(dxdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(dydesc);

#section init_code_struct

APPLY_SPECIFIC(xdesc) = NULL;
APPLY_SPECIFIC(dxdesc) = NULL;
APPLY_SPECIFIC(dydesc) = NULL;

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
                            cudnnSpatialTransformerDescriptor_t desc,
                            PyGpuArrayObject ** input_grad,
                            PyGpuArrayObject ** grid_grad,
                            cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    double alpha = 1.0, beta = 0.0;
    float af = alpha, bf = beta;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    int input_num_images, input_num_channels,
        input_height, input_width;
    int num_images, num_channels, height, width;

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
            "GpuDnnTransformerGradI: unsupported type for input in spatial transformer gradients" );
        return -1;
    }

    if ( grid->ga.typecode != GA_FLOAT &&
         grid->ga.typecode != GA_DOUBLE &&
         grid->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError,
            "GpuDnnTransformerGradI: unsupported data type for grid in spatial transformer gradients." );
        return -1;
    }

    if ( theano_prep_output( input_grad, PyGpuArray_NDIM( input ),
                             PyGpuArray_DIMS( input ), input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

    if ( theano_prep_output( grid_grad, PyGpuArray_NDIM( grid ),
                             PyGpuArray_DIMS( grid ), grid->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

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

    err = cudnnSpatialTfSamplerBackward( _handle, desc, alpha_p, APPLY_SPECIFIC(xdesc),
        PyGpuArray_DEV_DATA( input ), beta_p, APPLY_SPECIFIC(dxdesc),
        PyGpuArray_DEV_DATA( *input_grad ), alpha_p, APPLY_SPECIFIC(dydesc),
        PyGpuArray_DEV_DATA( dy ), PyGpuArray_DEV_DATA( grid ), beta_p,
        PyGpuArray_DEV_DATA( *grid_grad ) );

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

        return -1;
    }

    return 0;
}
