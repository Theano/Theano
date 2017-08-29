#section support_code_struct

cudnnSpatialTransformerDescriptor_t APPLY_SPECIFIC(sptf);
cudnnTensorDescriptor_t APPLY_SPECIFIC(xdesc);
cudnnTensorDescriptor_t APPLY_SPECIFIC(ydesc);

#section init_code_struct

APPLY_SPECIFIC(sptf) = NULL;
APPLY_SPECIFIC(xdesc) = NULL;
APPLY_SPECIFIC(ydesc) = NULL;

{
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    err = cudnnCreateSpatialTransformerDescriptor(&APPLY_SPECIFIC(sptf));
    if (err != CUDNN_STATUS_SUCCESS)
    {
        PyErr_Format(PyExc_MemoryError,
            "GpuDnnTransformerSampler: could not allocate spatial transformer descriptor (sptf): %s",
            cudnnGetErrorString( err ));
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(xdesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerSampler: failed to allocate cuDNN tensor descriptor xdesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }

    err = cudnnCreateTensorDescriptor( &APPLY_SPECIFIC(ydesc) );
    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerSampler: failed to allocate cuDNN tensor descriptor ydesc: %s",
            cudnnGetErrorString( err ) );
        FAIL;
    }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(sptf) != NULL)
    cudnnDestroySpatialTransformerDescriptor(APPLY_SPECIFIC(sptf));

if ( APPLY_SPECIFIC(xdesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(xdesc) );

if ( APPLY_SPECIFIC(ydesc) != NULL )
    cudnnDestroyTensorDescriptor( APPLY_SPECIFIC(ydesc) );

#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_sampler)(PyGpuArrayObject * input,
                                 PyGpuArrayObject * grid,
                                 PyGpuArrayObject ** output,
                                 cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    double alpha = 1.0, beta = 0.0;
    float af = alpha, bf = beta;
    size_t out_dims[4];
    int desc_dims[4];
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
            "GpuDnnTransformer: unsupported type for input in spatial transformer." );
        return 1;
    }

    out_dims[0] = (size_t) PyGpuArray_DIM(input, 0); // num_images
    out_dims[1] = (size_t) PyGpuArray_DIM(input, 1); // num_channels
    out_dims[2] = (size_t) PyGpuArray_DIM(grid, 1); // grid height
    out_dims[3] = (size_t) PyGpuArray_DIM(grid, 2); // grid width
    // Set output dimensions for the descriptor setup
    desc_dims[0] = (int) out_dims[0];
    desc_dims[1] = (int) out_dims[1];
    desc_dims[2] = (int) out_dims[2];
    desc_dims[3] = (int) out_dims[3];

    if ( out_dims[0] == 0 || out_dims[1] == 0 || out_dims[2] == 0 || out_dims[3] == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "GpuDnnTransformerSampler: one of the sampler dimensions is zero" );
        return 1;
    }

    if ( theano_prep_output( output, 4, out_dims, input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        PyErr_SetString( PyExc_MemoryError,
            "GpuDnnTransformerSampler: could not allocate memory for grid sampler" );
        return 1;
    }

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so the sampler method is currently not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor(APPLY_SPECIFIC(sptf), CUDNN_SAMPLER_BILINEAR,
        dt, 4, desc_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerSampler: could not initialize descriptor: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    if ( c_set_tensorNd( input, APPLY_SPECIFIC(xdesc) ) != 0 )
        return 1;

    if ( c_set_tensorNd( *output, APPLY_SPECIFIC(ydesc) ) != 0 )
        return 1;

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfSamplerForward( _handle, APPLY_SPECIFIC(sptf), alpha_p,
        APPLY_SPECIFIC(xdesc), PyGpuArray_DEV_DATA( input ), PyGpuArray_DEV_DATA( grid ),
        beta_p, APPLY_SPECIFIC(ydesc), PyGpuArray_DEV_DATA( *output ) );

    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    cuda_exit( gpu_ctx->ctx );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerSampler: could not create grid sampler: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}

