#section support_code_struct

cudnnSpatialTransformerDescriptor_t APPLY_SPECIFIC(sptf);

#section init_code_struct

APPLY_SPECIFIC(sptf) = NULL;

{
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    if ((err = cudnnCreateSpatialTransformerDescriptor(&APPLY_SPECIFIC(sptf))) != CUDNN_STATUS_SUCCESS)
    {
        PyErr_Format(PyExc_MemoryError,
            "GpuDnnTransformerGradT: could not allocate spatial transformer descriptor (sptf): %s",
            cudnnGetErrorString(err));
        FAIL;
    }
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(sptf) != NULL)
    cudnnDestroySpatialTransformerDescriptor(APPLY_SPECIFIC(sptf));

#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_gt)(PyGpuArrayObject * dgrid,
                            PyGpuArrayObject ** dtheta,
                            cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = dgrid->context;
    int num_images, height, width;
    int desc_dims[4];
    size_t dtheta_dims[3];
    cudnnDataType_t dt;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    switch(dgrid->ga.typecode)
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
            "GpuDnnTransformerGradT: unsupported data type for dgrid in spatial transformer." );
        return 1;
    }

    num_images = (int) PyGpuArray_DIM( dgrid, 0 );
    height = (int) PyGpuArray_DIM( dgrid, 1 );
    width = (int) PyGpuArray_DIM( dgrid, 2 );

    dtheta_dims[0] = num_images;
    dtheta_dims[1] = 2;
    dtheta_dims[2] = 3;

    if ( theano_prep_output( dtheta, 3, dtheta_dims, dgrid->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

    desc_dims[0] = num_images;
    // Assume number of channels is 1, because the information is not
    // available or relevant here
    desc_dims[1] = 1;
    desc_dims[2] = height;
    desc_dims[3] = width;

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

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( dgrid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*dtheta)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfGridGeneratorBackward( _handle, APPLY_SPECIFIC(sptf),
        PyGpuArray_DEV_DATA( dgrid ), PyGpuArray_DEV_DATA( *dtheta ) );

    cuda_record( dgrid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*dtheta)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    cuda_exit( gpu_ctx->ctx );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        PyErr_Format( PyExc_RuntimeError,
            "GpuDnnTransformerGradT: could not compute gradients of the affine transformation: %s",
            cudnnGetErrorString( err ) );

        return 1;
    }

    return 0;
}
