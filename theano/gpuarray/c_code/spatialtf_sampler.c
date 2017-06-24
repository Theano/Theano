#section support_code

typedef struct __spatialtf_context {
    cudnnTensorDescriptor_t xdesc;
    cudnnTensorDescriptor_t ydesc;
} spatialtf_context_t;

void spatialtf_context_init( spatialtf_context_t * ctx )
{
    ctx->xdesc = NULL;
    ctx->ydesc = NULL;
}

void spatialtf_context_destroy( spatialtf_context_t * ctx )
{
    if ( NULL != ctx->xdesc )
        cudnnDestroyTensorDescriptor( ctx->xdesc );

    if ( NULL != ctx->ydesc )
        cudnnDestroyTensorDescriptor( ctx->ydesc );
}

#section support_code_struct

int
spatialtf_sampler(PyGpuArrayObject * input,
                  PyGpuArrayObject * om,
                  PyGpuArrayObject * grid,
                  PyArrayObject * grid_dimensions,
                  cudnnSpatialTransformerDescriptor_t desc,
                  double alpha, double beta,
                  PyGpuArrayObject ** output,
                  cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    float af = alpha, bf = beta;
    spatialtf_context_t spatialtf_ctx;
    cudnnDataType_t dt;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    // Obtain grid dimensions
    npy_int * dimensions_data = (npy_int *)PyArray_DATA( grid_dimensions );
    const int width = dimensions_data[0];
    const int height = dimensions_data[1];
    const int num_channels = dimensions_data[2];
    const int num_images = dimensions_data[3];

    switch (grid->ga.typecode)
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
        PyErr_SetString(PyExc_TypeError,
                        "Unsupported type in spatial transformer sampler");
        return -1;
    }

    spatialtf_context_init( &spatialtf_ctx );

    cuda_enter( gpu_ctx->ctx );

    err = cudnnCreateTensorDescriptor( &(spatialtf_ctx.xdesc) );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "Could not create xdesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.xdesc, CUDNN_TENSOR_NCHW, dt,
        num_images, num_channels, height, width );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "Could not initialize xdesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    err = cudnnCreateTensorDescriptor( &(spatialtf_ctx.ydesc) );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "Could not create xdesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.ydesc, CUDNN_TENSOR_NCHW, dt,
        num_images, num_channels, height, width );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "Could not initialize ydesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    if ( NULL == *output )
    {
        *output = pygpu_zeros( PyGpuArray_NDIM(om), PyGpuArray_DIMS(om), input->ga.typecode,
            GA_C_ORDER, gpu_ctx, Py_None );

        if ( NULL == *output )
        {
            spatialtf_context_destroy( &spatialtf_ctx );
            cuda_exit( gpu_ctx->ctx );

            PyErr_SetString( PyExc_MemoryError,
                             "Could allocate memory for spatial transformer's grid sampler" );
            return -1;
        }
    }

    const void * input_data = PyGpuArray_DEV_DATA( input );
    const void * grid_data  = PyGpuArray_DEV_DATA( grid );
    void * out_data =  PyGpuArray_DEV_DATA( *output );

    err = cudnnSpatialTfSamplerForward( _handle, desc, alpha_p, spatialtf_ctx.xdesc,
        input_data, grid_data, beta_p, spatialtf_ctx.ydesc, out_data );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );
        return -1;
    }

    spatialtf_context_destroy( &spatialtf_ctx );
    cuda_exit( gpu_ctx->ctx );

    return 0;
}
