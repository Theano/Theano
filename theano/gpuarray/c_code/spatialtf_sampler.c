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
    // Number of color channels (feature maps) is the innermost dimension
    cudnnTensorFormat_t tf = CUDNN_TENSOR_NHWC;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    if ( PyArray_DIM( grid_dimensions, 0 ) != 4 )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "grid_dimensions must have 4 dimensions" );
        return -1;
    }

    // Obtain grid dimensions
    const int num_images = (int) *( (npy_int *) PyArray_GETPTR1( grid_dimensions, 0 ) );
    const int height = (int) *( (npy_int *) PyArray_GETPTR1( grid_dimensions, 1 ) );
    const int width = (int) *( (npy_int *) PyArray_GETPTR1( grid_dimensions, 2 ) );
    const int num_channels = (int) *( (npy_int *) PyArray_GETPTR1( grid_dimensions, 3 ) );

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
        PyErr_SetString( PyExc_TypeError,
                         "Unsupported type in spatial transformer sampler" );
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

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.xdesc, tf, dt, num_images,
        num_channels, height, width );

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

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.ydesc, tf, dt, num_images,
        num_channels, height, width );

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
        // (num_images, height, width, num_channels )
        const size_t out_dims[4] = { num_images, height, width, num_channels };

        *output = pygpu_zeros( 4, &(out_dims[0]), input->ga.typecode, GA_C_ORDER,
            gpu_ctx, Py_None );

        if ( NULL == *output )
        {
            spatialtf_context_destroy( &spatialtf_ctx );
            cuda_exit( gpu_ctx->ctx );

            PyErr_SetString( PyExc_MemoryError,
                             "Could allocate memory for spatial transformer's grid sampler" );
            return -1;
        }
    }
    else
    {
        GpuArray_memset( &( (*output)->ga ), 0 );
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &(input->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "input data is not C-contiguous" );
        return -1;
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &(grid->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "grid data is not C-contiguous" );
        return -1;
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &((*output)->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "theta data is not C-contiguous" );
        return -1;
    }

    const void * input_data = PyGpuArray_DEV_DATA( input );
    const void * grid_data = PyGpuArray_DEV_DATA( grid );
    void * out_data = PyGpuArray_DEV_DATA( *output );

    err = cudnnSpatialTfSamplerForward( _handle, desc, alpha_p, spatialtf_ctx.xdesc,
        input_data, grid_data, beta_p, spatialtf_ctx.ydesc, out_data );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );
        return -1;
    }

    cuda_record(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record(grid->ga.data, GPUARRAY_CUDA_WAIT_READ);
    cuda_record((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);

    spatialtf_context_destroy( &spatialtf_ctx );
    cuda_exit( gpu_ctx->ctx );

    return 0;
}
