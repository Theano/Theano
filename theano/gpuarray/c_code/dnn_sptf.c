#section support_code

typedef struct __spatialtf_context {
    PyGpuArrayObject * grid;
    cudnnTensorDescriptor_t xdesc;
    cudnnTensorDescriptor_t ydesc;
} spatialtf_context_t;

void spatialtf_context_init( spatialtf_context_t * ctx )
{
    if ( ctx == NULL )
        return;

    ctx->grid = NULL;
    ctx->xdesc = NULL;
    ctx->ydesc = NULL;
}

void spatialtf_context_destroy( spatialtf_context_t * ctx )
{
    Py_XDECREF( ctx->grid );

    if ( NULL != ctx->xdesc )
        cudnnDestroyTensorDescriptor( ctx->xdesc );

    if ( NULL != ctx->ydesc )
        cudnnDestroyTensorDescriptor( ctx->ydesc );
}

#section support_code_struct

int
dnn_sptf(PyGpuArrayObject * input,
         PyGpuArrayObject * theta,
         PyArrayObject * grid_dims,
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
    cudnnTensorFormat_t tf = CUDNN_TENSOR_NCHW;
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
                         "GpuDnnTransformer: unsupported type in spatial transformer sampler" );
        return -1;
    }

    if ( ! GpuArray_IS_C_CONTIGUOUS( &(input->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "GpuDnnTransformer: input data is not C-contiguous" );
        return -1;
    }

    if ( theta->ga.typecode != GA_FLOAT &&
         theta->ga.typecode != GA_DOUBLE &&
         theta->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError, "GpuDnnTransformer: unsupported data type for theta" );
        return -1;
    }
    else if ( PyGpuArray_NDIM( theta ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: theta must have three dimensions!" );
        return -1;
    }
    else if ( PyGpuArray_DIM( theta, 1 ) != 2 && PyGpuArray_DIM( theta, 2 ) != 3 )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: incorrect dimensions for theta, expected (%d, %d, %d), got (%d, %d, %d)",
                      PyGpuArray_DIMS( theta )[0], 2, 3, PyGpuArray_DIMS( theta )[0],
                      PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
        return -1;
    }
    else if ( ! GpuArray_IS_C_CONTIGUOUS( &(theta->ga) ) )
    {
        PyErr_SetString( PyExc_MemoryError,
                         "GpuDnnTransformer: theta is not C-contiguous" );
        return -1;
    }

    if ( PyArray_NDIM( grid_dims ) != 1 || PyArray_SIZE( grid_dims ) != 4 )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformer: grid_dims must have 4 elements." );
        return -1;
    }

    // Obtain grid dimensions
    const int num_images = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 0 ) );
    const int num_channels = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 1 ) );
    const int height = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 2 ) );
    const int width = (int) *( (npy_int *) PyArray_GETPTR1( grid_dims, 3 ) );
    const size_t gpu_grid_dims[4] = { num_images, height, width, 2 };

    if ( width == 0 || height == 0 || num_images == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformer: grid_dims has a dimension with value zero" );
        return -1;
    }

    spatialtf_context_init( &spatialtf_ctx );

    cuda_enter( gpu_ctx->ctx );

    spatialtf_ctx.grid = pygpu_empty(4, &(gpu_grid_dims[0]), input->ga.typecode, GA_C_ORDER,
        gpu_ctx, Py_None);

    if ( spatialtf_ctx.grid == NULL )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformer: could not allocate memory for grid of coordinates" );
        return -1;
    }

    err = cudnnCreateTensorDescriptor( &(spatialtf_ctx.xdesc) );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: could not create xdesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    // In the input tensor, we must use its width and height, instead
    // of the grid's width and height. The number of images and channels
    // should be the same as the grid dimensions
    const int input_num_images = (int) PyGpuArray_DIM( input, 0 );
    const int input_num_channels = (int) PyGpuArray_DIM( input, 1 );
    const int input_height = (int) PyGpuArray_DIM( input, 2 );
    const int input_width = (int) PyGpuArray_DIM( input, 3 );

    if ( input_num_images != num_images || input_num_channels != num_channels )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: expected input to have %d inputs, got %d inputs.",
                      num_images, input_num_images );
        return -1;
    }

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.xdesc, tf, dt, num_images,
        input_num_channels, input_height, input_width );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: failed to initialize xdesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    err = cudnnCreateTensorDescriptor( &(spatialtf_ctx.ydesc) );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: failed to create ydesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    err = cudnnSetTensor4dDescriptor( spatialtf_ctx.ydesc, tf, dt, num_images,
        input_num_channels, height, width );

    if ( err != CUDNN_STATUS_SUCCESS )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: failed to initialize ydesc: %s",
                      cudnnGetErrorString(err) );
        return -1;
    }

    const size_t out_dims[4] = { num_images, input_num_channels, height, width };

    if ( theano_prep_output( output, 4, out_dims, input->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

        PyErr_SetString( PyExc_MemoryError,
                         "GpuDnnTransformer: could not allocate memory for grid sampler" );
        return -1;
    }

    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfGridGeneratorForward( _handle, desc, PyGpuArray_DEV_DATA( theta ),
        PyGpuArray_DEV_DATA( spatialtf_ctx.grid ) );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_RuntimeError,
                      "GpuDnnTransformer: failed to create grid of coordinates: %s",
                      cudnnGetErrorString( err ) );
        return -1;
    }

    err = cudnnSpatialTfSamplerForward( _handle, desc, alpha_p, spatialtf_ctx.xdesc,
        PyGpuArray_DEV_DATA( input ), PyGpuArray_DEV_DATA( spatialtf_ctx.grid ),
        beta_p, spatialtf_ctx.ydesc, PyGpuArray_DEV_DATA( *output ) );

    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformer: failed to create grid sampler" );
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );
        return -1;
    }

    spatialtf_context_destroy( &spatialtf_ctx );
    cuda_exit( gpu_ctx->ctx );

    return 0;
}
