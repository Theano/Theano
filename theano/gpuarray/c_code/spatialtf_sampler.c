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
spatialtf_sampler(PyGpuArrayObject *input,
                  PyGpuArrayObject *om,
                  PyGpuArrayObject *grid,
                  cudnnSpatialTransformerDescriptor_t desc,
                  double alpha, double beta,
                  PyGpuArrayObject **output,
                  cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = input->context;
    void * alpha_p;
    void * beta_p;
    float af = alpha, bf = beta;
    spatialtf_context_t spatialtf_ctx;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    switch (input->ga.typecode)
    {
    case GA_DOUBLE:
        alpha_p = (void *)&alpha;
        beta_p = (void *)&beta;
        break;
    case GA_FLOAT:
    case GA_HALF:
        alpha_p = (void *)&af;
        beta_p = (void *)&bf;
        break;
    default:
        PyErr_SetString(PyExc_TypeError,
                        "Unsupported type in spatial transformer sampler");
        return -1;
    }

    if ( grid->ga.typecode != GA_FLOAT &&
         grid->ga.typecode != GA_DOUBLE &&
         grid->ga.typecode != GA_HALF )
    {
        PyErr_SetString( PyExc_TypeError, "Unsupported data type for grid" );
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

    if ( theano_prep_output( output, PyGpuArray_NDIM(om), PyGpuArray_DIMS(om), grid->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
    {
        spatialtf_context_destroy( &spatialtf_ctx );
        cuda_exit( gpu_ctx->ctx );

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

    err = cudnnSpatialTfSamplerForward( _handle, desc, alpha_p, spatialtf_ctx.xdesc,
        PyGpuArray_DEV_DATA( input ), PyGpuArray_DEV_DATA( grid ), beta_p,
        spatialtf_ctx.ydesc, PyGpuArray_DEV_DATA( *output ) );

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
