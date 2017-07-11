#section support_code_struct

int
APPLY_SPECIFIC(dnn_sptf_gt)(PyGpuArrayObject * dgrid,
                            cudnnSpatialTransformerDescriptor_t desc,
                            PyGpuArrayObject ** dtheta,
                            cudnnHandle_t _handle)
{
    PyGpuContextObject * gpu_ctx = dgrid->context;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;

    int num_images = (int) PyGpuArray_DIM( dgrid, 0 );

    const size_t dtheta_dims[3] = { num_images, 2, 3 };

    if ( theano_prep_output( dtheta, 3, &(dtheta_dims[0]), dgrid->ga.typecode,
                             GA_C_ORDER, gpu_ctx ) != 0 )
        return 1;

    cuda_enter( gpu_ctx->ctx );

    cuda_wait( dgrid->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*dtheta)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    err = cudnnSpatialTfGridGeneratorBackward( _handle, desc,
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
