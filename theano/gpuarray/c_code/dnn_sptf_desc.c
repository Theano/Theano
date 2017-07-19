#section support_code_apply

int APPLY_SPECIFIC(dnn_sptf_desc)(PyArrayObject * dims,
                                  cudnnSpatialTransformerDescriptor_t * desc,
                                  PARAMS_TYPE * params)
{
    cudnnStatus_t err;

    const int nimages = *((int *) PyArray_GETPTR1(dims, 0));
    const int nchannels = *((int *) PyArray_GETPTR1(dims, 1));
    const int height = *((int *) PyArray_GETPTR1(dims, 2));
    const int width = *((int *) PyArray_GETPTR1(dims, 3));

    if ( nimages == 0 || nchannels == 0 || height == 0 || width == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformerDescriptor: invalid grid dimensions" );
        return 1;
    }

    // num_images, num_channels, height, width
    const int out_tensor_dims[4] = { nimages, nchannels, height, width };

    err = cudnnCreateSpatialTransformerDescriptor( desc );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerDescriptor: could not allocate descriptor: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so it is not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor( *desc, CUDNN_SAMPLER_BILINEAR,
        params->dtype, 4, out_tensor_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerDescriptor: could not initialize descriptor: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}
