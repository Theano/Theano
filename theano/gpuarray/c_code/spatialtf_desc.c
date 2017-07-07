#section support_code_apply

int APPLY_SPECIFIC(spatialtf_desc)(npy_int32 dim_nimages,
                                   npy_int32 dim_nchannels,
                                   npy_int32 dim_height,
                                   npy_int32 dim_width,
                                   cudnnSpatialTransformerDescriptor_t * desc,
                                   PARAMS_TYPE * params)
{
    cudnnStatus_t err;

    const int nimages = (int) dim_nimages;
    const int nchannels = (int) dim_nchannels;
    const int height = (int) dim_height;
    const int width = (int) dim_width;

    if ( nimages == 0 || nchannels == 0 || height == 0 || width == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError, "Invalid grid dimensions" );
        return -1;
    }

    // num_images, num_channels, height, width
    const int out_tensor_dims[4] = { nimages, nchannels, height, width };

    err = cudnnCreateSpatialTransformerDescriptor( desc );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "Failed to allocate spatial transformer descriptor: %s",
            cudnnGetErrorString( err ) );
        return -1;
    }

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so it is not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor( *desc, CUDNN_SAMPLER_BILINEAR,
        params->dtype, 4, out_tensor_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "Failed to initialize spatial transformer descriptor: %s",
            cudnnGetErrorString( err ) );
        return -1;
    }

    return 0;
}
