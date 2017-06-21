#section support_code_apply

int APPLY_SPECIFIC(spatialtf_desc)(cudnnSpatialTransformerDescriptor_t * desc,
                                   PARAMS_TYPE * params)
{
    cudnnStatus_t err;
    
    // width, height, num_feature_maps, num_images
    const int out_tensor_dims[4] = { params->dim0, params->dim1, params->dim2, params->dim3 };

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
        params->precision, params->nb_dims, out_tensor_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError, 
            "Failed to initialize spatial transformer descriptor: %s",
            cudnnGetErrorString( err ) );
        return -1;
    }

    return 0;
}
