#section support_code_apply

int APPLY_SPECIFIC(dnn_sptf_desc)(PyArrayObject * out_dims,
                                  cudnnSpatialTransformerDescriptor_t * desc,
                                  PARAMS_TYPE * params)
{
    cudnnStatus_t err;

    const int nimages = (int) *((npy_int64 *) PyArray_GETPTR1(out_dims, 0));
    const int nchannels = (int) *((npy_int64 *) PyArray_GETPTR1(out_dims, 1));
    const int height = (int) *((npy_int64 *) PyArray_GETPTR1(out_dims, 2));
    const int width = (int) *((npy_int64 *) PyArray_GETPTR1(out_dims, 3));

    if ( nimages == 0 || nchannels == 0 || height == 0 || width == 0 )
    {
        PyErr_SetString( PyExc_RuntimeError,
                         "GpuDnnTransformerDesc: invalid grid dimensions" );
        return 1;
    }

    // num_images, num_channels, height, width
    const int out_tensor_dims[4] = { nimages, nchannels, height, width };

    err = cudnnCreateSpatialTransformerDescriptor( desc );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerDesc: could not allocate descriptor: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    // Currently, only the bilinear sampler is supported by cuDNN,
    // so it is not available as a parameter
    err = cudnnSetSpatialTransformerNdDescriptor( *desc, CUDNN_SAMPLER_BILINEAR,
        params->precision, 4, out_tensor_dims );
    if ( CUDNN_STATUS_SUCCESS != err )
    {
        PyErr_Format( PyExc_MemoryError,
            "GpuDnnTransformerDesc: could not initialize descriptor: %s",
            cudnnGetErrorString( err ) );
        return 1;
    }

    return 0;
}
