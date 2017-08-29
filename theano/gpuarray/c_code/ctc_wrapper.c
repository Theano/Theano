#section init_code

setup_ext_cuda();

#section support_code

typedef struct ctc_context {
    struct ctcOptions options;
    gpudata * workspace;
    int * input_lengths;
    int * flat_labels;
    int * label_lengths;
} ctc_context_t;

void ctc_context_init(ctc_context_t * context, PyGpuContextObject * gpu_context)
{
    memset(&(context->options), 0, sizeof(struct ctcOptions));
    context->options.loc = CTC_GPU;

    // Get CUDA function pointer to obtain stream
    CUstream (*getstream_func_ptr)(void *) = (CUstream (*)(void *)) gpuarray_get_extension( "cuda_get_stream" );
    context->options.stream = getstream_func_ptr(gpu_context->ctx);

    context->workspace = NULL;
    context->input_lengths = NULL;
    context->flat_labels = NULL;
    context->label_lengths = NULL;
}

void ctc_context_destroy(ctc_context_t * context)
{
    gpudata_release( context->workspace );

    free( context->input_lengths );

    free( context->flat_labels );

    free( context->label_lengths );
}

int ctc_check_result(ctcStatus_t retcode, const char * msg)
{
    if( CTC_STATUS_SUCCESS != retcode )
    {
        // Get error message from underlying library
        const char * ctc_msg = ctcGetStatusString( retcode );

        PyErr_Format( PyExc_RuntimeError,
                      "GpuConnectionistTemporalClassification: %s CTC error: %s",
                      msg,
                      ctc_msg );
        return 1;
    }
    return 0;
}

void create_contiguous_input_lengths( PyArrayObject * input_lengths_arr,
    int ** input_lengths )
{
    npy_int num_elements = PyArray_DIMS( input_lengths_arr )[0];

    *input_lengths = (int *) malloc( num_elements * sizeof(int) );

    if ( NULL == (*input_lengths) )
        return;

    for( npy_int elem_idx = 0; elem_idx < num_elements; ++elem_idx )
    {
        (*input_lengths)[elem_idx] = *( (npy_int *) PyArray_GETPTR1( input_lengths_arr, elem_idx ) );
    }
}

void create_flat_labels( PyArrayObject * label_matrix, int ** flat_labels,
    int ** label_lengths )
{
    npy_int rows = PyArray_DIMS( label_matrix )[0];
    npy_int cols = PyArray_DIMS( label_matrix )[1];

    *flat_labels = (int *) calloc( rows * cols, sizeof(int) );
    if ( NULL == (*flat_labels) )
        return;

    *label_lengths = (int *) calloc( rows, sizeof(int) );
    if ( NULL == (*label_lengths) )
    {
        free( *flat_labels );
        *flat_labels = NULL;
        return;
    }

    npy_int label_index = 0;
    for( npy_int row_idx = 0; row_idx < rows; ++row_idx )
    {
        npy_int label_length = 0;
        for( npy_int col_idx = 0; col_idx < cols; ++col_idx )
        {
            npy_int label = *( (npy_int *) PyArray_GETPTR2( label_matrix, row_idx, col_idx ) );
            if ( label >= 0 )  // negative values are assumed to be padding
            {
                (*flat_labels)[ label_index++ ] = label;
                ++label_length;
            }
        }
        (*label_lengths)[ row_idx ] = label_length;
    }
}

#section support_code_apply

int APPLY_SPECIFIC(ctc_cost_gpu)(PyGpuArrayObject   *  in_activations,
                                 PyArrayObject      *  in_labels,
                                 PyArrayObject      *  in_input_lengths,
                                 PyGpuArrayObject   ** out_costs,
                                 PyGpuArrayObject   ** out_gradients,
                                 PyGpuContextObject *  gpu_context)
{
    ctc_context_t ctc_object;
    ctc_context_t * context = &ctc_object;

    size_t gpu_workspace_size;
    int ctc_error = 0;

    const size_t num_activations = PyGpuArray_DIMS( in_activations )[0];
    const size_t minibatch_size = PyGpuArray_DIMS( in_activations )[1];
    const size_t alphabet_size = PyGpuArray_DIMS( in_activations )[2];
    const size_t cost_size = minibatch_size;

    const size_t grad_dims[3] = { num_activations, minibatch_size, alphabet_size };

    float * costs = NULL,
          * activations = NULL,
          * gradients = NULL;

    cuda_enter( gpu_context->ctx );

    ctc_context_init( context, gpu_context );

    switch (in_activations->ga.typecode)
    {
    case GA_FLOAT:
        activations = (float *) PyGpuArray_DEV_DATA( in_activations );
        break;
    default:
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        PyErr_SetString( PyExc_TypeError,
            "GpuConnectionistTemporalClassification: Unsupported type for activations." );

        return 1;
    }

    create_contiguous_input_lengths( in_input_lengths, &(context->input_lengths) );

    if ( NULL == context->input_lengths )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        PyErr_Format( PyExc_MemoryError,
            "GpuConnectionistTemporalClassification: Could not allocate memory for input lengths." );
        return 1;
    }

    // flatten labels to conform with library memory layout
    create_flat_labels( in_labels, &(context->flat_labels), &(context->label_lengths) );

    if ( ( NULL == context->label_lengths ) || ( NULL == context->flat_labels ) )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        PyErr_Format( PyExc_MemoryError,
            "GpuConnectionistTemporalClassification: Could not allocate memory for labels and their lengths." );
        return 1;
    }

    if ( theano_prep_output( out_costs, 1, &cost_size, in_activations->ga.typecode,
                             GA_C_ORDER, gpu_context ) != 0 )
    {
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        return 1;
    }

    GpuArray_memset( &((*out_costs)->ga), 0 );

    costs = (float *) PyGpuArray_DEV_DATA( *out_costs );

    if ( NULL != out_gradients )  // if gradient computation is not disabled
    {
        if ( theano_prep_output( out_gradients, 3, grad_dims, in_activations->ga.typecode,
                                 GA_C_ORDER, gpu_context ) != 0 )
        {
            ctc_context_destroy( context );

            cuda_exit( gpu_context->ctx );

            return 1;
        }

        GpuArray_memset( &((*out_gradients)->ga), 0 );

        gradients = (float *) PyGpuArray_DEV_DATA( *out_gradients );
    }

    ctc_error = ctc_check_result( get_workspace_size( context->label_lengths,
        context->input_lengths, alphabet_size, minibatch_size, context->options,
        &gpu_workspace_size ),
        "Failed to obtain CTC workspace size." );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        return 1;
    }

    context->workspace = gpudata_alloc( gpu_context->ctx, gpu_workspace_size, NULL, 0, NULL );

    if ( NULL == context->workspace )
    {
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        PyErr_Format( PyExc_MemoryError,
            "GpuConnectionistTemporalClassification: Failed to allocate memory for CTC workspace." );
        return 1;
    }

    cuda_wait( in_activations->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_wait( (*out_costs)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    if ( out_gradients != NULL )
        cuda_wait( (*out_gradients)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    ctc_error = ctc_check_result( compute_ctc_loss( activations, gradients,
        context->flat_labels, context->label_lengths, context->input_lengths,
        alphabet_size, minibatch_size, costs, *(void **)context->workspace,
        context->options ), "Failed to compute CTC loss function." );

    cuda_record( in_activations->ga.data, GPUARRAY_CUDA_WAIT_READ );
    cuda_record( (*out_costs)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
    if ( out_gradients != NULL )
        cuda_record( (*out_gradients)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        ctc_context_destroy( context );

        cuda_exit( gpu_context->ctx );

        return 1;
    }

    ctc_context_destroy( context );
    cuda_exit( gpu_context->ctx );

    return 0;
}
