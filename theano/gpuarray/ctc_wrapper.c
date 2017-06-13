#section support_code

typedef struct ctc_context {
    struct ctcOptions options;
    PyGpuArrayObject * workspace;
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
    Py_XDECREF( context->workspace );

    if ( NULL != context->input_lengths )
        free( context->input_lengths );

    if ( NULL != context->flat_labels )
        free( context->flat_labels );

    if ( NULL != context->label_lengths )
        free( context->label_lengths );
}

int ctc_check_result(ctcStatus_t retcode, const char * msg)
{
    if( CTC_STATUS_SUCCESS != retcode )
    {
        // Get error message from underlying library
        const char * ctc_msg = ctcGetStatusString( retcode );

        PyErr_Format( PyExc_RuntimeError,
                      "%s | CTC library error message: %s",
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

    *flat_labels = (int *) malloc( rows * cols * sizeof(int) );
    if ( NULL == (*flat_labels) )
        return;

    *label_lengths = (int *) malloc( rows * sizeof(int) );
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
    ctc_context_init( context, gpu_context );

    float * activations = NULL;
    switch (in_activations->ga.typecode)
    {
    case GA_FLOAT:
        activations = (float *) PyGpuArray_DEV_DATA( in_activations );
        break;
    default:
        ctc_context_destroy( context );
        PyErr_SetString(PyExc_TypeError, "Unsupported type for activations!");
        return 1;
    }

    create_contiguous_input_lengths( in_input_lengths, &(context->input_lengths) );

    if ( NULL == context->input_lengths )
    {
        PyErr_Format( PyExc_MemoryError,
            "Could not allocate storage for input lengths" );
        return 1;
    }

    // flatten labels to conform with library memory layout
    create_flat_labels( in_labels, &(context->flat_labels), &(context->label_lengths) );

    if ( ( NULL == context->label_lengths ) || ( NULL == context->flat_labels ) )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        PyErr_Format( PyExc_MemoryError,
            "Could not allocate storage for labels and their lengths" );
        return 1;
    }

    const size_t minibatch_size = PyGpuArray_DIMS( in_activations )[1];
    const size_t alphabet_size  = PyGpuArray_DIMS( in_activations )[2];

    float * costs = NULL;
    const size_t cost_size = minibatch_size;

    if (NULL == *out_costs ||  // symbolic variable has no real backing
        PyGpuArray_NDIM( *out_costs ) != 1 ||
        PyGpuArray_DIMS( *out_costs )[0] != cost_size)
    {
        Py_XDECREF( *out_costs );

        *out_costs = pygpu_empty( 1, &cost_size, GA_FLOAT, GA_C_ORDER,
            gpu_context, Py_None );

        if ( NULL == *out_costs )
        {
            // Destroy previous CTC context before returning exception
            ctc_context_destroy( context );

            PyErr_Format( PyExc_MemoryError,
                "Could not allocate storage for CTC costs");
            return 1;
        }
    }

    switch ( (*out_costs)->ga.typecode )
    {
    case GA_FLOAT:
        costs = (float *) PyGpuArray_DEV_DATA( *out_costs );
        break;
    default:
        ctc_context_destroy( context );
        PyErr_SetString(PyExc_TypeError, "Unsupported type for costs!");
        return 1;
    }

    float * gradients = NULL;

    if ( NULL != out_gradients )  // if gradient computation is not disabled
    {
        if ( NULL == *out_gradients ||
             PyGpuArray_NDIM( *out_gradients ) != 3 ||
             PyGpuArray_DIMS( *out_gradients )[0] != PyGpuArray_DIMS( in_activations )[0] ||
             PyGpuArray_DIMS( *out_gradients )[1] != PyGpuArray_DIMS( in_activations )[1] ||
             PyGpuArray_DIMS( *out_gradients )[2] != PyGpuArray_DIMS( in_activations )[2] )
        {
            Py_XDECREF( *out_gradients );

            const size_t * activation_dims = PyGpuArray_DIMS( in_activations );
            *out_gradients = pygpu_zeros( 3, activation_dims, GA_FLOAT, GA_C_ORDER,
                gpu_context, Py_None );

            if ( NULL == *out_gradients )
            {
                ctc_context_destroy( context );

                PyErr_Format( PyExc_MemoryError,
                    "Could not allocate storage for CTC gradients!" );
                return 1;
            }
        }
        else
        {
            GpuArray_memset( &((*out_gradients)->ga), 0 );
        }

        switch ( (*out_gradients)->ga.typecode )
        {
        case GA_FLOAT:
            gradients = (float *) PyGpuArray_DEV_DATA( *out_gradients );
            break;
        default:
            ctc_context_destroy( context );
            PyErr_SetString(PyExc_TypeError, "Unsupported type for gradients!");
            return 1;
        }
    }

    size_t gpu_workspace_size;
    int ctc_error = 0;

    ctc_error = ctc_check_result( get_workspace_size( context->label_lengths,
        context->input_lengths, alphabet_size, minibatch_size, context->options,
        &gpu_workspace_size ),
        "Failed to obtain CTC workspace size!" );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        return 1;
    }

    context->workspace = pygpu_empty(1, &gpu_workspace_size, GA_BYTE,
        GA_C_ORDER, gpu_context, Py_None );

    if ( NULL == context->workspace )
    {
        ctc_context_destroy( context );

        PyErr_Format( PyExc_MemoryError,
            "Failed to allocate memory for CTC workspace!" );
        return 1;
    }

    ctc_error = ctc_check_result( compute_ctc_loss( activations, gradients,
        context->flat_labels, context->label_lengths, context->input_lengths,
        alphabet_size, minibatch_size, costs, PyGpuArray_DEV_DATA(context->workspace),
        context->options ), "Failed to compute CTC loss function!" );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        ctc_context_destroy( context );
        return 1;
    }

    ctc_context_destroy( context );

    return 0;
}

int APPLY_SPECIFIC(ctc_cost_gpu_no_grad)(PyGpuArrayObject   *  in_activations,
                                         PyArrayObject      *  in_labels,
                                         PyArrayObject      *  in_input_lengths,
                                         PyGpuArrayObject   ** out_costs,
                                         PyGpuContextObject *  gpu_context)
{
    return APPLY_SPECIFIC(ctc_cost_gpu)(in_activations,
                                        in_labels,
                                        in_input_lengths,
                                        out_costs,
                                        NULL,
                                        gpu_context);
}
