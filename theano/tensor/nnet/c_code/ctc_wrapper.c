#section support_code

typedef struct ctc_context {
    struct ctcOptions options;
    void * workspace;
    int * input_lengths;
    int * flat_labels;
    int * label_lengths;
} ctc_context_t;

void ctc_context_init(ctc_context_t * context)
{
    struct ctcOptions * options = &(context->options);
    memset(options, 0, sizeof(struct ctcOptions));
    options->loc = CTC_CPU;
#if defined(_OPENMP)
    options->num_threads = omp_get_num_threads();
#else
    options->num_threads = 1;
#endif
    context->workspace = NULL;
    context->input_lengths = NULL;
    context->flat_labels = NULL;
    context->label_lengths = NULL;
}

void ctc_context_destroy(ctc_context_t * context)
{
    free( context->workspace );

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
                      "ConnectionistTemporalClassification: %s CTC error: %s",
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

    *input_lengths = (int *) calloc( num_elements, sizeof(int) );

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

int APPLY_SPECIFIC(ctc_cost_cpu)(PyArrayObject *  in_activations,
                                 PyArrayObject *  in_labels,
                                 PyArrayObject *  in_input_lengths,
                                 PyArrayObject ** out_costs,
                                 PyArrayObject ** out_gradients)
{
    ctc_context_t ctc_object;
    ctc_context_t * context = &ctc_object;
    ctc_context_init( context );

    if ( !PyArray_IS_C_CONTIGUOUS( in_activations ) )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "ConnectionistTemporalClassification: activations array must be C-contiguous." );
        return 1;
    }

    npy_float32 * activations = (npy_float32 *) PyArray_DATA( in_activations );

    create_contiguous_input_lengths( in_input_lengths, &(context->input_lengths) );

    if ( NULL == context->input_lengths )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        PyErr_Format( PyExc_MemoryError,
            "ConnectionistTemporalClassification: Could not allocate memory for input lengths" );
        return 1;
    }

    // flatten labels to conform with library memory layout
    create_flat_labels( in_labels, &(context->flat_labels), &(context->label_lengths) );

    if ( ( NULL == context->label_lengths ) || ( NULL == context->flat_labels ) )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        PyErr_Format( PyExc_MemoryError,
            "ConnectionistTemporalClassification: Could not allocate memory for labels and their lengths" );
        return 1;
    }

    npy_int minibatch_size = PyArray_DIMS( in_activations )[1];
    npy_int alphabet_size = PyArray_DIMS( in_activations )[2];

    npy_float32 * costs = NULL;
    npy_intp cost_size = minibatch_size;

    if ( (*out_costs) == NULL ||                       // Symbolic variable has no memory backing
         PyArray_NDIM( *out_costs ) != 1 ||            // or, matrix has the wrong size
         PyArray_DIMS( *out_costs )[0] != cost_size )
    {
        Py_XDECREF( *out_costs );
        // Allocate new matrix
        *out_costs = (PyArrayObject *) PyArray_ZEROS( 1, &cost_size, NPY_FLOAT32, 0 );

        if ( NULL == (*out_costs) )
        {
            // Destroy previous CTC context before returning exception
            ctc_context_destroy( context );

            PyErr_Format( PyExc_MemoryError,
                "ConnectionistTemporalClassification: Could not allocate memory for CTC costs" );
            return 1;
        }
    }

    costs = (npy_float32 *) PyArray_DATA( *out_costs );

    npy_float32 * gradients = NULL;

    if ( NULL != out_gradients )  // If gradient computation is not disabled
    {
        if ( NULL == (*out_gradients) ||  // Symbolic variable has no real backing
            PyArray_NDIM( *out_gradients ) != 3 ||
            PyArray_DIMS( *out_gradients )[0] != PyArray_DIMS( in_activations )[0] ||
            PyArray_DIMS( *out_gradients )[1] != PyArray_DIMS( in_activations )[1] ||
            PyArray_DIMS( *out_gradients )[2] != PyArray_DIMS( in_activations )[2] )
        {
            // Existing matrix is the wrong size. Make a new one.
            // Decrement ref counter to existing array
            Py_XDECREF( *out_gradients );
            // Allocate new array
            *out_gradients = (PyArrayObject *) PyArray_ZEROS(3, PyArray_DIMS( in_activations ),
                NPY_FLOAT32, 0);

            if ( NULL == (*out_gradients) )
            {
                // Destroy previous CTC context before returning exception
                ctc_context_destroy( context );

                PyErr_Format( PyExc_MemoryError,
                    "ConnectionistTemporalClassification: Could not allocate memory for CTC gradients!" );
                return 1;
            }
        }
        gradients = (npy_float32 *) PyArray_DATA( *out_gradients );
    }

    size_t cpu_workspace_size;
    int ctc_error;

    ctc_error = ctc_check_result( get_workspace_size( context->label_lengths,
        context->input_lengths, alphabet_size, minibatch_size, context->options,
        &cpu_workspace_size ),
        "Failed to obtain CTC workspace size." );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        return 1;
    }

    context->workspace = malloc( cpu_workspace_size );

    if ( NULL == context->workspace )
    {
        // Destroy previous CTC context before returning exception
        ctc_context_destroy( context );

        PyErr_Format( PyExc_MemoryError,
            "ConnectionistTemporalClassification: Failed to allocate memory for CTC workspace." );
        return 1;
    }

    ctc_error = ctc_check_result( compute_ctc_loss( activations, gradients,
        context->flat_labels, context->label_lengths, context->input_lengths,
        alphabet_size, minibatch_size, costs, context->workspace,
        context->options ), "Failed to compute CTC loss function." );

    if ( ctc_error )  // Exception is set by ctc_check_result, return error here
    {
        ctc_context_destroy( context );

        return 1;
    }

    ctc_context_destroy( context );

    return 0;
}
