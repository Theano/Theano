#section support_code

void create_contiguous_input_lengths( PyArrayObject * input_lengths_arr,
    int ** input_lengths )
{
    int num_elements = PyArray_DIMS( input_lengths_arr )[0];

    *input_lengths = (int *) malloc( num_elements * sizeof(int) );

    if ( NULL == (*input_lengths) )
        return;

    for( int i = 0; i < num_elements; ++i )
    {
        (*input_lengths)[i] = *( (int *) PyArray_GETPTR1( input_lengths_arr, i ) );
    }
}

void create_flat_labels( PyArrayObject * label_matrix, int ** flat_labels, 
    int ** label_lengths )
{
    int rows = PyArray_DIMS( label_matrix )[0];
    int cols = PyArray_DIMS( label_matrix )[1];

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

    int label_index = 0;
    for( int i = 0; i < rows; ++i )
    {
        int label_length = 0;
        for( int j = 0; j < cols; ++j )
        {
            int label = *( (int *) PyArray_GETPTR2( label_matrix, i, j ) );
            if ( label >= 0 )  // negative values are assumed to be padding
            {
                (*flat_labels)[ label_index++ ] = label;
                ++label_length;
            }
        }
        (*label_lengths)[ i ] = label_length;
    }
}

#section support_code_apply

int APPLY_SPECIFIC(ctc_cost_cpu)(PyArrayObject *  in_activations,
                                 PyArrayObject *  in_labels,
                                 PyArrayObject *  in_input_lengths,
                                 PyArrayObject ** out_costs,
                                 PyArrayObject ** out_gradients)
{
    // setup CTC computation parameters
    ctcOptions ctc_options;
    memset( &ctc_options, 0, sizeof(ctcOptions) );
    ctc_options.loc = CTC_CPU;
    ctc_options.num_threads = 1;

    npy_float32 * activations = NULL;
    PyArrayObject * activations_copy = NULL;

    if ( PyArray_IS_C_CONTIGUOUS( in_activations ) )
    {
        activations = (npy_float32 *) PyArray_DATA( in_activations );
    }
    else
    {
        activations_copy = PyArray_GETCONTIGUOUS( in_activations );
        if ( NULL != activations_copy )
        {
            activations = (npy_float32 *) PyArray_DATA( activations_copy );
        }
        else
        {
            PyErr_Format( PyExc_ValueError,
                          "Could not create a contiguous copy of activations array." );
            return 1;            
        }
    }

    int * input_lengths = NULL,
        * flat_labels = NULL,
        * label_lengths = NULL;

    create_contiguous_input_lengths( in_input_lengths, &input_lengths );

    if ( NULL == input_lengths )
    {
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for input lengths" );
        return 1;        
    }

    // flatten labels to conform with library memory layout
    create_flat_labels( in_labels, &flat_labels, &label_lengths );

    if ( ( NULL == label_lengths ) || ( NULL == flat_labels ) )
    {
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for labels and their lengths" );
        return 1;
    }

    npy_int minibatch_size = PyArray_DIMS( in_activations )[1];
    npy_int alphabet_size = PyArray_DIMS( in_activations )[2];

    void * ctc_cpu_workspace = NULL;

    npy_float32 * costs = NULL;
    npy_intp cost_size = minibatch_size;

    if ( NULL == (*out_costs) )
    {
        // Symbolic variable has no memory backing, so we create one
        *out_costs = (PyArrayObject *) PyArray_ZEROS( 1, &cost_size, NPY_FLOAT32, 0 );
    }
    else if ( PyArray_NDIM( *out_costs ) != 1 ||
              PyArray_DIMS( *out_costs )[0] != cost_size )  // matrix has the wrong size
    {
        Py_XDECREF( *out_costs ); 
        // Allocate new matrix
        *out_costs = (PyArrayObject *) PyArray_ZEROS( 1, &cost_size, NPY_FLOAT32, 0 );
    }

    if ( NULL == (*out_costs) )
    {
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for CTC costs" );
        return 1;
    }
    
    costs = (npy_float32 *) PyArray_DATA( *out_costs );

    if ( NULL == (*out_gradients) )
    {
        // Symbolic variable has no real backing, so create one.
        *out_gradients = (PyArrayObject*) PyArray_ZEROS( 3, PyArray_DIMS( in_activations ),
            NPY_FLOAT32, 0 );
    }
    else if ( PyArray_NDIM( *out_gradients ) != 3 
        || PyArray_DIMS( *out_gradients )[0] != PyArray_DIMS( in_activations )[0]
        || PyArray_DIMS( *out_gradients )[1] != PyArray_DIMS( in_activations )[1]
        || PyArray_DIMS( *out_gradients )[2] != PyArray_DIMS( in_activations )[2] )
    {
        // Existing matrix is the wrong size. Make a new one.
        // Decrement ref counter to existing array
        Py_XDECREF( *out_gradients ); 
        // Allocate new array
        *out_gradients = (PyArrayObject *) PyArray_ZEROS(3, PyArray_DIMS( in_activations ),
            NPY_FLOAT32, 0);
    }

    if ( NULL == (*out_gradients) )
    {
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for CTC gradients!" );
        return 1;        
    }

    npy_float32 * gradients = (npy_float32 *) PyArray_DATA( *out_gradients );

    ctcStatus_t status;
    size_t cpu_workspace_size;

    status = get_workspace_size( label_lengths, input_lengths, alphabet_size,
        minibatch_size, ctc_options, &cpu_workspace_size );

    if ( CTC_STATUS_SUCCESS != status )
    {
        PyErr_Format( PyExc_ValueError,
                      "Could not compute the CTC workspace size!" );
        return 1;    
    }

    ctc_cpu_workspace = malloc( cpu_workspace_size );


    status = compute_ctc_loss( activations, gradients, flat_labels,
        label_lengths, input_lengths, alphabet_size, minibatch_size,
        costs, ctc_cpu_workspace, ctc_options );

    if ( CTC_STATUS_SUCCESS != status )
    {
        PyErr_Format( PyExc_ValueError, "Failed to compute CTC loss!" );
        return 1;         
    }

    if ( NULL != activations_copy )
    {
        Py_XDECREF( activations_copy );
    }

    free( input_lengths );
    free( flat_labels );
    free( label_lengths );

    free( ctc_cpu_workspace );

    return 0;
}
