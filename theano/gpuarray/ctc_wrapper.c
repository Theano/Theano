#section support_code

struct ctcOptions;

typedef struct ctc_context {
    struct ctcOptions options;
    void * workspace;
    int * input_lengths;
    int * flat_labels;
    int * label_lengths;
} ctc_context_t;

void ctc_context_init(ctc_context_t * context)
{
    memset(&(context->options), 0, sizeof(struct ctcOptions));
    options->loc = CTC_GPU;
    options->stream = NULL;

    context->workspace = NULL;
    context->input_lengths = NULL;
    context->flat_labels = NULL;
    context->label_lengths = NULL;
}

void ctc_context_destroy(ctc_context_t * context)
{
    if ( NULL != context->workspace )
        free( context->workspace );

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

#section support_code_struct

int APPLY_SPECIFIC(ctc_cost_gpu)(PyGpuArrayObject   *  in_activations,
                                 PyGpuArrayObject   *  in_labels,
                                 PyGpuArrayObject   *  in_input_lengths,
                                 PyGpuArrayObject   ** out_costs,
                                 PyGpuArrayObject   ** out_gradients,
                                 PyGpuContextObject *  ctx)
{
    ctc_context_t ctc_ctx;
    ctc_context_init( &ctc_ctx );

    if ( !PyArray_IS_C_CONTIGUOUS( in_activations ) )
    {
        PyErr_SetString( PyExc_RuntimeError,
            "activations array must be C-contiguous." );
        return 1;
    }

    npy_float32 * activations = (npy_float32 *) PyArray_DATA( in_activations );

    // TODO: flatten input_lengths to conform with underlying library memory layout

    // TODO: flatten labels to conform with underlying library memory layout


    const npy_int minibatch_size = PyArray_DIMS( in_activations )[1];
    const npy_int alphabet_size  = PyArray_DIMS( in_activations )[2];

    npy_float32 * costs = NULL;
    const npy_intp cost_size = minibatch_size;

    if (NULL == *out_costs ||  // symbolic variable has no real backing
        PyArray_NDIM( *out_costs ) != 1 ||
        PyArray_DIMS( *out_costs )[0] != cost_size)
    {
        PY_XDECREF( *out_costs );

        *out_costs = pygpu_zeros(1, cost_size, GA_FLOAT, GA_C_ORDER,
            ctx, Py_None);

        if ( NULL == *out_costs )
        {
            // Destroy previous CTC context before returning exception
            ctc_context_destroy( &ctc_ctx );

            PyErr_Format( PyExc_MemoryError,
                "Could not allocate storage for CTC costs");
            return 1;
        }
    }

    costs = (npy_float32 *) PyArray_DATA( *out_costs );

    npy_float32 * gradients = NULL;

    if ( NULL != out_gradients )  // if gradient computation is not disabled
    {
        if ( NULL == *out_gradients ||
             PyArray_NDIM( *out_gradients ) != 3 ||
             PyArray_DIMS( *out_gradients )[0] != PyArray_DIMS( in_activations )[0] ||
             PyArray_DIMS( *out_gradients )[1] != PyArray_DIMS( in_activations )[1] ||
             PyArray_DIMS( *out_gradients )[2] != PyArray_DIMS( in_activations )[2] )
        {
            Py_XDECREF( *out_gradients );

            *out_gradients = pygpu_zeros( 3, PyArray_DIMS( in_activations ), NPY_FLOAT32, 0 );

            if ( NULL == *out_gradients )
            {
                ctc_context_destroy( &ctc_ctx );

                PyErr_Format( PyExc_MemoryError,
                    "Could not allocate storage for CTC gradients!" );
                return 1;
            }
        }

        gradients = (npy_float32 *) PyArray_DATA( *out_gradients );
    }

    ctc_context_destroy( &ctc_ctx );

    return 0;
}

int APPLY_SPECIFIC(ctc_cost_gpu_no_grad)(PyGpuArrayObject   *  in_activations,
                                         PyGpuArrayObject   *  in_labels,
                                         PyGpuArrayObject   *  in_input_lengths,
                                         PyGpuArrayObject   ** out_costs,
                                         PyGpuContextObject *  ctx)
{
    return APPLY_SPECIFIC(ctc_cost_gpu)(in_activations,
                                        in_labels,
                                        in_input_lengths,
                                        out_costs,
                                        NULL,
                                        ctx);
}
