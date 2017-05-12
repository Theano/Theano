#section support_code


#section support_code_apply

int APPLY_SPECIFIC(ctc_cost_cpu)(PyArrayObject * activations,
                                 PyArrayObject * labels,
                                 PyArrayObject * input_lengths,
                                 PyArrayObject **costs,
                                 PyArrayObject **gradients)
{
    npy_int minibatch_size = PyArray_DIMS(activations)[1];

    npy_int cost_size = minibatch_size;

    if ( NULL != (*costs) )
    {
        Py_XDECREF( *costs ); 
    }
    
    *costs = (PyArrayObject*)PyArray_ZEROS( 1, (npy_intp *)&cost_size,
        NPY_FLOAT32, 0 );

    if ( NULL == (*costs) )
    {
        // FIXME: should it be 'FAIL;' ???
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for CTC costs" );
        return 1;
    }

    if ( PyArray_NDIM( *gradients ) != 3 
        || PyArray_DIMS( *gradients )[0] != PyArray_DIMS( activations )[0]
        || PyArray_DIMS( *gradients )[1] != PyArray_DIMS( activations )[1]
        || PyArray_DIMS( *gradients )[2] != PyArray_DIMS( activations )[2])
    {
        // Existing matrix is the wrong size. Make a new one.
        // Decrement ref counter to existing array
        Py_XDECREF( *gradients ); 
        // Allocate new array
        *gradients = (PyArrayObject*)PyArray_ZEROS(3, PyArray_DIMS( activations ),
            NPY_FLOAT32, 0);
    }

    // Symbolic variable has no real backing, so create one.
    *gradients = (PyArrayObject*)PyArray_ZEROS( 3, PyArray_DIMS(activations),
        NPY_FLOAT32, 0 );

    if ( NULL == (*gradients) )
    {
        // FIXME: should it be 'FAIL;' ???
        PyErr_Format( PyExc_ValueError,
                      "Could not allocate storage for CTC gradients" );
        return 1;        
    }

    return 0;
}
