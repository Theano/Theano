/** C Implementation (with NumPy back-end) of BLAS functions used in Theano.
 * Used instead of BLAS when Theano flag ``blas.ldflags`` is empty.
 * This file contains some useful header code not templated.
 * File alt_blas_template.c currently contains template code for:
 * - [sd]gemm_
 * - [sd]gemv_
 * - [sd]dot_
 **/

#define alt_fatal_error(message) { if (PyErr_Occurred()) PyErr_Print(); if(message != NULL) fprintf(stderr, message); exit(-1); }

#define alt_trans_to_bool(trans)  (*trans != 'N' && *trans != 'n')

/**Template code for BLAS functions follows in file alt_blas_template.c
 * (as Python string to be used with old formatting).
 * PARAMETERS:
 * float_type: "float" or "double".
 * float_size: 4 for float32 (sgemm_), 8 for float64 (dgemm_).
 * npy_float: "NPY_FLOAT32" or "NPY_FLOAT64".
 * precision: "s" for single, "d" for double.
 * See blas_headers.py for current use.**/
