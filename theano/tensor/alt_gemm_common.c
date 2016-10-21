/** C Implementation of [sd]gemm_ based on NumPy
 * Used instead of blas when Theano config flag blas.ldflags is empty.
 * This file contains the common code for [sd]gemm_.
 * File alt_gemm_template.c contains template code for [sd]gemm_. **/

#define alt_fatal_error(message) { if(message != NULL) fprintf(stderr, message); exit(-1); }

#define alt_trans_to_bool(trans)  (*trans != 'N' && *trans != 'n')

/**Template code for [sd]gemm_ follows in file alt_gemm_template.c
 * (as Python string to be used with old formatting).
 * PARAMETERS:
 * float_type: "float" for sgemm_, "double" for dgemm_.
 * float_size: 4 for float32 (sgemm_), 8 for float64 (dgemm_).
 * npy_float: "NPY_FLOAT32" for sgemm_, "NPY_FLOAT64" for dgemm_.
 * name: "sgemm_" for sgemm_, "dgemm_" for dgemm_. 
 * See blas_headers.py for current use.**/
