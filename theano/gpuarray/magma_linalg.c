#section support_code_struct

float *APPLY_SPECIFIC(dwork);
magma_int_t *APPLY_SPECIFIC(piv);

#section init_code_struct

APPLY_SPECIFIC(dwork) = NULL;
APPLY_SPECIFIC(piv) = NULL;

#section cleanup_code_struct

if (APPLY_SPECIFIC(dwork) != NULL) {magma_free(APPLY_SPECIFIC(dwork));}
if (APPLY_SPECIFIC(piv) != NULL) {magma_free(APPLY_SPECIFIC(piv));}

#section support_code_struct

int APPLY_SPECIFIC(magma_matrix_inv)(PyGpuArrayObject *A,
                                     PyGpuArrayObject **_A_inv,
                                     PyGpuContextObject *ctx) {
  PyGpuArrayObject *A_inv = *_A_inv;

  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaMatrixInverse: requires data to be C-contiguous");
    return 1;
  }
  if (PyGpuArray_NDIM(A) != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaMatrixInverse: matrix rank error");
    return 1;
  }
  const size_t *x_dims = PyGpuArray_DIMS(A);
  if (x_dims[0] != x_dims[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaMatrixInverse: matrix is not square");
    return 1;
  }
#ifdef INPLACE
  Py_XDECREF(out);
  A_inv = A;
  Py_INCREF(out);
#else
  A_inv = theano_try_copy(A_inv, A);
  if (A_inv == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaMatrixInverse: failed to allocate memory");
    return 1;
  }
#endif
  {
    // magma matrix inverse
    magma_init();

    magma_int_t ldwork, info;
    magma_int_t N = x_dims[0];

    ldwork = N * magma_get_sgetri_nb(N);
    if (magma_smalloc(&APPLY_SPECIFIC(dwork), ldwork)) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "GpuMagmaMatrixInverse: failed to allocate magma working memory");
      return 1;
    }

    APPLY_SPECIFIC(piv) = (magma_int_t *)malloc(N * sizeof(magma_int_t));
    float *A_ptr = (float *) ((void **)A_inv->ga.data)[0];
    magma_sgetri_gpu(N, A_ptr, N, APPLY_SPECIFIC(piv), APPLY_SPECIFIC(dwork), ldwork, &info);
    if (info != 0) {
      PyErr_Format(
          PyExc_RuntimeError,
          "GpuMagmaMatrixInverse: magma_sgetri_gpu returned error %d", info);
      return 1;
    }
    magma_finalize();
  }
  *_A_inv = A_inv;
  return 0;
}
