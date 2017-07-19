#section init_code

setup_ext_cuda();

#section support_code_struct

int APPLY_SPECIFIC(magma_eigh)(PyGpuArrayObject *A_,
                               PyGpuArrayObject **D,
                               PyGpuArrayObject **V, // may be NULL
                               PARAMS_TYPE *params) {
  PyGpuArrayObject *A = NULL;
  magma_int_t N, liwork, *iwork_data = NULL;
  size_t d_dims[1], v_dims[2];
  magma_uplo_t uplo;
  magma_vec_t jobz;
  float *w_data = NULL, *wA_data = NULL, *work_data = NULL, lwork;
  int res = -1, info;

  if (A_->ga.typecode != GA_FLOAT) {
    PyErr_SetString(PyExc_TypeError,
                    "GpuMagmaEigh: Unsupported data type");
    return -1;
  }

  // This is early to match the exit() in the fail label.
  cuda_enter(params->context->ctx);

  if (!GpuArray_IS_C_CONTIGUOUS(&A_->ga)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaEigh: requires data to be C-contiguous");
    goto fail;
  }
  if (PyGpuArray_NDIM(A_) != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaEigh: matrix rank error");
    goto fail;
  }
  if (PyGpuArray_DIM(A_, 0) != PyGpuArray_DIM(A_, 1)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaEigh: matrix is not square");
    goto fail;
  }

  A = pygpu_copy(A_, GA_F_ORDER);
  if (A == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to change to column-major order");
    return -1;
  }

  // magma matrix eigen decomposition of a symmetric matrix
  N = PyGpuArray_DIM(A, 0);

  if (params->lower) {
    uplo = MagmaLower;
  } else {
    uplo = MagmaUpper;
  }
  if (params->compute_v) {
    jobz = MagmaVec;
  } else {
    jobz = MagmaNoVec;
  }

  if (MAGMA_SUCCESS != magma_smalloc_pinned(&w_data, N)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to allocate working memory");
    goto fail;
  }
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&wA_data, N * N)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to allocate working memory");
    goto fail;
  }

  // query for workspace size
  magma_ssyevd_gpu(jobz, uplo, N, NULL, N, NULL, NULL, N, &lwork,
                   -1, &liwork, -1, &info);

  if (MAGMA_SUCCESS != magma_smalloc_pinned(&work_data, (size_t)lwork)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to allocate working memory");
    goto fail;
  }
  if (MAGMA_SUCCESS != magma_imalloc_cpu(&iwork_data, liwork)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to allocate working memory");
    goto fail;
  }

  magma_ssyevd_gpu(jobz, uplo, N, (float *)PyGpuArray_DEV_DATA(A), N, w_data,
                   wA_data, N, work_data, (size_t)lwork, iwork_data, liwork,
                   &info);
  if (info > 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaEigh:  %d off-diagonal elements of an didn't converge to zero",
        info);
    goto fail;
  } else if (info < 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaEigh: magma_ssyevd_gpu argument %d has an illegal value", -info);
    goto fail;
  }

  d_dims[0] = N;
  if (theano_prep_output(D, 1, d_dims, A->ga.typecode, GA_C_ORDER, params->context) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaEigh: failed to allocate memory for the output");
    goto fail;
  }
  cudaMemcpy(PyGpuArray_DEV_DATA(*D), w_data, N * sizeof(float),
             cudaMemcpyDeviceToDevice);

  if (params->compute_v) {
    *V = theano_try_copy(*V, A);
    if (*V == NULL) {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMagmaEigh: failed to allocate memory for the output");
      goto fail;
    }
  }
  res = 0;
fail:
  if (w_data != NULL)
    magma_free_pinned(w_data);
  if (wA_data != NULL)
    magma_free_pinned(wA_data);
  if (work_data != NULL)
    magma_free_pinned(work_data);
  if (iwork_data != NULL)
    magma_free_cpu(iwork_data);
  Py_XDECREF(A);
  cuda_exit(params->context->ctx);
  return res;
}
