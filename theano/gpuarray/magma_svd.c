#section init_code

setup_ext_cuda();

#section support_code_struct

int APPLY_SPECIFIC(magma_svd)(PyGpuArrayObject *A, PyGpuArrayObject **U,
                              PyGpuArrayObject **S, PyGpuArrayObject **VT,
                              PyGpuContextObject *c) {
  magma_int_t M, N, K, ldu, ldv, M_U, N_VT, info;
  magma_vec_t jobu, jobv;
  size_t s_dims[1], u_dims[2], vt_dims[2];
  float *a_data = NULL, *s_data = NULL, *u_data = NULL, *vt_data = NULL,
        *work = NULL;
  float dummy[1];
  int res = -1, lwork;

  if (A->ga.typecode != GA_FLOAT) {
    PyErr_SetString(PyExc_TypeError,
                    "GpuMagmaMatrixInverse: Unsupported data type");
    return -1;
  }

  // This is early to match the exit() in the fail label.
  cuda_enter(c->ctx);
  magma_init();

  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaMatrixInverse: requires data to be C-contiguous");
    return 1;
  }
  if (PyGpuArray_NDIM(A) != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaMatrixInverse: matrix rank error");
    goto fail;
  }

  // magma matrix svd
  M = PyGpuArray_DIM(A, 0);
  N = PyGpuArray_DIM(A, 1);
  K = M < N ? M : N;

  if (MAGMA_SUCCESS !=  magma_smalloc_pinned(&a_data, M * N)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  cudaMemcpy(a_data, PyGpuArray_DEV_DATA(A), M * N * sizeof(float),
             cudaMemcpyDeviceToDevice);

  if (MAGMA_SUCCESS !=  magma_smalloc_pinned(&s_data, K)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }

  if (COMPUTE_UV) {
    if (FULL_MATRICES) {
      jobu = MagmaAllVec;
      jobv = MagmaAllVec;
    }
    else {
      jobu = MagmaSomeVec;
      jobv = MagmaSomeVec;
    }
    M_U  = (jobu == MagmaAllVec ? M : K);
    N_VT = (jobv == MagmaAllVec ? N : K);
    ldu = M;
    ldv = N_VT;

    if (MAGMA_SUCCESS != magma_smalloc_pinned(&u_data, M * M_U)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMagmaSVD: failed to allocate memory");
      goto fail;
    }
    if (MAGMA_SUCCESS != magma_smalloc_pinned(&vt_data, ldv * N)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMagmaSVD: failed to allocate memory");
      goto fail;
    }
  }
  else {
    jobu = MagmaNoVec;
    jobv = MagmaNoVec;
    ldu = M;
    ldv = N;
  }

  // query for workspace size
  magma_sgesvd(jobu, jobv, M, N, NULL, M, NULL, NULL, ldu, NULL, ldv,
               dummy, -1, &info);

  lwork = (magma_int_t) MAGMA_S_REAL(dummy[0]);
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&work, lwork)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate working memory");
    goto fail;
  }

  // compute svd
  magma_sgesvd(jobu, jobv, M, N, a_data, M, s_data,
               u_data, ldu, vt_data, ldv, work, lwork, &info);
  if (info > 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaSVD: magma_sgesvd_gpu %d superdiagonals failed to converge",
        info);
    goto fail;
  } else if (info < 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaSVD: magma_sgesvd_gpu argument %d has an illegal value", -info);
    goto fail;
  }

  s_dims[0] = K;
  if (theano_prep_output(S, 1, s_dims, A->ga.typecode, GA_C_ORDER, c) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  cudaMemcpy(PyGpuArray_DEV_DATA(*S), s_data, K * sizeof(float),
             cudaMemcpyDeviceToDevice);

  u_dims[0] = M; u_dims[1] = ldu;
  // choose fortran order to avoid transpose
  if (theano_prep_output(U, 2, u_dims, A->ga.typecode, GA_F_ORDER, c) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  cudaMemcpy(PyGpuArray_DEV_DATA(*U), u_data, M * ldu * sizeof(float),
             cudaMemcpyDeviceToDevice);
  /* GpuArray_transpose_inplace(&(*U)->ga, NULL); */

  vt_dims[0] = ldv; vt_dims[1] = N;
  // choose fortran order to avoid transpose
  if (theano_prep_output(VT, 2, vt_dims, A->ga.typecode, GA_F_ORDER, c) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  cudaMemcpy(PyGpuArray_DEV_DATA(*VT), vt_data, ldv * N * sizeof(float),
             cudaMemcpyDeviceToDevice);
  /* GpuArray_transpose_inplace(&(*VT)->ga, NULL); */

  res = 0;
fail:
  if (a_data != NULL)
    magma_free_pinned(a_data);
  if (s_data != NULL)
    magma_free_pinned(s_data);
  if (u_data != NULL)
    magma_free_pinned(u_data);
  if (vt_data != NULL)
    magma_free_pinned(vt_data);
  if (work != NULL)
    magma_free_pinned(work);
  magma_finalize();
  cuda_exit(c->ctx);
  return res;
}
