#section init_code

setup_ext_cuda();

#section support_code_struct

int APPLY_SPECIFIC(magma_svd)(PyGpuArrayObject *A,
#ifdef COMPUTE_UV
                              PyGpuArrayObject **U,
#endif
                              PyGpuArrayObject **S,
#ifdef COMPUTE_UV
                              PyGpuArrayObject **VT,
#endif
                              PyGpuContextObject *c) {
  magma_int_t *iwork = NULL, iunused[1];
  magma_int_t M, N, K, ldu, ldv, M_U, N_VT, info;
  magma_vec_t jobz;
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
  // reverse dimensions because MAGMA expects column-major matrices:
  M = PyGpuArray_DIM(A, 1);
  N = PyGpuArray_DIM(A, 0);
  K = std::min(M, N);

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

#ifdef COMPUTE_UV
#ifdef FULL_MATRICES
  jobz = MagmaAllVec;
#else
  jobz = MagmaSomeVec;
#endif
  M_U  = (jobz == MagmaAllVec ? M : K);
  N_VT = (jobz == MagmaAllVec ? N : K);
  ldu = M;
  ldv = N_VT;

  if (MAGMA_SUCCESS != magma_smalloc_pinned(&u_data, M_U * M)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&vt_data, N * N_VT)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
#else
  jobz = MagmaNoVec;
  ldu = M;
  ldv = N;
#endif

  // query for workspace size
  magma_sgesdd(jobz, M, N, NULL, M, NULL, NULL, ldu, NULL, ldv,
               dummy, -1, iunused, &info);

  lwork = (magma_int_t) MAGMA_S_REAL(dummy[0]);
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&work, lwork)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate working memory");
    goto fail;
  }

  if (MAGMA_SUCCESS != magma_imalloc_cpu(&iwork, 8*K)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate working memory");
    goto fail;
  }

  // compute svd
  magma_sgesdd(jobz, M, N, a_data, M, s_data,
               u_data, ldu, vt_data, ldv, work, lwork, iwork, &info);
  if (info > 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaSVD: the updating process of SBDSDC did not converge (error: %d)",
        info);
    goto fail;
  } else if (info < 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaSVD: magma_sgesdd_gpu argument %d has an illegal value", -info);
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

#ifdef COMPUTE_UV
  u_dims[0] = N; u_dims[1] = N_VT;
  if (theano_prep_output(U, 2, u_dims, A->ga.typecode, GA_C_ORDER, c) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  // magma expects column-major matrices. Exchange u_data -> VT and vt_data -> U
  // to match numpy.linalg.svd output
  cudaMemcpy(PyGpuArray_DEV_DATA(*U), vt_data, N * N_VT * sizeof(float),
             cudaMemcpyDeviceToDevice);

  vt_dims[0] = M_U; vt_dims[1] = M;
  if (theano_prep_output(VT, 2, vt_dims, A->ga.typecode, GA_C_ORDER, c) != 0){
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaSVD: failed to allocate memory");
    goto fail;
  }
  // magma expects column-major matrices. Exchange u_data -> VT and vt_data -> U
  // to match numpy.linalg.svd output
  cudaMemcpy(PyGpuArray_DEV_DATA(*VT), u_data, M_U * M * sizeof(float),
             cudaMemcpyDeviceToDevice);
#endif
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
  if (iwork != NULL)
    magma_free_cpu(iwork);
  magma_finalize();
  cuda_exit(c->ctx);
  return res;
}
