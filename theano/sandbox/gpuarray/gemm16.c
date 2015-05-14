#section init_code_struct

/* Why do we need this? */
size_t dim = 2048 * 32;
rand_buf = pygpu_empty(1, &dim, GA_UINT, GA_C_ORDER, pygpu_default_context(),
                       Py_None);
if (rand_buf == NULL) {
  FAIL;
}

#section support_code_struct

PyGpuArrayObject *rand_buf;

int gemm16(PyGpuArrayObject *C, float alpha,
           PyGpuArrayObject *A, PyGpuArrayObject *B,
           float beta, PyGpuArrayObject **out) {
  PyGpuArrayObject *AA = NULL;
  PyGpuArrayObject *BB = NULL;
  GpuKernel *gk;
  void *params[13];
  size_t grid[2];
  size_t threads[2];
  int res = 0;
  int flags = 0;
  int lda, ldb, ldc, n, m, k;
  int n128, n64;
  int size = 0;
  int vec = 0;
  static unsigned int nprocs = 0;
  char opA, opB;
  if (GpuArray_CHKFLAGS(&A->ga, GA_FARRAY) &&
      GpuArray_CHKFLAGS(&B->ga, GA_FARRAY)) {
    /*
     * The nervana kernels do not cover the case where both inputs are
     * trans so we need to copy one of them.  We choose the smallest
     * one.
     */
    if (PyGpuArray_DIM(A, 0) * PyGpuArray_DIM(A, 1) <
        PyGpuArray_DIM(B, 0) * PyGpuArray_DIM(B, 1)) {
      AA = pygpu_copy(A, GA_C_ORDER);
      if (AA == NULL) {
        res = 1;
        goto cleanup;
      }
      BB = B;
      Py_INCREF(BB);
    } else {
      BB = pygpu_copy(B, GA_C_ORDER);
      if (BB == NULL) {
        res = 1;
        goto cleanup;
      }
      AA = A;
      Py_INCREF(AA);
    }
  }
  if (GEMM16_INPLACE && GpuArray_CHKFLAGS(&C->ga, GA_CARRAY)) {
    Py_XDECREF(*out);
    *out = C;
    Py_INCREF(*out);
  } else {
    *out = theano_try_copy(*out, C);
    if (*out == NULL) {
      res = 1;
      goto cleanup;
    }
  }

  if (GpuArray_CHKFLAGS(&A->ga, GA_FARRAY))
    opA = 't';
  else
    opA = 'n';

  if (GpuArray_CHKFLAGS(&B->ga, GA_FARRAY))
    opB = 't';
  else
    opB = 'n';

  m = PyGpuArray_DIM(C, 0);
  n = PyGpuArray_DIM(C, 1);
  k = PyGpuArray_DIM(B, 0);

  /* Tuning code adapted from the python version */
  grid[0] = (m + 127) / 128;

  if (opA == 'n' && opB == 't')
    size = 128;
  else {
    if (n < 384-16) {
      n128 = n % 128;
      if (n128 < 112) {
        if (48 < n128 && n128 <= 64) {
          n64 = n / 64;
          if (nprocs == 0)
            if (C->ga.ops->property(C->context->ctx, NULL, NULL,
                                    GA_CTX_PROP_NUMPROCS, &nprocs)) {
              nprocs = 0;
              res = 1;
              goto cleanup;
            }
          n64 *= (grid[0] / nprocs);
          if (n64 > 1 || (opA == 't' && opB == 'n'))
            size = 64;
          else
            size = 32;
        } else {
          size = 32;
        }
      } else {
        size = 128;
      }
    } else {
      size = 128;
    }
  }

  grid[1] = (n + (size-1)) / size;
  if (size == 128)
    threads[0] = 256;
  else
    threads[0] = 128;
  threads[1] = 1;

  if ((opA == 't' && opB == 'n' && m % 8 == 0 && n % 8 == 0) ||
      (opA == 'n' && opB == 'n' && k % 16 == 0 && n % 8 == 0) ||
      (opA == 'n' && opB == '1' && k % 16 == 0))
    vec = 1;

  switch (size) {
  case 128:
    if (opA == 'n' && opB == 'n') {
      if (vec)
        gk = &k_nn_vec_128x128;
      else
        gk = &k_nn_128x128;
    } else if (opA == 'n' && opB == 't') {
      if (vec)
        gk = &k_nt_vec_128x128;
      else
        gk = &k_nt_128x128;
    } else if (opA == 't' && opB == 'n') {
      if (vec)
        gk = &k_tn_vec_128x128;
      else
        gk = &k_tn_128x128;
    }
    break;
  case 64:
    if (opA == 'n' && opB == 'n') {
      if (vec)
        gk = &k_nn_vec_128x64;
      else
        gk = &k_nn_128x64;
    } else if (opA == 't' && opB == 'n') {
      if (vec)
        gk = &k_tn_vec_128x64;
      else
        gk = &k_tn_128x64;
    }
    break;
  case 32:
    if (opA == 'n' && opB == 'n') {
      if (vec)
        gk = &k_nn_vec_128x32;
      else
        gk = &k_nn_128x32;
    } else if (opA == 't' && opB == 'n') {
      if (vec)
        gk = &k_tn_vec_128x32;
      else
        gk = &k_tn_128x32;
    }
    break;
  default:
    PyErr_SetString(PyExc_RuntimeError, "error selecting kernel");
    res = 1;
    goto cleanup;
  }

  params[0] = &rand_buf->ga;
  params[1] = &A->ga;
  params[2] = &B->ga;
  params[3] = &C->ga;
  params[4] = &lda;
  params[5] = &ldb;
  params[6] = &ldc;
  params[7] = &m;
  params[8] = &n;
  params[9] = &k;
  params[10] = &alpha;
  params[11] = &beta;
  params[12] = &flags;

  printf("%c%c_%s128x%d\n", opA, opB, vec ? "vec_" : "", size);

  printf("%zu %zu %zu %zu\n", rand_buf->ga.offset, A->ga.offset, B->ga.offset, C->ga.offset);
  printf("%p %p %p %p\n", *((void **)rand_buf->ga.data), *((void **)A->ga.data), *((void **)B->ga.data), *((void **)C->ga.data));

  if (GpuKernel_call2(gk, NULL, threads, grid, params) != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "error in gemm16 kernel call");
    res = 1;
  }

cleanup:
  Py_XDECREF(AA);
  Py_XDECREF(BB);
  return res;
}
