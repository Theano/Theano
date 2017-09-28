#section kernels

#kernel triu_kernel : size, size, size, *:
#include "cluda.h"

KERNEL void triu_kernel(const ga_size nthreads, const ga_size ncols,
                        const ga_size a_off, GLOBAL_MEM DTYPE_INPUT_0 *a) {
  a = (GLOBAL_MEM DTYPE_INPUT_0 *)(((char *)a) + a_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0; index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size ix = index / ncols;
    const ga_size iy = index % ncols;
    if (ix > iy) {
      a[index] = 0.0;
    }
  }
}

#section init_code

setup_ext_cuda();

#section support_code

static PyGpuArrayObject *pygpu_narrow(PyGpuArrayObject *src, size_t dim,
                                      size_t size) {
  PyGpuArrayObject *src_view = pygpu_view(src, Py_None);
  src_view->ga.dimensions[dim] = size;
  GpuArray_fix_flags(&src_view->ga);
  return pygpu_copy(src_view, GA_C_ORDER);
}

#section support_code_struct

int APPLY_SPECIFIC(magma_qr)(PyGpuArrayObject *A_,
                             PyGpuArrayObject **R,
                             PyGpuArrayObject **Q, // may be NULL
                             PARAMS_TYPE* params) {
  PyGpuArrayObject *A = NULL;
  magma_int_t M, N, K, nb, ldwork;
  size_t n2;
  float *tau_data = NULL;
  gpudata *work_data = NULL;
  int res = -1, info;
  A = A_;

  if (A->ga.typecode != GA_FLOAT) {
    PyErr_SetString(PyExc_TypeError,
                    "GpuMagmaQR: Unsupported data type");
    return -1;
  }

  // This is early to match the exit() in the fail label.
  cuda_enter(params->context->ctx);

  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaQR: requires data to be C-contiguous");
    goto fail;
  }
  if (PyGpuArray_NDIM(A) != 2) {
    PyErr_SetString(PyExc_ValueError, "GpuMagmaQR: matrix rank error");
    goto fail;
  }

  A = pygpu_copy(A_, GA_F_ORDER);
  if (A == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaQR: failed to change to column-major order");
    goto fail;
  }

  // magma matrix qr
  M = PyGpuArray_DIM(A, 0);
  N = PyGpuArray_DIM(A, 1);
  K = M < N ? M : N;

  if (MAGMA_SUCCESS != magma_smalloc_pinned(&tau_data, N * N)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaQR: failed to allocate working memory");
    goto fail;
  }

  nb = magma_get_sgeqrf_nb(M, N);
  ldwork = (2 * K + magma_roundup(N, 32)) * nb;
  work_data = gpudata_alloc(params->context->ctx, ldwork * sizeof(float), NULL, 0, NULL);
  if (work_data == NULL) {
    PyErr_SetString(PyExc_RuntimeError,
                    "GpuMagmaQR: failed to allocate working memory");
    goto fail;
  }

  // compute R
  magma_sgeqrf2_gpu(M, N, (float *)PyGpuArray_DEV_DATA(A), M, tau_data, &info);
  if (info != 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaQR: magma_sgeqrf2_gpu argument %d has an illegal value", -info);
    goto fail;
  }
  *R = pygpu_narrow(A, 0, K);
  if (*R == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "GpuMagmaQR: failed to narrow array");
    goto fail;
  }
  n2 = K * N;
  res = triu_kernel_scall(1, &n2, 0, n2, N, (*R)->ga.offset, (*R)->ga.data);
  if (res != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "GpuMagmaQR: triu_kernel %s.",
                 GpuKernel_error(&k_triu_kernel, res));
    goto fail;
  }

  if (params->complete) {
    // compute Q
    Py_XDECREF(A);
    A = pygpu_copy(A_, GA_F_ORDER);
    if (A == NULL) {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMagmaQR: failed to change to column-major order");
      return -1;
    }
    magma_sgeqrf_gpu(M, N, (float *)PyGpuArray_DEV_DATA(A), M, tau_data,
                     *(float **)work_data, &info);
    if (info != 0) {
      PyErr_Format(
                   PyExc_RuntimeError,
                   "GpuMagmaQR: magma_sgeqrf_gpu argument %d has an illegal value", -info);
      goto fail;
    }

    magma_sorgqr_gpu(M, K, K, (float *)PyGpuArray_DEV_DATA(A), M, tau_data,
                     *(float **)work_data, nb, &info);
    if (info != 0) {
      PyErr_Format(
                   PyExc_RuntimeError,
                   "GpuMagmaQR: magma_sorgqr_gpu argument %d has an illegal value", -info);
      goto fail;
    }
    *Q = pygpu_narrow(A, 1, K);
    if (*Q == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "GpuMagmaQR: failed to narrow array");
      goto fail;
    }
  }
  res = 0;
fail:
  if (tau_data != NULL)
    magma_free_pinned(tau_data);
  if (work_data != NULL)
    gpudata_release(work_data);
  Py_XDECREF(A);
  cuda_exit(params->context->ctx);
  return res;
}
