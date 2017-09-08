#section kernels

#kernel tril_kernel : size, size, size, *:
#include "cluda.h"

KERNEL void tril_kernel(const ga_size nthreads, const ga_size ncols,
                        const ga_size a_off, GLOBAL_MEM DTYPE_INPUT_0 *a) {
  a = (GLOBAL_MEM DTYPE_INPUT_0 *)(((char *)a) + a_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0; index < nthreads;
       index += LDIM_0 * GDIM_0) {
    unsigned int ix = index / ncols;
    unsigned int iy = index % ncols;
    if (ix < iy) {
      a[index] = 0.0;
    }
  }
}

#kernel triu_kernel : size, size, size, *:
#include "cluda.h"

KERNEL void triu_kernel(const ga_size nthreads, const ga_size ncols,
                        const ga_size a_off, GLOBAL_MEM DTYPE_INPUT_0 *a) {
  a = (GLOBAL_MEM DTYPE_INPUT_0 *)(((char *)a) + a_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0; index < nthreads;
       index += LDIM_0 * GDIM_0) {
    unsigned int ix = index / ncols;
    unsigned int iy = index % ncols;
    if (ix > iy) {
      a[index] = 0.0;
    }
  }
}

#section init_code

setup_ext_cuda();

#section support_code_struct

int APPLY_SPECIFIC(magma_cholesky)(PyGpuArrayObject *A, PyGpuArrayObject **L,
                                   PARAMS_TYPE *params) {
  const size_t *dims;
  size_t N, n2;
  magma_uplo_t ul;
  int res = -1, info;

  if (A->ga.typecode != GA_FLOAT) {
    PyErr_SetString(PyExc_TypeError,
                    "GpuMagmaCholesky: unsupported data type");
    return -1;
  }

  // This is early to match the exit() in the fail label.
  cuda_enter(params->context->ctx);

  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMagmaCholesky: requires data to be C-contiguous");
    goto fail;
  }
  if (PyGpuArray_NDIM(A) != 2) {
    PyErr_SetString(PyExc_ValueError, "GpuMagmaCholesky: matrix rank error");
    goto fail;
  }
  dims = PyGpuArray_DIMS(A);
  if (dims[0] != dims[1]) {
    PyErr_SetString(PyExc_ValueError, "GpuMagmaCholesky: matrix is not square");
    goto fail;
  }

  if (params->inplace) {
    Py_XDECREF(*L);
    *L = A;
    Py_INCREF(*L);
  } else {
    *L = theano_try_copy(*L, A);
    if (*L == NULL) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "GpuMagmaCholesky: failed to allocate memory for the output");
      goto fail;
    }
  }

  // magma matrix cholesky
  N = dims[0];
  n2 = N * N;

// Magma requires column-major order for the matrix A. Instead of changing
// matrix order which requires copying data, we can compute cholesky
// decomposition where we change parameters lower to upper and upper to
// lower.
  if (params->lower) {
    ul = MagmaUpper;
  }
  else {
    ul = MagmaLower;
  }
  magma_spotrf_gpu(ul, N, (float *)PyGpuArray_DEV_DATA(*L), N, &info);
  if (info > 0) {
    PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: the leading minor of "
                                     "order %d is not positive definite",
                 info);
    goto fail;
  } else if (info < 0) {
    PyErr_Format(
        PyExc_RuntimeError,
        "GpuMagmaCholesky: magma_spotrf_gpu argument %d has an illegal value",
        -info);
    goto fail;
  }

  if (params->lower) {
    res = tril_kernel_scall(1, &n2, 0, n2, N, (*L)->ga.offset, (*L)->ga.data);
    if (res != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: tril_kernel %s.",
                   GpuKernel_error(&k_tril_kernel, res));
      goto fail;
    }
  } else {
    res = triu_kernel_scall(1, &n2, 0, n2, N, (*L)->ga.offset, (*L)->ga.data);
    if (res != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: triu_kernel %s.",
                   GpuKernel_error(&k_triu_kernel, res));
      goto fail;
    }
  }
  res = 0;
fail:
  cuda_exit(params->context->ctx);
  return res;
}
