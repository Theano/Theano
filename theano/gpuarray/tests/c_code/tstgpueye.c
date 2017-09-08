#section kernels

#kernel eye : *, size, size, size :
#include <cluda.h>
/* The eye name will be used to generate supporting objects.  The only
   you probably need to care about is the kernel object which will be
   named 'k_' + <the name above> (k_eye in this case).  This name also
   has to match the kernel function name below.
 */

KERNEL void eye(GLOBAL_MEM DTYPE_OUTPUT_0 *a, ga_size a_off, ga_size n, ga_size m) {
  a = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)a) + a_off);
  ga_size nb = n < m ? n : m;
  for (ga_size i = LID_0; i < nb; i += LDIM_0) {
    a[i*m + i] = 1;
  }
}

#section support_code_struct

int APPLY_SPECIFIC(tstgpueye)(PyArrayObject *n, PyArrayObject *m,
                              PyGpuArrayObject **z, PARAMS_TYPE* params) {
  size_t dims[2] = {0, 0};
  size_t ls, gs;
  void *args[3];
  int err;

  dims[0] = ((DTYPE_INPUT_0 *)PyArray_DATA(n))[0];
  dims[1] = ((DTYPE_INPUT_1 *)PyArray_DATA(m))[0];

  Py_XDECREF(*z);
  *z = pygpu_zeros(2, dims,
                   params->typecode,
                   GA_C_ORDER,
                   params->context, Py_None);
  if (*z == NULL)
    return -1;

  ls = 1;
  gs = 256;
  /* The eye_call name comes from the kernel declaration above. */
  err = eye_call(1, &gs, &ls, 0, (*z)->ga.data, (*z)->ga.offset, dims[0], dims[1]);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: kEye: %s. n%lu, m=%lu.",
                 GpuKernel_error(&k_eye, err),
                 (unsigned long)dims[0], (unsigned long)dims[1]);
    return -1;
  }
  return 0;
}
