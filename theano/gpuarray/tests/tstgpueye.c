#section kernels

#kernel eye : *, size, size :

KERNEL void eye(GLOBAL_MEM DTYPE_o0 *a, ga_size n, ga_size m) {
  ga_size nb = n < m ? n : m;
  for (ga_size i = LID_0; i < nb; i += LDIM_0) {
    a[i*m + i] = 1;
  }
}

#section support_code_struct

int APPLY_SPECIFIC(tstgpueye)(PyArrayObject *n, PyArrayObject *m,
                              PyGpuArrayObject **z, PyGpuContextObject *ctx) {
  size_t dims[2] = {0, 0};
  size_t ls, gs;
  void *args[3];
  int err;

  dims[0] = ((DTYPE_INPUT_0 *)PyArray_DATA(n))[0];
  dims[1] = ((DTYPE_INPUT_1 *)PyArray_DATA(m))[0];

  Py_XDECREF(*z);
  *z = pygpu_zeros(2, dims,
                   TYPECODE,
                   GA_C_ORDER,
                   ctx, Py_None);
  if (*z == NULL)
    return -1;

  args[0] = (*z)->ga.data;
  args[1] = &dims[0];
  args[2] = &dims[1];
  ls = 1;
  gs = 256;
  err = GpuKernel_call(&k_eye, 1, &ls, &gs, 0, args);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: kEye: %s. n%lu, m=%lu.",
                 GpuKernel_error(&k_eye, err),
                 (unsigned long)dims[0], (unsigned long)dims[1]);
    return -1;
  }
  return 0;
}
