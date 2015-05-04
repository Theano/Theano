#ifndef THEANO_GPUARRAY_HELPER
#define THEANO_GPUARRAY_HELPER

#include <string.h>
#include <pygpu_api.h>

static int theano_size_check(PyGpuArray *a, unsigned int nd,
                             const size_t *dims, int typecode) {
  return (a->ga.nd == nd && a->ga.typecode == typecode &&
          memcmp(a->dims, dims, nd * sizeof(size_t)) == 0);
}

static int theano_prep_output(PyGpuArrayObject **out, unsigned int nd,
                             const size_t *dims, int typecode, ga_order ord,
                             PyGpuContextObject *c) {
  if (*out != NULL &&
      theano_size_check(*out, nd, dims, typecode)) {
    return 1;
  }

  Py_XDECREF(*out);
  *out = pygpu_empty(nd, dims, typecode, ord, c, Py_None);
  return (*out == NULL)? 0 : 1;
}

#endif
