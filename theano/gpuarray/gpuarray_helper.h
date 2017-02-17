#ifndef THEANO_GPUARRAY_HELPER
#define THEANO_GPUARRAY_HELPER

#include <string.h>
#include <gpuarray_api.h>
#include <numpy_compat.h>
#include <gpuarray/util.h>


static int theano_size_check(PyGpuArrayObject *a, unsigned int nd,
                             const size_t *dims, int typecode) {
  return (a->ga.nd == nd && a->ga.typecode == typecode &&
          memcmp(a->ga.dimensions, dims, nd * sizeof(size_t)) == 0);
}

static int theano_prep_output(PyGpuArrayObject **out, unsigned int nd,
                             const size_t *dims, int typecode, ga_order ord,
                             PyGpuContextObject *c) {
  if (*out != NULL &&
      theano_size_check(*out, nd, dims, typecode)) {
    return 0;
  }

  Py_XDECREF(*out);
  *out = pygpu_empty(nd, dims, typecode, ord, c, Py_None);
  return (*out == NULL) ? 1 : 0;
}

static PyGpuArrayObject *theano_try_copy(PyGpuArrayObject *out,
                                         PyGpuArrayObject *V) {
  if (out &&
      GpuArray_CHKFLAGS(&out->ga, GA_CARRAY) &&
      theano_size_check(out, PyGpuArray_NDIM(V),
                        PyGpuArray_DIMS(V),
                        V->ga.typecode)) {
    if (pygpu_move(out, V)) {
      Py_XDECREF(out);
      return NULL;
    }
  } else {
    Py_XDECREF(out);
    out = pygpu_copy(V, GA_C_ORDER);
  }
  return out;
}

static inline void *PyGpuArray_DEV_DATA(PyGpuArrayObject *a) {
  /* This is guaranteed to work and return the raw CUDA/OpenCL object on
   * all recent (as of June 2015) version of libgpuarray. This is also
   * promised to keep working in future versions. */
  char * p = *((char **)a->ga.data);

  /* This only works on cuda since we have a real pointer. */
  return (void *)(p + a->ga.offset);
}

#endif
