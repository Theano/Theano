#ifndef THEANO_MOD_HELPER
#define THEANO_MOD_HELPER

#include <Python.h>

#ifndef _WIN32
#define MOD_PUBLIC __attribute__((visibility ("default")))
#else
/* MOD_PUBLIC is only used in PyMODINIT_FUNC, which is declared
 * and implemented in mod.cu/cpp, not in headers, so dllexport
 * is always correct. */
#define MOD_PUBLIC __declspec( dllexport )
#endif

#ifdef __cplusplus
#define THEANO_EXTERN extern "C"
#else
#define THEANO_EXTERN
#endif

#if PY_MAJOR_VERSION < 3
#define THEANO_RTYPE void
#else
#define THEANO_RTYPE PyObject *
#endif

/* We need to redefine PyMODINIT_FUNC to add MOD_PUBLIC in the middle */
#undef PyMODINIT_FUNC
#define PyMODINIT_FUNC THEANO_EXTERN MOD_PUBLIC THEANO_RTYPE

#endif
