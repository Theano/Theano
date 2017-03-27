#ifndef _MKL_NDARRAY_H_
#define _MKL_NDARRAY_H_

#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdint.h>
#include "mkl_dnn.h"

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)(-1))
#endif

#ifndef Py_TYPE
#define Py_TYPE(o) ((o)->ob_type)
#endif

#define MAX_NDIM     (16)
#define MKL_FLOAT32  (11)
#define MKL_FLOAT64  (12)

char* MKL_TYPE[] = {"", "", "", "int16", "", "int32", "", "int64",
                    "", "", "", "float32", "float64", ""};

#if PY_MAJOR_VERSION >= 3
// Py3k treats all ints as longs. This one is not caught by npy_3kcompat.h.
#define PyNumber_Int PyNumber_Long

#include "numpy/npy_3kcompat.h"

// Py3k strings are unicode, these mimic old functionality.
//
// NOTE: npy_3kcompat.h replaces PyString_X with PyBytes_X, which breaks
// compatibility with some functions returning text.
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString PyUnicode_AsUTF8
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_Size PyUnicode_GET_SIZE
#define PyInt_FromSize_t PyLong_FromSize_t

// Python 3 expects a PyObject* as the first argument to PySlice_GetIndicesEx().
#define SLICE_CAST(x) (x)
#else
// Python 2 expects a PySliceObject* as the first argument to PySlice_GetIndicesEx().
#define SLICE_CAST(x) ((PySliceObject*)(x))
#endif // end #if PY_MAJOR_VERSION >= 3


/**
 * MKLNdarray: wrapper for MKL private data and layout
 * This is a Python type.
 */
typedef struct __MKLNdarray__{

    PyObject_HEAD
    PyObject * base;

    /* Type-specific fields go here. */
    int         nd;                             // the number of dimensions of the tensor, maximum is 16 (MAX_NDIM).
    int         dtype;                          // an integer type number is given here.
    size_t      data_size;                      // the number of bytes allocated for mkl_data
    size_t      user_structure[2 * MAX_NDIM];   // user layout: [size0, size1, ..., stride0, stride1, ..., 0, 0].
    dnnLayout_t private_layout;                 // layout instance create by MKL APIs
    void*       private_data;                   // data buffer
    void*       private_workspace;              // computation workspace for forward and backward
}MKLNdarray;


__attribute__((visibility ("default"))) int MKLNdarray_Check(const PyObject* ob);
__attribute__((visibility ("default"))) PyObject* MKLNdarray_New(int nd, int typenum);
__attribute__((visibility ("default"))) int MKLNdarray_CopyFromArray(MKLNdarray* self, PyArrayObject* obj);
__attribute__((visibility ("default"))) int MKLNdarray_set_structure(MKLNdarray* self, int nd, const size_t* dims);
__attribute__((visibility ("default"))) PyObject* MKLNdarray_CreateArrayObj(MKLNdarray* self);

__attribute__((visibility ("default"))) void* MKLNdarray_DATA(const MKLNdarray* self);
__attribute__((visibility ("default"))) void* MKLNdarray_WORKSPACE(const MKLNdarray* self);
__attribute__((visibility ("default"))) dnnLayout_t MKLNdarray_LAYOUT(const MKLNdarray* self);
__attribute__((visibility ("default"))) const size_t* MKLNdarray_DIMS(const MKLNdarray* self);
__attribute__((visibility ("default"))) const size_t* MKLNdarray_STRIDES(const MKLNdarray* self);
__attribute__((visibility ("default"))) int MKLNdarray_NDIM(const MKLNdarray* self);
__attribute__((visibility ("default"))) int MKLNdarray_TYPE(const MKLNdarray* self);

__attribute__((visibility ("default"))) int MKLNdarray_create_buffer_from_primitive(MKLNdarray *self,
                                                                                    const dnnPrimitive_t *prim,
                                                                                    dnnResourceType_t res_type);
#endif
