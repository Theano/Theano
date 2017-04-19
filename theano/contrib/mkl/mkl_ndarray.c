#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "mkl_ndarray.h"


int MKLNdarray_Check(const PyObject *ob);
PyObject* MKLNdarray_New(int nd, int typenum);
int MKLNdarray_CopyFromArray(MKLNdarray *self, PyArrayObject *obj);


/*
 * This function is called in MKLNdarray_dealloc.
 *
 * Release all allocated buffer and layout in self.
 *
 * If private_layout is a reference from another ndarray,
 * do not free it in this ndarray.
 *
 */
static int
MKLNdarray_uninit(MKLNdarray *self) {
    int rval = 0;
    if (self->dtype == MKL_FLOAT64) {  // for float64
        if (self->private_data) {
            rval = dnnReleaseBuffer_F64(self->private_data);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release data: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_data = NULL;
        }

        if ((self->base == NULL) && self->private_layout) {
            rval = dnnLayoutDelete_F64(self->private_layout);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release layout: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_layout = NULL;
        }

        if (self->private_workspace) {
            rval = dnnReleaseBuffer_F64(self->private_workspace);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release workspace: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_workspace = NULL;
        }
    } else {  // for float32
        if (self->private_data) {
            rval = dnnReleaseBuffer_F32(self->private_data);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release data: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_data = NULL;
        }

        if ((self->base == NULL) && self->private_layout) {
            rval = dnnLayoutDelete_F32(self->private_layout);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release layout: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_layout = NULL;
        }

        if (self->private_workspace) {
            rval = dnnReleaseBuffer_F32(self->private_workspace);

            if (rval != E_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release workspace: %d, line: %d",
                             rval, __LINE__);
            }
            self->private_workspace = NULL;
        }
    }

    self->data_size = 0;
    self->nd = -1;
    self->dtype = -1;
    Py_XDECREF(self->base);
    self->base = NULL;
    return rval;
}


/*
 * type: tp_dealloc
 *
 * This function will be called by Py_DECREF when object's reference count is reduced to zero.
 * DON'T call this function directly.
 *
 */
static void
MKLNdarray_dealloc(MKLNdarray *self) {
    if (Py_REFCNT(self) > 1) {
        printf("WARNING: MKLNdarray_dealloc called when there is still active reference to it.\n");
    }
    if (self != NULL) {
        MKLNdarray_uninit(self);
        Py_TYPE(self)->tp_free((PyObject*)self);
    }
}


/*
 * type:tp_new
 *
 * This function is used to create an instance of object.
 * Be first called when do a = MKLNdarray() in python code.
 *
 */
static PyObject*
MKLNdarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MKLNdarray* self = NULL;
    self = (MKLNdarray*)(type->tp_alloc(type, 0));

    if (self != NULL) {
        self->base              = NULL;
        self->nd                = -1;
        self->dtype             = -1;
        self->private_workspace = NULL;
        self->private_data      = NULL;
        self->private_layout    = NULL;
        self->data_size         = 0;

        memset((void*)(self->user_structure), 0, 2 * MAX_NDIM * sizeof (size_t));
    } else {
        PyErr_SetString(PyExc_MemoryError, "MKLNdarray_new: fail to create a new instance \n");
        return NULL;
    }
    return (PyObject*)self;
}


/*
 * type:tp_init
 *
 * This function is called after MKLNdarray_new.
 *
 * Initialize an instance. like __init__() in python code.
 *
 * args: need input a PyArrayObject here.
 *
 */
static int
MKLNdarray_init(MKLNdarray *self, PyObject *args, PyObject *kwds) {
    PyObject* arr = NULL;

    if (!PyArg_ParseTuple(args, "O", &arr))
        return -1;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_init: PyArrayObject arg required");
        return -1;
    }

    // do type conversion here. PyArrayObject -> MKLNdarray
    int rval = -1;
    if (self != NULL) {
        rval = MKLNdarray_CopyFromArray(self, (PyArrayObject*)arr);
        return rval;
    } else {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_init: input MKLNdarray* self is NULL");
        return -1;
    }
}


/*
 * type:tp_repr
 *
 * Return a string or a unicode object. like repr() in python code.
 *
 */
PyObject* MKLNdarray_repr(PyObject *self) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_repr: input PyObject* self is NULL");
        return NULL;
    }
    MKLNdarray* object = (MKLNdarray*)self;
    char cstr[64]; // 64 chars is enough for a string.
    sprintf(cstr, "ndim=%d, dtype=%s", object->nd, MKL_TYPE[object->dtype]);
    PyObject* out = PyString_FromFormat("%s%s%s", "MKLNdarray(", cstr, ")");
#if PY_MAJOR_VERSION >= 3
    PyObject* out2 = PyObject_Str(out);
    Py_DECREF(out);
    return out2;
#else
    return out;
#endif
}


/*
 * Get dims in user_structure.
 *
 * A pointer is returned.
 *
 */
const size_t*
MKLNdarray_DIMS(const MKLNdarray *self) {
    if (self != NULL) {
        return self->user_structure;
    } else {
        return NULL;
    }
}


/*
 * Get strides in user_structure.
 *
 * A pointer is returned. stride has a self->nd offset n user_structure
 *
 */
const size_t*
MKLNdarray_STRIDES(const MKLNdarray *self) {
    if (self != NULL) {
        return self->user_structure + self->nd;
    } else {
        return NULL;
    }
}


/*
 * Get ndim.
 *
 * An integer is returned.
 *
 */
int MKLNdarray_NDIM(const MKLNdarray *self) {
    if (self != NULL) {
        return self->nd;
    } else {
        return -1;
    }
}


/*
 * Get dtype.
 *
 * An integer is returned.
 *
 */
int MKLNdarray_TYPE(const MKLNdarray *self) {
    if (self != NULL) {
        return self->dtype;
    } else {
        return -1;
    }
}


/*
 * Get address of private_data.
 *
 * An void* pointer is returned.
 *
 */
void*
MKLNdarray_DATA(const MKLNdarray *self) {
    if (self != NULL) {
        return self->private_data;
    } else {
        return NULL;
    }
}


/*
 * Get address of private_workspace.
 *
 * An void* pointer is returned.
 *
 */
void*
MKLNdarray_WORKSPACE(const MKLNdarray *self) {
    if (self != NULL) {
        return self->private_workspace;
    } else {
        return NULL;
    }
}


/*
 * Get address of private_layout.
 *
 * An dnnLayout_t* pointer is returned.
 *
 */
dnnLayout_t
MKLNdarray_LAYOUT(const MKLNdarray *self) {
    if (self != NULL) {
        return self->private_layout;
    } else {
        return NULL;
    }
}


/*
 * Create private layout and buffer for a MKLNdarray according to input primitive.
 *
 * If res_type is dnnResourceWorkspace, private_workspace will be allocated for MKLNdarray.
 *
 */
int MKLNdarray_create_buffer_from_primitive(MKLNdarray *self, const dnnPrimitive_t *prim, dnnResourceType_t res_type) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: input MKLNdarray* self is NULL");
        return -1;
    }
    if (self->nd < 0 || (self->dtype != MKL_FLOAT32 && self->dtype != MKL_FLOAT64)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: Can't create layout and buffer for an uninitialized MKLNdarray");
        return -1;
    }

    if (NULL == prim) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: Can't create layout and buffer with an empty primtive");
        return -1;
    }

    int status = 0;
    if (self->dtype == MKL_FLOAT64) {  // for float64
        if (res_type == dnnResourceWorkspace) {
            if (self->private_workspace != NULL) {
                PyErr_SetString(PyExc_RuntimeError,
                                "MKLNdarray_create_buffer_from_primitive: Can't create buffer for workspace repeatly");
                return -1;
            }

            dnnLayout_t layout_workspace = NULL;
            status = dnnLayoutCreateFromPrimitive_F64(&layout_workspace, *prim, res_type);
            if (E_SUCCESS != status || NULL == layout_workspace) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create layout for workspace failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }

            status = dnnAllocateBuffer_F64(&(self->private_workspace), layout_workspace);
            if (E_SUCCESS != status || NULL == self->private_workspace) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create buffer for workspace failed: %d, line: %d",
                             status, __LINE__);
                dnnLayoutDelete_F64(layout_workspace);
                layout_workspace = NULL;
                return -1;
            }

            dnnLayoutDelete_F64(layout_workspace);
            layout_workspace = NULL;
        } else {
            if (NULL != self->private_layout || NULL != self->private_data) {
                PyErr_SetString(PyExc_RuntimeError,
                                "MKLNdarray_create_buffer_from_primitive: Can't create layout or buffer for MKLNdarray repeatly");
                return -1;
            }

            status = dnnLayoutCreateFromPrimitive_F64(&(self->private_layout), *prim, res_type);
            if (E_SUCCESS != status || NULL == self->private_layout) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create private layout failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }

            status = dnnAllocateBuffer_F64(&(self->private_data), self->private_layout);
            if (E_SUCCESS != status || NULL == self->private_data) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create private data failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }
            self->data_size = dnnLayoutGetMemorySize_F64(self->private_layout);
        }
    } else {  // for float32
        if (res_type == dnnResourceWorkspace) {
            if (self->private_workspace != NULL) {
                PyErr_SetString(PyExc_RuntimeError,
                                "MKLNdarray_create_buffer_from_primitive: Can't create buffer for workspace repeatly");
                return -1;
            }

            dnnLayout_t layout_workspace = NULL;
            status = dnnLayoutCreateFromPrimitive_F32(&layout_workspace, *prim, res_type);
            if (E_SUCCESS != status || NULL == layout_workspace) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create layout for workspace failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }

            status = dnnAllocateBuffer_F32(&(self->private_workspace), layout_workspace);
            if (E_SUCCESS != status || NULL == self->private_workspace) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create buffer for workspace failed: %d, line: %d",
                             status, __LINE__);
                dnnLayoutDelete_F32(layout_workspace);
                layout_workspace = NULL;
                return -1;
            }

            dnnLayoutDelete_F32(layout_workspace);
            layout_workspace = NULL;
        } else {
            if (NULL != self->private_layout || NULL != self->private_data) {
                PyErr_SetString(PyExc_RuntimeError,
                                "MKLNdarray_create_buffer_from_primitive: Can't create layout or buffer for MKLNdarray repeatly");
                return -1;
            }

            status = dnnLayoutCreateFromPrimitive_F32(&(self->private_layout), *prim, res_type);
            if (E_SUCCESS != status || NULL == self->private_layout) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create private layout failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }

            status = dnnAllocateBuffer_F32(&(self->private_data), self->private_layout);
            if (E_SUCCESS != status || NULL == self->private_data) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_create_buffer_from_primitive: Create private data failed: %d, line: %d",
                             status, __LINE__);
                return -1;
            }
            self->data_size = dnnLayoutGetMemorySize_F32(self->private_layout);
        }
    }
    return 0;
}


/*
 * In this function a plain layout is created for self according to user_structure.
 *
 * A private_data buffer is allocated for self according to the plain layout.
 *
 */
int MKLNdarray_create_buffer_from_structure(MKLNdarray *self) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_structure: input MKLNdarray* self is NULL");
        return -1;
    }
    if (self->nd <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_create_buffer_from_structure: Can't create mkl dnn layout and allocate buffer for a %d dimension MKLNdarray",
                     self->nd);
        return -1;
    }
    size_t ndim = self->nd;

    if (self->private_layout || self->private_data) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_create_buffer_from_structure: MKL layout and buffer have been allocated for %p \n", self);
        return -1;
    }

    size_t mkl_size[MAX_NDIM] = {0};
    size_t mkl_stride[MAX_NDIM] = {0};

    // nchw -> whcn
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i -1];
    }

    // float64
    if (self->dtype == MKL_FLOAT64) {
        int status = dnnLayoutCreate_F64(&(self->private_layout),
                                         ndim,
                                         mkl_size,
                                         mkl_stride);
        if (E_SUCCESS != status || NULL == self->private_layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: Call dnnLayoutCreate_F64 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F64(&(self->private_data), self->private_layout);
        if (E_SUCCESS != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: Call dnnAllocateBuffer_F64 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F64(self->private_layout);

    } else {  // float32
        int status = dnnLayoutCreate_F32(&(self->private_layout),
                                         ndim,
                                         mkl_size,
                                         mkl_stride);
        if (E_SUCCESS != status || NULL == self->private_layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: Call dnnLayoutCreate_F32 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F32(&(self->private_data), self->private_layout);
        if (E_SUCCESS != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: Call dnnAllocateBuffer_F32 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F32(self->private_layout);
    }

    return 0;
}


/*
 * Sometimes, we need to allocate buffer for a MKLNdarray from its private_layout.
 *
 */
int MKLNdarray_create_buffer_from_layout(MKLNdarray *self) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: input MKLNdarray* self is NULL");
        return -1;
    }

    if (self->private_layout == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: self->private_layout is NULL");
        return -1;
    }

    if (self->private_data != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: self->private_data is already existed");
        return -1;
    }

    int status = 0;
    if (self->dtype == MKL_FLOAT64) {
        status = dnnAllocateBuffer_F64(&(self->private_data), self->private_layout);
        if (E_SUCCESS != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_layout: Call dnnAllocateBuffer_F64 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F64(self->private_layout);
    } else {  // float32
        status = dnnAllocateBuffer_F32(&(self->private_data), self->private_layout);
        if (E_SUCCESS != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_layout: Call dnnAllocateBuffer_F32 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F32(self->private_layout);
    }
    return 0;
}


/*
 * If we want to create a MKLNdarray with a same private_layout with another
 * MKLNdarray. We set the pointer of private_layout toward that privat_layout
 * in another MKLNdarray, and increase the reference count of the MKLNdarray.
 *
 */
int MKLNdarray_copy_layout(MKLNdarray *self, const MKLNdarray *other) {
    if (self == NULL || other == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: input MKLNdarray* self or other is NULL");
        return -1;
    }
    assert (self->nd == other->nd);
    assert (self->dtype = other->dtype);
    assert (other->private_layout != NULL);

    if (self->private_layout) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: self->private_layout is already exsited");
        return -1;
    }

    self->private_layout = other->private_layout;
    self->base = (PyObject*)other;
    Py_INCREF(other);
    return 0;
}


/*
 * Set user_structure for self.
 *
 * nd: number of dimension. nd should <= 16.
 *
 * dims: dimension info
 *
 */
int MKLNdarray_set_structure(MKLNdarray *self, int nd, const size_t *dims) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_set_structure: input MKLNdarray* self is NULL");
        return -1;
    }
    // assert (self->nd == nd);
    if (nd > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", nd, MAX_NDIM);
        return -1;
    }

    self->user_structure[0] = dims[0];
    self->user_structure[2 * nd - 1] = 1;
    for (int i = 1; i < nd; i++) {
        // nchw
        self->user_structure[i] = dims[i];
        // chw, hw, w, 1
        self->user_structure[2 * nd - 1 - i] = self->user_structure[2 * nd - i] * dims[nd - i];
    }

    return 0;
}


/*
 * Copy/construct a plain MKLNdarray with dada/structure from a PyArrayObject.
 *
 * Need check the dtype and ndim of PyArrayObject.
 *
 */
int MKLNdarray_CopyFromArray(MKLNdarray *self, PyArrayObject *obj) {
    if (self == NULL || obj == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CopyFromArray: input self or obj is NULL");
        return -1;
    }
    int ndim = PyArray_NDIM(obj);
    npy_intp* d = PyArray_DIMS(obj);
    int typenum = PyArray_TYPE(obj);

    if (typenum != MKL_FLOAT32 && typenum != MKL_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_CopyFromArray: can only copy from float/double arrays");
        return -1;
    }

    if (ndim < 0 || ndim > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", ndim, MAX_NDIM);
        return -1;
    }

    self->dtype = typenum;
    self->nd = ndim;

    PyArrayObject* py_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)obj, typenum, self->nd, self->nd);
    if (!py_src) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_CopyFromArray: fail to cast obj to contiguous array");
        return -1;
    }

    size_t dims[MAX_NDIM] = {0};
    size_t user_size = 1;

    for (int i = 0; i < ndim; i++) {
        dims[i] = (size_t)d[i];
        user_size *= dims[i];
    }

    int err = MKLNdarray_set_structure(self, ndim, dims);
    if (err < 0) {
        return err;
    }

    // prepare user layout and mkl buffer
    err = MKLNdarray_create_buffer_from_structure(self);
    if (err < 0) {
        return err;
    }

    // copy data to mkl buffer
    size_t element_size = (size_t)PyArray_ITEMSIZE(py_src);
    // assert (user_size * element_size <= self->data_size);
    memcpy((void*)self->private_data, (void*)PyArray_DATA(py_src), user_size * element_size);
    Py_DECREF(py_src);
    return 0;
}


/*
 * Create a MKLNdarray object according to input dims and typenum.
 *
 * Set all elements to zero.
 *
 * n: number of dimension
 * dims: dimension info
 *
 */
PyObject* MKLNdarray_create_with_zeros(int n, size_t *dims, int typenum) {
    size_t total_elements = 1;
    if (n < 0 || n > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", n, MAX_NDIM);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (dims[i] != 0 && total_elements > (SIZE_MAX / dims[i])) {
            PyErr_Format(PyExc_RuntimeError,
                         "Can't store in size_t for the bytes requested %llu * %llu",
                         (unsigned long long)total_elements,
                         (unsigned long long)dims[i]);
            return NULL;
        }
        total_elements *= dims[i];
    }

    // total_elements now contains the size of the array
    size_t max = 0;
    if (typenum == MKL_FLOAT64)
        max = SIZE_MAX / sizeof (double);
    else
        max = SIZE_MAX / sizeof (float);

    if (total_elements > max) {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't store in size_t for the bytes requested %llu",
                     (unsigned long long)total_elements);
        return NULL;
    }

    size_t total_size = 0;
    if (typenum == MKL_FLOAT64)
        total_size = total_elements * sizeof (double);
    else
        total_size = total_elements * sizeof (float);

    MKLNdarray* rval = (MKLNdarray*)MKLNdarray_New(n, typenum);
    if (!rval) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_create_with_zeros: call to New failed");
        return NULL;
    }

    if (MKLNdarray_set_structure(rval, n, dims)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_create_with_zeros: syncing structure to mkl failed.");
        Py_DECREF(rval);
        return NULL;
    }

    if (MKLNdarray_create_buffer_from_structure(rval)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarrya_create_with_zeros: create buffer from structure failed.");
        Py_DECREF(rval);
        return NULL;
    }
    // Fill with zeros
    memset(rval->private_data, 0, total_size);
    return (PyObject*)rval;
}


/*
 * Get shape info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a tuple contains dimension info.
 */
static PyObject*
MKLNdarray_get_shape(MKLNdarray *self, void *closure) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_shape: input MKLNdarray* self is NULL");
        return NULL;
    }
    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_get_shape: MKLNdarray not initialized");
        return NULL;
    }

    PyObject* rval = PyTuple_New(self->nd);
    if (rval == NULL) {
        return NULL;
    }

    for (int i = 0; i < self->nd; i++) {
        if (PyTuple_SetItem(rval, i, PyInt_FromLong(MKLNdarray_DIMS(self)[i]))) {
            Py_XDECREF(rval);
            return NULL;
        }
    }

    return rval;
}


/*
 * Get dtype info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a string: 'float32' or 'float64'.
 *
 */
static PyObject*
MKLNdarray_get_dtype(MKLNdarray *self, void *closure) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_dtype: input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_get_dtype: MKLNdarray not initialized");
        return NULL;
    }

    PyObject * rval = PyString_FromFormat("%s", MKL_TYPE[self->dtype]);
    return rval;
}


/*
 * Get ndim info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a integer number. If self is not initialized, -1 will be returned.
 *
 */
static PyObject*
MKLNdarray_get_ndim(MKLNdarray *self, void *closure) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_ndim: input MKLNdarray* self is NULL");
        return NULL;
    } else {
        return PyInt_FromLong(self->nd);
    }
}


/*
 * Get size info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a integer number.
 *
 */
static PyObject*
MKLNdarray_get_size(MKLNdarray *self, void *closure) {
    size_t total_element = 1;
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_size: input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd <= 0) {
        total_element = 0;
    } else {
        for (int i = 0; i < self->nd; i++) {
            total_element *= self->user_structure[i];
        }
    }
    return PyInt_FromLong(total_element);
}


/*
 * Get base info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a PyObject.
 *
 */
static PyObject*
MKLNdarray_get_base(MKLNdarray *self, void *closure) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_base: input MKLNdarray* self is NULL");
        return NULL;
    }

    PyObject * base = self->base;
    if (!base) {
        base = Py_None;
    }
    Py_INCREF(base);
    return base;
}


/*
 * Create a PyArrayObject from a MKLNdarray.
 */
PyObject* MKLNdarray_CreateArrayObj(MKLNdarray *self) {
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CreateArrayObj: input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 ||
        self->private_data == NULL ||
        self->private_layout == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CreateArrayObj: Can't convert from an uninitialized MKLNdarray");
        return NULL;
    }

    npy_intp npydims[MAX_NDIM] = {0};
    for (int i = 0; i < self->nd; i++) {
        npydims[i] = (npy_intp)self->user_structure[i];
    }

    PyArrayObject* rval = NULL;
    if (self->dtype == MKL_FLOAT64) {
        // float64
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT64);
    } else {
        // float32
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT32);
    }

    if (!rval) {
        return NULL;
    }

    void* rval_data = PyArray_DATA(rval);
    dnnLayout_t layout_user = NULL;
    int status = -1;
    dnnPrimitive_t primitive = NULL;

    size_t mkl_size[MAX_NDIM] = {0};
    size_t mkl_stride[MAX_NDIM] = {0};

    // nchw -> whcn
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i -1];
    }

    if (self->dtype == MKL_FLOAT64) { // float64
        status = dnnLayoutCreate_F64(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (status != 0 || layout_user == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F64 failed: %d, line: %d",
                         status, __LINE__);
            Py_DECREF(rval);
            return NULL;
        }

        if (!dnnLayoutCompare_F64(self->private_layout, layout_user)) {
            status = dnnConversionCreate_F64(&primitive, self->private_layout, layout_user);
            if (E_SUCCESS != status || NULL == primitive) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: dnnConversionCreate_F64 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                return NULL;
            }

            status = dnnConversionExecute_F64(primitive, (void*)self->private_data, (void*)rval_data);
            if (E_SUCCESS != status) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: dnnConversionExecute_F64 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                return NULL;
            }
        } else {
            memcpy((void*)rval_data, (void*)self->private_data, PyArray_SIZE(rval) * sizeof (double));
        }

        if (NULL != layout_user) {
            dnnLayoutDelete_F64(layout_user);
            layout_user = NULL;
        }
        if (NULL != primitive) {
            dnnDelete_F64(primitive);
            primitive = NULL;
        }

    } else {  // float32
        status = dnnLayoutCreate_F32(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (status != E_SUCCESS || layout_user == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F32 failed: %d, line: %d",
                         status, __LINE__);
            Py_DECREF(rval);
            return NULL;
        }

        if (!dnnLayoutCompare_F32(self->private_layout, layout_user)) {
            status = dnnConversionCreate_F32(&primitive, self->private_layout, layout_user);
            if (E_SUCCESS != status || NULL == primitive) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: dnnConversionCreate_F32 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                return NULL;
            }

            status = dnnConversionExecute_F32(primitive, (void*)self->private_data, (void*)rval_data);
            if (E_SUCCESS != status) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: dnnConversionExecute_F32 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                return NULL;
            }
        } else {
            memcpy((void*)rval_data, (void*)self->private_data, PyArray_SIZE(rval) * sizeof (float));
        }

        if (NULL != layout_user) {
            dnnLayoutDelete_F32(layout_user);
            layout_user = NULL;
        }
        if (NULL != primitive) {
            dnnDelete_F32(primitive);
            primitive = NULL;
        }
    }

    return (PyObject*)rval;
}


/*
 * Create a new MKLNdarray instance and set all elements to zero.
 * This function will be called when do MKLNdarray.zeros(shape, typenum) in python code.
 *
 * shape: a tuple contains shape info. length of shape should <= MAX_NDIM
 * typenum: MKL_FLAOT32, MKL_FLOAT64. MKL_FLOAT32 by default.
 *
 * This function will call MKLNdarray_create_with_zeros to do detailed processing.
 *
 */
PyObject* MKLNdarray_Zeros(PyObject *_unused, PyObject *args) {
    if (!args) {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_Zeros: function takes at least 1 argument");
        return NULL;
    }

    PyObject* shape = NULL;
    int typenum = -1;

    if (!PyArg_ParseTuple(args, "O|i", &shape, &typenum)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Zeros: PyArg_ParseTuple failed \n");
        return NULL;
    }

    if (typenum != MKL_FLOAT32 && typenum != MKL_FLOAT64) {
        typenum = MKL_FLOAT32;
    }

    if (!PySequence_Check(shape)) {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    int shplen = PySequence_Length(shape);
    if (shplen <= 0 || shplen > MAX_NDIM) {
        PyErr_Format(PyExc_TypeError, "length of shape argument must be 1 ~ %d", MAX_NDIM);
        return NULL;
    }

    size_t newdims[MAX_NDIM] = {0};
    for (int i = shplen -1; i >= 0; i--) {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if (shp_el_obj == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Zeros: index out of bound in sequence");
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        if (shp_el < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "MKLNdarray_Zeros: shape must contain only non-negative values for size of a dimension");
            return NULL;
        }
        newdims[i] = (size_t)shp_el;
    }

    PyObject* rval = MKLNdarray_create_with_zeros(shplen, newdims, typenum);
    return (PyObject*)rval;
}


/*
 * type:tp_methods
 * Describe methos of a type. ml_name/ml_meth/ml_flags/ml_doc.
 * ml_name: name of method
 * ml_meth: PyCFunction, point to the C implementation
 * ml_flags: indicate how the call should be constructed
 * ml_doc: docstring
 *
 */
static PyMethodDef MKLNdarray_methods[] = {
    {"__array__",
        (PyCFunction)MKLNdarray_CreateArrayObj, METH_VARARGS,
        "Copy from MKL to a numpy ndarray."},

    {"zeros",
        (PyCFunction)MKLNdarray_Zeros, METH_STATIC | METH_VARARGS,
        "Create a new MklNdarray with specified shape, filled with zeros."},

    {NULL, NULL, 0, NULL}  /* Sentinel */
};


/* type:tp_members
 * Describe attributes of a type. name/type/offset/flags/doc.
 */
static PyMemberDef MKLNdarray_members[] = {
    {NULL}      /* Sentinel */
};


/*
 * type:tp_getset
 * get/set attribute of instances of this type. name/getter/setter/doc/closure
 *
 */
static PyGetSetDef MKLNdarray_getset[] = {
    {"shape",
        (getter)MKLNdarray_get_shape,
        NULL,
        "shape of this ndarray (tuple)",
        NULL},

    {"dtype",
        (getter)MKLNdarray_get_dtype,
        NULL,
        "the dtype of the element.",
        NULL},

    {"size",
        (getter)MKLNdarray_get_size,
        NULL,
        "the number of elements in this object.",
        NULL},

    {"ndim",
        (getter)MKLNdarray_get_ndim,
        NULL,
        "the number of dimensions in this objec.",
        NULL},

    {"base",
        (getter)MKLNdarray_get_base,
        NULL,
        "if this ndarray is a view, base is the original ndarray.",
        NULL},

    {NULL, NULL, NULL, NULL}  /* Sentinel*/
};


/*
 * type object.
 * If you want to define a new object type, you need to create a new type object.
 *
 */
static PyTypeObject MKLNdarrayType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "MKLNdarray",              /*tp_name*/
    sizeof(MKLNdarray),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MKLNdarray_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    MKLNdarray_repr,           /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
#endif
    "MKLNdarray objects",      /*tp_doc */
    0,                         /*tp_traverse*/
    0,                         /*tp_clear*/
    0,                         /*tp_richcompare*/
    0,                         /*tp_weaklistoffset*/
    0,                         /*tp_iter*/
    0,                         /*tp_iternext*/
    MKLNdarray_methods,        /*tp_methods*/
    MKLNdarray_members,        /*tp_members*/
    MKLNdarray_getset,         /*tp_getset*/
    0,                         /*tp_base*/
    0,                         /*tp_dict*/
    0,                         /*tp_descr_get*/
    0,                         /*tp_descr_set*/
    0,                         /*tp_dictoffset*/
    (initproc)MKLNdarray_init, /*tp_init*/
    0,                         /*tp_alloc*/
    MKLNdarray_new,            /*tp_new*/
};


/*
 * Check an input is an instance of MKLNdarray or not.
 * Same as PyArray_Check
 *
 */
int MKLNdarray_Check(const PyObject *ob) {
    return ((Py_TYPE(ob) == &MKLNdarrayType) ? 1 : 0);
}


/*
 * Try to new a MKLNdarray instance.
 * This function is different from MKLNdarray_new which is set for tp_new.
 *
 * MKLNdarray_new is call by python when python has allocate memory for it already.
 * MKLNdarray_New will be called manually in C/C++ code, so we need to call tp_alloc manually.
 *
 */
PyObject*
MKLNdarray_New(int nd, int typenum) {
    if (nd < 0 || nd > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_New: not support a %d-dim array. Try array which ndim is <= %d. line: %d",
                     nd, MAX_NDIM, __LINE__);
        return NULL;
    }

    MKLNdarray* self = (MKLNdarray*)(MKLNdarrayType.tp_alloc(&MKLNdarrayType, 0));
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_New: failed to allocate self");
        return NULL;
    }
    self->base              = NULL;
    self->nd                = nd;
    self->dtype             = typenum;
    self->private_data      = NULL;
    self->private_workspace = NULL;
    self->data_size         = 0;
    self->private_layout    = NULL;
    memset((void*)(self->user_structure), 0, 2 * MAX_NDIM * sizeof (size_t));

    return (PyObject*)self;
}


/*
 * Declare methods belong to this module but not in MKLNdarray.
 * Used in module initialization function.
 *
 * Users can access these methods by module.method_name() after they have imported this module.
 *
 */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}   /* Sentinel */
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef mkl_ndarray_moduledef =
{
    PyModuleDef_HEAD_INIT,
    "mkl_ndarray",
    "MKL implementation of a numpy ndarray-like object.",
    -1,
    module_methods
};

PyMODINIT_FUNC
PyInit_mkl_ndarray(void)
#else
PyMODINIT_FUNC
initmkl_ndarray(void)
#endif
{
    import_array();
    PyObject* m = NULL;

    if (PyType_Ready(&MKLNdarrayType) < 0) {
#if PY_MAJOR_VERSION == 3
        return NULL;
#else
        return;
#endif
    }

    // add attribute to MKLNdarrayType
    // if user has import MKLNdarrayType already, they can get typenum of float32 and float64
    // by MKLNdarray.float32 or MKLNdarray.float64
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float32", PyInt_FromLong(MKL_FLOAT32));
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float64", PyInt_FromLong(MKL_FLOAT64));
#if PY_MAJOR_VERSION == 3
    m = PyModule_Create(&mkl_ndarray_moduledef);
#else
    m = Py_InitModule3("mkl_ndarray", module_methods, "MKL implementation of a numpy ndarray-like object.");
#endif
    if (m == NULL) {
#if PY_MAJOR_VERSION == 3
        return NULL;
#else
        return;
#endif
    }
    Py_INCREF(&MKLNdarrayType);
    PyModule_AddObject(m, "MKLNdarray", (PyObject*)&MKLNdarrayType);
#if PY_MAJOR_VERSION == 3
    return m;
#else
    return;
#endif
}
