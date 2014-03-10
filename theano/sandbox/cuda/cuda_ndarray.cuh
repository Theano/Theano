#ifndef _CUDA_NDARRAY_H
#define _CUDA_NDARRAY_H

// Defines for Python 2/3 compatibility.
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

// Python 3 expects a PyObject* as the first argument to PySlice_GetIndicesEx().
#define SLICE_CAST(x) (x)
#else
// Python 2 expects a PySliceObject* as the first argument to PySlice_GetIndicesEx().
#define SLICE_CAST(x) ((PySliceObject*)(x))
#endif // end #if PY_MAJOR_VERSION >= 3

#ifndef Py_TYPE
#  define Py_TYPE(o) ((o)->ob_type)
#endif
#ifndef Py_REFCNT
#  define Py_REFCNT(o) ((o)->ob_refcnt)
#endif

#include <numpy/arrayobject.h>
#include <stdio.h>

#include <cublas.h>

#ifdef _WIN32
#ifdef _CUDA_NDARRAY_C
#define DllExport   __declspec( dllexport )
#else
#define DllExport   __declspec( dllimport )
#endif
#define ALWAYS_INLINE
#else //else _WIN32
#define DllExport
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

typedef float real;
#define REAL_TYPENUM 11

#ifdef __DEVICE_EMULATION__
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     1  //This prevents printf from getting tangled up
#else
#define NUM_VECTOR_OP_BLOCKS                4096 //Max number of blocks to launch.  Should be read from device properties. (#10)
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     256  //Should be read from device properties. (#10)
#endif

#if 1
// Do not wait after every kernel & transfer.
#define CNDA_THREAD_SYNC
#else
// This is useful for using normal profiling tools
#define CNDA_THREAD_SYNC cudaThreadSynchronize();
#endif


#ifndef SHARED_SIZE
#define SHARED_SIZE (16*1024)
#endif

#define VERBOSE_DEVICE_MALLOC 1
#define NO_VERBOSE_DEVICE_MALLOC 0

/**
 * Allocation and freeing of device memory should go through these functions so that the lib can track memory usage.
 *
 * device_malloc will set the Python error message before returning None.
 * device_free will return nonzero on failure (after setting the python error message)
 */
DllExport void * device_malloc(size_t size);
DllExport void * device_malloc(size_t size, int verbose);
DllExport int device_free(void * ptr);

template <typename T>
static T ceil_intdiv(T a, T b)
{
    return (a/b) + ((a % b) ? 1: 0);
}

/**
 * struct CudaNdarray
 *
 * This is a Python type.
 *
 */
struct CudaNdarray
{
    PyObject_HEAD

    /**
     * base:
     *  either NULL or a pointer to a fellow CudaNdarray into which this one is viewing.
     *  This pointer is never followed, except during Py_DECREF when we do not need it any longer.
     */
    PyObject * base;

    /* Type-specific fields go here. */
    //GpuTensorType::VoidTensor * vt;
    int nd; //the number of dimensions of the tensor
    // Client should acces host_structure via CudaNdarray_HOST_DIMS / CudaNdarray_HOST_STRIDES functions
    int * host_structure; //dim0, dim1, ... stride0, stride1, ...
    int data_allocated; //the number of bytes allocated for devdata


    //device pointers (allocated by cudaMalloc)
    mutable int dev_structure_fresh;
    //dev_structure should be accessed via the functions like
    //CudaNdarray_DEV_DIMS, otherwise may not be
    //synchronized with host_structure. The accessor functions will allocate it when needed.
    mutable int * dev_structure; //dim0, dim1, ..., stride0, stride1, ...
    real* devdata; //pointer to data element [0,..,0].
};


enum operator_t
{
    IADD=0,
    IDIV,
    CPY,
    N_ELEMWISE_OPS // This is to know the number of operation
};

/*
 * Return a CudaNdarray whose 'nd' dimensions are all 0.
 * if nd==-1, it is not initialized.
 */
DllExport PyObject *
CudaNdarray_New(int nd=-1);

/**
 * Return 1 for a CudaNdarray otw 0
 */
DllExport int
CudaNdarray_Check(const PyObject * ob);

/**
 * Return 1 for a CudaNdarray otw 0
 */
DllExport int
CudaNdarray_CheckExact(const PyObject * ob);

/**
 * Return true for a C-contiguous CudaNdarray, else false
 */
DllExport bool
CudaNdarray_is_c_contiguous(const CudaNdarray * self);

/****
 * Returns the number of elements necessary in host_structure and dev_structure for a given number of dimensions.
 */
DllExport int cnda_structure_size(int nd);

/*
 * This describes the shape of the ndarray. The array
 * of dimensions is itself stored on the host.
 * If you need to access the dimensions array from inside
 * a kernel, use CudaNdarray_DEVICE_DIMS.
 */
DllExport const int *
CudaNdarray_HOST_DIMS(const CudaNdarray * self);

DllExport const int *
CudaNdarray_HOST_STRIDES(const CudaNdarray * self);

DllExport const int *
CudaNdarray_HOST_LOG2DIMS(const CudaNdarray * self);

DllExport inline void ALWAYS_INLINE
cnda_mark_dev_structure_dirty(CudaNdarray * self)
{
    self->dev_structure_fresh = 0;
}


DllExport int
CudaNdarray_EqualAndIgnore(CudaNdarray *cnda1, CudaNdarray *cnda2, int ignoreSync, int ignoreBase);

// Default: do not ignore sync of dev and host structures in comparing, and do not ignore difference in base pointers
DllExport int
CudaNdarray_Equal(CudaNdarray *cnda1, CudaNdarray *cnda2);

/****
 *  Set the dimension[idx] to value d.
 *
 *  Updates the log2dim shadow array.
 *
 *  Does not sync structure to device.
 */
DllExport inline void ALWAYS_INLINE
CudaNdarray_set_dim(CudaNdarray * self, int idx, int d) 
{
    if ((idx >= self->nd) || (idx < 0) || (d < 0))
    {
        fprintf(stderr, "WARNING: probably bad CudaNdarray_set_dim arguments: self->ndim=%i, idx=%i stride=%i\n",
                self->nd, idx, d);
    }

    if (d != self->host_structure[idx])
    {
        self->host_structure[idx] = d;
        int log2d = (int)log2((double)d);
        self->host_structure[idx + 2*self->nd] = (d == (1 << log2d)) ? log2d : -1;
        cnda_mark_dev_structure_dirty(self);
    }
}


DllExport inline void ALWAYS_INLINE
CudaNdarray_set_stride(CudaNdarray * self, int idx, int s)
{
    if ((idx >= self->nd) || (idx < 0))
    {
        fprintf(stderr, "WARNING: probably bad CudaNdarray_set_stride arguments: %i %i\n", idx, s);
    }

    if (s != CudaNdarray_HOST_STRIDES(self)[idx])
    {
        self->host_structure[idx+self->nd] = s;
        cnda_mark_dev_structure_dirty(self);
    }
}

/***
 *  Update dependent variables from the contents of CudaNdarray_HOST_DIMS(self) and CudaNdarray_HOST_STRIDES(self)
 *
 *  This means: recalculate the log2dims and transfer structure to the card
 */
DllExport int cnda_copy_structure_to_device(const CudaNdarray * self);

/* CudaNdarray_DEV_DIMS gives the same information as CudaNdarray_HOST_DIMS,
 * but stored on the GPU. Use this pointer when it needs to be accessed
 * from inside a CUDA kernel.
 */
DllExport const int *CudaNdarray_DEV_DIMS(const CudaNdarray * self);
DllExport const int *CudaNdarray_DEV_STRIDES(const CudaNdarray * self);
DllExport const int *CudaNdarray_DEV_LOG2DIMS(const CudaNdarray * self);
DllExport float *CudaNdarray_DEV_DATA(const CudaNdarray * self);

// The following 4 macro are here to help make c code generator that work on
// both PyArray and CudaNdarray.  This is at least used for Subtensor and
// GpuSubtensor
#define CudaNdarray_DIMS CudaNdarray_HOST_DIMS
#define CudaNdarray_NDIM(self) self->nd
#define CudaNdarray_STRIDES CudaNdarray_HOST_STRIDES
#define CudaNdarray_BYTES CudaNdarray_DEV_DATA

/**
 * Return the number of elements in the ndarray (product of the dimensions)
 */
DllExport int CudaNdarray_SIZE(const CudaNdarray *self);

static PyObject *CudaNdarray_SIZE_Object(const CudaNdarray *self, void *closure);

/**
 * Allocate a new CudaNdarray with room for given number of dimensions
 *
 * No Storage space is allocated (and all dimensions are 0)
 */
DllExport PyObject * CudaNdarray_new_nd(const int nd);

/**
 * [Re]allocate a CudaNdarray with access to 'nd' dimensions.
 *
 * Note: This does not allocate storage for data, or free
 *       pre-existing storage.
 */
DllExport inline int ALWAYS_INLINE
CudaNdarray_set_nd(CudaNdarray * self, const int nd)
{
    if (nd != self->nd)
    {
        if (self->dev_structure)
        {
            if (device_free(self->dev_structure))
            {
                return -1;
            }
            self->dev_structure = NULL;
        }
        if (self->host_structure)
        {
            free(self->host_structure);
            self->host_structure = NULL;
            self->nd = -1;
        }
        if (nd == -1) return 0;

        self->host_structure = (int*)malloc(cnda_structure_size(nd)*sizeof(int));
        if (NULL == self->host_structure)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate dim or str");
            return -1;
        }
        //initialize all dimensions and strides to 0
        for (int i = 0; i < cnda_structure_size(nd); ++i)
        {
            self->host_structure[i] = 0;
        }
        //The device structure will be created in cnda_copy_structure_to_device
        //if needed.
        self->nd = nd;
        self->dev_structure_fresh = 0;
    }
    return 0;
}


/**
 * CudaNdarray_alloc_contiguous
 *
 * Allocate storage space for a tensor of rank 'nd' and given dimensions.
 * (No-op if self already has a contiguous tensor of the right dimensions)
 *
 * Note: CudaNdarray_alloc_contiguous is templated to work for both int dimensions and npy_intp dimensions
 */
template<typename inttype>
static int CudaNdarray_alloc_contiguous(CudaNdarray *self, const int nd, const inttype * dim)
{
    // allocate an empty ndarray with c_contiguous access
    // return 0 on success
    int size = 1; //set up the strides for contiguous tensor
    assert (nd >= 0);

    // Here we modify the host structure to have the desired shape and
    // strides. This does not cause the storage to be freed or reallocated.
    if (CudaNdarray_set_nd(self, nd))
    {
        return -1;
    }
    for (int i = nd-1; i >= 0; --i)
    {
        CudaNdarray_set_stride(self, i, (dim[i] == 1) ? 0 : size);
        CudaNdarray_set_dim(self, i, dim[i]);
        size = size * dim[i];
    }

    // If the allocated buffer is already of the right size, we don't need to
    // do anything else.
    // Note: self->data_allocated is 0 for a view, so views will fail this
    // check and be turned into independent arrays below.
    if (self->data_allocated == size)
    {
        return 0;
    }

    // The structure of self will be reused with newly allocated memory.
    // If self was a view, we should remove the reference to its base.
    // (If base was already NULL, the following has no effect.)
    Py_XDECREF(self->base);
    self->base = NULL;

    // If self is a view, do not try to free its memory
    if (self->data_allocated && device_free(self->devdata))
    {
        self->devdata = NULL;
        self->data_allocated = 0;
        return -1;
    }

    if (size < 0)
    {
        PyErr_Format(PyExc_AssertionError,
                     "size (%i) < 0",
                     size);
        return -1;
    }

    self->devdata = (float*)device_malloc(size*sizeof(real));
    if (size && !self->devdata)
    {
        CudaNdarray_set_nd(self, -1);
        self->data_allocated = 0;
        self->devdata = 0;
        return -1;
    }
    if (0)
        fprintf(stderr,
            "Allocated devdata %p (self=%p)\n",
            self->devdata,
            self);
    self->data_allocated = size;

    return 0;
}

/*
 * Return a CudaNdarray whose 'nd' dimensions are set to dims, and allocated.
 */
template<typename inttype> 
static PyObject *CudaNdarray_NewDims(int nd, const inttype * dims)
{
    CudaNdarray * rval = (CudaNdarray*)CudaNdarray_New();
    if (rval)
    {
        if (CudaNdarray_alloc_contiguous(rval, nd, dims))
        {
            Py_DECREF(rval);
            return NULL;
        }
    }
    return (PyObject*)rval;
}


/**
 * CudaNdarray_set_device_data
 *
 * Set self to be a view of given `data`, owned by existing CudaNdarray `base`.
 */
DllExport int CudaNdarray_set_device_data(CudaNdarray * self, float * data, PyObject * base);
DllExport int CudaNdarray_set_device_data(CudaNdarray * self, float * data, const CudaNdarray * base);

/**
 * Return an independent copy of self
 */
DllExport PyObject * CudaNdarray_DeepCopy(CudaNdarray * self, PyObject * memo);

/**
 * Return an independent copy of self
 */
DllExport PyObject * CudaNdarray_Copy(const CudaNdarray * self);

/**
 * Return a new object obtained by summing over the dimensions for which there is a 1 in the mask.
 */
DllExport PyObject * CudaNdarray_ReduceSum(CudaNdarray * self, PyObject * py_reduce_mask);

/**
 * Transfer the contents of numpy array `obj` to `self`.
 *
 * self is reallocated to have the correct dimensions if necessary.
 */
DllExport int CudaNdarray_CopyFromArray(CudaNdarray * self, PyArrayObject*obj);

/**
 * Transfer the contents of CudaNdarray `other` to `self`.
 *
 * self is reallocated to have the correct dimensions if necessary.
 * TODO: WRITEME: what does "if necessary" mean?
 * TODO: we use this to implement set/inc subtensor, where self is a view of
 *       the original tensor so that we write only to the subtensor. How
 *       do we ensure that self is not reallocated in this case?
 *
 *  unbroadcast: if true, this means that other is broadcastable in some
 *               dimensions, and the result, self, is not.
 *               ie, if unbroadcast=false, we must do the broadcasting
 *               operation as part of the copy.
 *               e.g. suppose self and other are 2D matrices and other
 *               has only one row. Then we need to copy this row several
 *               times when copying to self.
 */
DllExport int CudaNdarray_CopyFromCudaNdarray(CudaNdarray * self,
        const CudaNdarray * other, bool unbroadcast = false);

/**
 * Transfer the contents of CudaNdarray `self` to a new numpy ndarray.
 */
DllExport PyObject *
CudaNdarray_CreateArrayObj(CudaNdarray * self, PyObject *args = NULL);

DllExport PyObject *
CudaNdarray_ZEROS(int n, int * dims);

/**
 * True iff the strides look like [dim[nd-2], dim[nd-3], ... , dim[0], 1]
 */
DllExport inline bool ALWAYS_INLINE
CudaNdarray_is_c_contiguous(const CudaNdarray * self)
{
    bool c_contiguous = true;
    int size = 1;
    for (int i = self->nd-1; (i >= 0) && c_contiguous; --i)
    {
        if (CudaNdarray_HOST_DIMS(self)[i] == 1)
            continue;
        if (CudaNdarray_HOST_STRIDES(self)[i] != size)
        {
            c_contiguous = false;
        }
        size = size * CudaNdarray_HOST_DIMS(self)[i];
    }
    return c_contiguous;
}

DllExport PyObject * CudaNdarray_IS_C_Contiguous(CudaNdarray * self);

DllExport int CudaNdarray_gemm(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C);
DllExport int CudaNdarray_sgemv(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C);
DllExport int CudaNdarray_sger(float alpha, const CudaNdarray * x, const CudaNdarray * y, CudaNdarray* A);

DllExport int CudaNdarray_reduce_sum(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_prod(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_min(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_max(CudaNdarray * self, CudaNdarray * A);

DllExport int CudaNdarray_dimshuffle(CudaNdarray * self, unsigned int len, const int * pattern);
DllExport PyObject*
CudaNdarray_TakeFrom(CudaNdarray * self, PyObject *args);

int fprint_CudaNdarray(FILE * fd, const CudaNdarray *self);


DllExport PyObject * CudaNdarray_View(const CudaNdarray * self);
DllExport PyObject * CudaNdarray_inplace_add(PyObject* py_self, PyObject * py_other);
DllExport PyObject * CudaNdarray_Subscript(PyObject * py_self, PyObject * key);
DllExport int CudaNdarray_inplace_elemwise(PyObject* py_self, PyObject * py_other, operator_t fct_nb);

// Ensures that *arr is a pointer to a contiguous ndarray of the specified
// dimensions.
// *arr may initially be NULL, a pointer to an ndarray of the wrong size,
// or a pointer to an ndarray of the right size. In the last case it will
// not change.
DllExport int CudaNdarray_prep_output(CudaNdarray ** arr, int nd,
        const int * dims);

DllExport inline const char* ALWAYS_INLINE cublasGetErrorString(cublasStatus err){
    if(CUBLAS_STATUS_SUCCESS == err)
        return "success";
    else if(CUBLAS_STATUS_NOT_INITIALIZED == err)
        return "the library was not initialized";
    else if(CUBLAS_STATUS_ALLOC_FAILED == err)
        return "the resource allocation failed";
    else if(CUBLAS_STATUS_INVALID_VALUE == err)
        return "the parameters n<0 or incx,incy=0";
    else if(CUBLAS_STATUS_MAPPING_ERROR == err)
        return "an access to GPU memory space failed";
    else if(CUBLAS_STATUS_EXECUTION_FAILED == err)
        return "the function failed to launch on the GPU";
    else if(CUBLAS_STATUS_INTERNAL_ERROR == err)
        return "an internal operation failed";
    else
        return "unknow code";
}

#endif
/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
