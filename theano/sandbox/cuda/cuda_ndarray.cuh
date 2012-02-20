#ifndef _CUDA_NDARRAY_H
#define _CUDA_NDARRAY_H

#include <numpy/arrayobject.h>
#include <stdio.h>

#include <cublas.h>

#ifdef _WIN32
#ifdef _CUDA_NDARRAY_C
#define DllExport   __declspec( dllexport )
#else
#define DllExport   __declspec( dllimport )
#endif
#else
#define DllExport
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

#if 0
// Do not wait after every kernel & transfer.
#define CNDA_THREAD_SYNC
#else
// This is useful for using normal profiling tools
#define CNDA_THREAD_SYNC cudaThreadSynchronize();
#endif


#ifndef SHARED_SIZE
#define SHARED_SIZE (16*1024)
#endif

/**
 * Allocation and freeing of device memory should go through these functions so that the lib can track memory usage.
 *
 * device_malloc will set the Python error message before returning None.
 * device_free will return nonzero on failure (after setting the python error message)
 */
DllExport void * device_malloc(size_t size);
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
    // Client should acces host_structure via CudaNdarray_HOST_DIMS / CudaNdarray_HOST_STRIDES macros
    int * host_structure; //dim0, dim1, ... stride0, stride1, ...
    int data_allocated; //the number of bytes allocated for devdata


    //device pointers (allocated by cudaMalloc)
    mutable int dev_structure_fresh;
    //dev_structure should be accessed via macros, otherwise may not be synchronized
    int * dev_structure; //dim0, dim1, ..., stride0, stride1, ...
    real* devdata; //pointer to data element [0,..,0].
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

DllExport const int *
CudaNdarray_HOST_DIMS(const CudaNdarray * self);

DllExport const int *
CudaNdarray_HOST_STRIDES(const CudaNdarray * self);

DllExport const int *
CudaNdarray_HOST_LOG2DIMS(const CudaNdarray * self);

DllExport void
cnda_mark_dev_structure_dirty(CudaNdarray * self);

DllExport int
CudaNdarray_EqualAndIgnore(CudaNdarray *cnda1, CudaNdarray *cnda2, int ignoreSync, int ignoreBase);

// Default: do not ignore sync of dev and host structures in comparing, and do not ignore difference in base pointers
DllExport int
CudaNdarray_Equal(CudaNdarray *cnda1, CudaNdarray *cnda2);

/****
 *  Set the idx'th dimension to value d.
 *
 *  Updates the log2dim shaddow array.
 *
 *  Does not sync structure to host.
 */
DllExport void
CudaNdarray_set_dim(CudaNdarray * self, int idx, int d);

DllExport void
CudaNdarray_set_stride(CudaNdarray * self, int idx, int s);

/***
 *  Update dependent variables from the contents of CudaNdarray_HOST_DIMS(self) and CudaNdarray_HOST_STRIDES(self)
 *
 *  This means: recalculate the log2dims and transfer structure to the card
 */
DllExport int cnda_copy_structure_to_device(const CudaNdarray * self);

DllExport const int *CudaNdarray_DEV_DIMS(const CudaNdarray * self);
DllExport const int *CudaNdarray_DEV_STRIDES(const CudaNdarray * self);
DllExport const int *CudaNdarray_DEV_LOG2DIMS(const CudaNdarray * self);
DllExport float *CudaNdarray_DEV_DATA(const CudaNdarray * self);

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
 * Note: This does not allocate storage for data.
 */
DllExport int CudaNdarray_set_nd(CudaNdarray * self, const int nd);

/**
 * CudaNdarray_alloc_contiguous
 *
 * Allocate storage space for a tensor of rank 'nd' and given dimensions.
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
    if (CudaNdarray_set_nd(self, nd))
    {
        return -1;
    }
    //TODO: check if by any chance our current dims are correct,
    //      and strides already contiguous
    //      in that case we can return right here.
    for (int i = nd-1; i >= 0; --i)
    {
        CudaNdarray_set_stride(self, i, (dim[i] == 1) ? 0 : size);
        CudaNdarray_set_dim(self, i, dim[i]);
        size = size * dim[i];
    }

    if (CudaNdarray_is_c_contiguous(self) && (self->data_allocated == size))
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
        PyErr_SetString(PyExc_RuntimeError,
                        "Could not allocate memory on device");
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
DllExport int CudaNdarray_set_device_data(CudaNdarray * self, float * data, CudaNdarray * base);

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
 */
DllExport int CudaNdarray_CopyFromCudaNdarray(CudaNdarray * self, const CudaNdarray * other, bool unbroadcast = false);

/**
 * Transfer the contents of CudaNdarray `self` to a new numpy ndarray.
 */
DllExport PyObject *
CudaNdarray_CreateArrayObj(CudaNdarray * self);

DllExport PyObject *
CudaNdarray_ZEROS(int n, int * dims);

/**
 * True iff the strides look like [dim[nd-2], dim[nd-3], ... , dim[0], 1]
 */
DllExport bool CudaNdarray_is_c_contiguous(const CudaNdarray * self);
DllExport PyObject * CudaNdarray_IS_C_Contiguous(CudaNdarray * self);

DllExport int CudaNdarray_gemm(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C);
DllExport int CudaNdarray_sgemv(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C);
DllExport int CudaNdarray_sger(float alpha, const CudaNdarray * x, const CudaNdarray * y, CudaNdarray* A);

DllExport int CudaNdarray_reduce_sum(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_prod(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_min(CudaNdarray * self, CudaNdarray * A);
DllExport int CudaNdarray_reduce_max(CudaNdarray * self, CudaNdarray * A);

DllExport int CudaNdarray_dimshuffle(CudaNdarray * self, unsigned int len, const int * pattern);

static void fprint_CudaNdarray(FILE * fd, const CudaNdarray *self);

#endif
/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  c-file-offsets:((innamespace . 0)(inline-open . 0))
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
