#ifndef _CUDA_NDARRAY_H
#define _CUDA_NDARRAY_H

#include <numpy/arrayobject.h>
#include <stdio.h>

#include <cublas.h>

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
void * device_malloc(size_t size);
int device_free(void * ptr);

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
    int dev_structure_fresh;
	//dev_structure should be accessed via macros, otherwise may not be synchronized
    int * dev_structure; //dim0, dim1, ..., stride0, stride1, ...  
    real* devdata; //pointer to data element [0,..,0].
};

/*
 * Return a CudaNdarray whose 'nd' dimensions are all 0.
 */
PyObject * 
CudaNdarray_New(int nd);

/**
 * Return 1 for a CudaNdarray otw 0
 */
int 
CudaNdarray_Check(const PyObject * ob);

/**
 * Return 1 for a CudaNdarray otw 0
 */
int 
CudaNdarray_CheckExact(const PyObject * ob);

/****
 * Returns the number of elements necessary in host_structure and dev_structure for a given number of dimensions.
 */
int 
cnda_structure_size(int nd)
{
    // dim0, dim1, ...
    // str0, str1, ...
    // log2(dim0), log2(dim1), ...
    return nd + nd + nd;
}

const int * 
CudaNdarray_HOST_DIMS(const CudaNdarray * self)
{
    return self->host_structure;
}
const int * 
CudaNdarray_HOST_STRIDES(const CudaNdarray * self)
{
    return self->host_structure + self->nd;
}
const int * 
CudaNdarray_HOST_LOG2DIMS(const CudaNdarray * self)
{
    return self->host_structure + 2*self->nd;
}

void 
cnda_mark_dev_structure_dirty(CudaNdarray * self)
{
    self->dev_structure_fresh = 0;
}

int
CudaNdarray_EqualAndIgnore(CudaNdarray *cnda1, CudaNdarray *cnda2, int ignoreSync, int ignoreBase)
{
    int verbose = 1;

    if (!ignoreSync && cnda1->dev_structure_fresh != cnda2->dev_structure_fresh)
    {
        if(verbose) fprintf(stdout, "CUDANDARRAY_EQUAL FAILED : 1\n");
        return 0;
    }

    if (cnda1->nd != cnda2->nd)
    {
        if(verbose) fprintf(stdout, "CUDANDARRAY_EQUAL FAILED : 2\n");
        return 0;
    }

    for (int i=0; i < 2*cnda1->nd; i++)
    {
        if (cnda1->host_structure[i] != cnda2->host_structure[i])
        {
            if(verbose)
                fprintf(stdout, "CUDANDARRAY_EQUAL : host_structure : %d, %d, %d\n", i, cnda1->host_structure[i], cnda2->host_structure[i]);
            return 0;
        }
    }

    if (!ignoreBase && cnda1->base != cnda2->base)
    {
        if(verbose) fprintf(stdout, "CUDANDARRAY_EQUAL FAILED : 4");
        return 0;
    }
    else if (cnda1->data_allocated != cnda2->data_allocated)
    {
        if(verbose) fprintf(stdout, "CUDANDARRAY_EQUAL FAILED : 5");
        return 0;
    }
    else if (cnda1->data_allocated && cnda1->devdata != cnda2->devdata)
    {
        if(verbose) fprintf(stdout, "CUDANDARRAY_EQUAL FAILED : 6");
        // no need to check devdata if data is not allocated
        return 0;
    }

    return 1;
}

// Default: do not ignore sync of dev and host structures in comparing, and do not ignore difference in base pointers
int
CudaNdarray_Equal(CudaNdarray *cnda1, CudaNdarray *cnda2)
{
    return CudaNdarray_EqualAndIgnore(cnda1, cnda2, 0, 0);
}

/****
 *  Set the idx'th dimension to value d.
 *
 *  Updates the log2dim shaddow array.
 *
 *  Does not sync structure to host.
 */
void 
CudaNdarray_set_dim(CudaNdarray * self, int idx, int d)
{
    if ((idx >= self->nd) || (idx < 0) || (d < 0))
    {
        fprintf(stderr, "WARNING: probably bad CudaNdarray_set_dim arguments: %i %i\n", idx, d);
    }

    if (d != self->host_structure[idx])
    {
        self->host_structure[idx] = d;
        int log2d = (int)log2((double)d);
        self->host_structure[idx + 2*self->nd] = (d == (1 << log2d)) ? log2d : -1;
        cnda_mark_dev_structure_dirty(self);
    }
}
void 
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
int 
cnda_copy_structure_to_device(CudaNdarray * self)
{
    cublasSetVector(cnda_structure_size(self->nd), sizeof(int), self->host_structure, 1, self->dev_structure, 1);
    CNDA_THREAD_SYNC;
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying structure to device memory");
        return -1;
    }
    self->dev_structure_fresh = 1;
    return 0;
}

const int * 
CudaNdarray_DEV_DIMS(CudaNdarray * self)
{
    if (!self->dev_structure_fresh)
    {
        if (cnda_copy_structure_to_device(self))
            return NULL;
    }
    return self->dev_structure;
}
const int * 
CudaNdarray_DEV_STRIDES(CudaNdarray * self)
{
    if (!self->dev_structure_fresh)
    {
        if (cnda_copy_structure_to_device(self))
            return NULL;
    }
    return self->dev_structure + self->nd;
}
const int * 
CudaNdarray_DEV_LOG2DIMS(CudaNdarray * self)
{
    if (!self->dev_structure_fresh)
    {
        if (cnda_copy_structure_to_device(self))
            return NULL;
    }
    return self->dev_structure + 2*self->nd;
}
float * 
CudaNdarray_DEV_DATA(const CudaNdarray * self)
{
    return self->devdata;
}

/**
 * Return the number of elements in the ndarray (product of the dimensions)
 */
int 
CudaNdarray_SIZE(const CudaNdarray *self)
{
    if (self->nd == -1) return 0;
    int size = 1;
    for (int i = 0; i < self->nd; ++i)
    {
        size *= CudaNdarray_HOST_DIMS(self)[i];
    }
    return size;
}
static PyObject * 
CudaNdarray_SIZE_Object(const CudaNdarray *self, void *closure)
{
    return PyInt_FromLong(CudaNdarray_SIZE(self));
}


/**
 * Allocate a new CudaNdarray with nd==-1
 */
PyObject * CudaNdarray_new_null();

/**
 * Allocate a new CudaNdarray with room for given number of dimensions
 *
 * No Storage space is allocated (and all dimensions are 0)
 */
PyObject * CudaNdarray_new_nd(const int nd);

/**
 * [Re]allocate a CudaNdarray with access to 'nd' dimensions.
 *
 * Note: This does not allocate storage for data.
 */
int CudaNdarray_set_nd(CudaNdarray * self, const int nd)
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
        //initialize all dimensions and strides to 0
        for (int i = 0; i < cnda_structure_size(nd); ++i) self->host_structure[i] = 0;
        if (NULL == self->host_structure)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate dim or str");
            return -1;
        }
        int struct_size = cnda_structure_size(nd);
        if (struct_size)
        {
            self->dev_structure = (int*)device_malloc(struct_size* sizeof(int));
            if (NULL == self->dev_structure)
            {
                free(self->host_structure);
                self->host_structure = NULL;
                self->dev_structure = NULL;
                return -1;
            }
        }
        self->nd = nd;
        self->dev_structure_fresh = 0;
    }
    return 0;
}

/**
 * CudaNdarray_alloc_contiguous
 *
 * Allocate storage space for a tensor of rank 'nd' and given dimensions.
 *
 * Note: CudaNdarray_alloc_contiguous is templated to work for both int dimensions and npy_intp dimensions
 */
template<typename inttype>
int CudaNdarray_alloc_contiguous(CudaNdarray *self, const int nd, const inttype * dim)
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

    if (self->data_allocated != size)
    {
        if (device_free(self->devdata))
        {
            // Does this ever happen??  Do we need to set data_allocated or devdata to 0?
            return -1;
        }
        assert(size>0);
        self->devdata = (float*)device_malloc(size*sizeof(real));
        if (!self->devdata)
        {
            CudaNdarray_set_nd(self,-1);
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
    }
    return 0;
}

/*
 * Return a CudaNdarray whose 'nd' dimensions are set to dims, and allocated.
 */
template<typename inttype>
PyObject * 
CudaNdarray_NewDims(int nd, const inttype * dims)
{
    CudaNdarray * rval = (CudaNdarray*)CudaNdarray_new_null();
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
int CudaNdarray_set_device_data(CudaNdarray * self, float * data, CudaNdarray * base);

/**
 * Return an independent copy of self
 */
PyObject * CudaNdarray_DeepCopy(CudaNdarray * self, PyObject * memo);

/**
 * Return an independent copy of self
 */
PyObject * CudaNdarray_Copy(CudaNdarray * self);

/**
 * Return a new object obtained by summing over the dimensions for which there is a 1 in the mask.
 */
PyObject * CudaNdarray_ReduceSum(CudaNdarray * self, PyObject * py_reduce_mask);

/**
 * Transfer the contents of numpy array `obj` to `self`.
 *
 * self is reallocated to have the correct dimensions if necessary.
 */
int CudaNdarray_CopyFromArray(CudaNdarray * self, PyArrayObject*obj);

/**
 * Transfer the contents of CudaNdarray `other` to `self`.
 *
 * self is reallocated to have the correct dimensions if necessary.
 */
int CudaNdarray_CopyFromCudaNdarray(CudaNdarray * self, CudaNdarray * other, bool unbroadcast = false);

/**
 * Transfer the contents of CudaNdarray `self` to a new numpy ndarray.
 */
PyObject * 
CudaNdarray_CreateArrayObj(CudaNdarray * self);

/**
 * True iff the strides look like [dim[nd-2], dim[nd-3], ... , dim[0], 1]
 */
bool CudaNdarray_is_c_contiguous(const CudaNdarray * self);

int CudaNdarray_gemm(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C);


int CudaNdarray_reduce_sum(CudaNdarray * self, CudaNdarray * A);
int CudaNdarray_reduce_prod(CudaNdarray * self, CudaNdarray * A);
int CudaNdarray_reduce_min(CudaNdarray * self, CudaNdarray * A);
int CudaNdarray_reduce_max(CudaNdarray * self, CudaNdarray * A);

int CudaNdarray_dimshuffle(CudaNdarray * self, unsigned int len, const int * pattern);

void fprint_CudaNdarray(FILE * fd, const CudaNdarray *self)
{
    fprintf(fd, "CudaNdarray <%p, %p> nd=%i dev_structure_fresh=%d data_allocated=%d\n",
	    self, self->devdata, self->nd, self->dev_structure_fresh, self->data_allocated);
    fprintf(fd, "\tHOST_DIMS:      ");
    for (int i = 0; i < self->nd; ++i)
    {
        fprintf(fd, "%i\t", CudaNdarray_HOST_DIMS(self)[i]);
    }
    fprintf(fd, "\n\tHOST_STRIDES: ");
    for (int i = 0; i < self->nd; ++i)
    {
        fprintf(fd, "%i\t", CudaNdarray_HOST_STRIDES(self)[i]);
    }
    
    int data=0;
    fprintf(fd, "\n\tDEV_DIMS:      ");
    for (int i = 0; i < self->nd; ++i)
    {
        cublasGetVector(1, sizeof(int),
			self->dev_structure+i, 1,
			&data, 1);
	fprintf(fd, "%i\t", data);
    }
    fprintf(fd, "\n\tDEV_STRIDES: ");
    for (int i = 0; i < self->nd; ++i)
    {
        cublasGetVector(1, sizeof(int),
			self->dev_structure + self->nd+i, 1,
			&data, 1);
	fprintf(fd, "%i \t", data);
    }
    fprintf(fd, "\n");
}

#endif
