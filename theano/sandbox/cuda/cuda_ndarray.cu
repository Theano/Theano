#define _CUDA_NDARRAY_C

#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>
#include <iostream>

#include "cuda_ndarray.cuh"

//If true, when there is a gpu malloc or free error, we print the size of allocated memory on the device.
#define COMPUTE_GPU_MEM_USED 0

//If true, we fill with NAN allocated device memory.
#define ALLOC_MEMSET 0

/////////////////////////
// Alloc and Free
/////////////////////////

static int g_gpu_context_active = 0;


PyObject *
CudaNdarray_Dimshuffle(PyObject* _unused, PyObject* args);

/**
 *
 * In the test program I'm using, the _outstanding_mallocs decreases with every call.
 * This suggests there are more free() calls being made than alloc(), but I can't figure out why.
 *
 */
int _outstanding_mallocs[] = {0,0};
#if COMPUTE_GPU_MEM_USED
int _allocated_size = 0;
const int TABLE_SIZE = 10000;
struct table_struct{
    void* ptr;
    int size;
};
table_struct _alloc_size_table[TABLE_SIZE];
#endif
void * device_malloc(size_t size)
{
    void * rval=NULL;
    cudaError_t err = cudaMalloc(&rval, size);
    if (cudaSuccess != err)
    {
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error allocating %li bytes of device memory (%s). %d already allocated\n", (long)size, cudaGetErrorString(err),_allocated_size);
#else
        fprintf(stderr, "Error allocating %li bytes of device memory (%s).\n", (long)size, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "Error allocating %li bytes of device memory (%s).", (long)size, cudaGetErrorString(err));
        return NULL;
    }
    _outstanding_mallocs[0] += (rval != NULL);
#if COMPUTE_GPU_MEM_USED
    for(int i=0;i<TABLE_SIZE;i++){
        if(NULL==_alloc_size_table[i].ptr){
            _alloc_size_table[i].ptr=rval;
            _alloc_size_table[i].size=size;
            break;
        }
    }
    _allocated_size += size;
#endif
    //fprintf(stderr, "allocated %li bytes of device memory (%s). %d already allocated, ptr: %p\n", (long)size, cudaGetErrorString(err),_allocated_size,rval);

    if(ALLOC_MEMSET){
        //We init them to nan to make sure we catch more debug case.
        cudaMemset(rval, 0xFF, size);
        //printf("MEMSET\n");
    }
    return rval;
}
int device_free(void *ptr)
{
    // if there is no gpu context, the call to cudaFree will fail; skip it entirely
    if(!g_gpu_context_active) {
        return 0;
    }
    cudaError_t err =  cudaFree(ptr);
    if (cudaSuccess != err)
    {
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error freeing device pointer %p (%s).%d byte already allocated\n", ptr, cudaGetErrorString(err), _allocated_size);
#else
        fprintf(stderr, "Error freeing device pointer %p (%s).\n", ptr, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "error freeing device pointer %p (%s)", ptr, cudaGetErrorString(err));
        return -1;
    }
    _outstanding_mallocs[0] -= (ptr != NULL);
#if COMPUTE_GPU_MEM_USED
    int i=0;
    size_t total_freed = 0;
    for(;i<TABLE_SIZE;i++)
        if(_alloc_size_table[i].ptr==ptr){
            _allocated_size -= _alloc_size_table[i].size;
            total_freed += _alloc_size_table[i].size;
            _alloc_size_table[i].ptr=0;
            _alloc_size_table[i].size=0;

            break;
        }
    if(i==TABLE_SIZE)
        printf("Unallocated unknow size!\n");
    //fprintf(stderr, "freed %li bytes of device memory (%s). %d already allocated, ptr=%p\n", (long)total_freed, cudaGetErrorString(err),_allocated_size,ptr);
#endif
    return 0;
}
static PyObject *
outstanding_mallocs(PyObject* self, PyObject * args)
{
    return PyInt_FromLong(_outstanding_mallocs[0]);
}

/////////////////////////
// Static helper methods
/////////////////////////

static void
CudaNdarray_null_init(CudaNdarray*self)
{
    self->base = NULL;
    self->nd = -1;
    self->host_structure = NULL;
    self->data_allocated = 0;
    self->dev_structure_fresh = 1;
    self->dev_structure = NULL;
    self->devdata = NULL;
}

static int
CudaNdarray_uninit(CudaNdarray*self)
{
    int rval = 0;
    if (self->data_allocated) {
        assert(self->devdata);
        if (device_free(self->devdata))
        {
            fprintf(stderr,
                    "!!!! error freeing device memory %p (self=%p)\n",
                    self->devdata, self);
            rval = -1;
        }
        self->devdata = NULL;
        self->data_allocated = 0;
    }
    if (self->dev_structure)
    {
        if (device_free(self->dev_structure))
        {
            fprintf(stderr,
                    "!!!! error freeing dev_structure memory %p (self=%p)\n",
                    self->dev_structure, self);
            rval = -1;
        }
        self->dev_structure = NULL;
    }
    if (self->host_structure)
    {
        free(self->host_structure);
        self->host_structure = NULL;
    }
    self->nd = -1;
    Py_XDECREF(self->base);
    self->base = NULL;
    return rval;
}


//make the rightmost coords change fastest
//TODO: why does a downward for-loop not work????
//TODO: use the log2_dims and driver code to remove / and %
//TODO: skip the last division (when d == 0)
#define decl_k_elemwise_unary_rowmajor(name, F) \
__global__ void name (unsigned int numEls,  \
        unsigned int nd, \
        const int * dim,  \
        const float * a_data, const int * a_str, \
        float * z_data, const int * z_str) \
{ \
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int numThreads = blockDim.x * gridDim.x; \
 \
    for (unsigned int i = idx; i < numEls; i += numThreads) \
    { \
        unsigned int ii = i; \
        const float * a_i = a_data; \
        float * z_i = z_data; \
        for (unsigned int _d = 0; _d < nd; ++_d) \
        { \
            unsigned int d = nd - _d-1;  \
            /* i_d used to be unsigned, but their is a bug in nvcc 3.0. making it signed fix the bug.*/\
            int i_d = ii % dim[d]; /* i_d is our position in the d'th dimension   */ \
            ii = ii / dim[d]; \
            a_i += i_d * a_str[d]; /* increment our a and z pointers by i_d elements */ \
            z_i += i_d * z_str[d]; \
        } \
        z_i[0] = F(a_i[0]); \
    } \
}

template<typename T> __device__ T unary_copy(T a) { return a; }
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy, unary_copy<float>)

template<typename T> __device__ T unary_exp(T a) { return exp(a); }
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_exp, unary_exp<float>)

/////////////////////////////
// Satisfying reqs to be Type
/////////////////////////////

//DON'T use directly(if their is other CudaNdarray that point to it, it will cause problem)! use Py_DECREF() instead
static void
CudaNdarray_dealloc(CudaNdarray* self)
{
    if (0) std::cerr << "CudaNdarray dealloc " << self << " " << self->devdata << '\n';
    if(self->ob_refcnt>1)
      printf("WARNING:CudaNdarray_dealloc called when their is still active reference to it.\n");
    CudaNdarray_uninit(self);
    self->ob_type->tp_free((PyObject*)self);
    --_outstanding_mallocs[1];
    if (0)
    {
        fprintf(stderr, "device_malloc_counts: (device) %i (obj) %i\n",
                _outstanding_mallocs[0],
                _outstanding_mallocs[1]);
    }
}

static PyObject *
CudaNdarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CudaNdarray *self;

    self = (CudaNdarray *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        CudaNdarray_null_init(self);
        ++_outstanding_mallocs[1];
    }
    return (PyObject *)self;
}
static int
CudaNdarray_init(CudaNdarray *self, PyObject *args, PyObject *kwds)
{
    PyObject *arr=NULL;

    if (! PyArg_ParseTuple(args, "O", &arr))
        return -1;
    if (! PyArray_Check(arr))
    {
        PyErr_SetString(PyExc_TypeError, "PyArray arg required");
        return -1;
    }
    int rval = CudaNdarray_CopyFromArray(self, (PyArrayObject*)arr);
    return rval;
}
static PyMemberDef CudaNdarray_members[] =
{
    /*
    {"first", T_OBJECT_EX, offsetof(CudaNdarray, first), 0,
     "first name"},
    {"last", T_OBJECT_EX, offsetof(CudaNdarray, last), 0,
     "last name"},
    {"number", T_INT, offsetof(CudaNdarray, number), 0,
     "noddy number"},
     */
    {NULL}  /* Sentinel */
};

PyObject * CudaNdarray_CreateArrayObj(CudaNdarray * self)
{
    int verbose = 0;
    if(self->nd>=0 && CudaNdarray_SIZE(self)==0){
        npy_intp * npydims = (npy_intp*)malloc(self->nd * sizeof(npy_intp));
        assert (npydims);
        for (int i = 0; i < self->nd; ++i) npydims[i] = (npy_intp)(CudaNdarray_HOST_DIMS(self)[i]);
        PyObject * rval = PyArray_SimpleNew(self->nd, npydims, REAL_TYPENUM);
        free(npydims);
        if (!rval){
            return NULL;
        }
        assert (PyArray_ITEMSIZE(rval) == sizeof(real));
        return rval;
    }
    if ((self->nd < 0) || (self->devdata == 0))
    {
        PyErr_SetString(PyExc_ValueError, "can't copy from un-initialized CudaNdarray");
        return NULL;
    }
    CudaNdarray * contiguous_self = NULL;
    if (CudaNdarray_is_c_contiguous(self))
    {
        contiguous_self = self;
        Py_INCREF(contiguous_self);
        if (verbose) std::cerr << "CreateArrayObj already contiguous" << contiguous_self << '\n';
    }
    else
    {
        contiguous_self = (CudaNdarray*)CudaNdarray_Copy(self);
        if (verbose) std::cerr << "CreateArrayObj created contiguous" << contiguous_self << '\n';
    }
    if (!contiguous_self)
    {
        return NULL;
    }

    npy_intp * npydims = (npy_intp*)malloc(self->nd * sizeof(npy_intp));
    assert (npydims);
    for (int i = 0; i < self->nd; ++i)
        npydims[i] = (npy_intp)(CudaNdarray_HOST_DIMS(self)[i]);
    PyObject * rval = PyArray_SimpleNew(self->nd, npydims, REAL_TYPENUM);
    free(npydims);
    if (!rval)
    {
        Py_DECREF(contiguous_self);
        return NULL;
    }

    assert (PyArray_ITEMSIZE(rval) == sizeof(real));

    cublasGetVector(PyArray_SIZE(rval), sizeof(real),
                    contiguous_self->devdata, 1,
                    PyArray_DATA(rval), 1);
    CNDA_THREAD_SYNC;

    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to host");
        Py_DECREF(rval);
        rval = NULL;
    }

    Py_DECREF(contiguous_self);
    return rval;
}

// TODO-- we have two functions here, ZEROS and Zeros.
// ZEROS is meant to be called just from C code (you don't need to pass it PyObject * s)
// but this naming is very weird, makes it look like a macro
// we should figure out the correct convention and change to that
PyObject* CudaNdarray_ZEROS(int n, int * dims)
{

    int total_elements = 1;
    for(int i=0;i<n;i++)
        total_elements*=dims[i];

    // total_elements now contains the size of the array, in reals
    int total_size = total_elements * sizeof(real);

    CudaNdarray* rval = (CudaNdarray*)CudaNdarray_New();
    if (!rval)
    {
        PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_ZEROS: call to New failed");
        return NULL;
    }

    if (CudaNdarray_alloc_contiguous(rval, n, dims))
    {
        PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_ZEROS: allocation failed.");
        Py_DECREF(rval);
        return NULL;
    }

    // Fill with zeros
    //fprintf(stdout, "Sizeof: %d\n", total_size);
    if (cudaSuccess != cudaMemset(rval->devdata, 0, total_size))
    {
        PyErr_Format(PyExc_MemoryError, "CudaNdarray_ZEROS: Error memsetting %d bytes of device memory.", total_size);
        Py_DECREF(rval);
        return NULL;
    }

    if (cnda_copy_structure_to_device(rval))
    {
        PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_ZEROS: syncing structure to device failed");
        Py_DECREF(rval);
        return NULL;
    }
    return (PyObject*) rval;
}

// declared as a static method (hence "dummy" is not used)
// Based on _Copy and _dimshuffle
PyObject* CudaNdarray_Zeros(PyObject* dummy, PyObject* shape)
{
    if(!shape)
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_Zeros: function takes at least 1 argument (0 given)");
        return NULL;
    }
    if(!PySequence_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    int shplen = PySequence_Length(shape);

    if (shplen == 0)
    {
        return CudaNdarray_ZEROS(0, NULL);
    }

    int* newdims = (int *)malloc(sizeof(int) * shplen);

    if (!newdims)
    {
        PyErr_SetString(PyExc_MemoryError,
            "CudaNdarray_Zeros: Failed to allocate temporary space");
        return NULL;
    }

    // start from the end to compute strides
    for (int i = shplen-1; i >= 0; --i)
    {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if(shp_el_obj == NULL)
        {
            // shouldn't happen since we checked length before...
            PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_Zeros: Index out of bound in sequence");
            free(newdims);
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        if (shp_el < 0)
        {
            PyErr_SetString(PyExc_ValueError, "CudaNdarray_Zeros: shape must contain only non-negative values for size of a dimension");
            free(newdims);
            return NULL;
        }

        newdims[i] = shp_el;
    }

    PyObject* rval = CudaNdarray_ZEROS(shplen,newdims);

    free(newdims);

    return (PyObject*)rval;
}





PyObject * CudaNdarray_Copy(CudaNdarray * self)
{
    PyObject * rval = CudaNdarray_New();
    if ((!rval) || (-1 == self->nd))
    {
        return rval;
    }
    if (CudaNdarray_alloc_contiguous((CudaNdarray*)rval, self->nd, CudaNdarray_HOST_DIMS(self)))
    {
        Py_DECREF(rval);
        return NULL;
    }
    if (CudaNdarray_CopyFromCudaNdarray((CudaNdarray*)rval, self))
    {
        Py_DECREF(rval);
        return NULL;
    }
    return rval;
}
PyObject * CudaNdarray_DeepCopy(CudaNdarray * self, PyObject * memo)
{
    assert(PyDict_Check(memo));
    PyObject * selfkey = PyInt_FromLong((long)self);
    assert(selfkey);
    if (PyDict_Contains(memo, selfkey))
    {
        PyObject * rval = PyDict_GetItem(memo, selfkey);
        Py_DECREF(selfkey);
        Py_XINCREF(rval);
        return rval;
    }
    else
    {
        PyObject * rval = CudaNdarray_Copy(self);
        if (0) std::cerr << "DeepCopy created " << rval << " devdata " << ((CudaNdarray*)rval)->devdata << "\n";
        if (NULL == rval)
        {
            Py_DECREF(selfkey);
            return NULL;
        }
        if (PyDict_SetItem(memo, selfkey, rval))
        {
            Py_DECREF(rval);
            Py_DECREF(selfkey);
            return NULL;
        }
        Py_DECREF(selfkey);
        return rval;
    }
}
PyObject * CudaNdarray_ReduceSum(CudaNdarray * self, PyObject * py_reduce_mask)
{
    if (!PySequence_Check(py_reduce_mask))
    {
        PyErr_SetString(PyExc_TypeError, "reduce_mask must be sequence of ints");
        return NULL;
    }
    int len = PySequence_Length(py_reduce_mask);
    if (len != self->nd)
    {
        PyErr_SetString(PyExc_TypeError, "length of reduce_mask must match self->nd");
        return NULL;
    }
    CudaNdarray * self_sum = (CudaNdarray*)CudaNdarray_New();
    if (!self_sum)
    {
        return NULL;
    }
    //TODO: allocate a fixed size dimshuffle_pattern_cache on the stack,
    //      and use it if it is big enough.
    int * dimshuffle_pattern = (int*)malloc(len * 2 * sizeof(int));
    int * sum_dims = dimshuffle_pattern + len;
    int n_remaining_dims = 0;
    if (!dimshuffle_pattern)
    {
        Py_DECREF(self_sum);
        PyErr_SetString(PyExc_MemoryError, "failed to alloc internal storage");
        return NULL;
    }
    for (int i = 0; i < len; ++i)
    {
        PyObject *o_i = PySequence_GetItem(py_reduce_mask, i);
        int o_i_int = PyInt_AsLong(o_i);
        Py_XDECREF(o_i);
        if (PyErr_Occurred())
        {
            Py_DECREF(self_sum);
            free(dimshuffle_pattern);
            return NULL;
        }
        if (o_i_int) // this is a dimension over which we are reducing
        {
            sum_dims[i] = 1;
        }
        else
        {
            sum_dims[i] = CudaNdarray_HOST_DIMS(self)[i];
            dimshuffle_pattern[n_remaining_dims++] = i;
        }
    }
    if (0   || CudaNdarray_alloc_contiguous(self_sum, len, sum_dims)
            || CudaNdarray_reduce_sum(self_sum, self)
            || CudaNdarray_dimshuffle(self_sum, n_remaining_dims, dimshuffle_pattern))
    {
        Py_DECREF(self_sum);
        free(dimshuffle_pattern);
        return NULL;
    }
    free(dimshuffle_pattern);
    return (PyObject*)self_sum;
}

__global__ void k_copy_reshape_rowmajor(unsigned int numEls,
        unsigned int a_nd, const float * a_data, const int * a_dim, const int * a_str,
        unsigned int z_nd, float * z_data, const int * z_dim, const int * z_str)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads)
    {
        const float * a_i = a_data;
        unsigned int a_ii = i;
        for (unsigned int _d = 0; _d < a_nd; ++_d) //make the rightmost coords change fastest
        {
            unsigned int d = a_nd - _d-1;
            unsigned int a_i_d = a_ii % a_dim[d];
            a_ii = a_ii / a_dim[d];
            a_i += a_i_d * a_str[d];
        }
        unsigned int z_ii = i;
        float * z_i = z_data;
        for (unsigned int _d = 0; _d < z_nd; ++_d) //make the rightmost coords change fastest
        {
            unsigned int d = z_nd - _d-1;
            //i tried to make the for loop count down, but it didn't work!?
            unsigned int z_i_d = z_ii % z_dim[d];
            z_i += z_i_d * z_str[d];
            z_ii = z_ii / z_dim[d];
        }
        z_i[0] = a_i[0]; //copy one lousy float!
    }
}

// Reshape self to the new shape gived by the tuple shape.
//
// If self is c contiguous, it return a view. Otherwise it always do a copy.
// TODO: make it return a view when the strides allow it even if it is not
//       c contiguous
PyObject * CudaNdarray_Reshape(CudaNdarray * self, PyObject * shape)
{
    // check shape tuple
    unsigned int rval_nd;
    unsigned int * rval_dims;
    unsigned int rval_size = 1;

    if (PyTuple_Check(shape)){
        // copy shape to integer array
        rval_nd = PyTuple_Size(shape);
    }else if (PyInt_Check(shape)){
        rval_nd = 1;
    }else{
        PyErr_SetString(PyExc_TypeError, "shape must be tuple of integers or an integer");
        return NULL;
    }
    rval_dims = (unsigned int*)malloc(rval_nd * sizeof(int));

    if(PyTuple_Check(shape)){
        for (int i = 0; i < rval_nd; ++i)
        {
            rval_dims[i] = PyInt_AsLong(PyTuple_GetItem(shape, i)); //GetItem returns borrowed reference
            if (PyErr_Occurred()) //error in AsLong
            {
                free(rval_dims);
                return NULL;
            }
            if(rval_dims[i]<=0){
                PyErr_Format(PyExc_ValueError, "Reshape has invalid dimension %i (must be >0)",rval_dims[i]);
                free(rval_dims);
                return NULL;
            }
            rval_size = rval_size * rval_dims[i];
        }
    }else{
        rval_size = PyInt_AsLong(shape);
        rval_dims[0] = rval_size;
    }
    // calculate new size, assert same as old size
    if (rval_size != CudaNdarray_SIZE(self))
    {
        PyErr_Format(PyExc_ValueError, "size must remain unchanged, changed from %i to %i", CudaNdarray_SIZE(self), rval_size);
        free(rval_dims);
        return NULL;
    }
    if (rval_size==0)
    {
        PyObject * rval = CudaNdarray_NewDims(rval_nd, rval_dims);
        free(rval_dims);
        return rval;
    }

    if(CudaNdarray_is_c_contiguous(self))
    {
        //return a view, not a copy
        CudaNdarray * rval = (CudaNdarray * )CudaNdarray_New(rval_nd);

        if (!rval || 0 != rval->data_allocated
            ||CudaNdarray_set_device_data(rval, CudaNdarray_DEV_DATA(self), self))
        {
            Py_XDECREF(rval);
            free(rval_dims);
            return NULL;
        }
        //set dim and stride
        int size = 1;
        for (int i = rval_nd-1; i >= 0; --i)
        {
            CudaNdarray_set_stride(rval, i, (rval_dims[i] == 1) ? 0 : size);
            CudaNdarray_set_dim(rval, i, rval_dims[i]);
            size = size * rval_dims[i];
        }
        free(rval_dims);
        return (PyObject*)rval;
    }

    // allocate new space (TODO: test to see if we can re-use old one)
    CudaNdarray * rval = (CudaNdarray * )CudaNdarray_New();
    if (!rval || CudaNdarray_alloc_contiguous(rval, rval_nd, rval_dims)){
        Py_XDECREF(rval);
        free(rval_dims);
        return NULL;
    }

    // call worker routine
    unsigned int threads_per_block = std::min(rval_size, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
    unsigned int n_blocks = std::min(ceil_intdiv(rval_size,threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);
    k_copy_reshape_rowmajor<<<n_blocks,threads_per_block>>>(
            rval_size,
            self->nd,
            CudaNdarray_DEV_DATA(self), CudaNdarray_DEV_DIMS(self), CudaNdarray_DEV_STRIDES(self),
            rval->nd,
            CudaNdarray_DEV_DATA(rval), CudaNdarray_DEV_DIMS(rval), CudaNdarray_DEV_STRIDES(rval));

    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        Py_DECREF(rval);
        PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n", "k_copy_reshape_rowmajor", cudaGetErrorString(err));
        free(rval_dims);
        return NULL;
    }
    free(rval_dims);
    return (PyObject*)rval;
}
PyObject * CudaNdarray_View(CudaNdarray * self)
{
    CudaNdarray * rval = (CudaNdarray*)CudaNdarray_New(self->nd);
    if (!rval || CudaNdarray_set_device_data(rval, CudaNdarray_DEV_DATA(self), self))
    {
        Py_XDECREF(rval);
        rval = NULL;
    }
    else
    {
        for (int i = 0; i < self->nd; ++i)
        {
            CudaNdarray_set_dim(rval, i, CudaNdarray_HOST_DIMS(self)[i]);
            CudaNdarray_set_stride(rval, i, CudaNdarray_HOST_STRIDES(self)[i]);
        }
    }
    return (PyObject*)rval;
}

PyObject * CudaNdarray_SetStride(CudaNdarray * self, PyObject *args)
{
    int pos, stride;
    if (! PyArg_ParseTuple(args, "ii", &pos, &stride))
        return NULL;
    if ((pos < 0) || (pos >= self->nd))
    {
        PyErr_Format(PyExc_ValueError, "position argument out of legal range [0, %i)", self->nd);
        return NULL;
    }
    CudaNdarray_set_stride(self, pos, stride);
    if (cnda_copy_structure_to_device(self))
    {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}
PyObject * CudaNdarray_SetShapeI(CudaNdarray * self, PyObject *args)
{
    int pos, dim;
    if (! PyArg_ParseTuple(args, "ii", &pos, &dim))
        return NULL;
    if ((pos < 0) || (pos >= self->nd))
    {
        PyErr_Format(PyExc_ValueError, "position argument out of legal range [0, %i)", self->nd);
        return NULL;
    }
    CudaNdarray_set_dim(self, pos, dim);
    if (cnda_copy_structure_to_device(self))
    {
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
CudaNdarray_exp(CudaNdarray* self)
{
    CudaNdarray * rval = (CudaNdarray *)CudaNdarray_New();
    if ((NULL == rval) || CudaNdarray_alloc_contiguous(rval, self->nd, CudaNdarray_HOST_DIMS(self)))
    {
        Py_XDECREF(rval);
        return NULL;
    }
    unsigned int size = 1;
    for (int i = 0; i < self->nd; i++)
    {
        size *= (unsigned int) CudaNdarray_HOST_DIMS(self)[i];
    }
    unsigned int threads_per_block = std::min(size, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
    unsigned int n_blocks = std::min(ceil_intdiv(size,threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);
    k_elemwise_unary_rowmajor_exp<<<n_blocks,threads_per_block>>>(size, self->nd, CudaNdarray_DEV_DIMS(self),
            CudaNdarray_DEV_DATA(self), CudaNdarray_DEV_STRIDES(self),
            CudaNdarray_DEV_DATA(rval), CudaNdarray_DEV_STRIDES(rval));

    //TODO: don't do this right away, do it when we need the result
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        Py_DECREF(rval);
        PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n", "kExp", cudaGetErrorString(err));
        return NULL;
    }

    return (PyObject*)rval;
}
static PyMethodDef CudaNdarray_methods[] =
{
    {"__array__",
        (PyCFunction)CudaNdarray_CreateArrayObj, METH_NOARGS,
        "Copy from the device to a numpy ndarray"},
    {"__copy__",
        (PyCFunction)CudaNdarray_View, METH_NOARGS,
        "Create a shallow copy of this object. used by module copy"},
    {"__deepcopy__",
        (PyCFunction)CudaNdarray_DeepCopy, METH_O,
        "Create a copy of this object"},
    {"zeros",
        (PyCFunction)CudaNdarray_Zeros, METH_STATIC,
        "Create a new CudaNdarray with specified shape, filled with zeros."},
    {"copy",
        (PyCFunction)CudaNdarray_Copy, METH_NOARGS,
        "Create a copy of this object"},
    {"is_c_contiguous",
        (PyCFunction)CudaNdarray_IS_C_Contiguous, METH_NOARGS,
        "Return True is the object is c contiguous. False otherwise."},
    {"reduce_sum",
        (PyCFunction)CudaNdarray_ReduceSum, METH_O,
        "Reduce over the given dimensions by summation"},
    {"exp",
        (PyCFunction)CudaNdarray_exp, METH_NOARGS,
        "Return the exponential of all elements"},
    {"reshape",
        (PyCFunction)CudaNdarray_Reshape, METH_O,
        "Return a reshaped view (or copy) of this ndarray\n\
            The required argument is a tuple of integers specifying the shape of the new ndarray."},
    {"view",
        (PyCFunction)CudaNdarray_View, METH_NOARGS,
        "Return an alias of this ndarray"},
    {"_set_stride",
        (PyCFunction)CudaNdarray_SetStride, METH_VARARGS,
        "For integer arguments (i, s), set the 'i'th stride to 's'"},
    {"_set_shape_i",
        (PyCFunction)CudaNdarray_SetShapeI, METH_VARARGS,
        "For integer arguments (i, s), set the 'i'th shape to 's'"},
    {NULL, NULL, NULL, NULL}  /* Sentinel */
};


////////////////////
// Number protocol
////////////////////

__global__ void kAdd_contiguous(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + b[i];
    }
}

// Will be called by __add__ in Python
static PyObject *
CudaNdarray_add(PyObject* py_self, PyObject * py_other)
{
    if (! CudaNdarray_Check(py_self)) {
        PyErr_SetString(PyExc_TypeError, "need a CudaNdarray on left");
        return NULL;
    }
    if (! CudaNdarray_Check(py_other)) {
        PyErr_SetString(PyExc_TypeError, "need a CudaNdarray on right");
        return NULL;
    }
    CudaNdarray * self = (CudaNdarray *)py_self;
    CudaNdarray * other = (CudaNdarray *)py_other;
    if(!CudaNdarray_is_c_contiguous(self) || !CudaNdarray_is_c_contiguous(other)){
        PyErr_SetString(PyExc_TypeError, "We have implementet only the c_contiguous version for now.");
        return NULL;
    }

    //standard elemwise size checks
    if (self->nd != other->nd)
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_add: need same number of dims");
        return NULL;
    }
    //standard elemwise dim checks
    unsigned int size = 1;
    for (int i = 0; i< self->nd; ++i)
    {
        if (CudaNdarray_HOST_DIMS(self)[i] != CudaNdarray_HOST_DIMS(other)[i])
        {
            PyErr_SetString(PyExc_TypeError, "need same dimensions");
            return NULL;
        }
        size *= (unsigned int) CudaNdarray_HOST_DIMS(self)[i];
    }
    CudaNdarray * rval = (CudaNdarray *)CudaNdarray_New();
    if (!rval || CudaNdarray_alloc_contiguous(rval, self->nd, CudaNdarray_HOST_DIMS(self)))
    {
        Py_XDECREF(rval);
        return NULL;
    }

    if(CudaNdarray_SIZE((CudaNdarray *)py_self)==0 && CudaNdarray_SIZE((CudaNdarray *)py_other)==0){
      return (PyObject *) rval;
    }

    int threads_per_block = std::min(size, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
    int n_blocks = std::min(ceil_intdiv(size,(unsigned int)threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);
    kAdd_contiguous<<<n_blocks,threads_per_block>>>(
            self->devdata, other->devdata, rval->devdata, size);
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n", "kAdd", cudaGetErrorString(err));
        Py_DECREF(rval);
        return NULL;
    }
    return (PyObject *) rval;
}

enum operator_t
{
    IADD=0,
    IDIV,
    CPY,
    N_ELEMWISE_OPS // What this mean? It is not used
};

template <int operator_num>
__global__ void k_ielem_3(const int d0, const int d1, const int d2,
        float* a, const int sA0, const int sA1, const int sA2,
        const float* b, const int sB0, const int sB1, const int sB2){
    for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
        for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y){
            for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x){
                switch (operator_num)
                {
                  case IADD:
                    a[i0*sA0 + i1*sA1 + i2*sA2] += b[i0*sB0 + i1*sB1 + i2*sB2];
                    break;
                  case IDIV:
                    a[i0*sA0 + i1*sA1 + i2*sA2] /= b[i0*sB0 + i1*sB1 + i2*sB2];
                    break;
                  case CPY:
                    a[i0*sA0 + i1*sA1 + i2*sA2] = b[i0*sB0 + i1*sB1 + i2*sB2];
                    break;
                }
            }
        }
    }
}

template <int operator_num>
__global__ void k_ielem_4(const int d0, const int d1, const int d2, const int d3,
                         float* a, const int sA0, const int sA1,
                         const int sA2, const int sA3,
                         const float* b, const int sB0, const int sB1,
                         const int sB2, const int sB3){
    for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
        for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y){
            for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x){
                for (int i3 = threadIdx.y; i3 < d3; i3 += blockDim.y){
                    switch (operator_num) {
                        case IADD:
                            a[i0*sA0 + i1*sA1 + i2*sA2 + i3*sA3]
                            += b[i0*sB0 + i1*sB1 + i2*sB2 + i3*sB3];
                            break;
                        case IDIV:
                            a[i0*sA0 + i1*sA1 + i2*sA2 + i3*sA3]
                            /= b[i0*sB0 + i1*sB1 + i2*sB2 + i3*sB3];
                            break;
                        case CPY:
                            a[i0*sA0 + i1*sA1 + i2*sA2 + i3*sA3]
                            = b[i0*sB0 + i1*sB1 + i2*sB2 + i3*sB3];
                            break;
                    }
                }
            }
        }
    }
}

/*
CudaNdarray_inplace_elemwise
Compute elemwise, working inplace on A.
Currently implemented A / B, A + B and A = B
(the last is not tested and not used!)

py_self - the CudaNdarray that we'll modify (A)
py_other - the other argument (B)
fct_nb - which operation to perform (operator_t)

Returns 0 on success.
Returns -1 on failure, and sets Python exception.

*/
int
CudaNdarray_inplace_elemwise(PyObject* py_self, PyObject * py_other, operator_t fct_nb)
{
    int verbose = 0;
    void (*k3)(const int, const int, const int,
                    float*, const int, const int, const int,
                    const float*, const int, const int, const int);
    void (*k4)(const int, const int, const int, const int,
                    float*, const int, const int,
                    const int, const int,
                    const float*, const int, const int,
                    const int, const int);
    switch (fct_nb)
    {
        case IADD:
            k3 = k_ielem_3<IADD>;
            k4 = k_ielem_4<IADD>;
            break;
        case IDIV:
            k3 = k_ielem_3<IDIV>;
            k4 = k_ielem_4<IDIV>;
            break;
        case CPY:
            k3 = k_ielem_3<CPY>;
            k4 = k_ielem_4<CPY>;
            break;
        default:
            assert (0);
            PyErr_Format(
                PyExc_TypeError,
                "CudaNdarray_inplace_elemwise invalid fct_nb (%i).",
                (int)fct_nb);
            return -1;
    }
    if (!CudaNdarray_Check(py_self)) {
        PyErr_SetString(
            PyExc_TypeError,
            "CudaNdarray_inplace_elemwise need a CudaNdarray on left");
        return -1;
    }
    if (!CudaNdarray_Check(py_other)) {
        PyErr_SetString(
            PyExc_TypeError,
            "CudaNdarray_inplace_elemwise need a CudaNdarray on right");
        return -1;
    }

    CudaNdarray * self = (CudaNdarray *)py_self;
    CudaNdarray * other = (CudaNdarray *)py_other;

    if (verbose)
    {
        fprintf(stderr,
            "INPLACE ADD/DIV for self->nd=%d other->nd=%d\n",
            self->nd, other->nd);
    }

    //standard elemwise size checks
    if (self->nd != other->nd)
    {
        PyErr_Format(
            PyExc_TypeError,
            "CudaNdarray_inplace_elemwise: need same number of dims. Got %d and %d",
            self->nd, other->nd);
        return -1;
    }
    //standard elemwise dim checks
    unsigned int size = 1;
    for (int i = 0; i< self->nd; ++i)
    {
        if ((CudaNdarray_HOST_DIMS(self)[i] != CudaNdarray_HOST_DIMS(other)[i])
            && (CudaNdarray_HOST_DIMS(other)[i] != 1))
        {
            PyErr_SetString(
                PyExc_ValueError,
                "CudaNdarray_inplace_elemwise need same dimensions (or broadcastable dimension)");
            return -1;
        }
        // if we're broadcasting other, then make sure it has stride 0
        assert ((CudaNdarray_HOST_DIMS(self)[i] == CudaNdarray_HOST_DIMS(other)[i])
            || (CudaNdarray_HOST_STRIDES(other)[i] == 0));
        size *= (unsigned int) CudaNdarray_HOST_DIMS(self)[i];
    }

    if (size==0)
    {
        if (CudaNdarray_SIZE((CudaNdarray *)py_other))
        {
            PyErr_SetString(
                PyExc_ValueError,
                "CudaNdarray_inplace_elemwise cannot work inplace on an un-initialized array");
            return 0;
        }
        return 0;
    }

    switch(self->nd)
    {
        case 0:
            {
                dim3 n_blocks(1, 1, 1);
                dim3 n_threads(1);
                k3<<<n_blocks, n_threads>>>(
                        1, //d0
                        1, //d1
			1, //d2
                        CudaNdarray_DEV_DATA(self),
                        1, //strides
                        1,
                        1,
                        CudaNdarray_DEV_DATA(other),
                        1, //strides
                        1,
                        1);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Cuda error: %s: %s.\n",
                        "k3",
                        cudaGetErrorString(err));
                    return -1;
                }
            }
            break;
        case 1:
            {
                dim3 n_blocks(1, 1, 1);
                dim3 n_threads(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
                k3<<<n_blocks, n_threads>>>(
                        1, //dimensions
                        1,
                        CudaNdarray_HOST_DIMS(self)[0],
                        CudaNdarray_DEV_DATA(self),
                        1, //strides
                        1,
                        CudaNdarray_HOST_STRIDES(self)[0],
                        CudaNdarray_DEV_DATA(other),
                        1, //strides
                        1,
                        CudaNdarray_HOST_STRIDES(other)[0]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Cuda error: %s: %s.\n",
                        "k3",
                        cudaGetErrorString(err));
                    return -1;
                }
            }
            break;
        case 2:
            {
                //TODO:  if both self and other are f-contiguous
                //       Then flip the block and thread dimensions
                //       to make contiguous reads & writes
                dim3 n_blocks(1,
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[0],
                            NUM_VECTOR_OP_BLOCKS));
                dim3 n_threads(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[1],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
                k3<<<n_blocks, n_threads>>>(1,
                        CudaNdarray_HOST_DIMS(self)[0],
                        CudaNdarray_HOST_DIMS(self)[1],
                        CudaNdarray_DEV_DATA(self),
                        1,
                        CudaNdarray_HOST_STRIDES(self)[0],
                        CudaNdarray_HOST_STRIDES(self)[1],
                        CudaNdarray_DEV_DATA(other),
                        1,
                        CudaNdarray_HOST_STRIDES(other)[0],
                        CudaNdarray_HOST_STRIDES(other)[1]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Cuda error: %s: %s.\n",
                        "k3",
                        cudaGetErrorString(err));
                    return -1;
                }
            }
            break;
        case 3:
            {
                //TODO:  Dimshuffle so that at least one of the arrays
                //       has a contiguous dimension on the thread idx.
                dim3 n_blocks(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[0],
                            NUM_VECTOR_OP_BLOCKS),
                        CudaNdarray_HOST_DIMS(self)[1]);
                while (n_blocks.x * n_blocks.y > NUM_VECTOR_OP_BLOCKS)
                    n_blocks.y /= 2;
                dim3 n_threads(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
                k3<<<n_blocks, n_threads>>>(
                        CudaNdarray_HOST_DIMS(self)[0],
                        CudaNdarray_HOST_DIMS(self)[1],
                        CudaNdarray_HOST_DIMS(self)[2],
                        CudaNdarray_DEV_DATA(self),
                        CudaNdarray_HOST_STRIDES(self)[0],
                        CudaNdarray_HOST_STRIDES(self)[1],
                        CudaNdarray_HOST_STRIDES(self)[2],
                        CudaNdarray_DEV_DATA(other),
                        CudaNdarray_HOST_STRIDES(other)[0],
                        CudaNdarray_HOST_STRIDES(other)[1],
                        CudaNdarray_HOST_STRIDES(other)[2]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Cuda error: %s: %s.\n",
                        "k3",
                        cudaGetErrorString(err));
                    return -1;
                }
            }
            break;
        case 4:
            {
                dim3 n_blocks(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[0],
                            NUM_VECTOR_OP_BLOCKS),
                        CudaNdarray_HOST_DIMS(self)[1]
                        );
                while (n_blocks.x * n_blocks.y > NUM_VECTOR_OP_BLOCKS)
                    n_blocks.y /= 2;
                dim3 n_threads(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK)
                    //TODO: DON"T YOU NEED OT PUT DIMS[3] in here???
                            );
                k4<<<n_blocks, n_threads>>>(
                        CudaNdarray_HOST_DIMS(self)[0],
                        CudaNdarray_HOST_DIMS(self)[1],
                        CudaNdarray_HOST_DIMS(self)[2],
                        CudaNdarray_HOST_DIMS(self)[3],
                        CudaNdarray_DEV_DATA(self),
                        CudaNdarray_HOST_STRIDES(self)[0],
                        CudaNdarray_HOST_STRIDES(self)[1],
                        CudaNdarray_HOST_STRIDES(self)[2],
                        CudaNdarray_HOST_STRIDES(self)[3],
                        CudaNdarray_DEV_DATA(other),
                        CudaNdarray_HOST_STRIDES(other)[0],
                        CudaNdarray_HOST_STRIDES(other)[1],
                        CudaNdarray_HOST_STRIDES(other)[2],
                        CudaNdarray_HOST_STRIDES(other)[3]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Cuda error: %s: %s.\n",
                        "k4",
                        cudaGetErrorString(err));
                    return -1;
                }
            }
            break;
        case 5:
            {
                dim3 n_blocks(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[1],
                            NUM_VECTOR_OP_BLOCKS),
                        CudaNdarray_HOST_DIMS(self)[2]);
                while (n_blocks.x * n_blocks.y > NUM_VECTOR_OP_BLOCKS)
                    n_blocks.y /= 2;
                dim3 n_threads(
                        std::min(
                            CudaNdarray_HOST_DIMS(self)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK)
                    //TODO: DON"T YOU NEED OT PUT DIMS[3] in here???
                    );
                for (int i = 0; i < CudaNdarray_HOST_DIMS(self)[0]; ++i)
                {
                     k4<<<n_blocks, n_threads>>>(
                            CudaNdarray_HOST_DIMS(self)[1],
                            CudaNdarray_HOST_DIMS(self)[2],
                            CudaNdarray_HOST_DIMS(self)[3],
                            CudaNdarray_HOST_DIMS(self)[4],
                            CudaNdarray_DEV_DATA(self) + i * CudaNdarray_HOST_STRIDES(self)[0],
                            CudaNdarray_HOST_STRIDES(self)[1],
                            CudaNdarray_HOST_STRIDES(self)[2],
                            CudaNdarray_HOST_STRIDES(self)[3],
                            CudaNdarray_HOST_STRIDES(self)[4],
                            CudaNdarray_DEV_DATA(other) + i * CudaNdarray_HOST_STRIDES(other)[0],
                            CudaNdarray_HOST_STRIDES(other)[1],
                            CudaNdarray_HOST_STRIDES(other)[2],
                            CudaNdarray_HOST_STRIDES(other)[3],
                            CudaNdarray_HOST_STRIDES(other)[4]);
                    CNDA_THREAD_SYNC;
                    cudaError_t err = cudaGetLastError();
                    if( cudaSuccess != err)
                    {
                        PyErr_Format(
                            PyExc_RuntimeError,
                            "Cuda error: %s: %s.\n",
                            "k4",
                            cudaGetErrorString(err));
                        return -1;
                    }
                }
            }
            break;
        default:
        {
            PyErr_Format(
                PyExc_NotImplementedError,
                "inplace_elemwise w nd=%i\n",
                self->nd);
            return -1;
        }
    }
    return 0;
}

/*
 * We need this inplace Add to support IncSubTensor
 * It returns py_self on success with an additional reference. Else NULL.
 */
// Will be called by __iadd__ in Python
static PyObject *
CudaNdarray_inplace_add(PyObject* py_self, PyObject * py_other)
{
    if (CudaNdarray_inplace_elemwise(py_self, py_other, IADD))
    {
        return NULL;
    }
    Py_INCREF(py_self);
    return py_self;
}

/*
 * We need this inplace div for cuda/tests/test_basic_ops.py:test_shared_options
 * It returns py_self on success with an additional reference. Else NULL.
 */
// Will be called by __idiv__ in Python
static PyObject *
CudaNdarray_inplace_div(PyObject* py_self, PyObject * py_other)
{
    if (CudaNdarray_inplace_elemwise(py_self, py_other, IDIV))
    {
        return NULL;
    }
    Py_INCREF(py_self);
    return py_self;
}

static PyNumberMethods CudaNdarrayNumberMethods =
{
     (binaryfunc)CudaNdarray_add,  //binaryfunc nb_add;  __add__
     0,  //binaryfunc nb_subtract;      __sub__
     0,  //binaryfunc nb_multiply;      __mul__
     0,  //binaryfunc nb_divide;        __div__
     0,  //binaryfunc nb_remainder;     __mod__
     0,  //binaryfunc nb_divmod;        __divmod__
     0,  //ternaryfunc nb_power;        __pow__
     0,  //unaryfunc nb_negative;       __neg__
     0,  //unaryfunc nb_positive;       __pos__
     0,  //unaryfunc nb_absolute;       __abs__
     0,  //inquiry nb_nonzero;          __nonzero__     /* Used by PyObject_IsTrue */
     0,  //unaryfunc nb_invert;         __invert__
     0,  //binaryfunc nb_lshift;        __lshift__
     0,  //binaryfunc nb_rshift;        __rshift__
     0,  //binaryfunc nb_and;           __and__
     0,  //binaryfunc nb_xor;           __xor__
     0,  //binaryfunc nb_or;            __or__
     0,  //coercion nb_coerce;          __coerce__     /* Used by the coerce() function */
     0,  //unaryfunc nb_int;            __int__
     0,  //unaryfunc nb_long;           __long__
     0,  //unaryfunc nb_float;          __float__
     0,  //unaryfunc nb_oct;            __oct__
     0,  //unaryfunc nb_hex;            __hex__

     /* Added in release 2.0 */
     (binaryfunc)CudaNdarray_inplace_add,  //binaryfunc nb_inplace_add;  __iadd__
     0,  //binaryfunc nb_inplace_subtract;      __isub__
     0,  //binaryfunc nb_inplace_multiply;      __imul__
     (binaryfunc)CudaNdarray_inplace_div,  //binaryfunc nb_inplace_divide;        __idiv__
     0,  //binaryfunc nb_inplace_remainder;     __imod__
     0,  //ternaryfunc nb_inplace_power;        __ipow__
     0,  //binaryfunc nb_inplace_lshift;        __ilshift__
     0,  //binaryfunc nb_inplace_rshift;        __irshift__
     0,  //binaryfunc nb_inplace_and;           __iand__
     0,  //binaryfunc nb_inplace_xor;           __ixor__
     0,  //binaryfunc nb_inplace_or;            __ior__

     /* Added in release 2.2 */
     0,  //binaryfunc nb_floor_divide;          __floordiv__
     0,  //binaryfunc nb_true_divide;           __truediv__
     0,  //binaryfunc nb_inplace_floor_divide;  __ifloordiv__
     0,  //binaryfunc nb_inplace_true_divide;   __itruediv__

#if PY_MINOR_VERSION > 4
     /* Added in release 2.5 */
     0  //unaryfunc nb_index;  __index__
#endif
};


/////////////////////
// Mapping protocol
/////////////////////

// Will by called by __len__ in Python
static Py_ssize_t
CudaNdarray_len(PyObject * py_self)
{
    CudaNdarray * self = (CudaNdarray*) py_self;
    if (self->nd <= 0)
    {
        return (Py_ssize_t) 0;
    }
    else
    {
        return (Py_ssize_t) CudaNdarray_HOST_DIMS(self)[0];
    }
}

// Will by called by __getitem__ in Python
static PyObject *
CudaNdarray_Subscript(PyObject * py_self, PyObject * key)
{
    int verbose = 0;
    if (verbose) fprintf(stderr, "Subscript .... \n");
    CudaNdarray * self = (CudaNdarray*) py_self;
    PyObject * py_rval = NULL;
    CudaNdarray * rval = NULL;
    PyObject * intobj = NULL;

    //PyObject_Print(key, stderr, 0);

    if (key == Py_Ellipsis)
    {
        Py_INCREF(py_self);
        return py_self;
    }
    if ((intobj=PyNumber_Int(key))) //INDEXING BY INTEGER
    //else if (PyInt_Check(key)) //INDEXING BY INTEGER
    {
        int d_idx = PyInt_AsLong(intobj);
        Py_DECREF(intobj); intobj=NULL;
        //int d_idx = PyInt_AsLong(key);
        if (self->nd == 0)
        {
            PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
            return NULL;
        }
        int d_dim = CudaNdarray_HOST_DIMS(self)[0];
        int offset = 0;

        if ((d_idx >= 0) && (d_idx < d_dim))
        {
            //normal indexing
            offset += d_idx * CudaNdarray_HOST_STRIDES(self)[0];
        }
        else if ((d_idx < 0) && (d_idx >= -d_dim))
        {
            //end-based indexing
            // d_idx is negative
            offset += (d_dim + d_idx) * CudaNdarray_HOST_STRIDES(self)[0];
        }
        else
        {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return NULL;
        }

        //allocate our subtensor view
        py_rval = CudaNdarray_new_nd(self->nd - 1);
        rval = (CudaNdarray*) py_rval;
        if (!rval) return NULL;
        assert (0 == rval->data_allocated);

        //initialize the view's data pointer to our own.
        if (CudaNdarray_set_device_data(rval, CudaNdarray_DEV_DATA(self) + offset, self))
        {
            Py_DECREF(rval);
            return NULL;
        }
        for (int d = 1; d < self->nd; ++d)
        {
            CudaNdarray_set_stride(rval, d-1, CudaNdarray_HOST_STRIDES(self)[d]);
            CudaNdarray_set_dim(rval, d-1, CudaNdarray_HOST_DIMS(self)[d]);
        }
    }
    else
    {
        PyErr_Clear();
    }
    if (PySlice_Check(key)) //INDEXING BY SLICE
    {
        if (self->nd == 0)
        {
            PyErr_SetString(PyExc_ValueError, "cannot slice a 0-d array");
            return NULL;
        }

        int d_dim = CudaNdarray_HOST_DIMS(self)[0];
        Py_ssize_t start, stop, step, slen;
        if (PySlice_GetIndicesEx((PySliceObject*)key, d_dim, &start, &stop, &step, &slen))
        {
            return NULL;
        }
        if (verbose)
        {
            std::cerr << "start " << start << "\n";
            std::cerr << "stop " << stop << "\n";
            std::cerr << "step " << step << "\n";
            std::cerr << "slen " << slen << "\n";
        }

        //allocate our subtensor view
        py_rval = CudaNdarray_new_nd(self->nd);
        rval = (CudaNdarray*) py_rval;
        if (!rval) return NULL;
        assert (0 == rval->data_allocated);


        //initialize the view's data pointer to our own.
        if (CudaNdarray_set_device_data(rval,
                    CudaNdarray_DEV_DATA(self) + start * CudaNdarray_HOST_STRIDES(self)[0],
                    self))
        {
            Py_DECREF(rval);
            return NULL;
        }
        //initialize dimension 0 of rval
        CudaNdarray_set_stride(rval, 0, step * CudaNdarray_HOST_STRIDES(self)[0]);
        CudaNdarray_set_dim(rval, 0, slen);
        if (verbose) std::cerr << "rval stride " << CudaNdarray_HOST_STRIDES(rval)[0] << "\n";
        // initialize dimensions > 0 of rval
        for (int d = 1; d < self->nd; ++d)
        {
            CudaNdarray_set_stride(rval, d, CudaNdarray_HOST_STRIDES(self)[d]);
            CudaNdarray_set_dim(rval, d, CudaNdarray_HOST_DIMS(self)[d]);
        }
    }
    if (PyTuple_Check(key)) //INDEXING BY TUPLE
    {
        //elements of the tuple can be either integers or slices
        //the dimensionality of the view we will return is diminished for each slice in the tuple

        if (PyTuple_Size(key) > self->nd)
        {
            PyErr_SetString(PyExc_IndexError, "index error");
            return NULL;
        }

        //calculate the number of dimensions in the return value
        int rval_nd = self->nd;
        for (int d = 0; d < PyTuple_Size(key); ++d)
        {
            //On some paltform PyInt_Check(<type 'numpy.int64'>) return true, other it return false.
            //So we use PyArray_IsAnyScalar that should covert everything.
            rval_nd -= PyArray_IsAnyScalar(PyTuple_GetItem(key, d));
        }

        //allocate our subtensor view
        py_rval = CudaNdarray_new_nd(rval_nd);
        rval = (CudaNdarray*) py_rval;
        if (!rval) return NULL;
        assert (0 == rval->data_allocated);

        //initialize the view's data pointer to our own.
        if (CudaNdarray_set_device_data(rval, CudaNdarray_DEV_DATA(self), self))
        {
            Py_DECREF(rval);
            return NULL;
        }

        // rval_d will refer to the current dimension in the rval.
        // It will not be incremented for integer keys, but will be incremented for slice
        // keys
        int rval_d = 0;

        for (int d = 0; d < self->nd; ++d)
        {
            // keys can be shorter than self->nd.
            // when that happens, it means that the remaining dimensions are "full slices"
            if (d >=PyTuple_Size(key))
            {
                CudaNdarray_set_stride(rval, rval_d, CudaNdarray_HOST_STRIDES(self)[d]);
                CudaNdarray_set_dim(rval, rval_d, CudaNdarray_HOST_DIMS(self)[d]);
                ++rval_d;
            }
            else
            {
                PyObject * key_d = PyTuple_GetItem(key, d);

                if (PySlice_Check(key_d))
                {
                    Py_ssize_t start, stop, step, slen;
                    if (PySlice_GetIndicesEx((PySliceObject*)key_d, CudaNdarray_HOST_DIMS(self)[d], &start, &stop, &step, &slen))
                    {
                        Py_DECREF(rval);
                        return NULL;
                    }
                    rval->devdata += start * CudaNdarray_HOST_STRIDES(self)[d];
                    CudaNdarray_set_stride(rval, rval_d, step * CudaNdarray_HOST_STRIDES(self)[d]);
                    CudaNdarray_set_dim(rval, rval_d, slen);
                    if (0)
                    {
                        std::cerr << "start " << start << "\n";
                        std::cerr << "stop " << stop << "\n";
                        std::cerr << "step " << step << "\n";
                        std::cerr << "slen " << slen << "\n";
                    }
                    ++rval_d;
                }
                else if ((intobj=PyNumber_Int(key_d)))
                {
                    assert(PyArray_IsAnyScalar(key_d));
                    int d_idx = PyInt_AsLong(intobj);
                    Py_DECREF(intobj);
                    intobj = NULL;
                    int d_dim = CudaNdarray_HOST_DIMS(self)[d];

                    if ((d_idx >= 0) && (d_idx < d_dim))
                    {
                        //normal indexing
                        rval->devdata += d_idx * CudaNdarray_HOST_STRIDES(self)[d];
                    }
                    else if ((d_idx < 0) && (d_idx >= -d_dim))
                    {
                        //end-based indexing
                        rval->devdata += (d_dim + d_idx) * CudaNdarray_HOST_STRIDES(self)[d];
                    }
                    else
                    {
                        PyErr_SetString(PyExc_IndexError, "index out of bounds");
                        Py_DECREF(rval);
                        return NULL;
                    }
                }
                else
                {
                    PyErr_Clear(); // clear the error set by PyNumber_Int
                    PyErr_SetString(PyExc_IndexError, "index must be either int or slice");
                    Py_DECREF(rval);
                    return NULL;
                }
            }
        }
    }
    if (py_rval)
    {
        if (verbose) fprint_CudaNdarray(stderr, self);
        if (verbose) fprint_CudaNdarray(stderr, rval);
    }
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "Unknown key type");
        return NULL;
    }
    return py_rval;
}

// Will by called by __setitem__ in Python
// See http://docs.python.org/dev/py3k/c-api/object.html#PyObject_SetItem
// Doesn't handle broadcasting, e.g. a[:] = 5
// Can only be assigned from a CudaNdarray on the right side
// Or a ndarray when the left side part is c contiguous.
// Or a python scalar with value 0 when the left side part is c contiguous.
static int
CudaNdarray_setitem(PyObject *o, PyObject  *key, PyObject  *value)
{
    int verbose = 0;
    if (verbose) fprintf(stderr, "CudaNdarray_setitem start\n");
    // We try to copy directly into this CudaNdarray from the ndarray
    CudaNdarray* rval = (CudaNdarray*)CudaNdarray_Subscript(o, key);

    if(!rval){
        // CudaNdarray_Subscript failed and set the error msg.
        Py_XDECREF(rval);
        return -1;
    }

    if(rval != (CudaNdarray*)o &&
                (rval->data_allocated ||
                 // The new array should have a base
                 !(((CudaNdarray*)rval)->base) ||
                 // If the original array has no base, the base of the new
                 // array should be the original one
                 (!((CudaNdarray*)o)->base && ((CudaNdarray*)rval)->base != o) ||
                 // Else, the two arrays should have the same base
                 (((CudaNdarray*)o)->base && ((CudaNdarray*)rval)->base != ((CudaNdarray*)o)->base)))
    {
        // This case shouldn't happen, based on what I see in Subscript
        // but just in case it happens sometime in the future

        PyErr_Format(PyExc_RuntimeError, "__getitem__ must return a CudaNdarray that refers to the original CudaNdarray, not a copy. rval.base=%p o.base=%p o=%p",
                     (((CudaNdarray*)rval)->base), ((CudaNdarray*)o)->base, o);
        Py_DECREF(rval);
        return -1;
    }

    PyObject * intobj = NULL;
    if(CudaNdarray_Check(o)  && PyArray_Check(value)){
        if (verbose) fprintf(stderr, "CudaNdarray_setitem dest is a CudaNdarray and value is a ndarray\n");
        int typenum = PyArray_TYPE(value);
        if (typenum != REAL_TYPENUM){
            PyErr_SetString(PyExc_TypeError, "CudaNdarray.__setitem__: can only copy from float32 arrays");
            Py_XDECREF(rval);
            return -1;
        }
        if(! CudaNdarray_is_c_contiguous(rval)){
            PyErr_SetString(PyExc_NotImplementedError, "CudaNdarray.__setitem__: When the new value is an ndarray the part where we copy it to must be c contiguous.");
            Py_XDECREF(rval);
            return -1;
        }
        if(rval->nd != ((PyArrayObject*)value)->nd){
            PyErr_Format(PyExc_NotImplementedError, "CudaNdarray.__setitem__: need same number of dims. destination nd=%d, source nd=%d. broadcasting implemented only for zeroing values from python scalar.",
                         rval->nd,((PyArrayObject*)value)->nd);
            Py_XDECREF(rval);
            return -1;
        }
        for(int i=0 ; i<rval->nd ; i++){
          if(CudaNdarray_HOST_DIMS(rval)[i] != ((PyArrayObject*)value)->dimensions[i]){
            PyErr_Format(PyExc_ValueError, "CudaNdarray.__setitem__: need same dimensions for dim %d, destination=%d, source=%ld",
                i,
                CudaNdarray_HOST_DIMS(rval)[i],
                (long int)(((PyArrayObject*)value)->dimensions[i]));
            Py_XDECREF(rval);
            return -1;
          }
        }
        PyArrayObject * py_v = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)value, typenum,
                                rval->nd, rval->nd);
        cublasSetVector(PyArray_SIZE(py_v),
                        sizeof(real),
                        PyArray_DATA(py_v), 1,
                        rval->devdata, 1);
        CNDA_THREAD_SYNC;
        Py_XDECREF(py_v);
        Py_XDECREF(rval);
        if (CUBLAS_STATUS_SUCCESS != cublasGetError()){
          PyErr_SetString(PyExc_RuntimeError, "CudaNdarray.__setitem__: error copying ndarray data to device memory");
          return -1;
        }
        return 0;
    }
    else if ((intobj=PyNumber_Int(value)))
    {
        if (verbose) fprintf(stderr, "CudaNdarray_setitem dest and value is a python number\n");
        if(! CudaNdarray_is_c_contiguous(rval)){
            PyErr_SetString(PyExc_NotImplementedError,
                            "CudaNdarray.__setitem__: When the new value is a scalar of value 0 the part where we copy to must be c contiguous.");
            Py_XDECREF(rval);
            return -1;
        }

        long val = PyInt_AsLong(intobj);
        Py_DECREF(intobj); intobj=NULL;
        if (val == 0)
        {
            cudaError_t err = cudaMemset(rval->devdata, 0, CudaNdarray_SIZE(rval) * sizeof(real));
            Py_XDECREF(rval);
            if (err)
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "CudaNdarray.__setitem__: cudaMemset failed");
                return -1;
            }
            return 0;
        } else {
            Py_XDECREF(rval);
            PyErr_SetString(PyExc_NotImplementedError,
                                "CudaNdarray.__setitem__: we support setting only python scalar of value 0, numpy nd array and CudaNdarray.");
                return -1;
        }
    }

    PyErr_Clear(); // clear PyNumber_Int error.

    if(!CudaNdarray_Check(o) || !CudaNdarray_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray.__setitem__: left must be a CudaNdarrays and right must be a CudaNdarrays, an ndarray or a python scalar of value 0.");
        return -1;
    }

    if (verbose) fprintf(stderr, "CudaNdarray_setitem dest and value are CudaNdarray\n");
    if (cnda_copy_structure_to_device(rval))
    {
        PyErr_SetString(PyExc_RuntimeError, "CudaNdarray.__setitem__: syncing structure to device failed");
        Py_DECREF(rval);
        if (verbose) fprintf(stderr, "CudaNdarray_setitem error end\n");
        return -1;
    }

    PyObject *baseSavedForComparison = rval->base;

    if(CudaNdarray_CopyFromCudaNdarray(rval, (CudaNdarray*)value, true))
    {
        Py_DECREF((PyObject*)rval);
        if (verbose) fprintf(stderr, "CudaNdarray_setitem error end\n");
        return -1;
    }

    assert (rval->base == baseSavedForComparison);
    assert (rval->dev_structure_fresh);

    // Clean up locally-created references
    Py_DECREF(rval);

    return 0;
}


PyMappingMethods CudaNdarrayMappingMethods = {
    CudaNdarray_len, //lenfunc mp_length;               __len__
    CudaNdarray_Subscript, //binaryfunc mp_subscript;   __getitem__
    CudaNdarray_setitem //objobjargproc mp_ass_subscript;                __setitem__
};

////////////////////
//
////////////////////

static PyObject *
CudaNdarray_get_shape(CudaNdarray *self, void *closure)
{
    if (self->nd < 0)
    {
        PyErr_SetString(PyExc_ValueError, "CudaNdarray not initialized");
        return NULL;
    }
    PyObject * rval = PyTuple_New(self->nd);
    for (int i = 0; i < self->nd; ++i)
    {
        if (!rval || PyTuple_SetItem(rval, i, PyInt_FromLong(CudaNdarray_HOST_DIMS(self)[i])))
        {
            Py_XDECREF(rval);
            return NULL;
        }

    }
    return rval;
}

static int
CudaNdarray_set_shape(CudaNdarray *self, PyObject *value, void *closure)
{
    PyErr_SetString(PyExc_NotImplementedError, "TODO: call reshape");
    return -1;
}

static PyObject *
CudaNdarray_get_strides(CudaNdarray *self, void *closure)
{
    if (self->nd < 0)
    {
        PyErr_SetString(PyExc_ValueError, "CudaNdarray not initialized");
        return NULL;
    }
    PyObject * rval = PyTuple_New(self->nd);
    for (int i = 0; i < self->nd; ++i)
    {
        if (!rval || PyTuple_SetItem(rval, i, PyInt_FromLong(CudaNdarray_HOST_STRIDES(self)[i])))
        {
            Py_XDECREF(rval);
            return NULL;
        }

    }
    return rval;
}

static int
CudaNdarray_set_strides(CudaNdarray *self, PyObject *value, void *closure)
{
    PyErr_SetString(PyExc_NotImplementedError, "");
    return -1;
}

static PyObject *
CudaNdarray_get_dev_data(CudaNdarray *self, void *closure)
{
    float * p =  CudaNdarray_DEV_DATA(self);
    //printf("get_dev_data %p %li \n", p, (long int)p );
    return PyInt_FromLong((long int) CudaNdarray_DEV_DATA(self));
}

static int
CudaNdarray_set_dev_data(CudaNdarray *self, PyObject *value, void *closure)
{
    long int newdevdata = PyInt_AsLong(value);
    //printf("set_dev_data %p %li \n",(float*)newdevdata ,newdevdata);
    if (PyErr_Occurred())
    {
        return -1;
    }
    return  CudaNdarray_set_device_data(self, (float*)newdevdata, (CudaNdarray*)self->base);
}

static PyObject *
CudaNdarray_get_dtype(CudaNdarray *self, void *closure)
{
    return PyString_FromString("float32");
}

static PyObject *
CudaNdarray_get_ndim(CudaNdarray *self, void *closure)
{
    return PyInt_FromLong(self->nd);
}

static PyObject *
CudaNdarray_get_base(CudaNdarray *self, void *closure)
{
    PyObject * base = self->base;
    if (!base)
    {
        // We cannot return a NULL pointer, use None instead
        base = Py_None;
    }
    Py_INCREF(base);
    return base;
}

void put_in_dict(PyObject * dict, const char * key, int val)
{
  PyObject * k = PyString_FromString(key);
  PyObject * v = PyInt_FromLong(val);
  PyDict_SetItem(dict, k, v);
  Py_DECREF(k);
  Py_DECREF(v);
}

PyObject *
GetDeviceProperties(PyObject* _unused, PyObject* args)
{
  int dev_id = -1;
  if (! PyArg_ParseTuple(args, "i", &dev_id))
    return NULL;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev_id);

  PyObject * dict = PyDict_New();
  PyObject * str= PyString_FromString("name");
  PyObject * i = PyString_FromString(deviceProp.name);
  PyDict_SetItem(dict, str, i);
  Py_DECREF(str);
  Py_DECREF(i);

  put_in_dict(dict, "major", deviceProp.major);
  put_in_dict(dict, "minor", deviceProp.minor);
#if CUDART_VERSION >= 2020
  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  put_in_dict(dict, "driverVersion", driverVersion);
  put_in_dict(dict, "runtimeVersion", runtimeVersion);
#endif
#if CUDART_VERSION >= 2000

  put_in_dict(dict, "multiProcessorCount", deviceProp.multiProcessorCount);
  //if ConvertSMVer2Cores is not defined in cuda_runtime_api.h, the run time is too old.
  int sm_cores = -1;
  if(deviceProp.major==1)
    sm_cores = 32;
  else if(deviceProp.major==2 && deviceProp.minor==0)
    sm_cores = 32;
  else if(deviceProp.major==2 && deviceProp.minor==1)
    sm_cores = 48;
  put_in_dict(dict, "coresCount", sm_cores * deviceProp.multiProcessorCount);
#endif
  put_in_dict(dict, "totalConstMem", deviceProp.totalConstMem);
  put_in_dict(dict, "sharedMemPerBlock", deviceProp.sharedMemPerBlock);
  put_in_dict(dict, "regsPerBlock", deviceProp.regsPerBlock);
  put_in_dict(dict, "warpSize", deviceProp.warpSize);
  put_in_dict(dict, "maxThreadsPerBlock", deviceProp.maxThreadsPerBlock);
  put_in_dict(dict, "maxThreadsDim0", deviceProp.maxThreadsDim[0]);
  put_in_dict(dict, "maxThreadsDim1", deviceProp.maxThreadsDim[1]);
  put_in_dict(dict, "maxThreadsDim2", deviceProp.maxThreadsDim[2]);
  put_in_dict(dict, "maxGridSize0", deviceProp.maxGridSize[0]);
  put_in_dict(dict, "maxGridSize1", deviceProp.maxGridSize[1]);
  put_in_dict(dict, "maxGridSize2", deviceProp.maxGridSize[2]);
  put_in_dict(dict, "memPitch", deviceProp.memPitch);
  put_in_dict(dict, "textureAlignment", deviceProp.textureAlignment);
  put_in_dict(dict, "clockRate", deviceProp.clockRate);
#if CUDART_VERSION >= 2000
  put_in_dict(dict, "deviceOverlap", deviceProp.deviceOverlap);
#endif
#if CUDART_VERSION >= 2020
  put_in_dict(dict, "kernelExecTimeoutEnabled", deviceProp.kernelExecTimeoutEnabled);
  put_in_dict(dict, "integrated", deviceProp.integrated);
  put_in_dict(dict, "canMapHostMemory", deviceProp.canMapHostMemory);
  put_in_dict(dict, "computeMode", deviceProp.computeMode);
  //in the doc of this fct tell that 0 - Normal mode, 1 - only 1 context, 2 - no context
#endif
#if CUDART_VERSION >= 3000
  put_in_dict(dict, "concurrentKernels", deviceProp.concurrentKernels);
#endif
#if CUDART_VERSION >= 3010
  put_in_dict(dict, "ECCEnabled", deviceProp.ECCEnabled);
#endif
#if CUDART_VERSION >= 3020
  put_in_dict(dict, "tccDriver", deviceProp.tccDriver);
#endif

  return dict;
}

/*
 * Returns in *free and *total respectively, the free and total amount of memory available for allocation by the device in bytes.
 */
PyObject *
GetDeviceMemInfo(PyObject* _unused, PyObject* dummy)
{
    size_t free = 0, total = 0;
    if(g_gpu_context_active == 0){
        PyErr_Format(PyExc_RuntimeError, "No gpu device selected yet. Please make sure the gpu device was initialized by Theano before.");
        return NULL;
    }

    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess){
        PyErr_Format(PyExc_RuntimeError, "Error while getting memory info about the gpu %d");
        return NULL;
    }
    return PyTuple_Pack(2, PyLong_FromLong(free), PyLong_FromLong(total));
}

#if COMPUTE_GPU_MEM_USED
/*
 * Return the size in bytes that Theano currently have allocated on the gpu.
 */
PyObject *
GetTheanoAllocInfo(PyObject* _unused, PyObject* dummy)
{
    return PyLong_FromLong(_allocated_size);
}
#endif

static PyGetSetDef CudaNdarray_getset[] = {
    {"shape",
        (getter)CudaNdarray_get_shape,
        (setter)CudaNdarray_set_shape,
        "shape of this ndarray (tuple)",
        NULL},
    {"_strides",
        (getter)CudaNdarray_get_strides,
        (setter)CudaNdarray_set_strides,
        "data pointer strides (in elements)",
        NULL},
    //gpudata is needed to allow calling pycuda fct with CudaNdarray input.
    {"gpudata",
        (getter)CudaNdarray_get_dev_data,
        NULL,
        "device data pointer",
        NULL},
    {"_dev_data",
        (getter)CudaNdarray_get_dev_data,
        (setter)CudaNdarray_set_dev_data,
        "device data pointer",
        NULL},
    {"dtype",
        (getter)CudaNdarray_get_dtype,
        NULL,
        "The dtype of the element. Now always float32",
        NULL},
    {"size",
        (getter)CudaNdarray_SIZE_Object,
        NULL,
        "The number of elements in this object.",
        NULL},
    //mem_size is neede for pycuda.elementwise.ElementwiseKernel Why do they use size and mem_size of the same value?
    {"mem_size",
        (getter)CudaNdarray_SIZE_Object,
        NULL,
        "The number of elements in this object.",
        NULL},
    {"ndim",
        (getter)CudaNdarray_get_ndim,
        NULL,
        "The number of dimensions in this object.",
        NULL},
    {"base",
        (getter)CudaNdarray_get_base,
        NULL,
        "If this ndarray is a view, base is the original ndarray.",
        NULL},

    {NULL, NULL, NULL, NULL}  /* Sentinel */
};



static PyTypeObject CudaNdarrayType =
{
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "CudaNdarray",             /*tp_name*/
    sizeof(CudaNdarray),       /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)CudaNdarray_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    &CudaNdarrayNumberMethods, /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    &CudaNdarrayMappingMethods,/*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
    "CudaNdarray objects",     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    CudaNdarray_methods,       /* tp_methods */
    CudaNdarray_members,       /* tp_members */
    CudaNdarray_getset,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)CudaNdarray_init,/* tp_init */
    0,                         /* tp_alloc */
    CudaNdarray_new,           /* tp_new */
};

static __global__ void get_gpu_ptr_size(int* dst)
{
    dst[0] = sizeof(float*);
    dst[1] = sizeof(int);
}

PyObject *
CudaNdarray_ptr_int_size(PyObject* _unused, PyObject* args)
{
    int *gpu_data = (int*)device_malloc(sizeof(int)*2);
    if(gpu_data == NULL){
        return PyErr_Format(PyExc_MemoryError,
                            "CudaNdarray_ptr_int_size: Can't allocate memory on the gpu.");
    }
    get_gpu_ptr_size<<<1,1>>>(gpu_data);
    if (cudaSuccess != cublasGetError()){

        device_free(gpu_data);
        return PyErr_Format(PyExc_RuntimeError,
                            "CudaNdarray_ptr_int_size: error when calling the gpu code.");
    }

    // Transfer the result to cpu
    int gpu_sizes[] = {-1,-1};
    cublasGetVector(2, sizeof(int), gpu_data, 1, gpu_sizes, 1);
    device_free(gpu_data);

    if (CUBLAS_STATUS_SUCCESS != cublasGetError()){
        PyErr_SetString(PyExc_RuntimeError, "error copying data to from memory");
        return NULL;
    }
    return Py_BuildValue("iiii", gpu_sizes[0], sizeof(float*), sizeof(int), gpu_sizes[1]);
}

// Initialize the gpu.
// Takes one optional parameter, the device number.
// If provided, it sets that device to be the active device.
// If not provided (usually just to test whether the gpu is available at all),
// it does not set an active device.
// Raises EnvironmentError or ValueError (as appropriate) if the initialization failed.
PyObject *
CudaNdarray_gpu_init(PyObject* _unused, PyObject* args)
{
    int card_nb = 0;
    int card_number_provided = 1;

    PyArg_ParseTuple(args, "|i", &card_nb); // if we're given something wildly invalid, this will throw a TypeError

    if(PyTuple_Size(args) == 0) {
        card_number_provided = 0;
        card_nb = 0;
    }

    int deviceCount;
    cudaError err = cudaGetDeviceCount(&deviceCount);
    if(cudaSuccess != err) {
        return PyErr_Format(PyExc_EnvironmentError,
                            "Unable to get the number of gpus available: %s",
                            cudaGetErrorString(cudaGetLastError()));
    }

    // as soon as the first successful call to a cuda* function is made, a
    // gpu context has been created
    g_gpu_context_active = 1;

    if(deviceCount <= 0) {
        return PyErr_Format(PyExc_EnvironmentError,
                            "Can't use the GPU, no devices support CUDA");
    }
    if(card_number_provided && (card_nb < 0 || card_nb > (deviceCount - 1))) {
        return PyErr_Format(PyExc_ValueError,
                            "Bad device number %d. Only %d devices available.",
                            card_nb,
                            deviceCount);
    }

    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, card_nb);
    if(cudaSuccess != err) {
        return PyErr_Format(PyExc_EnvironmentError,
                            "Unable to get properties of gpu %i: %s",
                            card_nb,
                            cudaGetErrorString(cudaGetLastError()));
    }

    if(deviceProp.major == 9999 && deviceProp.minor == 9999 ){
        return PyErr_Format(PyExc_EnvironmentError,
                            "There is no device that supports CUDA");
    }

    if(card_number_provided) {
        err = cudaSetDevice(card_nb);
        if(cudaSuccess != err) {
            return PyErr_Format(PyExc_EnvironmentError,
                                "Unable to set device %i: %s",
                                card_nb,
                                cudaGetErrorString(cudaGetLastError()));
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *
CudaNdarray_active_device_number(PyObject* _unused, PyObject* _unused_args) {
    // NB: No cuda error checking here; keeps things simple, and it's not
    // really necessary.
    int currentDevice;
    cudaGetDevice(&currentDevice);
    return PyInt_FromLong(currentDevice);
}

PyObject *
CudaNdarray_active_device_name(PyObject* _unused, PyObject* _unused_args) {
    // NB: No cuda error checking here; keeps things simple, and it's not
    // really necessary.
    int currentDevice;
    cudaGetDevice(&currentDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, currentDevice);
    return PyString_FromString(deviceProp.name);
}

PyObject *
CudaNdarray_gpu_shutdown(PyObject* _unused, PyObject* _unused_args) {
    cudaThreadExit();
    g_gpu_context_active = 0; // context has now been closed down
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * This function is tested in theano/misc/test_pycuda_theano_simple.py
 */
PyObject *
CudaNdarray_from_gpu_pointer(PyObject* _unused, PyObject* args)
{
    int verbose = 0;
    PyObject *gpu_ptr = NULL;
    PyObject *shapes = NULL;
    PyObject *strides = NULL;
    PyObject *base = NULL;
    PyObject *rval = NULL;

    //args should consist of 3 python objects
    //The first is the gpu ptr
    //The second if the shape
    //The third if the strides
    if (! PyArg_ParseTuple(args, "OOOO", &gpu_ptr, &shapes, &strides, &base))
        return NULL;

    if (verbose) printf("In CudaNdarray_from_gpu_pointer\n");
    if (!PyLong_Check(gpu_ptr))
    {
        PyErr_Format(PyExc_Exception, "CudaNdarray_from_gpu_pointer: The gpu pointor is not an long");
        return NULL;
    }

    Py_ssize_t nd =  PyObject_Length(shapes);
    if (nd < 0)
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_from_gpu_pointer: Couldn't get length of second argument");
        return NULL;
    }
    Py_ssize_t nd_stride =  PyObject_Length(strides);
    if (nd_stride < 0)
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_from_gpu_pointer: Couldn't get length of third argument");
        return NULL;
    }

    if (nd != nd_stride)
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_from_gpu_pointer: We need the same number of shapes and strides");
        return NULL;
    }

    rval = CudaNdarray_New();

    if (CudaNdarray_set_nd((CudaNdarray *)rval, nd))
    {
        //CudaNdarray_set_nd set the error msg
        return NULL;
    }
    // set gpu pointeur
    assert(((CudaNdarray *)rval)->data_allocated == 0);
    if (CudaNdarray_set_device_data((CudaNdarray *)rval, (float *)PyInt_AsLong(gpu_ptr), base))
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray_from_gpu_pointer: Error while setting the gpu pointor");
        return NULL;

    }

    // Set dims and strides
    for (int i = nd-1; i >= 0; --i)
    {
        PyObject * idx = PyLong_FromLong(i);
        if (idx == NULL)
        {
            PyErr_SetString(PyExc_Exception, "CudaNdarray_from_gpu_pointer: Couldn't make long object to loop over list/tuple");
            return NULL;
        }
        PyObject* dim_ = PyObject_GetItem(shapes, idx);
        PyObject* strd_ = PyObject_GetItem(strides, idx);
        if (!PyInt_Check(dim_))
        {
            PyErr_Format(PyExc_Exception, "CudaNdarray_from_gpu_pointer: shapes[%d] is not an int", i);
            return NULL;
        }
        if (!PyInt_Check(strd_))
        {
            PyErr_Format(PyExc_Exception, "CudaNdarray_from_gpu_pointer: strides[%d] is not an int", i);
            return NULL;
        }
        int dim = PyInt_AsLong(dim_);
        int strd = PyInt_AsLong(strd_);
        CudaNdarray_set_stride((CudaNdarray *)rval, i, strd);
        CudaNdarray_set_dim((CudaNdarray *)rval, i, dim);
        Py_DECREF(idx);
        Py_DECREF(dim_);
        Py_DECREF(strd_);
    }
    if (verbose) printf("CudaNdarray_from_gpu_pointer normal return\n");
    return rval;
}

PyObject *
CudaNdarray_Dot(PyObject* _unused, PyObject* args)
{
    PyObject *l=NULL;
    PyObject *r=NULL;
    PyObject * rval = NULL;

    //args should consist of two python objects ("OO")
    if (! PyArg_ParseTuple(args, "OO", &l, &r))
        return NULL;

    if (!CudaNdarray_Check(l) || !CudaNdarray_Check(r))
    {
        PyErr_SetString(PyExc_TypeError, "CudaNdarray arguments required ");
        goto CudaNdarray_dot_fail;
    }
    if (((CudaNdarray*)l)->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "need 2d CudaNdarray arg for now");
        goto CudaNdarray_dot_fail;
    }
    if (((CudaNdarray*)r)->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "need 2d CudaNdarray arg for now");
        goto CudaNdarray_dot_fail;
    }
    rval = CudaNdarray_New();
    if (!rval)
    {
        goto CudaNdarray_dot_fail;
    }
    int dims[2];
    dims[0] = CudaNdarray_HOST_DIMS((CudaNdarray*)l)[0];
    dims[1] = CudaNdarray_HOST_DIMS((CudaNdarray*)r)[1];
    if (CudaNdarray_alloc_contiguous((CudaNdarray*)rval, 2, dims))
    {
        goto CudaNdarray_dot_fail;
    }
    if (CudaNdarray_gemm(1.0, (CudaNdarray*)l, (CudaNdarray*)r, 0.0, (CudaNdarray*)rval))
    {
        goto CudaNdarray_dot_fail;
    }

    return rval;

    CudaNdarray_dot_fail:
    Py_XDECREF(rval);
    return NULL;
}

static PyObject *
filter(PyObject* __unsed_self, PyObject *args) // args = (data, broadcastable, strict, storage)
{
    /*
     * TODO: DOC what this function should do in the various cases of
     * What is 'strict' supposed to mean in the context of this function?
     * What do we do with input that could be interpreted as matching the broadcastable pattern in strict vs. non-strict cases?
     *
     */
    PyObject *py_data=NULL;
    PyArrayObject * data = NULL;
    int strict = 0;
    PyObject * broadcastable=NULL;
    PyObject * storage=NULL;
    CudaNdarray * rval=NULL;

    //Python object references which are provided to the caller are borrowed references
    if (!PyArg_ParseTuple(args, "OOiO", &py_data, &broadcastable, &strict, &storage)) return NULL;

    if (!PyTuple_Check(broadcastable)){
        PyErr_SetString(PyExc_TypeError, "broadcastable arg should be a tuple of int.");
        return NULL;
    }
    Py_INCREF(py_data);
    Py_INCREF(broadcastable);

    CudaNdarray * cnda = (CudaNdarray*)py_data;

    if (strict || CudaNdarray_Check(py_data))
    {
        //TODO: support non-strict "casting" from a vt to the broadcastable/type/size that we need.
        if (!CudaNdarray_Check(py_data))
        {
            Py_DECREF(py_data);
            Py_DECREF(broadcastable);
            PyErr_SetString(PyExc_TypeError, "strict mode requires CudaNdarray");
            return NULL;
        }
        if (cnda->nd != PyTuple_Size(broadcastable))
        {
            Py_DECREF(py_data);
            Py_DECREF(broadcastable);
            PyErr_Format(PyExc_TypeError, "Wrong rank: %i vs %li", cnda->nd, (long)PyTuple_Size(broadcastable));
            return NULL;
        }
        for (int i = 0; i < cnda->nd; ++i)
        {
            if ((CudaNdarray_HOST_DIMS(cnda)[i] > 1) && PyInt_AsLong(PyTuple_GetItem(broadcastable, Py_ssize_t(i))))
            {
                PyErr_Format(PyExc_TypeError, "Non-unit size in broadcastable vt dimension %i", i);
                Py_DECREF(py_data);
                Py_DECREF(broadcastable);
                return NULL;
            }
        }
        Py_DECREF(broadcastable);
        return py_data;
    }
    else
    {
        data = (PyArrayObject*)PyArray_FromObject(py_data, REAL_TYPENUM, PyTuple_Size(broadcastable), PyTuple_Size(broadcastable));
        if (!data)
        {
            //err message already defined
            Py_DECREF(py_data);
            Py_DECREF(broadcastable);
            return NULL;
        }
        for (int i = 0; i < data->nd; ++i)
        {
            if ((data->dimensions[i] > 1) && PyInt_AsLong(PyTuple_GetItem(broadcastable, Py_ssize_t(i))))
            {
                PyErr_Format(PyExc_TypeError, "Non-unit size in broadcastable dimension %i", i);
                Py_DECREF(data);
                Py_DECREF(py_data);
                Py_DECREF(broadcastable);
                return NULL;
            }
        }
        if (storage && CudaNdarray_Check(storage))
        {
            rval = (CudaNdarray*) storage;
            Py_INCREF(rval);
        }
        else
        {
            rval = (CudaNdarray*) CudaNdarray_New();
        }
        if (rval)
        {
            if (CudaNdarray_CopyFromArray(rval, data))
            {
                Py_DECREF(rval);
                rval = NULL;
            }
        }
        Py_DECREF(data);
        Py_DECREF(py_data);
        Py_DECREF(broadcastable);
        return (PyObject*)rval;
    }
}

//TODO-- CudaNdarray_Dot and CudaNdarray_active_device_name are following different capitalization conventions.
//       Pick one and standardize it, this file is already annoying enough to grep through
static PyMethodDef module_methods[] = {
    {"dimshuffle", CudaNdarray_Dimshuffle, METH_VARARGS, "Returns the dimshuffle of a CudaNdarray."},
    {"dot", CudaNdarray_Dot, METH_VARARGS, "Returns the matrix product of two CudaNdarray arguments."},
    {"gpu_init", CudaNdarray_gpu_init, METH_VARARGS, "Select the gpu card to use; also usable to test whether CUDA is available."},
    {"active_device_name", CudaNdarray_active_device_name, METH_VARARGS, "Get the name of the active device."},
    {"active_device_number", CudaNdarray_active_device_number, METH_VARARGS, "Get the number of the active device."},
    {"gpu_shutdown", CudaNdarray_gpu_shutdown, METH_VARARGS, "Shut down the gpu."},
    {"device_properties", GetDeviceProperties, METH_VARARGS, "Return a dictionary with the device properties."},
    {"mem_info", GetDeviceMemInfo, METH_NOARGS, "Return a tuple with the free and total memory on the gpu in bytes."},
#if COMPUTE_GPU_MEM_USED
    {"theano_allocated", GetTheanoAllocInfo, METH_NOARGS, "Return the size in bytes of memory Theano currently have allocated on the gpu."},
#endif
    {"ptr_int_size", CudaNdarray_ptr_int_size, METH_VARARGS, "Return a tuple with the size of gpu pointer, cpu pointer and int in bytes."},
    {"filter", filter, METH_VARARGS, "filter(obj, broadcastable, strict, storage) returns a CudaNdarray initialized to obj if it matches the constraints of broadcastable.  strict=True prevents any numeric casting. If storage is a CudaNdarray it may be overwritten and used as the return value."},
    {"outstanding_mallocs", outstanding_mallocs, METH_VARARGS, "how many more mallocs have been called than free's"},
    {"from_gpu_pointer", CudaNdarray_from_gpu_pointer, METH_VARARGS, "Used to create a CudaNdarray from already allocated memory on the gpu.(example by pycuda)"},
    {NULL, NULL, NULL, NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initcuda_ndarray(void)
{
    import_array();

    PyObject* m;

    if (PyType_Ready(&CudaNdarrayType) < 0)
        return;

    m = Py_InitModule3("cuda_ndarray", module_methods,
                       "Example module that creates an extension type.");

    if (m == NULL)
        return;

    Py_INCREF(&CudaNdarrayType);
    PyModule_AddObject(m, "CudaNdarray", (PyObject *)&CudaNdarrayType);
#if COMPUTE_GPU_MEM_USED
    for(int i=0;i<TABLE_SIZE;i++){
        _alloc_size_table[i].ptr=NULL;
        _alloc_size_table[i].size=0;
    }
#endif
    //    cublasInit();
    //if (0&&CUBLAS_STATUS_SUCCESS != cublasGetError())
    //{
        //std::cerr << "WARNING: initcuda_ndarray: error initializing device\n";
    //}
    if (0) //TODO: is this necessary?
    {
        int deviceId = 0; // TODO: what number goes here?
        cudaSetDevice(deviceId);
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
            std::cerr << "Error in SetDevice:" << cudaGetErrorString(err) << "\n";
        }
    }
}


//////////////////////////////////////
//
// C API FOR CudaNdarray
//
//////////////////////////////////////

int
CudaNdarray_Check(const PyObject * ob)
{
    //TODO: doesn't work with inheritance
    return CudaNdarray_CheckExact(ob);
}
int
CudaNdarray_CheckExact(const PyObject * ob)
{
    return ((ob->ob_type == &CudaNdarrayType) ? 1 : 0);
}

PyObject *
CudaNdarray_New(int nd)
{
    CudaNdarray *self = (CudaNdarray *)CudaNdarrayType.tp_alloc(&CudaNdarrayType, 0);
    if (self == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "CudaNdarray_New failed to allocate self");
        return NULL;
    }
    CudaNdarray_null_init(self);

    if (nd == 0)
    {
        self->nd = 0;
    }
    else if (nd > 0)
    {
        if (CudaNdarray_set_nd(self, nd))
        {
            Py_DECREF(self);
            return NULL;
        }
    }
    ++_outstanding_mallocs[1];
    return (PyObject *)self;
}



//////////////////////////////
//
// Published helper functions
//
//////////////////////////////

int
cublas_init()
{
    cublasInit();
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error initializing device");
        return -1;
    }
    return 0;
}
int
cublas_shutdown()
{
    cublasShutdown();
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error shutting down device");
        return -1;
    }
    return 0;
}

int
CudaNdarray_CopyFromArray(CudaNdarray * self, PyArrayObject*obj)
{
    int err = CudaNdarray_alloc_contiguous(self, obj->nd, obj->dimensions);
    if (err) {
        return err;
    }

    int typenum = PyArray_TYPE(obj);
    if (typenum != REAL_TYPENUM)
    {
        PyErr_SetString(PyExc_TypeError, "can only copy from float arrays");
        return -1;
    }
    assert( 4 ==  PyArray_ITEMSIZE(obj));
    PyObject * py_src = PyArray_ContiguousFromAny((PyObject*)obj, typenum, self->nd, self->nd);
    if (!py_src) {
        return -1;
    }
    cublasSetVector(PyArray_SIZE(py_src),
            sizeof(real),
            PyArray_DATA(py_src), 1,
            self->devdata, 1);
    CNDA_THREAD_SYNC;
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to device memory");
        Py_DECREF(py_src);
        return -1;
    }
    Py_DECREF(py_src);
    return 0;
}

bool
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

PyObject *
CudaNdarray_new_nd(int nd)
{
    CudaNdarray * rval = (CudaNdarray*) CudaNdarray_New();
    if (!rval || CudaNdarray_set_nd(rval, nd))
    {
        Py_XDECREF(rval);
        rval = NULL;
    }
    return (PyObject *) rval;
}


/**
 * Initialize 'self' as a view of 'base', with memory storage 'data'
 */

int CudaNdarray_set_device_data(CudaNdarray * self, float * data, PyObject * base)
{
    if (self->data_allocated)
    {
        assert(self->devdata);
        if (device_free(self->devdata))
        {
            self->devdata = NULL;
            self->data_allocated = 0;
            return -1;
        }
    }
    // Get the original base object (base.base.base...)
    PyObject * orig_base = base;
    // base is not always a CudaNdarray. It can be a GpuArray from pycuda, ...
    while (orig_base && CudaNdarray_Check(orig_base) && ((CudaNdarray*) orig_base)->base)
    {
        // base_base is itself a view
        orig_base = ((CudaNdarray*) orig_base)->base;
    }
    //N.B. XDECREF and XINCREF are no-ops for NULL pointers
    if (self->base != orig_base)
    {
        Py_XDECREF(self->base);
        self->base = orig_base;
        Py_XINCREF(self->base);
    }
    self->data_allocated = 0;
    self->devdata = data;
    return 0;
}

static __global__ void k_copy_1d(const int N, const float * x, const int sx, float * y, const int sy)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x*blockDim.x)
    {
        y[i*sy] = x[i*sx];
    }
}

//copy from other into self
int CudaNdarray_CopyFromCudaNdarray(CudaNdarray * self, CudaNdarray * other, bool unbroadcast)
{
    int verbose = 0;
    if (verbose>1) fprintf(stderr, "CudaNdarray_CopyFromCudaNdarray\n");

    //standard elemwise size checks
    if (self->nd == -1)
    {
        PyErr_SetString(PyExc_TypeError, "can't copy into un-initialized CudaNdarray");
        return -1;
    }
    if (self->nd != other->nd)
    {
        PyErr_Format(PyExc_NotImplementedError, "CudaNdarray_CopyFromCudaNdarray: need same number of dims. destination nd=%d, source nd=%d. No broadcasting implemented.", self->nd, other->nd);
        return -1;
    }
    //standard elemwise dim checks (also compute total size)
    unsigned int size = 1;
    unsigned int size_source = 1;
    for (int i = 0; i< self->nd; ++i)
    {
        if ((CudaNdarray_HOST_DIMS(self)[i] != CudaNdarray_HOST_DIMS(other)[i])
            && (1!=CudaNdarray_HOST_DIMS(other)[i] || !unbroadcast) )
        {
          PyErr_Format(PyExc_ValueError, "need same dimensions for dim %d, destination=%d, source=%d",
                       i, CudaNdarray_HOST_DIMS(self)[i], CudaNdarray_HOST_DIMS(other)[i]);
            return -1;
        }
        size *= (unsigned int) CudaNdarray_HOST_DIMS(self)[i];
        size_source *= (unsigned int) CudaNdarray_HOST_DIMS(other)[i];
    }
    if (0 == size)
    {
        return 0; //nothing to copy, we're done.
    }
    if (CudaNdarray_is_c_contiguous(self) && CudaNdarray_is_c_contiguous(other) && size == size_source)
    {
        cublasScopy(size, CudaNdarray_DEV_DATA(other), 1, CudaNdarray_DEV_DATA(self), 1);
        CNDA_THREAD_SYNC;
        if (CUBLAS_STATUS_SUCCESS != cublasGetError())
        {
            PyErr_SetString(PyExc_RuntimeError, "Error copying memory");
            return -1;
        }
        return 0;
    }
    //TODO: rewrite these copy operations to be more efficient
    //      See, for example the transpose example in the cuda_sdk.
    switch (self->nd)
    {
        case 0: // scalar
            {
                // THIS CASE SHOULD NEVER HAPPEN BECAUSE SCALARS ARE ALWAYS C CONTIGUOUS
                assert(0);
            }; break;
        case 1: // vector
            {
                if (verbose) fprintf(stderr, "Copying non-contiguous vector\n");
                if (verbose) fprint_CudaNdarray(stderr, other);
                unsigned int n_blocks = std::min(size, (unsigned int)NUM_VECTOR_OP_BLOCKS);
                unsigned int n_threads = std::min(ceil_intdiv(size, n_blocks), (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                k_copy_1d<<<n_blocks, n_threads>>>(size,
                        CudaNdarray_DEV_DATA(other), CudaNdarray_HOST_STRIDES(other)[0],
                        CudaNdarray_DEV_DATA(self), CudaNdarray_HOST_STRIDES(self)[0]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s. (n_blocks=%i, n_threads_per_block=%i)\n", "k_copy_1d", cudaGetErrorString(err), n_blocks, n_threads);
                    return -1;
                }
            }; break;
        default:
            {
                assert (cudaSuccess == cudaGetLastError());
                if (verbose) fprintf(stderr, "Copying with default version unbroadcast=%d\n", unbroadcast);
                // call worker routine
                unsigned int n_blocks = std::min(size, (unsigned int)NUM_VECTOR_OP_BLOCKS);
                unsigned int threads_per_block = std::min(ceil_intdiv(size, n_blocks), (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                CudaNdarray * cuda_dims = other;
                if(unbroadcast)
                    cuda_dims = self;
                //copy from other into self
                k_elemwise_unary_rowmajor_copy<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)other->nd,
                        (const int *)CudaNdarray_DEV_DIMS(cuda_dims),
                        (const float*)CudaNdarray_DEV_DATA(other), (const int *)CudaNdarray_DEV_STRIDES(other),
                        CudaNdarray_DEV_DATA(self),  (const int *)CudaNdarray_DEV_STRIDES(self));
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    //fprint_CudaNdarray(stderr, self);
                    //fprint_CudaNdarray(stderr, other);
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s. (n_blocks=%i, n_threads_per_block=%i)\n", "k_elemwise_unary_rowmajor_copy", cudaGetErrorString(err), n_blocks, threads_per_block);
                    return -1;
                }
            }
    };
    return 0;
}

int CudaNdarray_gemm(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C)
{
    if (A->nd != 2) { PyErr_SetString(PyExc_ValueError, "non-matrix arg to gemm"); return -1; }
    if (B->nd != 2) { PyErr_SetString(PyExc_ValueError, "non-matrix arg to gemm"); return -1; }
    if (C->nd != 2) { PyErr_SetString(PyExc_ValueError, "non-matrix arg to gemm"); return -1; }

    // We must allow dimensions to be zeros.
    if ((CudaNdarray_HOST_DIMS(A)[1] != CudaNdarray_HOST_DIMS(B)[0])
            || (CudaNdarray_HOST_DIMS(A)[0] != CudaNdarray_HOST_DIMS(C)[0])
            || (CudaNdarray_HOST_DIMS(B)[1] != CudaNdarray_HOST_DIMS(C)[1]))
    {
        PyErr_Format(PyExc_ValueError, "dimension mismatch in args to gemm (%i,%i)x(%i,%i)->(%i,%i)",
                CudaNdarray_HOST_DIMS(A)[0],
                CudaNdarray_HOST_DIMS(A)[1],
                CudaNdarray_HOST_DIMS(B)[0],
                CudaNdarray_HOST_DIMS(B)[1],
                CudaNdarray_HOST_DIMS(C)[0],
                CudaNdarray_HOST_DIMS(C)[1]);
        return -1;
    }

    // a matrix has non-unit size and non-unit stride in both directions, we can't operate in-place
    // TODO: make a copy instead of returning in error
    if (((CudaNdarray_HOST_DIMS(A)[0] > 1) && (CudaNdarray_HOST_STRIDES(A)[0] != 1)) && ((CudaNdarray_HOST_DIMS(A)[1] > 1) && (CudaNdarray_HOST_STRIDES(A)[1] != 1)))
    { PyErr_SetString(PyExc_NotImplementedError, "non-unit stride in gemm arg"); return -1; }
    if (((CudaNdarray_HOST_DIMS(B)[0] > 1) && (CudaNdarray_HOST_STRIDES(B)[0] != 1)) && ((CudaNdarray_HOST_DIMS(B)[1] > 1) && (CudaNdarray_HOST_STRIDES(B)[1] != 1)))
    { PyErr_SetString(PyExc_NotImplementedError, "non-unit stride in gemm arg"); return -1; }
    if (((CudaNdarray_HOST_DIMS(C)[0] > 1) && (CudaNdarray_HOST_STRIDES(C)[0] != 1)) && ((CudaNdarray_HOST_DIMS(C)[1] > 1) && (CudaNdarray_HOST_STRIDES(C)[1] != 1)))
    { PyErr_SetString(PyExc_NotImplementedError, "non-unit stride in gemm arg"); return -1; }

    // the unit integer is divided logically into three fields of 4 bits
    // the lowermost 4 bits encode the stride pattern of the output
    // the next higher 4 bits encode the B variable (or y)
    // the next higher 4 bits encode the C variable (or x)
    //
    // the stride pattern for each input is encoded as 0 for unit stride from col to col (Row major)
    //                                                 1 for unit stride from row to row (Col major)

    // a stride of 0 implies a dimension of 1 - so we can actually define
    // a stride of 0 as a 'unit' stride because gemm will never use it.
    int unit = 0;
    if (CudaNdarray_HOST_STRIDES(A)[1] == 1 || CudaNdarray_HOST_STRIDES(A)[1] == 0) {
        unit |= (0x0 << 8);
    } else if (CudaNdarray_HOST_STRIDES(A)[0] == 1 || CudaNdarray_HOST_STRIDES(A)[0] == 0) {
        unit |= (0x1 << 8);
    } else {
        unit |= (0x2 << 8);
    }
    if (CudaNdarray_HOST_STRIDES(B)[1] == 1 || CudaNdarray_HOST_STRIDES(B)[1] == 0) {
        unit |= (0x0 << 4);
    } else if (CudaNdarray_HOST_STRIDES(B)[0] == 1 || CudaNdarray_HOST_STRIDES(B)[0] == 0) {
        unit |= (0x1 << 4);
    } else {
        unit |= (0x2 << 4);
    }
    if (CudaNdarray_HOST_STRIDES(C)[1] == 1 || CudaNdarray_HOST_STRIDES(C)[1] == 0) {
        unit |= (0x0 << 0);
    } else if (CudaNdarray_HOST_STRIDES(C)[0] == 1 || CudaNdarray_HOST_STRIDES(C)[0] == 0) {
        unit |= (0x1 << 0);
    } else {
        unit |= (0x2 << 0);
    }

    // I don't know if cudablas handles negative strides
    if (   (CudaNdarray_HOST_STRIDES(A)[0] < 0)
        || (CudaNdarray_HOST_STRIDES(A)[1] < 0)
        || (CudaNdarray_HOST_STRIDES(B)[0] < 0)
        || (CudaNdarray_HOST_STRIDES(B)[1] < 0)
        || (CudaNdarray_HOST_STRIDES(C)[0] < 0)
        || (CudaNdarray_HOST_STRIDES(C)[1] < 0))
    {
        PyErr_Format(PyExc_ValueError, "illegal strides in args to gemm (%i,%i)x(%i,%i)->(%i,%i)",
                CudaNdarray_HOST_STRIDES(A)[0],
                CudaNdarray_HOST_STRIDES(A)[1],
                CudaNdarray_HOST_STRIDES(B)[0],
                CudaNdarray_HOST_STRIDES(B)[1],
                CudaNdarray_HOST_STRIDES(C)[0],
                CudaNdarray_HOST_STRIDES(C)[1]);
        return -1;
    }

    /* create appropriate strides for malformed matrices that are row or column
     * vectors
     */
    int sa_0 = (CudaNdarray_HOST_DIMS(A)[0] > 1) ? CudaNdarray_HOST_STRIDES(A)[0] : CudaNdarray_HOST_DIMS(A)[1];
    int sa_1 = (CudaNdarray_HOST_DIMS(A)[1] > 1) ? CudaNdarray_HOST_STRIDES(A)[1] : CudaNdarray_HOST_DIMS(A)[0];
    int sb_0 = (CudaNdarray_HOST_DIMS(B)[0] > 1) ? CudaNdarray_HOST_STRIDES(B)[0] : CudaNdarray_HOST_DIMS(B)[1];
    int sb_1 = (CudaNdarray_HOST_DIMS(B)[1] > 1) ? CudaNdarray_HOST_STRIDES(B)[1] : CudaNdarray_HOST_DIMS(B)[0];
    int sc_0 = (CudaNdarray_HOST_DIMS(C)[0] > 1) ? CudaNdarray_HOST_STRIDES(C)[0] : CudaNdarray_HOST_DIMS(C)[1];
    int sc_1 = (CudaNdarray_HOST_DIMS(C)[1] > 1) ? CudaNdarray_HOST_STRIDES(C)[1] : CudaNdarray_HOST_DIMS(C)[0];

    float* a = CudaNdarray_DEV_DATA(A);
    float* b = CudaNdarray_DEV_DATA(B);
    float* c = CudaNdarray_DEV_DATA(C);
    char N = 'N';
    char T = 'T';
    //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
    //TODO: recognize the negative stride and make a copy of the offending argument,
    //rather than aborting
#define CHK_STRIDE_SGEMM(T0, T1, D0, D1, D2, a, x, sx, y, sy, b, z, sz) \
    if (sx == 0){sx = 1;}\
    if (sy == 0){sy = 1;}\
    if (sz == 0){sz = 1;}\
    if ((sx > 0) && (sy > 0) && (sz > 0)) { \
        cublasSgemm(T0, T1, D0, D1, D2, a, x, sx, y, sy, b, z, sz); \
    } else { \
        PyErr_SetString(PyExc_NotImplementedError, "negative stride to sGemm");\
        return -1; \
    }

    switch(unit)
    {
        case 0x000: CHK_STRIDE_SGEMM(N, N, CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(A)[1], alpha, b, sb_0, a, sa_0, beta, c, sc_0); break;
        case 0x100: CHK_STRIDE_SGEMM(N, T, CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(A)[1], alpha, b, sb_0, a, sa_1, beta, c, sc_0); break;
        case 0x010: CHK_STRIDE_SGEMM(T, N, CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(A)[1], alpha, b, sb_1, a, sa_0, beta, c, sc_0); break;
        case 0x110: CHK_STRIDE_SGEMM(T, T, CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(A)[1], alpha, b, sb_1, a, sa_1, beta, c, sc_0); break;
        case 0x001: CHK_STRIDE_SGEMM(T, T, CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(A)[1], alpha, a, sa_0, b, sb_0, beta, c, sc_1); break;
        case 0x101: CHK_STRIDE_SGEMM(N, T, CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(A)[1], alpha, a, sa_1, b, sb_0, beta, c, sc_1); break;
        case 0x011: CHK_STRIDE_SGEMM(T, N, CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(A)[1], alpha, a, sa_0, b, sb_1, beta, c, sc_1); break;
        case 0x111: CHK_STRIDE_SGEMM(N, N, CudaNdarray_HOST_DIMS(C)[0], CudaNdarray_HOST_DIMS(C)[1], CudaNdarray_HOST_DIMS(A)[1], alpha, a, sa_1, b, sb_1, beta, c, sc_1); break;
        default: PyErr_Format(PyExc_ValueError, "some matrix has no unit stride (unit=%i)", unit);
                 return -1;
    };
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        PyErr_Format(PyExc_RuntimeError, "cublassGemm failed (%s)",cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int CudaNdarray_sgemv(float alpha, const CudaNdarray * A, const CudaNdarray * B, float beta, CudaNdarray * C)
{
    /**
    * C <- alpha A B + beta C
    *    A : matrix
    *    B, C: vector
    *    alpha, beta: scalars
    */
    if (A->nd != 2) { PyErr_SetString(PyExc_ValueError, "non-matrix arg to gemv"); return -1; }
    if (B->nd != 1) { PyErr_SetString(PyExc_ValueError, "non-vector arg to gemv"); return -1; }
    if (C->nd != 1) { PyErr_SetString(PyExc_ValueError, "non-vector arg to gemv"); return -1; }

    // We must allow dimensions to be zeros.
    if ((CudaNdarray_HOST_DIMS(A)[1] != CudaNdarray_HOST_DIMS(B)[0])
            || (CudaNdarray_HOST_DIMS(A)[0] != CudaNdarray_HOST_DIMS(C)[0]))
    {
        PyErr_Format(PyExc_ValueError, "dimension mismatch in args to gemv (%i,%i)x(%i)->(%i)",
                CudaNdarray_HOST_DIMS(A)[0],
                CudaNdarray_HOST_DIMS(A)[1],
                CudaNdarray_HOST_DIMS(B)[0],
                CudaNdarray_HOST_DIMS(C)[0]);
        return -1;
    }

    // a matrix has non-unit size and non-unit stride in both directions, we can't operate in-place
    // TODO: make a copy instead of returning in error
    if (((CudaNdarray_HOST_DIMS(A)[0] > 1) && (CudaNdarray_HOST_STRIDES(A)[0] != 1)) && ((CudaNdarray_HOST_DIMS(A)[1] > 1) && (CudaNdarray_HOST_STRIDES(A)[1] != 1)))
    { PyErr_SetString(PyExc_NotImplementedError, "non-unit stride in gemv arg"); return -1; }

    // I don't know if cudablas handles negative strides
    if (   (CudaNdarray_HOST_STRIDES(A)[0] < 0)
        || (CudaNdarray_HOST_STRIDES(A)[1] < 0)
        || (CudaNdarray_HOST_STRIDES(B)[0] < 0)
        || (CudaNdarray_HOST_STRIDES(C)[0] < 0))
    {
        PyErr_Format(PyExc_ValueError, "illegal strides in args to gemv (%i,%i)x(%i)->(%i)",
                CudaNdarray_HOST_STRIDES(A)[0],
                CudaNdarray_HOST_STRIDES(A)[1],
                CudaNdarray_HOST_STRIDES(B)[0],
                CudaNdarray_HOST_STRIDES(C)[0]);
        return -1;
    }

    /* create appropriate strides for malformed matrices that are row or column
     * vectors
     */
    int sa_0 = (CudaNdarray_HOST_DIMS(A)[0] > 1) ? CudaNdarray_HOST_STRIDES(A)[0] : CudaNdarray_HOST_DIMS(A)[1];
    int sa_1 = (CudaNdarray_HOST_DIMS(A)[1] > 1) ? CudaNdarray_HOST_STRIDES(A)[1] : CudaNdarray_HOST_DIMS(A)[0];
    int sb_0 = (CudaNdarray_HOST_DIMS(B)[0] > 1) ? CudaNdarray_HOST_STRIDES(B)[0] : 1;
    int sc_0 = (CudaNdarray_HOST_DIMS(C)[0] > 1) ? CudaNdarray_HOST_STRIDES(C)[0] : 1;

    if (sa_0 == 1)
    {
        cublasSgemv('N',
                CudaNdarray_HOST_DIMS(A)[0], CudaNdarray_HOST_DIMS(A)[1],
                alpha,
                CudaNdarray_DEV_DATA(A), sa_1,
                CudaNdarray_DEV_DATA(B), sb_0,
                beta,
                CudaNdarray_DEV_DATA(C), sc_0);
    }
    else if (sa_1 == 1)
    {
        cublasSgemv('T',
                CudaNdarray_HOST_DIMS(A)[1], CudaNdarray_HOST_DIMS(A)[0],
                alpha,
                CudaNdarray_DEV_DATA(A), sa_0,
                CudaNdarray_DEV_DATA(B), sb_0,
                beta,
                CudaNdarray_DEV_DATA(C), sc_0);
    }
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "too many strides strides in sgemv");
        return -1;
    }
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        PyErr_Format(PyExc_RuntimeError, "cublassGemv failed (%s)",cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int CudaNdarray_sger(float alpha, CudaNdarray * x, CudaNdarray * y, CudaNdarray * A) {
    if (x->nd != 1) { PyErr_SetString(PyExc_ValueError, "non-vector arg x to sger"); return -1; }
    if (y->nd != 1) { PyErr_SetString(PyExc_ValueError, "non-vector arg y to sger"); return -1; }
    if (A->nd != 2) { PyErr_SetString(PyExc_ValueError, "non-matrix arg A to sger"); return -1; }

    if ((CudaNdarray_HOST_DIMS(A)[0] != CudaNdarray_HOST_DIMS(x)[0])
        || (CudaNdarray_HOST_DIMS(A)[1] != CudaNdarray_HOST_DIMS(y)[0])) {
        PyErr_Format(PyExc_ValueError,
                     "dimension mismatch in args to sger (%i)x(%i)->(%i,%i)",
                     CudaNdarray_HOST_DIMS(x)[0],
                     CudaNdarray_HOST_DIMS(y)[0],
                     CudaNdarray_HOST_DIMS(A)[0],
                     CudaNdarray_HOST_DIMS(A)[1]);
        return -1;
    }

    // Maybe this could work, but be safe for now
    if (!CudaNdarray_is_c_contiguous(A)) {
        PyErr_SetString(PyExc_NotImplementedError, "non-c continugous A in sger");
        return -1;
    }
    // Since Sger expects A in col-major, we invert x and y to fake this.
    int x_strides = CudaNdarray_HOST_STRIDES(x)[0];
    CudaNdarray * x_ = x;
    if(x_strides == 0){
        if(CudaNdarray_HOST_DIMS(x)[0] != 1){
            PyErr_Format(PyExc_RuntimeError,
                         "CudaNdarray_sger: Invalid input x(should not happen)."
                         " We received an CudaNdarray vector with a stride of 0"
                         " that have more then 1 elements!");
            return -1;
        }
        x_strides = 4;
    } else if(x_strides < 0){
        x_ = (CudaNdarray*)CudaNdarray_Copy(x);
        x_strides = CudaNdarray_HOST_STRIDES(x_)[0];
    }

    int y_strides = CudaNdarray_HOST_STRIDES(y)[0];
    CudaNdarray * y_ = y;
    if(y_strides == 0){
        if(CudaNdarray_HOST_DIMS(y)[0] != 1){
            PyErr_Format(PyExc_RuntimeError,
                         "CudaNdarray_sger: Invalid input y(should not happen)."
                         " We received an CudaNdarray vector with a stride of 0"
                         " that have more then 1 elements!");
            return -1;
        }
        y_strides = 4;
    } else if(y_strides < 0){
        y_ = (CudaNdarray*)CudaNdarray_Copy(y);
        y_strides = CudaNdarray_HOST_STRIDES(y_)[0];
    }

    if(CudaNdarray_SIZE(A)){
        cublasSger(CudaNdarray_HOST_DIMS(y)[0], CudaNdarray_HOST_DIMS(x)[0], alpha,
                   CudaNdarray_DEV_DATA(y_), y_strides,
                   CudaNdarray_DEV_DATA(x_), x_strides,
                   CudaNdarray_DEV_DATA(A), CudaNdarray_HOST_DIMS(A)[1]);
    }
    CNDA_THREAD_SYNC;
    if(x_ != x)
        Py_DECREF(x_);
    if(y_ != y)
        Py_DECREF(y_);

    cudaError_t err = cudaGetLastError();
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        PyErr_Format(PyExc_RuntimeError, "cublasSger failed (%s)",cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 *
 * Precondition:
 *  a->dim[d] == (dims_a[d]==0) ? (1 << log2_dims_a[d]) : dims_a[d]
 *  z->dim[d] == (z_str[d]==0) ? 1 : dims_a[d];
 *
 *  TODO: templatize this function to support other reductions.
 *  All that needs to change is the initial value for sum, and the reduction operator.
 */

static __global__ void kernel_reduce_sum(const unsigned int size_z,
        const unsigned int nd,
        const int * dims_a,
        const int * log2_dims_a,
        const int * a_str,
        const float * a_data,
        const int * z_str,
        float * z_data)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    //structure data contains the strides and dimensions of both a and z
    // a_dim[0], a_dim[1], ... a_dim[nd-1],
    // a_log2dim[0], a_log2dim[1], ... a_log2dim[nd-1],
    // a_str[0], ... a_str[nd-1],
    // z_str[0], ... z_str[nd-1]
    extern __shared__ int structure_data[];
    for (unsigned int i = threadIdx.x; i < nd; i += blockDim.x)
    {
        structure_data[i+0*nd] = dims_a[i];
        structure_data[i+1*nd] = log2_dims_a[i];
        structure_data[i+2*nd] = a_str[i];
        structure_data[i+3*nd] = z_str[i];
    }
    dims_a = structure_data;
    log2_dims_a = structure_data + nd;
    a_str = structure_data + 2*nd;
    z_str = structure_data + 3*nd;

    __syncthreads(); //wait for all the shared structure to be loaded

    for (unsigned int i = idx; i < size_z; i += numThreads)
    {
        unsigned int ii = i;
        const float * a_data_i = a_data;
        float * z_data_i = z_data;
        unsigned int n_reduce_elements = 1;
        unsigned int n_reduce_dims = 0;
        unsigned int reduce_dim0 = nd-1;


        //In this loop, we locate the initial element of the slice that we'd like to reduce with this thread
        //  At the same time, we [re]calculate the size of that slice (n_reduce_elements)
        for (unsigned int d = 0; d < nd; ++d)
        {
            if (a_str[d] && (!z_str[d])) // this means 'd' is a dimension we are reducing over
            {
                n_reduce_elements *= dims_a[d];
                n_reduce_dims += 1;
                reduce_dim0 = (d < reduce_dim0) ? d : reduce_dim0;
            }
            else //'d' is not a dimension that we are reducing over
            {
                unsigned int pos_d;
                if (log2_dims_a[d]==-1) //TODO: when things are working, use this switch
                {
                    // this branch is not preferred,
                    // because the manual said that integer mod and div operations are slow on gpu
                    pos_d = (ii % dims_a[d]);
                    ii = (ii / dims_a[d]);
                }
                else
                {
                    pos_d = (ii & ((1 << log2_dims_a[d])-1)); //take the lower log2_dims bits
                    ii = (ii >> log2_dims_a[d]);  //shift those lower log2_dims bits off of ii
                }
                a_data_i += pos_d * a_str[d];
                z_data_i += pos_d * z_str[d];
            }
        }
        // now we've got pointers a_data_i and z_data_i into element 0 of the slice over which we are reducing
        // do a similar loop

        float sum = 0.0f;
        switch(n_reduce_dims)
        {
            case 0:
                {
                    sum = a_data_i[0];
                }
                break;
            case 1:
                {
                    const int stride = a_str[reduce_dim0];
                    const float * a_data_i_max = a_data_i + dims_a[reduce_dim0] * stride;
                    while (a_data_i != a_data_i_max)
                    {
                        sum += a_data_i[0];
                        a_data_i += stride;
                    }
                }
                break;
            case 2:
                {
                    int rd = reduce_dim0+1;
                    for (; rd < nd; ++rd)
                    {
                        if (a_str[rd] && (!z_str[rd])) // this means 'rd' is a dimension we are reducing over
                            break;
                    }
                    const int stride0 = a_str[reduce_dim0];
                    const int stride1 = a_str[rd];
                    for (int ii = 0; ii < dims_a[rd]; ++ii)
                    {
                        const float * a_data_ri = a_data_i + ii * stride1;
                        const float * a_data_ri_max = a_data_ri + dims_a[reduce_dim0] * stride0;
                        while (a_data_ri != a_data_ri_max)
                        {
                            sum += a_data_ri[0];
                            a_data_ri += stride0;
                        }
                    }
                };
                break;
            default:
                {
                    for (unsigned int reduce_i = 0; reduce_i < n_reduce_elements; ++reduce_i)
                    {
                        //TODO: optimize this loop to work more like theano's Elemwise.  It's serial code.
                        unsigned int reduce_ii = reduce_i;
                        const float * a_data_ri = a_data_i;

                        //This loop finds the element in the a slice to add.
                        for (unsigned int rd = reduce_dim0; rd < nd; ++rd)
                        {
                            unsigned int pos_d;
                            if (a_str[rd] && (!z_str[rd])) // this means 'd' is a dimension we are reducing over
                            {
                                if (log2_dims_a[rd]==-1)
                                {
                                    // this branch is not preferred,
                                    // because the manual said that integer mod and div operations are slow on gpu
                                    pos_d = (reduce_ii % dims_a[rd]);
                                    reduce_ii = (reduce_ii / dims_a[rd]);
                                }
                                else
                                {
                                    pos_d = (reduce_ii & ((1 << log2_dims_a[rd])-1)); //take the lower log2_dims bits
                                    reduce_ii = (reduce_ii >> log2_dims_a[rd]);  //shift those lower log2_dims bits off of ii
                                }
                                a_data_ri += pos_d * a_str[rd];
                            }
                        }
                        sum += a_data_ri[0];
                    }
                }
        }
        z_data_i[0] = sum;
    }
}

static __global__ void kernel_reduce_sum_1011(
        const unsigned int d0,
        const unsigned int d1,
        const unsigned int d2,
        const unsigned int d3,
        const float *A, const int sA0, const int sA1, const int sA2, const int sA3,
        float * Z, const int sZ0)
{
    const int threadCount = blockDim.x * blockDim.y * blockDim.z;
    const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ float buf[];
    float mysum = 0.0f;

    if (warpSize != 32)
    {
        return;  //TODO: set error code
    }

    for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
    {
        float Ai = A[i0 * sA0 + blockIdx.x * sA1 + threadIdx.y * sA2 + threadIdx.x * sA3];
        mysum += Ai;
    }
    buf[threadNum] = mysum;
    __syncthreads();

    // rest of function is handled by one warp
    if (threadNum < warpSize)
    {
        for (int i = threadNum + warpSize; i < threadCount; i += warpSize)
        {
            mysum += buf[i];
        }
        buf[threadNum] = mysum;
        if (threadNum < 16)
        {
            //reduce so that threadNum 0 has the sum of everything
            if(threadNum + 16 < threadCount) buf[threadNum] += buf[threadNum+16];
            if(threadNum + 8 < threadCount) buf[threadNum] += buf[threadNum+8];
            if(threadNum + 4 < threadCount) buf[threadNum] += buf[threadNum+4];
            if(threadNum + 2 < threadCount) buf[threadNum] += buf[threadNum+2];
            if(threadNum + 1 < threadCount) buf[threadNum] += buf[threadNum+1];
            if (threadNum == 0)
            {
                Z[blockIdx.x*sZ0] = buf[0];
            }
        }
    }
}
/**
 * Dimensions in which the self has size 1 and A has size > 1 are considered summing dimensions
 * Dimensions in which self has size > 1 and A has size > 1 are considered non-summing dimensions, and in this case their sizes must be equal.
 */
int
CudaNdarray_reduce_sum(CudaNdarray * self, CudaNdarray * A)
{
    int verbose = 0;
    //check input rank
    if (self->nd != A->nd)
    {
        PyErr_Format(PyExc_TypeError, "Rank mismatch in CudaNdarray_sum: %i vs %i", self->nd, A->nd);
        return -1;
    }
    for (int i = 0; i < self->nd; ++i)
    {
        if ((CudaNdarray_HOST_DIMS(self)[i] > 1) && (CudaNdarray_HOST_DIMS(self)[i] != CudaNdarray_HOST_DIMS(A)[i]))
        {
            PyErr_Format(PyExc_TypeError, "Dimension mismatch in CudaNdarray_sum: self->dim[%i] == %i , A->dim[%i] = %i",
                    i, CudaNdarray_HOST_DIMS(self)[i], i, CudaNdarray_HOST_DIMS(A)[i]);
            return -1;
        }
    }

    int n_summations = (unsigned int)CudaNdarray_SIZE(self);
    if (verbose)
    {
        std::cerr << "reduce_sum n_summations " << n_summations  << '\n';
        std::cerr << "reduce_sum nd " << self->nd  << '\n';
        fprint_CudaNdarray(stderr, A);
        fprint_CudaNdarray(stderr, self);
    }
    if (0 && (A->nd == 4) //check to see if kernel_reduce_sum_1011 applies
            && (CudaNdarray_HOST_DIMS(self)[0] == 1)
            && (CudaNdarray_HOST_DIMS(self)[2] == 1)
            && (CudaNdarray_HOST_DIMS(self)[3] == 1)
       )
    {
        dim3 n_threads(CudaNdarray_HOST_DIMS(A)[3], CudaNdarray_HOST_DIMS(A)[2]);
        dim3 n_blocks(CudaNdarray_HOST_DIMS(A)[1]);
        while (n_threads.x * n_threads.y * n_threads.z < NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.z;
        n_threads.z -= 1;
        if (n_threads.z > 64) n_threads.z = 64;
        if (n_threads.z)
        {
            if (verbose) printf("trying kernel_reduce_sum_1011\n");
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            kernel_reduce_sum_1011<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_HOST_DIMS(A)[0],
                    CudaNdarray_HOST_DIMS(A)[1],
                    CudaNdarray_HOST_DIMS(A)[2],
                    CudaNdarray_HOST_DIMS(A)[3],
                    CudaNdarray_DEV_DATA(A),
                    CudaNdarray_HOST_STRIDES(A)[0],
                    CudaNdarray_HOST_STRIDES(A)[1],
                    CudaNdarray_HOST_STRIDES(A)[2],
                    CudaNdarray_HOST_STRIDES(A)[3],
                    CudaNdarray_DEV_DATA(self),
                    CudaNdarray_HOST_STRIDES(self)[1]);
            CNDA_THREAD_SYNC;
            if (cudaSuccess == cudaGetLastError()) return 0;
            if (verbose) printf("failed, falling back to kernel_reduce_sum\n");
        }
    }

    int n_threads_per_block = std::min(n_summations,
            NUM_VECTOR_OP_THREADS_PER_BLOCK);
    int n_blocks = std::min(ceil_intdiv(n_summations,n_threads_per_block),
            NUM_VECTOR_OP_BLOCKS);
    int n_structure_cache = self->nd * 4 * sizeof(int);

    if (verbose)
    {
        std::cerr << "n_blocks, n_threads_per_block " << n_blocks << ' ' << n_threads_per_block  << '\n';
    }
    assert (self->nd > 0);
    assert (self->nd == A->nd);
    kernel_reduce_sum<<<n_blocks, n_threads_per_block, n_structure_cache>>>(
            n_summations,
            self->nd,
            CudaNdarray_DEV_DIMS(A),
            CudaNdarray_DEV_LOG2DIMS(A),
            CudaNdarray_DEV_STRIDES(A),
            CudaNdarray_DEV_DATA(A),
            CudaNdarray_DEV_STRIDES(self),
            CudaNdarray_DEV_DATA(self));
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n", "kernel_reduce_sum", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
int
CudaNdarray_reduce_prod(CudaNdarray * self, const CudaNdarray * A)
{
    PyErr_SetString(PyExc_NotImplementedError, "");
    return -1;
}
int
CudaNdarray_reduce_min(CudaNdarray * self, const CudaNdarray * A)
{
    PyErr_SetString(PyExc_NotImplementedError, "");
    return -1;
}
int
CudaNdarray_reduce_max(CudaNdarray * self, const CudaNdarray * A)
{
    PyErr_SetString(PyExc_NotImplementedError, "");
    return -1;
}


/**
 *
 *  pattern is a permutation of [0, 1, ... self->nd-1] with the following twists:
 *  - an element 'd' of the permutation can be dropped if CudaNdarray_HOST_DIMS(self)[d] == 1
 *  - any number of '-1' elements can be in the pattern, and they will cause new ranks (with dim==1) to be inserted.
 *
 *  For example, if CudaNdarray_HOST_DIMS(self) == [4, 5, 1, 6], and pattern = [0,3,-1,-1, 1], then CudaNdarray_HOST_DIMS(self) would be modified to become:
 *     [4, 6, 1, 1, 5] (we dropped the original dim[2]==1, and inserted two singleton dimensions with the -1s.
 */
int
CudaNdarray_dimshuffle(CudaNdarray * self, unsigned int len, const int * pattern)
{
    //TODO: pass a workspace pointer to avoid the internal malloc
    int * newdims = (int *)malloc(sizeof(int) * (len + len + self->nd)); //we tack on the taken buffer here for speed of not having to malloc twice.
    int * newstrides = newdims + len;
    int * dims_taken = newstrides + len;
    if (!newdims)
    {
        PyErr_SetString(PyExc_MemoryError, "CudaNdarray_dimshuffle: Failed to allocate temporary space");
        return -1;
    }
    for (int i = 0; i < self->nd; ++i)
    {
        dims_taken[i] = 0;
    }
    for (int i = 0; i < len; ++i)
    {
        if (pattern[i] < 0)
        {
            newdims[i] = 1;
            newstrides[i] = 0;
        }
        else if(dims_taken[pattern[i]])
        {
            PyErr_Format(PyExc_ValueError, "Cudandarray_dimshuffle: invalid pattern for Cudandarray_dimshuffle. You used the dimensions %d multiple time",
                         pattern[i]);
            free(newdims);
            return -1;
        }
        else if (pattern[i]>= self->nd)
        {
            PyErr_Format(PyExc_ValueError, "Cudandarray_dimshuffle: invalid pattern for Cudandarray_dimshuffle. You asked for a dimensions that don't exist %d for a %d dims CudaNdarray",
                         pattern[i], self->nd);
            free(newdims);
            return -1;
        }
        else
        {
            newdims[i] = CudaNdarray_HOST_DIMS(self)[pattern[i]];
            newstrides[i] = CudaNdarray_HOST_STRIDES(self)[pattern[i]];
            dims_taken[pattern[i]] = 1;
        }
    }
    //Check if we dropped not broadcastable dims
    for (int i = 0; i < self->nd; ++i)
    {
        if (dims_taken[i]==0 && CudaNdarray_HOST_DIMS(self)[i]!=1)
        {
            PyErr_SetString(PyExc_ValueError, "Cudandarray_dimshuffle: You cannot drop a non-broadcastable dimension.");
            free(newdims);
            return -1;
        }
    }
    //swap this structure in for the one in self, and sync to the card
    if (CudaNdarray_set_nd(self, len))
    {
        free(newdims);
        return -1;
    }
    for (int i = 0; i < len; ++i)
    {
        CudaNdarray_set_dim(self, i, newdims[i]);
        CudaNdarray_set_stride(self, i, newstrides[i]);
    }
    if (cnda_copy_structure_to_device(self))
    {
        free(newdims);
        return -1;
    }
    free(newdims);
    return 0;
}



/**
 *
 *  This is the function that bind to python.
 *  See CudaNdarray_dimshuffle to call from C.
 *  We use -1 to mean 'x' as in Tensor Dimshuffle.
 */
PyObject *
CudaNdarray_Dimshuffle(PyObject* _unused, PyObject* args)
{
    PyObject * self = NULL;
    PyObject * pattern_object = NULL;
    int * pattern = NULL;
    PyObject * rval = NULL;
    int success = -1;
    //const int * dims = NULL;

    //args should consist of two python objects ("OO")
    if (! PyArg_ParseTuple(args, "OO", &self, &pattern_object))
        return NULL;

    if (!CudaNdarray_Check(self) )
    {
        PyErr_SetString(PyExc_TypeError, "First argument to cuda_ndarray.dimshuffle must be a CudaNdarray");
        return NULL;
    }

    //parse pattern_object into int * pattern

    Py_ssize_t pattern_dim =  PyObject_Length(pattern_object);

    if (pattern_dim < 0)
    {
        PyErr_SetString(PyExc_TypeError, "Couldn't get length of third argument to cuda_ndarray.dimshuffle");
        return NULL;
    }

    pattern = (int *) malloc( pattern_dim * sizeof(int));

    for (Py_ssize_t i = 0; i < pattern_dim; i++)
    {
        PyObject * idx = PyLong_FromLong(i);

        if (idx == NULL)
        {
            PyErr_SetString(PyExc_Exception, "Couldn't make long object to loop over list/tuple");
            goto CudaNdarray_dimshuffle_fail;
        }

        long elem_value = 0;

        PyObject * elem = PyObject_GetItem(pattern_object, idx);

        if (elem == NULL)
        {
            Py_XDECREF( elem);
            PyErr_SetString(PyExc_ValueError, "Third argument to dimshuffle must be list or tuple of integers");
            goto CudaNdarray_dimshuffle_fail;
        }

        elem_value = PyInt_AsLong(elem);

        if (elem_value == -1 && PyErr_Occurred() )
        {
            Py_XDECREF(elem);
            PyErr_SetString(PyExc_ValueError, "Third argument to dimshuffle must be list or tuple of integers");
            goto CudaNdarray_dimshuffle_fail;
        }

        pattern[i] = elem_value;

        Py_XDECREF( elem );
        Py_XDECREF( idx );
    }

    //allocate rval
    rval =  (PyObject *) CudaNdarray_View((CudaNdarray *) self);

    if (rval == NULL)
    {
        //CudaNdarray_New should have set the exception string
        goto CudaNdarray_dimshuffle_fail;
    }


    //printf("pattern_dim: %d\n",pattern_dim);
    //printf("pattern: %d %d\n",pattern[0],pattern[1]);
    //dims = CudaNdarray_HOST_DIMS( (CudaNdarray *) self);
    //printf("dims before: %d %d\n",dims[0],dims[1]);

    success = CudaNdarray_dimshuffle((CudaNdarray *) rval, pattern_dim, pattern);

    if (success != 0)
    {
        //Exception string should already be set by CudaNdarray_dimshuffle
        goto CudaNdarray_dimshuffle_fail;
    }

    free(pattern);

    return rval;

    CudaNdarray_dimshuffle_fail:

    if (pattern != NULL)
        free(pattern);

    Py_XDECREF(rval);
    return NULL;
}


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


int
CudaNdarray_Equal(CudaNdarray *cnda1, CudaNdarray *cnda2)
{
    return CudaNdarray_EqualAndIgnore(cnda1, cnda2, 0, 0);
}

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

PyObject *
CudaNdarray_SIZE_Object(const CudaNdarray *self, void *closure)
{
    return PyInt_FromLong(CudaNdarray_SIZE(self));
}

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

int CudaNdarray_set_device_data(CudaNdarray * self, float * data, CudaNdarray * base)
{
    return CudaNdarray_set_device_data(self, data, (PyObject *) base);
}

PyObject * CudaNdarray_IS_C_Contiguous(CudaNdarray * self)
{
    return PyBool_FromLong(CudaNdarray_is_c_contiguous(self));
}

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
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:encoding=utf-8:textwidth=79 :
