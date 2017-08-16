#section support_code_struct
cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnFilterDescriptor_t APPLY_SPECIFIC(kerns);

static int c_check_groups_for_conv(cudnnConvolutionDescriptor_t desc, int groups) {
#if CUDNN_MAJOR >= 7
  int desc_groups;
  if (groups > 1) {
    cudnnStatus_t err = cudnnGetConvolutionGroupCount(desc, &desc_groups);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
		   "error getting groups for convolution : %s",
		   cudnnGetErrorString(err));
      return -1;
    }
    if (groups != desc_groups) {
      PyErr_SetString(PyExc_MemoryError,
              "groups specified different from convolution descriptor");
      return -1;
    }
  }
  return 1;
#else
  return groups;  
#endif
}

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(kerns) = NULL;
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
	       "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateFilterDescriptor(&APPLY_SPECIFIC(kerns))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %s", 
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input));
if (APPLY_SPECIFIC(output) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output));
if (APPLY_SPECIFIC(kerns) != NULL)
  cudnnDestroyFilterDescriptor(APPLY_SPECIFIC(kerns));

#section support_code
#include <sstream>
#include <vector>
#include <string>
#if __cplusplus < 201103L
#include <tr1/unordered_map>
typedef std::tr1::unordered_map<std::string, AlgoRec> AlgoCache;
#else
#include <unordered_map>
typedef std::unordered_map<std::string, AlgoRec> AlgoCache;
#endif
#include "pthread.h"

#line 69 "dnn_conv_base.c"

using std::vector;
using std::string;

pthread_mutex_t  algoMutex;
AlgoCache        algoCache;

static cudnnStatus_t checkCudnnStatus(cudnnStatus_t err)
{
    if (err != CUDNN_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "CUDNN Error: %s",
                     cudnnGetErrorString(err));
    }    
    return err;
}

static int
c_get_largest_free_block_size(PyGpuContextObject *c) 
{
  size_t free = 0;
  
  int err2 = gpucontext_property(c->ctx, GA_CTX_PROP_LARGEST_MEMBLOCK, &free);
  if (err2 != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "Error when trying to find the "
                 "memory information on the GPU");
  }
  // Guess 4Mb if the info is not available
  if (free == 0) free = 4 * 1024 * 1024;
  return free;
}


static std::string shape(int* res, int size)
{
    std::stringstream s;
    if (size>0) {
      
      s<<res[0];
      for (int i=1; i< size; ++i)
        s <<',' << res[i];
    }
    return std::string(s.str().c_str());
}


static std::string shape(cudnnTensorDescriptor_t t)
{
    std::vector<int> res;
    std::vector<int> stride;
        
    int nbDims;
    cudnnDataType_t type;
    checkCudnnStatus(cudnnGetTensorNdDescriptor(t, 0, &type, &nbDims,0,0));
    res.resize(nbDims);
    stride.resize(nbDims);
    checkCudnnStatus(cudnnGetTensorNdDescriptor(t, nbDims, &type, &nbDims, res.data(), stride.data()));
    return shape(&res[0], nbDims) + shape(&stride[0], nbDims);
    
};

static std::string shape(cudnnFilterDescriptor_t t, cudnnDataType_t* type)
{
    cudnnTensorFormat_t format;
    int sizes = 8;
    
    std::vector<int> res(sizes);
    int outDims;
    checkCudnnStatus(cudnnGetFilterNdDescriptor(t, sizes, type, &format, &outDims, res.data()));
    return shape(&res[0], outDims);
};

static std::string shape(cudnnConvolutionDescriptor_t convDesc)
{
    const int maxDim = 5;
    int nDim=0;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t        computeType;
    
    int                                 padA[maxDim];
    int                                 strideA[maxDim];
    int                                 dilationA[maxDim];    

    checkCudnnStatus(
        cudnnGetConvolutionNdDescriptor( convDesc, maxDim,
                                         &nDim,
                                         &padA[0],
                                         &strideA[0],
                                         &dilationA[0],
                                         &mode,
                                         &computeType ));
    
    return std::string("-mode ") + (((int)mode==0) ? "conv" : "corr") + " -padA" + shape(padA,nDim) + " -convStrideA " + shape(strideA, nDim)  + " -dilationA " + shape(dilationA, nDim);
}

static bool all_aligned(cudnnDataType_t type, void* in, void* out, void* filter)
{
        size_t alignMask = (type == CUDNN_DATA_HALF) ? 0x7F : 0xFF ;
        // there have to be entries for both aligned and not
        if (((size_t)in | (size_t)out | (size_t)filter) & alignMask)
        {
            return false;
        }
        return true;
}

static std::string dnn_conv_shape(cudnnTensorDescriptor_t inputDesc, PyGpuArrayObject* input,
				  cudnnFilterDescriptor_t filterDesc, PyGpuArrayObject* filter,
				  cudnnConvolutionDescriptor_t convDesc,
				  PyGpuArrayObject* output, int groups)
{
    cudnnDataType_t  dType;
    std::stringstream s;
    int expected_output_dims[5] = {0};
    cudnnStatus_t err = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputDesc, filterDesc,
							      PyGpuArray_NDIM(filter), expected_output_dims);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "error computing convolution output dim: %s",
                   cudnnGetErrorString(err));
      return "";
    }
    if (PyGpuArray_NDIM(filter) == 4) {
      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
          (PyGpuArray_DIMS(output)[1] / groups  != expected_output_dims[1]) ||
          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3])) {
        PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%ldx%ld"
                     " but received gradient with shape %dx%dx% dx%d",
                     expected_output_dims[0], expected_output_dims[1] / groups,
                     expected_output_dims[2], expected_output_dims[3],
                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3]);
        return "";
      }
    } else if (PyGpuArray_NDIM(filter) == 5) {
      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
          (PyGpuArray_DIMS(output)[1] != expected_output_dims[1]) ||
          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3]) ||
          (PyGpuArray_DIMS(output)[4] != expected_output_dims[4])) {
        PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %ldx%ldx%ldx%ldx%ld"
                     " but received gradient with shape %ldx%ldx%ldx%ldx%ld",
                     expected_output_dims[0], expected_output_dims[1],
                     expected_output_dims[2], expected_output_dims[3],
                     expected_output_dims[4],
                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3],
                     PyGpuArray_DIMS(output)[4]);
        return "";
      }
    }
    
    s << "-g" << groups << " -dimA" << shape(inputDesc) << " -filtA" <<
      shape(filterDesc, &dType) << shape(convDesc);    
    
// there have to be entries for both aligned and not
    if (!all_aligned(dType, PyGpuArray_DEV_DATA(input), PyGpuArray_DEV_DATA(output), PyGpuArray_DEV_DATA(filter)))
    {
      s << " [unaligned] ";
    }
    return std::string(s.str().c_str());
}

static void dnn_conv_update_cache(const std::string& hash, const AlgoRec& rec)
{
  pthread_mutex_lock(&algoMutex);    
  algoCache[hash] = rec;
  pthread_mutex_unlock(&algoMutex);
}


static const AlgoRec* dnn_conv_check_cache(const std::string& hash)
{
  pthread_mutex_lock(&algoMutex);    
  bool cacheHit = false;
  const AlgoRec* ret = 0;
  
  // cout << "dnn_conv_check_cache: "<< hash << endl;
  
  AlgoCache::iterator hit = algoCache.find(hash);
  
  if (hit != algoCache.end())
    ret = &hit->second;

  pthread_mutex_unlock(&algoMutex);
  return ret;
}

