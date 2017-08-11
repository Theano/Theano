#section support_code
#include <cuda.h>
#include <sstream>
#include <vector>
#include <string>
#include "dnn_conv_find.h"
#if __cplusplus < 201103L
#include <tr1/unordered_map>
typedef std::tr1::unordered_map<std::string, AlgoRec> AlgoCache;
#else
#include <unordered_map>
typedef std::unordered_map<std::string, AlgoRec> AlgoCache;
#endif
#include "pthread.h"

#line 10 "dnn_conv_find.cc"

using std::vector;
using std::string;

pthread_mutex_t  algoMutex;
AlgoCache        algoCache;



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

std::string dnn_conv_shape(cudnnTensorDescriptor_t input, void* in,
                           cudnnFilterDescriptor_t filterDesc, void* filter,
                           cudnnConvolutionDescriptor_t convDesc,
                           void* out)
{
    cudnnDataType_t  dType;
     
    std::stringstream s;

    s << " -dimA" << shape(input) << " -filtA" << shape(filterDesc, &dType) << shape(convDesc);    
    
// there have to be entries for both aligned and not
    if (!all_aligned(dType, in, out, filter))
    {
        s << " [unaligned] ";
    }
    return std::string(s.str().c_str());
}

void dnn_conv_update_cache(const std::string& hash, const AlgoRec& rec)
{
  pthread_mutex_lock(&algoMutex);    
  algoCache[hash] = rec;
  pthread_mutex_unlock(&algoMutex);
}


const AlgoRec* dnn_conv_check_cache(const std::string& hash)
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

