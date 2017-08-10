#section support_code
#include <cuda.h>
#include <mutex>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "dnn_conv_find.h"

using std::vector;
using std::string;
using std::unique_lock;
using std::mutex;

typedef std::unordered_map<string, AlgoRec> AlgoCache;

mutex  algoMutex;
AlgoCache algoCache;



static std::string shape(int* res, int size)
{
    std::stringstream s;
    s<<res[0];
    for (int i=1; i< size; ++i)
        s <<',' << res[i];
    return std::string(s.str().c_str());
}


static std::string shape(cudnnTensorDescriptor_t t)
{
    std::vector<int> res;
    int nbDims;
    cudnnDataType_t type;
    checkCudnnStatus(cudnnGetTensorNdDescriptor(t, 0, &type, &nbDims, nullptr, nullptr));
    res.resize(nbDims * 2);
    checkCudnnStatus(cudnnGetTensorNdDescriptor(t, nbDims, &type, &nbDims, res.data(), res.data() + nbDims));
    res.resize(nbDims);
    return shape(&res[0], nbDims);
};

static std::string shape(cudnnFilterDescriptor_t t, cudnnDataType_t* type)
{

    cudnnTensorFormat_t format;

    int sizes=8;
    // checkCudnnStatus(cudnnGetFilterNdDescriptor(t, 0, nullptr, nullptr, &sizes, nullptr));
    std::vector<int> res;
    res.resize(sizes);
    int outDims;
    checkCudnnStatus(cudnnGetFilterNdDescriptor(t, sizes, type, &format, &outDims, res.data()));
    assert(outDims=sizes);
    return shape(&res[0], outDims);
    
};

static std::string shape(cudnnConvolutionDescriptor_t convDesc)
{
    const int maxDim = 64;
    int nDim=0;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t        computeType;
    
    int                                 padA[maxDim];
    int                                 strideA[maxDim];
    int                                 dilationA[maxDim];
    

    checkCudnnStatus(
        cudnnGetConvolutionNdDescriptor( convDesc, maxDim,
                                         &nDim,
                                         padA,
                                         strideA,
                                         dilationA,
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



    
extern "C" std::string dnn_conv_shape(cudnnTensorDescriptor_t input, void* in,
                           cudnnFilterDescriptor_t filterDesc, void* filter,
                           cudnnConvolutionDescriptor_t convDesc,
                           void* out)
{
    int deviceId;

    cudaGetDevice(&deviceId);
    cudnnDataType_t  dType;
     
    std::stringstream s;
    s << "GPU#" << deviceId << " -dimA" << shape(input) << " -filtA" << shape(filterDesc, &dType) << shape(convDesc);    
    
// there have to be entries for both aligned and not
    if (!all_aligned(dType, in, out, filter))
    {
        s << " [unaligned] ";
    }
    return std::string(s.str().c_str());
}

extern "C" void dnn_conv_update_cache(const std::string& hash, const AlgoRec& rec)
{
    unique_lock<mutex> lock(algoMutex);    
    algoCache[hash] = rec;
}


extern "C" const AlgoRec* dnn_conv_check_cache(const std::string& hash)
{
    unique_lock<mutex> lock(algoMutex);    
    bool cacheHit = false;
    // cout << "dnn_conv_check_cache: "<< hash << endl;

    AlgoCache::iterator hit = algoCache.find(hash);
    if (hit != algoCache.end())
        return &hit->second;
    return nullptr;
}

