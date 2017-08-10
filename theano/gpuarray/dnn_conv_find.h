#include <string>
#include <cuda.h>
#include <cudnn.h>

#if CUDNN_MAJOR < 7
    enum cudnnMathType_t { CUDNN_DEFAULT_MATH=0, CUDNN_TENSOR_OP_MATH = 1 };
#endif

inline void checkCudnnStatus(cudnnStatus_t err)
{
    if (err != CUDNN_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "CUDNN Error: %s",
                     cudnnGetErrorString(err));
    }    
}
    

/* a common struct for all 3 CUDNN enums */
struct AlgoRec {
        int algo;
        cudnnDataType_t dataType;
        size_t ws;
        cudnnMathType_t mathType;
};

const AlgoRec* dnn_conv_check_cache(const std::string&);

std::string dnn_conv_shape(cudnnTensorDescriptor_t input, void* in,
                           cudnnFilterDescriptor_t filterDesc, void* filter,
                           cudnnConvolutionDescriptor_t convDesc,
                           void* out);
void dnn_conv_update_cache(const std::string& hash, const AlgoRec& rec);

