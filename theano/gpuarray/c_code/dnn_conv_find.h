#pragma once
#include <string>
#include <cuda.h>
#include <cudnn.h>

#if CUDNN_MAJOR < 7
    enum cudnnMathType_t { CUDNN_DEFAULT_MATH=0, CUDNN_TENSOR_OP_MATH = 1 };
#endif

inline cudnnStatus_t checkCudnnStatus(cudnnStatus_t err)
{
    if (err != CUDNN_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "CUDNN Error: %s",
                     cudnnGetErrorString(err));
    }    
    return err;
}
    

/* a common struct for all 3 CUDNN enums */
struct AlgoRec {
        int algo;
        cudnnDataType_t dataType;
        size_t wsSize;
        cudnnMathType_t mathType;
};



