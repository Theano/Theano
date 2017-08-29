#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

#ifndef CUDNN_VERSION

#define CUDNN_VERSION -1
static inline int cudnnGetVersion() {
  return -1;
}
#endif

#if CUDNN_MAJOR < 7
    enum cudnnMathType_t { CUDNN_DEFAULT_MATH=0, CUDNN_TENSOR_OP_MATH = 1 };
#endif
/* a common struct for all 3 CUDNN enums */
struct AlgoRec {
        int algo;
        size_t wsSize;
        cudnnMathType_t mathType;
};

#endif
