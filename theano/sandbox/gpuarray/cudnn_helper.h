#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

#ifndef CUDNN_VERSION

#define CUDNN_VERSION -1
static inline int cudnnGetVersion() {
  return -1;
}
#endif

#include <assert.h>

// If needed, define element of the V4 interface in terms of elements of
// previous versions
#if defined(CUDNN_VERSION) && CUDNN_VERSION < 4000

#define CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING 5
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING 3

#endif

#endif
