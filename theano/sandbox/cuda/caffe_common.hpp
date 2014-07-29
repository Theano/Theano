// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

//#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
//#include <glog/logging.h>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n); \
i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#endif  // CAFFE_COMMON_HPP_
