#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

#ifndef CUDNN_VERSION

// Here we define the R2 API in terms of functions in the R1 interface
// This is only for what we use

static inline const char *cudnnGetErrorString(cudnnStatus_t err) {
  switch (err) {
  case CUDNN_STATUS_SUCCESS:
    return "The operation completed successfully.";
  case CUDNN_STATUS_NOT_INITIALIZED:
    return "The handle was not initialized(Is your driver recent enought?).";
  case CUDNN_STATUS_ALLOC_FAILED:
    return "Ressource allocation failed inside the library.";
  case CUDNN_STATUS_BAD_PARAM:
    return "An incorrect value was passed in.";
  case CUDNN_STATUS_ARCH_MISMATCH:
    return "The current GPU does not support the required features (only cc 3.0+ are supported).";
  case CUDNN_STATUS_MAPPING_ERROR:
    return "An access to GPU memory space failed (probably due to a failure to bind texture).";
  case CUDNN_STATUS_EXECUTION_FAILED:
    return "A kernel failed to execute.";
  case CUDNN_STATUS_INTERNAL_ERROR:
    return "An internal cuDNN operation failed.";
  case CUDNN_STATUS_NOT_SUPPORTED:
    return "The combination of parameters is not currently supported.";
  default:
    return "Unknown error code.";
  }
}

// some macros to help support cudnn R1 while using R2 code.
#define cudnnCreateTensorDescriptor cudnnCreateTensor4dDescriptor
#define cudnnDestroyTensorDescriptor cudnnDestroyTensor4dDescriptor
#define cudnnSetFilter4dDescriptor cudnnSetFilterDescriptor

typedef cudnnTensorDescriptor_t cudnnTensor4dDescriptor_t;

static inline cudnnStatus_t
cdnnGetConvolution2dForwardOutputDim(
  const cudnnConvolutionDescriptor_t convDesc,
  const cudnnTensorDescriptor_t inputTensorDesc,
  const cudnnFilterDescriptor_t filterDesc,
  int *n,
  int *c,
  int *h,
  int *w) {
  return cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD,
				   n, c, h, w);
}

typedef int cudnnConvolutionFwdAlgo_t;

static inline cudnnStatus_t
cudnnGetConvolutionForwardAlgorithm(
  cudnnHandle_t handle,
  const cudnnTensorDescriptor_t srcDesc,
  const cudnnFilterDescriptor_t filterDesc,
  const cudnnConvolutionDescriptor_t convDesc,
  const cudnnTensorDescriptor_t destDesc,
  cudnnConvolutionFwdPreference_t preference,
  size_t memoryLimitInbytes,
  cudnnConvolutionFwdAlgo_t *algo) {
  *algo = 0;
  return CUDNN_STATUS_SUCCESS;
}

static inline cudnnStatus_t
cudnnConvolutionForward_v2(
  cudnnHandle_t handle,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDest,
  const void *srcData,
  const cudnnFilterDescriptor_t filterDesc,
  const void *filterData,
  const cudnnConvolutionDescriptor_t convDesc,
  cudnnConvolutionFwdAlgo_t algo,
  void *workSpace,
  size_t workSpaceSizeInBytes,
  const void *beta,
  const cudnnTensorDescriptor_t destDesc,
  void *destData) {
  assert(*(float *)alpha == 1.0);
  assert(*(float *)beta == 0.0);
  return cudnnConvolutionForward(handle, srcDesc, srcData,
				 filterDesc, filterData,
				 convDesc, destDesc, destData,
				 CUDNN_RESULT_NO_ACCUMULATE);
}
#define cudnnConvolutionForward cudnnConvolutionForward_v2

static inline cudnnStatus_t
cudnnConvolutionBackwardFilter_v2(
  cudnnHandle_t	handle,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDesc,
  const void *srcData,
  const cudnnTensorDescriptor_t diffDesc,
  const void *diffData,
  const cudnnConvolutionDescriptor_t convDesc,
  const void *beta,
  const cudnnFilterDescriptor_t gradDesc,
  void *gradData) {
  assert(*(float *)alpha == 1.0);
  assert(*(float *)beta == 0.0);
  return cudnnConvolutionBackwardFilter(handle, srcDesc, srcData,
					diffDesc, diffData,
					convDesc, gradDesc, gradData,
					CUDNN_RESULT_NO_ACCUMULATE);
}

#define cudnnConvolutionBackwardFilter cudnnConvolutionBackwardFilter_v2

static inline cudnnStatus_t
cudnnConvolutionBackwardData_v2(
  cudnnHandle_t	handle,
  const void *alpha,
  const cudnnTensorDescriptor_t filterDesc,
  const void *filterData,
  const cudnnTensorDescriptor_t diffDesc,
  const void *diffData,
  const cudnnConvolutionDescriptor_t convDesc,
  const void *beta,
  const cudnnFilterDescriptor_t gradDesc,
  void *gradData) {
  assert(*(float *)alpha == 1.0);
  assert(*(float *)beta == 0.0);
  return cudnnConvolutionBackwardFilter(handle, filterDesc, filterData,
					diffDesc, diffData,
					convDesc, gradDesc, gradData,
					CUDNN_RESULT_NO_ACCUMULATE);
}

#define cudnnConvolutionBackwardData cudnnConvolutionBackwardData_v2

#endif

#endif
