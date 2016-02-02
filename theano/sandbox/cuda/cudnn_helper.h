#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

// If needed, define element of the V4 interface in terms of elements of
// previous versions
#if defined(CUDNN_VERSION) && CUDNN_VERSION < 4000

#define CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING 5
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING 3

#endif

#ifndef CUDNN_VERSION
#include <assert.h>

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

typedef cudnnTensor4dDescriptor_t cudnnTensorDescriptor_t;

static inline cudnnStatus_t
cudnnSetTensorNdDescriptor(
  cudnnTensorDescriptor_t tensorDesc,
  cudnnDataType_t dataType,
  int nbDims,
  const int dimA[],
  const int strideA[]) {
  if (nbDims != 4) return CUDNN_STATUS_NOT_SUPPORTED;
  return cudnnSetTensor4dDescriptorEx(
    tensorDesc, dataType,
    dimA[0], dimA[1], dimA[2], dimA[3],
    strideA[0], strideA[1], strideA[2], strideA[3]);
}

static inline cudnnStatus_t
cudnnGetConvolution2dForwardOutputDim(
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
typedef int cudnnConvolutionFwdPreference_t;

#define CUDNN_CONVOLUTION_FWD_NO_WORKSPACE 0

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
cudnnGetConvolutionForwardWorkspaceSize(
 cudnnHandle_t handle,
 const cudnnTensorDescriptor_t srcDesc,
 const cudnnFilterDescriptor_t filterDesc,
 const cudnnConvolutionDescriptor_t convDesc,
 const cudnnTensor4dDescriptor_t destDesc,
 cudnnConvolutionFwdAlgo_t algo,
 size_t *sizeInBytes) {
  *sizeInBytes = 0;
  return CUDNN_STATUS_SUCCESS;
}


static inline cudnnStatus_t
cudnnConvolutionForward_v2(
  cudnnHandle_t handle,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDesc,
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
  cudnnAccumulateResult_t r;
  if (*(float *)beta == 0.0) {
    r = CUDNN_RESULT_NO_ACCUMULATE;
  } else if (*(float *)beta == 1.0) {
    r = CUDNN_RESULT_ACCUMULATE;
  } else {
    assert(0 && "beta must be 0.0 or 1.0");
  }
  return cudnnConvolutionForward(handle, srcDesc, srcData,
				 filterDesc, filterData,
				 convDesc, destDesc, destData,
				 r);
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
  cudnnAccumulateResult_t r;
  if (*(float *)beta == 0.0) {
    r = CUDNN_RESULT_NO_ACCUMULATE;
  } else if (*(float *)beta == 1.0) {
    r = CUDNN_RESULT_ACCUMULATE;
  } else {
    assert(0 && "beta must be 0.0 or 1.0");
  }
  return cudnnConvolutionBackwardFilter(handle, srcDesc, srcData,
					diffDesc, diffData,
					convDesc, gradDesc, gradData,
					r);
}

#define cudnnConvolutionBackwardFilter cudnnConvolutionBackwardFilter_v2

static inline cudnnStatus_t
cudnnConvolutionBackwardData_v2(
  cudnnHandle_t	handle,
  const void *alpha,
  const cudnnFilterDescriptor_t filterDesc,
  const void *filterData,
  const cudnnTensorDescriptor_t diffDesc,
  const void *diffData,
  const cudnnConvolutionDescriptor_t convDesc,
  const void *beta,
  const cudnnTensorDescriptor_t gradDesc,
  void *gradData) {
  assert(*(float *)alpha == 1.0);
  cudnnAccumulateResult_t r;
  if (*(float *)beta == 0.0) {
    r = CUDNN_RESULT_NO_ACCUMULATE;
  } else if (*(float *)beta == 1.0) {
    r = CUDNN_RESULT_ACCUMULATE;
  } else {
    assert(0 && "beta must be 0.0 or 1.0");
  }
  /* This function needs the casting because its params are not
     declared as const */
  return cudnnConvolutionBackwardData(handle,
				      (cudnnFilterDescriptor_t)filterDesc,
				      filterData,
				      (cudnnTensorDescriptor_t)diffDesc,
				      diffData,
				      (cudnnConvolutionDescriptor_t)convDesc,
				      (cudnnTensorDescriptor_t)gradDesc,
				      gradData,
				      r);
}

#define cudnnConvolutionBackwardData cudnnConvolutionBackwardData_v2

static inline cudnnStatus_t
cudnnSetPoolingNdDescriptor(
  cudnnPoolingDescriptor_t poolingDesc,
  const cudnnPoolingMode_t mode,
  int nbDims,
  const int windowDimA[],
  const int paddingA[],
  const int strideA[]) {
  if (nbDims != 2) return CUDNN_STATUS_NOT_SUPPORTED;
  if (paddingA[0] != 0 || paddingA[1] != 0) return CUDNN_STATUS_NOT_SUPPORTED;
  return cudnnSetPoolingDescriptor(poolingDesc, mode,
                                   windowDimA[0], windowDimA[1],
                                   strideA[0], strideA[1]);
}

static inline cudnnStatus_t
cudnnGetPoolingNdDescriptor(
  const cudnnPoolingDescriptor_t poolingDesc,
  const int nbDimsRequested,
  cudnnPoolingMode_t *mode,
  int *nbDims,
  int windowA[],
  int paddingA[],
  int strideA[]) {
  int win0, win1, str0, str1;
  cudnnStatus_t err;
  if (nbDimsRequested < 2) return CUDNN_STATUS_NOT_SUPPORTED;
  err = cudnnGetPoolingDescriptor(poolingDesc, mode, &win0, &win1,
                                  &str0, &str1);
  if (err != CUDNN_STATUS_SUCCESS) return err;
  *nbDims = 2;
  paddingA[0] = 0;
  paddingA[1] = 0;
  windowA[0] = win0;
  windowA[1] = win1;
  strideA[0] = str0;
  strideA[1] = str1;
  return CUDNN_STATUS_SUCCESS;
}

static inline cudnnStatus_t
cudnnPoolingForward_v2(
  cudnnHandle_t handle,
  const cudnnPoolingDescriptor_t poolingDesc,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDesc,
  const void *srcData,
  const void *beta,
  const cudnnTensorDescriptor_t destDesc,
  void *destData) {
  if (*(float*)alpha != 1.0 || *(float *)beta != 0.0) return CUDNN_STATUS_NOT_SUPPORTED;
  return cudnnPoolingForward(handle, poolingDesc, srcDesc, srcData,
                             destDesc, destData);
}
#define cudnnPoolingForward cudnnPoolingForward_v2

static inline cudnnStatus_t
cudnnPoolingBackward_v2(
  cudnnHandle_t handle,
  const cudnnPoolingDescriptor_t poolingDesc,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDesc,
  const void *srcData,
  const cudnnTensorDescriptor_t srcDiffDesc,
  const void *srcDiffData,
  const cudnnTensorDescriptor_t destDesc,
  const void *destData,
  const void *beta,
  const cudnnTensorDescriptor_t destDiffDesc,
  void *destDiffData) {
  if (*(float*)alpha != 1.0 || *(float *)beta != 0.0) return CUDNN_STATUS_NOT_SUPPORTED;
  return cudnnPoolingBackward(handle, poolingDesc,
                              srcDesc, srcData,
                              srcDiffDesc, srcDiffData,
                              destDesc, destData,
                              destDiffDesc, destDiffData);
}
#define cudnnPoolingBackward cudnnPoolingBackward_v2

//Needed for R2 rc2
# define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING CUDNN_POOLING_AVERAGE
#else

// r2 rc1 and rc2 do not have the same macro defined
// I didn't checked if this the right combination, but as we do not wrap the padding interface, it is fine for now.
# define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ((cudnnPoolingMode_t)1)

#endif

#endif
