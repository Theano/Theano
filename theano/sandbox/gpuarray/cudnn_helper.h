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

#if CUDNN_VERSION < 3000
// Here we define the R3 API in terms of functions in the R2 interface
// This is only for what we use

typedef int cudnnConvolutionBwdDataAlgo_t;

#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 0
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 1
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT 2

static cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
  cudnnHandle_t handle,
  const cudnnFilterDescriptor_t filterDesc,
  const cudnnTensorDescriptor_t diffDesc,
  const cudnnConvolutionDescriptor_t convDesc,
  const cudnnTensorDescriptor_t gradDesc,
  cudnnConvolutionBwdDataAlgo_t algo,
  size_t *sizeInBytes) {
  *sizeInBytes = 0;
  return CUDNN_STATUS_SUCCESS;
}

static cudnnStatus_t cudnnConvolutionBackwardData_v3(
  cudnnHandle_t handle,
  const void *alpha,
  const cudnnFilterDescriptor_t filterDesc,
  const void *filterData,
  const cudnnTensorDescriptor_t diffDesc,
  const void *diffData,
  const cudnnConvolutionDescriptor_t convDesc,
  cudnnConvolutionBwdDataAlgo_t algo,
  void *workspace,
  size_t workspaceSizeInBytes,
  const void *beta,
  const cudnnTensorDescriptor_t gradDesc,
  void *gradData) {
  return cudnnConvolutionBackwardData(
    handle,
    alpha,
    filterDesc,
    filterData,
    diffDesc,
    diffData,
    convDesc,
    beta,
    gradDesc,
    gradData);
}

typedef int cudnnConvolutionBwdFilterAlgo_t;

#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 0
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 1
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT 2

static cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
  cudnnHandle_t handle,
  const cudnnTensorDescriptor_t filterDesc,
  const cudnnTensorDescriptor_t diffDesc,
  const cudnnConvolutionDescriptor_t convDesc,
  const cudnnFilterDescriptor_t gradDesc,
  cudnnConvolutionBwdDataAlgo_t algo,
  size_t *sizeInBytes) {
  *sizeInBytes = 0;
  return CUDNN_STATUS_SUCCESS;
}

static cudnnStatus_t cudnnConvolutionBackwardFilter_v3(
  cudnnHandle_t handle,
  const void *alpha,
  const cudnnTensorDescriptor_t srcDesc,
  const void *srcData,
  const cudnnTensorDescriptor_t diffDesc,
  const void *diffData,
  const cudnnConvolutionDescriptor_t convDesc,
  cudnnConvolutionBwdFilterAlgo_t algo,
  void *workspace,
  size_t workspaceSizeInBytes,
  const void *beta,
  const cudnnFilterDescriptor_t gradDesc,
  void *gradData) {
  return cudnnConvolutionBackwardFilter(
    handle,
    alpha,
    srcDesc,
    srcData,
    diffDesc,
    diffData,
    convDesc,
    beta,
    gradDesc,
    gradData);
}

#endif

#endif
