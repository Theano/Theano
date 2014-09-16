#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

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

#endif
