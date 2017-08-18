#section support_code_struct
cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnFilterDescriptor_t APPLY_SPECIFIC(kerns);

static int c_check_groups_for_conv(cudnnConvolutionDescriptor_t desc, int groups) {
#if CUDNN_MAJOR >= 7
  int desc_groups;
  if (groups > 1) {
    cudnnStatus_t err = cudnnGetConvolutionGroupCount(desc, &desc_groups);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
		   "error getting groups for convolution : %s",
		   cudnnGetErrorString(err));
      return -1;
    }
    if (groups != desc_groups) {
      PyErr_SetString(PyExc_MemoryError,
              "groups specified different from convolution descriptor");
      return -1;
    }
  }
  return 1;
#else
  return groups;
#endif
}

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(kerns) = NULL;
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
	       "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateFilterDescriptor(&APPLY_SPECIFIC(kerns))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %s",
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input));
if (APPLY_SPECIFIC(output) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output));
if (APPLY_SPECIFIC(kerns) != NULL)
  cudnnDestroyFilterDescriptor(APPLY_SPECIFIC(kerns));

#section support_code
#include <sstream>
#include <string>
#if __cplusplus < 201103L
#include <tr1/unordered_map>
typedef std::tr1::unordered_map<std::string, AlgoRec> AlgoCache;
#else
#include <unordered_map>
typedef std::unordered_map<std::string, AlgoRec> AlgoCache;
#endif
#include "pthread.h"

#line 73 "dnn_conv_base.c"

pthread_mutex_t  algoMutex;
AlgoCache        algoCache;

static cudnnStatus_t checkCudnnStatus(cudnnStatus_t err, const char* msg)
{
    if (err != CUDNN_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "CUDNN Error: %s: %s",
                     msg, cudnnGetErrorString(err));
    }
    return err;
}

static int
c_get_largest_free_block_size(PyGpuContextObject *c)
{
  size_t free = 0;

  int err2 = gpucontext_property(c->ctx, GA_CTX_PROP_LARGEST_MEMBLOCK, &free);
  if (err2 != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "Error when trying to find the "
                 "memory information on the GPU");
  }
  // Guess 4Mb if the info is not available
  if (free == 0) free = 4 * 1024 * 1024;
  return free;
}

/** Check if convolution output tensor has expected dimensions
    depending on given inputs and number of groups.
    return 0 if everything is ok, non-0 on error.
**/
static int dnn_check_convolution_output(cudnnConvolutionDescriptor_t convDesc,
                                        cudnnTensorDescriptor_t inputDesc,
                                        cudnnFilterDescriptor_t filterDesc,
                                        size_t tensorNdim,
                                        PyGpuArrayObject* output,
                                        int groups) {
    int expected_output_dims[5] = {0};
    cudnnStatus_t err = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputDesc, filterDesc,
                                                              tensorNdim, expected_output_dims);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError, "error computing convolution output dim: %s",
                   cudnnGetErrorString(err));
      return 1;
    }
    if (tensorNdim == 4) {
      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
          (PyGpuArray_DIMS(output)[1] / groups != expected_output_dims[1]) ||
          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3])) {
        PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %dx%dx%dx%d"
                     " but received %ldx%ldx%ldx%ld",
                     expected_output_dims[0], expected_output_dims[1] * groups,
                     expected_output_dims[2], expected_output_dims[3],
                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3]);
        return 1;
      }
    } else if (tensorNdim == 5) {
      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
          (PyGpuArray_DIMS(output)[1] / groups != expected_output_dims[1]) ||
          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3]) ||
          (PyGpuArray_DIMS(output)[4] != expected_output_dims[4])) {
        PyErr_Format(PyExc_ValueError, "impossible convolution output dim: expected %dx%dx%dx%dx%d"
                     " but received %ldx%ldx%ldx%ldx%ld",
                     expected_output_dims[0], expected_output_dims[1] * groups,
                     expected_output_dims[2], expected_output_dims[3],
                     expected_output_dims[4],
                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3],
                     PyGpuArray_DIMS(output)[4]);
        return 1;
      }
    }
    return 0;
}

static std::string shape(int* res, int size)
{
    std::ostringstream s;
    if (size > 0) {

      s << res[0];
      for (int i = 1; i < size; ++i)
        s <<',' << res[i];
    }
    return s.str();
}

static std::string shape(cudnnTensorDescriptor_t t)
{
    // cuDNN can handle up to CUDNN_DIM_MAX dimensions.
    int res[CUDNN_DIM_MAX];
    int stride[CUDNN_DIM_MAX];
    int nbDims;
    cudnnDataType_t type;
    checkCudnnStatus(cudnnGetTensorNdDescriptor(t, CUDNN_DIM_MAX, &type, &nbDims, res, stride),
                     "error getting tensor description");
    if (PyErr_Occurred()) return "";
    return shape(res, nbDims) + "," + shape(stride, nbDims);
};

static std::string shape(cudnnFilterDescriptor_t t, cudnnDataType_t* type)
{
    cudnnTensorFormat_t format;
    int res[CUDNN_DIM_MAX];
    int outDims;
    checkCudnnStatus(cudnnGetFilterNdDescriptor(t, CUDNN_DIM_MAX, type, &format, &outDims, res),
                     "error getting filter description");
    if (PyErr_Occurred()) return "";
    return shape(res, outDims);
};

static std::string shape(cudnnConvolutionDescriptor_t convDesc)
{
    int nDim;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t        computeType;

    int                                 padA[5];
    int                                 strideA[5];
    int                                 dilationA[5];

    checkCudnnStatus(
        cudnnGetConvolutionNdDescriptor( convDesc, 5,
                                         &nDim,
                                         &padA[0],
                                         &strideA[0],
                                         &dilationA[0],
                                         &mode,
                                         &computeType ),
        "error getting convolution description");
    if (PyErr_Occurred()) return "";

    return (std::string("-mode ") +
            ((mode == CUDNN_CONVOLUTION) ? "conv" : "cross") +
            " -pad " +
            shape(padA, nDim) +
            " -subsample " +
            shape(strideA, nDim) +
            " -dilation " +
            shape(dilationA, nDim));
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

static std::string dnn_conv_shape(cudnnTensorDescriptor_t inputDesc, PyGpuArrayObject* input,
				  cudnnFilterDescriptor_t filterDesc, PyGpuArrayObject* filter,
				  cudnnConvolutionDescriptor_t convDesc,
				  PyGpuArrayObject* output, int groups)
{
    cudnnDataType_t  dType;
    std::ostringstream s;
    int expected_output_dims[5] = {0};
    if (dnn_check_convolution_output(convDesc, inputDesc, filterDesc, PyGpuArray_NDIM(filter), output, groups) != 0)
        return "";
    std::string shapeInput = shape(inputDesc);
    std::string shapeFilter = shape(filterDesc, &dType);
    std::string shapeConvDesc = shape(convDesc);
    if (shapeInput.empty() || shapeFilter.empty() || shapeConvDesc.empty())
        return "";
    s << "-g " << groups << " -dim " << shapeInput << " -filt " <<
      shapeFilter << " " << shapeConvDesc;

    // there have to be entries for both aligned and not.
    if (!all_aligned(dType, PyGpuArray_DEV_DATA(input), PyGpuArray_DEV_DATA(output), PyGpuArray_DEV_DATA(filter)))
    {
      s << " [unaligned]";
    }
    return s.str();
}

static void dnn_conv_update_cache(const std::string& hash, const AlgoRec& rec)
{
  pthread_mutex_lock(&algoMutex);
  algoCache[hash] = rec;
  pthread_mutex_unlock(&algoMutex);
}

static const AlgoRec* dnn_conv_check_cache(const std::string& hash)
{
  pthread_mutex_lock(&algoMutex);
  const AlgoRec* ret = 0;

  AlgoCache::iterator hit = algoCache.find(hash);

  if (hit != algoCache.end())
    ret = &hit->second;

  pthread_mutex_unlock(&algoMutex);
  return ret;
}
