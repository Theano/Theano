// This uses a lot of code from Caffe (http://caffe.berkeleyvision.org/);
// sources are clearly marked. Below we reproduce the original license of
// the Caffe software.
/*
Copyright (c) 2014, The Regents of the University of California (Regents)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#undef _GLIBCXX_ATOMIC_BUILTINS


// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CUDA_NUM_THREADS = 1024;
#else
    const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


// (Adapted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu)
// Kernels for fast unfold + copy
__global__ void im3d2col_kernel(const int n, const float* data_im,
                                const int height, const int width, const int depth,
                                const int kernel_h, const int kernel_w, const int kernel_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                float* data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int d_out = index % depth_col;
    int w_index = index / depth_col;
    int w_out = w_index % width_col;
    int h_index = w_index / width_col;
    int h_out = h_index % height_col;

    int channel_in = h_index / height_col;
    //channel_in = 1;

    int channel_out = channel_in * kernel_h * kernel_w * kernel_d;

    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    int d_in = d_out * stride_d - pad_d;

    float* data_col_ptr = data_col;
    data_col_ptr += channel_out * (height_col * width_col * depth_col) +
      h_out * (width_col * depth_col) + w_out * depth_col + d_out;

    const float* data_im_ptr = data_im;
    data_im_ptr += channel_in * (height * width * depth) +
      h_in * (width * depth) + w_in * depth + d_in;

    for (int i = 0; i < kernel_h; ++i)
    {
      int h = h_in + i;
      for (int j = 0; j < kernel_w; ++j)
      {
        int w = w_in + j;
        for (int k = 0; k < kernel_d; ++k)
        {
          int d = d_in + k;
          *data_col_ptr = (h >= 0 && w >= 0 && d >= 0 &&
                           h < height && w < width && d < depth) ?
                           data_im_ptr[i * (width * depth) + j *depth + k] : 0;
          data_col_ptr += height_col * width_col * depth_col;
        }
      }
    }
  }
}

void im3d2col(const float* data_im, const int channels,
              const int height, const int width, const int depth,
              const int kernel_h, const int kernel_w, const int kernel_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              float* data_col)
{
  // We are going to launch channels * height_col * width_col * depth_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
  int num_kernels = channels * height_col * width_col * depth_col;
  im3d2col_kernel<<<GET_BLOCKS(num_kernels),
                    CUDA_NUM_THREADS>>>(num_kernels, data_im,
                                        height, width, depth,
                                        kernel_h, kernel_w, kernel_d,
                                        pad_h, pad_w, pad_d,
                                        stride_h, stride_w, stride_d,
                                        height_col, width_col, depth_col,
                                        data_col);
}


__global__ void col2im3d_kernel(const int n, const float* data_col,
                                const int height, const int width, const int depth,
                                const int channels,
                                const int patch_h, const int patch_w, const int patch_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                float* data_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0;
    int d = index % depth + pad_d;
    int w_index = index / depth;
    int w = w_index % width + pad_w;
    int h_index = w_index / width;
    int h = h_index % height + pad_h;
    int c = h_index / height;

    // compute the start and end of the output
    int d_col_start = (d < patch_d) ? 0 : (d - patch_d) / stride_d + 1;
    int d_col_end = min(d / stride_d + 1, depth_col);
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

    int offset =
      (c * patch_h * patch_w * patch_d + h * patch_w * patch_d + w * patch_d + d) * height_col * width_col * depth_col;

    int coeff_h_col = (1 - stride_h * patch_w * patch_d * height_col) * width_col * depth_col;
    int coeff_w_col = (1 - stride_w * patch_d * height_col * width_col) * depth_col;
    int coeff_d_col = (1 - stride_d * height_col * width_col * depth_col);
    for (int d_col = d_col_start; d_col < d_col_end; ++d_col)
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col + d_col * coeff_d_col];
      }
   }
    data_im[index] = val;
  }
}

void col2im3d(const float* data_col, const int channels,
              const int height, const int width, const int depth,
              const int patch_h, const int patch_w, const int patch_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              float* data_im)
{
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - patch_d) / stride_d + 1;
  int num_kernels = channels * height * width * depth;

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im3d_kernel<<<GET_BLOCKS(num_kernels),
                    CUDA_NUM_THREADS>>>(num_kernels, data_col,
                                        height, width, depth, channels,
                                        patch_h, patch_w, patch_d,
                                        pad_h, pad_w, pad_d,
                                        stride_h, stride_w, stride_d,
                                        height_col, width_col, depth_col,
                                        data_im);
}




// Theano op code
// Authors: Arjun Jain, Frederic Bastien, Jan Schluter, Nicolas Ballas
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
// Adaptation for 3d
CudaNdarray* corr3dMM(CudaNdarray *const bottom,
                      CudaNdarray *const weight,
                      CudaNdarray *const top,
                      const int direction,
                      const int dH = 1,
                      const int dW = 1,
                      const int dD = 1,
                      const int padH = 0,
                      const int padW = 0,
                      const int padD = 0)
{
    if (bottom->nd != 5)
    {
      PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires bottom of 5D");
      return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(bottom))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuCorr3dMM requires bottom to be C-contiguous, "
                   "but strides are: %d %d %d %d %d\n",
                   CudaNdarray_HOST_STRIDES(bottom)[0],
                   CudaNdarray_HOST_STRIDES(bottom)[1],
                   CudaNdarray_HOST_STRIDES(bottom)[2],
                   CudaNdarray_HOST_STRIDES(bottom)[3],
                   CudaNdarray_HOST_STRIDES(bottom)[4]);
      return 0;
    }
    if (weight->nd != 5)
    {
      PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires weight of 5D");
      return 0;
    }
    if (!CudaNdarray_is_c_contiguous(weight))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuCorr3dMM requires weight to be C-contiguous, "
                   "but strides are: %d %d %d %d %d\n",
                   CudaNdarray_HOST_STRIDES(weight)[0],
                   CudaNdarray_HOST_STRIDES(weight)[1],
                   CudaNdarray_HOST_STRIDES(weight)[2],
                   CudaNdarray_HOST_STRIDES(weight)[3],
                   CudaNdarray_HOST_STRIDES(weight)[4]);
      return 0;
    }

    if (top->nd != 5)
    {
      PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires top of 5D");
      return 0;
    }
    if (!CudaNdarray_is_c_contiguous(top))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuCorr3dMM requires top to be C-contiguous, "
                   "but strides are: %d %d %d %d %d\n",
                   CudaNdarray_HOST_STRIDES(top)[0],
                   CudaNdarray_HOST_STRIDES(top)[1],
                   CudaNdarray_HOST_STRIDES(top)[2],
                   CudaNdarray_HOST_STRIDES(top)[3],
                   CudaNdarray_HOST_STRIDES(top)[4]);
      return 0;
    }


    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth)
    const int batchSize = CudaNdarray_HOST_DIMS(bottom)[0];
    const int nChannels = CudaNdarray_HOST_DIMS(bottom)[1];
    const int bottomHeight = CudaNdarray_HOST_DIMS(bottom)[2];
    const int bottomWidth = CudaNdarray_HOST_DIMS(bottom)[3];
    const int bottomDepth = CudaNdarray_HOST_DIMS(bottom)[4];
    // weights: (nFilters, nChannels, rows, columns, depth)
    const int nFilters = CudaNdarray_HOST_DIMS(weight)[0];
    const int kH = CudaNdarray_HOST_DIMS(weight)[2];
    const int kW = CudaNdarray_HOST_DIMS(weight)[3];
    const int kD = CudaNdarray_HOST_DIMS(weight)[4];
    if (nChannels != CudaNdarray_HOST_DIMS(weight)[1])
    {
      PyErr_SetString(PyExc_ValueError,
                      "GpuCorr3dMM images and kernel must have the same stack size\n");
      return 0;
    }
    // top: (batchSize, nFilters, topHeight, topWidth, topDepth)
    const int topHeight = int((bottomHeight + 2*padH - kH) / dH) + 1;
    const int topWidth  = int((bottomWidth + 2*padW - kW) / dW) + 1;
    const int topDepth  = int((bottomDepth + 2*padD - kD) / dD) + 1;
    if (batchSize != CudaNdarray_HOST_DIMS(top)[0] ||
        nFilters != CudaNdarray_HOST_DIMS(top)[1] ||
        topHeight != CudaNdarray_HOST_DIMS(top)[2] ||
        topWidth != CudaNdarray_HOST_DIMS(top)[3] ||
        topDepth != CudaNdarray_HOST_DIMS(top)[4])
   {
     PyErr_Format(PyExc_ValueError,
                  "GpuCorr3dMM shape inconsistency:\n"
                  "  bottom shape: %d %d %d %d %d\n"
                  "  weight shape: %d %d %d %d %d\n"
                  "  top shape: %d %d %d %d %d (expected %d %d %d %d %d)\n",
                  batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth,
                  nFilters, nChannels, kH, kW, kD,
                  CudaNdarray_HOST_DIMS(top)[0], CudaNdarray_HOST_DIMS(top)[1],
                  CudaNdarray_HOST_DIMS(top)[2], CudaNdarray_HOST_DIMS(top)[3],
                  CudaNdarray_HOST_DIMS(top)[4],
                  batchSize, nFilters, topHeight, topWidth, topDepth);
        return 0;
    }

    // Create temporary columns
    int col_dim[2];
    col_dim[0] = nChannels * kW * kH * kD;
    col_dim[1] = topHeight * topWidth * topDepth;
    CudaNdarray* col = (CudaNdarray*) CudaNdarray_NewDims(2, col_dim);
    if (0 == col)
    {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuCorr3dMM failed to allocate working memory of %d x %d\n",
                   col_dim[0], col_dim[1]);
        return 0;
    }

    // Define some useful variables
    const int bottom_stride = CudaNdarray_HOST_STRIDES(bottom)[0];
    const int top_stride = CudaNdarray_HOST_STRIDES(top)[0];
    const int K_ = col_dim[0];
    const int N_ = col_dim[1];
    const int M_ = nFilters;
    const float one = 1.0f;
    const float zero = 0.0f;

    CudaNdarray *output;
    if (direction == 0)
    { // forward pass
      output = top;
      // valid correlation: im2col, then gemm
      // Iterate over batch
      for (int n = 0; n < batchSize; n++)
      {
        // First, im3d2col
        im3d2col(bottom->devdata + n * bottom_stride,
                 nChannels,
                 bottomHeight, bottomWidth, bottomDepth,
                 kH, kW, kD,
                 padH, padW, padD,
                 dH, dW, dD,
                 col->devdata);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuCorr3dMM encountered a CUDA error in im2col: %s\n"
                       "This could be a known bug in CUDA, please see the "
                       "GpuCorr3dMM() documentation.\n",
                       cudaGetErrorString(err));
          Py_DECREF(col);
          return 0;
        }
        // Second, gemm
        cublasStatus_t status = cublasSgemm(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            N_, M_, K_,
                                            &one,
                                            col->devdata, N_,
                                            weight->devdata, K_,
                                            &zero,
                                            top->devdata + n * top_stride, N_);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuCorr3dMM encountered a CUBLAS error: %s\n"
                       "This could be a known bug in CUDA, please see the "
                       "GpuCorr3dMM() documentation.\n",
                       cublasGetErrorString(status));
          Py_DECREF(col);
          return 0;
        }
      }
    }
    else if (direction == 1)
    {
      // backprop wrt. weights
      output = weight;
      // valid convolution: im2col, then gemm
      // Iterate over batch
      for (int n = 0; n < batchSize; n++)
      {
        // First, im2col
        im3d2col(bottom->devdata + n * bottom_stride, nChannels,
                 bottomHeight, bottomWidth, bottomDepth,
                 kH, kW, kD,
                 padH, padW, padD,
                 dH, dW, dD,
                 col->devdata);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
       {
         PyErr_Format(PyExc_RuntimeError,
                      "GpuCorr3dMM encountered a CUDA error in im2col: %s\n"
                      "This could be a known bug in CUDA, please see the "
                      "GpuCorr3dMM() documentation.\n",
                      cudaGetErrorString(err));
         Py_DECREF(col);
         return 0;
       }
        // Second, gemm
        // Note that we accumulate into weight. We do so by setting beta = 0
        // for the first iteration and beta = 1 for subsequent ones. (This
        // is faster than setting weight to all zeros before the loop.)
        cublasStatus_t status = cublasSgemm(handle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            K_, M_, N_,
                                            &one,
                                            col->devdata, N_,
                                            top->devdata + n * top_stride, N_,
                                            (n == 0) ? &zero : &one,
                                            weight->devdata, K_);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuCorr3dMM encountered a CUBLAS error: %s\n"
                       "This could be a known bug in CUDA, please see the "
                       "GpuCorr3dMM() documentation.\n",
                       cublasGetErrorString(status));
          Py_DECREF(col);
          return 0;
        }
      }
    }
    else if (direction == 2)
    {
      // backprop wrt. inputs
      output = bottom;
      // full convolution: gemm, then col2im3d
      // Iterate over batch
      for (int n = 0; n < batchSize; n++)
      {
        // gemm into columns
        cublasStatus_t status = cublasSgemm(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            N_, K_, M_,
                                            &one,
                                            top->devdata + n * top_stride, N_,
                                            weight->devdata, K_,
                                            &zero,
                                            col->devdata, N_);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuCorr3dMM encountered a CUBLAS error: %s\n"
                       "This could be a known bug in CUDA, please see the "
                       "GpuCorr3dMM() documentation.\n",
                       cublasGetErrorString(status));
          Py_DECREF(col);
          return 0;
        }
        // col2im3d back to the data
        col2im3d(col->devdata, nChannels,
                 bottomHeight, bottomWidth, bottomDepth,
                 kH, kW, kD,
                 padH, padW, padD,
                 dH, dW, dD, bottom->devdata + n * bottom_stride);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuCorr3dMM encountered a CUDA error in col2im: %s\n"
                       "This could be a known bug in CUDA, please see the "
                       "GpuCorr3dMM() documentation.\n",
                       cudaGetErrorString(err));
          Py_DECREF(col);
          return 0;
        }
      }
    }
    // Free temporary columns
    Py_DECREF(col);

    // Note that we don't change the refcount of the output matrix here. Output
    // allocation and refcounting is done in BaseGpuCorr3dMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}
