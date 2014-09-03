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
                                const int height, const int width,
                                const int depth,
                                const int kernel_h, const int kernel_w,
                                const int kernel_d,
                                const int pad_h, const int pad_w,
                                const int pad_d,
                                const int stride_h, const int stride_w,
                                const int stride_d,
                                const int height_col, const int width_col,
                                const int depth_col,
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


// Author: Arjun Jain
// Modified by: Nicolas Ballas
CudaNdarray* corr3dMM(const CudaNdarray *input,
                    CudaNdarray *weight,
                    CudaNdarray *output,
                    int dH = 1,
                    int dW = 1,
                    int dD = 1,
                    int padH = 0,
                    int padW = 0,
                    int padD = 0)
{

  cublasStatus_t status;

  if (input->nd != 5)
    PyErr_SetString(PyExc_ValueError, "required input of 5D");

  if (weight->nd != 5)
    PyErr_SetString(PyExc_ValueError, "required weight of 5D");

  if (CudaNdarray_HOST_DIMS(input)[1]  != CudaNdarray_HOST_DIMS(weight)[1])
  {
    PyErr_SetString(PyExc_ValueError,
                    "GpuCorr3dMM images and kernel must have the same stack size\n");
    return 0;
  }

  if (!CudaNdarray_is_c_contiguous(input))
  {
    PyErr_Format(PyExc_ValueError,
                 "GpuCorrMM requires bottom to be C-contiguous, "
                 "but strides are: %d %d %d %d %d\n",
                 CudaNdarray_HOST_STRIDES(input)[0],
                 CudaNdarray_HOST_STRIDES(input)[1],
                 CudaNdarray_HOST_STRIDES(input)[2],
                 CudaNdarray_HOST_STRIDES(input)[3],
                 CudaNdarray_HOST_STRIDES(input)[4]);
    return 0;
  }
  if (!CudaNdarray_is_c_contiguous(weight))
  {
    PyErr_Format(PyExc_ValueError,
                 "GpuCorrMM requires weight to be C-contiguous, "
                 "but strides are: %d %d %d %d %d\n",
                 CudaNdarray_HOST_STRIDES(weight)[0],
                 CudaNdarray_HOST_STRIDES(weight)[1],
                 CudaNdarray_HOST_STRIDES(weight)[2],
                 CudaNdarray_HOST_STRIDES(weight)[3],
                 CudaNdarray_HOST_STRIDES(weight)[4]);
    return 0;
  }


  // filters: (number of filters, nInputPlane, rows, columns, depth)
  int nOutputPlane = CudaNdarray_HOST_DIMS(weight)[0];
  int kH = CudaNdarray_HOST_DIMS(weight)[2];
  int kW = CudaNdarray_HOST_DIMS(weight)[3];
  int kD = CudaNdarray_HOST_DIMS(weight)[4];


  // input: (batch, nInputPlane, rows, columns, depth)
  long batchSize = CudaNdarray_HOST_DIMS(input)[0];
  int nInputPlane = CudaNdarray_HOST_DIMS(input)[1];
  long inputHeight  = CudaNdarray_HOST_DIMS(input)[2];
  long inputWidth   = CudaNdarray_HOST_DIMS(input)[3];
  long inputDepth   = CudaNdarray_HOST_DIMS(input)[4];

  long outputDepth = (inputDepth + 2 * padD - kD) / dD + 1;
  long outputWidth  = (inputWidth + 2 * padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - kH) / dH + 1;

  // check output, size (batchSize, nOutputPlane, outputHeight, outputWidth);
  if (batchSize != CudaNdarray_HOST_DIMS(output)[0] ||
      nOutputPlane != CudaNdarray_HOST_DIMS(output)[1] ||
      outputHeight != CudaNdarray_HOST_DIMS(output)[2] ||
      outputWidth != CudaNdarray_HOST_DIMS(output)[3] ||
      outputDepth != CudaNdarray_HOST_DIMS(output)[4])
  {
    PyErr_Format(PyExc_ValueError,
                 "GpuCorrMM outputs parameter don't have the good shape %d %d %d %d %d, %d %d %d %d %d\n",
                 batchSize, nOutputPlane, outputHeight, outputWidth, outputDepth,
                 CudaNdarray_HOST_DIMS(output)[0], CudaNdarray_HOST_DIMS(output)[1],
                 CudaNdarray_HOST_DIMS(output)[2], CudaNdarray_HOST_DIMS(output)[3],
                 CudaNdarray_HOST_DIMS(output)[4]);
    return 0;
  }
  // Create temporary columns
  int col_dim[2];
  col_dim[0] = nInputPlane * kW * kH * kD;
  col_dim[1]= outputHeight * outputWidth * outputDepth;
  CudaNdarray* columns = (CudaNdarray*) CudaNdarray_NewDims(2, col_dim);

  int ip_stride = CudaNdarray_HOST_STRIDES(input)[0];
  int op_stride = CudaNdarray_HOST_STRIDES(output)[0];

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++)
  {
    // Matrix mulitply per output:
    // 1. Extract columns:
    im3d2col(input->devdata + elt * ip_stride,
             nInputPlane,
             inputHeight, inputWidth, inputDepth,
             kH, kW, kD,
             padH, padW, padD,
             dH, dW, dD,
             columns->devdata);


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    float alpha = 1.0f; float beta = 0.0f;
    int m = CudaNdarray_HOST_DIMS(columns)[1];
    int n = CudaNdarray_HOST_DIMS(weight)[0];
    int k = CudaNdarray_HOST_DIMS(columns)[0];

    status = cublasSgemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, k,
                         &alpha,
                         columns->devdata, m,
                         weight->devdata, k,
                         &beta,
                         output->devdata + elt * op_stride, m);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "!!!! CUBLAS error: ";
      std::cerr << cublasGetErrorString(status) << "\n";
    }
  }

  Py_DECREF(columns);
  return output;
}
