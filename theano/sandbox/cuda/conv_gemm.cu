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


// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu)
// Kernels for fast unfold + copy
__global__ void im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  im2col_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
}

__global__ void col2im_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize
            + (w - w_col * stride_w);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset =
        (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
}


// Theano op code
// Authors: Arjun Jain, Frédéric Bastien, Jan Schlüter
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
CudaNdarray* corrMM(const CudaNdarray *input, 
                    CudaNdarray *weight,
                    CudaNdarray *output,
                    int mode,
                    int dH = 1,
                    int dW = 1,
                    int padH = 0,
                    int padW = 0)
{
    if (input->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires input of 4D");
    }
    
    if (weight->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires weight of 4D");
    }

    if (output->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires output of 4D");
    }

    // Extract some shape information for later and check shape consistency
    // inputs: (batchSize, nInputPlane, inputHeight, inputWidth)
    const int batchSize = CudaNdarray_HOST_DIMS(input)[0];
    const int nInputPlane = CudaNdarray_HOST_DIMS(input)[1];
    const int inputHeight = CudaNdarray_HOST_DIMS(input)[2];
    const int inputWidth = CudaNdarray_HOST_DIMS(input)[3];
    // filters: (nOutputPlane, nInputPlane, rows, columns)
    const int nOutputPlane = CudaNdarray_HOST_DIMS(weight)[0];
    const int kH = CudaNdarray_HOST_DIMS(weight)[2];
    const int kW = CudaNdarray_HOST_DIMS(weight)[3];
    if (nInputPlane != CudaNdarray_HOST_DIMS(weight)[1]) {
        PyErr_SetString(PyExc_ValueError,
                "GpuCorrMM images and kernel must have the same stack size\n");
        return NULL;
    }
    // outputs: (batchSize, nOutputPlane, outputHeight, outputWidth)
    int outputHeight, outputWidth;
    if (mode == 1) {  // valid correlation with padding and subsampling
        outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
        outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
    }
    else if (mode == 0) {  // full convolution with upsampling and cropping
        // these would be the shapes for a standard full convolution:
        //outputHeight = (inputHeight + 2*padH + kH - 2) / dH + 1;
        //outputWidth  = (inputWidth + 2*padW + kW - 2) / dW + 1;
        // but here, dH and dW are *upsampling* factors, and padding is reversed
        // (because the implementation was meant as a backward pass for a CNN)
        outputHeight = (inputHeight - 1) * dH + kH - 2*padH;
        outputWidth = (inputWidth - 1) * dW + kW - 2*padW;
    }
    if (batchSize != CudaNdarray_HOST_DIMS(output)[0] ||
            nOutputPlane != CudaNdarray_HOST_DIMS(output)[1] ||
            outputHeight != CudaNdarray_HOST_DIMS(output)[2] ||
            outputWidth != CudaNdarray_HOST_DIMS(output)[3]) {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM output parameter has wrong shape %d %d %d %d, expected %d %d %d %d\n",
                CudaNdarray_HOST_DIMS(output)[0], CudaNdarray_HOST_DIMS(output)[1],
                CudaNdarray_HOST_DIMS(output)[2], CudaNdarray_HOST_DIMS(output)[3],
                batchSize, nOutputPlane, outputHeight, outputWidth);
        return NULL;
    }

    if (mode == 1) {  // valid correlation: im2col, then gemm
        // Create temporary columns (col_data)
        int col_dim[2];
        col_dim[0] = nInputPlane * kW * kH;
        col_dim[1] = outputHeight * outputWidth;
        CudaNdarray* col_data = (CudaNdarray*)CudaNdarray_NewDims(2, col_dim);

        // Define some useful variables
        const int ip_stride = CudaNdarray_HOST_STRIDES(input)[0];
        const int op_stride = CudaNdarray_HOST_STRIDES(output)[0];
        const int K_ = col_dim[0];
        const int N_ = col_dim[1];
        const int M_ = nOutputPlane;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // First, im2col
            im2col(input->devdata + n * ip_stride, nInputPlane, inputHeight,
                    inputWidth, kH, kW, padH, padW, dH, dW, col_data->devdata);
            // Second, gemm
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N_, M_, K_,
                    &alpha,
                    col_data->devdata, N_,
                    weight->devdata, K_,
                    &beta,
                    output->devdata + n * op_stride, N_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n",
                        cublasGetErrorString(status));
                return NULL;
            }
        }
        // Free temporary columns
        Py_DECREF(col_data);

        /*
        // Original caffe code for comparison
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is for grouped convolution; we can ignore groups
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = (*top)[i]->mutable_gpu_data();
        Dtype* col_data = col_buffer_.mutable_gpu_data();
        const Dtype* weight = this->blobs_[0]->gpu_data();
        int weight_offset = M_ * K_;
        int col_offset = K_ * N_;
        int top_offset = M_ * N_;
        for (int n = 0; n < num_; ++n) {
          // First, im2col
          im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
              width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              col_data);
          // Second, innerproduct with groups
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
              (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
            == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
            cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N,
              N_, M_, K_,
              1.,
              col_data + col_offset * g, N_,
              weight + weight_offset * g, K_,
              0.,
              top_data + (*top)[i]->offset(n) + top_offset * g, N_);
          }
        }
        */
    }
    else if (mode == 0) {  // full convolution: gemm, then col2im
        // Create temporary columns (col_diff)
        int col_dim[2];
        col_dim[0] = nOutputPlane * kW * kH;
        col_dim[1] = inputHeight * inputWidth;
        CudaNdarray* col_diff = (CudaNdarray*)CudaNdarray_NewDims(2, col_dim);

        // Define some useful variables
        const int ip_stride = CudaNdarray_HOST_STRIDES(input)[0];
        const int op_stride = CudaNdarray_HOST_STRIDES(output)[0];
        const int K_ = col_dim[0];
        const int N_ = col_dim[1];
        const int M_ = nInputPlane;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // gemm into columns
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    N_, K_, M_,
                    &alpha,
                    input->devdata + n * ip_stride, N_,
                    weight->devdata, K_,
                    &beta,
                    col_diff->devdata, N_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n",
                        cublasGetErrorString(status));
                return NULL;
            }
            // col2im back to the data
            col2im(col_diff->devdata, nOutputPlane, outputHeight, outputWidth,
                    kH, kW, padH, padW, dH, dW, output->devdata + n * op_stride);
        }
        // Free temporary columns
        Py_DECREF(col_diff);

        /*
        // Original caffe code for comparison
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is the backward pass of a valid convolution, so
        // top_diff is the input, bottom_diff is the output, weights are weights
        Dtype* col_data = col_buffer_.mutable_gpu_data();
        Dtype* col_diff = col_buffer_.mutable_gpu_diff();
        Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
        for (int n = 0; n < num_; ++n) {
            // gradient w.r.t. bottom data, if necessary
            for (int g = 0; g < group_; ++g) {
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                  (Dtype)1., weight + weight_offset * g,
                  top_diff + top[i]->offset(n) + top_offset * g,
                  (Dtype)0., col_diff + col_offset * g);
              == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
              cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_T, N_, K_, M_,
                  1.,
                  top_diff + top[i]->offset(n) + top_offset * g, N_,
                  weight + weight_offset * g, K_,
                  0.,
                  col_diff + col_offset * g, N_);
            }
            // col2im back to the data
            col2im_gpu(col_diff, channels_, height_, width_,
                kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                bottom_diff + (*bottom)[i]->offset(n));
        }
        */
    }
    return output;
}

