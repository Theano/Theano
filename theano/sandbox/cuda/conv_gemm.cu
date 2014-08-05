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
#include <Python.h>
#include "cuda_ndarray.cuh"
#include "caffe_common.hpp"
// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// Reference code: https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
__global__ void im2col_kernel(const int n, const float* data_im,
                              const int height, const int width, const int ksize, const int pad,
                              const int stride, const int height_col, const int width_col,
                              float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                                             data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

void im2col(const float* data_im, const int channels,
            const int height, const int width, const int ksize, const int pad,
            const int stride, float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
    
  // Launch
  im2col_kernel <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>> (
     num_kernels, data_im, height, width, ksize, 
     pad, stride, 
     height_col, width_col, data_col
     );
}



// Author: Arjun Jain
CudaNdarray* corrMM(const CudaNdarray *input, 
				      CudaNdarray *weight,
				      CudaNdarray *output,
				      int padding = 0) 
{

  	cublasStatus_t status;

    if (input->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required input of 4D");
    }
    
    if (weight->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required weight of 4D");
    }
        
     // TODO: stride(dW, dH) and padding as function parameter
     int dH = 1; 
     int dW = 1;
     int kH = CudaNdarray_HOST_DIMS(weight)[2];
     int kW = CudaNdarray_HOST_DIMS(weight)[3];
     int nInputPlane = CudaNdarray_HOST_DIMS(input)[1]; 
     // filters: (number of filters, nInputPlane, rows, columns)
     int nOutputPlane = CudaNdarray_HOST_DIMS(weight)[0];
     long batchSize = CudaNdarray_HOST_DIMS(input)[0];
     if (CudaNdarray_HOST_DIMS(input)[2] != CudaNdarray_HOST_DIMS(input)[3]){
       PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM support only square images. Got %dx%d images\n",
		    CudaNdarray_HOST_DIMS(input)[2],
		    CudaNdarray_HOST_DIMS(input)[3]
		    );
       return NULL;
     }
     if (kW != kH){
       PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM support only square kernel. Got %dx%d kernel\n",
		    kW, kH
		    );
       return NULL;
     }
     if (CudaNdarray_HOST_DIMS(input)[1]  != CudaNdarray_HOST_DIMS(weight)[1]){
       PyErr_SetString(PyExc_ValueError,
                    "GpuCorrMM images and kernel must have the same stack size\n"
		    );
       return NULL;
     }
     long inputHeight  = CudaNdarray_HOST_DIMS(input)[2];
     long inputWidth   = CudaNdarray_HOST_DIMS(input)[3];
     long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
     long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;
     // check output, size (batchSize, nOutputPlane,
     //		outputHeight, outputWidth);
     
     if (batchSize != CudaNdarray_HOST_DIMS(output)[0] ||
	 nOutputPlane != CudaNdarray_HOST_DIMS(output)[1] ||
	 outputHeight != CudaNdarray_HOST_DIMS(output)[2] ||
	 outputWidth != CudaNdarray_HOST_DIMS(output)[3]){
       PyErr_SetString(PyExc_ValueError,
                    "GpuCorrMM outputs parameter don't have the good shape\n"
		    );
       return NULL;
     }
     // Create temporary columns
     int col_dim[2];
     col_dim[0] = nInputPlane*kW*kH;
     col_dim[1]= outputHeight*outputWidth;
     CudaNdarray* columns = (CudaNdarray*)CudaNdarray_NewDims(2,col_dim);


     int ip_stride = CudaNdarray_HOST_STRIDES(input)[0];

     int op_stride = CudaNdarray_HOST_STRIDES(output)[0];

     // For each elt in batch, do:
     for (int elt = 0; elt < batchSize; elt ++) {
     // Matrix mulitply per output:
       
        // 1. Extract columns:
        im2col(
           input->devdata + elt*ip_stride,
           nInputPlane, inputWidth, inputHeight, kW, padding, dW, 
           columns->devdata
         );
      
         
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
                output->devdata + elt * op_stride, m
                );

  	     if (status != CUBLAS_STATUS_SUCCESS) {
	       std::cerr << "!!!! CUBLAS error: ";
	       std::cerr << cublasGetErrorString(status) << "\n";
	      }

      }

    Py_DECREF(columns);
  return output;
}
               
               
				      
    
