// Copyright 2014 BVLC and contributors.
#undef _GLIBCXX_ATOMIC_BUILTINS
#include <Python.h>
#include "cuda_ndarray.cuh"
#include "caffe_common.hpp"
// Author: Arjun Jain
// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
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



CudaNdarray* validMM(const CudaNdarray *input, 
				      CudaNdarray *weight,
				      CudaNdarray *output) 
{

    // TODO: This needs to be done in the singleton!
    // Initialize CUBLAS
    cublasHandle_t handle;
  	cublasStatus_t status = cublasCreate(&handle);
  	if (status != CUBLAS_STATUS_SUCCESS) {
      		std::cerr << "!!!! CUBLAS initialization error\n";
	}

    if (input->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required input of 4D");
    }
    
    if (weight->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required weight of 4D");
    }
        
     // Reference code: https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
     // TODO: stride(dW, dH) and padding as function parameter
     int dH = 1; 
     int dW = 1;
     int padding = 0; 
     int kH = CudaNdarray_HOST_DIMS(weight)[2];
     int kW = CudaNdarray_HOST_DIMS(weight)[3];
     int nInputPlane = CudaNdarray_HOST_DIMS(input)[1]; 
     // filters: (number of filters, nInputPlane, rows, columns)
     int nOutputPlane = CudaNdarray_HOST_DIMS(weight)[0];
     long batchSize = CudaNdarray_HOST_DIMS(input)[0];
     assert(kW == kH); //filters must be square (kW == kH)
     assert(dW == dH); //stride must be square (dW == dH)
     long inputHeight  = CudaNdarray_HOST_DIMS(input)[2];
     long inputWidth   = CudaNdarray_HOST_DIMS(input)[3];
     long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
     long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;
     // Allocate output, size (batchSize, nOutputPlane, 
     //		outputHeight, outputWidth);
     int out_dim[4];
     out_dim[0] = batchSize;
     out_dim[1] = nOutputPlane;
     out_dim[2] = outputHeight;
     out_dim[3] = outputWidth;
     
     output = (CudaNdarray*)CudaNdarray_NewDims(4,out_dim);
     // Create temporary columns
     int col_dim[2];
     col_dim[0] = nInputPlane*kW*kH;
     col_dim[1]= outputHeight*outputWidth;
     CudaNdarray* columns = (CudaNdarray*)CudaNdarray_NewDims(2,col_dim);


      int ip_stride = CudaNdarray_HOST_DIMS(input)[1] *
        CudaNdarray_HOST_DIMS(input)[2] * 
        CudaNdarray_HOST_DIMS(input)[3];
        
     int op_stride = CudaNdarray_HOST_DIMS(output)[1] *
        CudaNdarray_HOST_DIMS(output)[2] * 
        CudaNdarray_HOST_DIMS(output)[3];

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
         int n = CudaNdarray_HOST_DIMS(weight)[1];
         int k = CudaNdarray_HOST_DIMS(columns)[0];
         
         //Caffe::getRef().getCublasHandle().get()
         status = cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                columns->devdata, m,
                weight->devdata, k,
                &beta,
                output->devdata + elt * op_stride, m
                );

  
		  cudaError_t err = cudaGetLastError();
		  if (err != cudaSuccess) {
		    printf("error in validMM: %s\n", cudaGetErrorString(err));
		  }

      }
    
    // TODO: How is columns and output deallocated? 
    // device_free(columns->devdata);
    // TODO: I did not kill the cublas context. If it comes from 
    // the singleton, we dont need to kill it.

 
  return output;
}
               
               
				      
    