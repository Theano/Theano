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

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp)
// Loops for fast unfold + copy
void im2col(const %(float_type)s* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    %(float_type)s* data_col) {
  // Implicit dilated kernel size
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c %% kernel_w;
    int h_offset = (c / kernel_w) %% kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(npy_intp)(c * height_col + h) * width_col + w] =
            data_im[(npy_intp)(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(npy_intp)(c * height_col + h) * width_col + w] = 0.;
      }
    }
  }
}

// Unlike the Caffe and Theano GPU verions, the data_im array is set to zero
// before the col2im call rather than doing it here. So, the result is just
// accumulated into data_im.
void col2im(const %(float_type)s* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, %(float_type)s* data_im) {
  // Implicit dilated patch
  int dil_patch_h = (patch_h - 1) * dilation_h + 1;
  int dil_patch_w = (patch_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c %% patch_w;
    int h_offset = (c / patch_w) %% patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(npy_intp)(c_im * height + h_pad) * width + w_pad] +=
            data_col[(npy_intp)(c * height_col + h) * width_col + w];

      }
    }
  }
}


// Theano op code
// GPU version authors: Arjun Jain, Frederic Bastien, Jan Schlueter
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
// CPU version author: Jesse Livezey
// CPU version adapted from GPU version
PyArrayObject* corrMM(PyArrayObject* bottom,
                      PyArrayObject* weight,
                      PyArrayObject* top,
                      const int direction,
                      const int dH = 1,
                      const int dW = 1,
                      const int dilH = 1,
                      const int dilW = 1,
                      const int padH = 0,
                      const int padW = 0)
{
    if (PyArray_NDIM(bottom) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM requires bottom of 4D");
        return NULL;
    }
    if (PyArray_TYPE(bottom) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM received bottom with wrong type.");
        return NULL;
    }

    if (PyArray_NDIM(weight) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM requires weight of 4D");
        return NULL;
    }
    if (PyArray_TYPE(weight) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM received weight with wrong type.");
        return NULL;
    }

    if (PyArray_NDIM(top) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM requires top of 4D");
        return NULL;
    }
    if (PyArray_TYPE(top) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "CorrMM received top with wrong type.");
        return NULL;
    }
    // Ensure data is contiguous
    bottom = PyArray_GETCONTIGUOUS(bottom);
    weight = PyArray_GETCONTIGUOUS(weight);
    top = PyArray_GETCONTIGUOUS(top);

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth)
    const int batchSize = PyArray_DIMS(bottom)[0];
    const int nChannels = PyArray_DIMS(bottom)[1];
    const int bottomHeight = PyArray_DIMS(bottom)[2];
    const int bottomWidth = PyArray_DIMS(bottom)[3];
    // weights: (nFilters, nChannels, rows, columns)
    const int nFilters = PyArray_DIMS(weight)[0];
    const int kH = PyArray_DIMS(weight)[2];
    const int kW = PyArray_DIMS(weight)[3];
    if (nChannels != PyArray_DIMS(weight)[1]) {
        PyErr_SetString(PyExc_ValueError,
                "CorrMM images and kernel must have the same stack size\n");
        return NULL;
    }
    // implicit dilated filter
    const int dil_kH = (kH - 1) * dilH + 1;
    const int dil_kW = (kW - 1) * dilW + 1;
    // top: (batchSize, nFilters, topHeight, topWidth)
    const int topHeightNoDH = (bottomHeight + 2*padH - dil_kH);
    const int topWidthNoDW  = (bottomWidth + 2*padW - dil_kW);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) %% y) == 0 ? 0 : 1)) : (x / y))
    const int topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const int topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
#undef _CONV_FLOORDIV
    if (batchSize != PyArray_DIMS(top)[0] ||
            nFilters != PyArray_DIMS(top)[1] ||
            topHeight != PyArray_DIMS(top)[2] ||
            topWidth != PyArray_DIMS(top)[3]) {
        PyErr_Format(PyExc_ValueError,
                "CorrMM shape inconsistency:\n"
                "  bottom shape: %%d %%d %%d %%d\n"
                "  weight shape: %%d %%d %%d %%d\n"
                "  top shape: %%ld %%ld %%ld %%ld (expected %%d %%d %%d %%d)\n",
                batchSize, nChannels, bottomHeight, bottomWidth,
                nFilters, nChannels, kH, kW,
                PyArray_DIMS(top)[0], PyArray_DIMS(top)[1],
                PyArray_DIMS(top)[2], PyArray_DIMS(top)[3],
                batchSize, nFilters, topHeight, topWidth);
        return NULL;
    }

    // Create temporary columns
    int max_threads = %(omp_get_max_threads)s;
    if (batchSize < max_threads) {
        max_threads = batchSize;
    }
    npy_intp col_dim[3];
    col_dim[0] = (npy_intp)max_threads;
    col_dim[1] = (npy_intp)(nChannels * kW * kH);
    col_dim[2] = (npy_intp)(topHeight * topWidth);

    //Change to PyArray_ZEROS which is faster than PyArray_EMPTY.
    PyArrayObject* col = (PyArrayObject*)PyArray_ZEROS(3,
            col_dim,
            PyArray_TYPE(top),
            0); 
    if (NULL == col) {
        PyErr_Format(PyExc_RuntimeError,
                "CorrMM failed to allocate working memory of"
                " %%ld x %%ld x %%ld\n",
                col_dim[0], col_dim[1], col_dim[2]);
        return NULL;
    }

    // Define some useful variables
    const int bottom_stride = PyArray_STRIDES(bottom)[0]/%(n_bytes)f;
    const int top_stride = PyArray_STRIDES(top)[0]/%(n_bytes)f;
    const int K_ = col_dim[1];
    const int N_ = col_dim[2];
    const int col_stride = (K_ * N_);
    const int M_ = nFilters;
    const %(c_float_type)s one = 1.0;
    const %(c_float_type)s zero = 0.0;
    char NTrans = 'N';
    char Trans = 'T';
    PyArrayObject *output;

    if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
        switch(direction) {
        case 0:
            output = top;
            break;
        case 1:
            output = weight;
            break;
        case 2:
            output = bottom;
            break;
        default:
            return NULL;
        }
        PyArray_FILLWBYTE(output, 0);
    }
    else if (direction == 0) {  // forward pass
        output = top;
        // valid correlation: im2col, then gemm
        // Iterate over batch
        int blas_threads_saved = %(blas_get_num_threads)s;
        // Always forcing gemm to one thread when OpenMP is enalbed for best and stable performance.
        %(blas_set_num_threads)s(1);
        %(omp_flags)s
        for (int n = 0; n < batchSize; ++n) {
            int tid = %(omp_get_thread_num)s;
            // First, im2col
            im2col((%(float_type)s*)PyArray_DATA(bottom) + n * bottom_stride, nChannels, bottomHeight,
                   bottomWidth, kH, kW, dilH, dilW, padH, padW, dH, dW,
                   (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride);
            // Second, gemm
            %(gemm)s(&NTrans, &NTrans,
                   &N_, &M_, &K_,
                   &one,
                   (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride, &N_,
                   (%(float_type)s*)PyArray_DATA(weight), &K_,
                   &zero,
                   (%(float_type)s*)PyArray_DATA(top) + n * top_stride, &N_);
        }
        // Restore to previous blas threads
        %(blas_set_num_threads)s(blas_threads_saved);

        /*
        // Original caffe code for comparison
        // Note that this code was translated from the Theano GPU code,
        // not the Caffe CPU code.
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is for grouped convolution; we can ignore groups here,
        // but the group-related offsets help explain what M_, N_ and K_ are
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
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        npy_intp weight_dim[2];
        weight_dim[0] = (npy_intp)max_threads;
        weight_dim[1] = (npy_intp)(M_ * K_);
        PyArrayObject* local_weight = (PyArrayObject*)PyArray_ZEROS(2,
                                   weight_dim, PyArray_TYPE(weight), 0);

        if (NULL == local_weight)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "CorrMM failed to allocate weight memory of %%ld x %%ld\n",
                    weight_dim[0], weight_dim[1]);
            return NULL;
        }
        
        // valid convolution: im2col, then gemm
        // Iterate over batch
        int blas_threads_saved = %(blas_get_num_threads)s;
        // Always forcing gemm to one thread when OpenMP is enalbed for best and stable performance.
        %(blas_set_num_threads)s(1);
        // OMP for batch-level paralization
        %(omp_flags)s
        for (int n = 0; n < batchSize; ++n) {
            int tid = %(omp_get_thread_num)s;
            // First, im2col
            im2col((%(float_type)s*)PyArray_DATA(bottom) + n * bottom_stride, nChannels, bottomHeight,
                   bottomWidth, kH, kW, dilH, dilW, padH, padW, dH, dW,
                   (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride);
            // Second, gemm
            // Note that we accumulate into weight. We do so by setting beta = 0
            // for the first iteration and beta = 1 for subsequent ones. (This
            // is faster than setting weight to all zeros before the loop.)
            %(gemm)s(&Trans, &NTrans,
                   &K_, &M_, &N_,
                   &one,
                   (%(float_type)s*)PyArray_DATA(col) + tid * col_stride, &N_,
                   (%(float_type)s*)PyArray_DATA(top) + n * top_stride, &N_,
                   (n == 0) ? &zero : &one,
                   (%(float_type)s*)PyArray_DATA(local_weight) + 
                   tid * weight_dim[1], &K_);
        }
        // Restore to previous blas threads
        %(blas_set_num_threads)s(blas_threads_saved);

        //aggregate weights
        memset((%(float_type)s*)PyArray_DATA(weight), 0, M_ * K_*sizeof(%(float_type)s));
        /*
         * Put index "j" into outer loop to get the
         * correct result when openmp is used.
         */
        %(omp_flags)s
        for(int j = 0; j < weight_dim[1]; ++j){
            for(int i = 0; i < max_threads; ++i){
                ((%(float_type)s*)PyArray_DATA(weight))[j] += 
                    *((%(float_type)s*)PyArray_DATA(local_weight) +
                    i * weight_dim[1] + j);
            }
        }
        Py_DECREF(local_weight);
        /*
        // Original caffe code for comparison
        // Note that this code was translated from the Theano GPU code,
        // not the Caffe CPU code.
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is for grouped convolution; we can ignore groups
        for (int n = 0; n < num_; ++n) {
          // Since we saved memory in the forward pass by not storing all col
          // data, we will need to recompute them.
          im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                     width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                     stride_h_, stride_w_, col_data);
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
            == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
            cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, K_, M_, N_,
                1.0,
                col_data + col_offset * g, N_,
                top_diff + top[i]->offset(n) + top_offset * g, N_,
                1.0,
                weight_diff + weight_offset * g, K_);
          }
        }
        */
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        // bottom is set to zero here rather than inside of col2im
        PyArray_FILLWBYTE(bottom, 0);
        // full convolution: gemm, then col2im
        // Iterate over batch

        int blas_threads_saved = %(blas_get_num_threads)s;
        // Always forcing gemm to one thread when OpenMP is enalbed for best and stable performance.
        %(blas_set_num_threads)s(1);
        %(omp_flags)s
        for (int n = 0; n < batchSize; ++n) {
            // gemm into columns
            int tid = %(omp_get_thread_num)s;
            %(gemm)s(&NTrans, &Trans,
                   &N_, &K_, &M_,
                   &one,
                   (%(float_type)s*)PyArray_DATA(top) + n * top_stride, &N_,
                   (%(float_type)s*)PyArray_DATA(weight), &K_,
                   &zero,
                   (%(float_type)s*)PyArray_DATA(col) + tid * col_stride, &N_);
            // col2im back to the data
            col2im((%(float_type)s*)PyArray_DATA(col) + tid * col_stride, nChannels, bottomHeight, bottomWidth,
                   kH, kW, dilH, dilW, padH, padW,
                   dH, dW, (%(float_type)s*)PyArray_DATA(bottom) + n * bottom_stride);
        }
        // Restore to previous blas threads
        %(blas_set_num_threads)s(blas_threads_saved);
        /*
        // Original caffe code for comparison
        // Note that this code was translated from the Theano GPU code,
        // not the Caffe CPU code.
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        for (int n = 0; n < num_; ++n) {
          // gradient w.r.t. bottom data, if necessary
          if (propagate_down[i]) {
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
        }
        */
    }
    // Free temporary columns
    Py_DECREF(col);
    // decref from contiguous check
    Py_DECREF(bottom);
    Py_DECREF(weight);
    Py_DECREF(top);

    // Note that we don't change the refcount of the output matrix here. Output
    // (re)allocation and refcounting is done in BaseCorrMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}

