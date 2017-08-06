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
void im3d2col(const %(float_type)s* data_im, const int channels,
    const int height, const int width, const int depth,
    const int kernel_h, const int kernel_w, const int kernel_d,
    const int dilation_h, const int dilation_w, const int dilation_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    %(float_type)s* data_col) {
  // Implicit dilated kernel size
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int dil_kernel_d = (kernel_d - 1) * dilation_d + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - dil_kernel_d) / stride_d + 1;
  int channels_col = channels * kernel_h * kernel_w * kernel_d;
  for (int c = 0; c < channels_col; ++c) {
    int d_offset = c %% kernel_d;
    int w_offset = (c / kernel_d) %% kernel_w;
    int h_offset = (c / kernel_w / kernel_d) %% kernel_h;
    int c_im = c / kernel_h / kernel_w / kernel_d;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset * dilation_w;
        for (int d = 0; d < depth_col; ++d) {
          int d_pad = d * stride_d - pad_d + d_offset * dilation_d;
          if (h_pad >= 0 && h_pad < height
              && w_pad >= 0 && w_pad < width
              && d_pad >= 0 && d_pad < depth)
            data_col[(npy_intp)((c * height_col + h) * width_col + w) * depth_col + d] =
              data_im[(npy_intp)((c_im * height + h_pad) * width + w_pad) * depth + d_pad];
          else
            data_col[(npy_intp)((c * height_col + h) * width_col + w) * depth_col + d] = 0.;
        }
      }
    }
  }
}

// Unlike the Caffe and Theano GPU verions, the data_im array is set to zero
// before the col2im call rather than doing it here. So, the result is just
// accumulated into data_im.
void col2im3d(const %(float_type)s* data_col, const int channels,
    const int height, const int width, const int depth,
    const int patch_h, const int patch_w, const int patch_d,
    const int dilation_h, const int dilation_w, const int dilation_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    %(float_type)s* data_im) {
  // Implicit dilated patch
  int dil_patch_h = (patch_h - 1) * dilation_h + 1;
  int dil_patch_w = (patch_w - 1) * dilation_w + 1;
  int dil_patch_d = (patch_d - 1) * dilation_d + 1;
  int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - dil_patch_d) / stride_d + 1;
  int num_kernels = channels * height * width * depth;
  int channels_col = channels * patch_h * patch_w * patch_d;
  for (int c = 0; c < channels_col; ++c) {
    int d_offset = c %% patch_d;
    int w_offset = (c / patch_d) %% patch_w;
    int h_offset = (c / patch_w / patch_d) %% patch_h;
    int c_im = c / patch_h / patch_w / patch_d;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset * dilation_w;
        for (int d = 0; d < depth_col; ++d) {
          int d_pad = d * stride_d - pad_d + d_offset * dilation_d;
          if (h_pad >= 0 && h_pad < height
              && w_pad >= 0 && w_pad < width
              && d_pad >= 0 && d_pad < depth)
            data_im[(npy_intp)((c_im * height + h_pad) * width + w_pad) * depth + d_pad] +=
              data_col[(npy_intp)((c * height_col + h) * width_col + w) * depth_col + d];
        }
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
PyArrayObject* corr3dMM(PyArrayObject* bottom,
                        PyArrayObject* weight,
                        PyArrayObject* top,
                        const int direction,
                        const int dH = 1,
                        const int dW = 1,
                        const int dD = 1,
                        const int dilH = 1,
                        const int dilW = 1,
                        const int dilD = 1,
                        const int padH = 0,
                        const int padW = 0,
                        const int padD = 0,
                        const int numgroups=1)
{
    if (PyArray_NDIM(bottom) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM requires bottom of 5D");
        return NULL;
    }
    if (PyArray_TYPE(bottom) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM received bottom with wrong type.");
        return NULL;
    }

    if (PyArray_NDIM(weight) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM requires weight of 5D");
        return NULL;
    }
    if (PyArray_TYPE(weight) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM received weight with wrong type.");
        return NULL;
    }

    if (PyArray_NDIM(top) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM requires top of 5D");
        return NULL;
    }
    if (PyArray_TYPE(top) != %(float_typenum)s)
    {
        PyErr_SetString(PyExc_ValueError, "Corr3dMM received top with wrong type.");
        return NULL;
    }
    // Ensure data is contiguous
    bottom = PyArray_GETCONTIGUOUS(bottom);
    weight = PyArray_GETCONTIGUOUS(weight);
    top = PyArray_GETCONTIGUOUS(top);

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth)
    const int batchSize = PyArray_DIMS(bottom)[0];
    const int nChannels = PyArray_DIMS(bottom)[1];
    const int bottomHeight = PyArray_DIMS(bottom)[2];
    const int bottomWidth = PyArray_DIMS(bottom)[3];
    const int bottomDepth = PyArray_DIMS(bottom)[4];
    // weights: (nFilters, nChannels, rows, columns, slices)
    const int nFilters = PyArray_DIMS(weight)[0];
    const int kH = PyArray_DIMS(weight)[2];
    const int kW = PyArray_DIMS(weight)[3];
    const int kD = PyArray_DIMS(weight)[4];
    if (nChannels != PyArray_DIMS(weight)[1] * numgroups) {
        PyErr_SetString(PyExc_ValueError,
                "Corr3dMM images and kernel must have the same stack size\n");
        return NULL;
    }
    if ((nFilters %% numgroups) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "CorrMM the number of filters must be divisible by the number of groups\n");
        return NULL;
    }
    // implicit dilated filter
    const int dil_kH = (kH - 1) * dilH + 1;
    const int dil_kW = (kW - 1) * dilW + 1;
    const int dil_kD = (kD - 1) * dilD + 1;
    // top: (batchSize, nFilters, topHeight, topWidth, topDepth)
    const int topHeightNoDH = (bottomHeight + 2*padH - dil_kH);
    const int topWidthNoDW  = (bottomWidth + 2*padW - dil_kW);
    const int topDepthNoDD  = (bottomDepth + 2*padD - dil_kD);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) %% y) == 0 ? 0 : 1)) : (x / y))
    const int topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const int topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
    const int topDepth  = _CONV_FLOORDIV_X(topDepthNoDD, dD) + 1;
#undef _CONV_FLOORDIV
    if (batchSize != PyArray_DIMS(top)[0] ||
            nFilters != PyArray_DIMS(top)[1] ||
            topHeight != PyArray_DIMS(top)[2] ||
            topWidth != PyArray_DIMS(top)[3] ||
            topDepth != PyArray_DIMS(top)[4]) {
        PyErr_Format(PyExc_ValueError,
                "Corr3dMM shape inconsistency:\n"
                "  bottom shape: %%d %%d %%d %%d %%d\n"
                "  weight shape: %%d %%d %%d %%d %%d\n"
                "  top shape: %%ld %%ld %%ld %%ld %%ld (expected %%d %%d %%d %%d %%d)\n",
                batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth,
                nFilters, nChannels / numgroups, kH, kW, kD,
                PyArray_DIMS(top)[0], PyArray_DIMS(top)[1],
                PyArray_DIMS(top)[2], PyArray_DIMS(top)[3], PyArray_DIMS(top)[4],
                batchSize, nFilters, topHeight, topWidth, topDepth);
        return NULL;
    }

    // Create temporary columns
    int max_threads = %(omp_get_max_threads)s;
    if (batchSize < max_threads) {
        max_threads = batchSize;
    }
    npy_intp col_dim[3];
    col_dim[0] = (npy_intp)max_threads;
    col_dim[1] = (npy_intp)(nChannels * kW * kH * kD);
    col_dim[2] = (npy_intp)(topHeight * topWidth * topDepth);

    //Change to PyArray_ZEROS which is faster than PyArray_EMPTY.
    PyArrayObject* col = (PyArrayObject*)PyArray_ZEROS(3,
            col_dim,
            PyArray_TYPE(top),
            0); 
    if (NULL == col) {
        PyErr_Format(PyExc_RuntimeError,
                "Corr3dMM failed to allocate working memory of"
                " %%ld x %%ld x %%ld\n",
                col_dim[0], col_dim[1], col_dim[2]);
        return NULL;
    }

    // Define some useful variables
    const int batch_bottom_stride = PyArray_STRIDES(bottom)[0]/%(n_bytes)f;
    const int group_bottom_stride = (PyArray_STRIDES(bottom)[1] * nChannels / numgroups)/%(n_bytes)f;
    const int batch_top_stride = PyArray_STRIDES(top)[0]/%(n_bytes)f;
    const int group_top_stride = (PyArray_STRIDES(top)[1] * nFilters / numgroups)/%(n_bytes)f;
    const int K_ = col_dim[1] / numgroups;
    const int N_ = col_dim[2];
    const int col_stride = (K_ * N_ * numgroups);
    const int group_col_stride = (K_ * N_);
    const int group_weight_stride = (PyArray_STRIDES(weight)[0] * nFilters / numgroups)/%(n_bytes)f;
    const int M_ = nFilters / numgroups;
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
        // valid correlation: im3d2col, then gemm
        // Iterate over batch
        int blas_threads_saved = %(blas_get_num_threads)s;
        // Always forcing gemm to one thread when OpenMP is enalbed for best and stable performance.
        %(blas_set_num_threads)s(1);
        %(omp_flags)s
        for (int n = 0; n < batchSize; ++n) {
            int tid = %(omp_get_thread_num)s;
            // First, im3d2col
            im3d2col((%(float_type)s*)PyArray_DATA(bottom) + n * batch_bottom_stride,
                     nChannels, bottomHeight, bottomWidth, bottomDepth,
                     kH, kW, kD, dilH, dilW, dilD, padH, padW, padD, dH, dW, dD,
                     (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride);

            for ( int g = 0; g < numgroups; ++g){
                // Second, gemm
                %(gemm)s(&NTrans, &NTrans,
                         &N_, &M_, &K_,
                         &one,
                         (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride + g * group_col_stride, &N_,
                         (%(float_type)s*)PyArray_DATA(weight) + g * group_weight_stride, &K_,
                         &zero,
                         (%(float_type)s*)PyArray_DATA(top) + n * batch_top_stride + g * group_top_stride, &N_);
            }
        }
        // Restore to previous blas threads
        %(blas_set_num_threads)s(blas_threads_saved);
    }
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        npy_intp weight_dim[2];
        weight_dim[0] = (npy_intp)max_threads;
        weight_dim[1] = (npy_intp)(M_ * K_ * numgroups);
        PyArrayObject* local_weight = (PyArrayObject*)PyArray_ZEROS(2,
                                   weight_dim, PyArray_TYPE(weight), 0);

        if (NULL == local_weight)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "Corr3dMM failed to allocate weight memory of %%ld x %%ld\n",
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
            im3d2col((%(float_type)s*)PyArray_DATA(bottom) + n * batch_bottom_stride,
                     nChannels, bottomHeight, bottomWidth, bottomDepth,
                     kH, kW, kD, dilH, dilW, dilD, padH, padW, padD, dH, dW, dD,
                     (%(float_type)s*)PyArray_DATA(col)+ tid * col_stride);

            for ( int g = 0; g < numgroups; ++g){
                // Second, gemm
                // Note that we accumulate into weight. We do so by setting beta = 0
                // for the first iteration and beta = 1 for subsequent ones. (This
                // is faster than setting weight to all zeros before the loop.)
                %(gemm)s(&Trans, &NTrans,
                         &K_, &M_, &N_,
                         &one,
                         (%(float_type)s*)PyArray_DATA(col) + tid * col_stride + g * group_col_stride, &N_,
                         (%(float_type)s*)PyArray_DATA(top) + n * batch_top_stride + g * group_top_stride, &N_,
                         (n == 0) ? &zero : &one,
                         (%(float_type)s*)PyArray_DATA(local_weight) + g * group_weight_stride +
                         tid * weight_dim[1], &K_);
            }
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
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        // bottom is set to zero here rather than inside of col2im
        PyArray_FILLWBYTE(bottom, 0);
        // full convolution: gemm, then col2im3d
        // Iterate over batch

        int blas_threads_saved = %(blas_get_num_threads)s;
        // Always forcing gemm to one thread when OpenMP is enalbed for best and stable performance.
        %(blas_set_num_threads)s(1);
        %(omp_flags)s
        for (int n = 0; n < batchSize; ++n) {

            int tid = %(omp_get_thread_num)s;
            for ( int g = 0; g < numgroups; ++g){
                // gemm into columns
                %(gemm)s(&NTrans, &Trans,
                         &N_, &K_, &M_,
                         &one,
                         (%(float_type)s*)PyArray_DATA(top) + n * batch_top_stride + g * group_top_stride, &N_,
                         (%(float_type)s*)PyArray_DATA(weight) + g * group_weight_stride, &K_,
                         &zero,
                         (%(float_type)s*)PyArray_DATA(col) + tid * col_stride + g * group_col_stride, &N_);
            }
            // col2im back to the data
            col2im3d((%(float_type)s*)PyArray_DATA(col) + tid * col_stride, nChannels,
                     bottomHeight, bottomWidth, bottomDepth,
                     kH, kW, kD, dilH, dilW, dilD, padH, padW, padD, dH, dW, dD,
                     (%(float_type)s*)PyArray_DATA(bottom) + n * batch_bottom_stride);
        }
        // Restore to previous blas threads
        %(blas_set_num_threads)s(blas_threads_saved);
    }
    // Free temporary columns
    Py_DECREF(col);
    // decref from contiguous check
    Py_DECREF(bottom);
    Py_DECREF(weight);
    Py_DECREF(top);

    // Note that we don't change the refcount of the output matrix here. Output
    // (re)allocation and refcounting is done in BaseCorr3dMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}

