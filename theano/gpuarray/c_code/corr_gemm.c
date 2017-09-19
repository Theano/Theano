#section kernels

#kernel dilated_im2col_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"
// TODO check kernel flags
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

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu)
// Kernels for fast unfold + copy
// GPU kernel for the case of dilation
KERNEL void dilated_im2col_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset,
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
    const ga_size height, const ga_size width,
    const ga_size kernel_h, const ga_size kernel_w,
    const ga_size dilation_h, const ga_size dilation_w,
    const ga_size pad_hl, const ga_size pad_wl,
    const ga_size stride_h, const ga_size stride_w,
    const ga_size height_col, const ga_size width_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_col,
    const ga_size offset_col) {
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    const ga_size h_index = index / width_col;
    const ga_size h_col = h_index % height_col;
    const ga_size w_col = index % width_col;
    const ga_size c_im = h_index / height_col;
    const ga_size c_col = c_im * kernel_h * kernel_w;
    const ga_size h_offset = h_col * stride_h - pad_hl;
    const ga_size w_offset = w_col * stride_w - pad_wl;
    GLOBAL_MEM DTYPE_INPUT_0 * data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im_ptr = data_im + data_im_offset;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (ga_size i = 0; i < kernel_h; ++i) {
      for (ga_size j = 0; j < kernel_w; ++j) {
        ga_size h_im = h_offset + i * dilation_h;
        ga_size w_im = w_offset + j * dilation_w;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

#kernel im2col_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

KERNEL void im2col_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset,
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
    const ga_size height, const ga_size width,
    const ga_size kernel_h, const ga_size kernel_w,
    const ga_size pad_hl, const ga_size pad_wl,
    const ga_size stride_h, const ga_size stride_w,
    const ga_size height_col, const ga_size width_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_col,
    const ga_size offset_col) {
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    const ga_size h_index = index / width_col;
    const ga_size h_col = h_index % height_col;
    const ga_size w_col = index % width_col;
    const ga_size c_im = h_index / height_col;
    const ga_size c_col = c_im * kernel_h * kernel_w;
    const ga_size h_offset = h_col * stride_h - pad_hl;
    const ga_size w_offset = w_col * stride_w - pad_wl;
    GLOBAL_MEM DTYPE_INPUT_0 * data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im_ptr = data_im + data_im_offset;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (ga_size i = 0; i < kernel_h; ++i) {
      for (ga_size j = 0; j < kernel_w; ++j) {
        ga_size h_im = h_offset + i ;
        ga_size w_im = w_offset + j ;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
           data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

// GPU kernel for the case of dilation
#kernel dilated_col2im_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size, size :
#include "cluda.h"

KERNEL void dilated_col2im_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_col, const ga_size offset_col,
    const ga_size height, const ga_size width, const ga_size channels,
    const ga_size kernel_h, const ga_size kernel_w,
    const ga_size dilation_h, const ga_size dilation_w,
    const ga_size pad_hl, const ga_size pad_wl,
    const ga_size stride_h, const ga_size stride_w,
    const ga_size height_col, const ga_size width_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset) {
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    DTYPE_INPUT_0 val = 0;
    const ga_size w_im = index % width + pad_wl;
    const ga_size h_im = (index / width) % height + pad_hl;
    const ga_size c_im = index / (width * height);
    ga_size kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    ga_size kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const ga_size w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const ga_size w_col_end = min(w_im / stride_w + 1, width_col);
    const ga_size h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const ga_size h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (ga_size h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (ga_size w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        ga_size h_k = (h_im - h_col * stride_h);
        ga_size w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          ga_size data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[data_im_offset + index] = val;
  }
}

#kernel col2im_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, *, size, size :
#include "cluda.h"

KERNEL void col2im_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_col, const ga_size offset_col,
    const ga_size height, const ga_size width, const ga_size channels,
    const ga_size kernel_h, const ga_size kernel_w,
    const ga_size pad_hl, const ga_size pad_wl,
    const ga_size stride_h, const ga_size stride_w,
    const ga_size height_col, const ga_size width_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset) {
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    DTYPE_INPUT_0 val = 0;
    const ga_size w_im = index % width + pad_wl;
    const ga_size h_im = (index / width) % height + pad_hl;
    const ga_size c_im = index / (width * height);
    // compute the start and end of the output
    const ga_size w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const ga_size w_col_end = min(w_im / stride_w + 1, width_col);
    const ga_size h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const ga_size h_col_end = min(h_im / stride_h + 1, height_col);
    // equivalent implementation, no dilation
    ga_size offset =
      (c_im * kernel_h * kernel_w + h_im * kernel_w + w_im) * height_col * width_col;
    ga_size coeff_h_col = (1 - stride_h * kernel_w * height_col) * width_col;
    ga_size coeff_w_col = (1 - stride_w * height_col * width_col);
    for (ga_size h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (ga_size w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[data_im_offset + index] = val;
  }
}

#section support_code

int rgemm(cb_order o, cb_transpose tA, cb_transpose tB,
          size_t M, size_t N, size_t K, double alpha,
          GpuArray *A, size_t offA, size_t lda,
          GpuArray *B, size_t offB, size_t ldb,
          double beta, GpuArray *C, size_t offC, size_t ldc) {
  switch (A->typecode) {
  case GA_FLOAT:
    return gpublas_sgemm(o, tA, tB,
                         M, N, K, alpha,
                         A->data, (A->offset / 4) + offA, lda,
                         B->data, (B->offset / 4) + offB, ldb,
                         beta,
                         C->data, (C->offset / 4) + offC, ldc);
  case GA_DOUBLE:
    return gpublas_dgemm(o, tA, tB,
                         M, N, K, alpha,
                         A->data, (A->offset / 8) + offA, lda,
                         B->data, (B->offset / 8) + offB, ldb,
                         beta,
                         C->data, (C->offset / 8) + offC, ldc);
  case GA_HALF:
    return gpublas_hgemm(o, tA, tB,
                         M, N, K, alpha,
                         A->data, (A->offset / 2) + offA, lda,
                         B->data, (B->offset / 2) + offB, ldb,
                         beta,
                         C->data, (C->offset / 2) + offC, ldc);
  default:
    return GA_UNSUPPORTED_ERROR;
  }
}

#section support_code_struct

int im2col(GpuArray *data_im, const size_t data_im_offset, const size_t channels,
    const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w,
    const size_t dilation_h, const size_t dilation_w,
    const size_t pad_hl, const size_t pad_hr,
    const size_t pad_wl, const size_t pad_wr,
    const size_t stride_h, const size_t stride_w,
    GpuArray *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  size_t dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  size_t dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  size_t height_col = (height + pad_hl + pad_hr - dil_kernel_h) / stride_h + 1;
  size_t width_col = (width + pad_wl + pad_wr - dil_kernel_w) / stride_w + 1;
  size_t num_kernels = channels * height_col * width_col;
  int err;
  if (dilation_h != 1 || dilation_w != 1) {
    err = dilated_im2col_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_im->data, data_im->offset, data_im_offset,
      height, width, kernel_h, kernel_w,
      dilation_h, dilation_w, pad_hl, pad_wl, stride_h, stride_w, height_col,
      width_col, data_col->data, data_col->offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: dilated_im2col_kernel: %s.",
                     GpuKernel_error(&k_dilated_im2col_kernel, err));
    }
  } else {
    err = im2col_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_im->data, data_im->offset, data_im_offset,
      height, width, kernel_h, kernel_w,
      pad_hl, pad_wl, stride_h, stride_w, height_col,
      width_col, data_col->data, data_col->offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: im2col_kernel: %s.",
                     GpuKernel_error(&k_im2col_kernel, err));
    }
  }
  return err;
}

int col2im(GpuArray *data_col, const size_t channels,
    const size_t height, const size_t width, const size_t patch_h, const size_t patch_w,
    const size_t dilation_h, const size_t dilation_w,
    const size_t pad_hl, const size_t pad_hr, const size_t pad_wl, const size_t pad_wr,
    const size_t stride_h, const size_t stride_w, GpuArray *data_im, const size_t data_im_offset) {
  size_t dil_patch_h = (patch_h - 1) * dilation_h + 1;
  size_t dil_patch_w = (patch_w - 1) * dilation_w + 1;
  size_t height_col = (height + pad_hl + pad_hr - dil_patch_h) / stride_h + 1;
  size_t width_col = (width + pad_wl + pad_wr - dil_patch_w) / stride_w + 1;
  size_t num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  int err;
  if (dilation_h != 1 || dilation_w != 1) {
    err = dilated_col2im_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_col->data, data_col->offset,
      height, width, channels, patch_h, patch_w,
      dilation_h, dilation_w, pad_hl, pad_wl, stride_h, stride_w,
      height_col, width_col, data_im->data, data_im->offset, data_im_offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: dilated_col2im_kernel: %s.",
                     GpuKernel_error(&k_dilated_col2im_kernel, err));
    }
  } else {
    err = col2im_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_col->data, data_col->offset,
      height, width, channels, patch_h, patch_w,
      pad_hl, pad_wl, stride_h, stride_w,
      height_col, width_col, data_im->data, data_im->offset, data_im_offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: col2im_kernel: %s.",
                     GpuKernel_error(&k_col2im_kernel, err));
    }
  }
  return err;
}


// Theano op code
// Authors: Arjun Jain, Frederic Bastien, Jan Schluter
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
PyGpuArrayObject* corrMM(PyGpuArrayObject *const bottom,
                         PyGpuArrayObject *const weight,
                         PyGpuArrayObject *const top,
                         const size_t direction,
                         const size_t dH = 1,
                         const size_t dW = 1,
                         const size_t dilH = 1,
                         const size_t dilW = 1,
                         const size_t padH_l = 0,
                         const size_t padH_r = 0,
                         const size_t padW_l = 0,
                         const size_t padW_r = 0,
                         const size_t numgroups = 1,
                         const size_t unshared = 0)
{
    if (PyGpuArray_NDIM(bottom) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires bottom of 4D");
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&bottom->ga))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires bottom to be C-contiguous, "
                "but strides are: %ld %ld %ld %ld\n",
                PyGpuArray_STRIDES(bottom)[0],
                PyGpuArray_STRIDES(bottom)[1],
                PyGpuArray_STRIDES(bottom)[2],
                PyGpuArray_STRIDES(bottom)[3]);
        return NULL;
    }

    if (PyGpuArray_NDIM(weight) != (unshared ? 6 : 4))
    {
        PyErr_Format(PyExc_ValueError, "GpuCorrMM requires weight of %dD", unshared ? 6 : 4);
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&weight->ga))
    {
        if (unshared) {
            PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM requires weight to be C-contiguous, "
                    "but strides are: %ld %ld %ld %ld %ld %ld\n",
                    PyGpuArray_STRIDES(weight)[0],
                    PyGpuArray_STRIDES(weight)[1],
                    PyGpuArray_STRIDES(weight)[2],
                    PyGpuArray_STRIDES(weight)[3],
                    PyGpuArray_STRIDES(weight)[4],
                    PyGpuArray_STRIDES(weight)[5]);
            return NULL;
        }
        else {  
            PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM requires weight to be C-contiguous, "
                    "but strides are: %ld %ld %ld %ld\n",
                    PyGpuArray_STRIDES(weight)[0],
                    PyGpuArray_STRIDES(weight)[1],
                    PyGpuArray_STRIDES(weight)[2],
                    PyGpuArray_STRIDES(weight)[3]);
            return NULL;
        }
    }

    if (PyGpuArray_NDIM(top) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires top of 4D");
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&top->ga))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires top to be C-contiguous, "
                "but strides are: %ld %ld %ld %ld\n",
                PyGpuArray_STRIDES(top)[0],
                PyGpuArray_STRIDES(top)[1],
                PyGpuArray_STRIDES(top)[2],
                PyGpuArray_STRIDES(top)[3]);
        return NULL;
    }

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth)
    const size_t batchSize = PyGpuArray_DIMS(bottom)[0];
    const size_t nChannels = PyGpuArray_DIMS(bottom)[1];
    const size_t bottomHeight = PyGpuArray_DIMS(bottom)[2];
    const size_t bottomWidth = PyGpuArray_DIMS(bottom)[3];
    // weights: (nFilters, nChannels, rows, columns)
    // or (nFilters, out_rows, out_columns, nChannels, rows, columns) -> for unshared
    const size_t nFilters = PyGpuArray_DIMS(weight)[0];

    const size_t kH = PyGpuArray_DIMS(weight)[unshared ? 4 : 2];
    const size_t kW = PyGpuArray_DIMS(weight)[unshared ? 5 : 3];
    if (nChannels != PyGpuArray_DIMS(weight)[unshared ? 3 : 1] * numgroups) {
        PyErr_SetString(PyExc_ValueError,
                "GpuCorrMM images and kernel must have the same stack size\n");
        return NULL;
    }
    if ((nFilters % numgroups) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "GPUCorrMM the number of filters must be divisible by the number of groups\n");
        return NULL;
    }
    // implicit dilated filter
    const size_t dil_kH = (kH - 1) * dilH + 1;
    const size_t dil_kW = (kW - 1) * dilW + 1;
    // top: (batchSize, nFilters, topHeight, topWidth)
    const size_t topHeightNoDH = (bottomHeight + padH_l + padH_r - dil_kH);
    const size_t topWidthNoDW  = (bottomWidth + padW_l + padW_r - dil_kW);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) % y) == 0 ? 0 : 1)) : (x / y))
    const size_t topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const size_t topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
#undef _CONV_FLOORDIV
    if (unshared) {
        if (topHeight != PyGpuArray_DIMS(weight)[1] ||
                topWidth != PyGpuArray_DIMS(weight)[2]) {
            PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM regions in kernel must match output regions:\n"
                    "  bottom shape: %ld %ld %ld %ld\n"
                    "  weight shape: %ld %ld %ld %ld %ld %ld"
                    " (expected %ld %ld %ld %ld %ld %ld)\n"
                    "  top shape(calculated): %ld %ld %ld %ld\n",
                    batchSize, nChannels, bottomHeight, bottomWidth,
                    nFilters, PyGpuArray_DIMS(weight)[1],
                    PyGpuArray_DIMS(weight)[2], nChannels / numgroups, kH, kW,
                    nFilters, topHeight, topWidth, nChannels / numgroups, kH, kW,
                    batchSize, nFilters, topHeight, topWidth);
            return NULL;
        }
        if (batchSize != PyGpuArray_DIMS(top)[0] ||
                nFilters != PyGpuArray_DIMS(top)[1] ||
                topHeight != PyGpuArray_DIMS(top)[2] ||
                topWidth != PyGpuArray_DIMS(top)[3]) {
            PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM shape inconsistency:\n"
                    "  bottom shape: %ld %ld %ld %ld\n"
                    "  weight shape: %ld %ld %ld %ld %ld %ld\n"
                    "  top shape: %ld %ld %ld %ld (expected %ld %ld %ld %ld)\n",
                    batchSize, nChannels, bottomHeight, bottomWidth,
                    nFilters, topHeight, topWidth, nChannels / numgroups, kH, kW,
                    PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                    PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
                    batchSize, nFilters, topHeight, topWidth);
            return NULL;
        }
    }
    else{
        if (batchSize != PyGpuArray_DIMS(top)[0] ||
                nFilters != PyGpuArray_DIMS(top)[1] ||
                topHeight != PyGpuArray_DIMS(top)[2] ||
                topWidth != PyGpuArray_DIMS(top)[3]) {
            PyErr_Format(PyExc_ValueError,
                    "GpuCorrMM shape inconsistency:\n"
                    "  bottom shape: %ld %ld %ld %ld\n"
                    "  weight shape: %ld %ld %ld %ld\n"
                    "  top shape: %ld %ld %ld %ld (expected %ld %ld %ld %ld)\n",
                    batchSize, nChannels, bottomHeight, bottomWidth,
                    nFilters, nChannels / numgroups, kH, kW,
                    PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                    PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
                    batchSize, nFilters, topHeight, topWidth);
            return NULL;
        }
    }

    int err = gpublas_setup(bottom->context->ctx);
    if (err != GA_NO_ERROR) {
        PyErr_SetString(PyExc_RuntimeError, "Can't setup blas");
        return NULL;
    }

    // Create temporary columns
    size_t col_dim[2];
    col_dim[0] = nChannels * kW * kH;
    col_dim[1] = topHeight * topWidth;
    PyGpuArrayObject* col = (PyGpuArrayObject*)pygpu_empty(2, col_dim,
                                                           bottom->ga.typecode,
                                                           GA_C_ORDER,
                                                           bottom->context,
                                                           Py_None);
    if (NULL == col) {
        PyErr_Format(PyExc_RuntimeError,
                "GpuCorrMM failed to allocate working memory of %ld x %ld\n",
                col_dim[0], col_dim[1]);
        return NULL;
    }

    // Define some useful variables
    const size_t batch_bottom_stride = PyGpuArray_STRIDES(bottom)[0] / gpuarray_get_elsize(bottom->ga.typecode);
    const size_t batch_top_stride = PyGpuArray_STRIDES(top)[0] / gpuarray_get_elsize(top->ga.typecode);
    const size_t group_bottom_stride = (PyGpuArray_STRIDES(bottom)[1] * nChannels / numgroups) / gpuarray_get_elsize(bottom->ga.typecode);
    const size_t group_top_stride = (PyGpuArray_STRIDES(top)[1] * nFilters / numgroups) / gpuarray_get_elsize(top->ga.typecode);
    const size_t group_weight_stride = (PyGpuArray_STRIDES(weight)[0] * nFilters / numgroups) / gpuarray_get_elsize(weight->ga.typecode);
    const size_t K_ = col_dim[0] / numgroups;
    const size_t N_ = col_dim[1];
    const size_t group_col_stride = (K_ * N_);
    const size_t M_ = nFilters / numgroups;

    PyGpuArrayObject *output;
    if (direction == 0) {  // forward pass
        output = top;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            err = GpuArray_memset(&output->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid correlation: im2col, then gemm
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
            // First, im2col
            err = im2col(&bottom->ga, n * batch_bottom_stride,
                         nChannels, bottomHeight,
                         bottomWidth, kH, kW, dilH, dilW,
                         padH_l, padH_r, padW_l, padW_r, dH, dW, &col->ga);
            if (err != GA_NO_ERROR) {
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            if (unshared) {
              for (size_t g = 0; g < numgroups; ++g) {
                for (size_t reg = 0; reg < N_; ++reg){
                  err = rgemm(cb_fortran, cb_no_trans, cb_no_trans,
                                      1, M_, K_, 1,
                                      &col->ga, g * group_col_stride + reg, N_,
                                      &weight->ga, g * group_weight_stride + reg * K_, K_ * N_,
                                      0,
                                      &top->ga, n * batch_top_stride + g * group_top_stride + reg, N_);
                  if (err != GA_NO_ERROR) {
                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM forward encountered an error running gemm: %d", err);
                      Py_DECREF(col);
                      return NULL;
                  }
                }
              }
            }
            else {
              for (size_t g = 0; g < numgroups; ++g){
                  err = rgemm(cb_fortran, cb_no_trans, cb_no_trans,
                              N_, M_, K_, 1,
                              &col->ga, g * group_col_stride, N_,
                              &weight->ga, g * group_weight_stride, K_,
                              0,
                              &top->ga, n * batch_top_stride + g * group_top_stride, N_);
                if (err != GA_NO_ERROR) {
                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM forward encountered an error running gemm: %d", err);
                    Py_DECREF(col);
                    return NULL;
                }
              }
            }
        }
    }
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            err = GpuArray_memset(&output->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. weights could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid convolution: im2col, then gemm
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
            // First, im2col
            err = im2col(&bottom->ga, n * batch_bottom_stride,
                         nChannels, bottomHeight,
                         bottomWidth, kH, kW, dilH, dilW,
                         padH_l, padH_r, padW_l, padW_r, dH, dW, &col->ga);
            if (err != GA_NO_ERROR) {
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            // Note that we accumulate into weight. We do so by setting beta = 0
            // for the first iteration and beta = 1 for subsequent ones. (This
            // is faster than setting weight to all zeros before the loop.)
            if (unshared) {
              for (size_t g = 0; g < numgroups; ++g) {
                for (size_t reg = 0; reg < N_; ++reg){
                  err = rgemm(cb_fortran, cb_trans, cb_no_trans,
                              K_, M_, 1, 1,
                              &col->ga, g * group_col_stride + reg, N_,
                              &top->ga, n * batch_top_stride + g * group_top_stride + reg, N_,
                              (n == 0) ? 0 : 1,
                              &weight->ga, g * group_weight_stride + reg * K_, K_ * N_);
                  if (err != GA_NO_ERROR) {
                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad weights encountered an error running gemm: %d", err);
                      Py_DECREF(col);
                      return NULL;
                  }
                }
              }
            }
            else{
              for(size_t g = 0; g < numgroups; g++){ 
                  err = rgemm(cb_fortran, cb_trans, cb_no_trans,
                              K_, M_, N_, 1,
                              &col->ga, g * group_col_stride, N_,
                              &top->ga, n * batch_top_stride + g * group_top_stride, N_,
                              (n == 0) ? 0 : 1,
                              &weight->ga, g * group_weight_stride, K_);
                if (err != GA_NO_ERROR) {
                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad weights encountered an error running gemm: %d", err);
                    Py_DECREF(col);
                    return NULL;
                }
              }
            }
        }
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            err = GpuArray_memset(&output->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. inputs could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // full convolution: gemm, then col2im
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
            // gemm into columns
            if (unshared) {
              for (size_t g = 0; g < numgroups; ++g){
                for (size_t reg = 0; reg < N_; ++reg) {
                  err = rgemm(cb_fortran, cb_no_trans, cb_trans,
                              1, K_, M_, 1,
                              &top->ga, n * batch_top_stride + g * group_top_stride + reg, N_,
                              &weight->ga, g * group_weight_stride + reg * K_, K_ * N_,
                              0,
                              &col->ga, g * group_col_stride + reg, N_);
                  if (err != GA_NO_ERROR) {
                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad inputs encountered an error running gemm: %d", err);
                      Py_DECREF(col);
                      return NULL;
                  }
                }
              }
            }
            else {
              for (size_t g = 0; g < numgroups; ++g){
                err = rgemm(cb_fortran, cb_no_trans, cb_trans,
                            N_, K_, M_, 1,
                            &top->ga, n * batch_top_stride + g * group_top_stride, N_,
                            &weight->ga, g * group_weight_stride, K_,
                            0,
                            &col->ga, g * group_col_stride, N_);
                if (err != GA_NO_ERROR) {
                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad inputs encountered an error running gemm: %d", err);
                    Py_DECREF(col);
                    return NULL;
                }
              }
            }
            // col2im back to the data
            err = col2im(&col->ga, nChannels, bottomHeight, bottomWidth,
                         kH, kW, dilH, dilW, padH_l, padH_r, padW_l, padW_r,
                         dH, dW, &bottom->ga, n * batch_bottom_stride);
            if (err != GA_NO_ERROR) {
                Py_DECREF(col);
                return NULL;
            }
        }
    }
    // Free temporary columns
    Py_DECREF(col);

    // Note that we don't change the refcount of the output matrix here. Output
    // (re)allocation and refcounting is done in BaseGpuCorrMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}
