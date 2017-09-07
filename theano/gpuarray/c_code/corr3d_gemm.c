#section kernels

#kernel dilated_im3d2col_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

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
KERNEL void dilated_im3d2col_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset,
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
    const ga_size height, const ga_size width, const ga_size depth,
    const ga_size kernel_h, const ga_size kernel_w, const ga_size kernel_d,
    const ga_size dilation_h, const ga_size dilation_w, const ga_size dilation_d,
    const ga_size pad_h, const ga_size pad_w, const ga_size pad_d,
    const ga_size stride_h, const ga_size stride_w, const ga_size stride_d,
    const ga_size height_col, const ga_size width_col, const ga_size depth_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_col,
    const ga_size offset_col) {
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    const ga_size w_index = index / depth_col;
    const ga_size h_index = w_index / width_col;
    const ga_size d_col = index % depth_col;
    const ga_size h_col = h_index % height_col;
    const ga_size w_col = w_index % width_col;
    const ga_size c_im = h_index / height_col;
    const ga_size c_col = c_im * kernel_h * kernel_w * kernel_d;
    const ga_size h_offset = h_col * stride_h - pad_h;
    const ga_size w_offset = w_col * stride_w - pad_w;
    const ga_size d_offset = d_col * stride_d - pad_d;
    GLOBAL_MEM DTYPE_INPUT_0 * data_col_ptr = data_col;
    data_col_ptr += c_col * (height_col * width_col * depth_col) +
      h_col * (width_col * depth_col) + w_col * depth_col + d_col;
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im_ptr = data_im + data_im_offset;
    data_im_ptr += c_im * (height * width * depth) +
      h_offset * (width * depth) + w_offset * depth + d_offset;
    for (ga_size i = 0; i < kernel_h; ++i) {
      ga_size h_im = h_offset + i * dilation_h;
      for (ga_size j = 0; j < kernel_w; ++j) {
        ga_size w_im = w_offset + j * dilation_w;
        for (ga_size k = 0; k < kernel_d; ++k) {
          ga_size d_im = d_offset + k * dilation_d;
          *data_col_ptr = (h_im >= 0 && w_im >= 0 && d_im >= 0 &&
                           h_im < height && w_im < width && d_im < depth) ?
                           data_im_ptr[i * dilation_h * (width * depth) +
                                       j * dilation_w * depth +
                                       k * dilation_d] : 0;
          data_col_ptr += height_col * width_col * depth_col;
        }
      }
    }
  }
}

#kernel im3d2col_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

KERNEL void im3d2col_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset,
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
    const ga_size height, const ga_size width, const ga_size depth,
    const ga_size kernel_h, const ga_size kernel_w, const ga_size kernel_d,
    const ga_size pad_h, const ga_size pad_w, const ga_size pad_d,
    const ga_size stride_h, const ga_size stride_w, const ga_size stride_d,
    const ga_size height_col, const ga_size width_col, const ga_size depth_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_col,
    const ga_size offset_col) {
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    const ga_size w_index = index / depth_col;
    const ga_size h_index = w_index / width_col;
    const ga_size d_col = index % depth_col;
    const ga_size h_col = h_index % height_col;
    const ga_size w_col = w_index % width_col;
    const ga_size c_im = h_index / height_col;
    const ga_size c_col = c_im * kernel_h * kernel_w * kernel_d;
    const ga_size h_offset = h_col * stride_h - pad_h;
    const ga_size w_offset = w_col * stride_w - pad_w;
    const ga_size d_offset = d_col * stride_d - pad_d;
    GLOBAL_MEM DTYPE_INPUT_0 * data_col_ptr = data_col;
    data_col_ptr += c_col * (height_col * width_col * depth_col) +
      h_col * (width_col * depth_col) + w_col * depth_col + d_col;
    GLOBAL_MEM const DTYPE_INPUT_0 * data_im_ptr = data_im + data_im_offset;
    data_im_ptr += c_im * (height * width * depth) +
      h_offset * (width * depth) + w_offset * depth + d_offset;
    for (ga_size i = 0; i < kernel_h; ++i) {
      ga_size h_im = h_offset + i;
      for (ga_size j = 0; j < kernel_w; ++j) {
        ga_size w_im = w_offset + j;
        for (ga_size k = 0; k < kernel_d; ++k) {
          ga_size d_im = d_offset + k;
          *data_col_ptr = (h_im >= 0 && w_im >= 0 && d_im >= 0 &&
                           h_im < height && w_im < width && d_im < depth) ?
                           data_im_ptr[i * (width * depth) + j * depth + k] : 0;
          data_col_ptr += height_col * width_col * depth_col;
        }
      }
    }
  }
}

// GPU kernel for the case of dilation
#kernel dilated_col2im3d_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size, size :
#include "cluda.h"

KERNEL void dilated_col2im3d_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_col,
    const ga_size offset_col,
    const ga_size height, const ga_size width, const ga_size depth,
    const ga_size channels,
    const ga_size kernel_h, const ga_size kernel_w, const ga_size kernel_d,
    const ga_size dilation_h, const ga_size dilation_w, const ga_size dilation_d,
    const ga_size pad_h, const ga_size pad_w, const ga_size pad_d,
    const ga_size stride_h, const ga_size stride_w, const ga_size stride_d,
    const ga_size height_col, const ga_size width_col, const ga_size depth_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset) {
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    DTYPE_INPUT_0 val = 0;
    const ga_size d_im = index % depth + pad_d;
    const ga_size w_index = index / depth;
    const ga_size w_im = w_index % width + pad_w;
    const ga_size h_index = w_index / width;
    const ga_size h_im = h_index % height + pad_h;
    const ga_size c_im = h_index / height;
    ga_size kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    ga_size kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    ga_size kernel_extent_d = (kernel_d - 1) * dilation_d + 1;
    // compute the start and end of the output
    const ga_size d_col_start =
        (d_im < kernel_extent_d) ? 0 : (d_im - kernel_extent_d) / stride_d + 1;
    const ga_size d_col_end = min(d_im / stride_d + 1, depth_col);
    const ga_size w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const ga_size w_col_end = min(w_im / stride_w + 1, width_col);
    const ga_size h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const ga_size h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (ga_size d_col = d_col_start; d_col < d_col_end; ++d_col) {
      for (ga_size h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (ga_size w_col = w_col_start; w_col < w_col_end; ++w_col) {
          ga_size h_k = (h_im - h_col * stride_h);
          ga_size w_k = (w_im - w_col * stride_w);
          ga_size d_k = (d_im - d_col * stride_d);
          if (h_k % dilation_h == 0 && w_k % dilation_w == 0 && d_k % dilation_d == 0) {
            h_k /= dilation_h;
            w_k /= dilation_w;
            d_k /= dilation_d;
            ga_size data_col_index = c_im * kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col +
                                     h_k             * kernel_w * kernel_d * height_col * width_col * depth_col +
                                     w_k                        * kernel_d * height_col * width_col * depth_col +
                                     d_k                                   * height_col * width_col * depth_col +
                                     h_col                                              * width_col * depth_col +
                                     w_col                                                          * depth_col +
                                     d_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_im[data_im_offset + index] = val;
  }
}

#kernel col2im3d_kernel : size, *, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, size, *, size, size :
#include "cluda.h"

KERNEL void col2im3d_kernel(const ga_size n,
    GLOBAL_MEM const DTYPE_INPUT_0 * data_col,
    const ga_size offset_col,
    const ga_size height, const ga_size width, const ga_size depth,
    const ga_size channels,
    const ga_size kernel_h, const ga_size kernel_w, const ga_size kernel_d,
    const ga_size pad_h, const ga_size pad_w, const ga_size pad_d,
    const ga_size stride_h, const ga_size stride_w, const ga_size stride_d,
    const ga_size height_col, const ga_size width_col, const ga_size depth_col,
    GLOBAL_MEM DTYPE_INPUT_0 * data_im,
    const ga_size offset_im,
    const ga_size data_im_offset) {
    // offset_im is the pointer offset for data_im.
    // data_im_offset is an offset of elements in the array
  data_im = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_im) + offset_im);
  data_col = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)data_col) + offset_col);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < (n); index += LDIM_0 * GDIM_0) {
    DTYPE_INPUT_0 val = 0;
    const ga_size d_im = index % depth + pad_d;
    const ga_size w_index = index / depth;
    const ga_size w_im = w_index % width + pad_w;
    const ga_size h_index = w_index / width;
    const ga_size h_im = h_index % height + pad_h;
    const ga_size c_im = h_index / height;

    // compute the start and end of the output
    const ga_size d_col_start = (d_im < kernel_d) ? 0 : (d_im - kernel_d) / stride_d + 1;
    const ga_size d_col_end = min(d_im / stride_d + 1, depth_col);
    const ga_size w_col_start = (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const ga_size w_col_end = min(w_im / stride_w + 1, width_col);
    const ga_size h_col_start = (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const ga_size h_col_end = min(h_im / stride_h + 1, height_col);

    ga_size offset =
      (c_im * kernel_h * kernel_w * kernel_d + h_im * kernel_w * kernel_d +
       w_im * kernel_d + d_im) * height_col * width_col * depth_col;

    ga_size coeff_h_col = (1 - stride_h * kernel_w * kernel_d * height_col) * width_col * depth_col;
    ga_size coeff_w_col = (1 - stride_w * kernel_d * height_col * width_col) * depth_col;
    ga_size coeff_d_col = (1 - stride_d * height_col * width_col * depth_col);
    for (ga_size d_col = d_col_start; d_col < d_col_end; ++d_col) {
      for (ga_size h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (ga_size w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col + d_col * coeff_d_col];
        }
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

int im3d2col(
    GpuArray *data_im, const size_t data_im_offset, const size_t channels,
    const size_t height, const size_t width, const size_t depth,
    const size_t kernel_h, const size_t kernel_w, const size_t kernel_d,
    const size_t dilation_h, const size_t dilation_w, const size_t dilation_d,
    const size_t pad_h, const size_t pad_w, const size_t pad_d,
    const size_t stride_h, const size_t stride_w, const size_t stride_d,
    GpuArray *data_col) {
  // We are going to launch channels * height_col * width_col * depth_col
  // kernels, each kernel responsible for copying a single-channel grid.
  size_t dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  size_t dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  size_t dil_kernel_d = (kernel_d - 1) * dilation_d + 1;
  size_t height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  size_t width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  size_t depth_col = (depth + 2 * pad_d - dil_kernel_d) / stride_d + 1;
  size_t num_kernels = channels * height_col * width_col * depth_col;
  int err;
  if (dilation_h != 1 || dilation_w != 1 || dilation_d != 1) {
    err = dilated_im3d2col_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_im->data, data_im->offset,
      data_im_offset, height, width, depth,
      kernel_h, kernel_w, kernel_d, dilation_h, dilation_w, dilation_d,
      pad_h, pad_w, pad_d, stride_h, stride_w, stride_d, height_col,
      width_col, depth_col, data_col->data, data_col->offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: dilated_im3d2col_kernel: %s.",
                     GpuKernel_error(&k_dilated_im3d2col_kernel, err));
    }
  } else {
    err = im3d2col_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_im->data, data_im->offset,
      data_im_offset, height, width, depth,
      kernel_h, kernel_w, kernel_d, pad_h, pad_w, pad_d,
      stride_h, stride_w, stride_d, height_col, width_col, depth_col,
      data_col->data, data_col->offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: im3d2col_kernel: %s.",
                     GpuKernel_error(&k_im3d2col_kernel, err));
    }
  }
  return err;
}

int col2im3d(GpuArray *data_col, const size_t channels,
    const size_t height, const size_t width, const size_t depth,
    const size_t patch_h, const size_t patch_w, const size_t patch_d,
    const size_t dilation_h, const size_t dilation_w, const size_t dilation_d,
    const size_t pad_h, const size_t pad_w, const size_t pad_d,
    const size_t stride_h, const size_t stride_w, const size_t stride_d,
    GpuArray *data_im, const size_t data_im_offset) {
  size_t dil_patch_h = (patch_h - 1) * dilation_h + 1;
  size_t dil_patch_w = (patch_w - 1) * dilation_w + 1;
  size_t dil_patch_d = (patch_d - 1) * dilation_d + 1;
  size_t height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  size_t width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  size_t depth_col = (depth + 2 * pad_d - dil_patch_d) / stride_d + 1;
  size_t num_kernels = channels * height * width * depth;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  int err;
  if (dilation_h != 1 || dilation_w != 1 || dilation_d != 1) {
    err = dilated_col2im3d_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_col->data, data_col->offset,
      height, width, depth, channels, patch_h, patch_w,
      patch_d, dilation_h, dilation_w, dilation_d, pad_h, pad_w, pad_d,
      stride_h, stride_w, stride_d, height_col, width_col, depth_col,
      data_im->data, data_im->offset, data_im_offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: dilated_col2im3d_kernel: %s.",
                     GpuKernel_error(&k_dilated_col2im3d_kernel, err));
    }
  }
  else{
    err = col2im3d_kernel_scall(
      1, &num_kernels, 0,
      num_kernels, data_col->data, data_col->offset,
      height, width, depth, channels, patch_h, patch_w,
      patch_d, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d,
      height_col, width_col, depth_col,
      data_im->data, data_im->offset, data_im_offset);
    if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "gpuarray error: col2im3d_kernel: %s.",
                     GpuKernel_error(&k_col2im3d_kernel, err));
    }
  }
  return err;
}


// Theano op code
// Authors: Arjun Jain, Frederic Bastien, Jan Schluter
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
// Adaptation for 3d
PyGpuArrayObject* corr3dMM(PyGpuArrayObject *const bottom,
                           PyGpuArrayObject *const weight,
                           PyGpuArrayObject *const top,
                           const size_t direction,
                           const size_t dH = 1,
                           const size_t dW = 1,
                           const size_t dD = 1,
                           const size_t dilH = 1,
                           const size_t dilW = 1,
                           const size_t dilD = 1,
                           const size_t padH = 0,
                           const size_t padW = 0,
                           const size_t padD = 0,
                           const size_t numgroups = 1)
{
    if (PyGpuArray_NDIM(bottom) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires bottom of 5D");
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&bottom->ga))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorr3dMM requires bottom to be C-contiguous, "
                "but strides are: %ld %ld %ld %ld %ld\n",
                PyGpuArray_STRIDES(bottom)[0],
                PyGpuArray_STRIDES(bottom)[1],
                PyGpuArray_STRIDES(bottom)[2],
                PyGpuArray_STRIDES(bottom)[3],
                PyGpuArray_STRIDES(bottom)[4]);
        return NULL;
    }

    if (PyGpuArray_NDIM(weight) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires weight of 5D");
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&weight->ga))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorr3dMM requires weight to be C-contiguous, "
                "but strides are: %ld %ld %ld %ld %ld\n",
                PyGpuArray_STRIDES(weight)[0],
                PyGpuArray_STRIDES(weight)[1],
                PyGpuArray_STRIDES(weight)[2],
                PyGpuArray_STRIDES(weight)[3],
                PyGpuArray_STRIDES(weight)[4]);
        return NULL;
    }

    if (PyGpuArray_NDIM(top) != 5)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires top of 5D");
        return NULL;
    }
    if (!GpuArray_IS_C_CONTIGUOUS(&top->ga))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorr3dMM requires top to be C-contiguous, "
                "but strides are: %ld %ld %ld %ld %ld\n",
                PyGpuArray_STRIDES(top)[0],
                PyGpuArray_STRIDES(top)[1],
                PyGpuArray_STRIDES(top)[2],
                PyGpuArray_STRIDES(top)[3],
                PyGpuArray_STRIDES(top)[4]);
        return NULL;
    }

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth)
    const size_t batchSize = PyGpuArray_DIMS(bottom)[0];
    const size_t nChannels = PyGpuArray_DIMS(bottom)[1];
    const size_t bottomHeight = PyGpuArray_DIMS(bottom)[2];
    const size_t bottomWidth = PyGpuArray_DIMS(bottom)[3];
    const size_t bottomDepth = PyGpuArray_DIMS(bottom)[4];
    // weights: (nFilters, nChannels, rows, columns, slices)
    const size_t nFilters = PyGpuArray_DIMS(weight)[0];
    const size_t kH = PyGpuArray_DIMS(weight)[2];
    const size_t kW = PyGpuArray_DIMS(weight)[3];
    const size_t kD = PyGpuArray_DIMS(weight)[4];
    if (nChannels != PyGpuArray_DIMS(weight)[1] * numgroups) {
        PyErr_SetString(PyExc_ValueError,
                "GpuCorr3dMM images and kernel must have the same stack size\n");
        return NULL;
    }
    if ((nFilters % numgroups) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "CorrMM the number of filters must be divisible by the number of groups\n");
        return NULL;
    }
    // implicit dilated filter
    const size_t dil_kH = (kH - 1) * dilH + 1;
    const size_t dil_kW = (kW - 1) * dilW + 1;
    const size_t dil_kD = (kD - 1) * dilD + 1;
    // top: (batchSize, nFilters, topHeight, topWidth, topDepth)
    const size_t topHeightNoDH = (bottomHeight + 2*padH - dil_kH);
    const size_t topWidthNoDW  = (bottomWidth + 2*padW - dil_kW);
    const size_t topDepthNoDD  = (bottomDepth + 2*padD - dil_kD);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) % y) == 0 ? 0 : 1)) : (x / y))
    const size_t topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const size_t topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
    const size_t topDepth  = _CONV_FLOORDIV_X(topDepthNoDD, dD) + 1;
#undef _CONV_FLOORDIV
    if (batchSize != PyGpuArray_DIMS(top)[0] ||
            nFilters != PyGpuArray_DIMS(top)[1] ||
            topHeight != PyGpuArray_DIMS(top)[2] ||
            topWidth != PyGpuArray_DIMS(top)[3] ||
            topDepth != PyGpuArray_DIMS(top)[4]) {
        PyErr_Format(PyExc_ValueError,
                "GpuCorr3dMM shape inconsistency:\n"
                "  bottom shape: %ld %ld %ld %ld %ld\n"
                "  weight shape: %ld %ld %ld %ld %ld\n"
                "  top shape: %ld %ld %ld %ld %ld (expected %ld %ld %ld %ld %ld)\n",
                batchSize, nChannels, bottomHeight, bottomWidth, bottomDepth,
                nFilters, nChannels / numgroups, kH, kW, kD,
                PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
                PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3], PyGpuArray_DIMS(top)[4],
                batchSize, nFilters, topHeight, topWidth, topDepth);
        return NULL;
    }

    int err = gpublas_setup(bottom->context->ctx);
    if (err != GA_NO_ERROR) {
        PyErr_SetString(PyExc_RuntimeError, "Can't setup blas");
        return NULL;
    }

    // Create temporary columns
    size_t col_dim[2];
    col_dim[0] = nChannels * kW * kH * kD;
    col_dim[1] = topHeight * topWidth * topDepth;
    PyGpuArrayObject* col = (PyGpuArrayObject*)pygpu_empty(2, col_dim,
                                                           bottom->ga.typecode,
                                                           GA_C_ORDER,
                                                           bottom->context,
                                                           Py_None);
    if (NULL == col)
    {
        PyErr_Format(PyExc_RuntimeError,
                "GpuCorr3dMM failed to allocate working memory of %ld x %ld\n",
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
                             "GpuCorr3dMM could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid correlation: im3d2col, then gemm
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
            // First, im3d2col
            err = im3d2col(
              &bottom->ga, n * batch_bottom_stride, nChannels, bottomHeight,
              bottomWidth, bottomDepth, kH, kW, kD, dilH, dilW, dilD,
              padH, padW, padD, dH, dW, dD, &col->ga);
            if (err != GA_NO_ERROR) {
                Py_DECREF(col);
                return NULL;
            }
            for ( size_t g = 0; g < numgroups; ++g){
                // Second, gemm
                err = rgemm(cb_fortran, cb_no_trans, cb_no_trans,
                            N_, M_, K_, 1,
                            &col->ga, g * group_col_stride, N_,
                            &weight->ga, g * group_weight_stride, K_,
                            0,
                            &top->ga, n * batch_top_stride + g * group_top_stride, N_);
            }
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorr3dMM forward encountered an error running gemm.");
                Py_DECREF(col);
                return NULL;
            }
        }
    }
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            err = GpuArray_memset(&output->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorr3dMM grad wrt. weights could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid convolution: im3col, then gemm
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
            // First, im3d2col
            err = im3d2col(
              &bottom->ga, n * batch_bottom_stride, nChannels, bottomHeight,
              bottomWidth, bottomDepth, kH, kW, kD, dilH, dilW, dilD,
              padH, padW, padD, dH, dW, dD, &col->ga);
            if (err != GA_NO_ERROR) {
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            // Note that we accumulate into weight. We do so by setting beta = 0
            // for the first iteration and beta = 1 for subsequent ones. (This
            // is faster than setting weight to all zeros before the loop.)
            for ( size_t g = 0; g < numgroups; ++g){
                err = rgemm(cb_fortran, cb_trans, cb_no_trans,
                            K_, M_, N_, 1,
                            &col->ga, g * group_col_stride, N_,
                            &top->ga, n * batch_top_stride + g * group_top_stride, N_,
                            (n == 0) ? 0 : 1,
                            &weight->ga, g * group_weight_stride, K_);
            }
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorr3dMM grad weights encountered an error running gemm.");
                Py_DECREF(col);
                return NULL;
            }
        }
        if (batchSize == 0) {
            err = GpuArray_memset(&weight->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorr3dMM grad weights could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
        }
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            err = GpuArray_memset(&output->ga, 0);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorr3dMM grad wrt. inputs could not fill the output with zeros: %d", err);
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // full convolution: gemm, then col2im3d
        // Iterate over batch
        for (size_t n = 0; n < batchSize; n++) {
          // gemm into columns
          for ( size_t g = 0; g < numgroups; ++g){
              err = rgemm(cb_fortran, cb_no_trans, cb_trans,
                          N_, K_, M_, 1,
                          &top->ga, n * batch_top_stride + g * group_top_stride, N_,
                          &weight->ga, g * group_weight_stride, K_,
                          0,
                          &col->ga, g * group_col_stride, N_);
          }
          if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuCorr3dMM grad inputs encountered an error running gemm.");
            Py_DECREF(col);
            return NULL;
          }
          // col2im3d back to the data
          err = col2im3d(&col->ga, nChannels,
                         bottomHeight, bottomWidth, bottomDepth,
                         kH, kW, kD, dilH, dilW, dilD, padH, padW, padD,
                         dH, dW, dD, &bottom->ga, n * batch_bottom_stride);
          if (err != GA_NO_ERROR) {
            Py_DECREF(col);
            return NULL;
          }
        }
    }
    // Free temporary columns
    Py_DECREF(col);

    // Note that we don't change the refcount of the output matrix here. Output
    // (re)allocation and refcounting is done in BaseGpuCorr3dMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}
