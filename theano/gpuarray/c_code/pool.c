#section kernels

#kernel max_pool2d_kernel : size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool2d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_h, const ga_size stride_w, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_OUTPUT_0 *z, const ga_size z_off)
{
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)z) + z_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size c = (index / pooled_width / pooled_height) % channels;
    const ga_size n = (index / pooled_width / pooled_height / channels);
    ga_int hstart = (ga_int)(ph*stride_h) - (ga_int)(pad_h);
    const ga_size hend = min(hstart + kernel_h, height);
    ga_int wstart = (ga_int)(pw*stride_w) - (ga_int)(pad_w);
    const ga_size wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const ga_size offset = (n*channels + c) * height * width;
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    DTYPE_OUTPUT_0 maxval = x_slice[hstart*width + wstart];

    for (ga_size h=hstart; h < hend; ++h) {
      for (ga_size w=wstart; w < wend; ++w) {
        // maximum in the region
        if (x_slice[h*width + w] > maxval) {
          maxval = x_slice[h*width + w];
        }
      }
    }
    z[index] = maxval;
  }
}

#kernel max_pool3d_kernel : size, size, size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool3d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   const ga_size depth, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, const ga_size kernel_d, const ga_size kernel_h,
   const ga_size kernel_w, const ga_size stride_d, const ga_size stride_h,
   const ga_size stride_w, const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_OUTPUT_0 *z, const ga_size z_off)
{
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)z) + z_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size pd = (index / pooled_width / pooled_height) % pooled_depth;
    const ga_size c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const ga_size n = (index / pooled_width / pooled_height / pooled_depth / channels);
    ga_int dstart = (ga_int)(pd*stride_d) - (ga_int)(pad_d);
    const ga_size dend = min(dstart + kernel_d, depth);
    ga_int hstart = (ga_int)(ph*stride_h) - (ga_int)(pad_h);
    const ga_size hend = min(hstart + kernel_h, height);
    ga_int wstart = (ga_int)(pw*stride_w) - (ga_int)(pad_w);
    const ga_size wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const ga_size offset = (n*channels + c) * depth * height * width;
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    DTYPE_OUTPUT_0 maxval = x_slice[(dstart*height + hstart)*width + wstart];

    for (ga_size d=dstart; d < dend; ++d) {
      for (ga_size h=hstart; h < hend; ++h) {
        for (ga_size w=wstart; w < wend; ++w) {
          // maximum in the region
          if (x_slice[(d*height + h)*width + w] > maxval) {
            maxval = x_slice[(d*height + h)*width + w];
          }
        }
      }
    }
    z[index] = maxval;
  }
}

#kernel ave_pool2d_kernel : size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, bool, bool, *, size:
#include "cluda.h"

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool2d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_h, const ga_size stride_w, const ga_size pad_h, const ga_size pad_w,
   const ga_bool inc_pad, const ga_bool sum_mode,
   GLOBAL_MEM DTYPE_OUTPUT_0 *z, const ga_size z_off)
{
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)z) + z_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size c = (index / pooled_width / pooled_height) % channels;
    const ga_size n = (index / pooled_width / pooled_height / channels);
    ga_int hstart = (ga_int)(ph*stride_h) - (ga_int)(pad_h);
    ga_size hend = min(hstart + kernel_h, height + pad_h);
    ga_int wstart = (ga_int)(pw*stride_w) - (ga_int)(pad_w);
    ga_size wend = min(wstart + kernel_w, width + pad_w);
    ga_size pool_size;
    if (inc_pad) {
      pool_size = (hend - hstart) * (wend - wstart);
    }
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    if (!inc_pad) {
      pool_size = (hend - hstart) * (wend - wstart);
    }

    const ga_size offset = (n*channels + c) * height * width;
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    DTYPE_OUTPUT_0 collector = 0;

    for (ga_size h=hstart; h < hend; ++h) {
      for (ga_size w=wstart; w < wend; ++w) {
        collector += x_slice[h * width + w];
      }
    }
    if (sum_mode) {
      z[index] = collector;
    }
    else {
      z[index] = collector / pool_size;
    }
  }
}

#kernel ave_pool3d_kernel : size, size, size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, size, size, size, bool, bool, *, size :
#include "cluda.h"

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool3d_kernel(const ga_size nthreads,
                              const ga_size num, const ga_size channels, const ga_size pooled_depth,
                              const ga_size pooled_height, const ga_size pooled_width,
                              const ga_size depth, const ga_size height, const ga_size width,
                              GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, const ga_size kernel_d, const ga_size kernel_h,
                              const ga_size kernel_w, const ga_size stride_d, const ga_size stride_h,
                              const ga_size stride_w, const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
                              const ga_bool inc_pad, const ga_bool sum_mode,
                              GLOBAL_MEM DTYPE_OUTPUT_0 *z, const ga_size z_off)
{
  // grid stride looping
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)z) + z_off);
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size pd = (index / pooled_width / pooled_height) % pooled_depth;
    const ga_size c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const ga_size n = (index / pooled_width / pooled_height / pooled_depth / channels);
    ga_int dstart = (ga_int)(pd*stride_d) - (ga_int)(pad_d);
    ga_size dend = min(dstart + kernel_d, depth + pad_d);
    ga_int hstart = (ga_int)(ph*stride_h) - (ga_int)(pad_h);
    ga_size hend = min(hstart + kernel_h, height + pad_h);
    ga_int wstart = (ga_int)(pw*stride_w) - (ga_int)(pad_w);
    ga_size wend = min(wstart + kernel_w, width + pad_w);
    ga_size pool_size;
    if (inc_pad) {
      pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    }
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dend = min(dend, depth);
    hend = min(hend, height);
    wend = min(wend, width);
    if (!inc_pad) {
      pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    }

    const ga_size offset = (n*channels + c) * depth * height * width;
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    DTYPE_OUTPUT_0 collector = 0;

    for (ga_size d=dstart; d < dend; ++d) {
      for (ga_size h=hstart; h < hend; ++h) {
        for (ga_size w=wstart; w < wend; ++w) {
          collector += x_slice[(d * height + h) * width + w];
        }
      }
    }
    if (sum_mode) {
      z[index] = collector;
    }
    else {
      z[index] = collector / pool_size;
    }
  }
}

#section support_code

// output shape for a given input padded shape, window shape and stride
// We use ssize_t in the max since this is done to avoid negative results.
#define OUTPUT_DIMS(in_dim, ws, st, ignore_border)        \
  (ignore_border ? (in_dim - ws)/st + 1 :                 \
   (st > ws ? (in_dim - 1)/st + 1 :                       \
    std::max<ssize_t>(0, (in_dim - 1 - ws + st)/st) + 1))

#section support_code_struct

int APPLY_SPECIFIC(pool)(PyGpuArrayObject *x,
                         PyArrayObject *ws,
                         PyArrayObject *stride,
                         PyArrayObject *pad,
                         PyGpuArrayObject **z,
                         PARAMS_TYPE* params) {
  bool max_pool = (params->mode == POOLING_MAX);
  bool inc_pad = (params->mode != POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
  bool sum_mode = (params->mode  == POOLING_SUM);
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuPool: requires data to be C-contiguous");
      return 1;
    }
  size_t ndims = PyArray_DIM(ws, 0);
  if (PyGpuArray_NDIM(x) != ndims + 2)
    {
      PyErr_SetString(PyExc_ValueError, "GpuPool: rank error");
      return 1;
    }
  // prepare output
  const size_t* x_dims = PyGpuArray_DIMS(x);
  size_t z_dims[5]; // avoid warning if use 2 + nd
  size_t w[3];
  size_t s[3];
  size_t p[3]; z_dims[0] = x_dims[0]; z_dims[1] = x_dims[1];
  int nonzero_padding = 0;
  for (int i = 0; i < ndims; i++) {
    w[i] = *((npy_int64*)PyArray_GETPTR1(ws, i));
    s[i] = *((npy_int64*)PyArray_GETPTR1(stride, i));
    p[i] = *((npy_int64*)PyArray_GETPTR1(pad, i));
    z_dims[2 + i] = OUTPUT_DIMS(x_dims[2 + i] + 2*p[i], w[i], s[i], params->ignore_border);
    if (p[i] > 0) {
      nonzero_padding = 1;
    }
  }
  if (!params->ignore_border && nonzero_padding) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuPool: padding works only with ignore_border=True");
    return 1;
  }

  if (theano_prep_output(z, PyGpuArray_NDIM(x), z_dims,
                         x->ga.typecode, GA_C_ORDER, params->context) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuPool: failed to allocate memory");
      return 1;
    }
  {
    // scope for running kernel
    int err;

    if (ndims == 2) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
      if (max_pool) {
        err = max_pool2d_kernel_scall(1, &num_kernels, 0, num_kernels,
                                      z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                      x_dims[2], x_dims[3],
                                      x->ga.data, x->ga.offset, w[0], w[1], s[0], s[1], p[0], p[1],
                                      (*z)->ga.data, (*z)->ga.offset);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: max_pool2d_kernel %s.",
                       GpuKernel_error(&k_max_pool2d_kernel, err));
          return 1;
        }
      } else {
        err = ave_pool2d_kernel_scall(1, &num_kernels, 0, num_kernels,
                                      z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                      x_dims[2], x_dims[3],
                                      x->ga.data, x->ga.offset,
                                      w[0], w[1], s[0], s[1], p[0], p[1],
                                      inc_pad, sum_mode,
                                      (*z)->ga.data, (*z)->ga.offset);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: ave_pool2d_kernel %s.",
                       GpuKernel_error(&k_ave_pool2d_kernel, err));
          return 1;
        }
      }
    }
    else if (ndims == 3) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
      if (max_pool) {
        err = max_pool3d_kernel_scall(1, &num_kernels, 0, num_kernels,
                                      z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                      x_dims[2], x_dims[3], x_dims[4],
                                      x->ga.data, x->ga.offset, w[0], w[1], w[2], s[0], s[1], s[2],
                                      p[0], p[1], p[2], (*z)->ga.data, (*z)->ga.offset);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: max_pool3d_kernel %s.",
                       GpuKernel_error(&k_max_pool2d_kernel, err));
          return 1;
        }
      } else {
        err = ave_pool3d_kernel_scall(1, &num_kernels, 0, num_kernels,
                                      z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                      x_dims[2], x_dims[3], x_dims[4],
                                      x->ga.data, x->ga.offset,
                                      w[0], w[1], w[2], s[0], s[1], s[2],
                                      p[0], p[1], p[2],
                                      inc_pad, sum_mode,
                                      (*z)->ga.data, (*z)->ga.offset);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: ave_pool3d_kernel %s.",
                       GpuKernel_error(&k_ave_pool3d_kernel, err));
          return 1;
        }
      }
    }
  }
  return 0;
}
