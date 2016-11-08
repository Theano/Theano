#section kernels

#kernel max_pool2d_kernel : size, size, size, size, size, size, size, *, size, size, size, size, size, size, * :

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool2d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_h, const ga_size stride_w, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_o0 *z)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size c = (index / pooled_width / pooled_height) % channels;
    const ga_size n = (index / pooled_width / pooled_height / channels);
    ga_int hstart = static_cast<ga_int>(ph*stride_h) - static_cast<ga_int>(pad_h);
    const ga_size hend = min(hstart + kernel_h, height);
    ga_int wstart = static_cast<ga_int>(pw*stride_w) - static_cast<ga_int>(pad_w);
    const ga_size wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const ga_size offset = (n*channels + c) * height * width;
    const DTYPE_i0* x_slice = x + offset;
    // TODO: use DTYPE_o0_MAX for max value
    DTYPE_o0 maxval = -__int_as_float(0x7f800000);  // ieee754 float max

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

#kernel max_pool3d_kernel : size, size, size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool3d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   const ga_size depth, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, const ga_size kernel_d, const ga_size kernel_h,
   const ga_size kernel_w, const ga_size stride_d, const ga_size stride_h,
   const ga_size stride_w, const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_o0 *z)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size pd = (index / pooled_width / pooled_height) % pooled_depth;
    const ga_size c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const ga_size n = (index / pooled_width / pooled_height / pooled_depth / channels);
    ga_int dstart = static_cast<ga_int>(pd*stride_d) - static_cast<ga_int>(pad_d);
    const ga_size dend = min(dstart + kernel_d, depth);
    ga_int hstart = static_cast<ga_int>(ph*stride_h) - static_cast<ga_int>(pad_h);
    const ga_size hend = min(hstart + kernel_h, height);
    ga_int wstart = static_cast<ga_int>(pw*stride_w) - static_cast<ga_int>(pad_w);
    const ga_size wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const ga_size offset = (n*channels + c) * depth * height * width;
    const DTYPE_i0* x_slice = x + offset;
    // TODO: use DTYPE_o0_MAX for max value
    DTYPE_o0 maxval = -__int_as_float(0x7f800000);  // ieee754 float max

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

#kernel ave_pool2d_kernel : size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool2d_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_h, const ga_size stride_w, const ga_size pad_h, const ga_size pad_w,
   const ga_bool inc_pad, const ga_bool sum_mode,
   GLOBAL_MEM DTYPE_o0 *z)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size c = (index / pooled_width / pooled_height) % channels;
    const ga_size n = (index / pooled_width / pooled_height / channels);
    ga_int hstart = static_cast<ga_int>(ph*stride_h) - static_cast<ga_int>(pad_h);
    ga_size hend = min(hstart + kernel_h, height + pad_h);
    ga_int wstart = static_cast<ga_int>(pw*stride_w) - static_cast<ga_int>(pad_w);
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
    const DTYPE_i0* x_slice = x + offset;
    DTYPE_o0 collector = 0;

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

#kernel ave_pool3d_kernel : size, size, size, size, size, size, size, size, size, *, size, size, size, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool3d_kernel(const ga_size nthreads,
                              const ga_size num, const ga_size channels, const ga_size pooled_depth,
                              const ga_size pooled_height, const ga_size pooled_width,
                              const ga_size depth, const ga_size height, const ga_size width,
                              GLOBAL_MEM const DTYPE_i0 *x, const ga_size kernel_d, const ga_size kernel_h,
                              const ga_size kernel_w, const ga_size stride_d, const ga_size stride_h,
                              const ga_size stride_w, const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
                              const ga_bool inc_pad, const ga_bool sum_mode,
                              GLOBAL_MEM DTYPE_o0 *z)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads;
       index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size pd = (index / pooled_width / pooled_height) % pooled_depth;
    const ga_size c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const ga_size n = (index / pooled_width / pooled_height / pooled_depth / channels);
    ga_int dstart = static_cast<ga_int>(pd*stride_d) - static_cast<ga_int>(pad_d);
    ga_size dend = min(dstart + kernel_d, depth + pad_d);
    ga_int hstart = static_cast<ga_int>(ph*stride_h) - static_cast<ga_int>(pad_h);
    ga_size hend = min(hstart + kernel_h, height + pad_h);
    ga_int wstart = static_cast<ga_int>(pw*stride_w) - static_cast<ga_int>(pad_w);
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
    const DTYPE_i0* x_slice = x + offset;
    DTYPE_o0 collector = 0;

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

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int nkernels, const int nthreads) {
  return (nkernels + nthreads - 1) / nthreads;
}

// output shape for a given input padded shape, window shape and stride
#define OUTPUT_DIMS(in_dim, ws, st)                       \
  (IGNORE_BORDER ? (in_dim - ws)/st + 1 :                 \
   (st > ws ? (in_dim - 1)/st + 1 :                       \
    std::max<size_t>(0, (in_dim - 1 - ws + st)/st) + 1))

#section support_code_struct

int APPLY_SPECIFIC(pool)(PyGpuArrayObject *x,
                         PyArrayObject *ws,
                         PyArrayObject *stride,
                         PyArrayObject *pad,
                         PyGpuArrayObject **z,
                         PyGpuContextObject *ctx) {
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
  for (int i = 0; i < ndims; i++) {
    w[i] = *((npy_intp*)PyArray_GETPTR1(ws, i));
    s[i] = *((npy_intp*)PyArray_GETPTR1(stride, i));
    p[i] = *((npy_intp*)PyArray_GETPTR1(pad, i));
    z_dims[2 + i] = OUTPUT_DIMS(x_dims[2 + i] + 2*p[i], w[i], s[i]);
  }

  if (theano_prep_output(z, PyGpuArray_NDIM(x), z_dims,
                         x->ga.typecode, GA_C_ORDER, ctx) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuPool: failed to allocate memory");
      return 1;
    }
  {
    // scope for running kernel
    size_t max_threads_dim;
    int err;

    // get the max threads per blocks
    err = gpucontext_property(ctx->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim);
    if (err != GA_NO_ERROR){
      PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_threads_dims");
      return 1;
    }
    size_t threads_per_block = max_threads_dim;

    if (ndims == 2) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
      size_t n_blocks = GET_BLOCKS(num_kernels, threads_per_block);
      if (MAX_POOL) {
        err = max_pool2d_kernel_call(1, &n_blocks, &threads_per_block, 0,
                                     num_kernels,
                                     z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                     x_dims[2], x_dims[3],
                                     x->ga.data, w[0], w[1], s[0], s[1], p[0], p[1],
                                     (*z)->ga.data);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: max_pool2d_kernel %s.",
                       GpuKernel_error(&k_max_pool2d_kernel, err));
          return 1;
        }
      } else {
        err = ave_pool2d_kernel_call(1, &n_blocks, &threads_per_block, 0,
                                     num_kernels,
                                     z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                     x_dims[2], x_dims[3],
                                     x->ga.data, w[0], w[1], s[0], s[1], p[0], p[1],
                                     INC_PAD, SUM_MODE, (*z)->ga.data);
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
      size_t n_blocks = GET_BLOCKS(num_kernels, threads_per_block);
      if (MAX_POOL) {
        err = max_pool3d_kernel_call(1, &n_blocks, &threads_per_block, 0,
                                     num_kernels,
                                     z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                     x_dims[2], x_dims[3], x_dims[4],
                                     x->ga.data, w[0], w[1], w[2], s[0], s[1], s[2],
                                     p[0], p[1], p[2], (*z)->ga.data);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError,
                       "GpuPool: max_pool3d_kernel %s.",
                       GpuKernel_error(&k_max_pool2d_kernel, err));
          return 1;
        }
      } else {
        err = ave_pool3d_kernel_call(1, &n_blocks, &threads_per_block, 0,
                                     num_kernels,
                                     z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                     x_dims[2], x_dims[3], x_dims[4],
                                     x->ga.data, w[0], w[1], w[2], s[0], s[1], s[2],
                                     p[0], p[1], p[2],
                                     INC_PAD, SUM_MODE, (*z)->ga.data);
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