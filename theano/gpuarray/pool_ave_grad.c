#section kernels

#kernel ave_pool2d_grad_kernel : size, size, size, size, size, size, size, *, *, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool2d_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size height,
   const ga_size width, const ga_size pooled_height, const ga_size pooled_width,
   GLOBAL_MEM const DTYPE_i0 *x, GLOBAL_MEM const DTYPE_i1 *gz,
   const ga_size kernel_h, const ga_size kernel_w, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_h, const ga_size pad_w, const ga_bool inc_pad, const ga_bool sum_mode,
   GLOBAL_MEM DTYPE_o0 *gx)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
    const ga_size w = index % width;
    const ga_size h = (index / width) % height;
    const ga_size c = (index / width / height) % channels;
    const ga_size n = (index / width / height / channels);
    const ga_size phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const ga_size phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const ga_size pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const ga_size pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    const ga_size offset = (n*channels + c) * pooled_height * pooled_width;
    const DTYPE_i1* gz_slice = gz + offset;
    DTYPE_o0 collector = 0;

    for (ga_size ph=phstart; ph < phend; ++ph) {
      for (ga_size pw=pwstart; pw < pwend; ++pw) {
        if (sum_mode) {
          collector += gz[ph*pooled_width + pw];
        } else {
          // figure out the pooling size
          const ga_size hstart = ph * stride_h - pad_h;
          const ga_size wstart = pw * stride_w - pad_w;
          const ga_size hend = min(hstart + kernel_h, height + pad_h);
          const ga_size wend = min(wstart + kernel_w, width + pad_w);
          const ga_size pool_size = (hend - hstart) * (wend - wstart);
          collector += gz_slice[ph*pooled_width + pw] / pool_size;
        }
      }
    }
    gx[index] = collector;
  }
}

#kernel ave_pool3d_grad_kernel : size, size, size, size, size, size, size, size, size, *, *, size, size, size, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void ave_pool3d_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size depth,
   const ga_size height, const ga_size width, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   GLOBAL_MEM const DTYPE_i0 *x, GLOBAL_MEM const DTYPE_i1 *gz,
   const ga_size kernel_d, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_d, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
   const ga_bool inc_pad, const ga_bool sum_mode, GLOBAL_MEM DTYPE_o0 *gx)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
    const ga_size w = index % width;
    const ga_size h = (index / width) % height;
    const ga_size d = (index / width / height) % depth;
    const ga_size c = (index / width / height / depth) % channels;
    const ga_size n = (index / width / height / depth / channels);
    const ga_size pdstart = (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    const ga_size pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
    const ga_size phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const ga_size phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const ga_size pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const ga_size pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    const ga_size offset = (n*channels + c) * pooled_depth * pooled_height * pooled_width;
    const DTYPE_i1* gz_slice = gz + offset;
    DTYPE_o0 collector = 0;

    for (ga_size pd=pdstart; pd < pdend; ++pd) {
      for (ga_size ph=phstart; ph < phend; ++ph) {
        for (ga_size pw=pwstart; pw < pwend; ++pw) {
          if (sum_mode) {
            collector += gz[ph*pooled_width + pw];
          } else {
            // figure out the pooling size
            const ga_size dstart = pd * stride_d - pad_d;
            const ga_size hstart = ph * stride_h - pad_h;
            const ga_size wstart = pw * stride_w - pad_w;
            const ga_size dend = min(dstart + kernel_d, depth + pad_d);
            const ga_size hend = min(hstart + kernel_h, height + pad_h);
            const ga_size wend = min(wstart + kernel_w, width + pad_w);
            const ga_size pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            collector += gz[ph*pooled_width + pw] / pool_size;
          }
        }
      }
    }
    gx[index] = collector;
  }
}

#section support_code_struct

int APPLY_SPECIFIC(ave_pool_grad)(PyGpuArrayObject *x,
                                  PyGpuArrayObject *gz,
                                  PyArrayObject *ws,
                                  PyArrayObject *stride,
                                  PyArrayObject *pad,
                                  PyGpuArrayObject **gx,
                                  PyGpuContextObject *ctx) {
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&gz->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuMaxPoolGrad: requires data to be C-contiguous");
      return 1;
    }
  size_t ndims = PyArray_DIM(ws, 0);
  if (PyGpuArray_NDIM(x) != ndims + 2
      || PyGpuArray_NDIM(gz) != ndims + 2)
    {
      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolGrad: rank error");
      return 1;
    }
  if (theano_prep_output(gx, PyGpuArray_NDIM(x), PyGpuArray_DIMS(x),
                         x->ga.typecode, GA_C_ORDER, ctx) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMaxPoolGrad: failed to allocate memory");
      return 1;
    }

  {
    // scope for running kernel
    size_t w[3];
    size_t s[3];
    size_t p[3];
    for(int i = 0; i < ndims; i++) {
      w[i] = *((npy_int64*)PyArray_GETPTR1(ws, i));
      s[i] = *((npy_int64*)PyArray_GETPTR1(stride, i));
      p[i] = *((npy_int64*)PyArray_GETPTR1(pad, i));
    }

    int err;
    const size_t* z_dims = PyGpuArray_DIMS(gz);
    const size_t* x_dims = PyGpuArray_DIMS(x);

    if (ndims == 2) {
      size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3];
      err = ave_pool2d_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                         x_dims[0], x_dims[1], x_dims[2], x_dims[3],
                                         z_dims[2], z_dims[3],
                                         x->ga.data, gz->ga.data,
                                         w[0], w[1], s[0], s[1], p[0], p[1],
                                         INC_PAD, SUM_MODE, (*gx)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuAveragePoolGrad: ave_pool2d_grad_kernel %s.",
                     GpuKernel_error(&k_ave_pool2d_grad_kernel, err));
        return 1;
      }
    } else if (ndims == 3) {
      size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4];
      err = ave_pool3d_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                         x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4],
                                         z_dims[2], z_dims[3], z_dims[4],
                                         x->ga.data, gz->ga.data,
                                         w[0], w[1], w[2], s[0], s[1], s[2],
                                         p[0], p[1], p[2], INC_PAD, SUM_MODE,
                                         (*gx)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuAveragePoolGrad: ave_pool3d_grad_kernel %s.",
                     GpuKernel_error(&k_ave_pool3d_grad_kernel, err));
        return 1;
      }
    }
  }
  return 0;
}
