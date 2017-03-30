#section kernels

#kernel max_pool2d_rop_kernel : size, size, size, size, size, size, size, *, *, size, size, size, size, size, size, * :

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool2d_rop_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, GLOBAL_MEM const DTYPE_i1 *ex,
   const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_h, const ga_size pad_w,
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
    const DTYPE_i1* ex_slice = ex + offset;
    DTYPE_o0 maxval = x_slice[hstart*width + wstart];
    DTYPE_o0 collector = ex_slice[hstart*width + wstart];

    for (ga_size h=hstart; h < hend; ++h) {
      for (ga_size w=wstart; w < wend; ++w) {
        // maximum in the region
        if (x_slice[h*width + w] > maxval) {
          maxval = x_slice[h*width + w];
          collector = ex_slice[h*width + w];
        }
      }
    }
    z[index] = collector;
  }
}

#kernel max_pool3d_rop_kernel : size, size, size, size, size, size, size, size, size, *, *, size, size, size, size, size, size, size, size, size, * :

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool3d_rop_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   const ga_size depth, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, GLOBAL_MEM const DTYPE_i1 *ex,
   const ga_size kernel_d, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_d, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
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
    const DTYPE_i1* ex_slice = ex + offset;
    DTYPE_o0 maxval = x_slice[(dstart*height + hstart)*width + wstart];
    DTYPE_o0 collector = ex_slice[(dstart*height + hstart)*width + wstart];

    for (ga_size d=dstart; d < dend; ++d) {
      for (ga_size h=hstart; h < hend; ++h) {
        for (ga_size w=wstart; w < wend; ++w) {
          // maximum in the region
          if (x_slice[(d*height + h)*width + w] > maxval) {
            maxval = x_slice[(d*height + h)*width + w];
            collector = ex_slice[(d*height + h)*width + w];
          }
        }
      }
    }
    z[index] = collector;
  }
}


#section support_code

// output shape for a given input padded shape, window shape and stride
#define OUTPUT_DIMS(in_dim, ws, st)                       \
  (IGNORE_BORDER ? (in_dim - ws)/st + 1 :                 \
   (st > ws ? (in_dim - 1)/st + 1 :                       \
    std::max<ssize_t>(0, (in_dim - 1 - ws + st)/st) + 1))

#section support_code_struct

int APPLY_SPECIFIC(max_pool_rop)(PyGpuArrayObject *x,
                                 PyGpuArrayObject *ex,
                                 PyArrayObject *ws,
                                 PyArrayObject *stride,
                                 PyArrayObject *pad,
                                 PyGpuArrayObject **z,
                                 PyGpuContextObject *ctx) {
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga) || !GpuArray_IS_C_CONTIGUOUS(&ex->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuMaxPoolRop: requires data to be C-contiguous");
      return 1;
    }
  size_t ndims = PyArray_DIM(ws, 0);
  if (PyGpuArray_NDIM(x) != ndims + 2 || PyGpuArray_NDIM(ex) != ndims + 2)
    {
      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolRop: rank error");
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
    z_dims[2 + i] = OUTPUT_DIMS(x_dims[2 + i] + 2*p[i], w[i], s[i]);
    if (p[i] > 0) {
      nonzero_padding = 1;
    }
  }
  if (!IGNORE_BORDER && nonzero_padding) {
    PyErr_SetString(PyExc_ValueError,
                    "GpuMaxPoolRop: padding works only with ignore_border=True");
    return 1;
  }

  if (theano_prep_output(z, PyGpuArray_NDIM(ex), z_dims,
                         ex->ga.typecode, GA_C_ORDER, ctx) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuMaxPoolRop: failed to allocate memory");
      return 1;
    }
  {
    // scope for running kernel
    int err;

    if (ndims == 2) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
      err = max_pool2d_rop_kernel_scall(1, &num_kernels, 0, num_kernels,
                                        z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                        x_dims[2], x_dims[3],
                                        x->ga.data, ex->ga.data,
                                        w[0], w[1], s[0], s[1], p[0], p[1],
                                        (*z)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuMaxPoolRop: max_pool2d_rop_kernel %s.",
                     GpuKernel_error(&k_max_pool2d_rop_kernel, err));
        return 1;
      }
    }
    else if (ndims == 3) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
      err = max_pool3d_rop_kernel_scall(1, &num_kernels, 0, num_kernels,
                                        z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                        x_dims[2], x_dims[3], x_dims[4],
                                        x->ga.data, ex->ga.data,
                                        w[0], w[1], w[2], s[0], s[1], s[2],
                                        p[0], p[1], p[2], (*z)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuMaxPoolRop: max_pool3d_rop_kernel %s.",
                     GpuKernel_error(&k_max_pool2d_rop_kernel, err));
        return 1;
      }
    }
  }
  return 0;
}
