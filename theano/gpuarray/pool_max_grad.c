#section kernels

#kernel max_pool2d_grad_kernel : size, size, size, size, size, size, size, *, *, *, size, size, size, size, size, size, * :

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ ga_double atomicAdd(ga_double *addr, ga_double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ ga_half atomicAdd(ga_half *addr, ga_half val) {
  ga_uint *base = (ga_uint *)((ga_size)addr & ~2);
  ga_uint old, assumed, sum, new_;
  old = *base;
  do {
    assumed = old;
    sum = __float2half_rn(__half2float(val) +
                          __half2float((ga_half)__byte_perm(
                              old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410)));
    new_ = __byte_perm(old, sum, ((ga_size)addr & 2) ? 0x5410 : 0x3254);
    old = atomicCAS(base, assumed, new_);
  } while (assumed != old);
  return (ga_half)__byte_perm(old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410);
}
#endif

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool2d_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size height,
   const ga_size width, const ga_size pooled_height, const ga_size pooled_width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, GLOBAL_MEM const DTYPE_INPUT_1 *z, GLOBAL_MEM const DTYPE_INPUT_2 *gz,
   const ga_size kernel_h, const ga_size kernel_w, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_h, const ga_size pad_w, GLOBAL_MEM DTYPE_OUTPUT_0 *gx)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
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
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    GLOBAL_MEM DTYPE_OUTPUT_0* gx_slice = gx + offset;
    ga_int maxidx = -1;

    bool maximum_found = false;
    for (ga_size h=hstart; h < hend && !maximum_found; ++h) {
      for (ga_size w=wstart; w < wend && !maximum_found; ++w) {
        if (z[index] == x_slice[h * width + w]) {
          maxidx = h * width + w;
          maximum_found = true;
        }
      }
    }
    if (maxidx != -1) {
      atomicAdd(&gx_slice[maxidx], gz[index]);
    }
  }
}

#kernel max_pool3d_grad_kernel : size, size, size, size, size, size, size, size, size, *, *, *, size, size, size, size, size, size, size, size, size, * :

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ ga_double atomicAdd(ga_double *addr, ga_double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ ga_half atomicAdd(ga_half *addr, ga_half val) {
  ga_uint *base = (ga_uint *)((ga_size)addr & ~2);
  ga_uint old, assumed, sum, new_;
  old = *base;
  do {
    assumed = old;
    sum = __float2half_rn(__half2float(val) +
                          __half2float((ga_half)__byte_perm(
                              old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410)));
    new_ = __byte_perm(old, sum, ((ga_size)addr & 2) ? 0x5410 : 0x3254);
    old = atomicCAS(base, assumed, new_);
  } while (assumed != old);
  return (ga_half)__byte_perm(old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410);
}
#endif

// (adopted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
KERNEL void max_pool3d_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size depth,
   const ga_size height, const ga_size width, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, GLOBAL_MEM const DTYPE_INPUT_1 *z, GLOBAL_MEM const DTYPE_INPUT_2 *gz,
   const ga_size kernel_d, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_d, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_OUTPUT_0 *gx)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
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
    GLOBAL_MEM const DTYPE_INPUT_0* x_slice = x + offset;
    GLOBAL_MEM DTYPE_OUTPUT_0* gx_slice = gx + offset;
    ga_int maxidx = -1;

    bool maximum_found = false;
    for (ga_size d=dstart; d < dend && !maximum_found; ++d) {
      for (ga_size h=hstart; h < hend && !maximum_found; ++h) {
        for (ga_size w=wstart; w < wend && !maximum_found; ++w) {
          if (z[index] == x_slice[(d*height + h)*width + w]) {
            maxidx = (d*height + h)*width + w;
            maximum_found = true;
          }
        }
      }
    }
    if (maxidx != -1) {
      atomicAdd(&gx_slice[maxidx], gz[index]);
    }
  }
}

#section support_code

static int theano_prep_zeros(PyGpuArrayObject **out, unsigned int nd,
                             const size_t *dims, int typecode, ga_order ord,
                             PyGpuContextObject *c) {
  if (*out != NULL && theano_size_check(*out, nd, dims, typecode)) {
    return 0;
  }

  Py_XDECREF(*out);
  *out = pygpu_zeros(nd, dims, typecode, ord, c, Py_None);
  return (*out == NULL) ? 1 : 0;
}

#section support_code_struct

int APPLY_SPECIFIC(max_pool_grad)(PyGpuArrayObject *x,
                                  PyGpuArrayObject *z,
                                  PyGpuArrayObject *gz,
                                  PyArrayObject *ws,
                                  PyArrayObject *stride,
                                  PyArrayObject *pad,
                                  PyGpuArrayObject **gx,
                                  PyGpuContextObject *ctx) {
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&z->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&gz->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuMaxPoolGrad: requires data to be C-contiguous");
      return 1;
    }
  size_t ndims = PyArray_DIM(ws, 0);
  if (PyGpuArray_NDIM(x) != ndims + 2
      || PyGpuArray_NDIM(z) != ndims + 2
      || PyGpuArray_NDIM(gz) != ndims + 2)
    {
      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolGrad: rank error");
      return 1;
    }
  if (theano_prep_zeros(gx, PyGpuArray_NDIM(x), PyGpuArray_DIMS(x),
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
    const size_t* x_dims = PyGpuArray_DIMS(x);
    const size_t* z_dims = PyGpuArray_DIMS(z);

    if (ndims == 2) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
      err = max_pool2d_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                         x_dims[0], x_dims[1], x_dims[2], x_dims[3],
                                         z_dims[2], z_dims[3],
                                         x->ga.data, z->ga.data, gz->ga.data,
                                         w[0], w[1], s[0], s[1], p[0], p[1],
                                         (*gx)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuMaxPoolGrad: max_pool2d_grad_kernel %s.",
                     GpuKernel_error(&k_max_pool2d_grad_kernel, err));
        return 1;
      }
    } else if (ndims == 3) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
      err = max_pool3d_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                         x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4],
                                         z_dims[2], z_dims[3], z_dims[4],
                                         x->ga.data, z->ga.data, gz->ga.data,
                                         w[0], w[1], w[2], s[0], s[1], s[2],
                                         p[0], p[1], p[2], (*gx)->ga.data);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuMaxPoolGrad: max_pool3d_grad_kernel %s.",
                     GpuKernel_error(&k_max_pool3d_grad_kernel, err));
        return 1;
      }
    }
  }
  return 0;
}
