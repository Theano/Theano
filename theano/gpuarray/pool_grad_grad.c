#section kernels

#kernel pool_grad_grad_kernel : size, size, size, size, size, size, size, *, *, *, size, size, size, size, * :

KERNEL void pool_grad_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_i0 *x, GLOBAL_MEM const DTYPE_i1 *z, GLOBAL_MEM const DTYPE_i2 *gx,
   const ga_size kernel_h, const ga_size kernel_w, const ga_size stride_h, const ga_size stride_w,
   GLOBAL_MEM DTYPE_o0 *gz)
{
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
    const ga_size pw = index % pooled_width;
    const ga_size ph = (index / pooled_width) % pooled_height;
    const ga_size c = (index / pooled_width / pooled_height) % channels;
    const ga_size n = (index / pooled_width / pooled_height / channels);
    const ga_size hstart = ph*stride_h;
    const ga_size hend = min(hstart + kernel_h, height);
    const ga_size wstart = pw*stride_w;
    const ga_size wend = min(wstart + kernel_w, width);

    const ga_size offset = (n*channels + c) * height * width;

    const DTYPE_i0* x_slice = x + offset;
    const DTYPE_i2* gx_slice = gx + offset;
    DTYPE_o0 gradient = 0;

    for (ga_size h=hstart; h < hend; ++h) {
      for (ga_size w=wstart; w < wend; ++w) {
        // maximum in the region
        if (z[index] == x_slice[h * width + w]) {
          gradient += gx_slice[h * width + w];
        }
      }
    }
    gz[index] = gradient;
  }
}
#section support_code_struct

int APPLY_SPECIFIC(pool_grad_grad)(PyGpuArrayObject *x,
                                   PyGpuArrayObject *z,
                                   PyGpuArrayObject *gx,
                                   PyGpuArrayObject **gz,
                                   PyGpuContextObject *ctx) {
  if (PyGpuArray_NDIM(x) != 4
      || PyGpuArray_NDIM(z) != 4
      || PyGpuArray_NDIM(gx) != 4)
    {
      PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMaxGradGrad: rank error");
      return 1;
    }
  if (NULL == *gz || theano_size_check(*gz, 4, PyGpuArray_DIMS(z), z->ga.typecode))
    {
      Py_XDECREF(*gz);
      *gz = pygpu_zeros(4, PyGpuArray_DIMS(z),
                        z->ga.typecode, GA_C_ORDER,
                        ctx, Py_None);
      if (NULL == *gz)
        {
          PyErr_SetString(PyExc_RuntimeError,
                          "GpuDownsampleFactorMaxGradGrad: failed to allocate memory");
          return 1;
        }
    }
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&z->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&gx->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&(*gz)->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuDownsampleFactorMaxGradGrad: requires data to be C-contiguous");
      return 1;
    }
  {
    // scope for running kernel
    size_t max_threads_dim;
    int err;
    const size_t* z_dims = PyGpuArray_DIMS(z);
    const size_t* x_dims = PyGpuArray_DIMS(x);

    // Get the max threads per blocks
    err = gpucontext_property(ctx->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim);
    if (err != GA_NO_ERROR){
      PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_threads_dims");
      return 1;
    }

    size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
    size_t threads_per_block = max_threads_dim;
    size_t n_blocks = (num_kernels + threads_per_block - 1) / threads_per_block;
    err = pool_grad_grad_kernel_call(1, &n_blocks, &threads_per_block, 0,
                                     num_kernels,
                                     z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                     x_dims[2], x_dims[3],
                                     x->ga.data, z->ga.data, gx->ga.data,
                                     DS0, DS1, ST0, ST1,
                                     (*gz)->ga.data);
    if (err != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDownsampleFactorMaxGradGrad: %s.",
                   GpuKernel_error(&k_pool_grad_grad_kernel, err));
      return 1;
    }
  }
  return 0;
}
