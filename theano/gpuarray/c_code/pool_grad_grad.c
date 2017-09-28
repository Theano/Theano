#section kernels

#kernel max_pool2d_grad_grad_kernel : size, size, size, size, size, size, size, *, size, *, size, *, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

KERNEL void max_pool2d_grad_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_height,
   const ga_size pooled_width, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, GLOBAL_MEM const DTYPE_INPUT_1 *z, const ga_size z_off, GLOBAL_MEM const DTYPE_INPUT_2 *gx, const ga_size gx_off,
   const ga_size kernel_h, const ga_size kernel_w, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_OUTPUT_0 *gz, const ga_size gz_off)
{
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_INPUT_1 *)(((GLOBAL_MEM char *)z) + z_off);
  gx = (GLOBAL_MEM DTYPE_INPUT_2 *)(((GLOBAL_MEM char *)gx) + gx_off);
  gz = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)gz) + gz_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
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
    GLOBAL_MEM const DTYPE_INPUT_2* gx_slice = gx + offset;
    DTYPE_OUTPUT_0 gradient = 0;

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

#kernel max_pool3d_grad_grad_kernel : size, size, size, size, size, size, size, size, size, *, size, *, size, *, size, size, size, size, size, size, size, size, size, size, *, size :
#include "cluda.h"

KERNEL void max_pool3d_grad_grad_kernel(const ga_size nthreads,
   const ga_size num, const ga_size channels, const ga_size pooled_depth,
   const ga_size pooled_height, const ga_size pooled_width,
   const ga_size depth, const ga_size height, const ga_size width,
   GLOBAL_MEM const DTYPE_INPUT_0 *x, const ga_size x_off, GLOBAL_MEM const DTYPE_INPUT_1 *z, const ga_size z_off, GLOBAL_MEM const DTYPE_INPUT_2 *gx, const ga_size gx_off,
   const ga_size kernel_d, const ga_size kernel_h, const ga_size kernel_w,
   const ga_size stride_d, const ga_size stride_h, const ga_size stride_w,
   const ga_size pad_d, const ga_size pad_h, const ga_size pad_w,
   GLOBAL_MEM DTYPE_OUTPUT_0 *gz, const ga_size gz_off)
{
  x = (GLOBAL_MEM DTYPE_INPUT_0 *)(((GLOBAL_MEM char *)x) + x_off);
  z = (GLOBAL_MEM DTYPE_INPUT_1 *)(((GLOBAL_MEM char *)z) + z_off);
  gx = (GLOBAL_MEM DTYPE_INPUT_2 *)(((GLOBAL_MEM char *)gx) + gx_off);
  gz = (GLOBAL_MEM DTYPE_OUTPUT_0 *)(((GLOBAL_MEM char *)gz) + gz_off);
  // grid stride looping
  for (ga_size index = GID_0 * LDIM_0 + LID_0;
       index < nthreads; index += LDIM_0 * GDIM_0) {
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
    GLOBAL_MEM const DTYPE_INPUT_2* gx_slice = gx + offset;
    DTYPE_OUTPUT_0 gradient = 0;

    for (ga_size d=dstart; d < dend; ++d) {
      for (ga_size h=hstart; h < hend; ++h) {
        for (ga_size w=wstart; w < wend; ++w) {
          // maximum in the region
          if (z[index] == x_slice[(d * height + h) * width + w]) {
            gradient += gx_slice[(d * height + h)* width + w];
          }
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
                                   PyArrayObject *ws,
                                   PyArrayObject *stride,
                                   PyArrayObject *pad,
                                   PyGpuArrayObject **gz,
                                   PyGpuContextObject *ctx) {
  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&z->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&gx->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuPoolingGradGrad: requires data to be C-contiguous");
      return 1;
    }
  size_t ndims = PyArray_DIM(ws, 0);
  if (PyGpuArray_NDIM(x) != ndims + 2
      || PyGpuArray_NDIM(z) != ndims + 2
      || PyGpuArray_NDIM(gx) != ndims + 2)
    {
      PyErr_SetString(PyExc_ValueError, "GpuPoolingGradGrad: rank error");
      return 1;
    }
  if (theano_prep_output(gz, PyGpuArray_NDIM(z), PyGpuArray_DIMS(z),
                         z->ga.typecode, GA_C_ORDER, ctx) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuPoolingGradGrad: failed to allocate memory");
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
    const size_t* z_dims = PyGpuArray_DIMS(z);
    const size_t* x_dims = PyGpuArray_DIMS(x);

    if (ndims == 2) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
      err = max_pool2d_grad_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                              z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                                              x_dims[2], x_dims[3],
                                              x->ga.data, x->ga.offset,
                                              z->ga.data, z->ga.offset,
                                              gx->ga.data, gx->ga.offset,
                                              w[0], w[1], s[0], s[1], p[0], p[1],
                                              (*gz)->ga.data, (*gz)->ga.offset);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuPoolingGradGrad: max_pool2d_grad_grad_kernel %s.",
                     GpuKernel_error(&k_max_pool2d_grad_grad_kernel, err));
        return 1;
      }
    }
    else if (ndims == 3) {
      size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
      err = max_pool3d_grad_grad_kernel_scall(1, &num_kernels, 0, num_kernels,
                                              z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                                              x_dims[2], x_dims[3], x_dims[4],
                                              x->ga.data, x->ga.offset,
                                              z->ga.data, z->ga.offset,
                                              gx->ga.data, gx->ga.offset,
                                              w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2],
                                              (*gz)->ga.data, (*gz)->ga.offset);
      if (err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError,
                     "GpuPoolingGradGrad: max_pool3d_grad_grad_kernel %s.",
                     GpuKernel_error(&k_max_pool3d_grad_grad_kernel, err));
        return 1;
      }
    }
  }
  return 0;
}
