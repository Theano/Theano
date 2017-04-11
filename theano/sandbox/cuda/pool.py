from __future__ import absolute_import, print_function, division

from theano import Apply
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.tensor.basic import as_tensor_variable


class GpuPool(GpuOp):
    """
    Implement the max and average pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'average':
            mode = 'average_inc_pad'
        self.mode = mode
        assert mode in ('max', 'sum', 'average_inc_pad', 'average_exc_pad')
        assert self.ndim in [2, 3]

    def make_node(self, inp, ws, stride, pad):
        inp = as_cuda_ndarray_variable(inp)
        assert (inp.type.ndim == self.ndim + 2)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, ws, stride, pad], [inp.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_headers(self):
        return ['<float.h>']

    def c_code(self, node, nodename, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        max_mode = int(self.mode == 'max')
        nd = self.ndim
        return """
        if (%(x)s->nd != 2 + %(nd)s)
        {
          PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMax: rank error");
          %(fail)s;
        }
        #define OUTPUT_DIMS(in_dim, ds, st) \
          (%(ignore_border)s ? (in_dim - ds)/st + 1 : \
            (st > ds ? (in_dim - 1)/st + 1 : \
                      max(0, (in_dim - 1 - ds + st)/st) + 1))

        int z_dims[5]; // avoid warning if use 2 + nd
        int w[3];
        int s[3];
        int p[3];
        z_dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
        z_dims[1] = CudaNdarray_HOST_DIMS(%(x)s)[1];
        for (int i = 0; i < %(nd)s; i++) {
          w[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
          s[i] = *((npy_intp*)PyArray_GETPTR1(%(stride)s, i));
          p[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
          z_dims[2 + i] = OUTPUT_DIMS(CudaNdarray_HOST_DIMS(%(x)s)[2 + i] + 2*p[i], w[i], s[i]);
        }

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != z_dims[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != z_dims[1])
            || (CudaNdarray_HOST_DIMS(%(z)s)[2] != z_dims[2])
            || (CudaNdarray_HOST_DIMS(%(z)s)[3] != z_dims[3]))
        {
          Py_XDECREF(%(z)s);
          %(z)s = (CudaNdarray*)CudaNdarray_New();
          if ((NULL == %(z)s)
              || CudaNdarray_alloc_contiguous(%(z)s, 2 + %(nd)s, z_dims))
          {
            Py_XDECREF(%(z)s);
            %(z)s = NULL;
            %(fail)s;
          }
        }
        {
          // scope for running kernel
          if (%(nd)s == 2) {
            size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
            if (%(max_mode)s) {
              kMaxPool2d_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                w[0], w[1], s[0], s[1], p[0], p[1]);
              CNDA_THREAD_SYNC;
              cudaError_t err = cudaGetLastError();
              if (cudaSuccess != err)
              {
                PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                             "kMaxPool2d_%(nodename)s", cudaGetErrorString(err));
                %(fail)s;
              }
            }
            else {
              kAvePool2d_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                w[0], w[1], s[0], s[1], p[0], p[1]);
              CNDA_THREAD_SYNC;
              cudaError_t err = cudaGetLastError();
              if (cudaSuccess != err)
              {
                PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                             "kMaxPool2d_%(nodename)s", cudaGetErrorString(err));
                %(fail)s;
              }
            }
          }
          else if (%(nd)s == 3) {
            size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
            if (%(max_mode)s) {
              kMaxPool3d_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
                num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[4],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_HOST_STRIDES(%(x)s)[4],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_HOST_STRIDES(%(z)s)[4],
                w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2]);
              CNDA_THREAD_SYNC;
              cudaError_t err = cudaGetLastError();
              if (cudaSuccess != err)
              {
                  PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                              "kMaxPool3d_%(nodename)s", cudaGetErrorString(err));
                  %(fail)s;
              }
            }
            else {
              kAvePool3d_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
                num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[4],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_HOST_STRIDES(%(x)s)[4],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_HOST_STRIDES(%(z)s)[4],
                w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2]);
              CNDA_THREAD_SYNC;
              cudaError_t err = cudaGetLastError();
              if (cudaSuccess != err)
              {
                  PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                              "kAvePool3d_%(nodename)s", cudaGetErrorString(err));
                  %(fail)s;
              }
            }
          }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        inc_pad = int(self.mode != 'average_exc_pad')
        sum_mode = int(self.mode == 'sum')
        return """
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CUDA_NUM_THREADS = 1024;
#else
const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
__global__ void kMaxPool2d_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_height, int pooled_width, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3,
  float * z, int zS0, int zS1, int zS2, int zS3,
  int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int c = (index / pooled_width / pooled_height) %% channels;
    int n = (index / pooled_width / pooled_height / channels);
    int hstart = ph*stride_h - pad_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw*stride_w - pad_w;
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const float* x_slice = x + n*xS0 + c*xS1;
    float maxval = -FLT_MAX;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (x_slice[h*xS2 + w*xS3] > maxval) {
          maxval = x_slice[h*xS2 + w*xS3];
        }
      }
    }
    z[n*zS0 + c*zS1 + ph*zS2 + pw*zS3] = maxval;
  }
}

__global__ void kMaxPool3d_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_depth, int pooled_height, int pooled_width,
  int depth, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3, int xS4,
  float * z, int zS0, int zS1, int zS2, int zS3, int zS4,
  int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h,
  int stride_w, int pad_d, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int pd = (index / pooled_width / pooled_height) %% pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) %% channels;
    int n = (index / pooled_width / pooled_height / pooled_depth / channels);
    int dstart = pd*stride_d - pad_d;
    int dend = min(dstart + kernel_d, depth);
    int hstart = ph*stride_h - pad_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw*stride_w - pad_w;
    int wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const float* x_slice = x + n*xS0 + c*xS1;
    float maxval = -FLT_MAX;

    for (int d=dstart; d < dend; ++d) {
      for (int h=hstart; h < hend; ++h) {
        for (int w=wstart; w < wend; ++w) {
          if (x_slice[d*xS2 + h*xS3 + w*xS4] > maxval) {
            maxval = x_slice[d*xS2 + h*xS3 + w*xS4];
          }
        }
      }
    }
    z[n*zS0 + c*zS1 + pd*zS2 + ph*zS3 + pw*zS4] = maxval;
  }
}

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
__global__ void kAvePool2d_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_height, int pooled_width, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3,
  float * z, int zS0, int zS1, int zS2, int zS3,
  int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int c = (index / pooled_width / pooled_height) %% channels;
    int n = (index / pooled_width / pooled_height / channels);
    int hstart = %(inc_pad)s ? ph*stride_h - pad_h : ph*stride_h;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wstart = %(inc_pad)s ? pw*stride_w - pad_w : pw*stride_w;
    int wend = min(wstart + kernel_w, width + pad_w);
    int pool_size;
    if (%(inc_pad)s) {
        pool_size = (hend - hstart) * (wend - wstart);
    }
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    if (!%(inc_pad)s) {
        pool_size = (hend - hstart) * (wend - wstart);
    }

    const float* x_slice = x + n*xS0 + c*xS1;
    float collector = 0;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
          collector += x_slice[h*xS2 + w*xS3];
      }
    }
    z[n*zS0 + c*zS1 + ph*zS2 + pw*zS3] = %(sum_mode)s ? collector : collector/pool_size;
  }
}

__global__ void kAvePool3d_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_depth, int pooled_height, int pooled_width,
  int depth, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3, int xS4,
  float * z, int zS0, int zS1, int zS2, int zS3, int zS4,
  int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h,
  int stride_w, int pad_d, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int pd = (index / pooled_width / pooled_height) %% pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) %% channels;
    int n = (index / pooled_width / pooled_height / pooled_depth / channels);
    int dstart = %(inc_pad)s ? pd*stride_d - pad_d : pd*stride_d;
    int dend = min(dstart + kernel_d, depth + pad_d);
    int hstart = ph*stride_h - pad_h;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wstart = pw*stride_w - pad_w;
    int wend = min(wstart + kernel_w, width + pad_w);
    int pool_size;
    if (%(inc_pad)s) {
        pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    }
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dend = min(dend, depth);
    hend = min(hend, height);
    wend = min(wend, width);
    if (!%(inc_pad)s) {
        pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    }

    const float* x_slice = x + n*xS0 + c*xS1;
    float collector = 0;

    for (int d=dstart; d < dend; ++d) {
      for (int h=hstart; h < hend; ++h) {
        for (int w=wstart; w < wend; ++w) {
          collector += x_slice[d*xS2 + h*xS3 + w*xS4];
        }
      }
    }
    z[n*zS0 + c*zS1 + pd*zS2 + ph*zS3 + pw*zS4] = %(sum_mode)s ? collector : collector/pool_size;
  }
}
        """ % locals()


class GpuMaxPoolGrad(GpuOp):
    """
    Implement the grad of max pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        assert self.mode == 'max'
        assert self.ndim in [2, 3]

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        inp = as_cuda_ndarray_variable(inp)
        assert (inp.type.ndim == self.ndim + 2)
        out = as_cuda_ndarray_variable(out)
        assert (out.type.ndim == self.ndim + 2)
        out_grad = as_cuda_ndarray_variable(out_grad)
        assert (out_grad.type.ndim == self.ndim + 2)

        assert (out_grad.type.ndim == inp.type.ndim)
        assert (inp.type.ndim == out.type.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, nodename, inp, out, sub):
        x, z, gz, ws, stride, pad = inp
        gx, = out
        nd = self.ndim
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2 + %(nd)s
            || %(z)s->nd != 2 + %(nd)s
            || %(gz)s->nd != 2 + %(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMaxGrad: rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (CudaNdarray_HOST_DIMS(%(gx)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[2] !=
                CudaNdarray_HOST_DIMS(%(x)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[3] !=
                CudaNdarray_HOST_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gx)s)
                || CudaNdarray_alloc_contiguous(%(gx)s, 2 + %(nd)s,
                                                CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            // scope for running kernel
            size_t w[3];
            size_t s[3];
            size_t p[3];
            for(int i = 0; i < %(nd)s; i++) {
                w[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
                s[i] = *((npy_intp*)PyArray_GETPTR1(%(stride)s, i));
                p[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
            }

            const int* x_dims = CudaNdarray_HOST_DIMS(%(x)s);

            if (%(nd)s == 2) {
                size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3];
                kMaxPool2dGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                    num_kernels, x_dims[0], x_dims[1], x_dims[2], x_dims[3],
                    CudaNdarray_HOST_DIMS(%(z)s)[2],
                    CudaNdarray_HOST_DIMS(%(z)s)[3],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_HOST_STRIDES(%(x)s)[2],
                    CudaNdarray_HOST_STRIDES(%(x)s)[3],
                    CudaNdarray_DEV_DATA(%(z)s),
                    CudaNdarray_HOST_STRIDES(%(z)s)[0],
                    CudaNdarray_HOST_STRIDES(%(z)s)[1],
                    CudaNdarray_HOST_STRIDES(%(z)s)[2],
                    CudaNdarray_HOST_STRIDES(%(z)s)[3],
                    CudaNdarray_DEV_DATA(%(gz)s),
                    CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                    CudaNdarray_DEV_DATA(%(gx)s),
                    CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                    w[0], w[1], s[0], s[1], p[0], p[1]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool2dGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }
            }
            else if (%(nd)s == 3) {
                size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4];
                kMaxPool3dGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                    num_kernels, x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4],
                    CudaNdarray_HOST_DIMS(%(z)s)[2],
                    CudaNdarray_HOST_DIMS(%(z)s)[3],
                    CudaNdarray_HOST_DIMS(%(z)s)[4],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_HOST_STRIDES(%(x)s)[2],
                    CudaNdarray_HOST_STRIDES(%(x)s)[3],
                    CudaNdarray_HOST_STRIDES(%(x)s)[4],
                    CudaNdarray_DEV_DATA(%(z)s),
                    CudaNdarray_HOST_STRIDES(%(z)s)[0],
                    CudaNdarray_HOST_STRIDES(%(z)s)[1],
                    CudaNdarray_HOST_STRIDES(%(z)s)[2],
                    CudaNdarray_HOST_STRIDES(%(z)s)[3],
                    CudaNdarray_HOST_STRIDES(%(z)s)[4],
                    CudaNdarray_DEV_DATA(%(gz)s),
                    CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[4],
                    CudaNdarray_DEV_DATA(%(gx)s),
                    CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[4],
                    w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool3dGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }

            }
        }""" % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)

        return """
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CUDA_NUM_THREADS = 1024;
#else
const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
__global__ void kMaxPool2dGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int height, int width, int pooled_height, int pooled_width,
  const float * x, int xS0, int xS1, int xS2, int xS3,
  const float * z, int zS0, int zS1, int zS2, int zS3,
  const float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
  float *gx, int gxS0, int gxS1, int gxS2, int gxS3,
  int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
{

  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index %% width;
    int h = (index / width) %% height;
    int c = (index / width / height) %% channels;
    int n = (index / width / height / channels);
    const int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    const float curr_x = x[n*xS0 + c*xS1 + h*xS2 + w*xS3];
    float gradient = 0;
    for (int ph=phstart; ph < phend; ++ph) {
      for (int pw=pwstart; pw < pwend; ++pw) {
        float curr_max = z[n*zS0 + c*zS1 + ph*zS2 + pw*zS3];
        if (curr_max == curr_x) {
          gradient += gz[n*gzS0 + c*gzS1 + ph*gzS2 + pw*gzS3];
        }
      }
    }
    gx[n*gxS0 + c*gxS1 + h*gxS2 + w*gxS3] = gradient;
  }
}

__global__ void kMaxPool3dGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int depth, int height, int width,
  int pooled_depth, int pooled_height, int pooled_width,
  const float * x, int xS0, int xS1, int xS2, int xS3, int xS4,
  const float * z, int zS0, int zS1, int zS2, int zS3, int zS4,
  const float * gz, int gzS0, int gzS1, int gzS2, int gzS3, int gzS4,
  float *gx, int gxS0, int gxS1, int gxS2, int gxS3, int gxS4,
  int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h,
  int stride_w, int pad_d, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index %% width;
    int h = (index / width) %% height;
    int d = (index / width / height) %% depth;
    int c = (index / width / height / depth) %% channels;
    int n = (index / width / height / depth / channels);
    const int pdstart = (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    const int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
    const int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    const float curr_x = x[n*xS0 + c*xS1 + d*xS2 + h*xS3 + w*xS4];
    float gradient = 0;
    for (int pd=pdstart; pd < pdend; ++pd) {
      for (int ph=phstart; ph < phend; ++ph) {
        for (int pw=pwstart; pw < pwend; ++pw) {
          float curr_max = z[n*zS0 + c*zS1 + pd*zS2 + ph*zS3 + pw*zS4];
          if (curr_max == curr_x) {
            gradient += gz[n*gzS0 + c*gzS1 + pd*gzS2 + ph*gzS3 + pw*gzS4];
          }
        }
      }
    }
    gx[n*gxS0 + c*gxS1 + d*gxS2 + h*gxS3 + w*gxS4] = gradient;
  }
}
        """ % locals()


class GpuAveragePoolGrad(GpuOp):
    """
    Implement the grad of average pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'average':
            mode = 'average_inc_pad'
        self.mode = mode
        assert self.mode in ('sum', 'average_inc_pad', 'average_exc_pad')
        assert self.ndim in [2, 3]

    def make_node(self, inp, out_grad, ws, stride, pad):
        inp = as_cuda_ndarray_variable(inp)
        assert (inp.type.ndim == self.ndim + 2)
        out_grad = as_cuda_ndarray_variable(out_grad)
        assert (out_grad.type.ndim == self.ndim + 2)

        assert (out_grad.type.ndim == inp.type.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out_grad, ws, stride, pad], [inp.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, nodename, inp, out, sub):
        x, gz, ws, stride, pad = inp
        gx, = out
        nd = self.ndim
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2 + %(nd)s
            || %(gz)s->nd != 2 + %(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMaxGrad: rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (CudaNdarray_HOST_DIMS(%(gx)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[2] !=
                CudaNdarray_HOST_DIMS(%(x)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[3] !=
                CudaNdarray_HOST_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gx)s)
                || CudaNdarray_alloc_contiguous(%(gx)s, 2 + %(nd)s,
                                                CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            // scope for running kernel
            size_t w[3];
            size_t s[3];
            size_t p[3];
            for(int i = 0; i < %(nd)s; i++) {
                w[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
                s[i] = *((npy_intp*)PyArray_GETPTR1(%(stride)s, i));
                p[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
            }

            const int* x_dims = CudaNdarray_HOST_DIMS(%(x)s);

            if (%(nd)s == 2) {
                size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3];
                kAvePool2dGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                    num_kernels, x_dims[0], x_dims[1], x_dims[2], x_dims[3],
                    CudaNdarray_HOST_DIMS(%(gz)s)[2],
                    CudaNdarray_HOST_DIMS(%(gz)s)[3],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_HOST_STRIDES(%(x)s)[2],
                    CudaNdarray_HOST_STRIDES(%(x)s)[3],
                    CudaNdarray_DEV_DATA(%(gz)s),
                    CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                    CudaNdarray_DEV_DATA(%(gx)s),
                    CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                    w[0], w[1], s[0], s[1], p[0], p[1]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool2dGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }
            }
            else if (%(nd)s == 3) {
                size_t num_kernels = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3] * x_dims[4];
                kAvePool3dGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                    num_kernels, x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4],
                    CudaNdarray_HOST_DIMS(%(gz)s)[2],
                    CudaNdarray_HOST_DIMS(%(gz)s)[3],
                    CudaNdarray_HOST_DIMS(%(gz)s)[4],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_HOST_STRIDES(%(x)s)[2],
                    CudaNdarray_HOST_STRIDES(%(x)s)[3],
                    CudaNdarray_HOST_STRIDES(%(x)s)[4],
                    CudaNdarray_DEV_DATA(%(gz)s),
                    CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                    CudaNdarray_HOST_STRIDES(%(gz)s)[4],
                    CudaNdarray_DEV_DATA(%(gx)s),
                    CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                    CudaNdarray_HOST_STRIDES(%(gx)s)[4],
                    w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool3dGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }

            }
        }""" % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        sum_mode = int(self.mode == 'sum')

        return """
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CUDA_NUM_THREADS = 1024;
#else
const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)
__global__ void kAvePool2dGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int height, int width, int pooled_height, int pooled_width,
  const float * x, int xS0, int xS1, int xS2, int xS3,
  const float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
  float *gx, int gxS0, int gxS1, int gxS2, int gxS3,
  int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
{

  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index %% width;
    int h = (index / width) %% height;
    int c = (index / width / height) %% channels;
    int n = (index / width / height / channels);
    const int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    float collector = 0;
    for (int ph=phstart; ph < phend; ++ph) {
      for (int pw=pwstart; pw < pwend; ++pw) {
        if (%(sum_mode)s) {
          collector += gz[n*gzS0 + c*gzS1 + ph*gzS2 + pw*gzS3];
        } else {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = min(hstart + kernel_h, height + pad_h);
          int wend = min(wstart + kernel_w, width + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          collector += gz[n*gzS0 + c*gzS1 + ph*gzS2 + pw*gzS3] / pool_size;
        }
      }
    }
    gx[n*gxS0 + c*gxS1 + h*gxS2 + w*gxS3] = collector;
  }
}

__global__ void kAvePool3dGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int depth, int height, int width,
  int pooled_depth, int pooled_height, int pooled_width,
  const float * x, int xS0, int xS1, int xS2, int xS3, int xS4,
  const float * gz, int gzS0, int gzS1, int gzS2, int gzS3, int gzS4,
  float *gx, int gxS0, int gxS1, int gxS2, int gxS3, int gxS4,
  int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h,
  int stride_w, int pad_d, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index %% width;
    int h = (index / width) %% height;
    int d = (index / width / height) %% depth;
    int c = (index / width / height / depth) %% channels;
    int n = (index / width / height / depth / channels);
    const int pdstart = (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    const int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
    const int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

    float collector = 0;
    for (int pd=pdstart; pd < pdend; ++pd) {
      for (int ph=phstart; ph < phend; ++ph) {
        for (int pw=pwstart; pw < pwend; ++pw) {
          if (%(sum_mode)s) {
            collector += gz[n*gzS0 + c*gzS1 + pd*gzS2 + ph*gzS3 + pw*gzS4];
          } else {
            int dstart = pd * stride_d - pad_d;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int dend = min(dstart + kernel_d, depth + pad_d);
            int hend = min(hstart + kernel_h, height + pad_h);
            int wend = min(wstart + kernel_w, width + pad_w);
            int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            collector += gz[n*gzS0 + c*gzS1 + pd*gzS2 + ph*gzS3 + pw*gzS4] / pool_size;
          }
        }
      }
    }
    gx[n*gxS0 + c*gxS1 + d*gxS2 + h*gxS3 + w*gxS4] = collector;
  }
}
        """ % locals()


class GpuMaxPoolGradGrad(GpuOp):
    """
    Implement the grad of max pooling grad on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        assert self.mode == 'max'
        assert self.ndim in [2, 3]

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        inp = as_cuda_ndarray_variable(inp)
        assert (inp.type.ndim == self.ndim + 2)
        out = as_cuda_ndarray_variable(out)
        assert (out.type.ndim == self.ndim + 2)
        out_grad = as_cuda_ndarray_variable(out_grad)
        assert (out_grad.type.ndim == self.ndim + 2)

        assert (out_grad.type.ndim == inp.type.ndim)
        assert (inp.type.ndim == out.type.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, nodename, inp, out, sub):
        x, z, gx, ws, stride, pad = inp
        gz, = out
        nd = self.ndim
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2 + %(nd)s
            || %(z)s->nd != 2 + %(nd)s
            || %(gx)s->nd != 2 + %(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "GpuDownsampleFactorMaxGradGrad: rank error");
            %(fail)s;
        }
        if ((NULL == %(gz)s)
            || (CudaNdarray_HOST_DIMS(%(gz)s)[0] !=
                CudaNdarray_HOST_DIMS(%(z)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[1] !=
                CudaNdarray_HOST_DIMS(%(z)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[2] !=
                CudaNdarray_HOST_DIMS(%(z)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gz)s)[3] !=
                CudaNdarray_HOST_DIMS(%(z)s)[3]))
        {
            Py_XDECREF(%(gz)s);
            %(gz)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gz)s)
                || CudaNdarray_alloc_contiguous(%(gz)s, 2 + %(nd)s,
                                                CudaNdarray_HOST_DIMS(%(z)s)))
            {
                Py_XDECREF(%(gz)s);
                %(gz)s = NULL;
                %(fail)s;
            }
        }
        {
            // scope for running kernel
            size_t w[3];
            size_t s[3];
            size_t p[3];
            for (int i = 0; i < %(nd)s; i++) {
              w[i] = *((npy_intp*)PyArray_GETPTR1(%(ws)s, i));
              s[i] = *((npy_intp*)PyArray_GETPTR1(%(stride)s, i));
              p[i] = *((npy_intp*)PyArray_GETPTR1(%(pad)s, i));
            }

            const int* z_dims = CudaNdarray_HOST_DIMS(%(z)s);

            if (%(nd)s == 2) {
                size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3];
                kMaxPool2dGradGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                  num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3],
                  CudaNdarray_HOST_DIMS(%(x)s)[2],
                  CudaNdarray_HOST_DIMS(%(x)s)[3],
                  CudaNdarray_DEV_DATA(%(x)s),
                  CudaNdarray_HOST_STRIDES(%(x)s)[0],
                  CudaNdarray_HOST_STRIDES(%(x)s)[1],
                  CudaNdarray_HOST_STRIDES(%(x)s)[2],
                  CudaNdarray_HOST_STRIDES(%(x)s)[3],
                  CudaNdarray_DEV_DATA(%(z)s),
                  CudaNdarray_HOST_STRIDES(%(z)s)[0],
                  CudaNdarray_HOST_STRIDES(%(z)s)[1],
                  CudaNdarray_HOST_STRIDES(%(z)s)[2],
                  CudaNdarray_HOST_STRIDES(%(z)s)[3],
                  CudaNdarray_DEV_DATA(%(gz)s),
                  CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                  CudaNdarray_DEV_DATA(%(gx)s),
                  CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                  w[0], w[1], s[0], s[1], p[0], p[1]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool2dGradGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }
            } else if (%(nd)s == 3) {
                size_t num_kernels = z_dims[0] * z_dims[1] * z_dims[2] * z_dims[3] * z_dims[4];
                kMaxPool3dGradGrad_%(nodename)s<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                  num_kernels, z_dims[0], z_dims[1], z_dims[2], z_dims[3], z_dims[4],
                  CudaNdarray_HOST_DIMS(%(x)s)[2],
                  CudaNdarray_HOST_DIMS(%(x)s)[3],
                  CudaNdarray_HOST_DIMS(%(x)s)[4],
                  CudaNdarray_DEV_DATA(%(x)s),
                  CudaNdarray_HOST_STRIDES(%(x)s)[0],
                  CudaNdarray_HOST_STRIDES(%(x)s)[1],
                  CudaNdarray_HOST_STRIDES(%(x)s)[2],
                  CudaNdarray_HOST_STRIDES(%(x)s)[3],
                  CudaNdarray_HOST_STRIDES(%(x)s)[4],
                  CudaNdarray_DEV_DATA(%(z)s),
                  CudaNdarray_HOST_STRIDES(%(z)s)[0],
                  CudaNdarray_HOST_STRIDES(%(z)s)[1],
                  CudaNdarray_HOST_STRIDES(%(z)s)[2],
                  CudaNdarray_HOST_STRIDES(%(z)s)[3],
                  CudaNdarray_HOST_STRIDES(%(z)s)[4],
                  CudaNdarray_DEV_DATA(%(gz)s),
                  CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                  CudaNdarray_HOST_STRIDES(%(gz)s)[4],
                  CudaNdarray_DEV_DATA(%(gx)s),
                  CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[3],
                  CudaNdarray_HOST_STRIDES(%(gx)s)[4],
                  w[0], w[1], w[2], s[0], s[1], s[2], p[0], p[1], p[2]);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,  "Cuda error: %%s: %%s.",
                                "kMaxPool3dGradGrad_%(nodename)s", cudaGetErrorString(err));
                    %(fail)s;
                }
            }

        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CUDA_NUM_THREADS = 1024;
#else
const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void kMaxPool2dGradGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_height, int pooled_width, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3,
  const float * z, int zS0, int zS1, int zS2, int zS3,
  float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
  const float *gx, int gxS0, int gxS1, int gxS2, int gxS3,
  int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int c = (index / pooled_width / pooled_height) %% channels;
    int n = (index / pooled_width / pooled_height / channels);
    int hstart = ph*stride_h - pad_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw*stride_w - pad_w;
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const float* x_slice = x + n*xS0 + c*xS1;
    const float* gx_slice = gx + n*gxS0 + c*gxS1;
    // maximum in the region
    const float curr_max = z[n*zS0 + c*zS1 + ph*zS2 + pw*zS3];
    float gradient = 0;

    for (int h=hstart; h < hend; ++h) {
      for (int w=wstart; w < wend; ++w) {
        // essentially: z[n,c,ph,pw] == x[n,c,h,w]
        if (curr_max == x_slice[h*xS2 + w*xS3]) {
          gradient += gx_slice[h*gxS2 + w*gxS3];
        }
      }
    }
    gz[n*gzS0 + c*gzS1 + ph*gzS2 + pw*gzS3] = gradient;
  }
}

__global__ void kMaxPool3dGradGrad_%(nodename)s(size_t nthreads,
  int num, int channels, int pooled_depth, int pooled_height, int pooled_width,
  int depth, int height, int width,
  const float * x, int xS0, int xS1, int xS2, int xS3, int xS4,
  const float * z, int zS0, int zS1, int zS2, int zS3, int zS4,
  float * gz, int gzS0, int gzS1, int gzS2, int gzS3, int gzS4,
  const float *gx, int gxS0, int gxS1, int gxS2, int gxS3, int gxS4,
  int kernel_d, int kernel_h, int kernel_w, int stride_d, int stride_h,
  int stride_w, int pad_d, int pad_h, int pad_w)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index %% pooled_width;
    int ph = (index / pooled_width) %% pooled_height;
    int pd = (index / pooled_width / pooled_height) %% pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) %% channels;
    int n = (index / pooled_width / pooled_height / pooled_depth / channels);
    int dstart = pd*stride_d - pad_d;
    int dend = min(dstart + kernel_d, depth);
    int hstart = ph*stride_h - pad_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw*stride_w - pad_w;
    int wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const float* x_slice = x + n*xS0 + c*xS1;
    const float* gx_slice = gx + n*gxS0 + c*gxS1;
    // maximum in the region
    const float curr_max = z[n*zS0 + c*zS1 + pd*zS2 + ph*zS3 + pw*zS4];
    float gradient = 0;

    for (int d=dstart; d < dend; ++d) {
      for (int h=hstart; h < hend; ++h) {
        for (int w=wstart; w < wend; ++w) {
          // essentially: z[n,c,pd,ph,pw] == x[n,c,d,h,w]
          if (curr_max == x_slice[d*xS2 + h*xS3 + w*xS4]) {
            gradient += gx_slice[d*gxS2 + h*gxS3 + w*gxS4];
          }
        }
      }
    }
    gz[n*gzS0 + c*gzS1 + pd*gzS2 + ph*gzS3 + pw*gzS4] = gradient;
  }
}
        """ % locals()
