import numpy
import theano
from theano import Apply, tensor, scalar, Constant
from theano.tensor import DimShuffle, discrete_dtypes

from theano.gradient import grad_undefined, grad_not_implemented

from theano.sandbox.cuda import cuda_available, GpuOp, GpuElemwise

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray, opt, GpuFromHost,
                                     HostFromGpu, host_from_gpu,
                                     GpuDimShuffle)

class SparseBlockGemvSS(GpuOp):
    """
    This op computes the dot product of specified pieces of vectors
    and matrices, returning pieces of vectors.

    It computes something like this for each j:

      o[j] = sum_over_i(dot(W[i, j], h[i])) + o[j]

    The i and j are taken from the inputIdx and outputIdx lists
    respectively.

    This should not be directly called since the interface is subject
    to change without notice.  Use the sparse_block_dot_SS() function
    for a stable interface.
    """
    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __str__(self):
        return "SparseBlockGemvSS%s" % ("{inplace}" if self.inplace else "")

    def make_node(self, o, W, h, inputIdx, outputIdx):
        o = basic_ops.as_cuda_ndarray_variable(o)
        W = basic_ops.as_cuda_ndarray_variable(W)
        h = basic_ops.as_cuda_ndarray_variable(h)
        assert o.ndim == 3
        assert W.ndim == 4
        assert h.ndim == 3
        assert inputIdx.ndim == 2
        assert outputIdx.ndim == 2

        assert inputIdx.type.dtype in discrete_dtypes
        assert outputIdx.type.dtype in discrete_dtypes

        return Apply(self, [o, W, h, inputIdx, outputIdx],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_support_code(self):
        return """
        __global__ void
        SparseBlockGemv_fill_lists(
int maxi, int maxj,
const float **inp_list,
float **out_list,
const float **W_list,
const float *W, int W_str_0, int W_str_1,
const float *h, int h_str_0, int h_str_1,
float *out, int o_str_0, int o_str_1,
const npy_intp *iIdx, int iI_str_0,
const npy_intp *oIdx, int oI_str_0
        ) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int b = blockIdx.z;
  if (i >= maxi || j >= maxj) return;
  int p = i + j * maxi + b * maxi * maxj;
  inp_list[p] = &h[b * h_str_0 + i * h_str_1];
  out_list[p] = &out[b * o_str_0 + j * o_str_1];
  W_list[p] = &W[iIdx[b*iI_str_0+i] * W_str_0 +
                 oIdx[b*oI_str_0+j] * W_str_1];
}

__global__ void _sgemvBH_N_a1_b1_small(const float *A[], int lda,
                                       const float *x[], int incx,
                                       float *y[], int incy,
                                       int b, int m, int n) {
  for (int p = blockIdx.y * blockDim.y + threadIdx.y; p < b;
       p += gridDim.y * blockDim.y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
      float yi = 0.0f;
      const float *Ap = A[p] + i;
      const float *xp = x[p];
      #pragma unroll 32
      for (int j = 0; j < n; j++) {
        yi += Ap[0] * xp[0];
        Ap += lda;
        xp += incx;
      }
      atomicAdd(&y[p][i*incy], yi);
    }
  }
}

__global__ void _sgemvBH_T_a1_b1_small(const float *A[], int lda,
                                       const float *x[], int incx,
                                       float *y[], int incy,
                                       int b, int m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int p = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || p >= b) return;
  float yi = 0.0f;
  const float *Ap = A[p] + i * lda;
  const float *xp = x[p];
  # pragma unroll 32
  for (int j = 0; j < n; j++) {
    yi += Ap[j] * xp[0];
    xp += incx;
  }
  atomicAdd(&y[p][i*incy], yi);
}

static cublasStatus_t SgemvBatched(cublasHandle_t handle,
                                   cublasOperation_t trans,
                                   int m, int n,
                                   const float *alpha,
                                   const float *A[], int lda,
                                   const float *x[], int incx,
                                   const float *beta,
                                   float *y[], int incy, int batchCount) {
  dim3 block(m, batchCount, 1);
  dim3 grid(1, 1, 1);
  cublasPointerMode_t mode;
  cudaError_t err;
  if (m < 512) {
    block.x = 32;
    if (batchCount > 16)
      block.y = 16;
    else
      block.y = batchCount;
  } else {
    block.x = 512;
    block.y = 1;
  }
  grid.x = (m + block.x - 1) / block.x;
  grid.y = (batchCount + block.y - 1) / block.y;
  if (grid.x * grid.y > 65535) {
    grid.y = (65535 / grid.x);
  }
  cublasGetPointerMode(handle, &mode);
  if (mode != CUBLAS_POINTER_MODE_HOST)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (*alpha != 1.0 || *beta != 1.0)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_N)
    _sgemvBH_N_a1_b1_small<<<grid, block>>>(A, lda, x, incx,
                                            y, incy,
                                            batchCount, m, n);
  else if (trans == CUBLAS_OP_T)
    _sgemvBH_T_a1_b1_small<<<grid, block>>>(A, lda, x, incx,
                                            y, incy,
                                            batchCount, m, n);
  else
    return CUBLAS_STATUS_INVALID_VALUE;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return CUBLAS_STATUS_SUCCESS;
}

static int SparseBlockGemv_copy(PyArrayObject *a, npy_intp *b) {
  cudaError_t err;
  PyArrayObject *aa = (PyArrayObject *)PyArray_Cast(a, NPY_INTP);
  if (aa == NULL) { return -1; }
  err = cudaMemcpyAsync(b, PyArray_DATA(aa), PyArray_NBYTES(aa),
                        cudaMemcpyHostToDevice);
  Py_DECREF(aa);
  if (err != cudaSuccess) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot copy index data to GPU");
    return -1;
  }
  return 0;
}
"""

    def c_support_code_apply(self, node, nodename):
        return """
        /* Statics are initialized with 0 */
        static const float **%(n)s_inp_list;
        static float **%(n)s_out_list;
        static const float **%(n)s_W_list;
        static size_t %(n)s_list_len;
        static npy_intp *%(n)s_iIdx;
        static size_t %(n)s_iIdx_len;
        static npy_intp *%(n)s_oIdx;
        static size_t %(n)s_oIdx_len;

        static int %(n)s_prep(int b, int i, int j, int outsize) {
          int s = b*i*j;
          if (%(n)s_list_len < s) {
            cudaFree(%(n)s_inp_list);
            cudaFree(%(n)s_out_list);
            cudaFree(%(n)s_W_list);
            if (cudaMalloc(&%(n)s_inp_list, s*sizeof(float *)) != cudaSuccess) return -1;
            if (cudaMalloc(&%(n)s_out_list, s*sizeof(float *)) != cudaSuccess) return -1;
            if (cudaMalloc(&%(n)s_W_list, s*sizeof(float *)) != cudaSuccess) return -1;
            %(n)s_list_len = s;
          }
          if (%(n)s_iIdx_len < b*i) {
            cudaFree(%(n)s_iIdx);
            if (cudaMalloc(&%(n)s_iIdx, b*i*sizeof(npy_intp)) != cudaSuccess) return -1;
            %(n)s_iIdx_len = b*i;
          }
          if (%(n)s_oIdx_len < b*j) {
            cudaFree(%(n)s_oIdx);
            if (cudaMalloc(&%(n)s_oIdx, b*j*sizeof(npy_intp)) != cudaSuccess) return -1;
            %(n)s_oIdx_len = b*j;
          }
          return 0;
        }
        """ % dict(n=nodename)

    def c_code(self, node, nodename, inputs, outputs, sub):
        o, W, h, inputIdx, outputIdx = inputs
        out = outputs[0]

        if self.inplace:
            res = """
Py_XDECREF(%(out)s);
%(out)s = %(o)s;
Py_INCREF(%(out)s);
""" % dict(out=out, o=o)
        else:
            res = """
if (CudaNdarray_prep_output(&%(out)s, 3, CudaNdarray_HOST_DIMS(%(o)s)))
{
  PyErr_SetString(PyExc_RuntimeError, "Cannot allocate output");
  %(fail)s
}
if (CudaNdarray_CopyFromCudaNdarray(%(out)s, %(o)s)) {
  PyErr_SetString(PyExc_RuntimeError, "Cannot copy data to output");
  %(fail)s
}
""" % dict(out=out, o=o, fail=sub['fail'])

        return res + """
        if (%(name)s_prep(CudaNdarray_HOST_DIMS(%(o)s)[0],
                          CudaNdarray_HOST_DIMS(%(h)s)[1],
                          CudaNdarray_HOST_DIMS(%(o)s)[1],
                          CudaNdarray_HOST_DIMS(%(o)s)[2]) == -1) {
          PyErr_SetString(PyExc_RuntimeError,
                          "Could not allocate working memory.");
          %(fail)s
        }
        if (SparseBlockGemv_copy(%(inputIdx)s, %(name)s_iIdx) == -1)
          { %(fail)s }
        if (SparseBlockGemv_copy(%(outputIdx)s, %(name)s_oIdx) == -1)
          { %(fail)s }
        { /* Prepare lists for the batch */
          dim3 block;
          dim3 grid;
          block.x = CudaNdarray_HOST_DIMS(%(h)s)[1];
          block.y = CudaNdarray_HOST_DIMS(%(o)s)[1];
          grid.z = CudaNdarray_HOST_DIMS(%(o)s)[0]; // batch size
          if (block.x > 32) {
            grid.x = (block.x + 31) / 32;
            block.x = 32;
          }
          if (block.x * block.y > 512) {
            grid.y = (block.y + 15) / 16;
            block.y = 16;
          }
          SparseBlockGemv_fill_lists<<<grid, block>>>(
CudaNdarray_HOST_DIMS(%(h)s)[1], CudaNdarray_HOST_DIMS(%(o)s)[1],
%(name)s_inp_list,
%(name)s_out_list,
%(name)s_W_list,
CudaNdarray_DEV_DATA(%(W)s),
CudaNdarray_HOST_STRIDES(%(W)s)[0], CudaNdarray_HOST_STRIDES(%(W)s)[1],
CudaNdarray_DEV_DATA(%(h)s),
CudaNdarray_HOST_STRIDES(%(h)s)[0], CudaNdarray_HOST_STRIDES(%(h)s)[1],
CudaNdarray_DEV_DATA(%(out)s),
CudaNdarray_HOST_STRIDES(%(out)s)[0], CudaNdarray_HOST_STRIDES(%(out)s)[1],
%(name)s_iIdx, PyArray_DIM(%(inputIdx)s, 1),
%(name)s_oIdx, PyArray_DIM(%(outputIdx)s, 1));
        }
        { /* Run SgemvBatched */
          float alpha = 1.0f;
          float beta = 1.0f;
          cublasStatus_t err;
          cublasOperation_t transA = CUBLAS_OP_N;
          int lda = CudaNdarray_HOST_STRIDES(%(W)s)[2];
          if (lda == 1) {
            transA = CUBLAS_OP_T;
            lda = CudaNdarray_HOST_STRIDES(%(W)s)[3];
          }
          if (lda == 0) lda = 1;
          err = SgemvBatched(handle, transA,
                             CudaNdarray_HOST_DIMS(%(o)s)[2],
                             CudaNdarray_HOST_DIMS(%(h)s)[2], &alpha,
                             %(name)s_W_list, lda, %(name)s_inp_list,
                             CudaNdarray_HOST_STRIDES(%(h)s)[2],
                             &beta, %(name)s_out_list,
                             CudaNdarray_HOST_STRIDES(%(o)s)[2],
                             CudaNdarray_HOST_DIMS(%(o)s)[1] *
                             CudaNdarray_HOST_DIMS(%(h)s)[1] *
                             CudaNdarray_HOST_DIMS(%(o)s)[0]);
          if (err != CUBLAS_STATUS_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError, "SgemvBatched failed");
            %(fail)s
          }
        }
        // And we're done!
        """ % dict(out=out, h=h, o=o, inputIdx=inputIdx, outputIdx=outputIdx,
                   W=W, fail=sub['fail'], name=nodename)

    def c_code_cache_version(self):
        return (10,)

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        Wgrad = sparse_block_outer_ss(W.zeros_like(),
                                      h, go, inputIdx, outputIdx)
        hgrad = sparse_block_gemv_ss(h.zeros_like(),
                                     W.dimshuffle((1, 0, 3, 2)),
                                     go,
                                     outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


sparse_block_gemv_ss = SparseBlockGemvSS(False)
sparse_block_gemv_ss_inplace = SparseBlockGemvSS(True)


class SparseBlockOuterSS(GpuOp):
    """
    This computes the outer product of two sets of pieces of vectors
    updating a full matrix with the results.

    It computes something like this:

      o[i, j] = (alpha * outer(x[i], y[j])) + o[i, j]

    The i and j are taken from the xIdx and yIdx lists respectively.

    This op should not be called directly since its interface is
    subject to change without notice.  It is involved in the gradient
    of SparseBlockGemvSS.
    """
    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __str__(self):
        return "SparseBlockOuterSS%s" % ("{inplace}" if self.inplace else "")

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None):
        one = tensor.constant(numpy.asarray(1.0, dtype='float32'))
        o = basic_ops.as_cuda_ndarray_variable(o)
        x = basic_ops.as_cuda_ndarray_variable(x)
        y = basic_ops.as_cuda_ndarray_variable(y)
        if alpha is None:
            alpha = one
        return Apply(self, [o, x, y, xIdx, yIdx, alpha],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_support_code(self):
        return """
__global__ void
SparseBlockOuter_fill_lists(
int maxi, int maxj,
const float **x_list,
const float **y_list,
float **out_list,
const float *x, int x_str_0, int x_str_1,
const float *y, int y_str_0, int y_str_1,
float *out, int o_str_0, int o_str_1,
const npy_intp *xIdx, int xI_str_0,
const npy_intp *yIdx, int yI_str_0
) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int b = blockIdx.z;
  if (i >= maxi || j >= maxj) return;
  int p = i + j * maxi + b * maxi * maxj;
  x_list[p] = &x[b * x_str_0 + i * x_str_1];
  y_list[p] = &y[b * y_str_0 + j * y_str_1];
  out_list[p] = &out[xIdx[b * xI_str_0 + i] * o_str_0 +
                     yIdx[b * yI_str_0 + j] * o_str_1];
}

/* This is tuned for smaller sizes (< 512) since it's what we get normally */
__global__ void _sgerBH_gen_small(const float *x[], int incx,
                                  const float *y[], int incy,
                                  float alpha,
                                  float *A[], int lda,
                                  int b, int m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
  for (int p = blockIdx.z; p < b; p += gridDim.z) {
    atomicAdd(&A[p][j * lda + i],
              alpha * x[p][i * incx] * y[p][j * incy]);
  }
}

static cublasStatus_t SgerBatched(cublasHandle_t handle, int m, int n,
                                  const float *alpha,
                                  const float *x[], int incx,
                                  const float *y[], int incy,
                                  float *A[], int lda,
                                  int batchCount) {
  dim3 block(m, n, 1);
  dim3 grid(1, 1, batchCount);
  cublasPointerMode_t mode;
  cudaError_t err;
  if (incx == 1) {
    if (block.x > 32) {
      grid.x = (block.x + 31)/32;
      block.x = 32;
    }
    if (block.x * block.y > 512) {
      grid.y = (block.y + 15) / 16;
      block.y = 16;
    }
  } else {
    if (block.y > 32) {
      grid.y = (block.y + 31)/32;
      block.y = 32;
    }
    if (block.x * block.y > 512) {
      grid.x = (block.x + 15) / 16;
      block.x = 16;
    }
  }
  if (grid.x * grid.y * grid.z > 65535) {
    if (grid.x * grid.y > 65535)
      return CUBLAS_STATUS_INVALID_VALUE;
    grid.z = (65535 / (grid.x * grid.y));
  }
  cublasGetPointerMode(handle, &mode);
  if (mode == CUBLAS_POINTER_MODE_HOST) {
    _sgerBH_gen_small<<<grid, block>>>(x, incx, y, incy, *alpha, A, lda,
                                       batchCount, m, n);
  } else {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  err = cudaGetLastError();
  if (err != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return CUBLAS_STATUS_SUCCESS;
}

static int SparseBlockOuter_copy(PyArrayObject *a, npy_intp *b) {
  cudaError_t err;
  PyArrayObject *aa = (PyArrayObject *)PyArray_Cast(a, NPY_INTP);
  if (aa == NULL) { return -1; }
  err = cudaMemcpyAsync(b, PyArray_DATA(aa), PyArray_NBYTES(aa),
                        cudaMemcpyHostToDevice);
  Py_DECREF(aa);
  if (err != cudaSuccess) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot copy index data to GPU");
    return -1;
  }
  return 0;
}
"""

    def c_support_code_apply(self, node, name):
        return """
/* statics are initialized with 0 */
static float **%(n)s_out_list;
static const float **%(n)s_x_list;
static const float **%(n)s_y_list;
static size_t %(n)s_list_len;
static npy_intp *%(n)s_xIdx;
static size_t %(n)s_xIdx_len;
static npy_intp *%(n)s_yIdx;
static size_t %(n)s_yIdx_len;

static int %(n)s_prep(int b, int i, int j) {
  int s = b*i*j;
  if (%(n)s_list_len < s) {
    cudaFree(%(n)s_x_list);
    cudaFree(%(n)s_y_list);
    cudaFree(%(n)s_out_list);
    if (cudaMalloc(&%(n)s_x_list, s*sizeof(float *)) != cudaSuccess) return -1;
    if (cudaMalloc(&%(n)s_y_list, s*sizeof(float *)) != cudaSuccess) return -1;
    if (cudaMalloc(&%(n)s_out_list, s*sizeof(float *)) != cudaSuccess) return -1;
    %(n)s_list_len = s;
  }
  if (%(n)s_xIdx_len < b*i) {
    cudaFree(%(n)s_xIdx);
    if (cudaMalloc(&%(n)s_xIdx, b*i*sizeof(npy_intp)) != cudaSuccess)
      return -1;
    %(n)s_xIdx_len = b*i;
  }
  if (%(n)s_yIdx_len < b*j) {
    cudaFree(%(n)s_yIdx);
    if (cudaMalloc(&%(n)s_yIdx, b*j*sizeof(npy_intp)) != cudaSuccess)
      return -1;
    %(n)s_yIdx_len = b*j;
  }
  return 0;
}
""" % dict(n=name)

    def c_code(self, node, name, inputs, outputs, sub):
        o, x, y, xIdx, yIdx, alpha = inputs
        out = outputs[0]
        if self.inplace:
            res = """
Py_XDECREF(%(out)s);
%(out)s = %(o)s;
Py_INCREF(%(out)s);
""" % dict(out=out, o=o)
        else:
            res = """
if (CudaNdarray_prep_output(&%(out)s, 4, CudaNdarray_HOST_DIMS(%(o)s)))
{
  PyErr_SetString(PyExc_RuntimeError, "Cannot allocate output");
  %(fail)s
}
if (CudaNdarray_CopyFromCudaNdarray(%(out)s, %(o)s)) {
  PyErr_SetString(PyExc_RuntimeError, "Cannot copy data to output");
  %(fail)s
}
""" % dict(out=out, o=o, fail=sub['fail'])

        return res + """
if (%(name)s_prep(CudaNdarray_HOST_DIMS(%(x)s)[0],
                  CudaNdarray_HOST_DIMS(%(x)s)[1],
                  CudaNdarray_HOST_DIMS(%(y)s)[1]) == -1) {
  PyErr_SetString(PyExc_RuntimeError, "Could not allocate working memory.");
  %(fail)s
}
if (SparseBlockOuter_copy(%(xIdx)s, %(name)s_xIdx) == -1)
 { %(fail)s }
if (SparseBlockOuter_copy(%(yIdx)s, %(name)s_yIdx) == -1)
 { %(fail)s }
{
  dim3 block;
  dim3 grid;
  block.x = CudaNdarray_HOST_DIMS(%(x)s)[1];
  block.y = CudaNdarray_HOST_DIMS(%(y)s)[1];
  grid.z = CudaNdarray_HOST_DIMS(%(x)s)[0];
  if (block.x > 32) {
    grid.x = (block.x + 31) / 32;
    block.x = 32;
  }
  if (block.x * block.y > 512) {
    grid.y = (block.y + 15) / 16;
    block.y = 16;
  }
  SparseBlockOuter_fill_lists<<<grid, block>>>(
CudaNdarray_HOST_DIMS(%(x)s)[1], CudaNdarray_HOST_DIMS(%(y)s)[1],
%(name)s_x_list,
%(name)s_y_list,
%(name)s_out_list,
CudaNdarray_DEV_DATA(%(x)s), CudaNdarray_HOST_STRIDES(%(x)s)[0], CudaNdarray_HOST_STRIDES(%(x)s)[1],
CudaNdarray_DEV_DATA(%(y)s), CudaNdarray_HOST_STRIDES(%(y)s)[0], CudaNdarray_HOST_STRIDES(%(y)s)[1],
CudaNdarray_DEV_DATA(%(out)s),
CudaNdarray_HOST_STRIDES(%(out)s)[0], CudaNdarray_HOST_STRIDES(%(out)s)[1],
%(name)s_xIdx, PyArray_DIM(%(xIdx)s, 1),
%(name)s_yIdx, PyArray_DIM(%(yIdx)s, 1));
}
{
  cublasStatus_t err;
  int str_y = CudaNdarray_HOST_STRIDES(%(y)s)[2];
  if (str_y == 0) str_y = 1;
  int str_x = CudaNdarray_HOST_STRIDES(%(x)s)[2];
  if (str_x == 0) str_x = 1;
  int str_out = CudaNdarray_HOST_STRIDES(%(out)s)[2];
  if (str_out == 0) str_out = 1;
  err = SgerBatched(handle,
    CudaNdarray_HOST_DIMS(%(y)s)[2], CudaNdarray_HOST_DIMS(%(x)s)[2],
    (float *)PyArray_GETPTR1(%(alpha)s, 0), %(name)s_y_list, str_y,
    %(name)s_x_list, str_x,
    %(name)s_out_list, str_out,
    CudaNdarray_HOST_DIMS(%(x)s)[0] *
    CudaNdarray_HOST_DIMS(%(x)s)[1] *
    CudaNdarray_HOST_DIMS(%(y)s)[1]);
  if (err != CUBLAS_STATUS_SUCCESS) {
    if (err == CUBLAS_STATUS_INVALID_VALUE) {
       /* The current code would be much too slow for sizes any larger
          than this. */
       PyErr_SetString(PyExc_ValueError,
                       "SgerBatched failed, probably because you have your "
                       "block size too big. The current limit is 65535 for "
                       "iSize * oSize.");
    } else {
      PyErr_SetString(PyExc_RuntimeError, "SgerBatched failed");
    }
    %(fail)s
  }
}""" % dict(x=x, y=y, out=out, xIdx=xIdx, yIdx=yIdx, name=name,
            alpha=alpha, fail=sub['fail'])

    def c_code_cache_version(self):
        return (9,)


sparse_block_outer_ss = SparseBlockOuterSS(False)
sparse_block_outer_ss_inplace = SparseBlockOuterSS(True)


if cuda_available:
    @opt.register_opt()
    @opt.local_optimizer([sparse_block_gemv_ss], inplace=True)
    def local_inplace_blocksparse_gemv(node):
        if node.op == sparse_block_gemv_ss:
            return [sparse_block_gemv_ss_inplace(*node.inputs)]

    @opt.register_opt()
    @opt.local_optimizer([sparse_block_outer_ss], inplace=True)
    def local_inplace_blocksparse_outer(node):
        if node.op == sparse_block_outer_ss:
            return [sparse_block_outer_ss_inplace(*node.inputs)]

    def grab_ger(v):
        # We need to do some digging because apparently the
        # cut_transfers op does not run before us.
        if v.owner is not None:
            if isinstance(v.owner.op, SparseBlockOuterSS):
                return v.owner
            elif (isinstance(v.owner.op, GpuFromHost) and
                  v.owner.inputs[0].owner is not None and
                  isinstance(v.owner.inputs[0].owner.op, HostFromGpu)):
                return grab_ger(v.owner.inputs[0].owner.inputs[0])
            else:
                return None

    # Should be run before elemwise fusion
    @opt.register_opt()
    @opt.local_optimizer([GpuElemwise])
    def local_merge_blocksparse_alpha(node):
        """
GpuElemwise{mul}(lr, SparseBlockOuterSS) -> SparseBlockOuterSS(..., alpha=lr)
        """
        def grab_lr(v):
            if v.owner is not None:
                n = v.owner
                if (isinstance(n.op, GpuDimShuffle) and
                      n.op.new_order == ('x', 'x', 'x', 'x')):
                    return host_from_gpu(n.inputs[0])
                elif (isinstance(n.op, DimShuffle) and
                      n.op.new_order == ('x', 'x', 'x', 'x')):
                    return n.inputs[0]
                elif isinstance(n.op, GpuFromHost):
                      return grab_lr(n.inputs[0])
                else:
                    return None
            else:
                if (isinstance(v, Constant) and
                    v.broadcastable == (True, True, True, True)):
                    return v.dimshuffle(())

        if (isinstance(node.op, GpuElemwise) and
            node.op.scalar_op == scalar.mul and
            node.nin == 2):
            ger = grab_ger(node.inputs[0])
            if ger is None:
                ger = grab_ger(node.inputs[1])
                lr = grab_lr(node.inputs[0])
            else:
                lr = grab_lr(node.inputs[1])
            if lr is None or ger is None:
                return None
            alpha = lr * ger.inputs[5]
            return [sparse_block_outer_ss(*(ger.inputs[:5] + [alpha]))]

    @opt.register_opt()
    @opt.local_optimizer([GpuElemwise])
    def local_merge_blocksparse_output(node):
        if (isinstance(node.op, GpuElemwise) and
            (node.op.scalar_op == scalar.sub or
             node.op.scalar_op == scalar.add) and
            node.nin == 2):
            ger = grab_ger(node.inputs[0])
            W = node.inputs[1]
            if ger is None:
                ger = grab_ger(node.inputs[1])
                W = node.inputs[0]
            if ger is None:
                return None
            if node.op.scalar_op == scalar.sub:
                alpha = -ger.inputs[5]
                W = W - ger.inputs[0]
            else:
                alpha = ger.inputs[5]
                W = W + ger.inputs[0]
            return [sparse_block_outer_ss(*([W] + ger.inputs[1:5] +
                                            [alpha]))]


def sparse_block_dot_SS(W, h, inputIdx, b, outputIdx):
    """
    Compute the dot product (plus bias) of the specified pieces of vectors
    and matrices.

    Parameters
    ----------
    var: shape, comment
    W: (iBlocks, oBlocks, iSize, oSize), weight matrix
    h: (batch, iWin, iSize), input from lower layer (sparse)
    inputIdx: (batch, iWin), indexes of the input blocks
    b: (oBlocks, oSize), bias vector
    outputIdx: (batch, oWin), indexes of the output blocks

    returns (batch, oWin, oSize), dot(W[i, j], h[i]) + b[j]
         but b[j] is only added once

    Notation
    --------
    - `batch` is the number of examples in a minibatch (batch size).
    - `iBlocks` is the total number of blocks in the input (from lower layer).
    - `iSize` is the size of each of these input blocks.
    - `iWin` is the number of blocks that will be used as inputs. Which blocks
      will be used is specified in `inputIdx`.
    - `oBlocks` is the number or possible output blocks.
    - `oSize` is the size of each of these output blocks.
    - `oWin` is the number of output blocks that will actually be computed.
      Which blocks will be computed is specified in `outputIdx`.
    """
    assert inputIdx.ndim == h.ndim - 1
    assert outputIdx.ndim == inputIdx.ndim
    if h.ndim == 2:
        h = h.dimshuffle('x', 0, 1)
        inputIdx = inputIdx.dimshuffle('x', 0)
        outputIdx = outputIdx.dimshuffle('x', 0)
    return sparse_block_gemv_ss(b.take(outputIdx, axis=0), W, h,
                                inputIdx, outputIdx)
