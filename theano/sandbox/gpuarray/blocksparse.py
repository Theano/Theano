from __future__ import absolute_import, print_function, division
import logging

import numpy
from theano import Op, Apply, tensor
from theano.tensor import discrete_dtypes

from theano.gradient import grad_undefined

from .basic_ops import as_gpuarray_variable, GpuKernelBase, Kernel

_logger = logging.getLogger('theano.sandbox.gpuarray.blocksparse')

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

class GpuSparseBlockGemv(Op):
    """
    GPU version of SparseBlockGemv. Check SparseBlockGemv's docstring for more
    information.

    This should not be directly called since the interface is subject
    to change without notice.  Use the sandbox.blocksparse.sparse_block_dot()
    function for a stable interface.
    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, o, W, h, inputIdx, outputIdx):
        ctx = infer_context(o, W, h)
        o = as_gpuarray_variable(o, ctx)
        W = as_gpuarray_variable(W, ctx)
        h = as_gpuarray_variable(h, ctx)
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

    def c_code(self, node, nodename, inputs, outputs, sub):
        o, W, h, inputIdx, outputIdx = inputs
        typecode = o.type.typecode
        out = outputs[0]

        if self.inplace:
            res = """
Py_XDECREF(%(out)s);
%(out)s = %(o)s;
Py_INCREF(%(out)s);
""" % dict(out=out, o=o)
        else:
            res = """
%(out)s = theano_try_copy(%(out)s, %(o)s);
if (%(out)s == NULL) {
  // Error already set
  %(fail)s
}
""" % dict(out=out, o=o, typecode=typecode, fail=sub['fail'], ctx=sub['params'])

        return res + """{
        gpudata **W_list = NULL;
        gpudata **inp_list = NULL;
        gpudata **out_list = NULL;
        size_t *offW = NULL;
        size_t *offInp = NULL;
        size_t *offOut = NULL;

        { /* Prepare lists for the batch */
          size_t maxi = PyGpuArray_DIMS(%(h)s)[1];
          size_t maxj = PyGpuArray_DIMS(%(o)s)[1];
          size_t maxb = PyGpuArray_DIMS(%(o)s)[0];
          ssize_t h_str_0 = PyGpuArray_STRIDES(%(h)s)[0];
          ssize_t h_str_1 = PyGpuArray_STRIDES(%(h)s)[1];
          ssize_t o_str_0 = PyGpuArray_STRIDES(%(o)s)[0];
          ssize_t o_str_1 = PyGpuArray_STRIDES(%(o)s)[1];
          ssize_t W_str_0 = PyGpuArray_STRIDES(%(W)s)[0];
          ssize_t W_str_1 = PyGpuArray_STRIDES(%(W)s)[1];

          W_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
          offW = calloc(sizof(size_t), maxi * maxj * maxb);
          inp_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
          offInp = calloc(sizof(size_t), maxi * maxj * maxb);
          out_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
          offOut = calloc(sizof(size_t), maxi * maxj * maxb);
          if (W_list == NULL || offW == NULL ||
              inp_list == NULL || offInp == NULL ||
              out_list == NULL || offOut == NULL) {
            free(W_list);
            free(offW);
            free(inp_list);
            free(offInp);
            free(out_list);
            free(offOut);
            PyErr_NoMemory();
            %(fail)s
          }
          for (size_t i = 0; i < maxi; i++) {
            for (size_t j = 0; j < maxj; j++) {
              for (size_t b = 0; b < maxb; b++) {
                size_t p = i + j * maxi + b * maxi * maxj;
                inp_list[p] = %(h)s->ga.data;
                offInp[p] = b * h_str_0 + i * h_str_1 + %(h)s->ga.offset;
                out_list[p] = %(o)s->ga.data;
                outInp[p] = b * o_str_0 + j * o_str_1 + %(o)s->ga.offset;
                W_list[p] = %(W)s->ga.data;
                offW[p] = *(%(inputIdx)s_DTYPE *)PyArray_GETPTR2(%(inputIdx)s, b, i) * W_str_0 + *(%(outputIdx)s_DTYPE *)PyArray_GETPTR2(%(outputIdx)s, b, j) * W_str_1 + %(W)s->ga.offset;
              }
            }
          }
        }
        { /* Run XgemvBatched */
          int err;
          cb_transpose transA = cb_no_trans;
          size_t lda = PyGpuArray_STRIDES(%(W)s)[2];
          if (lda == sizeof(float)) {
            transA = cb_trans;
            lda = PyGpuArray_STRIDES(%(W)s)[3];
          }

          if (%(typecode)s == GA_FLOAT) {
            err = blas_ops->sgemvBatch(cb_c, transA,
                             PyGpuArray_DIMS(%(o)s)[2],
                             PyGpuArray_DIMS(%(h)s)[2], 1,
                             W_list, offW, lda,
                             inp_list, offInp, PyGpuArray_STRIDES(%(h)s)[2],
                             1, out_list, offOut, PyGpuArray_STRIDES(%(o)s)[2],
                             PyGpuArray_DIMS(%(o)s)[1] * PyGpuArray_DIMS(%(h)s)[1] * PyGpuArray_DIMS(%(o)s)[0], 0);

          } else if (%(typecode)s == GA_DOUBLE) {
            err = blas_ops->dgemvBatch(cb_c, transA,
                             PyGpuArray_DIMS(%(o)s)[2],
                             PyGpuArray_DIMS(%(h)s)[2], 1,
                             W_list, offW, lda,
                             inp_list, offInp, PyGpuArray_STRIDES(%(h)s)[2],
                             1, out_list, offOut, PyGpuArray_STRIDES(%(o)s)[2],
                             PyGpuArray_DIMS(%(o)s)[1] * PyGpuArray_DIMS(%(h)s)[1] * PyGpuArray_DIMS(%(o)s)[0], 0);
          }
          free(W_list);
          free(offW);
          free(inp_list);
          free(offInp);
          free(out_list);
          free(offOut);
          if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError, "SgemvBatched failed(%%s)",
                         cublasGetErrorString(err));
            %(fail)s
          }
        }
        // And we're done!
        }""" % dict(out=out, h=h, o=o, inputIdx=inputIdx, outputIdx=outputIdx,
                   W=W, fail=sub['fail'], name=nodename)

    def c_code_cache_version(self):
        return ()

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        Wgrad = gpu_sparse_block_outer(W.zeros_like(),
                                       h, go, inputIdx, outputIdx)
        hgrad = gpu_sparse_block_gemv(h.zeros_like(),
                                      W.dimshuffle((1, 0, 3, 2)),
                                      go,
                                      outputIdx, inputIdx)
        return [go, Wgrad, hgrad,
                grad_undefined(self, 3, inputIdx,
                               "grad of inputIdx makes no sense"),
                grad_undefined(self, 4, outputIdx,
                               "grad of outputIdx makes no sense")]


gpu_sparse_block_gemv = GpuSparseBlockGemv(False)
gpu_sparse_block_gemv_inplace = GpuSparseBlockGemv(True)


class GpuSparseBlockOuter(GpuOp):
    """
    GPU version of SparseBlockOuter. See SparseBlockOuter's docstring for more
    information.

    This op should not be called directly since its interface is
    subject to change without notice.  It is involved in the gradient
    of GpuSparseBlockGemv. The gradient is not implemented.
    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

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
"""

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
%(out)s = theano_try_copy(%(out)s, %(o)s);
if (%(out)s == NULL) {
  // Error already set
  %(fail)s
}
""" % dict(out=out, o=o, fail=sub['fail'])

        return res + """
{
  size_t maxi = PyGpuArray_DIMS(%(x)s)[1];
  size_t maxj = PyGpuArray_DIMS(%(y)s)[1];
  size_t maxb = PyGpuArray_DIMS(%(x)s)[0];

  ssize_t x_str_0 = PyGpuArray_STRIDES(%(x)s)[0];
  ssize_t x_str_1 = PyGpuArray_STRIDES(%(x)s)[1];
  ssize_t y_str_0 = PyGpuArray_STRIDES(%(y)s)[0];
  ssize_t y_str_1 = PyGpuArray_STRIDES(%(y)s)[1];
  ssize_t o_str_0 = PyGpuArray_STRIDES(%(out)s)[0];
  ssize_t o_str_1 = PyGpuArray_STRIDES(%(out)s)[1];

  o_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
  offOut = calloc(sizof(size_t), maxi * maxj * maxb);
  x_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
  offX = calloc(sizof(size_t), maxi * maxj * maxb);
  y_list = calloc(sizof(gpudata *), maxi * maxj * maxb);
  offY = calloc(sizof(size_t), maxi * maxj * maxb);
  if (W_list == NULL || offW == NULL ||
      inp_list == NULL || offInp == NULL ||
      out_list == NULL || offOut == NULL) {
            free(o_list);
            free(offOut);
            free(x_list);
            free(offX);
            free(y_list);
            free(offY);
            PyErr_NoMemory();
            %(fail)s
          }
  for (size_t i = 0; i < maxi; i++) {
    for (size_t j = 0; j < maxj; j++) {
      for (size_t b = 0; b < maxb; b++) {
        size_t p = i + j * maxi + b * maxi * maxj;
        x_list[p] = %(x)s->ga.data;
        offX[p] = b * x_str_0 + i * x_str_1 + %(x)s->ga.offset;
        y_list[p] = %(y)s->ga.data;
        offY[p] = b * y_str_0 + j * y_str_1 + %(y)s->ga.offset;
        out_list[p] = %(out)s->ga.data;
        offOut[p] = *(%(xIdx)s_DTYPE *)PyArray_GETPTR2(%(xIdx)s, b, i) * o_str_0 + *(%(yIdx)s_DTYPE *)PyArray_GETPTR2(%(yIdx)s, b, j) * o_str_1 + %(out)s->ga.offset;
      }
    }
  }
{
  ga_ssize str_y = CudaNdarray_HOST_STRIDES(%(y)s)[2];
  ga_ssize str_x = CudaNdarray_HOST_STRIDES(%(x)s)[2];
  ga_ssize str_out = CudaNdarray_HOST_STRIDES(%(out)s)[2];
  int err;

  err = blas_ops->sgerBatch(cb_fortran,
    PyGpuArray_DIMS(%(y)s)[2], PyGpuArray_DIMS(%(x)s)[2],
    *(float *)PyArray_GETPTR1(%(alpha)s, 0),
    y_list, offY, str_y, x_list, offX, str_x, out_list, offOut, str_out,
    PyGpuArray_DIMS(%(x)s)[0] * PyGpuArray_DIMS(%(x)s)[1] * PyGpuArray_DIMS(%(y)s)[1], 0);
  free(o_list);
  free(offOut);
  free(x_list);
  free(offX);
  free(y_list);
  free(offY);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "sgerBatch failed");
    %(fail)s
  }
}""" % dict(x=x, y=y, out=out, xIdx=xIdx, yIdx=yIdx, name=name,
            alpha=alpha, fail=sub['fail'])

    def c_code_cache_version(self):
        return (11,)


gpu_sparse_block_outer = GpuSparseBlockOuter(False)
gpu_sparse_block_outer_inplace = GpuSparseBlockOuter(True)
