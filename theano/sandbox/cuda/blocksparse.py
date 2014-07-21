import numpy
import theano
from theano import Apply, tensor, scalar, Constant
from theano.tensor import DimShuffle

from theano.gradient import grad_undefined, grad_not_implemented

from theano.sandbox.cuda import cuda_available, GpuOp, GpuElemwise

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray, opt, GpuFromHost,
                                     HostFromGpu, host_from_gpu,
                                     GpuDimShuffle)

import theano.misc.pycuda_init
from theano.misc.pycuda_init import pycuda_available
if pycuda_available:
    import pycuda.gpuarray
    from theano.misc.pycuda_utils import to_cudandarray

try:
    import scikits.cuda
    from scikits.cuda import cublas
    import scikits.cuda.misc
    scikits.cuda.misc.init()
    scikits_cuda_available = True
except ImportError:
    scikits_cuda_available = False


def gemm_batched(tA, tB, m, n, k, Al, lda, Bl, ldb, Cl, ldc,
                 alpha=numpy.float32(1.0), beta=numpy.float32(0.0)):
    assert Al.shape[0] == Bl.shape[0]
    assert Al.shape[0] == Cl.shape[0]

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSgemmBatched(handle, tA, tB, m, n, k, alpha,
                              Al.ptr, lda, Bl.ptr, ldb,
                              beta, Cl.ptr, ldc,
                              Cl.shape[0])


def gemv(alpha, A, x, beta, y):
    assert A.shape[0] == x.shape[0]
    assert A.shape[1] == y.shape[0]

    if A.strides[0] == 1:
        n, m = 0, 1
        trans = 't'
    else:
        n, m = 1, 0
        trans = 'n'

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSgemv(handle, trans, A.shape[n], A.shape[m], alpha,
                       A.gpudata, A.strides[m],
                       x.gpudata, x.strides[0],
                       beta, y.gpudata, y.strides[0])


def ger(alpha, x, y, A):
    assert A.shape[1] == x.shape[0]
    assert A.shape[0] == y.shape[0]

    handle = scikits.cuda.misc._global_cublas_handle

    cublas.cublasSger(handle, A.shape[1], A.shape[0], alpha,
                      x.gpudata, x.strides[0],
                      y.gpudata, y.strides[0],
                      A.gpudata, A.strides[0])


class SparseBlockGemvSS(GpuOp):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "SparseBlockGemvSS"

    def make_node(self, o, W, h, inputIdx, outputIdx):
        o = basic_ops.as_cuda_ndarray_variable(o)
        W = basic_ops.as_cuda_ndarray_variable(W)
        h = basic_ops.as_cuda_ndarray_variable(h)
        assert o.ndim == 2
        assert W.ndim == 4
        assert h.ndim == 2
        assert inputIdx.ndim == 1
        assert outputIdx.ndim == 1

        assert 'int' in inputIdx.type.dtype
        assert 'int' in outputIdx.type.dtype

        return Apply(self, [o, W, h, inputIdx, outputIdx],
                     [o.type()])

    def c_support_code(self):
        return """
        // This is NOT batch-ready
        __global__ void
        SparseBlockGemv_fill_lists(
int n,
const float **inp_list,
float **out_list,
const float **W_list,
const float *W, int W_str_0, int W_str_1,
const float *h, int h_str_0,
float *outB, int o_str_0, int o_str_1,
const npy_intp *iIdx,
const npy_intp *oIdx
        ) {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;
          int p = i + j * blockDim.x * gridDim.x;
          if (p >= n) return;
          inp_list[p] = &h[i * h_str_0];
          out_list[p] = &outB[i * o_str_0 + j * o_str_1];
          W_list[p] = &W[iIdx[i] * W_str_0 + oIdx[j] * W_str_1];
        }

        static int SparseBlockGemv_copy(PyArrayObject *a, npy_intp *b) {
          cudaError_t err;
          PyArrayObject *aa = (PyArrayObject *)PyArray_Cast(a, NPY_INTP);
          if (aa == NULL) { return -1; }
          err = cudaMemcpy(b, PyArray_DATA(aa), PyArray_NBYTES(aa),
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
        static float *%(n)s_outB;
        static size_t %(n)s_outB_size;
        static const float **%(n)s_inp_list;
        static float **%(n)s_out_list;
        static const float **%(n)s_W_list;
        static size_t %(n)s_list_len;
        static npy_intp *%(n)s_iIdx;
        static size_t %(n)s_iIdx_len;
        static npy_intp *%(n)s_oIdx;
        static size_t %(n)s_oIdx_len;

        // This is batch-ready
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
          if (%(n)s_outB_size < s*outsize) {
            cudaFree(%(n)s_outB);
            if (cudaMalloc(&%(n)s_outB, s*outsize*sizeof(float)) != cudaSuccess) return -1;
            %(n)s_outB_size = s*outsize;
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

    def perform(self, node, inputs, outputs):
        o, W, h, inputIdx, outputIdx = inputs
        out = outputs[0]

        dd = (o.shape[0] * h.shape[0],)
        weightHostB = numpy.empty(dd, dtype='intp')
        outputHostB = numpy.empty(dd, dtype='intp')
        inputHostB = numpy.empty(dd, dtype='intp')

        outputBatched = pycuda.gpuarray.GPUArray((h.shape[0], o.shape[0], o.shape[1]), dtype='float32')

        k = 0
        for j in range(o.shape[0]):
            out_id = outputIdx[j]
            for i in range(h.shape[0]):
                inp_id = inputIdx[i]
                weightHostB[k] = W[inp_id, out_id].gpudata
                outputHostB[k] = outputBatched[i, j].ptr
                inputHostB[k] = h[i].gpudata
                k += 1

        weightB = pycuda.gpuarray.to_gpu(weightHostB)
        inputB = pycuda.gpuarray.to_gpu(inputHostB)
        outputB = pycuda.gpuarray.to_gpu(outputHostB)

        tA = 'n'
        lda = W.strides[2]
        if lda == 1:
            tA = 't'
            lda = W.strides[3]


        gemm_batched(tA, 'n', o.shape[1], 1, h.shape[1],
                     weightB, lda, inputB, h.strides[0],
                     outputB, o.strides[0],
                     beta=numpy.asarray(0.0, dtype='float32'))

        outputBatchedG = to_cudandarray(outputBatched)
        out[0] = o + outputBatchedG.reduce_sum([1, 0, 0])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_code(self, node, nodename, inputs, outputs, sub):
        o, W, h, inputIdx, outputIdx = inputs
        out = outputs[0]

        return """
        if (%(name)s_prep(1, // NOT batch-ready
                          CudaNdarray_HOST_DIMS(%(h)s)[0],
                          CudaNdarray_HOST_DIMS(%(o)s)[0],
                          CudaNdarray_HOST_DIMS(%(o)s)[1]) == -1) {
          PyErr_SetString(PyExc_RuntimeError,
                          "Could not allocate working memory.");
          %(fail)s
        }
        {
          // NOT batch-ready
          int dims[3];
          dims[0] = 1; // This is to facilitate the reduction at the end.
          dims[1] = CudaNdarray_HOST_DIMS(%(o)s)[0];
          dims[2] = CudaNdarray_HOST_DIMS(%(o)s)[1];
          if (CudaNdarray_prep_output(&%(out)s, 3, dims)) {
            PyErr_SetString(PyExc_RuntimeError, "Cannot allocate output");
            %(fail)s
          }
        }
        // This is batch-ready
        if (SparseBlockGemv_copy(%(inputIdx)s, %(name)s_iIdx) == -1)
          { %(fail)s }
        if (SparseBlockGemv_copy(%(outputIdx)s, %(name)s_oIdx) == -1)
          { %(fail)s }
        { /* Prepare lists for the batch */
          // NOT batch-ready
          dim3 block;
          block.x = CudaNdarray_HOST_DIMS(%(h)s)[0];
          block.y = CudaNdarray_HOST_DIMS(%(o)s)[0];
          SparseBlockGemv_fill_lists<<<block, 1>>>(
block.x*block.y,
%(name)s_inp_list,
%(name)s_out_list,
%(name)s_W_list,
CudaNdarray_DEV_DATA(%(W)s),
CudaNdarray_HOST_STRIDES(%(W)s)[0], CudaNdarray_HOST_STRIDES(%(W)s)[1],
CudaNdarray_DEV_DATA(%(h)s), CudaNdarray_HOST_STRIDES(%(h)s)[0],
%(name)s_outB,
CudaNdarray_HOST_DIMS(%(o)s)[0] * CudaNdarray_HOST_DIMS(%(o)s)[1],
CudaNdarray_HOST_DIMS(%(o)s)[1],
%(name)s_iIdx,
%(name)s_oIdx);
        }
        { /* Run SgemmBatched */
          float alpha = 1.0;
          float beta = 0.0;
          cublasStatus_t err;
          cublasOperation_t transA = CUBLAS_OP_N;
          int lda = CudaNdarray_HOST_STRIDES(%(W)s)[2];
          if (lda == 1) {
            transA = CUBLAS_OP_T;
            lda = CudaNdarray_HOST_STRIDES(%(W)s)[3];
          }
          err = cublasSgemmBatched(handle, transA, CUBLAS_OP_N,
                                   CudaNdarray_HOST_DIMS(%(o)s)[1], 1,
                                   CudaNdarray_HOST_DIMS(%(h)s)[1], &alpha,
                                   %(name)s_W_list, lda, %(name)s_inp_list,
                                   CudaNdarray_HOST_STRIDES(%(h)s)[0],
                                   &beta, %(name)s_out_list,
                                   CudaNdarray_HOST_STRIDES(%(o)s)[0],
                                   CudaNdarray_HOST_DIMS(%(o)s)[0] *
                                   CudaNdarray_HOST_DIMS(%(h)s)[0]);
          if (err != CUBLAS_STATUS_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError, "SgemmBatched failed");
            %(fail)s
          }
        }
        { /* Perform final reduction and add biases */
          CudaNdarray *tmp;
          int p[2];
          p[0] = 1;
          p[1] = 2;
          tmp = (CudaNdarray *)CudaNdarray_new_nd(3);
          if (tmp == NULL) { %(fail)s }
          CudaNdarray_set_dim(tmp, 0, CudaNdarray_HOST_DIMS(%(h)s)[0]);
          CudaNdarray_set_stride(tmp, 0, CudaNdarray_HOST_DIMS(%(o)s)[0] *
                                 CudaNdarray_HOST_DIMS(%(o)s)[1]);
          CudaNdarray_set_dim(tmp, 1, CudaNdarray_HOST_DIMS(%(o)s)[0]);
          CudaNdarray_set_stride(tmp, 1, CudaNdarray_HOST_DIMS(%(o)s)[1]);
          CudaNdarray_set_dim(tmp, 2, CudaNdarray_HOST_DIMS(%(o)s)[1]);
          CudaNdarray_set_stride(tmp, 2, 1);
          CudaNdarray_set_device_data(tmp, %(name)s_outB, (PyObject *)NULL);
          if (CudaNdarray_reduce_sum(%(out)s, tmp) ||
              CudaNdarray_dimshuffle(%(out)s, 2, p)) {
            Py_DECREF(tmp);
            %(fail)s;
          }
          Py_DECREF(tmp);
          if (CudaNdarray_inplace_add((PyObject *)%(out)s, (PyObject *)%(o)s) == NULL) {
            %(fail)s;
          }
        }
        // And we're done!
        """ % dict(out=out, h=h, o=o, inputIdx=inputIdx, outputIdx=outputIdx,
                   W=W, fail=sub['fail'], name=nodename)

    def c_code_cache_version(self):
        return (3,)

    def grad(self, inputs, grads):
        o, W, h, inputIdx, outputIdx = inputs
        go = grads[0]

        # might revise that interface to not have a huge output
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


sparse_block_gemv_ss = SparseBlockGemvSS()


class SparseBlockOuterSS(GpuOp):
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

    def make_node(self, o, x, y, xIdx, yIdx, alpha=None, beta=None):
        one = tensor.constant(numpy.asarray(1.0, dtype='float32'))
        o = basic_ops.as_cuda_ndarray_variable(o)
        x = basic_ops.as_cuda_ndarray_variable(x)
        y = basic_ops.as_cuda_ndarray_variable(y)
        if alpha is None:
            alpha = one
        if beta is None:
            beta = one
        return Apply(self, [o, x, y, xIdx, yIdx, alpha, beta],
                     [o.type()])

    def perform(self, node, inputs, outputs):
        o, x, y, xIdx, yIdx, alpha, beta = inputs
        out = outputs[0]

        if not self.inplace:
            o = o.copy()

        dd = (x.shape[0] * y.shape[0],)
        xHostB = numpy.empty(dd, dtype='intp')
        yHostB = numpy.empty(dd, dtype='intp')
        outHostB = numpy.empty(dd, dtype='intp')

        k = 0
        for j in range(y.shape[0]):
            out_id = yIdx[j]
            for i in range(x.shape[0]):
                inp_id = xIdx[i]
                outHostB[k] = o[inp_id, out_id].gpudata
                xHostB[k] = x[i].gpudata
                yHostB[k] = y[j].gpudata
                k += 1

        xB = pycuda.gpuarray.to_gpu(xHostB)
        yB = pycuda.gpuarray.to_gpu(yHostB)
        outB = pycuda.gpuarray.to_gpu(outHostB)

        gemm_batched('n', 't', y.shape[1], x.shape[1], 1,
                     yB, y.strides[0], xB, x.strides[0],
                     outB, o.strides[2],
                     alpha=alpha, beta=beta)

        out[0] = o

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_support_code(self):
        return """
__global__ void
SparseBlockOuter_fill_lists(
int n,
const float **x_list,
const float **y_list,
float **out_list,
const float *x, int x_str_0,
const float *y, int y_str_0,
float *out, int o_str_0, int o_str_1,
const npy_intp *xIdx,
const npy_intp *yIdx
) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int p = i + j * blockDim.x * gridDim.x;
  if (p >= n) return;
  x_list[p] = &x[i * x_str_0];
  y_list[p] = &y[j * y_str_0];
  out_list[p] = &out[xIdx[i] * o_str_0 + yIdx[j] * o_str_1];
}

static int SparseBlockOuter_copy(PyArrayObject *a, npy_intp *b) {
  cudaError_t err;
  PyArrayObject *aa = (PyArrayObject *)PyArray_Cast(a, NPY_INTP);
  if (aa == NULL) { return -1; }
  err = cudaMemcpy(b, PyArray_DATA(aa), PyArray_NBYTES(aa),
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

// This is batch-ready
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
        o, x, y, xIdx, yIdx, alpha, beta = inputs
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
if (%(name)s_prep(1, CudaNdarray_HOST_DIMS(%(x)s)[0],
                  CudaNdarray_HOST_DIMS(%(y)s)[0]) == -1) {
  PyErr_SetString(PyExc_RuntimeError, "Could not allocate working memory.");
  %(fail)s
}
if (SparseBlockOuter_copy(%(xIdx)s, %(name)s_xIdx) == -1)
 { %(fail)s }
if (SparseBlockOuter_copy(%(yIdx)s, %(name)s_yIdx) == -1)
 { %(fail)s }
{
  dim3 block;
  block.x = CudaNdarray_HOST_DIMS(%(x)s)[0];
  block.y = CudaNdarray_HOST_DIMS(%(y)s)[0];
  SparseBlockOuter_fill_lists<<<block, 1>>>(
block.x * block.y,
%(name)s_x_list,
%(name)s_y_list,
%(name)s_out_list,
CudaNdarray_DEV_DATA(%(x)s), CudaNdarray_HOST_STRIDES(%(x)s)[0],
CudaNdarray_DEV_DATA(%(y)s), CudaNdarray_HOST_STRIDES(%(y)s)[0],
CudaNdarray_DEV_DATA(%(out)s),
CudaNdarray_HOST_STRIDES(%(out)s)[0], CudaNdarray_HOST_STRIDES(%(out)s)[1],
%(name)s_xIdx,
%(name)s_yIdx);
}
{
  cublasStatus_t err;
  err = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    CudaNdarray_HOST_DIMS(%(y)s)[1], CudaNdarray_HOST_DIMS(%(x)s)[1], 1,
    (float *)PyArray_GETPTR1(%(alpha)s, 0), %(name)s_y_list,
    CudaNdarray_HOST_STRIDES(%(y)s)[0], %(name)s_x_list,
    CudaNdarray_HOST_STRIDES(%(x)s)[0], (float *)PyArray_GETPTR1(%(beta)s, 0),
    %(name)s_out_list, CudaNdarray_HOST_STRIDES(%(out)s)[2],
    CudaNdarray_HOST_DIMS(%(x)s)[0] * CudaNdarray_HOST_DIMS(%(y)s)[0]);
  if (err != CUBLAS_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "SgemmBatched failed");
    %(fail)s
  }
}""" % dict(x=x, y=y, out=out, xIdx=xIdx, yIdx=yIdx, name=name,
            alpha=alpha, beta=beta, fail=sub['fail'])

    def c_code_cache_version(self):
        return (1,)


sparse_block_outer_ss = SparseBlockOuterSS(False)
sparse_block_outer_ss_inplace = SparseBlockOuterSS(True)


if cuda_available:
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
            return [sparse_block_outer_ss(*(ger.inputs[:5] +
                                           [alpha, ger.inputs[6]]))]

    @opt.register_opt()
    @opt.local_optimizer([GpuElemwise])
    def local_merge_blocksparse_beta(node):
        if (isinstance(node.op, GpuElemwise) and
            node.op.scalar_op == scalar.sub and
            node.nin == 2):
            ger = grab_ger(node.inputs[0])
            W = node.inputs[1]
            if ger is None:
                ger = grab_ger(node.inputs[1])
                W = node.inputs[0]
            if ger is None:
                return None
            return [sparse_block_outer_ss(*([W] + ger.inputs[1:5] +
                                            [-ger.inputs[5], ger.inputs[6]]))]


def sparse_block_dot_SS(W, h, inputIdx, b, outputIdx):
    """
    var: shape, comment
    W: (iBlocks, oBlocks, iSize, oSize), weight matrix
    h: (iWin, iSize), input from lower layer (sparse)
    inputIdx: (iWin,), indexes of the input blocks
    b: (oBlocks, oSize), bias vector
    outputIdx: (oWin,), indexes of the output blocks

    returns (oBlocks, oSize), dot(W[i, j], h[i]) + b[j]
         but b[j] is only added once
    """
    return sparse_block_gemv_ss(b.take(outputIdx, axis=0), W, h,
                                inputIdx, outputIdx)
