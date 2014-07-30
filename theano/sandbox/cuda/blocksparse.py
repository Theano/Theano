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

class SparseBlockGemvSS(GpuOp):
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

        assert 'int' in inputIdx.type.dtype
        assert 'int' in outputIdx.type.dtype

        return Apply(self, [o, W, h, inputIdx, outputIdx],
                     [o.type()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def c_support_code(self):
        return """
        __global__ void
        SparseBlockGemv_fill_lists(
int n,
const float **inp_list,
float **out_list,
const float **W_list,
const float *W, int W_str_0, int W_str_1,
const float *h, int h_str_0, int h_str_1,
float *outB, int o_str_0, int o_str_1, int o_str_2,
const npy_intp *iIdx, int iI_str_0,
const npy_intp *oIdx, int oI_str_0
        ) {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;
          int b = threadIdx.z + blockDim.z * blockIdx.z;
          int p = i + j * blockDim.x * gridDim.x +
                  b * blockDim.y * gridDim.y * blockDim.x * gridDim.x;
          if (p >= n) return;
          inp_list[p] = &h[b * h_str_0 + i * h_str_1];
          out_list[p] = &outB[b * o_str_0 + i * o_str_1 + j * o_str_2];
          W_list[p] = &W[iIdx[b*iI_str_0+i] * W_str_0 +
                         oIdx[b*oI_str_0+j] * W_str_1];
        }

        __global__ void
        SparseBlockGemv_reduce(
int red_dim,
float *outB, int i_str_0, int i_str_1, int i_str_2, int i_str_3,
float *out, int o_str_0, int o_str_1, int o_str_2
        ) {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;
          int b = threadIdx.z + blockDim.z * blockIdx.z;
          float s = 0.0;
          float *oB = &outB[b * i_str_0 + i * i_str_2 + j * i_str_3];
          for (int k = 0; k < red_dim; k++) {
            s += oB[k * i_str_1];
          }
          out[b * o_str_0 + i * o_str_1 + j * o_str_2] += s;
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
          // NOT batch-ready
          dim3 block;
          block.x = CudaNdarray_HOST_DIMS(%(h)s)[1];
          block.y = CudaNdarray_HOST_DIMS(%(o)s)[1];
          block.z = CudaNdarray_HOST_DIMS(%(o)s)[0]; // batch size
          SparseBlockGemv_fill_lists<<<block, 1>>>(
block.x*block.y*block.z,
%(name)s_inp_list,
%(name)s_out_list,
%(name)s_W_list,
CudaNdarray_DEV_DATA(%(W)s),
CudaNdarray_HOST_STRIDES(%(W)s)[0], CudaNdarray_HOST_STRIDES(%(W)s)[1],
CudaNdarray_DEV_DATA(%(h)s), CudaNdarray_HOST_STRIDES(%(h)s)[0], CudaNdarray_HOST_STRIDES(%(h)s)[1],
%(name)s_outB,
CudaNdarray_HOST_DIMS(%(h)s)[1] * CudaNdarray_HOST_DIMS(%(o)s)[1] * CudaNdarray_HOST_DIMS(%(o)s)[2],
CudaNdarray_HOST_DIMS(%(o)s)[1] * CudaNdarray_HOST_DIMS(%(o)s)[2],
CudaNdarray_HOST_DIMS(%(o)s)[2],
%(name)s_iIdx, PyArray_DIM(%(inputIdx)s, 1),
%(name)s_oIdx, PyArray_DIM(%(outputIdx)s, 1));
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
                                   CudaNdarray_HOST_DIMS(%(o)s)[2], 1,
                                   CudaNdarray_HOST_DIMS(%(h)s)[2], &alpha,
                                   %(name)s_W_list, lda, %(name)s_inp_list,
                                   CudaNdarray_HOST_STRIDES(%(h)s)[1],
                                   &beta, %(name)s_out_list,
                                   CudaNdarray_HOST_STRIDES(%(o)s)[1],
                                   CudaNdarray_HOST_DIMS(%(o)s)[1] *
                                   CudaNdarray_HOST_DIMS(%(h)s)[1] *
                                   CudaNdarray_HOST_DIMS(%(o)s)[0]);
          if (err != CUBLAS_STATUS_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError, "SgemmBatched failed");
            %(fail)s
          }
        }
        { /* Perform final reduction and add biases */
          dim3 block;
          block.x = CudaNdarray_HOST_DIMS(%(o)s)[1];
          block.y = CudaNdarray_HOST_DIMS(%(o)s)[2];
          block.z = CudaNdarray_HOST_DIMS(%(o)s)[0];
          SparseBlockGemv_reduce<<<block, 1>>>(
CudaNdarray_HOST_DIMS(%(h)s)[1],
%(name)s_outB,
CudaNdarray_HOST_DIMS(%(h)s)[1] *
CudaNdarray_HOST_DIMS(%(o)s)[1] *
CudaNdarray_HOST_DIMS(%(o)s)[2],
CudaNdarray_HOST_DIMS(%(o)s)[1] *
CudaNdarray_HOST_DIMS(%(o)s)[2],
CudaNdarray_HOST_DIMS(%(o)s)[2],
1,
CudaNdarray_DEV_DATA(%(out)s),
CudaNdarray_HOST_STRIDES(%(out)s)[0],
CudaNdarray_HOST_STRIDES(%(out)s)[1],
CudaNdarray_HOST_STRIDES(%(out)s)[2]);
        }
        // And we're done!
        """ % dict(out=out, h=h, o=o, inputIdx=inputIdx, outputIdx=outputIdx,
                   W=W, fail=sub['fail'], name=nodename)

    def c_code_cache_version(self):
        return (5,)

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


sparse_block_gemv_ss = SparseBlockGemvSS(False)
sparse_block_gemv_ss_inplace = SparseBlockGemvSS(True)


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
const float *x, int x_str_0, int x_str_1,
const float *y, int y_str_0, int y_str_1,
float *out, int o_str_0, int o_str_1,
const npy_intp *xIdx, int xI_str_0,
const npy_intp *yIdx, int yI_str_0
) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int b = threadIdx.z + blockDim.z * blockIdx.z;
  int p = i + j * blockDim.x * gridDim.x +
          b * blockDim.y * gridDim.y * blockDim.x * gridDim.x;
  if (p >= n) return;
  x_list[p] = &x[b * x_str_0 + i * x_str_1];
  y_list[p] = &y[b * x_str_0 + j * y_str_1];
  out_list[p] = &out[xIdx[b * xI_str_0 + i] * o_str_0 +
                     yIdx[b * yI_str_0 + j] * o_str_1];
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
  block.x = CudaNdarray_HOST_DIMS(%(x)s)[1];
  block.y = CudaNdarray_HOST_DIMS(%(y)s)[1];
  block.z = CudaNdarray_HOST_DIMS(%(x)s)[0];
  SparseBlockOuter_fill_lists<<<block, 1>>>(
block.x * block.y * block.z,
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
  err = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    CudaNdarray_HOST_DIMS(%(y)s)[2], CudaNdarray_HOST_DIMS(%(x)s)[2], 1,
    (float *)PyArray_GETPTR1(%(alpha)s, 0), %(name)s_y_list,
    CudaNdarray_HOST_STRIDES(%(y)s)[1], %(name)s_x_list,
    CudaNdarray_HOST_STRIDES(%(x)s)[1], (float *)PyArray_GETPTR1(%(beta)s, 0),
    %(name)s_out_list, CudaNdarray_HOST_STRIDES(%(out)s)[2],
    CudaNdarray_HOST_DIMS(%(x)s)[0] *
    CudaNdarray_HOST_DIMS(%(x)s)[1] *
    CudaNdarray_HOST_DIMS(%(y)s)[1]);
  if (err != CUBLAS_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "SgemmBatched failed");
    %(fail)s
  }
}""" % dict(x=x, y=y, out=out, xIdx=xIdx, yIdx=yIdx, name=name,
            alpha=alpha, beta=beta, fail=sub['fail'])

    def c_code_cache_version(self):
        return (2,)


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
    h: (batch, iWin, iSize), input from lower layer (sparse)
    inputIdx: (batch, iWin), indexes of the input blocks
    b: (oBlocks, oSize), bias vector
    outputIdx: (batch, oWin), indexes of the output blocks

    returns (oBlocks, oSize), dot(W[i, j], h[i]) + b[j]
         but b[j] is only added once
    """
    assert inputIdx.ndim == h.ndim - 1
    assert outputIdx.ndim == inputIdx.ndim
    if h.ndim == 2:
        h = h.dimshuffle('x', 0, 1)
        inputIdx = inputIdx.dimshuffle('x', 0)
        outputIdx = outputIdx.dimshuffle('x', 0)
    return sparse_block_gemv_ss(b.take(outputIdx, axis=0), W, h,
                                inputIdx, outputIdx)
