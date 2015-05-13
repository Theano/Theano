import numpy
import theano
from theano import Op, Apply, Variable, tensor

from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.gof import local_optimizer
from theano.scalar import as_scalar, constant

from . import opt
from .basic_ops import (as_gpuarray_variable, gpu_alloc, gpu_from_host,
                        host_from_gpu, GpuAllocEmpty)
from .opt_util import alpha_merge, output_merge
from .pycuda_helper import ensure_pycuda_context


try:
    from nervanagpu.nervanagpu import GPUTensor, NervanaGPU
    nerv = NervanaGPU()
except ImportError:
    GPUTensor = None


def to_gputensor(a):
    assert a.flags.c_contiguous or a.flags.f_contiguous
    return GPUTensor(a.shape, dtype=a.dtype, base=a,
                     gpudata=a.gpudata + a.offset,
                     strides=a.strides, is_trans=a.flags.f_contiguous)


def ensure_float(val, name):
    if not isinstance(val, Variable):
        val = constant(val)
    if hasattr(val, 'ndim') and val.ndim == 0:
        val = as_scalar(val)
    if not isinstance(val.type, theano.scalar.Scalar):
        raise TypeError("%s: expected a scalar value" % (name,))
    if not val.type.dtype == 'float32':
        raise TypeError("%s: type is not float32" % (name,))
    return val


class Gemm16(Op):
    __props__ = ('relu', 'inplace')
    _f16_ok = True

    def __init__(self, relu=False, inplace=False):
        self.relu = relu
        # relu = True will require more work in optimizations.
        assert self.relu == False
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, C, alpha, A, B, beta):
        if GPUTensor is None:
            raise RuntimeError("Can't use Gemm16: nervanagpu not found")

        A = as_gpuarray_variable(A)
        B = as_gpuarray_variable(B)
        C = as_gpuarray_variable(C)

        alpha = ensure_float(alpha, 'alpha')
        beta = ensure_float(beta, 'beta')

        assert C.dtype == A.dtype == B.dtype == 'float16'

        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs):
        ctx = ensure_pycuda_context()
        C, alpha, A, B, beta = inputs
        # The nervana code does not support the case where both inputs
        # are trans, so we need to copy one if them if that is the
        # case. We copy the smaller one.
        if A.flags.f_contiguous and B.flags.f_contiguous:
            if A.size < B.size:
                A = A.copy()
            else:
                B = B.copy()
        inplace = self.inplace
        if inplace and not C.flags.forc:
            inplace = False
        if not inplace:
            C = C.copy()
        At = to_gputensor(A)
        Bt = to_gputensor(B)
        Ct = to_gputensor(C)
        nerv.dot(At, Bt, Ct, alpha=alpha, beta=beta, relu=False)
        outputs[0][0] = C


@opt.register_opt()
@local_optimizer([tensor.Dot])
def local_dot_to_gemm16(node):
    if (type(node.op) == tensor.Dot and
            node.inputs[0].dtype == 'float16' and
            node.inputs[1].dtype == 'float16' and
            node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2):
        fgraph = node.inputs[0].fgraph
        A = gpu_from_host(node.inputs[0])
        B = gpu_from_host(node.inputs[1])
        C = GpuAllocEmpty(dtype='float16')(
            shape_i(A, 0, fgraph), shape_i(B, 1, fgraph))
        return [host_from_gpu(Gemm16()(C, 1.0, A, B, 0.0))]


@opt.register_opt()
@alpha_merge(Gemm16, alpha_in=1, beta_in=4, nd=2)
def local_gemm16_alpha_merge(node, *inputs):
    return [Gemm16(relu=node.op.relu)(*inputs)]


@opt.register_opt()
@output_merge(Gemm16, alpha_in=1, beta_in=4, out_in=0, nd=2)
def local_gemm16_output_merge(node, *inputs):
    return [Gemm16(relu=node.op.relu)(*inputs)]


@local_optimizer([Gemm16], inplace=True)
def local_gemm16_inplace(node):
    if type(node.op) != Gemm16 or node.op.inplace:
        return
    inputs = list(node.inputs)
    C = inputs[0]
    if (C.owner and
            isinstance(C.owner.op, GpuAllocEmpty) and
            len(C.clients) > 1):
        inputs[0] = C.owner.op(*C.owner.inputs)
    return [Gemm16(relu=node.op.relu, inplace=True)(*inputs)]

optdb.register('local_gemm16_inplace',
               tensor.opt.in2out(local_gemm16_inplace,
                                 name='local_gemm16_inplace'),
               70.0, 'fast_run', 'inplace', 'gpuarray')
