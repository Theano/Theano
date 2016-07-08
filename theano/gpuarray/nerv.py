from __future__ import absolute_import, print_function, division
import os.path
import theano
from theano import Apply, Variable, tensor

from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.gof import local_optimizer, COp
from theano.scalar import as_scalar, constant

from . import opt
from .basic_ops import (as_gpuarray_variable, GpuAllocEmpty,
                        infer_context_name, gpu_alloc_empty)
from .type import gpu_context_type
from .opt_util import alpha_merge, output_merge

try:
    from nervanagpu.nervanagpu import GPUTensor, NervanaGPU
    nerv = NervanaGPU()
except ImportError:
    GPUTensor = None
    nerv = None


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


class Gemm16(COp):
    """
    Gemm for float16 using the nervena kernels.
    """
    __props__ = ('relu', 'inplace')
    _f16_ok = True
    params_type = gpu_context_type
    KERN_NAMES = ('nn_128x128', 'nn_128x64', 'nn_128x32',
                  'nn_vec_128x128', 'nn_vec_128x64', 'nn_vec_128x32',
                  'tn_128x128', 'tn_128x64', 'tn_128x32',
                  'tn_vec_128x128', 'tn_vec_128x64', 'tn_vec_128x32',
                  'tn_vec_128x16', 'nt_128x128', 'nt_vec_128x128')

    def __init__(self, relu=False, inplace=False):
        COp.__init__(self, ["gemm16.c"], "gemm16")
        self.relu = relu
        # relu = True will require more work in optimizations.
        assert self.relu is False
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, C, alpha, A, B, beta):
        if GPUTensor is None:
            raise RuntimeError("Can't use Gemm16: nervanagpu not found")
        ctx_name = infer_context_name(C, A, B)

        A = as_gpuarray_variable(A, ctx_name)
        B = as_gpuarray_variable(B, ctx_name)
        C = as_gpuarray_variable(C, ctx_name)

        alpha = ensure_float(alpha, 'alpha')
        beta = ensure_float(beta, 'beta')

        assert C.dtype == A.dtype == B.dtype == 'float16'

        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def c_headers(self):
        return ['gpuarray/types.h', 'numpy_compat.h', 'gpuarray_helper.h',
                'string.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def get_op_params(self):
        return [('GEMM16_INPLACE', '1' if self.inplace else '0')]

    @staticmethod
    def cubin_to_code(name):
        fname = 'hgemm_{0}.cubin'.format(name)
        with open(os.path.join(nerv.cubin_path, fname)) as f:
            cubin = f.read()
        bcode = ','.join(hex(ord(c)) for c in cubin)
        return "static const char bin_%s[] = { %s };" % (name, bcode)

    @staticmethod
    def init_gpukernel(name, fail):
        return """
bcode = bin_%(name)s;
sz = sizeof(bin_%(name)s);
if (GpuKernel_init(&k_%(name)s, c->ctx, 1, &bcode, &sz,
                   "hgemm_%(name)s", 13, types, GA_USE_BINARY, NULL)
    != GA_NO_ERROR) {
  PyErr_SetString(PyExc_RuntimeError, "Could not initialize kernel %(name)s");
  %(fail)s;
}
""" % dict(name=name, fail=fail)

    def c_support_code(self):
        codel = []
        for name in self.KERN_NAMES:
            codel.append(Gemm16.cubin_to_code(name))
        return '\n'.join(codel)

    def c_support_code_struct(self, node, nodename):
        codel = []
        for name in self.KERN_NAMES:
            codel.append("GpuKernel k_{0};".format(name))
        codel.append(super(Gemm16, self).c_support_code_struct(node, nodename))
        return '\n'.join(codel)

    def c_init_code_struct(self, node, nodename, sub):
        codel = [super(Gemm16, self).c_init_code_struct(node, nodename, sub)]
        for name in self.KERN_NAMES:
            codel.append("memset(&k_{0}, 0, sizeof(GpuKernel));".format(name))
        codel.append("const char *bcode;")
        codel.append("size_t sz;")
        codel.append("PyGpuContextObject *c = %s;" % (sub['params'],))
        codel.append("int types[13] = {GA_BUFFER, GA_BUFFER, GA_BUFFER, "
                     "GA_BUFFER, GA_INT, GA_INT, GA_INT, GA_INT, GA_INT, "
                     "GA_INT, GA_FLOAT, GA_FLOAT, GA_INT};")
        for name in self.KERN_NAMES:
            codel.append(self.init_gpukernel(name, sub['fail']))
        return '\n'.join(codel)

    def c_cleanup_code_struct(self, node, nodename):
        codel = []
        for name in self.KERN_NAMES:
            codel.append("GpuKernel_clear(&k_{0});".format(name))
        return '\n'.join(codel)


@opt.register_opt('fast_compile')
@opt.op_lifter([tensor.Dot])
@opt.register_opt2([tensor.Dot], 'fast_compile')
def local_gpua_dot_to_gemm16(op, ctx_name, inputs, outputs):
    if nerv is None:
        return
    A = inputs[0]
    B = inputs[1]
    if (A.ndim == 2 and B.ndim == 2 and
            A.dtype == 'float16' and B.dtype == 'float16'):
        fgraph = getattr(outputs[0], 'fgraph', None)
        C = gpu_alloc_empty(ctx_name, dtype='float16')(
            shape_i(A, 0, fgraph), shape_i(B, 1, fgraph))
        return Gemm16()(C, 1.0, A, B, 0.0)


@opt.register_opt()
@alpha_merge(Gemm16, alpha_in=1, beta_in=4)
def local_gemm16_alpha_merge(node, *inputs):
    return [Gemm16(relu=node.op.relu)(*inputs)]


@opt.register_opt()
@output_merge(Gemm16, alpha_in=1, beta_in=4, out_in=0)
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
