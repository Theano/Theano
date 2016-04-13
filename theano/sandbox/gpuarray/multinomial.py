from theano import Apply
from theano.gof import COp
from .basic_ops import as_gpuarray_variable, infer_context_name
from .type import GpuArrayType

class GPUAMultinomialFromUniform(COp):

    def __init__(self):
        COp.__init__(self, ['multinomial.c'], 'APPLY_SPECIFIC(multinomial)')

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        ctx_name = infer_context_name(pvals, unis)

        pvals = as_gpuarray_variable(pvals, ctx_name)
        unis = as_gpuarray_variable(unis, ctx_name)

        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = GpuArrayType(broadcastable=br, dtype="float32")()

        return Apply(self, [pvals, unis], [out])

    def c_code_cache_version(self):
        return (8,)
