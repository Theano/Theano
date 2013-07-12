import numpy
from theano import Op, Apply, scalar
from pygpu.tools import ScalarArg, ArrayArg
from pygpu.elemwise import ElemwiseKernel

from basic_ops import as_gpuarray_variable
from type import GpuArrayType

from theano.gof.utils import MethodNotDefined

def _is_scalar(v):
    False

def make_argument(v, name):
    if _is_scalar(v):
        return ScalarArg(numpy.dtype(v.type.dtype), name)
    else:
        return ArrayArg(numpy.dtype(v.type.dtype), name)

def ensure_out(o, ref):
    if o is None:
        return ref._empty_like_me()
    else:
        return o

class GpuElemwise(Op):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op):
        self.scalar_op = scalar_op
        self.destroy_map = {}

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._rehash()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.scalar_op == other.scalar_op)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.scalar_op)

    def __str__(self):
        return "GpuElemwise{%s}(gpuarray)" % (self.scalar_op,)

    def make_node(self, *inputs):
        _inputs = [as_gpuarray_variable(i) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError("Wrong argument count", (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('mismatched rank amongst inputs')

        broadcastable = []
        for d in xrange(_inputs[0].type.ndim):
            bcast_d = True
            for i in _inputs:
                if not i.type.broadcastable[d]:
                    bcast_d = False
                    break
            broadcastable.append(bcast_d)
        assert len(broadcastable) == _inputs[0].type.ndim

        assert self.nout > 0
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(inputs)]
        scal_ins = [scalar.Scalar(i.dtype) for i in inputs]
                          
        res = Apply(self, _inputs, 
                    [GpuArrayType(o.dtype, broadcastable)()
                     for o in self.scalar_op.output_types(scal_ins)])

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(res.outputs)]
        scal_out = [scalar.Scalar(o.dtype) for o in res.outputs]

        fake_node = Apply(self.scalar_op, [i() for i in scal_ins],
                          [o() for o in scal_out])

        kcode = self.scalar_op.c_code(fake_node, 'kcode',
                                      [i.expr() for i in inps],
                                      [o.expr() for o in outs],
                                      sub=dict(fail='return;'))
        res.tag.kcode = kcode

        support_code = ""
        try:
            support_code += self.scalar_op.c_support_code_apply(fake_node, 'kcode')
        except MethodNotDefined:
            pass
        try:
            support_code += self.scalar_op.c_support_code()
        except MethodNotDefined:
            pass

        k = ElemwiseKernel(None, inps+outs, kcode, preamble=support_code)
        res.tag.kernel = k

        return res

    def perform(self, node, inps, out):
        k = node.tag.kernel
        outs = [ensure_out(o[0], inps[0]) for o in out]
        
        k.call_dimspec(*(inps+outs), broadcast=True)

        for o, og in zip(out, outs):
            o[0] = og
