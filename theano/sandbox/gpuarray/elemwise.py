import copy
from itertools import izip

import numpy
from theano import Op, Apply, scalar, config
from theano.tensor.elemwise import Elemwise, DimShuffle, CAReduceDtype

try:
    import pygpu
    from pygpu.tools import ScalarArg, ArrayArg
    from pygpu.elemwise import ElemwiseKernel
    from pygpu.reduction import ReductionKernel
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable, HideC
from theano.sandbox.gpuarray.type import GpuArrayType

from theano.gof.utils import MethodNotDefined


def _is_scalar(v):
    False


def make_argument(v, name):
    if _is_scalar(v):
        return ScalarArg(numpy.dtype(v.type.dtype), name)
    else:
        return ArrayArg(numpy.dtype(v.type.dtype), name)


def ensure_allocated(storage, shape, dtype):
    odat = storage[0]
    if odat is not None:
        if odat.shape != shape:
            # It is unsafe to try to resize odat,
            # we have to allocate output storage.
            odat = None
    if odat is None:
        odat = pygpu.empty(shape, dtype=dtype)
    storage[0] = odat
    return odat


def as_C_string_const(s):
    return '\n'.join('"%s\\n"' % (l.replace('"', '\\"'))
                     for l in s.split('\n'))


class GpuElemwise(HideC, Elemwise):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __str__(self):
        if self.name is not None:
            return self.name
        items = str(sorted(self.inplace_pattern.items()))
        return "GpuElemwise{%s}%s<gpuarray>" % (self.scalar_op, items)

    def make_node(self, *inputs):
        res = Elemwise.make_node(self, *inputs)
        outputs = [GpuArrayType(broadcastable=o.type.broadcastable,
                                dtype=o.type.dtype)() for o in res.outputs]
        inputs = [as_gpuarray_variable(i) for i in inputs]
        res = Apply(self, inputs, outputs)
        # Try to generate the kernel to catch SupportCodeErrors
        k = self.generate_kernel(res, 'test')
        return res

    def generate_kernel(self, node, nodename):
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(node.inputs)]
        scal_ins = [scalar.Scalar(i.dtype) for i in node.inputs]

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(node.outputs) if not n in self.inplace_pattern]
        scal_out = [scalar.Scalar(o.dtype) for o in node.outputs]

        fake_node = Apply(self.scalar_op, [i() for i in scal_ins],
                          [o() for o in scal_out])

        try:
            code = self.scalar_op.c_support_code_apply(fake_node, nodename)
            if code:
                raise SupportCodeError(code)
        except MethodNotDefined:
            pass

        support_code = ""
        try:
            support_code = self.scalar_op.c_support_code()
        except MethodNotDefined:
            pass

        if (support_code.strip() != "#define THEANO_MACRO_MOD(x,y) (x % y)" and
            support_code.strip() != ""):
            # The macro is fine, the C++ struct is not.
            raise SupportCodeError(support_code)

        scal_out = []
        oi = 0
        for n in range(len(fake_node.outputs)):
            if n in self.inplace_pattern:
                scal_out.append(inps[self.inplace_pattern[n]].name+'[i]')
            else:
                scal_out.append(outs[oi].name+'[i]')
                oi += 1

        kop = self.scalar_op.c_code(fake_node, nodename+'_scalar',
                                    [i.name+'[i]' for i in inps],
                                    scal_out,
                                    dict(fail='return;'))

        # Translate types for scalar composite ops (except complex).
        support_code += """
#define npy_float64 ga_double
#define npy_float32 ga_float
#define npy_uint8 ga_ubyte
#define npy_int8 ga_byte
#define npy_uint16 ga_ushort
#define npy_int16 ga_short
#define npy_uint32 ga_uint
#define npy_int32 ga_int
#define npy_uint64 ga_ulong
#define npy_int64 ga_long
"""
        return ElemwiseKernel(None, inps+outs, kop, preamble=support_code)

    def c_support_code_apply(self, node, nodename):
        # This is useless by itself, but will serve an eventual c_code
        # implementation
        k = self.generate_kernel(node, nodename)

        nd = node.inputs[0].type.ndim
        res = []
        for i in range(1, nd):
            var = "static const char %s_%s[] = " % (nodename, str(i))
            res.append(var + as_C_string_const(k.render_basic(i)) + ';')
            res.append("static const gpukernel *%s_%s_k = NULL;" % (nodename,
                                                                    str(i)))
        var = "static const char %s_c[] = " % (nodename,)
        res.append(var + as_C_string_const(k.contig_src) + ';')
        res.append("static const gpukernel *%s_c_k = NULL;" % (nodename,))
        return '\n'.join(res)

    def perform(self, node, inputs, output_storage):
        # Try to reuse the kernel from a previous call to hopefully
        # avoid recompiling
        if not hasattr(node, '_cache_elemwise_k'):
            node._cache_elemwise_k = self.generate_kernel(node, "kcode")

        out_shape = []
        for values in izip(*[input.shape for input in inputs]):
            if any(v == 0 for v in values):
                # All non-broadcasted dimensions should be zero
                assert max(values) <= 1
                out_shape.append(0)
            else:
                out_shape.append(max(values))
        out_shape = tuple(out_shape)

        args = copy.copy(inputs)
        for n, (stor, out) in enumerate(izip(output_storage, node.outputs)):
            if n in self.inplace_pattern:
                stor[0] = inputs[self.inplace_pattern[n]]
            else:
                args.append(ensure_allocated(stor, out_shape, out.type.dtype))

        # the dict call is there to avoid a syntax error in python < 2.6
        node._cache_elemwise_k(*args, **dict(broadcast=True))
        if config.gpuarray.sync:
            output_storage[0][0].sync()


class SupportCodeError(Exception):
    """
    We do not support certain things (such as the C++ complex struct)
    """


class GpuDimShuffle(HideC, DimShuffle):
    def make_node(self, input):
        res = DimShuffle.make_node(self, input)
        otype = GpuArrayType(dtype=res.outputs[0].type.dtype,
                             broadcastable=res.outputs[0].type.broadcastable)
        input = as_gpuarray_variable(input)
        return Apply(self, [input], [otype()])

    def __str__(self):
        if self.inplace:
            s = "InplaceGpuDimShuffle{%s}"
        else:
            s = "GpuDimShuffle{%s}"
        return s % (','.join(str(x) for x in self.new_order))

    def perform(self, node, inp, out):
        input, = inp
        storage, = out

        res = input

        res = res.transpose(self.shuffle+self.drop)

        shape = list(res.shape[:len(self.shuffle)])
        for augm in self.augment:
            shape.insert(augm, 1)
        res = res.reshape(shape)

        if not self.inplace:
            res = res.copy()

        storage[0] = res

    def c_support_code_apply(self, node, name):
        def copy_shape(nd_out):
            stmts = []
            e = 0
            for d in range(nd_out):
                if d in self.augment:
                    stmts.append("sh[%s] = 1;" % (d,))
                else:
                    stmts.append("sh[%s] = tmp->ga.dimensions[%s];" % (d, e))
                    e += 1
            return '\n            '.join(stmts)

        return """
        static const unsigned int %(name)s_ax[] = {%(shuffle)s};

        static PyGpuArrayObject *%(name)s_f(PyGpuArrayObject *a) {
            PyGpuArrayObject *res, *tmp;
            size_t sh[%(nd_out)s];

            tmp = pygpu_transpose(a, %(name)s_ax);
            if (!tmp) return NULL;
            %(copy_shape)s
            res = pygpu_reshape(tmp, %(nd_out)s, sh, GA_ANY_ORDER, 1, -1);
            Py_DECREF(tmp);
            return res;
        }
        """ % dict(shuffle=', '.join(str(a) for a in (self.shuffle+self.drop)),
                   name=name, nd_out=len(self.new_order),
                   copy_shape=copy_shape(len(self.new_order)))

    def c_code(self, node, name, inputs, outputs, sub):
        d = dict(name=name, fail=sub['fail'], inp=inputs[0], out=outputs[0],
                 nd=len(self.input_broadcastable))
        process = """
        PyGpuArrayObject *tmp = NULL;
        if (%(inp)s->ga.nd != %(nd)s) {
            PyErr_SetString(PyExc_TypeError, "input nd");
            %(fail)s
        }

        Py_XDECREF(%(out)s);
        %(out)s = %(name)s_f(%(inp)s);
        if (%(out)s == NULL) {%(fail)s}
        """ % d

        if not self.inplace:
            process += """
            tmp = pygpu_copy(%(out)s, GA_ANY_ORDER);
            Py_DECREF(%(out)s);
            if (!tmp) {
                %(out)s = NULL;
                %(fail)s
            }
            %(out)s = tmp;
            """ % d
        return process

    def c_code_cache_version(self):
        return (3,)


class GpuCAReduce(HideC, CAReduceDtype):
    def __init__(self, scalar_op, axis=None, dtype=None, acc_dtype=None):
        if not hasattr(scalar_op, 'identity'):
            raise ValueError("No identity on scalar op")
        CAReduceDtype.__init__(self, scalar_op, axis=axis, dtype=dtype,
                               acc_dtype=acc_dtype)

    def __str__(self):
        ax = ''
        if self.axis is not None:
            ax = '{%s}' % (', '.join(str(x) for x in self.axis),)
        return "GpuReduce{%s}%s" % (self.scalar_op, ax)

    def make_node(self, input):
        res = CAReduceDtype.make_node(self, input)
        input = as_gpuarray_variable(input)
        otype = GpuArrayType(dtype=res.outputs[0].dtype,
                             broadcastable=res.outputs[0].broadcastable)

        if res.op.axis is not None:
            redux = []
            for i in range(len(input.type.broadcastable)):
                redux.append(i in res.op.axis)
                # since redux is just another way to describe what is in axis
                # it doesn't need to be compared in __eq__ or __hash__
            res.op.redux = redux

        return Apply(res.op, [input], [otype()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        if self.axis is None:
            redux = [True] * node.inputs[0].ndim
        else:
            redux = self.redux
        acc_dtype = getattr(self, 'acc_dtype', None)
        if acc_dtype is None:
            acc_dtype = node.outputs[0].type.dtype
        if any(redux):
            node._cache_reduction_k = self.generate_kernel(node, acc_dtype,
                                                           redux)
        return super(GpuCAReduce, self).make_thunk(node, storage_map,
                                                   compute_map, no_recycling)

    def generate_kernel(self, node, odtype, redux):
        if isinstance(self.scalar_op, scalar.basic.Add):
            reduce_expr = "a + b"
        elif isinstance(self.scalar_op, scalar.basic.Mul):
            reduce_expr = "a * b"
        else:
            raise NotImplementedError()
        return ReductionKernel(pygpu.get_default_context(), odtype,
                               self.scalar_op.identity, reduce_expr, redux,
                               arguments=[make_argument(node.inputs[0], 'a')],
                               init_nd=node.inputs[0].ndim
        )

    def perform(self, node, inp, out):
        input, = inp
        output, = out

        if self.axis is None:
            redux = [True] * input.ndim
        else:
            redux = self.redux

        if any(redux):
            output[0] = node._cache_reduction_k(input).astype(copy=False,
                                             dtype=node.outputs[0].type.dtype)
        else:
            output[0] = pygpu.gpuarray.array(input, copy=True,
                                             dtype=node.outputs[0].type.dtype)
