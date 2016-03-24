from __future__ import absolute_import, print_function, division
import copy
from theano.compat import izip
import numpy

import theano
from theano import Apply, scalar, config
from theano import scalar as scal
from six.moves import StringIO, xrange
from theano.gof.utils import MethodNotDefined
from theano.scalar import Scalar
from theano.tensor.elemwise import (Elemwise, DimShuffle, CAReduceDtype)

try:
    import pygpu
    from pygpu import gpuarray
    from pygpu.tools import ScalarArg, ArrayArg
    from pygpu.elemwise import ElemwiseKernel
    from pygpu.reduction import ReductionKernel
    from pygpu.gpuarray import dtype_to_typecode, dtype_to_ctype
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, HideC, GpuKernelBase, Kernel,
                        infer_context_name)
from .type import GpuArrayType
from .fp16_help import load_w, write_w


def _is_scalar(v):
    False


def make_argument(v, name):
    if _is_scalar(v):
        return ScalarArg(numpy.dtype(v.type.dtype), name)
    else:
        return ArrayArg(numpy.dtype(v.type.dtype), name)


def ensure_allocated(storage, shape, dtype, ctx):
    odat = storage[0]
    if odat is not None:
        if odat.shape != shape:
            # It is unsafe to try to resize odat,
            # we have to allocate output storage.
            odat = None
    if odat is None:
        odat = pygpu.empty(shape, dtype=dtype, context=ctx)
    storage[0] = odat
    return odat


def as_C_string_const(s):
    return '\n'.join('"%s\\n"' % (l.replace('"', '\\"'))
                     for l in s.split('\n'))


class GpuElemwise(GpuKernelBase, HideC, Elemwise):
    """
    Elemwise on the GPU.

    """
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)
    _f16_ok = True

    def __str__(self):
        if self.name is not None:
            return self.name
        items = str(sorted(self.inplace_pattern.items()))
        return "GpuElemwise{%s}%s<gpuarray>" % (self.scalar_op, items)

    def make_node(self, *inputs):
        ctx_name = infer_context_name(*inputs)
        res = Elemwise.make_node(self, *inputs)
        outputs = [GpuArrayType(broadcastable=o.type.broadcastable,
                                context_name=ctx_name,
                                dtype=o.type.dtype)() for o in res.outputs]
        if len(outputs) > 1:
            raise NotImplementedError()
        inputs = [as_gpuarray_variable(i, ctx_name) for i in inputs]
        node = Apply(self, inputs, outputs)

        # Try to generate the kernel to catch SupportCodeErrors
        try:
            scal_ins = [scalar.get_scalar_type(i.dtype) for i in node.inputs]

            scal_out = [scalar.get_scalar_type(o.dtype) for o in node.outputs]

            fake_node = Apply(self.scalar_op, [i() for i in scal_ins],
                              [o() for o in scal_out])
            code = self.scalar_op.c_support_code_apply(fake_node, "test")
            if code:
                raise SupportCodeError(code)
        except MethodNotDefined:
            pass
        try:
            support_code = self.scalar_op.c_support_code()
            if (support_code.strip() != "#define THEANO_MACRO_MOD(x,y) (x % y)" and
                    support_code.strip() != ""):
                # The macro is fine, the C++ struct is not.
                raise SupportCodeError(support_code)
        except MethodNotDefined:
            pass

        return node

    def get_params(self, node):
        return node.inputs[0].type.context

    def generate_kernel(self, node, nodename):
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(node.inputs)]
        scal_v_ins = [scalar.get_scalar_type(i.dtype) for i in node.inputs]

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(node.outputs) if n not in self.inplace_pattern]
        scal_v_outs = [scalar.get_scalar_type(o.dtype) for o in node.outputs]

        fake_node = Apply(self.scalar_op, [i() for i in scal_v_ins],
                          [o() for o in scal_v_outs])

        scal_in = [i.name + '[i]' if i.dtype != 'float16' else
                   '__half2float(' + i.name + '[i])' for i in inps]

        scal_out = []
        oi = 0
        scal_f16 = []
        for n in range(len(node.outputs)):
            if n in self.inplace_pattern:
                arg = inps[self.inplace_pattern[n]]
            else:
                arg = outs[oi]
                oi += 1
            if arg.dtype == 'float16':
                scal_f16.append(('tmpf16%i' % (len(scal_f16),), arg))
                scal_out.append(scal_f16[-1][0])
            else:
                scal_out.append(arg.name + '[i]')

        kop = self.scalar_op.c_code(fake_node, nodename + '_scalar',
                                    scal_in, scal_out,
                                    dict(fail='return;'))

        if scal_f16:
            # if we have float16 scalars on output we have to wrap
            # them and insert a stand-in float32 variable since
            # float16 arithemtic is not available
            code = ["{"]
            for f in scal_f16:
                code.append('ga_float %s;' % (f[0],))
            # XXX: The replace is an ugly hack to make sure temp
            # variables inthe middle are float32
            code.append(kop.replace('npy_float16', 'ga_float'))
            for f in scal_f16:
                code.append('%s[i] = __float2half_rn(%s);' % (f[1].name, f[0]))
            code.append('}')
            kop = '\n'.join(code)

        support_code = ""
        try:
            # We accept only some c_support_code().
            # This filter is done in the make_node()
            support_code += self.scalar_op.c_support_code()
        except MethodNotDefined:
            pass
        for npy, ga in [("npy_uint8", "ga_ubyte"),
                        ("npy_uint16", "ga_ushort"),
                        ("npy_uint32", "ga_uint"),
                        ("npy_uint64", "ga_ulong"),
                        ("npy_int8", "ga_byte"),
                        ("npy_int16", "ga_short"),
                        ("npy_int32", "ga_int"),
                        ("npy_int64", "ga_long"),
                        ("npy_float16", "ga_half"),
                        ("npy_float32", "ga_float"),
                        ("npy_float64", "ga_double"),
                        ]:
            kop = kop.replace(npy, ga)
        return ElemwiseKernel(self.get_params(node), inps + outs, kop,
                              preamble=support_code)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_support_code(self):
        return self.scalar_op.c_support_code()

    def _gpu_kernel_code(self, node, nodename):
        # This is useless by itself, but will serve an eventual c_code
        # implementation
        k = self.generate_kernel(node, nodename)
        nd = node.inputs[0].type.ndim
        res = []
        for i in range(0, nd + 1):
            res.append(k.render_basic(i, name="elem_" + str(i)) + ';')
        res.append(k.contig_src + ';')

        return '\n'.join(res)

    def gpu_kernels(self, node, nodename):
        src = self._gpu_kernel_code(node, nodename)
        nd = node.outputs[0].ndim
        params = ['uintp']
        params.extend('uintp' for _ in range(nd))
        num_inputs = len(node.inputs)
        num_outputs = len(node.outputs)
        for n in range(num_inputs + num_outputs):
            if (n - len(node.inputs)) in self.inplace_pattern:
                continue
            params.extend([gpuarray.GpuArray, 'uintp'])
            params.extend('intp' for _ in range(nd))
        acc_dtype = getattr(self, 'acc_dtype', None)
        if acc_dtype is None:
            acc_dtype = node.outputs[0].type.dtype
        return [Kernel(code=src, name="elem_%d" % nd, params=params,
                       flags=Kernel.get_flags(node.inputs[0].type.dtype,
                                              acc_dtype,
                                              node.outputs[0].type.dtype),
                       objvar='elem_%d_%s' % (nd, nodename))]

    def c_code(self, node, name, inputs, outputs, sub):
        if node.inputs[0].type.context.kind != 'cuda':
            raise MethodNotDefined('cuda only')
        nd = node.outputs[0].ndim
        fail = sub["fail"]
        initial_dims = ','.join('1' for i in xrange(nd))
        opname = str(self.scalar_op)
        ctx = sub['params']

        # check that all inputs have valid dimensions
        emitted_inames = {}
        num_kernel_params = 1 + nd + len(inputs + outputs) * (2 + nd)
        code = """
        size_t n_blocks = 0;
        size_t threads_per_block = 0;
        size_t numEls = 0;
        const ssize_t zero = 0;
        void *kernel_params[%(num_kernel_params)d] = {0};
        int err;
        """ % locals()
        if nd > 0:
            code += """
            size_t dims[%(nd)s] = {%(initial_dims)s};
            """ % locals()
        else:
            code += """
            size_t *dims = NULL;
            """
        for idx, iname in enumerate(inputs):
            if iname in emitted_inames:
                assert emitted_inames[iname] is node.inputs[idx]
                continue

            broadcasts = map(int, node.inputs[idx].broadcastable)
            broadcasts = ', '.join(map(str, broadcasts))
            nd = node.inputs[idx].ndim
            if nd > 0:
                code += """
                int broadcasts_%(iname)s[%(nd)s] = {%(broadcasts)s};
                """ % locals()
            else:
                code += """
                int *broadcasts_%(iname)s = NULL;
                """ % locals()
            emitted_inames[iname] = node.inputs[idx]

        # check that all inputs have valid dimensions
        emitted_inames = {}
        for idx, iname in enumerate(inputs):
            if iname in emitted_inames:
                continue
            code += """
        if (%(nd)s != PyGpuArray_NDIM(%(iname)s))
        {
            PyErr_Format(PyExc_TypeError,
                         "need %(nd)s dims, not %%u",
                         PyGpuArray_NDIM(%(iname)s));
            %(fail)s;
        }
        for (int i = 0; i< %(nd)s; ++i)
        {
            dims[i] = (dims[i] == 1) ? PyGpuArray_DIMS(%(iname)s)[i] : dims[i];
            if ((!(broadcasts_%(iname)s[i] &&
                 PyGpuArray_DIMS(%(iname)s)[i] == 1)) &&
                (dims[i] != PyGpuArray_DIMS(%(iname)s)[i]))
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " %(idx)d (indices start at 0) has shape[%%d] == %%llu"
                             ", but the output's size on that axis is %%llu.",
                             i,
                             (unsigned long long)PyGpuArray_DIMS(%(iname)s)[i],
                             (unsigned long long)dims[i]
                            );
                %(fail)s;
            }
        }
            """ % locals()
            emitted_inames[iname] = True
        # check that all outputs have valid dimensions
        for idx, oname in enumerate(outputs):
            typecode = dtype_to_typecode(node.outputs[idx].dtype)
            if idx not in self.inplace_pattern.keys():
                code += """
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != PyGpuArray_DIMS(%(oname)s)[i])
            {
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
            }
        }
        if (%(oname)s && !GpuArray_CHKFLAGS(&(%(oname)s->ga), GA_C_CONTIGUOUS))
        {
            Py_XDECREF(%(oname)s);
            %(oname)s = NULL;
        }
        if (NULL == %(oname)s)
        {
            %(oname)s = pygpu_empty(%(nd)d, dims,
                            %(typecode)s, GA_C_ORDER,
                            %(ctx)s, Py_None);
            if (!%(oname)s) {
                %(fail)s
            }
        }
        """ % locals()
            else:
                input_idx = self.inplace_pattern[idx]
                iname = inputs[input_idx]
                code += """
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_INCREF(%(oname)s);
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != PyGpuArray_DIMS(%(oname)s)[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Output dimension mis-match. Output"
                             " %(idx)d (indices start at 0), working inplace"
                             " on input %(input_idx)s, has shape[%%i] == %%llu"
                             ", but the output's size on that axis is %%llu.",
                             i,
                             (unsigned long long)PyGpuArray_DIMS(%(oname)s)[i],
                             (unsigned long long)dims[i]
                            );
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        """ % locals()
        z = outputs[0]
        code += """numEls = PyGpuArray_SIZE(%(z)s);

        //first use at least a full warp
        threads_per_block = std::min(numEls, (size_t)32); //WARP SIZE

        //next start adding multiprocessors
        // UP TO NUMBER OF MULTIPROCESSORS, use 30 for now.
        n_blocks = std::min(numEls/threads_per_block +
                               (numEls %% threads_per_block?1:0),
                           (size_t)30);

        // next start adding more warps per multiprocessor
        if (threads_per_block * n_blocks < numEls)
            threads_per_block = std::min(numEls/n_blocks, (size_t) 256);

        """ % locals()

        kname = 'elem_%d_%s' % (nd, name)
        param = ["(void *)&numEls"]
        for i in range(nd):
            param.append("(void *)&%(z)s->ga.dimensions[%(i)d]" % dict(z=outputs[0],
                                                                       i=i))
        for n, (name, var) in enumerate(zip(inputs + outputs,
                                            node.inputs + node.outputs)):
            if (n - len(inputs)) in self.inplace_pattern:
                continue
            dtype = dtype_to_ctype(var.dtype)
            param.append("(void *)%(name)s->ga.data" % locals())
            param.append("(void *)&%(name)s->ga.offset" % locals())
            for i in range(nd):
                param.append("PyGpuArray_DIMS(%(name)s)[%(i)d] == 1 ? (void *)&zero: (void *)&PyGpuArray_STRIDES(%(name)s)[%(i)d]" % locals())
        for n, p in enumerate(param):
            code += "kernel_params[%(n)d] = %(p)s;\n" % locals()
        code += """
        err = GpuKernel_call(&%(kname)s, 1, &threads_per_block, &n_blocks, 0, kernel_params);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "gpuarray error: %(kname)s: %%s.",
                         GpuKernel_error(&%(kname)s, err));
            %(fail)s;
        }
        """ % dict(kname=kname, fail=fail)
        if config.gpuarray.sync:
            code += """
            err = GpuArray_sync(&%(z)s->ga);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(kname)s: %%s.",
                             GpuKernel_error(&%(kname)s, err));
                %(fail)s;
            }
            """ % locals()
        return str(code)

    def perform(self, node, inputs, output_storage, ctx):
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
                args.append(ensure_allocated(stor, out_shape, out.type.dtype, ctx))

        node._cache_elemwise_k(*args, broadcast=True)
        if config.gpuarray.sync:
            output_storage[0][0].sync()

    def c_code_cache_version(self):
        ver = self.scalar_op.c_code_cache_version()
        if ver:
            return (4, ver)
        else:
            return ver


class SupportCodeError(Exception):
    """
    We do not support certain things (such as the C++ complex struct).

    """


class GpuDimShuffle(HideC, DimShuffle):
    """
    DimShuffle on the GPU.

    """
    _f16_ok = True

    def make_node(self, input):
        ctx_name = infer_context_name(input)
        res = DimShuffle.make_node(self, input)
        otype = GpuArrayType(dtype=res.outputs[0].type.dtype,
                             broadcastable=res.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        input = as_gpuarray_variable(input, ctx_name)
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

        res = res.transpose(self.shuffle + self.drop)

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
        """ % dict(shuffle=', '.join(str(a) for a in (self.shuffle + self.drop)),
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
        return (5,)


class GpuCAReduceCuda(GpuKernelBase, HideC, CAReduceDtype):
    """
    GpuCAReduceCuda is a Reduction along some dimensions by a scalar op.

    Parameters
    ----------
    reduce_mask
        The dimensions along which to reduce. The `reduce_mask` is a tuple of
        booleans (actually integers 0 or 1) that specify for each input
        dimension, whether to reduce it (1) or not (0).
    pre_scalar_op
        If present, must be a scalar op with only 1 input. We will execute it
        on the input value before reduction.

    Examples
    --------
    When scalar_op is a theano.scalar.basic.Add instance:

      - reduce_mask == (1,) sums a vector to a scalar

      - reduce_mask == (1,0) computes the sum of each column in a matrix

      - reduce_mask == (0,1) computes the sum of each row in a matrix

      - reduce_mask == (1,1,1) computes the sum of all elements in a 3-tensor.

    Notes
    -----
    Any reduce_mask of all zeros is a sort of 'copy', and may be removed during
    graph optimization.

    This Op is a work in progress.

    This op was recently upgraded from just GpuSum a general CAReduce. Not
    many code cases are supported for scalar_op being anything other than
    scal.Add instances yet.

    Important note: if you implement new cases for this op, be sure to
    benchmark them and make sure that they actually result in a speedup.
    GPUs are not especially well-suited to reduction operations so it is
    quite possible that the GPU might be slower for some cases.

    """
    __props__ = ('axis', 'reduce_mask', 'dtype', 'acc_dtype', 'scalar_op',
                 'pre_scalar_op')
    _f16_ok = True

    def __init__(self, scalar_op, axis=None,
                 reduce_mask=None, dtype=None, acc_dtype=None,
                 pre_scalar_op=None):
        if reduce_mask is not None:
            reduce_mask = tuple(reduce_mask)
        self.reduce_mask = reduce_mask

        # used to make sure that calls to scalar op
        # have unique name arguments
        self._n_scalar_op_calls = 0
        CAReduceDtype.__init__(self, scalar_op, axis=axis,
                               dtype=dtype, acc_dtype=acc_dtype)
        self.pre_scalar_op = pre_scalar_op
        if pre_scalar_op:
            assert pre_scalar_op.nin == 1

    def __str__(self):
        pre = ""
        if self.pre_scalar_op:
            pre = "pre=%s,red=" % str(self.pre_scalar_op)
        ax = ''
        if self.axis is not None:
            ax = '{%s}' % (', '.join(str(x) for x in self.axis),)
        return "GpuCAReduceCuda{%s%s}%s" % (pre, str(self.scalar_op), ax)

    def __setstate__(self, d):
        self.__dict__.update(d)
        # For unpickling of old ops.
        if not hasattr(self, "pre_scalar_op"):
            self.pre_scalar_op = None

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        if x.type.context.kind != 'cuda':
            raise TypeError("GpuCAReduceCuda doesn't work for non-cuda devices")
        ret = super(GpuCAReduceCuda, self).make_node(x)
        self = copy.copy(self)
        self.axis = ret.op.axis
        if self.pre_scalar_op:
            # Currently we only tested pre_scalar_op that don't cause
            # upcast.
            assert Elemwise(self.pre_scalar_op)(x).dtype == x.dtype
        if self.reduce_mask is None:
            if self.axis is None:
                reduce_mask = [1] * x.type.ndim
            else:
                reduce_mask = [0] * x.type.ndim
                for a in self.axis:
                    assert reduce_mask[a] == 0
                    reduce_mask[a] = 1
            self.reduce_mask = tuple(reduce_mask)

        if (x.type.ndim != len(self.reduce_mask)):
            raise TypeError("x must have rank %i" % len(self.reduce_mask))
        if ("complex" in x.dtype or
                "complex" in ret.outputs[0].dtype or
                "complex" in self._acc_dtype(x.dtype)):
            raise NotImplementedError("We don't support complex in gpu reduction")
        return Apply(self, [x], [GpuArrayType(ret.outputs[0].dtype,
                                              ret.outputs[0].type.broadcastable,
                                              context_name=x.type.context_name)()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def perform(self, node, inp, out, ctx):
        theano.Op.perform(self, node, inp, out, ctx)

    def supports_c_code(self, inputs):
        """
        Returns True if the current op and reduce pattern has functioning C code.

        """
        # If we don't even have the right method, we certainly
        # don't support the C code
        # (This is the test that used to be implemented by
        # local_gpu_sum)
        pattern = (''.join(str(i) for i in self.reduce_mask))
        if not hasattr(self, 'c_code_reduce_%s' % pattern):
            return False

        # Now that this is a general reduction op, we might
        # have a method for a pattern, but that pattern
        # might not be implemented for the current scalar op.
        # To detect this more complicated situation, we
        # make fake arguments to c_code, try to run them,
        # and see if NotImplementedError gets raised.

        node = self.make_node(*inputs)

        name = 'fake_name'

        inp = ['fake_input_name_%d' % i for i in xrange(len(inputs))]
        out = ['fake_output_name_%d' % i for i in xrange(len(node.outputs))]

        sub = {'fail': 'fake failure code', 'params': 'fake context'}

        try:
            self.c_code(node, name, inp, out, sub)
            self.c_support_code_apply(node, name)
        except NotImplementedError:
            return False
        return True

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        nd_in = node.inputs[0].type.ndim
        nd_out = node.outputs[0].type.ndim
        # For complex, we need to use theano_complex* in the c code to
        # have it run. But libgpuarray don't understand it.
        in_dtype = node.inputs[0].type.dtype_specs()[1]
        out_dtype = node.outputs[0].type.dtype_specs()[1]
        gin_dtype = "npy_" + node.inputs[0].dtype
        gout_dtype = "npy_" + node.outputs[0].dtype
        assert nd_in - nd_out == sum(self.reduce_mask)

        sio = StringIO()
        fail = sub['fail']
        ctx = sub['params']

        # check input
        print("""
        if (PyGpuArray_NDIM(%(x)s) != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError,
                         "required nd=%(nd_in)s, got nd=%%u", PyGpuArray_NDIM(%(x)s));
            %(fail)s;
        }
        """ % locals(), file=sio)

        # It might be nice to use a property of the op class to do this,
        # but tensor.elemwise.CAReduce has this exact same check so I guess
        # this is OK to do
        if self.scalar_op in [scal.minimum, scal.maximum]:
            conds = ["(PyGpuArray_DIMS(%s)[%d] == 0)" % (x, i)
                     for i in xrange(nd_in)
                     if self.reduce_mask[i]]
            assert len(conds) > 0
            cond = "(" + " || ".join(conds) + ")"
            print("""
            if %(cond)s
            {
                PyErr_Format(PyExc_ValueError," tried to reduce a 0-length axis.");
                %(fail)s;
            }
            """ % locals(), file=sio)

        #
        # alloc an output if we need one
        #

        # check the basics of out output
        print("""
        if (  !%(z)s
           || (PyGpuArray_NDIM(%(z)s) != %(nd_out)s)
        """ % locals(), file=sio)

        # ensure that the output has the right non-reduced dimensions
        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print(" || (PyGpuArray_DIMS(%(z)s)[%(j)s] != PyGpuArray_DIMS(%(x)s)[%(i)d]) " % locals(), file=sio)
                j += 1

        print("""
           )
        {
            """ % locals(), file=sio)
        if nd_out > 0:
            print("size_t new_dims[%(nd_out)s]; " % locals(), file=sio)
        else:
            print("size_t *new_dims=NULL; ", file=sio)

        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print('new_dims[%(j)s] = PyGpuArray_DIMS(%(x)s)[%(i)s];' % locals(), file=sio)
                j += 1
        out_typecode = dtype_to_typecode(gout_dtype[4:])
        print("""
            Py_XDECREF(%(z)s);
            %(z)s = pygpu_empty(%(nd_out)s, new_dims,
                                %(out_typecode)s, GA_C_ORDER,
                                %(ctx)s, Py_None);
            if (NULL == %(z)s)
            {
                PyErr_Format(PyExc_RuntimeError, "Failed to allocate output");
                %(fail)s;
            }
        }
        """ % locals(), file=sio)

        # \begin bracket the reduction in a check that there is
        # actually work to do
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "GpuArray_memset(&%(z)s->ga, 0)" % locals()
        # TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            scalar_op = self.scalar_op
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduceCuda not implemented when input shape is 0"
                         " for this scalar_op: %(scalar_op)s");
            %(fail)s;
            """ % locals()
        print("""
        if (PyGpuArray_SIZE(%(z)s) && ! PyGpuArray_SIZE(%(x)s)){
            %(zero_shp)s;
        }
        else if (PyGpuArray_SIZE(%(z)s))
        {
        """ % locals(), file=sio)

        #
        # Now perform the reduction
        #

        if all(i == 1 for i in self.reduce_mask):
            # check if the tensor is ccontiguous, if true, use the c_code_reduce_ccontig code.
            # TODO: check if we are ccontiguous when we un-dimshuffle
            # TODO: if only some dims are ccontiguous, call version with less dims.
            print('if(%(x)s->ga.flags & GA_C_CONTIGUOUS){' % locals(),
                  file=sio)
            self.c_code_reduce_ccontig(sio, node, name, x, z, fail)
            print("}else{", file=sio)
            getattr(self, 'c_code_reduce_%s' %
                    (''.join(str(i) for i in self.reduce_mask)))(
                sio, node, name, x, z, fail)
            print("}", file=sio)
        else:
            getattr(self, 'c_code_reduce_%s' % (''.join(
                str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)

        # \end bracket the reduction ...
        print("""
        }
        """ % locals(), file=sio)

        return sio.getvalue()

    def _makecall(self, node, name, x, z, fail, pattern=None, extra_dims=(), extra_strides=()):
        """
        Return a string for making a kernel call.

        The return value looks something like:

            .. code-block:: c

                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
                if (verbose)
                    printf("running kernel_reduce_10_%(name)s\\n");
                size_t n_shared = sizeof(%(acc_dtype)s) * n_threads[0] * n_threads[1] * n_threads[2];
                void *kernel_params[] = {
                        (void *)&PyGpuArray_DIMS(%(x)s)[0],
                        (void *)&PyGpuArray_DIMS(%(x)s)[1],
                        (void *)%(x)s->ga.data,
                        (void *)&%(x)s->ga.offset,
                        (void *)&stride_A0,
                        (void *)&stride_A1,
                        (void *)%(z)s->ga.data,
                        (void *)&%(z)s->ga.offset,
                        (void *)&stride_Z0};
                int err = GpuKernel_call(&%(k_var)s, 3, n_threads, n_blocks, n_shared, kernel_params);
                %(err_check)s
        """
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        sio = StringIO()
        if pattern is None:
            pattern = ''.join(str(c) for c in self.reduce_mask)
        ndim = len(self.reduce_mask)
        nd_out = ndim - sum(self.reduce_mask)
        shapes_format = "shape=(%s)" % ",".join(["%llu"] * node.inputs[0].ndim)
        shapes_data = ",".join(["(size_t) PyGpuArray_DIMS(%s)[%d]" % (x, i)
                                for i in range(node.inputs[0].ndim)])
        k_var = "kernel_reduce_%(pattern)s_%(name)s" % locals()
        params = []

        for i in xrange(ndim):
            params.append("(void *)&PyGpuArray_DIMS(%(x)s)[%(i)s]" % locals())
        for declaration, value in extra_dims:
            print(declaration % locals(), file=sio)
            params.append(value)
        params.append("(void *)%(x)s->ga.data" % locals())
        params.append("(void *)&%(x)s->ga.offset" % locals())
        for i in xrange(ndim):
            print("""
            ssize_t stride_A%(i)d = PyGpuArray_STRIDES(%(x)s)[%(i)s]/sizeof(%(in_dtype)s);
            """ % locals(), file=sio)
            params.append("(void *)&stride_A%(i)d" % locals())
        for declaration, value in extra_strides:
            print(declaration % locals(), file=sio)
            params.append(value)

        params.append("(void *)%(z)s->ga.data" % locals())
        params.append("(void *)&%(z)s->ga.offset" % locals())
        for i in xrange(nd_out):
            print("""
            ssize_t stride_Z%(i)d = PyGpuArray_STRIDES(%(z)s)[%(i)s]/sizeof(%(out_dtype)s);
            """ % locals(), file=sio)
            params.append("(void *)&stride_Z%(i)d" % locals())
        kernel_params = ', '.join(params)
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        print("""
            if (verbose)
                printf("running kernel_reduce_%(pattern)s_%(name)s\\n");
            size_t n_shared = sizeof(%(acc_dtype)s) * n_threads[0] * n_threads[1] * n_threads[2];
            void *kernel_params[] = { %(kernel_params)s };
            if (verbose>1)
                printf("n_threads[0]=%%lu, n_threads[1]=%%lu, "
                       "n_threads[2]=%%lu, n_threads=%%lu, "
                       "n_blocks[0]=%%lu, n_blocks[1]=%%lu, n_blocks[2]=%%lu, "
                       "n_blocks=%%lu, n_shared=%%d, %(shapes_format)s\\n",
                                  n_threads[0],n_threads[1],
                                  n_threads[2],
                                  n_threads[0]*n_threads[1]*
                                  n_threads[2],
                                  n_blocks[0],n_blocks[1],n_blocks[2],
                                  n_blocks[0]*n_blocks[1]*n_blocks[2],
                                  n_shared, %(shapes_data)s);
            int err = GpuKernel_call(&%(k_var)s, 3, n_threads, n_blocks, n_shared, kernel_params);
            %(err_check)s
            """ % locals(), file=sio)

        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
            """ % locals()
        print("""
            %(sync)s
        """ % locals(), file=sio)
        return sio.getvalue()

    def _k_decl(self, node, nodename, pattern=None,
                ndim=None, reduce_mask=None):
        """
        Return a string to declare a kernel function.

        The result will look something like this:

        .. code-block:: c

            KERNEL void kernel_reduce_110_%(nodename)s(
                    const ga_size d0,
                    const ga_size d1,
                    const ga_size d2,
                    const %(in_type)s *A,
                    const ga_size offset_A,
                    const ga_ssize sA0,
                    const ga_ssize sA1,
                    const ga_ssize sA2,
                    %(out_type)s * Z,
                    const ga_size offset_Z,
                    const ga_ssize sZ0)

        Since the nodename is unique, we don't need to put the name
        of the scalar_op in here.

        """
        in_dtype = node.inputs[0].dtype
        out_dtype = node.outputs[0].dtype
        in_type = gpuarray.dtype_to_ctype(in_dtype)
        out_type = gpuarray.dtype_to_ctype(out_dtype)
        if reduce_mask is None:
            reduce_mask = self.reduce_mask
        if ndim is None:
            ndim = len(reduce_mask)
        if pattern is None:
            pattern = ''.join(str(i) for i in reduce_mask)
        kname = "kernel_reduce_%(pattern)s" % locals()
        k_var = "kernel_reduce_%(pattern)s_%(nodename)s" % locals()
        params = []
        sio = StringIO()

        print("""
            KERNEL void %(kname)s(
        """ % locals(), file=sio)
        for i in xrange(ndim):
            params.append('uintp')
            print("""
                    const ga_size d%(i)s,
        """ % locals(), file=sio)
        params.append(gpuarray.GpuArray)
        params.append('uintp')
        print("""
                    const %(in_type)s *A, const ga_size offset_A,
        """ % locals(), file=sio)
        for i in xrange(ndim):
            params.append('intp')
            print("""
                    const ga_ssize sA%(i)s,
        """ % locals(), file=sio)
        params.append(gpuarray.GpuArray)
        params.append('uintp')
        print("""
                    %(out_type)s * Z, const ga_size offset_Z
        """ % locals(), file=sio)
        for i in xrange(ndim - sum(reduce_mask)):
            params.append('intp')
            print("""
                    , const ga_ssize sZ%(i)s
        """ % locals(), file=sio)
        print(")", file=sio)
        return sio.getvalue(), kname, params, k_var

    def _k_init(self, node, nodename):
        in_dtype = node.inputs[0].dtype
        out_dtype = node.outputs[0].dtype
        acc_dtype = self._acc_dtype(node.inputs[0].dtype)
        # We need to use theano_complex* and not npy_complex*
        in_type = gpuarray.dtype_to_ctype(in_dtype)
        out_type = gpuarray.dtype_to_ctype(out_dtype)
        acc_type = gpuarray.dtype_to_ctype(acc_dtype)

        return """
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = 0;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                //This is caught in cuda/init.py when we init the gpu. I keep
                //it here to ease finding code that rely on this.
                if (warpSize != 32)
                {
                    Z[0] = -666;
                    return;
                }

        """ % locals()

    def _assign_init(self, first_item):
        """
        This return the initial value for myresult.
        If the scalar op have an identity value, return it.

        Otherwise, check that the scalar op is maximum or minimum
        and return first_item. It should be the first element of the reduction.
        As the maximum and minimum of the same value don't change, this work.

        """
        if hasattr(self.scalar_op, 'identity'):
            return str(self.scalar_op.identity)
        else:
            assert isinstance(self.scalar_op, (scal.Maximum,
                                               scal.Minimum))
            if self.pre_scalar_op:  # TODO: multiple dtypes
                # dtype = node.inputs[0].dtype
                dtype = 'float32'

                dummy_var = scal.Scalar(dtype=dtype)()

                dummy_node = self.pre_scalar_op.make_node(dummy_var)

                dummy_name = 'assign_init_pre_scalar_op' + str(self._n_scalar_op_calls)
                self._n_scalar_op_calls += 1
                t = self.pre_scalar_op.c_code(dummy_node, dummy_name,
                                              (first_item,), ("",), {})
                assert t.startswith(' = ')
                first_item = t[3:]
                if first_item[-1] == ';':
                    first_item = first_item[:-1]

            return first_item

    def _assign_reduce(self, node, name, left, right, sub, pre):
        """

        Parameters
        ----------
        node
            The node argument to this op's c_code.
        name
            The name argument to this op's c_code.
        left
            A C code string identifying an lvalue.
        right
            A C code string identifying an expression.
        sub
            The sub argument to this op's c_code.
        pre
            If True, we will add the pre_scalar_op.c_code.

        Returns
        -------
        str
            C code to reduce left and right, assigning the result to left.

        """

        x, = node.inputs
        in_dtype = x.dtype
        out_dtype = node.outputs[0].dtype

        dummy_left = Scalar(dtype=out_dtype)()
        dummy_right = Scalar(dtype=in_dtype)()

        dummy_node = self.scalar_op.make_node(dummy_left, dummy_right)

        dummy_name = name + '_scalar_op' + str(self._n_scalar_op_calls)
        self._n_scalar_op_calls += 1

        if pre and self.pre_scalar_op:
            assert left == "myresult"
            dummy_node = self.pre_scalar_op.make_node(dummy_left)
            dummy_name = name + '_scalar_op' + str(self._n_scalar_op_calls)
            self._n_scalar_op_calls += 1
            t = self.pre_scalar_op.c_code(dummy_node, dummy_name,
                                          (right,), ("",), sub)
            assert t.startswith(' = ')
            right = t[3:]
            if right[-1] == ';':
                right = right[:-1]

        return self.scalar_op.c_code(dummy_node, dummy_name, (left, right),
                                     (left,), sub)

    def _k_reduce_buf(self, z_pos, node, name, sub):
        """
        WRITEME

        Parameters
        ----------
        node, name, sub
            These should be passed through from the original call to c_code.

        """
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        write_out = write_w(node.outputs[0].dtype)

        # This code (the code in new_version) is currently ignored.
        # Code produced later in this function is returned instead.
        # The code here works with all nvidia driver
        # But only for powers or multiples of 2!
        new_version = """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();


        if (threadNum >= ((threadCount >> 1) * 2))
        {
            int idx = threadNum - (threadCount >> 1) * 2;"""

        new_version += self._assign_reduce(node, name, 'buf[idx]',
                                           'buf[threadNum]', sub, False)

        new_version += """
        }
        __syncthreads();

        // Works for power of 2 only.
        int nTotalThreads = threadCount; // Total number of active threads
        while(nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1);        // divide by two
            // only the first half of the threads will be active.

            if (threadNum < halfPoint)
            {
              // Get the shared value stored by another thread
              %(acc_dtype)s temp = buf[threadNum + halfPoint];
              """

        new_version += self._assign_reduce(node, name,
                                           'buf[threadNum]', 'temp', sub, False)

        new_version += """
            }
            __syncthreads();

            nTotalThreads = (nTotalThreads >> 1);        // divide by two.
        }
            __syncthreads();

        if (threadNum == 0)
        {
            %(z_pos)s = %(write_out)s(buf[0]);
        }
            __syncthreads();"""

        new_version = new_version % locals()

        current_version = """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < warpSize)
        {
            //round up all the partial sums into the first `warpSize` elements
            for (int i = threadNum + warpSize; i < threadCount; i += warpSize)
            {
                """
        current_version += self._assign_reduce(node, name,
                                               'myresult', 'buf[i]',
                                               sub, False) + """
            }
            buf[threadNum] = myresult;
        /*Comment this optimization as it don't work on Fermi GPU.
        TODO: find why it don't work or put the GPU compute capability into the version
            // no sync because only one warp is running
            if(threadCount >32)
            {"""
        for num in [16, 8, 4, 2, 1]:
            current_version += self._assign_reduce(node, name,
                                                   'buf[threadNum]',
                                                   'buf[threadNum+%d]' % num,
                                                   sub, False)
            current_version += """
            """
        current_version += """
                if (threadNum == 0)
                {
                    %(z_pos)s = %(write_out)s(buf[0]);
                }

            }
            else */
            if (threadNum < 16)
            {
                //reduce so that threadNum 0 has the reduction of everything
                """
        for num in [16, 8, 4, 2, 1]:
            this_if = "if (threadNum + %d < threadCount) " % num + \
                self._assign_reduce(node, name,
                                    'buf[threadNum]', 'buf[threadNum+%d]' % num,
                                    sub, False)
            current_version += this_if
            current_version += """
            """
        current_version += """
                if (threadNum == 0)
                {
                    %(z_pos)s = %(write_out)s(buf[0]);
                }
            }
        }
        """

        current_version = current_version % locals()

        return current_version

    # Threads must be organized as: threadNum%nb_reduce correspond to the same sum
    # nb_reduce<=warpSize
    def _k_reduce_buf_multiple(self, z_pos, node, name, nb_reduce):
        reduce_fct = self._assign_reduce(node, name, 'myresult', 'buf[i]', {}, False)
        write_out = write_w(node.outputs[0].dtype)

        return """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < %(nb_reduce)s)
        {
            //round up all the partial sums into the first `nb_reduce` elements
            for (int i = threadNum + %(nb_reduce)s; i < threadCount; i += %(nb_reduce)s)
            {
                %(reduce_fct)s;
            }
            %(z_pos)s = %(write_out)s(myresult);
        }
        """ % locals()

    def c_code_reduce_ccontig(self, sio, node, name, x, z, fail):
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "GpuArray_memset(&%(z)s->ga, 0)" % locals()
        # TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduceCuda not implemented when input shape is 0 for this scalar_op");
            %(fail)s;
            """ % locals()

        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        k_var = "kernel_reduce_ccontig_%(name)s" % locals()
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
            """ % locals()
        print("""
        {
          if(PyGpuArray_SIZE(%(x)s)==0){
            %(zero_shp)s;
          }else{
            int verbose = 0;
            size_t numEls = PyGpuArray_SIZE(%(x)s);
            size_t n_threads = std::min(numEls, (size_t) 256);
            size_t n_blocks = 1;
            void *kernel_params[] = {(void *)&numEls,
                                     (void *)%(x)s->ga.data,
                                     (void *)&%(x)s->ga.offset,
                                     (void *)%(z)s->ga.data,
                                     (void *)&%(z)s->ga.offset};
            if (verbose) printf("running kernel_reduce_ccontig_%(name)s"
                                " n_threads=%%llu, size=%%llu, ndim=%%u\\n",
                                n_threads, numEls,
                                PyGpuArray_NDIM(%(x)s));
            size_t n_shared = sizeof(%(acc_dtype)s) * n_threads;
            int err = GpuKernel_call(&%(k_var)s, 1, &n_threads, &n_blocks, n_shared, kernel_params);
            %(err_check)s
            %(sync)s
         }
        }
        """ % locals(), file=sio)

    def c_code_reduce_1(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
            size_t n_blocks[3] = {1, 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_11(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;

            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 256), 1, 1};
            while (n_threads[1] * n_threads[0] <= 256) ++n_threads[1];
            n_threads[1] -= 1;
            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[0])
                n_threads[1] = PyGpuArray_DIMS(%(x)s)[0];

            size_t n_blocks[3] = {1, 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_01X(self, sio, node, name, x, z, fail, N):
        """

        Parameters
        ----------
        N
            The number of 1 in the pattern N=1 -> 01, N=2 -> 011 N=3 ->0111
            Work for N=1,2,3.

        """

        assert N in [1, 2, 3]
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        makecall = self._makecall(node, name, x, z, fail)
        N_pattern = ''.join(['1'] * N)
        param_dim = ",".join(["PyGpuArray_DIMS(%s)[%d]" % (x, i)
                              for i in xrange(N + 1)])
        strides_dim = ",".join(["PyGpuArray_STRIDES(%s)[%d]/sizeof(%s)"
                                % (x, i, in_dtype) for i in xrange(N + 1)])

        threads_y = """
            //get as many y threads as we can fit
            while (n_threads[0] * (n_threads[1]+1) <= 256)
            {
                if (n_threads[1] < PyGpuArray_DIMS(%(x)s)[%(N)s-1])
                    n_threads[1] += 1;
                else
                    break;
            }""" % locals()

        threads_z = """
            //get as many z threads as we can fit
            while (n_threads[0] * n_threads[1] * (n_threads[2]+1) <= 256)
            {
                if (n_threads[2] < PyGpuArray_DIMS(%(x)s)[%(N)s-2])
                    n_threads[2] += 1;
                else
                    break;
            }
            //Maximum for Fermi GPU on that dimensions.
            n_threads[2] = std::min(n_threads[2], (size_t)64);
        """ % locals()

        if len(self.reduce_mask) == 2:
            threads_y = ''
            threads_z = ''

        if len(self.reduce_mask) == 3:
            threads_z = ''

        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[%(N)s], (size_t) 256), 1, 1};
            %(threads_y)s
            %(threads_z)s
            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_01(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 1)

    def c_code_reduce_011(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 2)

    def c_code_reduce_0111(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 3)

    def c_code_reduce_10(self, sio, node, name, x, z, fail):
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        k_var = "kernel_reduce_10_%(name)s" % locals()
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
            """ % locals()
        print("""
    {
        int verbose = 0;
        if(PyGpuArray_STRIDES(%(x)s)[0]>
           PyGpuArray_STRIDES(%(x)s)[1]){
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                size_t n_threads[3] = {32, 1, 1};

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                GpuKernel *%(k_var)s = &kernel_reduce_010_AD_%(name)s;
                size_t A = 1;
                size_t B = PyGpuArray_DIMS(%(x)s)[0];
                size_t C = PyGpuArray_DIMS(%(x)s)[1];
                size_t D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                size_t n_blocks[3] = {A, D, 1};
                if (n_blocks[0] > 4096) n_blocks[0] = 4096;
                if (n_blocks[0]*n_blocks[1] > 4096) n_blocks[1] = 4096/n_blocks[0];
                ssize_t stride_A0 = 1;
                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
                ssize_t stride_Z0 = 1;
                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
                void *kernel_params[] = {
                        (void *)&A, (void *)&B, (void *)&C, (void *)&D,
                        (void *)%(x)s->ga.data,
                        (void *)&%(x)s->ga.offset,
                        (void *)&stride_A0, (void *)&stride_A1, (void *)&stride_A2,
                        (void *)%(z)s->ga.data,
                        (void *)&%(z)s->ga.offset,
                        (void *)&stride_Z0, (void *)&stride_Z1};
                int err = GpuKernel_call(%(k_var)s, 3, n_threads, n_blocks, 0, kernel_params);
                %(err_check)s
                %(sync)s
        }else{
            GpuKernel *%(k_var)s = &kernel_reduce_010_%(name)s;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
            size_t n_blocks[3] = {1, std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 4096), 1};
            if (verbose) {
              fprintf(stderr,
                "running kernel_reduce_10_%(name)s n_blocks=(%%llu,%%llu)\\n",
                (unsigned long long)n_blocks[0],
                (unsigned long long)n_blocks[1]);
            }
            assert(PyGpuArray_DIMS(%(x)s)[1] == PyGpuArray_DIMS(%(z)s)[0]);
            size_t n_shared = sizeof(%(acc_dtype)s) * n_threads[0];
            size_t dim_0 = 1;
            ssize_t stride_A0 = 1;
            ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
            ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
            ssize_t stride_Z0 = 1;
            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
            void *kernel_params[] = {
                    (void *)&dim_0,
                    (void *)&PyGpuArray_DIMS(%(x)s)[0],
                    (void *)&PyGpuArray_DIMS(%(x)s)[1],
                    (void *)%(x)s->ga.data, (void *)&%(x)s->ga.offset,
                    (void *)&stride_A0, (void *)&stride_A1, (void *)&stride_A2,
                    (void *)%(z)s->ga.data, (void *)&%(z)s->ga.offset,
                    (void *)&stride_Z0, (void *)&stride_Z1};
            int err = GpuKernel_call(%(k_var)s, 3, n_threads, n_blocks, n_shared, kernel_params);
            %(err_check)s
            %(sync)s
        }
    }
        """ % locals(), file=sio)

    def c_code_reduce_010(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        makecall_inner = self._makecall(node, name, x, z, fail,
                                        pattern="010_inner")
        pattern = ''.join(str(i) for i in self.reduce_mask)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        k_var = "kernel_reduce_010_AD_%(name)s" % locals()
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
            """ % locals()
        print("""
        {
            //int n_summations = PyGpuArray_DIMS(%(x)s)[0] * PyGpuArray_DIMS(%(x)s)[2];

            //if ((n_summations >= 15 * 32) && (PyGpuArray_DIMS(%(x)s)[2]>=16))
            if (1) // if the alternative is less buggy, consider not using this branch
            {
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                size_t n_threads[3] = {32, 1, 1};

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                size_t A = PyGpuArray_DIMS(%(x)s)[0];
                size_t B = PyGpuArray_DIMS(%(x)s)[1];
                size_t C = PyGpuArray_DIMS(%(x)s)[2];
                size_t D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                size_t n_blocks[3] = {A, D, 1};
                if (n_blocks[0] > 4096) n_blocks[0] = 4096;
                if (n_blocks[0]*n_blocks[1] > 4096) n_blocks[1] = 4096/n_blocks[0];
                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s);
                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1]/sizeof(%(out_dtype)s);
                void *kernel_params[] = {
                        (void *)&A, (void *)&B, (void *)&C, (void *)&D,
                        (void *)%(x)s->ga.data,
                        (void *)&%(x)s->ga.offset,
                        (void *)&stride_A0, (void *)&stride_A1, (void *)&stride_A2,
                        (void *)%(z)s->ga.data,
                        (void *)&%(z)s->ga.offset,
                        (void *)&stride_Z0, (void *)&stride_Z1};
                int err = GpuKernel_call(&%(k_var)s, 3, n_threads, n_blocks, 0, kernel_params);
                %(err_check)s
                %(sync)s
            }
            else
            {
                int verbose = 2;

                  size_t n_threads[3] = {std::min((size_t) 32, PyGpuArray_DIMS(%(x)s)[2]), 1, 1};
                  while(    (n_threads[0]*(n_threads[1]+1)<=256)
                         && (n_threads[1]<PyGpuArray_DIMS(%(x)s)[1])){
                      n_threads[1]++;
                  }

                  size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)4096), 1, 1};
                  n_blocks[1] = std::min(
                      ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
                                  (size_t)n_threads[0]),
                      (size_t)(4096 / n_blocks[0])
                      );
                if(std::min(std::min(PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
                                     PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s)),
                            PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s))
                   ==PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s)
                  && n_blocks[1]==ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
                                             (size_t)n_threads[0])){
                  if(verbose>1)
                    printf("n_block.x.1=%%d, n_block.x.2=%%d, n_block.y.1=%%d, n_block.y.2=%%d,\\n",
                           PyGpuArray_DIMS(%(x)s)[0],4096,
                           ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],(size_t)n_threads[0]),
                                       (size_t)(4096 / n_blocks[0]));
                  assert(n_threads[0]<=32);
                  %(makecall_inner)s
                }else{
                  n_threads[0] = std::min(PyGpuArray_DIMS(%(x)s)[1],
                                         (size_t) 256);
                  n_blocks[0] = std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)4096);
                  n_blocks[1] = std::min(
                      PyGpuArray_DIMS(%(x)s)[2],
                      (size_t)(4096 / n_blocks[0])
                      );
                  %(makecall)s
                }
                %(sync)s
            }
        }
        """ % locals(), file=sio)

    def c_code_reduce_0101(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};
            while (n_threads[0] * n_threads[1] <= 256)
            {
                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1]) break;
                n_threads[1] += 1;
            }
            n_threads[1] -= 1;
            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[0], PyGpuArray_DIMS(%(x)s)[2], 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_100(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        k_var = "kernel_reduce_010_AD_%(name)s" % locals()
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
            """ % locals()
        # use threadIdx.x for i0
        # use blockIdx.x for i1
        # use blockIdx.y for i2
        print("""
        {
            int verbose = 0;
            if (PyGpuArray_STRIDES(%(x)s)[2] != sizeof(%(in_dtype)s)){
              printf("slow\\n");
                size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
                size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)4096), 1, 1};
                while (n_blocks[0] * (n_blocks[1]+1) <= 4096 &&
                       n_blocks[1] <= PyGpuArray_DIMS(%(x)s)[2])
                {
                    n_blocks[1] += 1;
                }
                %(makecall)s
            }
            else
            {   // reuse 010_AD kernel, we transpose the 2 first dim
                // See the reduction for the real 010_AD kernel for
                // explanation. We do this to get coalesced read.
                size_t n_threads[3] = {32, 1, 1};

                size_t A = PyGpuArray_DIMS(%(x)s)[1];
                size_t B = PyGpuArray_DIMS(%(x)s)[0];
                size_t C = PyGpuArray_DIMS(%(x)s)[2];
                size_t D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                size_t n_blocks[3] = {A, D, 1};
                if (n_blocks[0] > 4096) n_blocks[0] = 4096;
                if (n_blocks[0]*n_blocks[1] > 4096) n_blocks[1] = 4096/n_blocks[0];
                size_t n_shared = 0;
                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s);
                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1]/sizeof(%(out_dtype)s);
                void *kernel_params[] = {
                        (void *)&A, (void *)&B, (void *)&C, (void *)&D,
                        (void *)%(x)s->ga.data,
                        (void *)&%(x)s->ga.offset,
                        (void *)&stride_A0, (void *)&stride_A1, (void *)&stride_A2,
                        (void *)%(z)s->ga.data,
                        (void *)&%(z)s->ga.offset,
                        (void *)&stride_Z0, (void *)&stride_Z1};
                int err = GpuKernel_call(&%(k_var)s, 3, n_threads, n_blocks, 0, kernel_params);
                %(err_check)s
                %(sync)s
            }
        }
        """ % locals(), file=sio)

    def c_code_reduce_110(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 256), 1, 1};
            while (n_threads[0]*n_threads[1] <= 256)
            {
                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads[1] += 1;
            }
            n_threads[1] -= 1;

            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[2], 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_001(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};
            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};
            while (n_blocks[0] * n_blocks[1] <= 4096)
            {
                if (n_blocks[1] > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_blocks[1] += 1;
            }
            n_blocks[1] -= 1;
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_101(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail,
                                  extra_dims=[("size_t one = 1;", "(void *) &one")],
                                  extra_strides=[("ssize_t sone = 1;", "(void *) &sone")],
                                  pattern="1011")
        print("""
        {
            int verbose = 0;
//            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3],
//                                            (size_t) 256), 1, 1};
            size_t n_threads[3] = {1, 1, 1};

            while (n_threads[0] * (n_threads[1]+1) <= 256) ++n_threads[1];
            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[2])
                n_threads[1] = PyGpuArray_DIMS(%(x)s)[2];

            while (n_threads[0] * n_threads[1] * (n_threads[2]+1) <= 256)
                ++n_threads[2];
            if (n_threads[2] > 64)
                n_threads[2] = 64;
            if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
                n_threads[2] = PyGpuArray_DIMS(%(x)s)[0];

            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[1], 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};

            //get as many y threads as we can fit
            while (n_threads[0] * n_threads[1] <= 256)
            {
                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_threads[1] += 1;
            }
            n_threads[1] -= 1;

            //get as many z threads as we can fit
            while (n_threads[0] * n_threads[1] * n_threads[2] <= 256)
            {
                if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads[2] += 1;
            }
            n_threads[2] -= 1;
            //Maximum for Fermi GPU on that dimensions.
            n_threads[2] = std::min(n_threads[2], (size_t)64);

            size_t n_blocks[3] = {1, 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_0011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        print("""
        {
            int verbose = 0;

            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};

            while (n_blocks[0] * n_blocks[1] <= 4096 &&
                   n_blocks[1] < PyGpuArray_DIMS(%(x)s)[1])
            {
                n_blocks[1] += 1;
            }

            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};
            while (n_threads[0] * n_threads[1] <= 256
                   && n_threads[1] < PyGpuArray_DIMS(%(x)s)[2]
                   && n_threads[0] * n_threads[1] * sizeof(%(acc_dtype)s) <=(15*1024-200))
            {
                n_threads[1] += 1;
            }

            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_1111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};

            //get as many y threads as we can fit
            while (n_threads[0] * n_threads[1] <= 256)
            {
                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_threads[1] += 1;
            }
            n_threads[1] -= 1;

            //get as many z threads as we can fit
            while (n_threads[0] * n_threads[1] * n_threads[2] <= 256)
            {
                if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads[2] += 1;
            }
            n_threads[2] -= 1;

            //Maximum for Fermi GPU on that dimensions.
            n_threads[2] = std::min(n_threads[2], (size_t)64);

            size_t n_blocks[3] = {1, 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_1011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};

            while (n_threads[0] * (n_threads[1]+1) <= 256) ++n_threads[1];
            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[2])
                n_threads[1] = PyGpuArray_DIMS(%(x)s)[2];

            while (n_threads[0] * n_threads[1] * (n_threads[2]+1) <= 256) ++n_threads[2];
            if (n_threads[2] > 64)
                n_threads[2] = 64;
            if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
                n_threads[2] = PyGpuArray_DIMS(%(x)s)[0];

            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[1], 1, 1};
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_cache_version_apply(self, node):
        version = [18]  # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(
            self.scalar_op,
            [Scalar(dtype=input.type.dtype)() for input in node.inputs],
            [Scalar(dtype=output.type.dtype)() for output in node.outputs])
        version.extend(self.scalar_op.c_code_cache_version_apply(scalar_node))
        for i in node.inputs + node.outputs:
            version.extend(Scalar(dtype=i.type.dtype).c_code_cache_version())
        version.extend(self.kernel_version(node))
        if all(version):
            return tuple(version)
        else:
            return ()

    def gpu_kernels(self, node, nodename):
        nd_in = len(self.reduce_mask)
        in_dtype = node.inputs[0].dtype
        out_dtype = node.outputs[0].dtype
        acc_dtype = self._acc_dtype(node.inputs[0].dtype)
        flags = Kernel.get_flags(in_dtype, acc_dtype, out_dtype)
        in_type = gpuarray.dtype_to_ctype(in_dtype)
        out_type = gpuarray.dtype_to_ctype(out_dtype)
        acc_type = gpuarray.dtype_to_ctype(acc_dtype)
        load_in = load_w(in_dtype)
        write_out = write_w(out_dtype)
        kernels = []

        if all(i == 1 for i in self.reduce_mask):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[0])")
            kname = "kernel_reduce_ccontig"
            k_var = "kernel_reduce_ccontig_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0,
                    const %(in_type)s *A, const ga_size offset_A,
                    %(out_type)s *Z, const ga_size offset_Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = %(reduce_init)s;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    %(reduce_fct)s
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            params = [
                'uintp',
                gpuarray.GpuArray, 'uintp',
                gpuarray.GpuArray, 'uintp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1,):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[0])")
            kname = "kernel_reduce_1"
            k_var = "kernel_reduce_1_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0,
                    %(out_type)s * Z, const ga_size offset_Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = %(reduce_init)s;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    %(reduce_fct)s
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            params = [
                'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp',
                gpuarray.GpuArray, 'uintp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 1):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[0])")
            kname = "kernel_reduce_11"
            k_var = "kernel_reduce_11_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0, const ga_size d1,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0, const ga_ssize sA1,
                    %(out_type)s * Z, const ga_size offset_Z)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y*blockDim.x + threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = %(reduce_init)s;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        %(reduce_fct)s;
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp',
                gpuarray.GpuArray, 'uintp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        # 01, 011, 0111
        if (0 == self.reduce_mask[0] and
                all(self.reduce_mask[1:]) and
                nd_in in[2, 3, 4]):
            # this kernel uses one block for each row.
            # threads per block for each element per row.

            N_pattern = ''.join(['1'] * (nd_in - 1))
            # TODO: is it faster to hardcode sA3, etc. in the later
            # code, rather than have the for_* variables declare them
            # and the later code use their names?
            if nd_in == 2:
                for_i1 = "for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)"
                first_i1 = 'threadIdx.x'
                sA1 = 'sA1'
                for_i2 = "int i2=0, sA2=0;"
                sA2 = '0'
                first_i2 = '0'
                for_i3 = "int i3=0, sA3=0;"
                sA3 = '0'
                first_i3 = '0'
            if nd_in == 3:
                for_i1 = "for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)"
                first_i1 = 'threadIdx.y'
                sA1 = 'sA1'
                for_i2 = "for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)"
                first_i2 = 'threadIdx.x'
                sA2 = 'sA2'
                for_i3 = "int i3=0, sA3=0;"
                first_i3 = 0
                sA3 = '0'
            if nd_in == 4:
                for_i1 = "for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)"
                first_i1 = 'threadIdx.z'
                sA1 = 'sA1'
                for_i2 = "for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)"
                first_i2 = 'threadIdx.y'
                sA2 = 'sA2'
                for_i3 = "for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)"
                first_i3 = 'threadIdx.x'
                sA3 = 'sA3'

            reducebuf = self._k_reduce_buf('Z[i0 * sZ0]', node,
                                           nodename, sub={})
            param_dim = ",".join(["const ga_size d%d" % i
                                  for i in xrange(nd_in)])
            param_strides = ",".join(["const ga_ssize sA%d" % i
                                      for i in xrange(nd_in)])
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_init = self._assign_init(load_in + "(A[%(first_i3)s * %(sA3)s + %(first_i2)s * %(sA2)s + %(first_i1)s * %(sA1)s + i0 * sA0])" % locals())
            reduce_fct = self._assign_reduce(
                node, nodename, "myresult",
                load_in + "(A[i3 * sA3 + i2 * sA2 + i1 * sA1 + i0 * sA0])",
                {}, True)
            sio = StringIO()
            print("""
                %(decl)s{
                    %(init)s
                    for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
                      myresult = %(reduce_init)s;
                      %(for_i1)s{
                        %(for_i2)s{
                          %(for_i3)s{
                            %(reduce_fct)s;
                          }
                        }
                      }
                      %(reducebuf)s
                    }
                }
                """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (0, 1, 0) or self.reduce_mask == (1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            # TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2*sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i0 * sA0 + threadIdx.x * sA1 + i2 * sA2])")
            kname = "kernel_reduce_010"
            k_var = "kernel_reduce_010_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0, const ga_size d1, const ga_size d2,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0, const ga_ssize sA1, const ga_ssize sA2,
                    %(out_type)s * Z, const ga_size offset_Z,
                    const ga_ssize sZ0, const ga_ssize sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }


                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        %(acc_type)s myresult = %(reduce_init)s;
                        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }

            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp', 'intp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask in [(0, 1, 0), (1, 0), (1, 0, 0)]:
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(X[a * sX0 + b * sX1 + c * sX2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(X[a * sX0 + 0 * sX1 + c * sX2])")
            kname = "kernel_reduce_010_AD"
            k_var = "kernel_reduce_010_AD_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size A, const ga_size B, const ga_size C, const ga_size D,
                    const %(in_type)s *X, const ga_size offset_X,
                    const ga_ssize sX0, const ga_ssize sX1, const ga_ssize sX2,
                    %(out_type)s * Z, const ga_size offset_Z,
                    const ga_ssize sZ0, const ga_ssize sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                %(acc_type)s myresult = 0;
                X = (const %(in_type)s *)(((char *)X)+offset_X);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int a = blockIdx.x; a < A; a += gridDim.x)
                {
                    for (int i2_D = blockIdx.y; i2_D < D; i2_D += gridDim.y)
                    {
                        int c = i2_D * 32 + threadIdx.x;
                        if (c < C)
                        {
                            myresult = %(reduce_init)s;
                            for (int b = 0; b < B; ++b)
                            {
                                %(reduce_fct)s;
                            }
                            Z[a * sZ0 + c * sZ1] = %(write_out)s(myresult);
                        }
                    }
                }

            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp', 'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp', 'intp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (0, 1, 0):
            #
            # This kernel is optimized when the inner most dimensions
            # have the smallest stride.

            # this kernel uses one block for multiple column(up to 32TODO),
            # threads per block for each element per column.

            # thread.x = dim 2 contiguous
            # thread.y = dim 1
            # block.x = dim 0
            # block.y = dim 1 rest
            init = self._k_init(node, nodename)
            decl, kname, params, k_var = self._k_decl(node, nodename, pattern="010_inner")
            reducebuf = self._k_reduce_buf_multiple('Z[i0 * sZ0 + i2*sZ1]',
                                                    node, nodename,
                                                    'blockDim.x')
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i0 * sA0 + 0 * sA1 + i2 * sA2])")
            sio = StringIO()
            print("""
            %(decl)s
            {
             if(warpSize<blockDim.x){
               //TODO: set error code
               Z[0] = -666;
               return;
              }

              %(init)s
              for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
              {
                for (int i2 = blockIdx.y*blockDim.x+threadIdx.x; i2 < d2; i2 += gridDim.y*blockDim.x)
                 {
                  myresult = %(reduce_init)s;
                  for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                  {
                      %(reduce_fct)s;
                  }
                  %(reducebuf)s
                 }
              }
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            # TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[blockIdx.x * sZ0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + blockIdx.x * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[blockIdx.x * sA2])")
            kname = "kernel_reduce_110"
            k_var = "kernel_reduce_110_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0, const ga_size d1, const ga_size d2,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0, const ga_ssize sA1, const ga_ssize sA2,
                    %(out_type)s * Z, const ga_size offset_Z,
                    const ga_ssize sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = %(reduce_init)s;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    //TODO: set error code
                    Z[blockIdx.x * sZ0] = %(write_out)s(-666);
                    return;
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        %(reduce_fct)s;
                    }
                }

                %(reducebuf)s
            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp', 'intp',
                gpuarray.GpuArray, 'uintp',
                'intp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 0, 0):
            reducebuf = self._k_reduce_buf('Z[i1 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i1 * sA1 + i2 * sA2])")
            sio = StringIO()
            print("""
            %(decl)s
            {
                %(init)s
                for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                {
                    for (int i1 = blockIdx.x; i1 < d1; i1 += gridDim.x)
                    {
                        myresult = %(reduce_init)s;
                        for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                        {
                            %(reduce_fct)s
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node,
                                           nodename, sub={})
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[0])")
            sio = StringIO()
            print("""
            %(decl)s
            {
                %(init)s
                myresult = %(reduce_init)s;
                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (0, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i0 * sA0 + i1 * sA1])")
            kname = "kernel_reduce_001"
            k_var = "kernel_reduce_001_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0, const ga_size d1, const ga_size d2,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0, const ga_ssize sA1, const ga_ssize sA2,
                    %(out_type)s * Z, const ga_size offset_Z,
                    const ga_ssize sZ0, const ga_ssize sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        %(acc_type)s myresult = %(reduce_init)s;
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp', 'intp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (0, 0, 1, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i0 * sA0 + i1 * sA1])")
            sio = StringIO()
            print("""
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        %(acc_type)s myresult = %(reduce_init)s;
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (0, 1, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[i0 * sA0 + i2 * sA2])")
            sio = StringIO()
            print("""
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        %(acc_type)s myresult = %(reduce_init)s;
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename,
                                           sub={})
            decl, kname, params, k_var = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[0])")
            sio = StringIO()
            print("""
            %(decl)s
            {
                %(init)s
                myresult = %(reduce_init)s;
              for (int i0 = 0; i0 < d0; i0++)
                for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        if self.reduce_mask == (1, 0, 1, 1) or self.reduce_mask == (1, 0, 1):
            reducebuf = self._k_reduce_buf('Z[blockIdx.x*sZ0]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             load_in + "(A[i0 * sA0 + blockIdx.x * sA1 + i2 * sA2 + i3 * sA3])",
                                             {}, True)
            reduce_init = self._assign_init(load_in + "(A[blockIdx.x * sA1])")
            kname = "kernel_reduce_1011"
            k_var = "kernel_reduce_1011_" + nodename
            sio = StringIO()
            print("""
            KERNEL void %(kname)s(
                    const ga_size d0, const ga_size d1, const ga_size d2, const ga_size d3,
                    const %(in_type)s *A, const ga_size offset_A,
                    const ga_ssize sA0, const ga_ssize sA1, const ga_ssize sA2, const ga_ssize sA3,
                    %(out_type)s * Z, const ga_size offset_Z,
                    const ga_ssize sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_type)s buf[];
                %(acc_type)s myresult = %(reduce_init)s;
                A = (const %(in_type)s *)(((char *)A)+offset_A);
                Z = (%(out_type)s *)(((char *)Z)+offset_Z);

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
            params = [
                'uintp', 'uintp', 'uintp', 'uintp',
                gpuarray.GpuArray, 'uintp',
                'intp', 'intp', 'intp', 'intp',
                gpuarray.GpuArray, 'uintp',
                'intp'
                ]
            kernels.append(Kernel(code=sio.getvalue(), name=kname,
                                  params=params, flags=flags, objvar=k_var))
        return kernels


class GpuCAReduceCPY(GpuKernelBase, HideC, CAReduceDtype):
    """
    CAReduce that reuse the python code from gpuarray.

    """
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
        ctx_name = infer_context_name(input)
        res = CAReduceDtype.make_node(self, input)
        input = as_gpuarray_variable(input, ctx_name)
        otype = GpuArrayType(dtype=res.outputs[0].dtype,
                             broadcastable=res.outputs[0].broadcastable,
                             context_name=ctx_name)

        if res.op.axis is not None:
            redux = []
            for i in range(len(input.type.broadcastable)):
                redux.append(i in res.op.axis)
                # since redux is just another way to describe what is in axis
                # it doesn't need to be compared in __eq__ or __hash__
            res.op.redux = redux

        return Apply(res.op, [input], [otype()])

    def get_params(self, node):
        return node.outputs[0].type.context

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        # cache the kernel object
        self.get_kernel_cache(node)
        return super(GpuCAReduceCPY, self).make_thunk(
            node, storage_map, compute_map, no_recycling)

    def get_kernel_cache(self, node):
        attr = '@cache_reduction_k'
        if self.axis is None:
            redux = [True] * node.inputs[0].ndim
        else:
            redux = self.redux
        if not hasattr(node, attr):
            acc_dtype = getattr(self, 'acc_dtype', None)
            if acc_dtype is None:
                acc_dtype = node.outputs[0].type.dtype
            if any(redux):
                setattr(node, attr, self.generate_kernel(node, acc_dtype,
                                                         redux))

        if any(redux):
            return getattr(node, attr)

    def gpu_kernels(self, node, name):
        if not any(getattr(self, 'redux', [node.inputs[0].ndim != 0])):
            # Some OpenCL compilers do not accept no-arguments kernels
            src = "KERNEL void reduk(GLOBAL_MEM float *a) {}"
            params = ['float32']
        else:
            k = self.get_kernel_cache(node)
            _, src, _, _ = k._get_basic_kernel(k.init_local_size,
                                               node.inputs[0].ndim)
            nd = node.inputs[0].ndim
            params = ['uint32', gpuarray.GpuArray]
            params.extend('uint32' for _ in range(nd))
            params.append(gpuarray.GpuArray)
            params.append('uint32')
            params.extend('int32' for _ in range(nd))
        acc_dtype = getattr(self, 'acc_dtype', None)
        if acc_dtype is None:
            acc_dtype = node.outputs[0].type.dtype
        return [Kernel(code=src, name="reduk", params=params,
                       flags=Kernel.get_flags(node.inputs[0].type.dtype,
                                              acc_dtype,
                                              node.outputs[0].type.dtype),
                       objvar='k_reduk_' + name)]

    def c_code(self, node, name, inp, out, sub):
        if not any(getattr(self, 'redux', [node.inputs[0].ndim != 0])):
            # We special case the no-reduction case since the gpu
            # kernel has trouble handling it.
            return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_copy(%(inp)s, GA_ANY_ORDER);
        if (!%(out)s) {
            %(fail)s
        }

        if (%(sync)d)
            GpuArray_sync(&%(out)s->ga);
        """ % dict(out=out[0], inp=inp[0], fail=sub['fail'],
                   sync=bool(config.gpuarray.sync))
        k = self.get_kernel_cache(node)
        _, src, _, ls = k._get_basic_kernel(k.init_local_size,
                                            node.inputs[0].ndim)
        if self.axis is None:
            redux = [True] * node.inputs[0].ndim
        else:
            redux = self.redux
        acc_dtype = getattr(self, 'acc_dtype', None)
        if acc_dtype is None:
            acc_dtype = node.outputs[0].type.dtype
        input = inp[0]
        output = out[0]
        nd_out = node.outputs[0].ndim
        code = """
        size_t gs = 1;
        size_t ls;
        unsigned int n = 1;
        unsigned int proxy_dim[%(nd_in)s];
        unsigned int proxy_off;
        int proxy_str[%(nd_in)s];
        void *args[%(n_args)s];
        PyGpuArrayObject *tmp;
        int err;
""" % dict(n_args=4 + (node.inputs[0].ndim * 2), nd_in=node.inputs[0].ndim)

        if nd_out != 0:
            code += """
        size_t out_dims[%(nd_out)s];
        int need_out = %(output)s == NULL || %(output)s->ga.nd != %(nd_out)s;
""" % dict(nd_out=nd_out, output=output)
            j = 0
            for i in range(node.inputs[0].ndim):
                if not self.redux[i]:
                    code += """
         out_dims[%(j)s] = %(input)s->ga.dimensions[%(i)s];
         if (!need_out)
             need_out |= %(output)s->ga.dimensions[%(j)s] != out_dims[%(j)s];
""" % dict(j=j, i=i, input=input, output=output)
                    j += 1
            code += """
         if (need_out) {
             %(output)s = pygpu_empty(%(nd_out)s, out_dims, %(out_type)s, GA_C_ORDER, %(ctx)s, Py_None);
             if (!%(output)s) {
                 %(fail)s
             }
         }
        """ % dict(output=output, nd_out=nd_out, fail=sub['fail'],
                   ctx=sub['params'],
                   out_type=dtype_to_typecode(node.outputs[0].type.dtype))
        else:
            code += """
        if (%(output)s == NULL || %(output)s->ga.nd != 0) {
            Py_XDECREF(%(output)s);
            %(output)s = pygpu_empty(0, NULL, %(out_type)s, GA_C_ORDER,
                                     %(ctx)s, Py_None);
            if (!%(output)s) {
                %(fail)s
            }
        }
        """ % dict(output=output, fail=sub['fail'], ctx=sub['params'],
                   out_type=dtype_to_typecode(node.outputs[0].type.dtype))

        if acc_dtype != node.outputs[0].type.dtype:
            code += """
        tmp = pygpu_empty(%(output)s->ga.nd, %(output)s->ga.dimensions,
                          %(acc_type)s, GA_C_ORDER, %(ctx)s, Py_None);
        if (!tmp) %(fail)s
        """ % dict(output=output, fail=sub['fail'], ctx=sub['params'],
                   acc_type=dtype_to_typecode(acc_dtype))
        else:
            code += """
        tmp = %(output)s;
        Py_INCREF(tmp);
        """ % dict(output=output)

        # We need the proxies since we are passing a pointer to the
        # data into the call and therefore we need a real copy of the
        # data in the proper type.
        code += """
        args[0] = &n;
        args[1] = tmp->ga.data;
        """ % dict(output=output)

        p = 2
        for i in range(node.inputs[0].ndim):
            code += """
        proxy_dim[%(i)s] = %(input)s->ga.dimensions[%(i)s];
        args[%(p)s] = &proxy_dim[%(i)s];
        n *= %(input)s->ga.dimensions[%(i)s];
        """ % dict(i=i, p=p, input=input)
            p += 1
            if not redux[i]:
                code += "gs *= %(input)s->ga.dimensions[%(i)s];" % dict(input=input, i=i)

        code += """
        args[%(p)s] = %(input)s->ga.data;
        proxy_off = %(input)s->ga.offset;
        args[%(p)s+1] = &proxy_off;
        """ % dict(p=p, input=input)
        p += 2

        for i in range(node.inputs[0].ndim):
            code += """
        proxy_str[%(i)s] = %(input)s->ga.strides[%(i)s];
        args[%(p)s] = &proxy_str[%(i)s];
        """ % dict(p=p, i=i, input=input)
            p += 1

        code += """
        if (gs == 0) gs = 1;
        n /= gs;
        ls = %(ls)s;
        err = GpuKernel_call(&%(k_var)s, 1, &ls, &gs, 0, args);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "gpuarray error: GpuCAReduceCPY: %%s.",
                         GpuKernel_error(&%(k_var)s, err));
            %(fail)s
        }

        if (%(cast_out)d) {
            err = GpuArray_move(&%(output)s->ga, &tmp->ga);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: GpuCAReduceCPY [cast]: %%s.",
                             GpuArray_error(&tmp->ga, err));
                %(fail)s
            }
        } else {
            Py_XDECREF(%(output)s);
            %(output)s = tmp;
        }

        if (%(sync)d) {
            err = GpuArray_sync(&%(output)s->ga);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: GpuCAReduceCPY: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s
            }
        }
        """ % dict(k_var='k_reduk_' + name, sync=bool(config.gpuarray.sync),
                   ls=ls, fail=sub['fail'], output=output, input=input,
                   cast_out=bool(acc_dtype != node.outputs[0].type.dtype))

        return code

    def c_code_cache_version(self):
        return (2, self.GpuKernelBase_version)

    def generate_kernel(self, node, odtype, redux):
        if isinstance(self.scalar_op, scalar.basic.Add):
            reduce_expr = "a + b"
        elif isinstance(self.scalar_op, scalar.basic.Mul):
            reduce_expr = "a * b"
        else:
            raise NotImplementedError()
        return ReductionKernel(node.inputs[0].type.context, odtype,
                               self.scalar_op.identity, reduce_expr, redux,
                               arguments=[make_argument(node.inputs[0], 'a')],
                               init_nd=node.inputs[0].ndim)

    def perform(self, node, inp, out, ctx):
        input, = inp
        output, = out

        if self.axis is None:
            redux = [True] * input.ndim
        else:
            redux = self.redux

        if any(redux):
            output[0] = self.get_kernel_cache(node)(input).astype(
                copy=False, dtype=node.outputs[0].type.dtype)
        else:
            output[0] = pygpu.gpuarray.array(input, copy=True,
                                             dtype=node.outputs[0].type.dtype,
                                             context=ctx)
# To allow reloading old pickled files
GpuCAReduce = GpuCAReduceCPY
