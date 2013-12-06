import copy
from itertools import izip

import numpy
from theano import Op, Apply, scalar, config
from theano.tensor.elemwise import Elemwise, DimShuffle, CAReduceDtype
from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler

try:
    import pygpu
    from pygpu.tools import ScalarArg, ArrayArg
    from pygpu.elemwise import ElemwiseKernel
    from pygpu.reduction import ReductionKernel
    from pygpu.gpuarray import dtype_to_typecode, dtype_to_ctype
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import (as_gpuarray_variable, HideC,
                                               GpuKernelBase)
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
        node = Apply(self, inputs, outputs)

        # Try to generate the kernel to catch SupportCodeErrors
        try:
            inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                    enumerate(node.inputs)]
            scal_ins = [scalar.Scalar(i.dtype) for i in node.inputs]

            outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                    enumerate(node.outputs) if not n in self.inplace_pattern]
            scal_out = [scalar.Scalar(o.dtype) for o in node.outputs]

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

    def generate_kernel(self, node, nodename):
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(node.inputs)]
        scal_ins = [scalar.Scalar(i.dtype) for i in node.inputs]

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(node.outputs) if not n in self.inplace_pattern]
        scal_out = [scalar.Scalar(o.dtype) for o in node.outputs]

        fake_node = Apply(self.scalar_op, [i() for i in scal_ins],
                          [o() for o in scal_out])

        scal_out = []
        oi = 0
        for n in range(len(node.outputs)):
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
        support_code = """
#ifdef _MSC_VER
#define signed __int8 int8_t
#define unsigned __int8 uint8_t
#define signed __int16 int16_t
#define unsigned __int16 uint16_t
#define signed __int32 int32_t
#define unsigned __int32 uint32_t
#define signed __int64 int64_t
#define unsigned __int64 uint64_t
#else
#include <stdint.h>
#endif
#define ga_bool uint8_t
#define ga_byte int8_t
#define ga_ubyte uint8_t
#define ga_short int16_t
#define ga_ushort uint16_t
#define ga_int int32_t
#define ga_uint uint32_t
#define ga_long int64_t
#define ga_ulong uint64_t
#define ga_float float
#define ga_double double
#define ga_half uint16_t

"""
        return ElemwiseKernel(None, inps+outs, kop, preamble=support_code)

    def c_headers(self):
        return ['cuda.h', '<compyte/extension.h>', '<compyte/numpy_compat.h>']

    def c_compiler(self):
        return NVCC_compiler

    def c_support_code_apply(self, node, nodename):
        # This is useless by itself, but will serve an eventual c_code
        # implementation
        k = self.generate_kernel(node, nodename)
        nd = node.inputs[0].type.ndim
        import pycuda._cluda
        res = ["CUdeviceptr (*cuda_get_ptr)(gpudata *g);",
               pycuda._cluda.CLUDA_PREAMBLE]
        for i in range(0, nd + 1):
            res.append(k.render_basic(i, name="elem_" + str(i)) + ';')
        res.append(k.contig_src + ';')

        return '\n'.join(res)

    def c_init_code(self):
        return ['cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))'
                'compyte_get_extension("cuda_get_ptr");']

    def c_code(self, node, name, inputs, outputs, sub):
        nd = node.outputs[0].ndim
        fail = sub["fail"]
        initial_dims = ','.join('1' for i in xrange(nd))
        opname = str(self.scalar_op)

        #check that all inputs have valid dimensions
        emitted_inames = {}
        code = """
        int n_blocks = 0;
        int threads_per_block = 0;
        size_t numEls = 0;
        """
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

        #check that all inputs have valid dimensions
        emitted_inames = {}
        for idx, iname in enumerate(inputs):
            if iname in emitted_inames:
                continue
            code += """
        //std::cerr << "C_CODE %(opname)s checking input %(iname)s\\n";
        if (%(nd)s != PyGpuArray_NDIM(%(iname)s))
        {
            PyErr_Format(PyExc_TypeError,
                         "need %(nd)s dims, not %%i",
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
                //std::cerr << "C_CODE %(opname)s checking input %(iname)s failed\\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " %(idx)d (indices start at 0) has shape[%%i] == %%i"
                             ", but the output's size on that axis is %%i.",
                             i,
                             PyGpuArray_DIMS(%(iname)s)[i],
                             dims[i]
                            );
                %(fail)s;
            }
        }
            """ % locals()
            emitted_inames[iname] = True
        #check that all outputs have valid dimensions
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
                            pygpu_default_context(), Py_None);
            if (!%(oname)s) {
                        //TODO, this check don't seam good.
                        //TODO, set exception?
                            %(fail)s
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << PyGpuArray_NDIM(%(oname)s) << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
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
                             " on input %(input_idx)s, has shape[%%i] == %%i"
                             ", but the output's size on that axis is %%i.",
                             i,
                             PyGpuArray_DIMS(%(oname)s)[i],
                             dims[i]
                            );
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << PyGpuArray_NDIM(%(oname)s) << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
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

                //std::cerr << "calling callkernel returned\\n";
        """ % locals()

        code += "elem_%(nd)s<<<n_blocks, threads_per_block>>>(numEls,\n" % locals()
        param = []
        for i in range(nd):
            param.append("%(z)s->ga.dimensions[%(i)d]" % dict(z=outputs[0],
                                                              i=i))
        for n, (name, var) in enumerate(zip(inputs + outputs,
                                       node.inputs + node.outputs)):
            if (n - len(inputs)) in self.inplace_pattern:
                continue
            dtype = var.dtype
            param.append("(npy_%(dtype)s*)(cuda_get_ptr(%(name)s->ga.data))" % locals())
            param.append("%(name)s->ga.offset" % locals())
            for i in range(nd):
                param.append("PyGpuArray_DIMS(%(name)s)[%(i)d] == 1 ? 0 : PyGpuArray_STRIDES(%(name)s)[%(i)d]" % locals())
        code += ',\n'.join(param) + ");\n"
        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(zz)s->ga);\n" % dict(zz=zz)
        return str(code)

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

        node._cache_elemwise_k(*args, broadcast=True)
        if config.gpuarray.sync:
            output_storage[0][0].sync()

    def c_code_cache_version(self):
        ver = self.scalar_op.c_code_cache_version()
        if ver:
            return (1, ver)
        else:
            return ver


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


class GpuCAReduce(GpuKernelBase, HideC, CAReduceDtype):
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
        # cache the kernel object
        self.get_kernel_cache(node)
        return super(GpuCAReduce, self).make_thunk(node, storage_map,
                                                   compute_map, no_recycling)

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

    def c_kernel_code(self, node):
        if not any(getattr(self, 'redux', [node.inputs[0].ndim != 0])):
            # Some OpenCL compilers do not accept no-arguments kernels
            return "KERNEL void reduk(GLOBAL_MEM float *a) {}"
        else:
            k = self.get_kernel_cache(node)
            _, src, _, _ = k._get_basic_kernel(k.init_local_size,
                                               node.inputs[0].ndim)
            return src

    def c_kernel_name(self):
        return "reduk"

    def c_kernel_params(self, node):
        if not any(getattr(self, 'redux', [node.inputs[0].ndim != 0])):
            return ["GA_FLOAT"]
        else:
            # Make sure this is synced with the call definition in
            # pygpu/reduction.py
            nd = node.inputs[0].ndim
            res = ["GA_UINT", "GA_BUFFER"]
            res.extend("GA_UINT" for _ in range(nd))
            res.append("GA_BUFFER")
            res.append("GA_UINT")
            res.extend("GA_INT" for _ in range(nd))
            return res

    def c_kernel_flags(self, node):
        acc_dtype = getattr(self, 'acc_dtype', None)
        if acc_dtype is None:
            acc_dtype = node.outputs[0].type.dtype
        return self._get_kernel_flags(node.inputs[0].type.dtype,
                                      acc_dtype,
                                      node.outputs[0].type.dtype)

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
             %(output)s = pygpu_empty(%(nd_out)s, out_dims, %(out_type)s, GA_C_ORDER, pygpu_default_context(), Py_None);
             if (!%(output)s) {
                 %(fail)s
             }
         }
""" % dict(output=output, nd_out=nd_out, fail=sub['fail'],
           out_type=dtype_to_typecode(node.outputs[0].type.dtype))
        else:
            code += """
        if (%(output)s == NULL || %(output)s->ga.nd != 0) {
            Py_XDECREF(%(output)s);
            %(output)s = pygpu_empty(0, NULL, %(out_type)s, GA_C_ORDER,
                                     pygpu_default_context(), Py_None);
            if (!%(output)s) {
                %(fail)s
            }
        }
""" % dict(output=output, fail=sub['fail'],
           out_type=dtype_to_typecode(node.outputs[0].type.dtype))

        if acc_dtype != node.outputs[0].type.dtype:
            code += """
        tmp = pygpu_empty(%(output)s->ga.nd, %(output)s->ga.dimensions,
                          %(acc_type)s, GA_C_ORDER, pygpu_default_context(),
                          Py_None);
        if (!tmp) %(fail)s
""" % dict(output=output, fail=sub['fail'], acc_type=dtype_to_typecode(acc_dtype))
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
        args[1] = &tmp->ga;
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
        args[%(p)s] = &%(input)s->ga;
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
        err = GpuKernel_call(&%(k_var)s, 0, %(ls)s, gs, args);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "compyte error: GpuCAReduce: %%s.",
                         GpuKernel_error(&%(k_var)s, err));
            %(fail)s
        }

        if (%(cast_out)d) {
            err = GpuArray_move(&%(output)s->ga, &tmp->ga);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "compyte error: GpuCAReduce [cast]: %%s.",
                             GpuArray_error(&tmp->ga, err));
                %(fail)s
            }
        } else {
            Py_XDECREF(%(output)s);
            %(output)s = tmp;
        }

        if (%(sync)d)
            GpuArray_sync(&%(output)s->ga);
""" % dict(k_var=self.c_kernel_obj(name), sync=bool(config.gpuarray.sync),
           ls=ls, fail=sub['fail'], output=output, input=input,
           cast_out=bool(acc_dtype != node.outputs[0].type.dtype))

        return code

    def c_code_cache_version(self):
        return (0,)

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
                               init_nd=node.inputs[0].ndim)

    def perform(self, node, inp, out):
        input, = inp
        output, = out

        if self.axis is None:
            redux = [True] * input.ndim
        else:
            redux = self.redux

        if any(redux):
            output[0] = self.get_kernel_cache(node)(input).astype(copy=False,
                                             dtype=node.outputs[0].type.dtype)
        else:
            output[0] = pygpu.gpuarray.array(input, copy=True,
                                             dtype=node.outputs[0].type.dtype)
