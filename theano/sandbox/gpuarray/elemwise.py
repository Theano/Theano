import copy
from itertools import izip
from StringIO import StringIO

import numpy

import theano
from theano import Apply, scalar, config
from theano import scalar as scal
from theano.scalar import Scalar
from theano.tensor.elemwise import (Elemwise, DimShuffle, CAReduceDtype)
from theano.sandbox.gpuarray.comp import NVCC_compiler

try:
    import pygpu
    from pygpu import gpuarray
    from pygpu.tools import ScalarArg, ArrayArg
    from pygpu.elemwise import ElemwiseKernel
    from pygpu.reduction import ReductionKernel
    from pygpu.gpuarray import dtype_to_typecode, dtype_to_ctype
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import (as_gpuarray_variable, HideC,
                                               GpuKernelBase, Kernel)
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

    def generate_kernel(self, node, nodename):
        inps = [make_argument(i, 'i%d' % (n,)) for n, i in
                enumerate(node.inputs)]
        scal_ins = [scalar.get_scalar_type(i.dtype) for i in node.inputs]

        outs = [make_argument(o, 'o%d' % (n,)) for n, o in
                enumerate(node.outputs) if not n in self.inplace_pattern]
        scal_out = [scalar.get_scalar_type(o.dtype) for o in node.outputs]

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
        try:
            #We accept only some c_support_code().
            #This filter is done in the make_node()
            support_code += self.scalar_op.c_support_code()
        except MethodNotDefined:
            pass
        for npy, ga in [("npy_uint8", "ga_ubyte"),
                        ("npy_uint16", "ga_ushort"),
                        ("npy_uin32", "ga_uint"),
                        ("npy_uin64", "ga_ulong"),
                        ("npy_int8", "ga_byte"),
                        ("npy_int16", "ga_short"),
                        ("npy_int32", "ga_int"),
                        ("npy_int64", "ga_long"),
                        ("npy_float32", "ga_float"),
                        ("npy_float64", "ga_double"),
            ]:
            kop = kop.replace(npy, ga)
        return ElemwiseKernel(None, inps+outs, kop, preamble=support_code)

    def c_headers(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>']

    def c_compiler(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        return NVCC_compiler

    def c_support_code(self):
        return self.scalar_op.c_support_code()

    def c_support_code_apply(self, node, nodename):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        # This is useless by itself, but will serve an eventual c_code
        # implementation
        k = self.generate_kernel(node, nodename)
        nd = node.inputs[0].type.ndim
        CLUDA_PREAMBLE = """
#define local_barrier() __syncthreads();

#define WITHIN_KERNEL __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_ARG /* empty */
#define REQD_WG_SIZE(X,Y,Z) __launch_bounds__(X*Y*Z, 1)

#define LID_0 threadIdx.x
#define LID_1 threadIdx.y
#define LID_2 threadIdx.z

#define GID_0 blockIdx.x
#define GID_1 blockIdx.y
#define GID_2 blockIdx.z

#define LDIM_0 blockDim.x
#define LDIM_1 blockDim.y
#define LDIM_2 blockDim.z

#define GDIM_0 gridDim.x
#define GDIM_1 gridDim.y
#define GDIM_2 gridDim.z
"""
        res = [CLUDA_PREAMBLE]
        for i in range(0, nd + 1):
            res.append(k.render_basic(i, name="elem_" + str(i)) + ';')
        res.append(k.contig_src + ';')

        return '\n'.join(res)

    def c_init_code(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        return ['setup_ext_cuda();']

    def c_code(self, node, name, inputs, outputs, sub):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
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
            dtype = dtype_to_ctype(var.dtype)
            param.append("(%(dtype)s*)(cuda_get_ptr(%(name)s->ga.data))" % locals())
            param.append("%(name)s->ga.offset" % locals())
            for i in range(nd):
                param.append("PyGpuArray_DIMS(%(name)s)[%(i)d] == 1 ? 0 : PyGpuArray_STRIDES(%(name)s)[%(i)d]" % locals())
        code += ',\n'.join(param) + ");\n"
        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(z)s->ga);\n" % dict(z=z)
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
            return (2, ver)
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
        return (4,)


class GpuCAReduceCuda(HideC, CAReduceDtype):
    """GpuCAReduceCuda is a Reduction along some dimensions by a scalar op.

    The dimensions along which to reduce is specified by the
    `reduce_mask` that you pass to the constructor.  The `reduce_mask`
    is a tuple of booleans (actually integers 0 or 1) that specify for
    each input dimension, whether to reduce it (1) or not (0).

    For example, when scalar_op is a theano.scalar.basic.Add instance:

      - reduce_mask == (1,) sums a vector to a scalar

      - reduce_mask == (1,0) computes the sum of each column in a matrix

      - reduce_mask == (0,1) computes the sum of each row in a matrix

      - reduce_mask == (1,1,1) computes the sum of all elements in a 3-tensor.

    :note: any reduce_mask of all zeros is a sort of 'copy', and may
           be removed during graph optimization

    This Op is a work in progress.

    This op was recently upgraded from just GpuSum a general CAReduce. Not
    many code cases are supported for scalar_op being anything other than
    scal.Add instances yet.

    Important note: if you implement new cases for this op, be sure to
    benchmark them and make sure that they actually result in a speedup.
    GPUs are not especially well-suited to reduction operations so it is
    quite possible that the GPU might be slower for some cases.

    pre_scalar_op: if present, must be a scalar op with only 1
    input. We will execute it on the input value before reduction.

    """

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

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.axis == other.axis and
                self.reduce_mask == other.reduce_mask and
                self.dtype == other.dtype and
                self.acc_dtype == other.acc_dtype and
                self.scalar_op == other.scalar_op and
                self.pre_scalar_op == other.pre_scalar_op)

    def __hash__(self):
        return (hash(type(self)) ^
                hash(self.axis) ^
                hash(self.reduce_mask) ^
                hash(self.dtype) ^
                hash(self.acc_dtype) ^
                hash(type(self.scalar_op)) ^
                hash(type(self.pre_scalar_op)))

    def __str__(self):
        pre = ""
        if self.pre_scalar_op:
            pre = "pre=%s,red=" % str(self.pre_scalar_op)
        ax = ''
        if self.axis is not None:
            ax = '{%s}' % (', '.join(str(x) for x in self.axis),)
        return "GpuCAReduceCuda{%s%s}%s" % (pre,str(self.scalar_op), ax)

    def __setstate__(self, d):
        self.__dict__.update(d)
        # For unpickling of old ops.
        if not hasattr(self, "pre_scalar_op"):
            self.pre_scalar_op = None

    def make_node(self, x):
        x = as_gpuarray_variable(x)
        ret = super(GpuCAReduceCuda, self).make_node(x)
        self = copy.copy(self)
        self.axis = ret.op.axis
        if self.pre_scalar_op:
            # Currently we only tested pre_scalar_op that don't cause
            # upcast.
            d1 = self.__class__(scalar_op=self.scalar_op)(Elemwise(self.pre_scalar_op)(x))
            assert d1.dtype == ret.outputs[0].dtype
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
        return Apply(self, [x], [GpuArrayType(ret.outputs[0].dtype,
                                              ret.outputs[0].type.broadcastable)()])

    """
    This method must be commented, because there's no way
    to communicate that it's OK to call for + but not for
    max
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        # reduce_max is declared but does nothing but
        # raise NotImplementedError.
        # We can't call it here anyway because it hasn't
        # been added to the python bindings yet
        z[0] = x.reduce_sum(self.reduce_mask)
    """
    def perform(self, node, inp, out):
        raise MethodNotDefined("")

    def supports_c_code(self, inputs):
        """ Returns True if the current op and reduce pattern
            has functioning C code """

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

        sub = {'fail': 'fake failure code'}

        try:
            self.c_code(node, name, inp, out, sub)
            self.c_support_code_apply(node, name)
        except NotImplementedError:
            return False
        return True

    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>']

    def c_compiler(self):
        return NVCC_compiler

    def c_init_code(self):
        return ['setup_ext_cuda();']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        nd_in = node.inputs[0].type.ndim
        nd_out = node.outputs[0].type.ndim
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        assert nd_in - nd_out == sum(self.reduce_mask)

        sio = StringIO()
        fail = sub['fail']

        #check input
        print >> sio, """
        if (PyGpuArray_NDIM(%(x)s) != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError,
                         "required nd=%(nd_in)s, got nd=%%i", PyGpuArray_NDIM(%(x)s));
            %(fail)s;
        }
        """ % locals()

        # It might be nice to use a property of the op class to do this,
        # but tensor.elemwise.CAReduce has this exact same check so I guess
        # this is OK to do
        if self.scalar_op in [scal.minimum, scal.maximum]:
            conds = ["(PyGpuArray_DIMS(%s)[%d] == 0)" % (x, i)
                     for i in xrange(nd_in)
                     if self.reduce_mask[i]]
            assert len(conds) > 0
            cond = "(" + " || ".join(conds) + ")"
            print >> sio, """
            if %(cond)s
            {
                PyErr_Format(PyExc_ValueError," tried to reduce a 0-length axis.");
                %(fail)s;
            }
            """ %locals()

        #
        # alloc an output if we need one
        #

        # check the basics of out output
        print >> sio, """
        if (  !%(z)s
           || (PyGpuArray_NDIM(%(z)s) != %(nd_out)s)
        """ % locals()

        #ensure that the output has the right non-reduced dimensions
        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print >> sio, " || (PyGpuArray_DIMS(%(z)s)[%(j)s] != PyGpuArray_DIMS(%(x)s)[%(i)d]) " % locals()
                j += 1

        print >> sio, """
           )
        {
            """ % locals()
        if nd_out > 0:
            print >> sio, "size_t new_dims[%(nd_out)s]; " % locals()
        else:
            print >> sio, "size_t *new_dims=NULL; "

        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print >> sio, 'new_dims[%(j)s] = PyGpuArray_DIMS(%(x)s)[%(i)s];' % locals()
                j += 1
        out_typecode = dtype_to_typecode(out_dtype[4:])
        print >> sio, """
            Py_XDECREF(%(z)s);
            %(z)s = pygpu_empty(%(nd_out)s, new_dims,
                                %(out_typecode)s, GA_C_ORDER,
                                pygpu_default_context(),
                                Py_None);
            if (NULL == %(z)s)
            {
                PyErr_Format(PyExc_RuntimeError, "Failed to allocate output");
                %(fail)s;
            }
        }
        """ % locals()

        # \begin bracket the reduction in a check that there is
        # actually work to do
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "cudaMemset((%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset), 0, PyGpuArray_SIZE(%(z)s) * sizeof(%(out_dtype)s))" % locals()
        #TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            scalar_op = self.scalar_op
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduceCuda not implemented when input shape is 0"
                         " for this scalar_op: %(scalar_op)s");
            %(fail)s;
            """ % locals()
        print >> sio, """
        if (PyGpuArray_SIZE(%(z)s) && ! PyGpuArray_SIZE(%(x)s)){
            %(zero_shp)s;
        }
        else if (PyGpuArray_SIZE(%(z)s))
        {
        """ % locals()

        #
        # Now perform the reduction
        #

        if all(i == 1 for i in self.reduce_mask):
            #check if the tensor is ccontiguous, if true, use the c_code_reduce_ccontig code.
            #TODO: check if we are ccontiguous when we un-dimshuffle
            #TODO: if only some dims are ccontiguous, call version with less dims.
            print >> sio, 'if(%(x)s->ga.flags & GA_C_CONTIGUOUS){'%locals()
            self.c_code_reduce_ccontig(sio, node, name, x, z, fail)
            print >> sio, "}else{"
            getattr(self, 'c_code_reduce_%s'%(''.join(
                str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)
            print >> sio, "}"
        else:
            getattr(self, 'c_code_reduce_%s'%(''.join(
                str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)

        # \end bracket the reduction ...
        print >> sio, """
        }
        """ % locals()

        return sio.getvalue()

    def _makecall(self, node, name, x, z, fail, pattern=None):
        """Return a string for making a kernel call.

            The return value looks something like:

            .. code-block:: c

                if (verbose)
                    printf("running kernel_reduce_10_%(name)s\\n");
                int n_shared = sizeof(%(acc_dtype)s) * n_threads.x * n_threads.y * n_threads.z;
                kernel_reduce_10_%(name)s<<<n_blocks, n_threads,
                                                n_shared>>>(
                        PyGpuArray_DIMS(%(x)s)[0],
                        PyGpuArray_DIMS(%(x)s)[1],
                        (%(in_dtype)s *)(((char *)cuda_get_ptr(%(x)s->ga.data))+%(x)s->ga.offset),
                        PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
                        PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s),
                        (%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset),
                        PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s)
                        );
                [
        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(z)s->ga);\n" % dict(z=z)
                ]
                if (cudaSuccess != cudaGetLastError())
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: ... );
                    %(fail)s;
                }
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
        shapes_data = ",".join(["(unsigned long long) PyGpuArray_DIMS(%s)[%d]" % (x, i)
                                for i in range(node.inputs[0].ndim)])

        print >> sio, """
            if (verbose)
                printf("running kernel_reduce_%(pattern)s_%(name)s\\n");
            int n_shared = sizeof(%(acc_dtype)s) * n_threads.x * n_threads.y * n_threads.z;
            if (verbose>1)
                printf("n_threads.x=%%d, n_threads.y=%%d, n_threads.z=%%d,"
                       " nb_threads=%%d, n_blocks.x=%%d, n_blocks.y=%%d,"
                       " nb_block=%%d, n_shared=%%d, %(shapes_format)s\\n",
                                  n_threads.x,n_threads.y,n_threads.z,
                                  n_threads.x*n_threads.y*n_threads.z,
                                  n_blocks.x,n_blocks.y,
                                  n_blocks.x*n_blocks.y, n_shared, %(shapes_data)s);
            kernel_reduce_%(pattern)s_%(name)s<<<n_blocks, n_threads, n_shared>>>(
            """ % locals()
        for i in xrange(ndim):
            print >> sio, """
                    PyGpuArray_DIMS(%(x)s)[%(i)s],
            """ % locals()
        print >> sio, """
                    (%(in_dtype)s *)(((char *)cuda_get_ptr(%(x)s->ga.data))+%(x)s->ga.offset)
            """ % locals()
        for i in xrange(ndim):
            print >> sio, """
                    ,PyGpuArray_STRIDES(%(x)s)[%(i)s]/sizeof(%(in_dtype)s)
            """ % locals()
        print >> sio, """
                    ,(%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset)
            """ % locals()
        for i in xrange(nd_out):
            print >> sio, """
                    ,PyGpuArray_STRIDES(%(z)s)[%(i)s]/sizeof(%(out_dtype)s)
            """ % locals()
        sync = ""
        if config.gpuarray.sync:
            sync = """GpuArray_sync(&%(z)s->ga);""" % locals()
        print >> sio, """
                    );
            %(sync)s
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s."
                    " (grid: %%i x %%i; block: %%i x %%i x %%i)"
                    " %(shapes_format)s \\n",
                    "kernel_reduce_%(pattern)s_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    %(shapes_data)s);
                %(fail)s;
            }
        """ % locals()
        return sio.getvalue()

    def _k_decl(self, node, nodename, pattern=None,
                ndim=None, reduce_mask=None):
        """Return a string to declare a kernel function

        The result will look something like this:

        .. code-block:: c

            static __global__ void kernel_reduce_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const %(in_dtype)s *A,
                    const int sA0,
                    const int sA1,
                    const int sA2,
                    %(out_dtype)s * Z,
                    const int sZ0)

            Since the nodename is unique, we don't need to put the name
            of the scalar_op in here.

        """
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        if reduce_mask is None:
            reduce_mask = self.reduce_mask
        if ndim is None:
            ndim = len(reduce_mask)
        if pattern is None:
            pattern = ''.join(str(i) for i in reduce_mask)
        sio = StringIO()

        print >> sio, """
            static __global__ void kernel_reduce_%(pattern)s_%(nodename)s(
        """ % locals()
        for i in xrange(ndim):
            print >> sio, """
                    const int d%(i)s,
        """ % locals()
        print >> sio, """
                    const %(in_dtype)s *A,
        """ % locals()
        for i in xrange(ndim):
            print >> sio, """
                    const int sA%(i)s,
        """ % locals()
        print >> sio, """
                    %(out_dtype)s * Z
        """ % locals()
        for i in xrange(ndim - sum(reduce_mask)):
            print >> sio, """
                    , const int sZ%(i)s
        """ % locals()
        print >> sio, ")"
        return sio.getvalue()

    def _k_init(self, node, nodename):
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)

        return """
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = 0;

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
            if self.pre_scalar_op: # TODO, multi_dtype!
                #dtype = node.inputs[0].dtype
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
            node: the node argument to this op's c_code
            name: the name argument to this op's c_code
            left: a C code string identifying an lvalue
            right: a C code string identifying an expression
            sub: the sub argument to this op's c_code
            pre: If True, we will add the pre_scalar_op.c_code

            returns C code to reduce left and right, assigning the
            result to left."""

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

        node, name, sub: these should be passed through from the original
        call to c_code
        """
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)

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
            %(z_pos)s = buf[0];
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
                    %(z_pos)s = buf[0];
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
                                    'buf[threadNum]','buf[threadNum+%d]' % num,
                                    sub, False)
            current_version += this_if
            current_version += """
            """
        current_version += """
                if (threadNum == 0)
                {
                    %(z_pos)s = buf[0];
                }
            }
        }
        """

        current_version = current_version % locals()

        return current_version

    #Threads must be organized as: threadNum%nb_reduce correspond to the same sum
    #nb_reduce<=warpSize
    def _k_reduce_buf_multiple(self, z_pos, node, name, nb_reduce):
        reduce_fct = self._assign_reduce(node, name, 'myresult', 'buf[i]', {}, False)
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
            %(z_pos)s = myresult;
        }
        """ % locals()

    def c_code_reduce_ccontig(self, sio, node, name, x, z, fail):
        """
        WRITEME
        IG: I believe, based on how this is called in c_code, that it
        is for the case where we are reducing on all axes and x is
        C contiguous.
        """
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "cudaMemset((%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset), 0, PyGpuArray_SIZE(%(z)s) * sizeof(%(out_dtype)s))" % locals()
        #TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduceCuda not implemented when input shape is 0 for this scalar_op");
            %(fail)s;
            """ % locals()

        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        sync = ""
        if config.gpuarray.sync:
            sync = """GpuArray_sync(&%(z)s->ga);""" % locals()
        print >> sio, """
        {
          if(PyGpuArray_SIZE(%(x)s)==0){
            %(zero_shp)s;
          }else{
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_SIZE(%(x)s),
                            (size_t) 256));
            dim3 n_blocks(1);
            if (verbose) printf("running kernel_reduce_ccontig_%(name)s"
                                " n_threads.x=%%d, size=%%d, ndim=%%d\\n",
                                n_threads.x,PyGpuArray_SIZE(%(x)s),
                                PyGpuArray_NDIM(%(x)s));
            int n_shared = sizeof(%(acc_dtype)s) * n_threads.x;
            kernel_reduce_ccontig_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    PyGpuArray_SIZE(%(x)s),
                    (%(in_dtype)s *)(((char *)cuda_get_ptr(%(x)s->ga.data))+%(x)s->ga.offset),
                    (%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset));
            %(sync)s
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s."
                             " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_ccontig_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
         }
        }
        """ % locals()

    def c_code_reduce_1(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[0],
                            (size_t) 256));
            dim3 n_blocks(1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_11(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[1],
                            (size_t) 256));
            while (n_threads.y * n_threads.x <= 256) ++n_threads.y;
            n_threads.y -= 1;
            if (n_threads.y > PyGpuArray_DIMS(%(x)s)[0])
                n_threads.y = PyGpuArray_DIMS(%(x)s)[0];

            dim3 n_blocks(1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_01X(self, sio, node, name, x, z, fail, N):
        """
        :param N: the number of 1 in the pattern N=1 -> 01, N=2 -> 011 N=3 ->0111
                  Work for N=1,2,3
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
            while (n_threads.x * (n_threads.y+1) <= 256)
            {
                if (n_threads.y < PyGpuArray_DIMS(%(x)s)[%(N)s-1])
                    n_threads.y += 1;
                else
                    break;
            }""" % locals()

        threads_z = """
            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * (n_threads.z+1) <= 256)
            {
                if (n_threads.z < PyGpuArray_DIMS(%(x)s)[%(N)s-2])
                    n_threads.z += 1;
                else
                    break;
            }
            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);
        """ % locals()

        if len(self.reduce_mask) == 2:
            threads_y = ''
            threads_z = ''

        if len(self.reduce_mask) == 3:
            threads_z = ''

        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[%(N)s],
                            (size_t) 256));
            %(threads_y)s
            %(threads_z)s
            dim3 n_blocks(std::min(PyGpuArray_DIMS(%(x)s)[0],
                                   (size_t) 4096));
            %(makecall)s
        }
        """ % locals()

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
        sync = ""
        if config.gpuarray.sync:
            sync = """GpuArray_sync(&%(z)s->ga);""" % locals()
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[0],
                            (size_t) 256));
            dim3 n_blocks(1,
                std::min(PyGpuArray_DIMS(%(x)s)[1],
                    (size_t) 4096));
            if (verbose) {
              fprintf(stderr,
                "running kernel_reduce_10_%(name)s n_blocks=(%%i,%%i)\\n",
                n_blocks.x,
                n_blocks.y);
            }
            assert( PyGpuArray_DIMS(%(x)s)[1] == PyGpuArray_DIMS(%(z)s)[0]);
            int n_shared = sizeof(%(acc_dtype)s) * n_threads.x;
            kernel_reduce_010_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    1,
                    PyGpuArray_DIMS(%(x)s)[0],
                    PyGpuArray_DIMS(%(x)s)[1],
                    (%(in_dtype)s *)(((char *)cuda_get_ptr(%(x)s->ga.data))+%(x)s->ga.offset),
                    1,
                    PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
                    PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s),
                    (%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset),
                    1,
                    PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s)
                    );
            %(sync)s
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s."
                    " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_010_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_code_reduce_010(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        makecall_inner = self._makecall(node, name, x, z, fail,
                                        pattern="010_inner")
        pattern = ''.join(str(i) for i in self.reduce_mask)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        sync = ""
        if config.gpuarray.sync:
            sync = """GpuArray_sync(&%(z)s->ga);""" % locals()
        print >> sio, """
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
                dim3 n_threads(32,1,1);

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                int A = PyGpuArray_DIMS(%(x)s)[0];
                int B = PyGpuArray_DIMS(%(x)s)[1];
                int C = PyGpuArray_DIMS(%(x)s)[2];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                dim3 n_blocks(A,D);
                if (n_blocks.x > 4096) n_blocks.x = 4096;
                if (n_blocks.x*n_blocks.y > 4096) n_blocks.y = 4096/n_blocks.x;
                int n_shared = 0;
                kernel_reduce_010_AD_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                A,B,C,D,
                        (%(in_dtype)s *)(((char *)cuda_get_ptr(%(x)s->ga.data))+%(x)s->ga.offset),
                        PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
                        PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s),
                        PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s),
                        (%(out_dtype)s *)(((char *)cuda_get_ptr(%(z)s->ga.data))+%(z)s->ga.offset),
                        PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s),
                        PyGpuArray_STRIDES(%(z)s)[1]/sizeof(%(out_dtype)s)
                        );
                %(sync)s
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                        "Cuda error: %%s: %%s."
                        " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_010_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
            else
            {
                int verbose = 2;

                  dim3 n_threads(std::min((size_t) 32,
                                          PyGpuArray_DIMS(%(x)s)[2]));
                  while(    (n_threads.x*(n_threads.y+1)<=256)
                         && (n_threads.y<PyGpuArray_DIMS(%(x)s)[1])){
                      n_threads.y++;
                  }

                  dim3 n_blocks(std::min(PyGpuArray_DIMS(%(x)s)[0],
                                (size_t)4096));
                  n_blocks.y = std::min(
                      ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
                                  (size_t)n_threads.x),
                      (size_t)(4096 / n_blocks.x)
                      );
                if(std::min(std::min(PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
                                     PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s)),
                            PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s))
                   ==PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s)
                  && n_blocks.y==ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
                                             (size_t)n_threads.x)){
                  if(verbose>1)
                    printf("n_block.x.1=%%d, n_block.x.2=%%d, n_block.y.1=%%d, n_block.y.2=%%d,\\n",
                           PyGpuArray_DIMS(%(x)s)[0],4096,
                           ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],(size_t)n_threads.x),
                                       (size_t)(4096 / n_blocks.x));
                  assert(n_threads.x<=32);
                  %(makecall_inner)s
                }else{
                  n_threads.x = std::min(PyGpuArray_DIMS(%(x)s)[1],
                                         (size_t) 256);
                  n_blocks.x = std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)4096);
                  n_blocks.y = std::min(
                      PyGpuArray_DIMS(%(x)s)[2],
                      (size_t)(4096 / n_blocks.x)
                      );
                  %(makecall)s
                }
                %(sync)s
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_%(pattern)s_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
        }
        """ % locals()

    def c_code_reduce_0101(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[3],
                             (size_t) 256));
            while (n_threads.x * n_threads.y <= 256)
            {
                if (n_threads.y > PyGpuArray_DIMS(%(x)s)[1]) break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;
            dim3 n_blocks(PyGpuArray_DIMS(%(x)s)[0], PyGpuArray_DIMS(%(x)s)[2]);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_100(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        # use threadIdx.x for i0
        # use blockIdx.x for i1
        # use blockIdx.y for i2
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[0],
                             (size_t) 256));
            dim3 n_blocks(std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)4096));
            while (n_blocks.x * (n_blocks.y+1) <= 4096 && n_blocks.y <= PyGpuArray_DIMS(%(x)s)[2])
            {
                n_blocks.y += 1;
            }
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_110(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[1],
                             (size_t) 256));
            while (n_threads.x*n_threads.y <= 256)
            {
                if (n_threads.y > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            dim3 n_blocks(PyGpuArray_DIMS(%(x)s)[2]);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_001(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[2],
                             (size_t) 256));
            dim3 n_blocks(
                    std::min(PyGpuArray_DIMS(%(x)s)[0],
                             (size_t) 4096));
            while (n_blocks.x * n_blocks.y <= 4096)
            {
                if (n_blocks.y > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_blocks.y += 1;
            }
            n_blocks.y -= 1;
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[2],
                             (size_t) 256));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= 256)
            {
                if (n_threads.y > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= 256)
            {
                if (n_threads.z > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;
            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_0011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)
        print >> sio, """
        {
            int verbose = 0;

            dim3 n_blocks(
                    std::min(PyGpuArray_DIMS(%(x)s)[0],
                             (size_t) 4096));

            while (n_blocks.x * n_blocks.y <= 4096 &&
                   n_blocks.y < PyGpuArray_DIMS(%(x)s)[1])
            {
                n_blocks.y += 1;
            }

            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[3],
                             (size_t) 256));
            while (n_threads.x * n_threads.y <= 256
                   && n_threads.y < PyGpuArray_DIMS(%(x)s)[2]
                   && n_threads.x * n_threads.y * sizeof(%(acc_dtype)s) <=(15*1024-200))
            {
                n_threads.y += 1;
            }

            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_1111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[2],
                             (size_t) 256));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= 256)
            {
                if (n_threads.y > PyGpuArray_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= 256)
            {
                if (n_threads.z > PyGpuArray_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;

            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_1011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(PyGpuArray_DIMS(%(x)s)[3],
                             (size_t) 256));

            while (n_threads.x * (n_threads.y+1) <= 256) ++n_threads.y;
            if (n_threads.y > PyGpuArray_DIMS(%(x)s)[2])
                n_threads.y = PyGpuArray_DIMS(%(x)s)[2];

            while (n_threads.x * n_threads.y * (n_threads.z+1) <= 256) ++n_threads.z;
            if (n_threads.z > 64)
                n_threads.z = 64;
            if (n_threads.z > PyGpuArray_DIMS(%(x)s)[0])
                n_threads.z = PyGpuArray_DIMS(%(x)s)[0];

            dim3 n_blocks(PyGpuArray_DIMS(%(x)s)[1]);
            %(makecall)s
        }
        """ % locals()

    def c_code_cache_version_apply(self, node):
        version = [11]  # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(self.scalar_op,
                [Scalar(dtype=input.type.dtype)() for input in node.inputs],
                [Scalar(dtype=output.type.dtype)() for output in node.outputs])
        version.extend(self.scalar_op.c_code_cache_version())
        for i in node.inputs + node.outputs:
            version.extend(Scalar(dtype=i.type.dtype).c_code_cache_version())
        if all(version):
            return tuple(version)
        else:
            return ()

    def c_support_code_apply(self, node, nodename):
        sio = StringIO()
        nd_in = len(self.reduce_mask)
        in_dtype = "npy_" + node.inputs[0].dtype
        out_dtype = "npy_" + node.outputs[0].dtype
        acc_dtype = "npy_" + self._acc_dtype(node.inputs[0].dtype)

        if all(i == 1 for i in self.reduce_mask):
            #this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print >> sio, """
            static __global__ void kernel_reduce_ccontig_%(nodename)s(
                    const unsigned int d0,
                    const %(in_dtype)s *A,
                    %(out_dtype)s * Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = %(reduce_init)s;

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
            """ % locals()
        if self.reduce_mask == (1,):
            #this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print >> sio, """
            static __global__ void kernel_reduce_1_%(nodename)s(
                    const unsigned int d0,
                    const %(in_dtype)s *A, const int sA0,
                    %(out_dtype)s * Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = %(reduce_init)s;

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
            """ % locals()
        if self.reduce_mask == (1, 1):
            #this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")

            print >> sio, """
            static __global__ void kernel_reduce_11_%(nodename)s(
                    const int d0,
                    const int d1,
                    const %(in_dtype)s *A, const int sA0, const int sA1,
                    %(out_dtype)s * Z)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y*blockDim.x + threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = %(reduce_init)s;

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
            """ % locals()
        #01, 011, 0111
        if (0 == self.reduce_mask[0] and
            all(self.reduce_mask[1:]) and
            nd_in in[2, 3, 4]):
            # this kernel uses one block for each row.
            # threads per block for each element per row.

            N_pattern = ''.join(['1'] * (nd_in - 1))
            # TODO: is it faster to hardcode sA3, etc. in the later code, rather
            # than have the for_* variables declare them and the later code use
            # their names?
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
            param_dim = ",".join(["const int d%d" % i
                                  for i in xrange(nd_in)])
            param_strides = ",".join(["const int sA%d" % i
                                      for i in xrange(nd_in)])
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_init = self._assign_init("A[%(first_i3)s * %(sA3)s + %(first_i2)s * %(sA2)s + %(first_i1)s * %(sA1)s + i0 * sA0]" % locals())
            reduce_fct = self._assign_reduce(
                node, nodename, "myresult",
                "A[i3 * sA3 + i2 * sA2 + i1 * sA1 + i0 * sA0]",
                {}, True)
            print >> sio, """
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
                """ % locals()
        if self.reduce_mask == (0, 1, 0) or self.reduce_mask == (1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            #TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2*sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + threadIdx.x * sA1 + i2 * sA2]")
            print >> sio, """
            static __global__ void kernel_reduce_010_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const %(in_dtype)s *A, const int sA0,
                    const int sA1, const int sA2,
                    %(out_dtype)s * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }


                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        %(acc_dtype)s myresult = %(reduce_init)s;
                        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }

            }
            """ % locals()
        if self.reduce_mask == (0, 1, 0):
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "X[a * sX0 + b * sX1 + c * sX2]",
                                             {}, True)
            reduce_init = self._assign_init("X[a * sX0 + 0 * sX1 + c * sX2]")
            print >> sio, """
            static __global__ void kernel_reduce_010_AD_%(nodename)s(
                    const int A,
                    const int B,
                    const int C,
                    const int D,
                    //const int E, // THIS is 32
                    const %(in_dtype)s *X, const int sX0,
                    const int sX1, const int sX2,
                    %(out_dtype)s * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                %(acc_dtype)s myresult = 0;

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
                            Z[a * sZ0 + c * sZ1] = myresult;
                        }
                    }
                }

            }
            """ % locals()
        if self.reduce_mask == (0, 1, 0):
            #
            # This kernel is optimized when the inner most dimensions
            # have the smallest stride.

            # this kernel uses one block for multiple column(up to 32TODO),
            # threads per block for each element per column.

#thread.x = dim 2 contiguous
#thread.y = dim 1
#block.x = dim 0
#block.y = dim 1 rest
            init = self._k_init(node, nodename)
            decl = self._k_decl(node, nodename, pattern="010_inner")
            reducebuf = self._k_reduce_buf_multiple('Z[i0 * sZ0 + i2*sZ1]',
                                                    node, nodename,
                                                    'blockDim.x')
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + 0 * sA1 + i2 * sA2]")
            print >> sio, """
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
            """ % locals()
        if self.reduce_mask == (1, 1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            #TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[blockIdx.x * sZ0]', node, nodename, sub = {})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + blockIdx.x * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[blockIdx.x * sA2]")
            print >> sio, """
            static __global__ void kernel_reduce_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const %(in_dtype)s *A, const int sA0,
                    const int sA1, const int sA2,
                    %(out_dtype)s * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    //TODO: set error code
                    Z[blockIdx.x * sZ0] = -666;
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
            """ % locals()
        if self.reduce_mask == (1, 0, 0):
            reducebuf = self._k_reduce_buf('Z[i1 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i1 * sA1 + i2 * sA2]")
            print >> sio, """
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
            """ % locals()
        if self.reduce_mask == (1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node,
                                           nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print >> sio, """
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
            """ % locals()
        if self.reduce_mask == (0, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i1 * sA1]")
            print >> sio, """
            static __global__ void kernel_reduce_001_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const %(in_dtype)s *A, const int sA0,
                    const int sA1, const int sA2,
                    %(out_dtype)s * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        %(acc_dtype)s myresult = %(reduce_init)s;
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals()
        if self.reduce_mask == (0, 0, 1, 1):
             # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i1 * sA1]")
            print >> sio, """
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        %(acc_dtype)s myresult = %(reduce_init)s;
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
            """ % locals()
        if self.reduce_mask == (0, 1, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i2 * sA2]")
            print >> sio, """
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        %(acc_dtype)s myresult = %(reduce_init)s;
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
            """ % locals()
        if self.reduce_mask == (1, 1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename,
                                           sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print >> sio, """
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
            """ % locals()
        if self.reduce_mask == (1, 0, 1, 1):
            reducebuf = self._k_reduce_buf('Z[blockIdx.x*sZ0]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + blockIdx.x * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[blockIdx.x * sA1]")
            print >> sio, """
            static __global__ void kernel_reduce_1011_%(nodename)s(
                    const unsigned int d0,
                    const unsigned int d1,
                    const unsigned int d2,
                    const unsigned int d3,
                    const %(in_dtype)s *A, const int sA0, const int sA1,
                    const int sA2, const int sA3,
                    %(out_dtype)s * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ %(acc_dtype)s buf[];
                %(acc_dtype)s myresult = %(reduce_init)s;

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
            """ % locals()
        print >> sio, """
        template <typename T>
        static T ceil_intdiv(T a, T b)
        {
            return (a/b) + ((a % b) ? 1: 0);
        }
        """
        return sio.getvalue()


class GpuCAReduceCPY(GpuKernelBase, HideC, CAReduceDtype):
    """CAReduce that reuse the python code from gpuarray.

    Too slow for now as it only have a python interface.

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
        return super(GpuCAReduceCPY, self).make_thunk(node, storage_map,
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
                       objvar='k_reduk_'+name)]

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

        if (%(sync)d)
            GpuArray_sync(&%(output)s->ga);
""" % dict(k_var='k_reduk_'+name, sync=bool(config.gpuarray.sync),
           ls=ls, fail=sub['fail'], output=output, input=input,
           cast_out=bool(acc_dtype != node.outputs[0].type.dtype))

        return code

    def c_code_cache_version(self):
        return (0, self.GpuKernelBase_version)

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
# To allow reloading old pickled files
GpuCAReduce = GpuCAReduceCPY
