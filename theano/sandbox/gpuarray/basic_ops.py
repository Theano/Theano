from __future__ import absolute_import, print_function, division
import os

import numpy

from theano import Op, Apply, Type, Variable
from theano import tensor, config
from theano.gradient import grad_undefined
from theano.tensor.basic import Alloc, Join, Split

from theano.gof import HideC
from theano.gof.utils import MethodNotDefined

from collections import deque

from six import string_types, iterbytes
from six.moves import xrange

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .type import (GpuArrayType, GpuArrayConstant, gpu_context_type,
                   get_context, ContextNotDefined)
from .fp16_help import write_w


def as_gpuarray_variable(x, context_name):
    """
    This will attempt to convert `x` into a variable on the GPU.

    It can take either a value of another variable.  If `x` is already
    suitable, it will be returned as-is.

    Parameters
    ----------
    x
        Object to convert
    context_name : str or None
        target context name for the result

    """
    # If this is already some form of variable, try to avoid an extra transfer
    if isinstance(x, Variable):
        while True:
            # If we are already a GpuArrayVariable in the right context
            # then there is nothing to do.
            if (isinstance(x.type, GpuArrayType) and
                    x.type.context_name == context_name):
                return x

            # If x is the result of a transfer, try to dig through.
            if getattr(x, 'owner', None):
                if isinstance(x.owner.op, HostFromGpu):
                    x = x.owner.inputs[0]
                    continue
                if isinstance(x.owner.op, GpuFromHost):
                    x = x.owner.inputs[0]
                    continue
                if isinstance(x.owner.op, GpuToGpu):
                    x = x.owner.inputs[0]
                    continue

            # If none of the conditions where met, then continue with
            # the rest of the body
            break

        # If we couldn't deal with transfers, then maybe it's a tensor
        if isinstance(x.type, tensor.TensorType):
            return GpuFromHost(context_name)(x)

    # Try _as_GpuArrayVariable if possible
    if hasattr(x, '_as_GpuArrayVariable'):
        return x._as_GpuArrayVariable(context_name)

    # If it didn't work try for a constant
    ctx = get_context(context_name)

    if isinstance(x, gpuarray.GpuArray):
        if x.context.ptr != ctx.ptr:
            x = x.transfer(ctx)

    x = gpuarray.asarray(x, context=ctx)

    bcast = [(s == 1) for s in x.shape]
    return GpuArrayConstant(GpuArrayType(dtype=x.dtype,
                                         broadcastable=bcast,
                                         context_name=context_name),
                            x)


def infer_context_name(*vars):
    """
    Infer the context name to use from the inputs given

    """
    # We try to infer the closest context first
    # TODO: What to do in case of context conflicts?
    #       We currently use a first found wins approach.
    todo = deque()
    todo.extendleft(vars)
    while todo:
        v = todo.pop()
        if isinstance(v.type, GpuArrayType):
            return v.type.context_name
        if hasattr(v.tag, 'context_name'):
            return v.tag.context_name
        if v.owner:
            if isinstance(v.owner.op, HostFromGpu):
                return v.owner.inputs[0].type.context_name
            if len(v.owner.inputs) == 1:
                todo.extendleft(v.owner.inputs)
    # If we can't find a context try None if it exists
    try:
        get_context(None)
        return None
    except ContextNotDefined:
        raise ValueError("Could not infer context from inputs")


class Kernel(object):
    """
    This class groups together all the attributes of a gpu kernel.

    """

    def __init__(self, code, params, name, flags,
                 codevar=None, binvar=None, objvar=None):
        self.code = code
        self.params = params
        self.name = name
        self.flags = flags
        if codevar is None:
            codevar = 'kcode_' + name
        self.codevar = codevar
        if binvar is None:
            binvar = 'kbin_' + name
        self.binvar = binvar
        if objvar is None:
            objvar = 'k_' + name
        self.objvar = objvar

    @staticmethod
    def get_flags(*types):
        def get_dtype(t):
            if isinstance(t, string_types):
                return numpy.dtype(t)
            elif isinstance(t, Type):
                return t.dtype
            elif isinstance(t, Variable):
                return t.type.dtype
            else:
                raise TypeError("can't get a dtype from %s" % (type(t),))
        dtypes = [get_dtype(t) for t in types]
        flags = dict(cluda=True)
        if any(d == numpy.float64 for d in dtypes):
            flags['have_double'] = True
        if any(d.itemsize < 4 for d in dtypes):
            flags['have_small'] = True
        if any(d.kind == 'c' for d in dtypes):
            flags['have_complex'] = True
        if any(d == numpy.float16 for d in dtypes):
            flags['have_half'] = True
        return flags

    def _get_c_flags(self):
        res = []
        if self.flags.get('cluda', False):
            res.append('GA_USE_CLUDA')
        if self.flags.get('have_double', False):
            res.append('GA_USE_DOUBLE')
        if self.flags.get('have_small', False):
            res.append('GA_USE_SMALL')
        if self.flags.get('have_complex', False):
            res.append('GA_USE_COMPLEX')
        if self.flags.get('have_half', False):
            res.append('GA_USE_SMALL')
        return '|'.join(res)

    def _get_c_types(self):
        def m(t):
            if t == gpuarray.GpuArray:
                return "GA_BUFFER"
            else:
                return str(gpuarray.dtype_to_typecode(t))
        return ', '.join(m(t) for t in self.params)


class GpuKernelBase(object):
    """
    Base class for operations that need to compile kernels.

    It is not mandatory to use this class, but it helps with a lot of
    the small things that you have to pay attention to.

    """
    params_type = gpu_context_type

    def gpu_kernels(self, node, name):
        """
        This is the method to override. This should return an iterable
        of Kernel objects that describe the kernels this op will need.

        """
        raise MethodNotDefined('gpu_kernels')

    def c_headers(self):
        try:
            o = super(GpuKernelBase, self).c_headers()
        except MethodNotDefined:
            o = []
        return o + ['gpuarray/types.h']

    def _generate_kernel_bin(self, k, ctx):
        gk = gpuarray.GpuKernel(k.code, k.name, k.params, context=ctx,
                                **k.flags)
        bin = gk._binary
        bcode = ','.join(hex(c) for c in iterbytes(bin))
        return ("""static const char %(bname)s[] = { %(bcode)s };""" %
                dict(bname=k.binvar, bcode=bcode))

    def _generate_kernel_code(self, k):
        code = '\\n'.join(l for l in k.code.split('\n'))
        code = code.replace('"', '\\"')
        return ("""static const char *%(cname)s = "%(code)s";""" %
                dict(cname=k.codevar, code=code))

    def _generate_kernel_vars(self, k):
        return """GpuKernel %(kname)s;""" % dict(kname=k.objvar)

    def c_support_code(self):
        return """
        template <typename T>
        static T ceil_intdiv(T a, T b)
        {
            return (a/b) + ((a % b) ? 1: 0);
        }
        """

    def c_support_code_apply(self, node, name):
        kernels = self.gpu_kernels(node, name)
        ctx = self.get_params(node)
        bins = '\n'.join(self._generate_kernel_bin(k, ctx) for k in kernels)
        codes = '\n'.join(self._generate_kernel_code(k) for k in kernels)
        return '\n'.join([bins, codes])

    def c_support_code_struct(self, node, name):
        kernels = self.gpu_kernels(node, name)
        return '\n'.join(self._generate_kernel_vars(k) for k in kernels)

    def _generate_zeros(self, k):
        return """memset(&%(v)s, 0, sizeof(%(v)s));""" % dict(v=k.objvar)

    def _generate_kernel_init(self, k, fail, ctx):
        return """{
  int err;
  int types[%(numargs)u] = {%(types)s};
  const char *bcode = %(bvar)s;
  size_t sz = sizeof(%(bvar)s);
  if (GpuKernel_init(&%(ovar)s, %(ctx)s->ops, %(ctx)s->ctx, 1, &bcode, &sz,
                     "%(kname)s", %(numargs)u, types, GA_USE_BINARY, NULL)
      != GA_NO_ERROR) {
    if ((err = GpuKernel_init(&%(ovar)s, %(ctx)s->ops, %(ctx)s->ctx, 1,
                              &%(cname)s, NULL, "%(kname)s", %(numargs)u,
                              types, %(flags)s, NULL)) != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuKernel_init error %%d: %%s",
                   err, Gpu_error(%(ctx)s->ops, %(ctx)s->ctx, err));
      %(fail)s
    }
  }
}""" % dict(numargs=len(k.params), types=k._get_c_types(), bvar=k.binvar,
            ovar=k.objvar, kname=k.name, cname=k.codevar,
            flags=k._get_c_flags(), fail=fail, ctx=ctx)

    def c_init_code_struct(self, node, name, sub):
        ctx = sub['params']
        kernels = self.gpu_kernels(node, name)
        inits_0 = '\n'.join(self._generate_zeros(k) for k in kernels)
        inits = '\n'.join(self._generate_kernel_init(k, sub['fail'], ctx)
                          for k in kernels)
        return '\n'.join([inits_0, inits])

    def _generate_kernel_cleanup(self, k):
        return "GpuKernel_clear(&%(ovar)s);" % dict(ovar=k.objvar)

    def c_cleanup_code_struct(self, node, name):
        kernels = self.gpu_kernels(node, name)
        cleanups = '\n'.join(self._generate_kernel_cleanup(k) for k in kernels)
        return cleanups

    # This is a shorthand for if your op only has a fixed version
    # You can reimplement it, but make sure to call kernel_version()
    def c_code_cache_version_apply(self, node):
        return (self.c_code_cache_version(), self.kernel_version(node))

    def kernel_version(self, node):
        """
        If you override :meth:`c_code_cache_version_apply`, call this
        method to have the version of the kernel support code and
        device.

        Parameters
        ----------
        node : apply node
            The node that we need the cache version for.

        """
        return (3, self.get_params(node).bin_id)


class HostFromGpu(Op):
    """
    Transfer data to CPU.

    """
    __props__ = ()
    _f16_ok = True

    def __str__(self):
        return 'HostFromGpu(gpuarray)'

    def make_node(self, x):
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x],
                     [tensor.TensorType(dtype=x.dtype,
                                        broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = numpy.asarray(x)

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        GpuArray %(name)s_ga_s;
        GpuArray *%(name)s_ga = NULL;
        int %(name)serr;
        PyArray_Descr *%(name)s_dtype;
        if (!GpuArray_ISONESEGMENT(&%(inp)s->ga)) {
            if (GpuArray_copy(&%(name)s_ga_s, &%(inp)s->ga, GA_C_ORDER) != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't make contiguous copy");
                %(fail)s;
            }
            %(name)s_ga = &%(name)s_ga_s;
        } else {
            %(name)s_ga = &%(inp)s->ga;
        }
        %(name)s_dtype = typecode_to_dtype(%(name)s_ga->typecode);
        Py_XDECREF(%(out)s);
        // PyArray_Empty below steals a reference to the dtype we pass it
        // so we need an extra one to spare.
        Py_INCREF(%(name)s_dtype);
        %(out)s = (PyArrayObject *)PyArray_Empty(%(inp)s->ga.nd,
                                (npy_intp *)%(inp)s->ga.dimensions,
                                %(name)s_dtype,
                                (%(inp)s->ga.flags & GA_F_CONTIGUOUS) &&
                                !(%(inp)s->ga.flags & GA_C_CONTIGUOUS));
        if (%(out)s == NULL) {
            if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
            %(fail)s
        }
        Py_BEGIN_ALLOW_THREADS
        %(name)serr = GpuArray_read(PyArray_DATA(%(out)s),
                                    PyArray_NBYTES(%(out)s),
                                    %(name)s_ga);
        Py_END_ALLOW_THREADS
        if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
        if (%(name)serr != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Could not read device data.");
            %(fail)s
        }
        """ % {'name': name, 'fail': sub['fail'], 'inp': inputs[0],
               'out': outputs[0]}

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inputs, grads):
        gz, = grads
        return [GpuFromHost(inputs[0].type.context_name)(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        return [self(ev)]

    def infer_shape(self, node, xshp):
        return xshp

host_from_gpu = HostFromGpu()


class GpuFromHost(Op):
    """
    Transfer data to GPU.

    """
    __props__ = ('context_name',)
    _f16_ok = True
    params_type = gpu_context_type

    def __init__(self, context_name):
        self.context_name = context_name

    def __str__(self):
        return 'GpuFromHost<%s>' % (self.context_name,)

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              context_name=self.context_name,
                                              dtype=x.dtype)()])

    def get_params(self, node):
        return get_context(self.context_name)

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = gpuarray.array(x, context=ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(as_gpuarray_variable(
                gz, context_name=self.context_name))]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        return [self(ev)]

    def infer_shape(self, node, xshp):
        return xshp

    def c_headers(self):
        return ["gpuarray_helper.h"]

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        PyArrayObject *%(name)s_tmp;
        %(name)s_tmp = PyArray_GETCONTIGUOUS(%(inp)s);
        int err;
        if (%(name)s_tmp == NULL)
          %(fail)s

        if (%(out)s != NULL && GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) &&
            theano_size_check(%(out)s, PyArray_NDIM(%(name)s_tmp),
                              (size_t *)PyArray_DIMS(%(name)s_tmp),
                              get_typecode((PyObject *)PyArray_DESCR(%(name)s_tmp)))) {
          Py_BEGIN_ALLOW_THREADS
          err = GpuArray_write(&%(out)s->ga, PyArray_DATA(%(name)s_tmp),
                               PyArray_NBYTES(%(name)s_tmp));
          Py_END_ALLOW_THREADS
          Py_DECREF(%(name)s_tmp);
          if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError, "Could not write data to gpu");
            %(fail)s;
          }
        } else {
          Py_XDECREF(%(out)s);
          // This method will release the GIL when needed.
          %(out)s = pygpu_fromhostdata(PyArray_DATA(%(name)s_tmp),
                                       get_typecode((PyObject *)PyArray_DESCR(%(name)s_tmp)),
                                       PyArray_NDIM(%(name)s_tmp),
                                       (size_t *)PyArray_DIMS(%(name)s_tmp),
                                       (ssize_t *)PyArray_STRIDES(%(name)s_tmp),
                                       %(ctx)s,
                                       Py_None);
          Py_DECREF(%(name)s_tmp);
          if (%(out)s == NULL) {
              %(fail)s
          }
        }
        """ % {'name': name, 'inp': inputs[0], 'ctx': sub['params'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (9,)


class GpuToGpu(Op):
    """
    Transfer data between GPUs.

    """
    __props__ = ('context_name',)
    _f16_ok = True
    params_type = gpu_context_type

    def __init__(self, context_name):
        self.context_name = context_name

    def __str__(self):
        return 'GpuToGpu<%s>' % (self.context_name,)

    def make_node(self, x):
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              context_name=self.context_name,
                                              dtype=x.dtype)()])

    def get_params(self, node):
        return get_context(self.context_name)

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = x.transfer(ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [GpuToGpu(inputs[0].type.context_name)(gz)]

    def R_op(self, inputs, eval_points):
        return self(eval_points[0])

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_transfer(%(inp)s, %(ctx)s, 0);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % {'inp': inputs[0], 'ctx': sub['params'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (0,)


class GpuAlloc(HideC, Alloc):
    """
    Allocate initialized memory on the GPU.

    Parameters
    ----------
    context_name : str
        The name of the context in which to allocate memory
    memset_0 : bool
        It's only an optimized version. True, it means the
        value is always 0, so the c code call memset as it is faster.

    """

    __props__ = ('memset_0', 'context_name')
    _f16_ok = True
    params_type = gpu_context_type

    def __init__(self, context_name, memset_0=False):
        self.context_name = context_name
        self.memset_0 = memset_0

    def get_params(self, node):
        return get_context(self.context_name)

    def __str__(self):
        # Hide the memset parameter when not used to prevent confusion.
        if self.memset_0:
            m = "{memset_0=True}"
        else:
            m = ""
        return "%s<%s>%s" % (self.__class__.__name__, self.context_name, m)

    def make_node(self, value, *shape):
        value = as_gpuarray_variable(value, context_name=self.context_name)
        sh, bcast = self.validate_shape(shape)
        if value.ndim > len(sh):
            TypeError("The GpuAlloc value to use has more dimensions "
                      "than the specified shape", value.ndim, len(sh))
        otype = value.type.clone(broadcastable=bcast)
        return Apply(self, [value] + sh, [otype()])

    def c_headers(self):
        return ['<numpy_compat.h>']

    def perform(self, node, inputs, outs, ctx):
        out, = outs
        v = inputs[0]
        sh = tuple(map(int, inputs[1:]))
        if out[0] is None or out[0].shape != sh:
            if self.memset_0:
                out[0] = gpuarray.zeros(sh, dtype=v.dtype, context=ctx)
            else:
                out[0] = gpuarray.empty(sh, dtype=v.dtype, context=ctx)
                out[0][...] = v
        else:
            out[0][...] = v
        if config.gpuarray.sync:
            out[0].sync()

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        ndim = len(inp[1:])
        zz, = out

        memset_0 = int(self.memset_0)
        code = """
        int i;
        size_t %(name)s_shape[%(ndim)s];
        """ % dict(name=name, ndim=ndim)

        for i, shp_i in enumerate(inp[1:]):
            code += """
        %(name)s_shape[%(i)s] = ((dtype_%(shp_i)s *)PyArray_DATA(%(shp_i)s))[0];
        """ % dict(name=name, i=i, shp_i=shp_i)

        code += """
        int need_new_out = (NULL == %(zz)s || %(zz)s->ga.nd != %(ndim)s);

        if (!need_new_out)
            for (i = 0; i < %(ndim)s; i++)
                need_new_out |= %(zz)s->ga.dimensions[i] != %(name)s_shape[i];

        if (need_new_out && (%(memset_0)s)) {
            //pygpu_zeros can be faster then empty followed by memset.
            Py_XDECREF(%(zz)s);
            %(zz)s = pygpu_zeros(%(ndim)s, %(name)s_shape,
                                 %(vv)s->ga.typecode, GA_C_ORDER,
                                 %(ctx)s, Py_None);
            if (!%(zz)s) {
                %(fail)s
            }
        } else {
            if (need_new_out) {
                Py_XDECREF(%(zz)s);
                %(zz)s = pygpu_empty(%(ndim)s, %(name)s_shape,
                                     %(vv)s->ga.typecode, GA_C_ORDER,
                                     %(ctx)s, Py_None);
                if (!%(zz)s) {
                    %(fail)s
                }
            }
            if (%(memset_0)s && GpuArray_ISONESEGMENT(&%(zz)s->ga))
            {
                int err = GpuArray_memset(&%(zz)s->ga, 0);
                if (err != GA_NO_ERROR)
                {
                    PyErr_Format(PyExc_MemoryError,
                                 "GpuAlloc: Error memsetting %%llu"
                                 " element of device memory to 0.",
                                 (unsigned long long)PyGpuArray_SIZE(%(zz)s));
                    %(fail)s;
                }
            }
            else if (GpuArray_setarray(&%(zz)s->ga, &%(vv)s->ga) !=
                     GA_NO_ERROR) {
                PyErr_SetString(PyExc_ValueError, "setarray failed");
                %(fail)s
            }
        }
        """ % dict(name=name, ndim=ndim, zz=zz, vv=vv, ctx=sub['params'],
                   fail=sub['fail'], memset_0=memset_0)

        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(zz)s->ga);" % dict(zz=zz)

        return code

    def c_code_cache_version(self):
        return (3,)

    def do_constant_folding(self, node):
        from . import subtensor, blas
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            # The following ops work inplace of their input id 0.
            elif (client[1] == 0 and
                  # Ops that will work inplace on the Alloc. So if they
                  # get constant_folded, they would copy the
                  # constant and this is less efficients.

                  # Not doing the constant folding could also lower
                  # the peak memory usage, as we the "constant" won't
                  # always exists.
                  isinstance(client[0].op,
                             (subtensor.GpuIncSubtensor,
                              subtensor.GpuAdvancedIncSubtensor1,
                              subtensor.GpuAdvancedIncSubtensor1_dev20,
                              blas.GpuGemm, blas.GpuGemv,
                              blas.GpuGer)
                             )):
                return False
            # If the clients is a transfer, we don't want to fold. We
            # let the moving opt finish before deciding what to do.
            elif isinstance(client[0].op, HostFromGpu):
                return False
        return True


class GpuAllocEmpty(HideC, Alloc):
    """
    Allocate uninitialized memory on the GPU.

    """
    __props__ = ('dtype', 'context_name')
    _f16_ok = True
    params_type = gpu_context_type

    def __init__(self, dtype, context_name):
        self.dtype = dtype
        self.context_name = context_name

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, *shape):
        sh, bcast = self.validate_shape(shape)
        output = GpuArrayType(dtype=self.dtype, broadcastable=bcast,
                              context_name=self.context_name)()
        output.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        # The outut can contain nan/inf.
        output.type.filter_checks_isfinite = False
        output.tag.nan_guard_mode_check = False
        return Apply(self, sh, [output])

    def debug_perform(self, node, inputs, out_, ctx):
        self.perform(node, inputs, out_, ctx)
        out_[0][0][:] = -123456789

    def perform(self, node, inputs, out_, ctx):
        out = out_[0]
        sh = [int(i) for i in inputs]
        if out[0] is None or out[0].shape != sh:
            out[0] = pygpu.empty(sh, dtype=self.dtype, context=ctx)
        # if out[0] is the right shape, we just return it

    def c_headers(self):
        return ['<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_code(self, node, name, inp, out, sub):
        ndim = len(inp)
        zz = out[0]
        fail = sub['fail']

        code = ["""
int i;
size_t shape[%(ndim)s];
""" % dict(ndim=ndim)]

        for i, shp_i in enumerate(inp):
            code.append("""
shape[%(i)s] = ((dtype_%(shp_i)s *)PyArray_DATA(%(shp_i)s))[0];
""" % dict(i=i, shp_i=shp_i))

        code.append("""
if (theano_prep_output(&%(zz)s, %(ndim)s, shape, %(type)s, GA_C_ORDER,
                       %(ctx)s)) {
  %(fail)s
}
""" % dict(zz=zz, ndim=ndim, type=gpuarray.dtype_to_typecode(self.dtype),
           fail=fail, ctx=sub['params']))

        return ''.join(code)

    def c_code_cache_version(self):
        return (1,)

    def do_constant_folding(self, node):
        return False

    def infer_shape(self, node, input_shapes):
        return [node.inputs]

    def grad(self, *args):
        # Don't reuse the grad implementation from Alloc
        raise NotImplementedError("grad disabled")


def empty_like(var):
    return GpuAllocEmpty(var.type.dtype, var.type.context_name)(*var.shape)


class GpuContiguous(Op):
    """
    Return a C contiguous version of the input.

    This may either pass the object as-is (if already C contiguous) or
    make a copy.

    """
    __props__ = ()
    view_map = {0: [0]}
    _f16_ok = True

    def grad(self, inputs, dout):
        x, = inputs
        dout, = dout
        dout = as_gpuarray_variable(dout, context_name=infer_context_name(x))

        return [dout]

    def make_node(self, input):
        input = as_gpuarray_variable(input,
                                     context_name=infer_context_name(input))
        return Apply(self, [input], [input.type()])

    def c_headers(self):
        return ['<numpy_compat.h>']

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        z, = out
        fail = sub['fail']
        str = """
        {
            if (GpuArray_IS_C_CONTIGUOUS(&(%(input)s->ga))){
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if ((NULL == %(z)s)""" % locals()
        for i in xrange(len(node.inputs[0].type.broadcastable)):
            str += "\n|| (PyGpuArray_DIMS(%(input)s)[%(i)s] != PyGpuArray_DIMS(%(z)s)[%(i)s])" % locals()
        str += """
                || !GpuArray_IS_C_CONTIGUOUS(&(%(z)s->ga)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = pygpu_copy(%(input)s, GA_C_ORDER);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            }else if(pygpu_move(%(z)s, %(input)s) == -1) {
                %(fail)s;
            }
        }
        """ % locals()
        return str

gpu_contiguous = GpuContiguous()


class GpuReshape(HideC, tensor.Reshape):
    """
    Reshape for GPU variables.

    """

    _f16_ok = True

    # __hash__, __eq__, __str__ come from tensor.Reshape
    def make_node(self, x, shp):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, context_name=ctx_name)
        res = host_from_gpu(x).reshape(shp, ndim=self.ndim)
        otype = GpuArrayType(dtype=res.dtype,
                             broadcastable=res.broadcastable,
                             context_name=ctx_name)
        return Apply(self, [x, shp], [otype()])

    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to GpuReshape.perform'
                             ' has incorrect length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)

        if shp.prod() != x.size:
            # We need to do check here to raise the same error as NumPy.
            # We should make pygpu do the same.
            ss = 1
            nb_m1 = 0
            for i in shp:
                if i == -1:
                    nb_m1 += 1
                else:
                    ss *= i
            if nb_m1 > 1:
                raise ValueError("Only one -1 is accepted in the new shape")
            elif nb_m1 == 1:
                if (x.size % ss) != 0:
                    raise ValueError("When using -1 in new shape, the computed new shape must be an multiple of the original shape.")
            else:
                raise ValueError("total size of new array must be unchanged")
        out[0] = x.reshape(tuple(shp))

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shape = inputs
        output, = outputs
        new_ndim = self.ndim
        sdtype = node.inputs[1].type.dtype_specs()[1]
        fail = sub['fail']
        return """
        size_t old_size = 1, new_size = 1;
        size_t new_dims[%(new_ndim)s];
        int compute_axis = -1;

        assert (PyArray_NDIM(%(shape)s) == 1);
        if (PyArray_DIM(%(shape)s, 0) != %(new_ndim)s)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: given shape is of incorrect "
                         "length (%%d should be %%d).",
                         PyArray_DIM(%(shape)s, 0), %(new_ndim)s);
            %(fail)s;
        }

        for (size_t i = 0; i < %(x)s->ga.nd; ++i)
            old_size *= %(x)s->ga.dimensions[i];

        for (size_t i = 0; i < %(new_ndim)s; ++i)
        {
            new_dims[i] = ((%(sdtype)s*)(
                    PyArray_BYTES(%(shape)s) +
                    i * PyArray_STRIDES(%(shape)s)[0]))[0];
            if (new_dims[i] == -1)
            {
                if (compute_axis != -1)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "GpuReshape: only one -1 is accepted "
                                 "in the new shape, but got two at "
                                 "indices %%d and %%zu.",
                                 compute_axis, i);
                    %(fail)s;
                }
                compute_axis = i;
            }
            else
                new_size *= new_dims[i];
        }

        if (compute_axis == -1 && new_size != old_size)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: trying to reshape an array of "
                         "total size %%zu into an array of total size "
                         "%%zu.", old_size, new_size);
            %(fail)s;
        }
        else if (compute_axis != -1 && old_size %% new_size != 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: -1 axis found at index %%d in "
                         "new shape but the total size of the array "
                         "(%%zu) is not divisible by the given shapes "
                         "(%%zu).", compute_axis, old_size, new_size);
            %(fail)s;
        }

        Py_XDECREF(%(output)s);
        %(output)s = pygpu_reshape(%(x)s, %(new_ndim)s, new_dims,
                                   GA_C_ORDER, 0, compute_axis);
        if (%(output)s == NULL)
        {
            %(fail)s;
        }
        """ % locals()


class GpuJoin(HideC, Join):
    """
    Join for GPU.

    """
    _f16_ok = True
    params_type = gpu_context_type

    def make_node(self, axis, *tensors):
        node = Join.make_node(self, axis, *tensors)

        ctx_name = infer_context_name(*tensors)

        def agv(v):
            return as_gpuarray_variable(v, context_name=ctx_name)

        return Apply(self, [node.inputs[0]] + list(map(agv, tensors)),
                     [GpuArrayType(broadcastable=node.outputs[0].broadcastable,
                                   dtype=node.outputs[0].dtype,
                                   context_name=ctx_name)()])

    def get_params(self, node):
        return node.outputs[0].type.context

    def perform(self, node, axis_and_tensors, out_, ctx):
        out, = out_
        axis = int(axis_and_tensors[0])
        if axis < -axis_and_tensors[1].ndim:
            raise IndexError
        if axis < 0:
            axis += axis_and_tensors[1].ndim
        tensors = axis_and_tensors[1:]
        out[0] = pygpu.concatenate(tensors, axis=axis, context=ctx).astype(
            node.outputs[0].dtype)

    def c_code_cache_version(self):
        return (2,)

    def c_support_code(self):
        return """
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#endif
"""

    def c_code(self, node, name, inputs, out_, sub):
        copy_to_list = []
        restype = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        for i, inp in enumerate(inputs[1:]):
            copy_to_list.append("als[%s] = &%s->ga;" % (i, inp))
        return """
const GpuArray **als = (const GpuArray **)PyMem_Malloc(sizeof(GpuArray *) *
                                                       %(n)s);
if (als == NULL) {
  PyErr_NoMemory();
  %(fail)s
}
%(copy_inputs_to_list)s
Py_XDECREF(%(out)s);
{
int axis = PyInt_AsLong((PyObject *)%(axis)s);
if (axis < 0) {
  if (axis == -1 && PyErr_Occurred()) {
    %(fail)s
  }
  axis += als[0]->nd;
  if (axis < 0) {
    PyErr_SetString(PyExc_IndexError, "invalid axis");
    %(fail)s
  }
}
%(out)s = pygpu_concatenate(als, %(n)s, axis,
                            %(restype)s, (PyObject *)&PyGpuArrayType,
                            %(ctx)s);
}
PyMem_Free(als);
if (%(out)s == NULL)
  %(fail)s
        """ % dict(n=len(inputs[1:]), fail=sub['fail'], out=out_[0],
                   axis=inputs[0], copy_inputs_to_list='\n'.join(copy_to_list),
                   restype=restype, ctx=sub['params'])

gpu_join = GpuJoin()


class GpuSplit(HideC, Split):
    """
    Split for GPU.

    """
    def make_node(self, x, axis, splits):
        node = Split.make_node(self, x, axis, splits)
        x = as_gpuarray_variable(x, infer_context_name(x))
        outs = [GpuArrayType(dtype=o.dtype, broadcastable=o.broadcastable,
                             context_name=x.type.context_name)()
                for o in node.outputs]
        return Apply(self, [x] + node.inputs[1:], outs)
    # we reuse the perform of the CPU op, which is suitable


class GpuEye(GpuKernelBase, Op):
    """
    Eye for GPU.

    """
    __props__ = ('dtype', 'context_name')
    _f16_ok = True

    def __init__(self, dtype=None, context_name=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype
        self.context_name = context_name

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, n, m, k):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        k = tensor.as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        otype = GpuArrayType(dtype=self.dtype,
                             broadcastable=(False, False),
                             context_name=self.context_name)

        # k != 0 isn't implemented on the GPU yet.
        assert tensor.get_scalar_constant_value(k) == 0
        return Apply(self, [n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i])
                for i in xrange(3)]

    def gpu_kernels(self, node, name):
        code = """
KERNEL void k(GLOBAL_MEM %(ctype)s *a, ga_size n, ga_size m) {
    ga_size nb = n < m ? n : m;
    for (ga_size i = LID_0; i < nb; i += LDIM_0) {
        a[i*m + i] = %(write_a)s(1);
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype),
            name=name, write_a=write_w(self.dtype))
        return [Kernel(
                code=code, name="k",
                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(self.dtype),
                objvar='k_eye_' + name)]

    def c_code(self, node, name, inp, out, sub):
        n, m = inp
        z, = out
        fail = sub['fail']
        ctx = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        sync = bool(config.gpuarray.sync)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        size_t dims[2] = {0, 0};
        size_t ls, gs;
        void *args[3];
        int err;

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        Py_CLEAR(%(z)s);

        %(z)s = pygpu_zeros(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            %(ctx)s, Py_None);
        if (%(z)s == NULL) {
            %(fail)s
        }

        args[0] = %(z)s->ga.data;
        args[1] = &dims[0];
        args[2] = &dims[1];
        ls = 1;
        gs = 256;
        err = GpuKernel_call(&%(kname)s, 1, &ls, &gs, 0, args);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "gpuarray error: kEye: %%s. n%%lu, m=%%lu.",
                         GpuKernel_error(&%(kname)s, err),
                         (unsigned long)dims[0], (unsigned long)dims[1]);
            %(fail)s;
        }

        if(%(sync)d)
            GpuArray_sync(&%(z)s->ga);
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (5,)
