from __future__ import absolute_import, print_function, division
import os
import copy
import re
import numpy as np

import theano
from theano import Op, Apply, Type, Variable
from theano import tensor, config
from theano.gradient import grad_undefined
from theano.scalar import (bool as bool_t,
                           int32 as int32_t)
from theano.tensor.basic import (
    Alloc, AllocEmpty, alloc_validate_shape, Join, Split)

from theano.gof import HideC, COp, ParamsType
from theano.gof.utils import MethodNotDefined
from theano.gof.opt import copy_stack_trace

from collections import deque

from six import string_types
from six.moves import xrange
from six import iteritems

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .type import (GpuArrayType, GpuArrayConstant, gpu_context_type,
                   get_context, ContextNotDefined, EQ_MAP)
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
            return copy_stack_trace(x, GpuFromHost(context_name)(x))

    # Try _as_GpuArrayVariable if possible
    if hasattr(x, '_as_GpuArrayVariable'):
        return copy_stack_trace(x, x._as_GpuArrayVariable(context_name))

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


def gpuarray_helper_inc_dir():
    return os.path.join(os.path.dirname(__file__), 'c_code')


class Kernel(object):
    """
    This class groups together all the attributes of a gpu kernel.

    `params` should contain the data type for each argument.  Buffer
    arguments should use the GpuArray class as the data type and
    scalar should use their equivalent numpy dtype.  For ga_size and
    ga_ssize, use gpuarray.SIZE and gpuarray.SSIZE.

    If the `ctypes` flags is set to `True` then it should be a C
    string which represent the typecode to use.

    `flags` can contain the following keys whose values are booleans:

        have_double
            the kernel uses double-typed variables somewhere
        have_small
            the kernel uses variables whose type takes less than 4
            bytes somewhere
        have_complex
            the kernel uses complex values somewhere
        have_half
            the kernel uses half-floats somewhere
        ctypes
            the `params` list consists of C typecodes

    It can also have the key `cflags` which is a string of C flag
    values like this `"GA_USE_DOUBLE|GA_USE_SMALL"`.

    Parameters
    ----------
    code: str
        The source code of the kernel.
    params: list
        list of parameter types.
    name: str
        the name of the kernel function in the source.
    flags: dict
        dictionary of flags
    codevar: str
        the name of the variable for the code object.
        (defaults to `kcode_` + name)
    objvar: str
        the name of the variable for the kernel object.
        (defaults to `k_` + name)
    fname: str
        the name of the function wrapper.
        (defaults to name + `_call`)
    sname: str
        the name of the scheduled call function
        (defaults to name _ `_scall`)

    """

    def __init__(self, code, params, name, flags,
                 codevar=None, objvar=None, fname=None, sname=None):
        self.code = code
        self.params = params
        self.name = name
        self.flags = flags
        if codevar is None:
            codevar = 'kcode_' + name
        self.codevar = codevar
        if objvar is None:
            objvar = 'k_' + name
        self.objvar = objvar
        if fname is None:
            fname = name + '_call'
        self.fname = fname
        if sname is None:
            sname = name + '_scall'
        self.sname = sname

    @staticmethod
    def get_flags(*types):
        def get_dtype(t):
            if isinstance(t, string_types):
                return np.dtype(t)
            elif isinstance(t, Type):
                return t.dtype
            elif isinstance(t, Variable):
                return t.type.dtype
            else:
                raise TypeError("can't get a dtype from %s" % (type(t),))
        dtypes = [get_dtype(t) for t in types]
        flags = dict()
        if any(d == np.float64 for d in dtypes):
            flags['have_double'] = True
        if any(d.itemsize < 4 for d in dtypes):
            flags['have_small'] = True
        if any(d.kind == 'c' for d in dtypes):
            flags['have_complex'] = True
        if any(d == np.float16 for d in dtypes):
            flags['have_half'] = True
        return flags

    def _get_c_flags(self):
        res = []
        if self.flags.get('cflags', '') != '':
            res.append(self.flags['cflags'])
        if self.flags.get('have_double', False):
            res.append('GA_USE_DOUBLE')
        if self.flags.get('have_small', False):
            res.append('GA_USE_SMALL')
        if self.flags.get('have_complex', False):
            res.append('GA_USE_COMPLEX')
        if self.flags.get('have_half', False):
            res.append('GA_USE_HALF')
        res = '|'.join(res)
        if not res:
            return '0'
        return res

    def _get_py_flags(self):
        res = dict(self.flags)
        cflags = res.pop('cflags', '')
        for fl in cflags.split('|'):
            fl = fl.strip()
            if fl == 'GA_USE_DOUBLE':
                res['have_double'] = True
            if fl == 'GA_USE_SMALL':
                res['have_small'] = True
            if fl == 'GA_USE_COMPLEX':
                res['have_complex'] = True
            if fl == 'GA_USE_HALF':
                res['have_half'] = True
        return res

    def _get_c_types(self):
        def m(t):
            if t == gpuarray.GpuArray:
                return "GA_BUFFER"
            else:
                return str(gpuarray.dtype_to_typecode(t))
        return ', '.join(m(t) for t in self.params)


def get_ctype(dtype):
    if dtype is gpuarray.GpuArray:
        return "gpudata *"
    elif isinstance(dtype, np.dtype):
        return 'npy_' + dtype.name
    elif dtype == gpuarray.SIZE:
        return "size_t"
    elif dtype == gpuarray.SSIZE:
        return "ssize_t"
    else:
        dtype = np.dtype(dtype)
        return 'npy_' + dtype.name


class GpuKernelBase(object):
    """
    Base class for operations that need to compile kernels.

    It is not mandatory to use this class, but it helps with a lot of
    the small things that you have to pay attention to.

    """
    params_type = gpu_context_type

    def get_params(self, node):
        # Default implementation, suitable for most sub-classes.
        # To be necessarly overridden in a subclass that uses a ParamsType.
        assert (self.params_type is gpu_context_type and
                node.inputs and
                isinstance(node.inputs[0].type, GpuArrayType))
        return node.inputs[0].type.context

    def get_gpu_context(self, node):
        # Private method used to retrieve GPU context, instead of
        # directly using self.get_params(node), as this latter may be overridden.
        if isinstance(self.params_type, ParamsType) and self.params_type.has_type(gpu_context_type):
            # Get field name of gpu_context_type into ParamsType object.
            gpu_context_field = self.params_type.get_field(gpu_context_type)
            # Get Params object (self.get_params() should have been overridden).
            wrap = self.get_params(node)
            # Get GPU context from Params object.
            return getattr(wrap, gpu_context_field)
        assert self.params_type is gpu_context_type
        return self.get_params(node)

    def get_gpu_context_c_name(self, params_c_name):
        # Private method used to retrieve C name of GPU context variable,
        # instead of directly using sub['params'], as params may not be a GPU context
        # (e.g. for sub-classes that use ParamsType).
        if isinstance(self.params_type, ParamsType) and self.params_type.has_type(gpu_context_type):
            return "(%s->%s)" % (params_c_name, self.params_type.get_field(gpu_context_type))
        assert self.params_type is gpu_context_type
        return params_c_name

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
        return o + ['gpuarray/types.h', 'numpy/npy_common.h']

    def c_header_dirs(self):
        try:
            o = super(GpuKernelBase, self).c_header_dirs()
        except MethodNotDefined:
            o = []
        # We rely on the input types for the directory to gpuarray includes
        return o + [np.get_include()]

    def _generate_kernel_code(self, k):
        code = '\\n'.join(l for l in k.code.split('\n'))
        code = code.replace('"', '\\"')
        return ("""static const char *%(cname)s_unsigned = "%(code)s";
                static const char *%(cname)s = (char *)%(cname)s_unsigned;
                """ %
                dict(cname=k.codevar, code=code))

    def _generate_kernel_vars(self, k):
        return """GpuKernel %(kname)s;""" % dict(kname=k.objvar)

    def _generate_kernel_wrap(self, k):
        args = []
        setargs = []
        for i, p in enumerate(k.params):
            args.append("{} arg{}".format(get_ctype(p), i))
            if p is gpuarray.GpuArray:
                setarg = "GpuKernel_setarg(&{0}, {1}, arg{1});"
            else:
                setarg = "GpuKernel_setarg(&{0}, {1}, &arg{1});"
            setargs.append(setarg.format(k.objvar, i))

        args = ', '.join(args)
        setargs = '\n  '.join(setargs)

        return """
int {fname}(unsigned int _nd, size_t *_gdim, size_t *_ldim, size_t _shared,
                  {args}) {{
  {setargs}

  return GpuKernel_call(&{kname}, _nd, _gdim, _ldim, _shared, NULL);
}}

int {sname}(unsigned int _nd, size_t *_n, size_t _shared, {args}) {{
  size_t _gs = 0;
  size_t _ls = 0;
  int _err;

  if (_nd != 1) return GA_UNSUPPORTED_ERROR;

  _err = GpuKernel_sched(&{kname}, _n[0], &_gs, &_ls);
  if (_err != GA_NO_ERROR)
    return _err;

  {setargs}

  return GpuKernel_call(&{kname}, 1, &_gs, &_ls, _shared, NULL);
}}
        """.format(args=args, fname=k.fname, setargs=setargs, sname=k.sname,
                   kname=k.objvar)

    def c_support_code_apply(self, node, name):
        kernels = self.gpu_kernels(node, name)
        codes = '\n'.join(self._generate_kernel_code(k) for k in kernels)
        return codes

    def c_support_code_struct(self, node, name):
        kernels = self.gpu_kernels(node, name)
        kvars = '\n'.join(self._generate_kernel_vars(k) for k in kernels)
        wrappers = '\n'.join(self._generate_kernel_wrap(k) for k in kernels)
        return kvars + '\n' + wrappers

    def _generate_zeros(self, k):
        return """memset(&%(v)s, 0, sizeof(%(v)s));""" % dict(v=k.objvar)

    def _generate_kernel_init(self, k, fail, ctx):
        return """{
  int err;
  int types[%(numargs)u] = {%(types)s};
  if ((err = GpuKernel_init(&%(ovar)s, %(ctx)s->ctx, 1,
                            &%(cname)s, NULL, "%(kname)s", %(numargs)u,
                            types, %(flags)s, NULL)) != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "GpuKernel_init error %%d: %%s",
                 err, gpucontext_error(%(ctx)s->ctx, err));
    %(fail)s
  }
}""" % dict(numargs=len(k.params), types=k._get_c_types(),
            ovar=k.objvar, kname=k.name, cname=k.codevar,
            flags=k._get_c_flags(), fail=fail, ctx=ctx)

    def c_init_code_struct(self, node, name, sub):
        ctx = self.get_gpu_context_c_name(sub['params'])
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
        v = self.c_code_cache_version()
        if not v:
            return ()
        return (v, self.kernel_version(node))

    def kernel_version(self, node):
        """
        If you override :meth:`c_code_cache_version_apply`, call this
        method to have the version of the kernel support code.

        Parameters
        ----------
        node : apply node
            The node that we need the cache version for.

        """
        return (9,)


def forward_string_meth(name):
    def f(*args):
        res = getattr(GpuKernelBase, name)(*args)
        try:
            res = res + '\n' + getattr(COp, name)(*args)
        except MethodNotDefined:
            pass
        return res
    f.__name__ = name
    return f


def get_dtype(s):
    if s == '*':
        return gpuarray.GpuArray
    if s == 'size':
        return gpuarray.SIZE
    if s == 'ssize':
        return gpuarray.SSIZE
    else:
        return np.dtype(s)


class CGpuKernelBase(COp, GpuKernelBase):
    """
    Class to combine GpuKernelBase and COp.

    It adds a new section type 'kernels' where you can define kernels
    with the '#kernel' tag
    """
    SECTIONS = copy.copy(COp.SECTIONS)
    SECTIONS.add('kernels')

    kernel_re = re.compile(r'^#kernel ([a-zA-Z_].*?)$', re.MULTILINE)

    get_params = GpuKernelBase.get_params
    c_support_code_apply = forward_string_meth('c_support_code_apply')
    c_support_code_struct = forward_string_meth('c_support_code_struct')
    c_init_code_struct = forward_string_meth('c_init_code_struct')
    c_cleanup_code_struct = forward_string_meth('c_cleanup_code_struct')

    def c_code_cache_version_apply(self, node):
        return GpuKernelBase.c_code_cache_version_apply(self, node)

    def _type_macros(self, node):
        define_template = "#define %s %s\n"
        undef_template = "#undef %s\n"
        define_macros = []
        undef_macros = []
        for i, v in enumerate(node.inputs):
            if isinstance(v.type, GpuArrayType):
                macro_name = "DTYPE_INPUT_%d" % (i,)
                macro_value = pygpu.gpuarray.dtype_to_ctype(v.dtype)
                define_macros.append(
                    define_template %
                    (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)
        for i, v in enumerate(node.outputs):
            if isinstance(v.type, GpuArrayType):
                macro_name = "DTYPE_OUTPUT_%d" % (i,)
                macro_value = pygpu.gpuarray.dtype_to_ctype(v.dtype)
                define_macros.append(
                    define_template %
                    (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

        return ''.join(define_macros), ''.join(undef_macros)

    def gpu_kernels(self, node, name):
        if hasattr(self, '_cached_kernels'):
            return self._cached_kernels
        if 'kernels' in self.code_sections:
            code = self.code_sections['kernels']
            split = self.kernel_re.split(code)
            if split[0].strip() != '':
                raise ValueError("Stray code in kernels section before the "
                                 "first #kernel statement.")
            def_macros, undef_macros = self._type_macros(node)
            n = 1
            res = []
            while n < len(split):
                kspec = split[n]
                kcode = split[n + 1]
                splt2 = kspec.split(':')
                if len(splt2) != 3:
                    raise ValueError("Bad kernel spec: %s" % (kspec,))
                kname = splt2[0].strip()
                ktypes = [get_dtype(s.strip()) for s in splt2[1].split(',')]
                kflags = splt2[2].strip()
                kcode = def_macros + '\n' + kcode + '\n' + undef_macros
                res.append(Kernel(kcode, ktypes, kname,
                                  flags=dict(cflags=kflags)))
                n += 2
            self._cached_kernels = res
            return res
        else:
            return GpuKernelBase.gpu_kernels(self, node, name)


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
        out_var = tensor.TensorType(dtype=x.dtype,
                                    broadcastable=x.broadcastable)()
        # Keep the special comparison if there is one.
        values_eq_approx = getattr(x.tag, 'values_eq_approx', None)
        if values_eq_approx:
            out_var.tag.values_eq_approx = EQ_MAP.get(values_eq_approx,
                                                      values_eq_approx)
        return Apply(self, [x], [out_var])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = np.asarray(x)

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
        if "complex" in x.dtype:
            raise TypeError("complex not supported in the new gpuarray back-end.", x)
        out_var = GpuArrayType(broadcastable=x.broadcastable,
                               context_name=self.context_name,
                               dtype=x.dtype)()
        # Keep the special comparison if there is one.
        values_eq_approx = getattr(x.tag, 'values_eq_approx', None)
        if values_eq_approx:
            out_var.tag.values_eq_approx = EQ_MAP.get(values_eq_approx,
                                                      values_eq_approx)
        return Apply(self, [x], [out_var])

    def get_params(self, node):
        return get_context(self.context_name)

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = gpuarray.array(x, context=ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [as_gpuarray_variable(
                gz, context_name=self.context_name).transfer('cpu')]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        return [self(ev)]

    def infer_shape(self, node, xshp):
        return xshp

    def c_headers(self):
        return ["gpuarray_helper.h"]

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        PyArrayObject *%(name)s_tmp;
        %(name)s_tmp = PyArray_GETCONTIGUOUS(%(inp)s);
        int err;
        if (%(name)s_tmp == NULL)
          %(fail)s

        if (%(out)s == NULL || !GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) ||
            !theano_size_check(%(out)s, PyArray_NDIM(%(name)s_tmp),
                               (size_t *)PyArray_DIMS(%(name)s_tmp),
                               get_typecode((PyObject *)PyArray_DESCR(%(name)s_tmp)))) {
          Py_XDECREF(%(out)s);
          %(out)s = pygpu_empty(PyArray_NDIM(%(name)s_tmp),
                                (size_t *)PyArray_DIMS(%(name)s_tmp),
                                get_typecode((PyObject *)PyArray_DESCR(%(name)s_tmp)),
                                GA_C_ORDER, %(ctx)s, Py_None);
          if (%(out)s == NULL) {
            Py_DECREF(%(name)s_tmp);
            %(fail)s;
          }
        }

        Py_BEGIN_ALLOW_THREADS
        err = GpuArray_write(&%(out)s->ga, PyArray_DATA(%(name)s_tmp),
                             PyArray_NBYTES(%(name)s_tmp));
        Py_END_ALLOW_THREADS
        Py_DECREF(%(name)s_tmp);
        if (err != GA_NO_ERROR) {
          PyErr_Format(PyExc_RuntimeError, "Could not write data to gpu");
          %(fail)s;
        }
        """ % {'name': name, 'inp': inputs[0], 'ctx': sub['params'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (10,)


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
        %(out)s = pygpu_empty(%(inp)s->ga.nd,
                              %(inp)s->ga.dimensions,
                              %(inp)s->ga.typecode,
                              GpuArray_IS_C_CONTIGUOUS(&(%(inp)s->ga)) ? GA_C_ORDER:GA_F_ORDER,
                              %(ctx)s, Py_None);
        if (%(out)s == NULL) {
            %(fail)s
        }
        if (pygpu_transfer(%(out)s, %(inp)s)) {
            %(fail)s
        }
        """ % {'inp': inputs[0], 'ctx': sub['params'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (1,)


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
    params_type = ParamsType(context=gpu_context_type, memset_0=bool_t)

    def __init__(self, context_name, memset_0=False):
        self.context_name = context_name
        self.memset_0 = memset_0

    def get_params(self, node):
        return self.params_type.get_params(context=get_context(self.context_name),
                                           memset_0=self.memset_0)

    def __str__(self):
        # Hide the memset parameter when not used to prevent confusion.
        if self.memset_0:
            m = "{memset_0=True}"
        else:
            m = ""
        return "%s<%s>%s" % (self.__class__.__name__, self.context_name, m)

    def make_node(self, value, *shape):
        value = as_gpuarray_variable(value, context_name=self.context_name)
        sh, bcast = alloc_validate_shape(shape)
        if value.ndim > len(sh):
            TypeError("The GpuAlloc value to use has more dimensions "
                      "than the specified shape", value.ndim, len(sh))
        otype = value.type.clone(broadcastable=bcast)
        return Apply(self, [value] + sh, [otype()])

    def c_headers(self):
        return ['<numpy_compat.h>']

    def perform(self, node, inputs, outs, params):
        out, = outs
        v = inputs[0]
        sh = tuple(map(int, inputs[1:]))
        if out[0] is None or out[0].shape != sh:
            if self.memset_0:
                out[0] = gpuarray.zeros(sh, dtype=v.dtype, context=params.context)
            else:
                out[0] = gpuarray.empty(sh, dtype=v.dtype, context=params.context)
                out[0][...] = v
        else:
            out[0][...] = v

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        ndim = len(inp[1:])
        zz, = out

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

        if (need_new_out && (%(params)s->memset_0)) {
            //pygpu_zeros can be faster then empty followed by memset.
            Py_XDECREF(%(zz)s);
            %(zz)s = pygpu_zeros(%(ndim)s, %(name)s_shape,
                                 %(vv)s->ga.typecode, GA_C_ORDER,
                                 %(params)s->context, Py_None);
            if (!%(zz)s) {
                %(fail)s
            }
        } else {
            if (need_new_out) {
                Py_XDECREF(%(zz)s);
                %(zz)s = pygpu_empty(%(ndim)s, %(name)s_shape,
                                     %(vv)s->ga.typecode, GA_C_ORDER,
                                     %(params)s->context, Py_None);
                if (!%(zz)s) {
                    %(fail)s
                }
            }
            if (%(params)s->memset_0 && GpuArray_ISONESEGMENT(&%(zz)s->ga))
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
        """ % dict(name=name, ndim=ndim, zz=zz, vv=vv, params=sub['params'],
                   fail=sub['fail'])

        return code

    def c_code_cache_version(self):
        return (4,)

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
                              subtensor.GpuAdvancedIncSubtensor,
                              blas.GpuGemm, blas.GpuGemv,
                              blas.GpuGer)
                             )):
                return False
            # If the clients is a transfer, we don't want to fold. We
            # let the moving opt finish before deciding what to do.
            elif isinstance(client[0].op, HostFromGpu):
                return False
        return True


class GpuAllocEmpty(HideC, AllocEmpty):
    """
    Allocate uninitialized memory on the GPU.

    """
    __props__ = ('dtype', 'context_name')
    _f16_ok = True
    params_type = ParamsType(context=gpu_context_type,
                             typecode=int32_t)

    def __init__(self, dtype, context_name):
        self.dtype = dtype
        self.context_name = context_name

    @property
    def typecode(self):
        return gpuarray.dtype_to_typecode(self.dtype)

    def get_params(self, node):
        return self.params_type.get_params(context=get_context(self.context_name),
                                           typecode=self.typecode)

    def make_node(self, *shape):
        sh, bcast = alloc_validate_shape(shape)
        output = GpuArrayType(dtype=self.dtype, broadcastable=bcast,
                              context_name=self.context_name)()
        output.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        # The outut can contain nan/inf.
        output.type.filter_checks_isfinite = False
        output.tag.nan_guard_mode_check = False
        return Apply(self, sh, [output])

    def debug_perform(self, node, inputs, out_, params):
        self.perform(node, inputs, out_, params)
        out_[0][0][:] = -123456789

    def perform(self, node, inputs, out_, params):
        out = out_[0]
        sh = [int(i) for i in inputs]
        if out[0] is None or out[0].shape != sh:
            out[0] = pygpu.empty(sh, dtype=self.dtype, context=params.context)
        # if out[0] is the right shape, we just return it

    def c_headers(self):
        return ['<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

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
if (theano_prep_output(&%(zz)s, %(ndim)s, shape, %(params)s->typecode, GA_C_ORDER,
                       %(params)s->context)) {
  %(fail)s
}
""" % dict(zz=zz, ndim=ndim, fail=fail, params=sub['params']))

        return ''.join(code)

    def c_code_cache_version(self):
        return (2,)

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

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_headers(self):
        return ['<gpuarray_helper.h>']

    def c_code_cache_version(self):
        return (4,)

    def c_code(self, node, name, inp, out, sub):
        return """
        {
            if (GpuArray_IS_C_CONTIGUOUS(&(%(input)s->ga))) {
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if (NULL == %(z)s
                || !theano_size_check(%(z)s, PyGpuArray_NDIM(%(input)s), PyGpuArray_DIMS(%(input)s),
                                      %(input)s->ga.typecode)
                || !GpuArray_IS_C_CONTIGUOUS(&(%(z)s->ga)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = pygpu_copy(%(input)s, GA_C_ORDER);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            } else if(pygpu_move(%(z)s, %(input)s) == -1) {
                %(fail)s;
            }
        }
        """ % dict(input=inp[0], z=out[0], fail=sub['fail'])

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
        shp = tensor.as_tensor_variable(shp)
        res = x.transfer('cpu').reshape(shp, ndim=self.ndim)
        otype = GpuArrayType(dtype=res.dtype,
                             broadcastable=res.broadcastable,
                             context_name=ctx_name)
        return Apply(self, [x, shp], [otype()])

    def perform(self, node, inp, out_, params):
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
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shape = inputs
        output, = outputs
        sdtype = node.inputs[1].type.dtype_specs()[1]
        just_fail = sub['fail']
        fail = """{
        free(new_dims);
        %(just_fail)s
        }""" % dict(just_fail=just_fail)
        params = sub['params']
        return """
        size_t old_size = 1, new_size = 1;
        size_t* new_dims = NULL;
        int compute_axis = -1;

        assert (PyArray_NDIM(%(shape)s) == 1);
        if (PyArray_DIM(%(shape)s, 0) != %(params)s->ndim)
        {
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: given shape is of incorrect "
                         "length (%%d should be %%d).",
                         PyArray_DIM(%(shape)s, 0), %(params)s->ndim);
            %(just_fail)s;
        }

        new_dims = (size_t*) malloc(sizeof(size_t) * %(params)s->ndim);
        if (new_dims == NULL) {
            PyErr_NoMemory();
            %(just_fail)s
        }

        for (size_t i = 0; i < %(x)s->ga.nd; ++i)
            old_size *= %(x)s->ga.dimensions[i];

        for (size_t i = 0; i < %(params)s->ndim; ++i)
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
        %(output)s = pygpu_reshape(%(x)s, %(params)s->ndim, new_dims,
                                   GA_C_ORDER, 0, compute_axis);
        free(new_dims);
        if (%(output)s == NULL)
        {
            %(just_fail)s;
        }
        """ % locals()


class GpuJoin(HideC, Join):
    """
    Join for GPU.

    """
    _f16_ok = True
    __props__ = ("view",)
    params_type = gpu_context_type

    def __init__(self, view=-1):
        self.view = view
        if view != -1:
            # since the first input is always the axis, the tensors
            # start from index 1.
            self.view_map = {0: [1 + view]}

    def __str__(self):
        return Join.__str__(self)

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
        view = self.view
        axis = int(axis_and_tensors[0])
        tensors = axis_and_tensors[1:]

        if axis < -axis_and_tensors[1].ndim:
            raise IndexError
        if axis < 0:
            axis += axis_and_tensors[1].ndim
        # we check these tensors for being empty.
        if (view != -1) and np.all(
                [tensor.shape[axis] == 0 for tensor in
                 tensors[0:view] + tensors[view + 1:]]):
            out[0] = tensors[view]
        else:
            out[0] = pygpu.concatenate(tensors, axis=axis, context=ctx).astype(
                node.outputs[0].dtype)

    def c_code_cache_version(self):
        return (3,)

    def c_support_code(self):
        return """
        #if PY_MAJOR_VERSION >= 3
        #define PyInt_AsLong PyLong_AsLong
        #endif
        """

    def c_headers(self):
        return ['<numpy_compat.h>']

    def c_code(self, node, name, inputs, out_, sub):
        axis, tensors = inputs[0], inputs[1:]
        copy_to_list = []
        restype = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        view = self.view
        non_empty_tensor = tensors[view]
        for i, inp in enumerate(tensors):
            copy_to_list.append("als[%s] = &%s->ga;" % (i, inp))

        n = len(tensors)
        fail = sub['fail']
        out = out_[0]
        copy_inputs_to_list = '\n'.join(copy_to_list)
        ctx = sub['params']

        code = """
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

            int tensors_lens_sum;
            if(%(view)s != -1) {
                tensors_lens_sum = 0;
                for(int i=0; i < %(n)s; i++){
                    tensors_lens_sum += als[i]->dimensions[axis];
                }
                tensors_lens_sum -= PyGpuArray_DIM(%(non_empty_tensor)s, axis);
            }

            if(%(view)s != -1 && tensors_lens_sum == 0) {
                Py_INCREF(%(non_empty_tensor)s);
                %(out)s = %(non_empty_tensor)s;
            }else{
                %(out)s = pygpu_concatenate(als, %(n)s, axis,
                                            %(restype)s, (PyObject *)&PyGpuArrayType,
                                            %(ctx)s);
            }
        }
        PyMem_Free(als);
        if (%(out)s == NULL)
            %(fail)s

        """ % locals()
        return code

gpu_join = GpuJoin()


class GpuSplit(HideC, Split):
    """
    Split for GPU.

    """
    _f16_ok = True

    def __init__(self, len_splits):
        super(GpuSplit, self).__init__(len_splits)
        # The GPU version of Split returns splits as views of the input.
        self.view_map = {}
        for i in xrange(self.len_splits):
            self.view_map[i] = [0]

    def make_node(self, x, axis, splits):
        node = Split.make_node(self, x, axis, splits)
        x = as_gpuarray_variable(x, infer_context_name(x))
        outs = [GpuArrayType(dtype=o.dtype, broadcastable=o.broadcastable,
                             context_name=x.type.context_name)()
                for o in node.outputs]
        return Apply(self, [x] + node.inputs[1:], outs)

    # we reuse the perform of the CPU op, which is suitable

    def c_code_cache_version(self):
        return (2,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inputs, outputs, sub):
        if self.len_splits == 0:
            # There are no outputs, then nothing to do.
            return ''

        # outputs_pointers lists the addresses of the pointers to the outputs.
        outputs_pointers = '&' + (', &'.join(outputs))
        x, axis, splits = inputs
        fail = sub['fail']
        splits_dtype = node.inputs[2].type.dtype_specs()[1]
        axis_dtype = node.inputs[1].type.dtype_specs()[1]
        expected_splits_count = self.len_splits

        main_code = """
        int ndim = PyGpuArray_NDIM(%(x)s);
        int axis = (int)(*(%(axis_dtype)s*)PyArray_GETPTR1(%(axis)s, 0));
        int splits_count = PyArray_DIM(%(splits)s, 0);
        size_t len_along_axis, sum_of_splits = 0;
        %(splits_dtype)s current_split_length;
        size_t* split_points = NULL;
        GpuArray* split_views = NULL;
        GpuArray** split_views_pointers = NULL;
        int i, j;
        PyGpuArrayObject** outputs[] = {%(outputs_pointers)s};

        /* Check inputs. */

        if (splits_count != %(expected_splits_count)s) {
            PyErr_Format(PyExc_ValueError,
                "GpuSplit: splits count (%%d) != expected count (%%d).", splits_count, %(expected_splits_count)s);
            %(fail)s
        }
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            PyErr_Format(PyExc_IndexError, "GpuSplit: invalid axis %%d for a %%d-D array.", axis, ndim);
            %(fail)s
        }
        len_along_axis = PyGpuArray_DIM(%(x)s, axis);
        for (i = 0; i < splits_count; ++i) {
            current_split_length = *(%(splits_dtype)s*)PyArray_GETPTR1(%(splits)s, i);
            if (current_split_length < 0) {
                PyErr_Format(PyExc_ValueError,
                    "GpuSplit: you try to take a negative number (%%ld) of elements.", current_split_length);
                %(fail)s
            }
            sum_of_splits += current_split_length;
        }
        if (sum_of_splits != len_along_axis) {
            PyErr_Format(PyExc_ValueError, "GpuSplit: the splits sums to %%ld, expected %%ld.", sum_of_splits, len_along_axis);
            %(fail)s
        }

        /* Compute splits views. */

        split_points = (size_t*) malloc((splits_count - 1) * sizeof(size_t));
        if (split_points == NULL) {
            PyErr_NoMemory();
            %(fail)s
        }
        split_points[0] = (size_t) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, 0) );
        for(i = 1; i < splits_count - 1; ++i) {
            split_points[i] = split_points[i - 1] + (size_t) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, i) );
        }
        split_views = (GpuArray*) malloc(splits_count * sizeof(GpuArray));
        split_views_pointers = (GpuArray**) malloc(splits_count * sizeof(GpuArray*));
        if (split_views == NULL || split_views_pointers == NULL) {
            PyErr_NoMemory();
            free(split_views_pointers);
            free(split_views);
            free(split_points);
            %(fail)s
        }
        for (i = 0; i < splits_count; ++i) {
            split_views_pointers[i] = split_views + i;
        }
        if (GpuArray_split(split_views_pointers, &%(x)s->ga, splits_count - 1, split_points, axis) != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "GpuSplit: unable to compute split.");
            for (i = 0; i < splits_count; ++i) {
                GpuArray_clear(split_views_pointers[i]);
            }
            free(split_views_pointers);
            free(split_views);
            free(split_points);
            %(fail)s
        }

        /* Put split views into outputs. */
        for (i = 0; i < splits_count; ++i) {
            PyGpuArrayObject** output = outputs[i];
            Py_XDECREF(*output);
            *output = pygpu_fromgpudata(
                split_views[i].data,
                split_views[i].offset,
                split_views[i].typecode,
                split_views[i].nd,
                split_views[i].dimensions,
                split_views[i].strides,
                %(x)s->context,
                1, // output is writable
                Py_None, Py_None
            );
            if (*output == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "GpuSplit: unable to update an output from a split view.");
                for (j = 0; j < splits_count; ++j) {
                    GpuArray_clear(split_views_pointers[j]);
                }
                free(split_views_pointers);
                free(split_views);
                free(split_points);
                %(fail)s
            }
        }

        /* Free memory. */
        for (i = 0; i < splits_count; ++i) {
           GpuArray_clear(split_views_pointers[i]);
        }
        free(split_views_pointers);
        free(split_views);
        free(split_points);
        """

        return main_code % locals()


@theano.compile.profiling.register_profiler_printer
def profile_printer(message, compile_time, fct_call_time,
                    apply_time, apply_cimpl, outputs_size, file):
    if any([x.op.__class__.__name__.lower().startswith("gpu")
            for x in apply_time.keys()]):
        local_time = sum(apply_time.values())
        print('', file=file)
        print('Some info useful for gpu:', file=file)

        fgraphs = set()
        for node in apply_time.keys():
            fgraphs.add(node.fgraph)

        cpu = 0
        gpu = 0
        trans = 0
        for node, t in iteritems(apply_time):
            if isinstance(node.op, (HostFromGpu, GpuFromHost)):
                trans += t
            elif node.op.__class__.__name__.lower().startswith("gpu"):
                gpu += t
            else:
                cpu += t
        print('', file=file)
        print("    Spent %.3fs(%.2f%%) in cpu Op, %.3fs(%.2f%%) in gpu Op and %.3fs(%.2f%%) transfert Op" % (
            cpu, cpu / local_time * 100, gpu, gpu / local_time * 100,
            trans, trans / local_time * 100), file=file)

        print('', file=file)
        print("    Theano function input that are float64", file=file)
        print("    <fct name> <input name> <input type> <str input>", file=file)
        for fg in fgraphs:
            for i in fg.inputs:
                if hasattr(i.type, 'dtype') and i.type.dtype == 'float64':
                    print('        ', fg.name, i.name, i.type, i, file=file)

        print('', file=file)
        print("    List of apply that don't have float64 as input but have float64 in outputs", file=file)
        print("    (Useful to know if we forgot some cast when using floatX=float32 or gpu code)", file=file)
        print('    <Apply> <Apply position> <fct name> <inputs type> <outputs type>', file=file)
        for fg in fgraphs:
            for idx, node in enumerate(fg.toposort()):
                if (any(hasattr(i, 'dtype') and i.dtype == 'float64'
                        for i in node.outputs) and
                    not any(hasattr(i, 'dtype') and i.dtype == 'float64'
                            for i in node.inputs)):

                    print('        ', str(node), idx, fg.name, end=' ',
                          file=file)
                    print(str([getattr(i, 'dtype', None)
                               for i in node.inputs]), end=' ', file=file)
                    print(str([getattr(i, 'dtype', None)
                               for i in node.outputs]), file=file)
        print('', file=file)


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

        return Apply(self, [n, m, k], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i])
                for i in xrange(3)]

    def gpu_kernels(self, node, name):
        code = """#include "cluda.h"

KERNEL void eye(GLOBAL_MEM %(ctype)s *a, ga_size a_off,
                ga_size n, ga_size m, ga_ssize k) {
    a = (GLOBAL_MEM %(ctype)s *)(((GLOBAL_MEM char *)a) + a_off);
    ga_ssize coff = max(k, (ga_ssize) 0);
    ga_ssize roff = -min(k, (ga_ssize) 0);
    ga_size nb = (ga_size) min(n - roff, m - coff);
    for (ga_size i = LID_0; i < nb; i += LDIM_0) {
        a[(i + roff)*m + i + coff] = %(write_a)s(1);
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype),
            name=name, write_a=write_w(self.dtype))
        return [Kernel(
                code=code, name="eye",
                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
                        gpuarray.SIZE, gpuarray.SSIZE],
                flags=Kernel.get_flags(self.dtype),
                objvar='k_eye_' + name)]

    def c_code(self, node, name, inp, out, sub):
        if len(inp) == 2:
            n, m = inp
            k = 0
        elif len(inp) == 3:
            n, m, k = inp

        z, = out
        fail = sub['fail']
        ctx = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        size_t dims[2] = {0, 0};
        size_t ls, gs;
        ssize_t k;
        size_t col_off;
        size_t row_off;
        int err;

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        k = ((dtype_%(k)s*)PyArray_DATA(%(k)s))[0];

        Py_CLEAR(%(z)s);

        %(z)s = pygpu_zeros(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            %(ctx)s, Py_None);
        if (%(z)s == NULL) {
            %(fail)s
        }

        ls = 1;
        gs = 256;
        col_off = (size_t) (k > 0?k:0);
        row_off = (size_t) (k < 0?-k:0);
        if (row_off < dims[0] && col_off < dims[1]) {
            err = eye_call(1, &gs, &ls, 0, %(z)s->ga.data, %(z)s->ga.offset,
                           dims[0], dims[1], k);
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: kEye: %%s. n%%lu, m=%%lu.",
                             GpuKernel_error(&%(kname)s, err),
                             (unsigned long)dims[0], (unsigned long)dims[1]);
                %(fail)s;
            }
        }

        """ % locals()

        return s

    def c_code_cache_version(self):
        return (10,)
