import os

import numpy

import theano
from theano import Op, Apply
from theano import tensor, scalar, config
from theano.scalar import Scalar
from theano.tensor.basic import Alloc, Join, Split

from theano.gof.python25 import any
from theano.gof.utils import MethodNotDefined
from theano.compat import PY3

try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from type import GpuArrayType


def as_gpuarray_variable(x):
    # This is needed to lower the number of useless transfer
    # introduced during optimization.  This speed up optimization and
    # "canonicalize" the graph, so it make easier making some
    # optimization.
    if (hasattr(x, 'fgraph') and
        len(x.clients) == 1 and
        x.owner and
        isinstance(x.owner.op, HostFromGpu)):
        return x.owner.inputs[0]
    if hasattr(x, '_as_GpuArrayVariable'):
        return x._as_GpuArrayVariable()
    # TODO we need to have the cuda -> gpu path taken care of.
    tensor_x = tensor.as_tensor_variable(x)
    return gpu_from_host(tensor_x)


def as_gpuarray(x):
    return gpuarray.array(x, copy=False)


class HideC(object):
    def __hide(*args):
        raise MethodNotDefined()

    c_code = __hide
    c_code_cleanup = __hide

    c_headers = __hide
    c_header_dirs = __hide
    c_libraries = __hide
    c_lib_dirs = __hide

    c_support_code = __hide
    c_support_code_apply = __hide

    c_compile_args = __hide
    c_no_compile_args = __hide
    c_init_code = __hide
    c_init_code_apply = __hide

    def c_code_cache_version(self):
        return ()

    def c_code_cache_version_apply(self, node):
        return self.c_code_cache_version()


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
            if isinstance(t, (str, unicode)):
                return numpy.dtype(t)
            elif isinstance(t, Type):
                return t.dtype
            elif isinstance(t, Variable):
                return t.type.dtype
            else:
                raise TypeError, "can't get a dtype from %s" % (type(t),)
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
    def gpu_kernels(self, node, name):
        """
        This is the method to override.  This should return an
        iterable of Kernel objects that describe the kernels this op
        will need.
        """
        raise MethodNotDefined, 'gpu_kernels'

    def c_headers(self):
        try:
            o = super(GpuKernelBase, self).c_headers()
        except MethodNotDefined:
            o = []
        return o + ['gpuarray/types.h']

    def _generate_kernel_bin(self, k):
        gk = gpuarray.GpuKernel(k.code, k.name, k.params, **k.flags)
        bin = gk._binary
        bcode = ','.join(hex(ord(c)) for c in bin)
        return ("""static const char %(bname)s[] = { %(bcode)s };""" %
                dict(bname=k.binvar, bcode=bcode))

    def _generate_kernel_code(self, k):
        code = '\\n'.join(l for l in k.code.split('\n'))
        code = code.replace('"', '\\"')
        return ("""static const char *%(cname)s = "%(code)s";""" %
                dict(cname=k.codevar, code=code))

    def _generate_kernel_vars(self, k):
        return """static GpuKernel %(kname)s;""" % dict(kname=k.objvar)

    def c_support_code_apply(self, node, name):
        kernels = self.gpu_kernels(node, name)
        bins = '\n'.join(self._generate_kernel_bin(k) for k in kernels)
        codes = '\n'.join(self._generate_kernel_code(k) for k in kernels)
        vars = '\n'.join(self._generate_kernel_vars(k) for k in kernels)
        return '\n'.join([bins, codes, vars])

    def _generate_kernel_init(self, k, err):
        if PY3:
            error_out = "NULL"
        else:
            error_out = ""
        return """{
  int types[%(numargs)u] = {%(types)s};
  const char *bcode = %(bvar)s;
  size_t sz = sizeof(%(bvar)s);
  PyGpuContextObject *c = pygpu_default_context();
  if (GpuKernel_init(&%(ovar)s, c->ops, c->ctx, 1, &bcode, &sz, "%(kname)s",
                     %(numargs)u, types, GA_USE_BINARY) != GA_NO_ERROR) {
    if ((%(err)s = GpuKernel_init(&%(ovar)s, c->ops, c->ctx, 1, &%(cname)s,
                                  NULL, "%(kname)s", %(numargs)u, types,
                                  %(flags)s)) != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuKernel_init error %%d: %%s",
                   %(err)s, Gpu_error(c->ops, c->ctx, %(err)s));
      return %(error_out)s;
    }
  }
}""" % dict(numargs=len(k.params), types=k._get_c_types(), bvar=k.binvar,
            ovar=k.objvar, kname=k.name, err=err, cname=k.codevar,
            flags=k._get_c_flags(), error_out=error_out)

    def c_init_code_apply(self, node, name):
        err = 'err_' + name
        kernels = self.gpu_kernels(node, name)
        inits ='\n'.join(self._generate_kernel_init(k, err) for k in kernels)
        return ("int %(err)s;\n" % dict(err=err)) + inits

    def _GpuKernelBase_version(self):
        ctx = gpuarray.get_default_context()
        return (2, ctx.kind, ctx.devname)

    GpuKernelBase_version = property(_GpuKernelBase_version)


class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

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
        %(name)serr = GpuArray_read(PyArray_DATA(%(out)s),
                                    PyArray_NBYTES(%(out)s),
                                    %(name)s_ga);
        if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
        if (%(name)serr != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Could not read device data.");
            %(fail)s
        }
        """ % {'name': name, 'fail': sub['fail'], 'inp': inputs[0],
               'out': outputs[0]}

    def c_code_cache_version(self):
        return (1,)

    def grad(self, inputs, grads):
        gz, = grads
        return [gpu_from_host(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, tensor.TensorType):
            return [gpu_from_host(ev)]
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp


host_from_gpu = HostFromGpu()


class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'GpuFromHost(gpuarray)'

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        type = node.outputs[0].type
        z[0] = gpuarray.array(x)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(as_gpuarray_variable(gz))]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, GpuArrayType):
            return [host_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_fromhostdata(PyArray_DATA(%(inp)s),
                                     get_typecode((PyObject *)PyArray_DESCR(%(inp)s)),
                                     PyArray_NDIM(%(inp)s),
                                     (size_t *)PyArray_DIMS(%(inp)s),
                                     (ssize_t *)PyArray_STRIDES(%(inp)s),
                                     pygpu_default_context(),
                                     Py_None);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % {'name': name, 'inp': inputs[0],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (4,)

gpu_from_host = GpuFromHost()


class GpuFromCuda(Op):
    view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'GpuFromCuda'

    def make_node(self, x):
        from theano.sandbox.cuda import CudaNdarrayType
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = gpuarray.array(numpy.asarray(x))

    def grad(self, inputs, grads):
        gz, = grads
        return [cuda_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, GpuArrayType):
            return [cuda_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp

    def c_headers(self):
        return ['<cuda_ndarray.cuh>', '<gpuarray/extension.h>',
                '<gpuarray/types.h>', '<cuda.h>']

    def c_header_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def c_lib_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_libraries(self):
        return ['cudart', 'cublas', 'cuda']

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        gpudata *(*cuda_make_buf)(void *c, CUdeviceptr p, size_t sz);
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *))gpuarray_get_extension("cuda_get_ctx");',
                'cuda_make_buf = (gpudata *(*)(void *, CUdeviceptr, size_t))gpuarray_get_extension("cuda_make_buf");']

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        int %(name)serr;
        gpudata *%(name)sdata;
        CUcontext %(name)scur;
        size_t *%(name)sdims;
        ssize_t *%(name)sstr;

        cuCtxGetCurrent(&%(name)scur);
        if (%(name)scur != cuda_get_ctx(pygpu_default_context()->ctx)) {
            PyErr_SetString(PyExc_ValueError, "Ambient cuda context is not the same as output context.");
            %(fail)s
        }
        %(name)sdims = (size_t *)calloc(%(in)s->nd, sizeof(size_t));
        if (%(name)sdims == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Can't allocate dimensions.");
            %(fail)s
        }
        %(name)sstr = (ssize_t *)calloc(%(in)s->nd, sizeof(ssize_t));
        if (%(name)sstr == NULL) {
            free(%(name)sdims);
            PyErr_SetString(PyExc_MemoryError, "Can't allocate strides.");
            %(fail)s
        }

        for (unsigned int i = 0; i < %(in)s->nd; i++) {
            %(name)sdims[i] = (size_t)CudaNdarray_HOST_DIMS(%(in)s)[i];
            %(name)sstr[i] = (ssize_t)CudaNdarray_HOST_STRIDES(%(in)s)[i]*4;
        }

        %(name)sdata = cuda_make_buf(pygpu_default_context()->ctx,
                                     (CUdeviceptr)%(in)s->devdata,
                                     ((size_t)%(in)s->data_allocated)*4);
        if (%(name)sdata == NULL) {
            Py_DECREF(%(out)s);
            free(%(name)sdims);
            free(%(name)sstr);
            PyErr_SetString(PyExc_MemoryError, "Could not allocate gpudata structure.");
            %(fail)s
        }
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_fromgpudata(%(name)sdata, 0, GA_FLOAT, %(in)s->nd,
                                    %(name)sdims, %(name)sstr,
                                    pygpu_default_context(), 1,
                                    (PyObject *)%(in)s,
                                    (PyObject *)&PyGpuArrayType);
        pygpu_default_context()->ops->buffer_release(%(name)sdata);
        free(%(name)sdims);
        free(%(name)sstr);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % {'name': name, 'in': inputs[0], 'out': outputs[0],
               'fail': sub['fail']}

    def c_code_cache_version(self):
        return (5,)

gpu_from_cuda = GpuFromCuda()


class CudaFromGpu(Op):
    view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'CudaFromGpu'

    def make_node(self, x):
        from theano.sandbox.cuda import CudaNdarrayType
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        if x.type.dtype != 'float32':
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        from theano.sandbox.cuda import filter as cuda_filter
        x, = inp
        z, = out
        z[0] = cuda_filter(theano._asarray(x, dtype='float32'),
                           tuple([0] * x.ndim), 0, z[0])

    def grad(self, inputs, grads):
        gz, = grads
        return [gpu_from_cuda(gz)]

    def R_op(self, inputs, eval_points):
        from theano.sandbox.cuda import CudaNdArrayType
        ev, = eval_points
        if (isinstance(ev, CudaNdarrayType)):
            return [gpu_from_cuda(ev)]
        else:
            return [ev]

    def infer_shape(self, node, shp):
        return shp

    def c_headers(self):
        return ['<cuda_ndarray.cuh>', '<gpuarray/extension.h>', '<cuda.h>']

    def c_header_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def c_lib_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_libraries(self):
        return ['cudart', 'cublas', 'cuda']

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *ctx))gpuarray_get_extension("cuda_get_ctx");',
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))gpuarray_get_extension("cuda_get_ptr");']

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        int %(name)serr = 0, %(name)si;
        CUcontext %(name)scur;

        cuCtxGetCurrent(&%(name)scur);
        if (%(name)scur != cuda_get_ctx(pygpu_default_context()->ctx)) {
            PyErr_SetString(PyExc_ValueError, "Ambient cuda context is not the same as output context.");
            %(fail)s
        }

        if (GpuArray_sync(&%(inp)s->ga) != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Could not sync GpuArray");
            %(fail)s
        }
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray *)CudaNdarray_new_nd(%(inp)s->ga.nd);
        if (!%(out)s) {
            %(fail)s
        }
        for (%(name)si = 0; %(name)si < %(inp)s->ga.nd; %(name)si++) {
            CudaNdarray_set_dim(%(out)s, %(name)si, %(inp)s->ga.dimensions[%(name)si]);
            CudaNdarray_set_stride(%(out)s, %(name)si, %(inp)s->ga.strides[%(name)si]/4);
        }
        %(name)serr = CudaNdarray_set_device_data(%(out)s,
          (float *)(((char *)cuda_get_ptr(%(inp)s->ga.data))+%(inp)s->ga.offset),
                                          (PyObject *)%(inp)s);
        if (%(name)serr) {
           %(fail)s
        }
        """ % {'name': name, 'inp': inputs[0], 'out': outputs[0],
               'fail': sub['fail']}

    def c_code_cache_version(self):
        return (3,)


cuda_from_gpu = CudaFromGpu()


class GpuAlloc(HideC, Alloc):
    def __init__(self, memset_0=False):
        """memset_0 is only an optimized version. True, it mean the
        value is always 0, so the c code call memset as it is faster.

        """
        self.memset_0 = memset_0

    def __eq__(self, other):
        return type(self) == type(other) and self.memset_0 == other.memset_0

    def __hash__(self):
        return hash(type(self)) ^ hash(self.memset_0)

    def __str__(self):
        #Hide the memset parameter when not used to prevent confusion.
        if self.memset_0:
            s = "%s{memset_0=%s}" % (self.__class__.__name__, self.memset_0)
        else:
            s = self.__class__.__name__
        return s

    def make_node(self, value, *shape):
        res = Alloc.make_node(self, value, *shape)
        value = as_gpuarray_variable(value)
        otype = GpuArrayType(dtype=res.outputs[0].dtype,
                             broadcastable=res.outputs[0].broadcastable)
        return Apply(self, [value] + res.inputs[1:], [otype()])

    def c_headers(self):
        return ['<numpy_compat.h>']

    def perform(self, node, inputs, outs):
        out, = outs
        v = inputs[0]
        sh = tuple(map(int, inputs[1:]))
        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and numpy.asarray(v)[0].item() == 0:
                out[0] = gpuarray.zeros(sh, dtype=v.dtype)
            else:
                out[0] = gpuarray.empty(sh, dtype=v.dtype)
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
                                 pygpu_default_context(), Py_None);
            if (!%(zz)s) {
                %(fail)s
            }
        } else {
            if (need_new_out) {
                Py_XDECREF(%(zz)s);
                %(zz)s = pygpu_empty(%(ndim)s, %(name)s_shape,
                                     %(vv)s->ga.typecode, GA_C_ORDER,
                                     pygpu_default_context(), Py_None);
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
                                 "GpuAlloc: Error memsetting %%d"
                                 " element of device memory to 0.",
                                 PyGpuArray_SIZE(%(zz)s));
                    %(fail)s;
                }
            }
            else if (GpuArray_setarray(&%(zz)s->ga, &%(vv)s->ga) !=
                     GA_NO_ERROR) {
                PyErr_SetString(PyExc_ValueError, "setarray failed");
                %(fail)s
            }
        }
        """ % dict(name=name, ndim=ndim, zz=zz, vv=vv,
                   fail=sub['fail'], memset_0=memset_0)

        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(zz)s->ga);" % dict(zz=zz)

        return code

    def c_code_cache_version(self):
        return (2,)

    def do_constant_folding(self, node):
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (#The following ops work inplace of their input id 0.
                  client[1] == 0 and
                  isinstance(client[0].op, (
                    #Ops that will work inplace on the Alloc. So if they
                    #get constant_folded, they would copy the
                    #constant and this is less efficients.

                    #Not doing the constant folding could also lower
                    #the peak memory usage, as we the "constant" won't
                    #always exists.
                      #theano.tensor.subtensor.AdvancedIncSubtensor,
                      theano.sandbox.gpuarray.subtensor.GpuIncSubtensor,
                      theano.sandbox.gpuarray.subtensor.GpuAdvancedIncSubtensor1,
                      theano.sandbox.gpuarray.subtensor.GpuAdvancedIncSubtensor1_dev20,
                      theano.sandbox.gpuarray.blas.GpuGemm,
                      theano.sandbox.gpuarray.blas.GpuGemv,
                      theano.sandbox.gpuarray.blas.GpuGer,
                  ))):
                return False
            #If the clients is a transfer, we don't want to fold. We
            #let the moving opt finish before deciding what to do.
            elif isinstance(client[0].op, HostFromGpu):
                return False
        return True

gpu_alloc = GpuAlloc()


class GpuReshape(HideC, tensor.Reshape):
    """
    Implement Reshape on the gpu.
    """
    # __hash__, __eq__, __str__ come from tensor.Reshape
    def make_node(self, x, shp):
        x = as_gpuarray_variable(x)
        res = host_from_gpu(x).reshape(shp, ndim=self.ndim)
        otype = GpuArrayType(dtype=res.dtype,
                             broadcastable=res.broadcastable)
        return Apply(self, [x, shp], [otype()])

    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to GpuReshape.perform'
                             ' has incorrect length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)
        s = shp.prod()

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


class GpuJoin(HideC, Join):
    def make_node(self, axis, *tensors):
        node = Join.make_node(self, axis, *tensors)

        return Apply(self, [node.inputs[0]] + map(as_gpuarray_variable,
                                                  tensors),
                     [GpuArrayType(broadcastable=node.outputs[0].broadcastable,
                                   dtype=node.outputs[0].dtype)()])

    def perform(self, node, axis_and_tensors, out_):
        out, = out_
        axis = int(axis_and_tensors[0])
        tensors = axis_and_tensors[1:]
        out[0] = pygpu.concatenate(tensors, axis=axis).astype(
            node.outputs[0].dtype)

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, out_, sub):
        copy_to_list = []
        restype=pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        for i, inp in enumerate(inputs[1:]):
            copy_to_list.append("als[%s] = &%s->ga;" % (i, inp))
        return """
GpuArray **als = (GpuArray **)PyMem_Malloc(sizeof(GpuArray *) * %(n)s);
if (als == NULL) {
  PyErr_NoMemory();
  %(fail)s
}
%(copy_inputs_to_list)s
Py_XDECREF(%(out)s);
%(out)s = pygpu_concatenate(als, %(n)s, PyInt_AsLong((PyObject *)%(axis)s),
                            %(restype)s, (PyObject *)&PyGpuArrayType,
                            pygpu_default_context());
PyMem_Free(als);
if (%(out)s == NULL)
  %(fail)s
        """ % dict(n=len(inputs[1:]), fail=sub['fail'], out=out_[0],
                   axis=inputs[0], copy_inputs_to_list='\n'.join(copy_to_list),
                   restype=restype)


gpu_join = GpuJoin()


class GpuSplit(HideC, Split):
    def make_node(self, x, axis, splits):
        node = Split.make_node(self, x, axis, splits)
        x = as_gpuarray_variable(x)
        outs = [GpuArrayType(dtype=o.dtype, broadcastable=o.broadcastable)()
                for o in node.outputs]
        return Apply(self, [x] + node.inputs[1:], outs)
    # we reuse the perform of the CPU op, which is suitable


class GpuEye(GpuKernelBase, Op):
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, n, m, k):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        k = tensor.as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        otype = GpuArrayType(dtype=self.dtype,
                             broadcastable=(False, False))

        # k != 0 isn't implemented on the GPU yet.
        assert tensor.get_scalar_constant_value(k) == 0
        return Apply(self, [n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ hash(type(self))

    def gpu_kernels(self, node, name):
        code = """
KERNEL void k(GLOBAL_MEM %(ctype)s *a, ga_size n, ga_size m) {
    ga_size nb = n < m ? n : m;
    for (ga_size i = LID_0; i < nb; i += LDIM_0) {
        a[i*m + i] = 1;
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype), name=name)
        return [Kernel(
                code=code, name="k",
                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(self.dtype),
                objvar='k_eye_'+name,
                )]

    def c_code(self, node, name, inp, out, sub):
        n, m = inp
        z, = out
        fail = sub['fail']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        sync = bool(config.gpuarray.sync)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        size_t dims[2] = {0, 0};
        void *args[3];
        int err;

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        Py_CLEAR(%(z)s);

        %(z)s = pygpu_zeros(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            pygpu_default_context(), Py_None);
        if (%(z)s == NULL) {
            %(fail)s
        }

        args[0] = &%(z)s->ga;
        args[1] = &dims[0];
        args[2] = &dims[1];
        err = GpuKernel_call(&%(kname)s, 0, 1, 256, args);
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
        return (3, self.GpuKernelBase_version)
