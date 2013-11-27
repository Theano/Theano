import os

import numpy

import theano
from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar, config
from theano.scalar import Scalar
from theano.tensor.basic import Alloc

from theano.gof.python25 import all, any
from theano.gof.utils import MethodNotDefined
from theano.compat import PY3

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from type import GpuArrayType


def as_gpuarray_variable(x):
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

    def c_code_cache_version(self):
        return ()

    def c_code_cache_version_apply(self, node):
        return self.c_code_cache_version()


class GpuKernelBase(object):
    GpuKernelBase_version = 0

    def c_kernel_code(self):
        """
        Return the source code of the kernel.
        """
        raise AttributeError("c_kernel_code", type(self))

    def c_kernel_params(self):
        """
        Return the list of typecodes for kernel parameters.

        The list can contain strings ( "GA_BUFFER" ) or direct int values.
        """
        raise AttributeError("c_kernel_params", type(self))

    def c_kernel_name(self):
        """
        Return the name of the kernel in the source.
        """
        raise AttributeError("c_kernel_name", type(self))

    def c_kernel_flags(self):
        """
        Return a string representing the C flags for the kernel.

        Example:
          "GA_USE_CLUDA|GA_USE_DOUBLE"

        self._get_kernel_flags(*dtypes) returns an appropritate string
        for the result of this function.
        """
        raise AttributeError("c_kernel_flags", type(self))

    def c_kernel_codevar(self):
        return 'kcode_' + type(self).__name__ + '_' + hex(hash(self))[2:]

    def c_kernel_obj(self):
        return 'k_' + type(self).__name__ + '_' + hex(hash(self))[2:]

    def _get_kernel_flags(self, *dtypes):
        dtypes = [numpy.dtype(d) for d in dtypes]
        flags = ['GA_USE_CLUDA']
        if any(d == numpy.float64 for d in dtypes):
            flags.append('GA_USE_DOUBLE')
        if any(d.itemsize < 4 for d in dtypes):
            flags.append('GA_USE_SMALL')
        return '|'.join(flags)

    def c_headers(self):
        return ['compyte/types.h']

    def c_support_code(self):
        kcode = self.c_kernel_code()
        vname = self.c_kernel_codevar()
        kname = self.c_kernel_obj()
        code = '\\n'.join(l for l in kcode.split('\n'))
        return """static const char *%(vname)s = "%(code)s";
static GpuKernel %(kname)s;""" % dict(vname=vname, kname=kname,code=code)

    def c_init_code(self):
        types = self.c_kernel_params()
        numargs = len(types)
        name = self.c_kernel_name()
        vname = self.c_kernel_codevar()
        kname = self.c_kernel_obj()
        flags = self.c_kernel_flags()
        # TODO: find a way to release the kernel once the module is unloaded
        error_out = ""
        if PY3:
            error_out = "NULL"
        return ["""
int types[%(numargs)u] = {%(types)s};
if (GpuKernel_init(&%(kname)s, pygpu_default_context()->ops,
                   pygpu_default_context()->ctx, 1, &%(vname)s, NULL,
                   "%(name)s", %(numargs)s, types, %(flags)s) != GA_NO_ERROR) {
    PyErr_SetString(PyExc_RuntimeError, "Error initializing kernel");
    return %(error_out)s;
}
""" % dict(types=','.join(types), numargs=numargs, kname=kname, name=name,
           vname=vname, flags=flags, error_out=error_out)]


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
                                        broadcastable=x.broadcastable,)()])

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
        if isintance(ev, GpuArrayType):
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
        if isintance(ev, GpuArrayType):
            return [cuda_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp

    def c_headers(self):
        return ['<cuda_ndarray.cuh>', '<compyte/extension.h>',
                '<compyte/types.h>', '<cuda.h>']

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
        return ['cuda_get_ctx = (CUcontext (*)(void *))compyte_get_extension("cuda_get_ctx");',
                'cuda_make_buf = (gpudata *(*)(void *, CUdeviceptr, size_t))compyte_get_extension("cuda_make_buf");']

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
        return ['<cuda_ndarray.cuh>', '<compyte/extension.h>', '<cuda.h>']

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
        return ['cuda_get_ctx = (CUcontext (*)(void *ctx))compyte_get_extension("cuda_get_ctx");',
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");']

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
        return ['<compyte/numpy_compat.h>']

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

    def c_kernel_code(self):
        return """
KERNEL void k(GLOBAL_MEM %(ctype)s *a, ga_size n, ga_size m) {
    ga_size nb = n < m ? n : m;
    for (ga_size i = LID_0; i < nb; i += LDIM_0) {
        a[i*m + i] = 1;
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype))

    def c_kernel_params(self):
        return ["GA_BUFFER", "GA_SIZE", "GA_SIZE"]

    def c_kernel_name(self):
        return "k"

    def c_kernel_flags(self):
        return self._get_kernel_flags(self.dtype)

    def c_code(self, node, name, inp, out, sub):
        n, m = inp
        z, = out
        fail = sub['fail']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        sync = bool(config.gpuarray.sync)
        kname = self.c_kernel_obj()
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
                         "compyte error: kEye: %%s. n%%lu, m=%%lu.",
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
