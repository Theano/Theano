import numpy

import theano
from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar, config
from theano.scalar import Scalar

from theano.gof.python25 import all, any

import pygpu
from pygpu import gpuarray, elemwise

from type import GpuArrayType

def as_gpuarray_variable(x):
    if hasattr(x, '_as_GpuArrayVariable'):
        return x._as_GpuArrayVariable()
    # TODO we need to have the cuda -> gpu path taken care of.
    tensor_x = tensor.as_tensor_variable(x)
    return gpu_from_host(tensor_x)


def as_gpuarray(x):
    return gpuarray.array(x, copy=False)


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
        %(name)s_dtype = typecode_to_dtype(%(inp)s->ga.typecode);
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
        type = node.outputs[0].type
        return """
        PyArrayObject *%(name)s_tmp;
        int %(name)serr;
        if ((PyObject *)GpuArray_default_context == Py_None) {
            PyErr_SetString(PyExc_ValueError, "No default context, gpuarray not initialized?");
            %(fail)s
        }
        %(name)s_tmp = PyArray_GETCONTIGUOUS(%(inp)s);
        if (%(name)s_tmp == NULL) {
            // PyArray_GETCONTIGUOUS sets an error message if it fails
            %(fail)s
        }
        %(out)s = new_GpuArray((PyObject *)&GpuArrayType, GpuArray_default_context);
        if (%(out)s == NULL) {
            Py_DECREF(%(name)s_tmp);
            // new_GpuArray calls __new__ which will set an error message
            // if it returns NULL.
            %(fail)s
        }
        %(name)serr = GpuArray_empty(&%(out)s->ga,
                                     GpuArray_default_context->ops,
                                     GpuArray_default_context->ctx,
                                     %(typecode)s,
                                     PyArray_NDIM(%(inp)s),
                                     (size_t *)PyArray_DIMS(%(inp)s),
                                     GA_C_ORDER);
        if (%(name)serr != GA_NO_ERROR) {
            Py_DECREF(%(name)s_tmp);
            Py_DECREF(%(out)s);
            %(out)s = NULL;
            PyErr_SetString(PyExc_MemoryError, "Can't allocate device memory for result.");
            %(fail)s
        }
        %(name)serr = GpuArray_write(&%(out)s->ga, PyArray_DATA(%(name)s_tmp),
                                     PyArray_NBYTES(%(name)s_tmp));
        Py_DECREF(%(name)s_tmp);
        if (%(name)serr != GA_NO_ERROR) {
            Py_DECREF(%(out)s);
            PyErr_SetString(PyExc_RuntimeError, "Could not copy array data to device");
            %(fail)s
        }
        """ % {'name': name, 'inp': inputs[0],
               'out': outputs[0], 'fail': sub['fail'],
               'typecode': type.typecode}

    def c_code_cache_version(self):
        return (0,)

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
        from theano.sandbox.cuda import CudaNdArrayType
        if not isinstance(x.type, CudaNdArrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)]())

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
        return ['cuda_ndarray.cuh', 'compyte/extension.h', 'cuda.h']

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

    def c_conpiler(self):
        from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
        return NVCC_compiler

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx) = compyte_get_extension('cuda_get_ctx');
        gpudata *(*cuda_make_buf)(void *c, CUdeviceptr p, size_t sz) = compyte_get_extension('cuda_make_buf');
        """

    def c_code(self, node, name, input, output, sub):
        return """
        int %(name)serr;
        gpudata *%(name)sdata;
        CUcontext *%(name)scur;
        size_t *%(name)sdims;
        ssize_t *%(name)sstr;

        cuCtxGetCurrent(&%(name)scur);
        if (%(name)scur != cuda_get_ctx(GpuArray_default_context->ctx)) {
            PyErr_SetString(PyErr_ValueError, "Ambient cuda context is not the same as output context.");
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
            %(name)sdims[i] = (size_t)CudaNdArray_HOST_DIMS(%(in)s)[i];
            %(name)sstr[i] = (ssize_t)CudaNdArray_HOST_STRIDES(%(in)s)[i];
        }

        Py_XDECREF(%(out)s);
        %(out)s = new_GpuArray((PyObject *)&GpuArrayType);
        if (%(out)s == NULL) {
            free(%(name)sdims);
            free(%(name)sstr);
            %(fail)s
        }

        %(name)sdata = cuda_make_buf(GpuArray_default_context->ctx,
                                     (CUdeviceptr)%(in)s->devdata,
                                     (size_t)%(in)s->data_allocated);
        if (%(name)sdata == NULL) {
            Py_DECREF(%(out)s);
            free(%(name)sdims);
            free(%(name)sstr);
            PyErr_SetString(PyExc_MemoryError, "Could not allocate gpudata structure.");
            %(fail)s
        }
        %(name)serr = GpuArray_fromdata(&%(out)s->ga,
                                        GpuArray_default_context->ops,
                                        %(name)sdata, 0, GA_FLOAT, %(in)s->nd,
                                        %(name)sdims, %(name)sstr, 1);
        free(%(name)sdims);
        free(%(name)sstr);
        if (%(name)serr != GA_NO_ERROR) {
            Py_DECREF(%(out)s);
            PyErr_SetString(PyExc_MemoryError, "Could not allocate GpuArray structure.");
            %(fail)s
        }
        Py_INCREF(%(in)s);
        %(out)s->base = %(in)s;
        """ % {'name':name, 'in': inputs[0], 'out': outputs[0],
               'fail': sub['fail']}

    def c_code_cache_version(self):
        return (0,)

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
        from theano.sandbox.cuda import CudaNdArrayType
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable,
                                                 dtype=x.dtype)()])

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
        return ['cuda_ndarray.cuh', 'compyte/extension.h', 'cuda.h']

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

    def c_conpiler(self):
        from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
        return NVCC_compiler

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx) = compyte_get_extension('cuda_get_ctx');
        CUdeviceptr (*cuda_get_ptr)(gpudata *g) = compyte_get_extension('cuda_get_ptr');
        size_t (*cuda_get_sz)(gpudata *g) = compyte_get_extension('cuda_get_sz');
        """

    def c_code(self, node, name, input, output, sub):
        return """
        int err = 0, i;
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray *)CudaNdarray_new_nd(%(inp)s->nd);
        if (!%(out)s) {
            %(fail)s
        }
        for (i = 0; i < %(inp)s->nd; i++) {
            CudaNdarray_set_dim(%(out)s, i, %(inp)s->dimensions[i]);
            CudaNdarray_set_stride(%(out)s, i, %(inp)s->strides[i]);
        }
        err = CudaNdarray_set_device_data(%(out),
          (float *)(((char *)cuda_get_ptr(%(inp)s.ga->data))+%(inp).ga.offset),
                                          (PyObject *)%(inp)s);
        if (err) {
           %(fail)s
        }
        """ % {'inp': inputs[0], 'out': output[0], 'fail': sub['fail']}


cuda_from_gpu = CudaFromGpu()


class GpuAlloc(Op):

    def __str__(self):
        return 'GpuAlloc'

    def make_node(self, value, *shape):
        v = as_gpuarray_variable(value)
        sh = [tensor.as_tensor_variable(s) for s in shape]
        if v.ndim != len(shape):
            raise TypeError(
                'GpuAlloc requires value of same dimensions as shape',
                value, len(shape))
        bcast = []
        for s in sh:
            if s.type.dtype[:3] not in ('int', 'uint'):
                raise TypeError('Shape arguments must be integers', s)
            try:
                const_shp = tensor.get_constant_value(s)
            except TypeError:
                const_shp = None
            bcast.append(numpy.all(1 == const_shp))
        otype = GpuArrayType(dtype=v.dtype, broadcastable=bcast)
        return Apply(self, [v] + sh, [otype()])

    def perform(self, node, inputs, outs):
        out, = outs
        v = inputs[0]
        sh = tuple(map(int, inputs[1:]))
        if out[0] is None or out[0].shape != sh:
            out[0] = gpuarray.empty(sh, dtype=v.dtype)
        out[0][...] = v

    def infer_shape(self, node, input_shapes):
        return [node.inputs[1:]]

    def grad(self, input, grads):
        return [None for i in inputs]

gpu_alloc = GpuAlloc()
