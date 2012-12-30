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


def as_gpuarray(x, kind, context):
    return gpuarray.array(x, kind=kind, context=context, copy=False)


class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

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
        GpuArray *%(name)s_ga;
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
            PyErr_SetSring(PyExc_RuntimeError, "Could not read device data.");
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

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        type = node.outputs[0].type
        z[0] = gpuarray.array(x, kind=type.kind, context=type.context)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

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
        %(name)s_tmp = PyArray_GETCONTIGUOUS(%(inp)s);
        if (%(name)s_tmp == NULL) {
            %(fail)s
        }
        %(out)s = new_GpuArray((PyObject *)&GpuArrayType);
        if (%(out)s == NULL) {
            Py_DECREF(%(name)s_tmp);
            %(fail)s
        }
        %(name)serr = GpuArray_empty(&%(out)s->ga, compyte_get_ops("%(kind)s"),
                                     (void *)%(ctx)s, %(typecode)s,
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
        """ % {'name': name, 'kind': type.kind, 'ctx': hex(type.context),
               'inp': inputs[0], 'out': outputs[0], 'fail': sub['fail'],
               'typecode': type.typecode}
    # Don't implement c_code_cache_version since we harcode the ctx address
    # in the code block and this will not work across processes


gpu_from_host = GpuFromHost()


class GpuFromCuda(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        if not isinstance(x.type, CudaNdArrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype)]())

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if globals.kind == 'cuda':
            base = x
            while hasattr(base, 'base') and base.base is not None:
                base = base.base
            # TODO: I know how to do this in C, but I don't know about python.
            #       Is perform() actually required to work?
            raise NotImplementedError("How are we going to get a gpudata pointer from here")
            x[0] = gpuarray.from_gpudata(b, 0, x.dtype, x.shape,
                                         base=base, kind=globals.kind,
                                         context=globals.context,
                                         strides=x.strides)
        else:
            z[0] = gpuarray.array(numpy.asarray(x), kind=globals.kind,
                                  context=globals.context)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isintance(ev, GpuArrayType):
            return [host_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp
