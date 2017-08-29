from __future__ import absolute_import, print_function, division
import os.path

import theano
from theano import Apply, Op

from theano.gof import LocalOptGroup
from .basic_ops import (GpuArrayType, 
                        as_gpuarray_variable, gpu_contiguous, infer_context_name)

try:
    import pygpu
except ImportError as e:
    # To make sure theano is importable
    pass


class SortGenOp(Op):

    def c_headers(self):
        return ['<blas_api.h>', '<numpy_compat.h>',
                '<c_code/gpuarray_helper.h>', '<gpuarray/sort.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), os.path.dirname(__file__)]

    def valid_input_type(self, in_type):
        supported_types = ('uint32', 'int32', 'float32', 'float64', 'uint8',
                           'int8', 'unit16', 'int16')
        return in_type in supported_types


class SortOp(SortGenOp):
    """
    Sort on the GPU.
    
    """
    def make_node(self, x):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, ctx_name)

        assert x.ndim == 1
        assert SortGenOp.valid_input_type(self, x.dtype) == True

        return Apply(self, [x], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], inp=inp[0], fail=sub['fail'], name=name)

        code = """
               int err = GA_NO_ERROR;
               if (!GpuArray_ISONESEGMENT(&%(inp)s->ga)) {
                   PyErr_SetString(PyExc_RuntimeError,
                                   "Input must be one segment");
                   %(fail)s
               }
               if (theano_prep_output(&%(out)s, %(inp)s->ga.nd,
                                 %(inp)s->ga.dimensions, %(inp)s->ga.typecode,
                                 GA_C_ORDER, %(inp)s->context) ) {
                   %(fail)s
               }
               err = GpuArray_sort(&%(out)s->ga, &%(inp)s->ga, 1, NULL);
               if (err != GA_NO_ERROR) {
                   PyErr_SetString(PyExc_RuntimeError,"GpuArray_sort failed");
                   %(fail)s
               }
                """ % vars

        return code


sort_gpu = SortOp()


class ArgSortOp(SortGenOp):
    """
    Argsort on the GPU.
    
    """
    def make_node(self, x):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, ctx_name)
        bcast = x.type.broadcastable

        assert x.ndim == 1
        assert SortGenOp.valid_input_type(self, x.dtype) == True

        return Apply(self, [x], [GpuArrayType(dtype='uint64', context_name=ctx_name,
                                              broadcastable=bcast)()])

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], inp=inp[0], fail=sub['fail'], name=name)

        code = """
               int err = GA_NO_ERROR;
               if (!GpuArray_ISONESEGMENT(&%(inp)s->ga)) {
                   PyErr_SetString(PyExc_RuntimeError,
                                   "Input must be one segment");
                   %(fail)s
               }
               if (theano_prep_output(&%(out)s, PyGpuArray_NDIM(%(inp)s),
                                  PyGpuArray_DIMS(%(inp)s), GA_ULONG,GA_C_ORDER,
                                  %(inp)s->context) ) {
                   %(fail)s
               }
               GpuArray dst;
               GpuArray_empty(&dst, GpuArray_context(&%(inp)s->ga),
                              %(inp)s->ga.typecode, %(inp)s->ga.nd,
                              %(inp)s->ga.dimensions, GA_C_ORDER);
               err = GpuArray_sort(&dst, &%(inp)s->ga, 1, &%(out)s->ga);
               GpuArray_clear(&dst);
               if (err != GA_NO_ERROR) {
                   PyErr_SetString(PyExc_RuntimeError, "GpuArray_sort failed");
                   %(fail)s
               }
               """ % vars

        return code


argsort_gpu = ArgSortOp()

