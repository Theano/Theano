from __future__ import absolute_import, print_function, division
import os.path
from six import integer_types

import theano
from theano import Apply, config, Op

from theano.compile import optdb
from theano.gof import LocalOptGroup, ParamsType
from theano.scalar import bool as bool_t
from theano.tensor.basic import as_tensor_variable
from theano.tensor.opt import in2out

from .basic_ops import (GpuArrayType, CGpuKernelBase,
                        as_gpuarray_variable, gpu_contiguous, infer_context_name)
from .opt_util import inplace_allocempty

try:
    import pygpu
    from pygpu import blas
except ImportError as e:
    # To make sure theano is importable
    pass


class SortGenOp(Op):
    def c_headers(self):
        return ['<blas_api.h>', '<numpy_compat.h>', '<gpuarray_helper.h>',
                '<gpuarray/sort.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), os.path.dirname(__file__)]

    def c_init_code(self):
        return ['import_pygpu__blas();']

    def valid_input_type(self, in_type):
        supported_types = ('uint32', 'int32', 'float32', 'float64', 'uint8',
                           'int8', 'unit16', 'int16')
        return in_type in supported_types

class SortOp(SortGenOp):

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
                                   "Input must be contiguous");
                   %(fail)s   
               }
               theano_prep_output(&%(out)s, %(inp)s->ga.nd,
                                 %(inp)s->ga.dimensions, %(inp)s->ga.typecode,
                                 GA_C_ORDER, %(inp)s->context);
               err = GpuArray_sort(&%(out)s->ga, &%(inp)s->ga, 1, NULL);
               if (err != GA_NO_ERROR) {
                   PyErr_SetString(PyExc_RuntimeError,"GpuArray_sort failed");
                   %(fail)s
               }
                """ % vars

        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars

        return code

# TODO add sort dir to params
sort_ascending = SortOp()
sort_descending = SortOp()


class ArgSortOp(SortGenOp):

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
                                   "Input must be contiguous");
                   %(fail)s   
               }
               theano_prep_output(&%(out)s, %(inp)s->ga.nd,
                                  %(inp)s->ga.dimensions, GA_ULONG,GA_C_ORDER,
                                  %(inp)s->context);
               GpuArray dst;
               GpuArray_empty(&dst, GpuArray_context(&%(inp)s->ga),
                              %(inp)s->ga.typecode, %(inp)s->ga.nd,
                              %(inp)s->ga.dimensions, GA_C_ORDER);
               err = GpuArray_sort(&dst, &%(inp)s->ga, 1, &%(out)s->ga);
               if (err != GA_NO_ERROR) {
                   PyErr_SetString(PyExc_RuntimeError, "Memset failed");
                   %(fail)s
               }
               GpuArray_clear(&dst);
               """ % vars

        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars

        return code

# TODO add sort dir to params
argsort_ascending = ArgSortOp()
argsort_descending = ArgSortOp()