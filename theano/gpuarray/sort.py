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
        return ['<blas_api.h>', '<numpy_compat.h>', '<gpuarray_helper.h>', '<gpuarray/sort.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), os.path.dirname(__file__)]

    def c_init_code(self):
        return ['import_pygpu__blas();']

class SortOp(SortGenOp):
    __props__ = ()

    def make_node(self, x):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, ctx_name)

        assert x.ndim == 1

        return Apply(self, [x], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], inp=inp[0], fail=sub['fail'], name=name)

        code = """
               int err;
               theano_prep_output(&%(out)s, %(inp)s->ga.nd, %(inp)s->ga.dimensions,%(inp)s->ga.typecode, GA_C_ORDER, %(inp)s->context);
               err = GpuArray_sort(&%(out)s->ga, &%(inp)s->ga, 1, NULL); 
               """ % vars
        return code

# TODO add sort dir to params
sortOp = SortOp()