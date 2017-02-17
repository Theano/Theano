from __future__ import division, absolute_import, print_function
import numpy as np
from six.moves import xrange

import theano
from theano import tensor, config, Apply, Op
from theano.gradient import grad_undefined

from .config import mode_with_gpu, test_ctx_name

from ..basic_ops import CGpuKernelBase
from ..type import GpuArrayType, get_context


from pygpu.gpuarray import dtype_to_typecode


# This is an implementation to test that CGpuKernelBase works and also
# to use as an example in the docs.  It is not used for user graphs.
class GpuEye(CGpuKernelBase, Op):
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
        CGpuKernelBase.__init__(self, ['tstgpueye.c'],
                                'APPLY_SPECIFIC(tstgpueye)')

    def get_params(self, node):
        return get_context(self.context_name)

    def c_headers(self):
        return ['<gpuarray/types.h>', '<gpuarray/kernel.h>']

    def make_node(self, n, m):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        assert n.ndim == 0
        assert m.ndim == 0
        otype = GpuArrayType(dtype=self.dtype,
                             broadcastable=(False, False),
                             context_name=self.context_name)

        return Apply(self, [n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i])
                for i in xrange(2)]

    def get_op_params(self):
        return [('TYPECODE', str(dtype_to_typecode(self.dtype)))]


def test_cgpukernelbase():
    op = GpuEye(dtype='int32', context_name=test_ctx_name)

    f = theano.function([], op(4, 5), mode=mode_with_gpu)

    r = f()

    assert (np.asarray(r) == np.eye(4, 5, dtype='int32')).all()
