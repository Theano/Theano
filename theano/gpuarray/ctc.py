from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op
from theano import config
import theano.tensor as T
from theano.tensor.extra_ops import cpu_contiguous
from .basic_ops import (gpu_contiguous, as_gpuarray_variable,
                        infer_context_name, CGpuKernelBase)
import theano.tensor.nnet.ctc
from .type import GpuArrayType
from .opt import register_opt, op_lifter, register_opt2
from theano.gradient import grad_undefined
from theano import gof
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize

import os
import pygpu

ctc_enabled = config.ctc.enabled


class GpuConnectionistTemporalClassification(CGpuKernelBase, Op):
    """
    GPU wrapper for Baidu CTC loss function.
    """
    __props__ = ('context_name', 'compute_grad',)

    func_file = "./ctc_wrapper.c"
    func_name = "APPLY_SPECIFIC(ctc_cost_gpu)"

    def __init__(self, compute_grad=True, context_name=None):
        if not compute_grad:
            self.func_name = "APPLY_SPECIFIC(ctc_cost_gpu_no_grad)"
        self.compute_grad = compute_grad
        self.context_name = context_name

        Op.__init__(self)
        CGpuKernelBase.__init__(self, self.func_file, self.func_name)

        self.costs = GpuArrayType(dtype='float32',
                                  broadcastable=(False,),
                                  context_name=self.context_name)

        if self.compute_grad:
            self.gradients = GpuArrayType(dtype='float32',
                                          broadcastable=(False, False, False,),
                                          context_name=self.context_name)

        if config.ctc.root == "":
            raise ValueError('ctc.root variable is not set, please set it '
                             'to the root directory of the CTC library in '
                             'your system.')

    def c_lib_dirs(self):
        dirs = []
        if ctc_enabled:
            # We assume here that the compiled library (libwarpctc.so) is available
            # at the build directory of the CTC root directory.
            dirs.append(os.path.join(config.ctc.root, "build"))
        return dirs

    def c_libraries(self):
        return ["warpctc", "gpuarray"]

    def c_header_dirs(self):
        dirs = [os.path.dirname(__file__), pygpu.get_include()]
        if ctc_enabled:
            # We assume here that the header is available at the include directory
            # of the CTC root directory.
            dirs.append(os.path.join(config.ctc.root, "include"))
        dirs.append('/usr/local/cuda/include')
        return dirs

    def c_headers(self):
        return ['ctc.h', 'numpy_compat.h', 'gpuarray_helper.h', 'gpuarray/types.h',
            'gpuarray_api.h', 'gpuarray/array.h', 'gpuarray/util.h', '<cuda_runtime.h>']

    def make_node(self, activations, labels, input_lengths):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'GpuConnectionistTemporalClassification Op '
                               'can not be constructed.')

        context = infer_context_name(activations, labels, input_lengths)
        assert context == self.context_name

        t_activations = as_gpuarray_variable(activations,
                                             context_name=self.context_name)
        # Ensure activations array is C-contiguous
        t_activations = gpu_contiguous(t_activations)

        # Labels and input lengths are always on the CPU
        t_labels = T.as_tensor_variable(labels)
        t_input_lengths = T.as_tensor_variable(input_lengths)

        if t_activations.type.dtype != 'float32':
            raise TypeError('Activations must use the float32 type!')

        if t_labels.type.dtype != 'int32':
            raise TypeError('Labels must use the int32 type!')

        if t_input_lengths.type.dtype != 'int32':
            raise TypeError('Label lengths must use the int32 type!')

        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0

        out_params = [as_gpuarray_variable(self.costs(), context_name=self.context_name)]
        if self.gradients is not None:
            out_params.append(as_gpuarray_variable(self.gradients(),
                              context_name=self.context_name))

        return theano.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                            outputs=out_params)

    def grad(self, inputs, grads):
        return [as_gpuarray_variable(self.gradients(), context_name=self.context_name),
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2])]

def ctc(activations, labels, input_lengths):
    return GpuConnectionistTemporalClassification()(activations, labels,
                                                    input_lengths)

# Disable gradient computation if not needed
@register_canonicalize
@register_stabilize
@local_optimizer([GpuConnectionistTemporalClassification])
def local_GpuConnectionistTemporalClassification_no_grad(node):
    if isinstance(node.op, GpuConnectionistTemporalClassification):
        if len(node.outputs) > 1:
            if len(node.outputs[1].clients) == 0:   # gradient is not used
                node.op = GpuConnectionistTemporalClassification(compute_grad=False)
                node.outputs = node.outputs[:1]   # costs only