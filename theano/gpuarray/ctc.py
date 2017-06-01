from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op
from theano import config
import theano.tensor as T
from .basic_ops import (gpu_contiguous, as_gpuarray_variable,
                        infer_context_name, CGpuKernelBase)
import theano.tensor.nnet.ctc
from .type import GpuArrayType
from .opt import register_opt, op_lifter, register_opt2
from theano.gradient import grad_undefined

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

        CGpuKernelBase.__init__(self, self.func_file, self.func_name)

        self.costs_type = GpuArrayType(dtype='float32',
                                       broadcastable=(False,),
                                       context_name=self.context_name)

        if self.compute_grad:
            self.grads_type = GpuArrayType(dtype='float32',
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
        return ["warpctc"]

    def c_header_dirs(self):
        dirs = []
        if ctc_enabled:
            # We assume here that the header is available at the include directory
            # of the CTC root directory.
            dirs.append(os.path.join(config.ctc.root, "include"))
        return dirs + CGpuKernelBase.c_header_dirs(self)

    def c_headers(self):
        return ["ctc.h"] + CGpuKernelBase.c_headers(self)

    def make_node(self, activations, labels, input_lengths):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'ConnectionistTemporalClassification Op '
                               'can not be constructed.')

        context = infer_context_name(activations, labels, input_lengths)
        assert context == self.context_name

        t_activations = as_gpuarray_variable(activations,
                                             context_name=self.context_name)
        # Ensure activations array is C-contiguous
        t_activations = gpu_contiguous(t_activations)

        t_labels = as_gpuarray_variable(labels, context_name=self.context_name)
        t_input_lengths = as_gpuarray_variable(input_lengths,
                                               context_name=self.context_name)

        if t_activations.type.dtype != 'float32':
            raise TypeError('Activations must use the float32 type!')

        if t_labels.type.dtype != 'int32':
            raise TypeError('Labels must use the int32 type!')

        if t_input_lengths.type.dtype != 'int32':
            raise TypeError('Label lengths must use the int32 type!')

        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0

        out_params = [self.costs_type()]
        if self.grads_type is not None:
            out_params.append(self.grads_type())

        return theano.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                            outputs=out_params)

    def grad(self, inputs, output_grads):
        return [grad_undefined(self, 0, inputs[0]),
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2])]

def ctc(activations, labels, input_lengths):
    return GpuConnectionistTemporalClassification()(activations, labels,
                                                    input_lengths)