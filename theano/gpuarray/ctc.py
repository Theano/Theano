from __future__ import absolute_import, print_function, division

import theano
from theano import Op
from theano import config
import theano.tensor as T
from .basic_ops import (gpu_contiguous, as_gpuarray_variable,
                        infer_context_name, CGpuKernelBase)
import theano.tensor.nnet.ctc
from .type import GpuArrayType
from .elemwise import GpuDimShuffle
from theano.gradient import grad_undefined
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
        return dirs

    def c_headers(self):
        return ['ctc.h', 'numpy_compat.h', 'gpuarray_helper.h', 'gpuarray/types.h',
                'gpuarray_api.h', 'gpuarray/array.h', 'gpuarray/util.h']

    def make_node(self, activations, labels, input_lengths):
        """
        Parameters
        ----------
        activations
            Three-dimensional tensor, which has a shape of (t, m, p), where
            t is the time index, m is the minibatch index, and p is the index
            over the probabilities of each symbol in the alphabet. The memory
            layout is assumed to be in C-order, which consists in the slowest
            to the fastest changing dimension, from left to right. In this case,
            p is the fastest changing dimension.
        labels
            A 1-D tensor of all the labels for the minibatch.
        input_lengths
            A 1-D tensor with the number of time steps for each sequence in
            the minibatch.

        """

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

    def grad(self, inputs, output_grads):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'GpuConnectionistTemporalClassification Op '
                               'can not be constructed.')
        z = output_grads[0]
        grad_shuffle = GpuDimShuffle(input_broadcastable=(False, False, False),
                                     new_order=(1, 0, 2))(self.gradients())
        grad_bdot = T.basic.batched_dot(z, grad_shuffle)
        grad_shuffle_reverse = GpuDimShuffle(input_broadcastable=(False, False, False),
                                             new_order=(1, 0, 2))(grad_bdot)
        return [grad_shuffle_reverse,
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
