from __future__ import absolute_import, print_function, division

import theano
from theano import (config, gof)
import theano.tensor as T
from .basic_ops import (gpu_contiguous, as_gpuarray_variable, infer_context_name)
import theano.tensor.nnet.ctc
from .type import (GpuArrayType, gpu_context_type)
from .elemwise import GpuDimShuffle
from theano.gradient import grad_undefined
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize

import os
from . import pygpu

ctc_enabled = config.ctc.enabled


class GpuConnectionistTemporalClassification(gof.COp):
    """
    GPU wrapper for Baidu CTC loss function.

    Parameters
    ----------
    compute_grad
        If set to True, enables the computation of gradients of the CTC loss function.
    Returns
    -------
    GPU Op
        An instance of the GPU CTC loss computation Op
    """
    __props__ = ('compute_grad',)

    func_file = "./ctc_wrapper.c"
    func_name = "APPLY_SPECIFIC(ctc_cost_gpu)"

    params_type = gpu_context_type

    def __init__(self, compute_grad=True):
        if not compute_grad:
            self.func_name = "APPLY_SPECIFIC(ctc_cost_gpu_no_grad)"
        self.compute_grad = compute_grad

        gof.COp.__init__(self, self.func_file, self.func_name)

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
                'gpuarray_api.h', 'gpuarray/array.h', 'gpuarray/util.h', 'gpuarray/extension.h']

    def get_params(self, node):
        return node.inputs[0].type.context

    def make_node(self, activations, labels, input_lengths):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'GpuConnectionistTemporalClassification Op '
                               'can not be constructed.')

        context = infer_context_name(activations, labels, input_lengths)

        t_activations = as_gpuarray_variable(activations,
                                             context_name=context)
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

        costs = GpuArrayType(dtype='float32',
                             broadcastable=(False,),
                             context_name=context)()
        outputs = [costs]

        if self.compute_grad:
            gradients = GpuArrayType(dtype='float32',
                                     broadcastable=(False, False, False,),
                                     context_name=context)()
            outputs += [gradients]

        return theano.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                            outputs=outputs)

    def L_op(self, inputs, outputs, output_grads):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'GpuConnectionistTemporalClassification Op '
                               'can not be constructed.')
        # Gradients computed by Op
        gradients = outputs[1]
        # Gradients of original function, to compose chain rule
        grad_op = output_grads[0]
        grad_shuffle = GpuDimShuffle(input_broadcastable=(False, False, False,),
                                     new_order=(1, 0, 2))(gradients)
        grad_bdot = T.basic.batched_dot(grad_op, grad_shuffle)
        grad_shuffle_reverse = GpuDimShuffle(input_broadcastable=(False, False, False,),
                                             new_order=(1, 0, 2))(grad_bdot)
        return [grad_shuffle_reverse,
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2])]


def gpu_ctc(activations, labels, input_lengths):
    """
    Compute CTC loss function on the GPU.

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

    Returns
    -------
    1-D tensor
        Cost of each example in the minibatch. Tensor is of shape
        (time index, minibatch index, probabilities).
    """
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
