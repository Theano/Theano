from __future__ import absolute_import, print_function, division

import theano
from theano import (config, gof)
import theano.tensor as T
from .basic_ops import (gpu_contiguous, as_gpuarray_variable, infer_context_name, gpuarray_helper_inc_dir)
import theano.tensor.nnet.ctc
from .type import (GpuArrayType, gpu_context_type)
from .elemwise import GpuDimShuffle
from theano.gradient import grad_undefined
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.nnet.ctc import ctc_available

import os
import sys
from . import pygpu


class GpuConnectionistTemporalClassification(gof.COp):
    """
    GPU wrapper for Baidu CTC loss function.

    Parameters
    ----------
    compute_grad
        If set to True, enables the computation of gradients of the CTC loss function.
    """
    __props__ = ('compute_grad',)

    _cop_num_inputs = 3
    _cop_num_outputs = 2

    func_file = "./c_code/ctc_wrapper.c"
    func_name = "APPLY_SPECIFIC(ctc_cost_gpu)"

    params_type = gpu_context_type

    def __init__(self, compute_grad=True):
        if not ctc_available():
            raise RuntimeError('Baidu CTC is not available and '
                               'GpuConnectionistTemporalClassification Op '
                               'can not be constructed.')

        self.compute_grad = compute_grad
        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0

        gof.COp.__init__(self, self.func_file, self.func_name)

    def c_lib_dirs(self):
        lib_dirs = []
        if ctc_available.path is not None:
            lib_dirs += [ctc_available.path]
        return lib_dirs

    def c_compile_args(self):
        if ctc_available.path is not None:
            if sys.platform != 'darwin' and ' ' in ctc_available.path:
                return ['-Wl,-rpath,"' + ctc_available.path + '"']
            else:
                return ['-Wl,-rpath,' + ctc_available.path]
        return []

    def c_libraries(self):
        return ["warpctc", "gpuarray"]

    def c_header_dirs(self):
        dirs = [gpuarray_helper_inc_dir(), pygpu.get_include(),
                config.cuda.include_path]
        if config.ctc.root != '':
            dirs.append(os.path.join(config.ctc.root, "include"))
        return dirs

    def c_headers(self):
        return ['ctc.h', 'numpy_compat.h', 'gpuarray/ext_cuda.h',
                'gpuarray_helper.h', 'gpuarray/types.h', 'gpuarray_api.h',
                'gpuarray/array.h', 'gpuarray/util.h', 'gpuarray/extension.h']

    def get_params(self, node):
        return node.inputs[0].type.context

    def make_node(self, activations, labels, input_lengths):
        context_name = infer_context_name(activations)
        t_activations = as_gpuarray_variable(activations,
                                             context_name=context_name)
        # Ensure activations array is C-contiguous
        t_activations = gpu_contiguous(t_activations)

        # Labels and input lengths are always on the CPU
        t_labels = T.as_tensor_variable(labels)
        t_input_lengths = T.as_tensor_variable(input_lengths)

        if t_activations.type.dtype != 'float32':
            raise TypeError('activations must use the float32 type.')

        if t_activations.ndim != 3:
            raise ValueError('activations must have 3 dimensions.')

        if t_labels.type.dtype != 'int32':
            raise TypeError('labels must use the int32 type.')

        if t_labels.ndim != 2:
            raise ValueError('labels must have 2 dimensions.')

        if t_input_lengths.type.dtype != 'int32':
            raise TypeError('input_lengths must use the int32 type.')

        if t_input_lengths.ndim != 1:
            raise ValueError('input_lengths must have 1 dimension.')

        costs = GpuArrayType(dtype='float32',
                             broadcastable=(False,),
                             context_name=context_name)()
        outputs = [costs]

        if self.compute_grad:
            gradients = GpuArrayType(dtype='float32',
                                     broadcastable=(False, False, False,),
                                     context_name=context_name)()
            outputs += [gradients]

        return theano.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                            outputs=outputs)

    def L_op(self, inputs, outputs, output_grads):
        # Gradients computed by Op
        assert self.compute_grad and len(outputs) == 2
        gradients = outputs[1]
        assert gradients is not None

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
        A 2-D tensor of all the labels for the minibatch. In each row, there
        is a sequence of target labels. Negative values are assumed to be padding,
        and thus are ignored. Blank symbol is assumed to have index 0 in the
        alphabet.
    input_lengths
        A 1-D tensor with the number of time steps for each sequence in
        the minibatch.

    Returns
    -------
    1-D array
        Cost of each example in the minibatch.
    """
    return GpuConnectionistTemporalClassification()(activations, labels, input_lengths)


# Disable gradient computation if not needed
@register_canonicalize("fast_compile")
@local_optimizer([GpuConnectionistTemporalClassification])
def local_gpu_ctc_no_grad(node):
    if isinstance(node.op, GpuConnectionistTemporalClassification):
        if len(node.outputs) > 1:
            if len(node.outputs[1].clients) == 0:   # gradient is not used
                return [GpuConnectionistTemporalClassification(compute_grad=False)(*node.inputs), None]
    return False
