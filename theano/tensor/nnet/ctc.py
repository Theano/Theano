from __future__ import (division, absolute_import, print_function)
import os
import os.path
import theano.tensor as T
from theano import config
from theano import gof
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize
from theano.tensor.extra_ops import cpu_contiguous
from theano.gradient import grad_undefined

ctc_enabled = config.ctc.enabled


class ConnectionistTemporalClassification(gof.COp, gof.OpenMPOp):
    """
    CTC loss function wrapper.

    Notes
    -----
    Using the wrapper requires that Baidu's warp-ctc library is installed and the
    configuration variables `config.ctc.enabled` and `config.ctc.root` be properly
    set.

    Parameters
    ----------
    compute_grad
        If set to True, enables the computation of gradients of the CTC loss function.
    """
    __props__ = ('compute_grad', 'default_output',)

    _cop_num_inputs = 3
    _cop_num_outputs = 2

    func_file = "./ctc_wrapper.c"
    func_name = "APPLY_SPECIFIC(ctc_cost_cpu)"

    def __init__(self, compute_grad=True):
        if not ctc_enabled:
            raise RuntimeError('Baidu CTC is not enabled and '
                               'ConnectionistTemporalClassification Op '
                               'can not be constructed.')
        elif config.ctc.root == "":
            raise ValueError('ctc.root variable is not set, please set it '
                             'to the root directory of the CTC library in '
                             'your system.')

        gof.COp.__init__(self, self.func_file, self.func_name)
        gof.OpenMPOp.__init__(self)

        self.compute_grad = compute_grad
        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0

    def c_lib_dirs(self):
        dirs = []
        if ctc_enabled:
            # Find the directory that contains libwarpctc.so
            lib_found = False
            for lib_dir in ["build", "lib", "lib64"]:
                lib_path = os.path.join(config.ctc.root, lib_dir)
                if os.path.isdir(lib_path) and os.path.exists(lib_path):
                    lib_found = os.path.exists(os.path.join(lib_path, "libwarpctc.so"))
                    if lib_found:
                        dirs.append(lib_path)
                        break

            if not lib_found:
                raise RuntimeError('libwarpctc.so could not be found. ',
                                   'Please check the config.ctc.root variable.')
        return dirs

    def c_libraries(self):
        return ["warpctc"]

    def c_header_dirs(self):
        dirs = []
        if ctc_enabled:
            # We assume here that the header is available at the include directory
            # of the CTC root directory.
            dirs.append(os.path.join(config.ctc.root, "include"))
        return dirs

    def c_compile_args(self):
        return gof.OpenMPOp.c_compile_args(self)

    def c_headers(self):
        return ["ctc.h"] + gof.OpenMPOp.c_headers(self)

    def make_node(self, activations, labels, input_lengths):
        t_activations = T.as_tensor_variable(activations)
        # Ensure activations array is C-contiguous
        t_activations = cpu_contiguous(t_activations)

        t_labels = T.as_tensor_variable(labels)
        t_input_lengths = T.as_tensor_variable(input_lengths)

        if t_activations.type.dtype != 'float32':
            raise TypeError('Activations must use the float32 type!')

        if t_labels.type.dtype != 'int32':
            raise TypeError('Labels must use the int32 type!')

        if t_input_lengths.type.dtype != 'int32':
            raise TypeError('Label lengths must use the int32 type!')

        costs = T.fvector(name="ctc_cost")
        outputs = [costs]
        if self.compute_grad:
            self.gradients = T.ftensor3(name="ctc_grad")
            outputs += [self.gradients]

        return gof.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                         outputs=outputs)

    def L_op(self, inputs, outputs, output_grads):
        gradients = self.gradients
        grad_op = output_grads[0]
        total_grad = T.basic.batched_dot(grad_op, gradients.dimshuffle(1, 0, 2)).dimshuffle(1, 0, 2)
        return [total_grad,
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2])]


def ctc(activations, labels, input_lengths):
    """
    Compute CTC loss function.

    Notes
    -----
    Using the loss function requires that Baidu's warp-ctc library is installed and the
    configuration variables `config.ctc.enabled` and `config.ctc.root` be properly set.

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
    1-D array
        Cost of each example in the minibatch.
    """
    return ConnectionistTemporalClassification()(activations, labels, input_lengths)


# Disable gradient computation if not needed
@register_canonicalize
@register_stabilize
@local_optimizer([ConnectionistTemporalClassification])
def local_ctc_no_grad(node):
    if isinstance(node.op, ConnectionistTemporalClassification):
        if len(node.outputs) > 1:
            if len(node.outputs[1].clients) == 0:   # gradient is not used
                node.op.compute_grad = False
                return [ConnectionistTemporalClassification(compute_grad=False)(*node.inputs), None]
    return False
