import numpy as np
import theano
import theano.tensor as T
from theano import gof
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize
from theano.gradient import grad_undefined

import os

class ConnectionistTemporalClassification(gof.COp):
    __props__ = ()

    func_file = "./ctc_wrapper.c"
    func_name = "APPLY_SPECIFIC(ctc_cost_cpu)"

    def __init__(self, computeGradient=True):
        super(ConnectionistTemporalClassification, self).__init__(self.func_file,
                                                                  self.func_name)
        self.computeGradient = computeGradient
        self.costs = T.fvector(name="ctc_cost")
        if self.computeGradient:
            self.gradients = T.ftensor3(name="ctc_grad")

        try:
            self.ctcLibDir = os.environ["CTC_LIB"]
        except KeyError:
            raise EnvironmentError("CTC_LIB environment variable is not set.")

    def c_lib_dirs(self):
        return [os.path.join(self.ctcLibDir, "build")]

    def c_libraries(self):
        return ["warpctc"]

    def c_header_dirs(self):
        return [os.path.join(self.ctcLibDir, "include")]

    def c_headers(self):
        return ["ctc.h"]

    def make_node(self, activations, labels, input_lengths=None):
        t_activations = T.as_tensor_variable(activations)
        t_labels = T.as_tensor_variable(labels)
        t_input_lengths = T.cast(activations.shape[0], dtype="int32") * \
           T.ones_like(activations[0,:,0], dtype=np.int32)

        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0 

        return gof.Apply(self, inputs=[t_activations, t_labels, t_input_lengths],
                         outputs=[self.costs, self.gradients])

    def grad(self, inputs, output_grads):
        # self.gradients.shape = [seqLen, batchSize, outputSize]
        # output_grads[0].shape = [batchSize]  (one cost per sequence)
        # So, reshape output_grads to [1, batchSize, 1] for broadcasting
        output_grad = output_grads[0].reshape( (1, -1, 1) )
        return [output_grad * self.gradients,
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2])]

def ctc(activations, labels, input_lengths=None):
    return ConnectionistTemporalClassification()(activations, labels, input_lengths)

# Disable gradient computation if not needed
@register_canonicalize 
@register_stabilize 
@local_optimizer([ConnectionistTemporalClassification]) 
def local_ConnectionistTemporalClassification_no_grad(node): 
  if isinstance(node.op, ConnectionistTemporalClassification): 
    if len(node.outputs) > 1:
      if len(node.outputs[1].clients) == 0:   # gradient is not used
        node.op = ConnectionistTemporalClassification(computeGradient=False)
        node.outputs = node.outputs[:1]   # costs only