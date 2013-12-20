from theano import gof
from theano import gradient as G
from theano.compile.function_module import orig_function
from theano.gof import ops_with_inner_function


class OpFromGraph(gof.Op):
    """This create an `Op` from a list of input variables and a list of output
    variables.

    The signature is similar to theano.function() and the resulting
    `Op` perform will do the same operation as::
      function(inputs, outputs, **kwargs)


    OpFromGraph takes an additional input, grad_depth. If grad_depth
    is n, OpFromGraph will make special Ops for gradients up to the
    nth level, allowing the user to differentiate this op up to n
    times. The parameter defaults to 1. If grad_depth == 0, the op
    will not be differentiable.

    Example:
      x, y, z = tensor.scalars('xyz')
      e = x + y * z
      op = OpFromGraph([x, y, z], [e], linker='c')
      # op behaves like a normal theano op
      e2 = op(x, y, z) + op(z, y, x)
      fn = function([x, y, z], [e2])


    TODO: -examples
          - support shared var
          - __hash__, __eq__ otherwise won't merge
          - c_code() to remove the double overhead?
          - move call to function to make_thunk().
          - opt to unfold it, work inplace on inputs
          - move grad stuff from __init__ to grad()
    """

    def __init__(self, inputs, outputs, grad_depth=1, **kwargs):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list', outputs)
        for i in inputs + outputs:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                        'inputs and outputs must be Variable instances', i)
        if 'updates' in kwargs:
            raise TypeError('updates are not allowed in kwargs')

        shared_inputs = [var for var in gof.graph.inputs(outputs)
                              if isinstance(var, SharedVariable)]
        if shared_inputs:
            raise NotImplementedError("OpFromGraph do not support SharedVariable in the inner graph")
        # TODO: the graph may have implicit inputs like
        #       SharedVariable instances.
        #       what impact to they have on the validity of this Op?
        self.fn = orig_function(inputs, outputs, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self.input_types = [input.type for input in inputs]
        self.output_types = [output.type for output in outputs]

        if grad_depth > 0:
            output_grads = [t() for t in self.output_types]
            # OpFromGraph doesn't implement a connection_pattern, so for now we regard
            # all inputs and outputs as connected. This will compute the right numerical
            # value for the gradients but could fail to raise the disconnected inputs error
            # in some cases.
            gs = G.grad(cost=None, known_grads=dict(zip(self.outputs, output_grads)),
                    wrt=self.inputs, disconnected_inputs='ignore')
            self.grad_ops = []
            for g in gs:
                if g is None:
                    self.grad_ops.append(lambda *args: None)
                else:
                    # It is normal if some inputs are not needed in order
                    # to compute the gradient, so we ignore them.
                    self.grad_ops.append(OpFromGraph(inputs + output_grads,
                                                     [g],
                                                     grad_depth=grad_depth - 1,
                                                     on_unused_input='ignore'))

    def __eq__(self, other):
        #TODO: recognize a copy
        return self is other

    def __hash__(self):
        #TODO: use internal variables in hash
        return hash(type(self))

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s"
                        % (type, input.type))
        return gof.Apply(self,
                         inputs,
                         [type() for type in self.output_types])

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in zip(outputs, variables):
            ##TODO: when function's output-borrowing semantics are correct,
            # we wont need this copy anymore
            output[0] = variable.copy()

    def grad(self, inputs, output_grads):
        if hasattr(self, 'grad_ops'):
            return [go(*(inputs + output_grads)) for go in self.grad_ops]
        else:
            raise NotImplementedError

# Since OpFromGraph contains a Theano compiled function, we should let
# DebugMode know about it
ops_with_inner_function[OpFromGraph] = 'fn'
