
from theano import gof
from theano import gradient as G
from function_module import orig_function


class OpFromGraph(gof.Op):
    """
    This create an L{Op} from a list of input variables and a list of output
    variables.

    The signature is the same as the signature of L{FunctionFactory}
    and/or function and the resulting L{Op}'s perform will do the same
    operation as::
      function(inputs, outputs, **kwargs)

    Take note that the following options, if provided, must take the
    value(s) listed below:
      unpack_single = False
      borrow_outputs = False

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
    """
    
    def __init__(self, inputs, outputs, grad_depth = 1, **kwargs):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list', outputs)
        for i in inputs + outputs:
            if not isinstance(i, gof.Variable):
                raise TypeError('inputs and outputs must be Variable instances', i)
        if 'updates' in kwargs:
            raise TypeError('updates are not allowed in kwargs')
        # TODO: the graph may have implicit inputs like Value and SharedVariable instances.
        #       what impact to they have on the validity of this Op?
        self.fn = orig_function(inputs, outputs, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self.input_types = [input.type for input in inputs]
        self.output_types = [output.type for output in outputs]

        if grad_depth > 0:
            output_grads = [t() for t in self.output_types]
            gd = G.grad_sources_inputs(zip(self.outputs, output_grads), self.inputs)
            gs = map(gd.get, self.inputs)
            self.grad_ops = []
            for g in gs:
                if g is None:
                    self.grad_ops.append(lambda *args: None)
                else:
                    self.grad_ops.append(OpFromGraph(inputs + output_grads,
                                                     [g],
                                                     grad_depth = grad_depth - 1))
    def __eq__(self, other):
        #TODO: recognize a copy
        return self is other

    def __hash__(self):
        #TODO: use internal variables in hash
        return hash(type(self))

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s" % (type, input.type))
        return gof.Apply(self,
                         inputs,
                         [type() for type in self.output_types])

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in zip(outputs, variables):
            ##TODO: when function's output-borrowing semantics are correct, we wont need this
            # copy anymore
            output[0] = variable.copy()

    def grad(self, inputs, output_grads):
        if hasattr(self, 'grad_ops'):
            return [go(*(inputs + output_grads)) for go in self.grad_ops]
        else:
            raise NotImplementedError


