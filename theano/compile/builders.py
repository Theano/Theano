from __future__ import absolute_import, print_function, division
import theano
from theano import gof
from theano.compat import izip
from theano.compile.function_module import orig_function
from theano.compile import SharedVariable, rebuild_collect_shared
from theano.gof import ops_with_inner_function
from theano.gof.graph import io_connection_pattern

from functools import reduce


class OpFromGraph(gof.Op):
    """
    This creates an `Op` from inputs and outputs lists of variables.

    The signature is similar to theano.function() and the resulting
    `Op`'s perform will do the same operation as::

        orig_function(inputs, outputs, **kwargs)

    TODO:
        - examples for a multi-layer mlp. where?
        - __hash__, __eq__ otherwise won't merge, try
          gof.opt.is_same_graph_with_merge(op1.new_outputs, op2,
          new_outputs)
        - c_code() to remove the double overhead?
        - opt to unfold it, work inplace on inputs
        - grad() make it support DisconnectedType and the new interface
        - check how it works with updates.
        - add test with constant as input or inside the inner graph.
        - Add support for the GPU? Probably just need an opt to remove transfer
        - Add support to pickle this Op.
        - Add support/test with random generator

    Notes
    -----
    - We support shared variables in the inner graph. This is automatic and
      invisible to the user. They can be as input to the node or in the
      inner graph.
    - We support unused inputs. This is needed for the grad.

    Examples
    --------

    Example 1:

    .. code-block:: python

        from theano import function, OpFromGraph, tensor
        x, y, z = tensor.scalars('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        # op behaves like a normal theano op
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    Example 2 with shared variable:

    .. code-block:: python

        import numpy
        import theano
        from theano import config, function, OpFromGraph, tensor
        x, y, z = tensor.scalars('xyz')
        s = theano.shared(numpy.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = OpFromGraph([x, y, z], [e])
        # op behaves like a normal theano op
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    """

    def __init__(self, inputs, outputs, **kwargs):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list', outputs)
        for i in inputs + outputs:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                    'inputs and outputs must be Variable instances', i)
        if 'updates' in kwargs or 'givens' in kwargs:
            raise TypeError('updates and givens are not allowed in kwargs')

        # To support correctly shared variables the inner fct should
        # not see them. Otherwise their is problem with the gradient.
        self.shared_inputs = [var for var in gof.graph.inputs(outputs)
                              if isinstance(var, SharedVariable)]
        shared_vars = [var.type() for var in self.shared_inputs]
        new = rebuild_collect_shared(outputs, inputs=inputs + shared_vars,
                                     replace=dict(izip(self.shared_inputs,
                                                       shared_vars)),
                                     copy_inputs_over=False)
        (new_inputs, new_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(new_inputs) == len(inputs) + len(self.shared_inputs)
        assert len(new_outputs) == len(outputs)
        assert not update_d
        assert not update_expr
        assert not shared_inputs

        self.new_inputs = new_inputs
        self.new_outputs = new_outputs
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs
        self.input_types = [input.type for input in inputs]
        self.output_types = [output.type for output in outputs]

    def __eq__(self, other):
        # TODO: recognize a copy
        return self is other

    def __hash__(self):
        # TODO: use internal variables in hash
        return hash(type(self))

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s" %
                                (type, input.type))
        return gof.Apply(self,
                         list(inputs) + self.shared_inputs,
                         [type() for type in self.output_types])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        ret = super(OpFromGraph, self).make_thunk(node, storage_map,
                                                  compute_map, no_recycling)
        if not hasattr(self, "fn"):
            self.fn = orig_function(self.new_inputs,
                                    self.new_outputs,
                                    **self.kwargs)
        return ret

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in zip(outputs, variables):
            # TODO: when function's output-borrowing semantics are correct,
            # we wont need this copy anymore
            output[0] = variable.copy()

    def connection_pattern(self, node):
        """
        Return connection pattern of subfgraph defined by inputs and outputs.

        """
        return io_connection_pattern(self.new_inputs, self.new_outputs)

    def infer_shape(self, node, shapes):
        out_shp = theano.scan_module.scan_utils.infer_shape(self.new_outputs,
                                                            self.new_inputs,
                                                            shapes)

        # Clone the output shape so that shape are computed from outer inputs.
        # Note:
        # Here we can do it more simply like:
        #      ret = [theano.clone(shp, replace=repl) for shp in out_shp]
        # But  doing it multiple time could duplicate common subgraph between
        # each shape call. Theano optimizer will clean this up later, but this
        # will ask extra work to the optimizer.
        repl = dict(zip(self.new_inputs, node.inputs))
        cloned = theano.clone(reduce(tuple.__add__, out_shp), replace=repl)
        ret = []
        used = 0
        for i in range(len(out_shp)):
            nb = len(out_shp[i])
            ret.append(cloned[used: used + nb])
            used += nb

        return ret

    def grad(self, inputs, output_grads):
        if hasattr(self, "grad_ops"):
            grad_ops = self.grad_ops
        else:
            gs = theano.gradient.grad(cost=None,
                                      known_grads=dict(izip(self.new_outputs,
                                                            output_grads)),
                                      wrt=self.new_inputs,
                                      disconnected_inputs='ignore')

            grad_ops = []
            for g in gs:
                if g is None:
                    grad_ops.append(lambda *args: None)
                else:
                    # It is normal if some inputs are not needed in order
                    # to compute the gradient, so we ignore them.
                    grad_ops.append(OpFromGraph(self.new_inputs + output_grads,
                                                [g],
                                                on_unused_input='ignore'))
            self.grad_ops = grad_ops

        return [go(*(inputs + output_grads)) for go in grad_ops]

# Since OpFromGraph contains a Theano compiled function, we should let
# DebugMode know about it
ops_with_inner_function[OpFromGraph] = 'fn'
