"""Define new Ops from existing Ops"""
from __future__ import absolute_import, print_function, division
from functools import reduce

import theano
from theano import gof
from theano.compat import izip
from theano.compile.function_module import orig_function
from theano.compile import SharedVariable, rebuild_collect_shared
from theano.compile import optdb
from theano.gof import ops_with_inner_function
from theano.gof.graph import io_connection_pattern


class BaseOpFromGraph(gof.Op):
    """
    General syntax is similar to theano.function
    Currently does not support 'updates' or 'givens'argument

    Parameters
    ----------

    inputs: list of variables
    outputs: list of variables
    grad_overrides: None or function or list of [None|function]
        Used to override default gradient routine.
        Overriding function must take two list as inputs: original inputs
        and upstream gradients
        If is None, will use default gradient routine.
        If is function, must return list of Variable.
        If is list, each function must return a single Variable. The order
            of the list must corresponds to inputs

    Notes
    -----

    You can use regular function for grad_overrides, but the function
    must take (list of) Variable as input and output.
    """
    def __init__(self, inputs, outputs, grad_overrides=None, **kwargs):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list', outputs)
        for i in inputs+outputs:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                    'inputs and outputs must be Variable instances', i)
        if 'updates' in kwargs or 'givens' in kwargs:
            raise TypeError('updates and givens are not allowed here')


        # To correctly support shared variables the inner fct should
        # not see them. Otherwise there is a problem with the gradient.
        self.shared_inputs = [var for var in gof.graph.inputs(outputs)
                              if isinstance(var, SharedVariable)]
        shared_vars = [var.type() for var in self.shared_inputs]

        new = rebuild_collect_shared(outputs, inputs=inputs + shared_vars,
                                     replace=dict(izip(
                                         self.shared_inputs, shared_vars)),
                                     copy_inputs_over=False)
        (internal_inputs, internal_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(internal_inputs) == len(inputs) + len(self.shared_inputs)
        assert len(internal_outputs) == len(outputs)
        assert not update_d
        assert not update_expr
        assert not shared_inputs


        self.internal_inputs = internal_inputs
        self.internal_outputs = internal_outputs
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs
        self.input_types = [inp.type for inp in inputs]
        self.output_types = [out.type for out in outputs]
        # used to cache gradient for subgraph
        self.grad_ops = grad_overrides
        # should be True after 1st call to grad()
        self.cached_grad_ops = False

    def __eq__(self, other):
        # TODO: recognize a copy
        return self is other

    def __hash__(self):
        # TODO: use internal variables in hash
        return hash(type(self))

    def grad(self, inputs, output_grads):
        if self.cached_grad_ops:
            return self.grad_ops(inputs+output_grads)

        grad_inps = self.internal_inputs + output_grads
        upstream_grads = dict(izip(self.internal_outputs, output_grads))
        if self.grad_ops is not None:
            grad_ops_l = self.grad_ops
            if isinstance(grad_ops_l, list):
                assert len(grad_ops_l) <= len(self.internal_inputs)
                if len(grad_ops_l)<len(self.internal_inputs):
                    grad_ops_l += [None]*(
                        len(self.internal_inputs) - len(grad_ops_l))
                # It is normal if some inputs are not needed in order
                # to compute the gradient, so we ignore them.
                gs = [go if go else type(self)(
                    grad_inps,
                    theano.gradient.grad(
                        cost=None,
                        known_grads=upstream_grads,
                        wrt=[inp],
                        disconnected_inputs='ignore'),
                    on_unused_input='ignore'
                ) for go, inp in izip(grad_ops_l, self.internal_inputs)]
                # since BaseOpFromGraph only accepts and outputs list,
                # additional filtering is needed
                grad_ops = lambda inps:[
                    (go(inps) if ov else go(*inps))
                    for go, ov in izip(gs, grad_ops_l)]
            else:
                grad_ops = grad_ops_l
            self.grad_ops = grad_ops
        else:
            gs = theano.gradient.grad(
                cost=None,
                known_grads=upstream_grads,
                wrt=self.internal_inputs,
                disconnected_inputs='ignore')
            grad_ops_l = []
            for g in gs:
                if g is None:
                    grad_ops_l.append(lambda *args: None)
                else:
                    grad_ops_l.append(type(self)(grad_inps,
                                                [g],
                                                on_unused_input='ignore'))
            grad_ops = lambda inps:[go(*inps) for go in grad_ops_l]
            self.grad_ops = grad_ops
        self.cached_grad_ops = True
        return grad_ops(inputs+output_grads)

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s" %
                                (type, input.type))

        apply_node = gof.Apply(self,
                         list(inputs) + self.shared_inputs,
                         [type() for type in self.output_types])
        apply_node.internal_inputs = self.internal_inputs
        apply_node.internal_outputs = self.internal_outputs
        return apply_node

    def connection_pattern(self, node):
        """
        Return connection pattern of subfgraph defined by inputs and outputs.

        """
        return io_connection_pattern(self.internal_inputs, self.internal_outputs)

    def infer_shape(self, node, shapes):
        out_shp = theano.scan_module.scan_utils.infer_shape(
            self.internal_outputs,
            self.internal_inputs,
            shapes)

        # Clone the output shape so that shape are computed from outer inputs.
        # Note:
        # Here we can do it more simply like:
        #      ret = [theano.clone(shp, replace=repl) for shp in out_shp]
        # But  doing it multiple time could duplicate common subgraph between
        # each shape call. Theano optimizer will clean this up later, but this
        # will ask extra work to the optimizer.
        repl = dict(zip(self.internal_inputs, node.inputs))
        cloned = theano.clone(reduce(tuple.__add__, out_shp), replace=repl)
        ret = []
        used = 0
        for i in range(len(out_shp)):
            nb = len(out_shp[i])
            ret.append(cloned[used: used + nb])
            used += nb

        return ret
    def perform(self, node, inputs, outputs):
        raise NotImplementedError()

class OpFromGraphPrecompiled(BaseOpFromGraph):
    """WRITEME"""
    def prepare_node(self, node, storage_map, compute_map, impl):
        if not hasattr(self, "fn") and impl == 'py':
            self.fn = orig_function(self.internal_inputs,
                                    self.internal_outputs,
                                    **self.kwargs)

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in zip(outputs, variables):
            # TODO: when function's output-borrowing semantics are correct,
            # we wont need this copy anymore
            output[0] = variable.copy()

class OpFromGraphInline(BaseOpFromGraph):
    """WRITEME"""
    def perform(self, node, inputs, outputs):
        raise RuntimeError('OpFromGraphInline is not supposed to be executed at runtime')
        pass

@gof.local_optimizer([OpFromGraphInline])
def inline_ofg_expansion(node):
    op = node.op
    if not isinstance(op, OpFromGraphInline):
        return False
    outputs = theano.clone(
        op.internal_outputs, {
            u:v for u,v in izip(
                node.op.internal_inputs, node.inputs)})
    return outputs

optdb.register(
    'inline_ofg_expansion',
    gof.opt.in2out(inline_ofg_expansion),
    0.5, 'fast_compile', 'fast_run')

ops_with_inner_function[OpFromGraphPrecompiled] = 'fn'

# for backward compatibility
OpFromGraph = OpFromGraphPrecompiled

# APIs for OpFromGraph*
def op_from_graph(
    inputs, outputs, inline=False, grad_overrides=None, **kwargs):
    cls_opfromgraph = OpFromGraphInline if inline else OpFromGraphPrecompiled
    return cls_opfromgraph(
        inputs, outputs, grad_overrides=grad_overrides, **kwargs)

#BELOW is the original OpFromGraph
'''
class OpFromGraph(gof.Op):
    """
    This creates an `Op` from inputs and outputs lists of variables.

    The signature is similar to theano.function() and the resulting
    `Op`'s perform will do the same operation as::

        orig_function(inputs, outputs, **kwargs)

    TODO:
        - examples for a multi-layer mlp. where?
        - __hash__, __eq__ otherwise won't merge, try
          gof.opt.is_same_graph_with_merge(op1.internal_outputs, op2,
          internal_outputs)
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

        import numpy as np
        import theano
        from theano import config, function, OpFromGraph, tensor
        x, y, z = tensor.scalars('xyz')
        s = theano.shared(np.random.rand(2, 2).astype(config.floatX))
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

        # To correctly support shared variables the inner fct should
        # not see them. Otherwise there is a problem with the gradient.
        self.shared_inputs = [var for var in gof.graph.inputs(outputs)
                              if isinstance(var, SharedVariable)]
        shared_vars = [var.type() for var in self.shared_inputs]
        new = rebuild_collect_shared(outputs, inputs=inputs + shared_vars,
                                     replace=dict(izip(self.shared_inputs,
                                                       shared_vars)),
                                     copy_inputs_over=False)
        (internal_inputs, internal_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(internal_inputs) == len(inputs) + len(self.shared_inputs)
        assert len(internal_outputs) == len(outputs)
        assert not update_d
        assert not update_expr
        assert not shared_inputs

        self.internal_inputs = internal_inputs
        self.internal_outputs = internal_outputs
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

    def prepare_node(self, node, storage_map, compute_map, impl):
        if not hasattr(self, "fn") and impl == 'py':
            self.fn = orig_function(self.internal_inputs,
                                    self.internal_outputs,
                                    **self.kwargs)

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
        return io_connection_pattern(self.internal_inputs, self.internal_outputs)

    def infer_shape(self, node, shapes):
        out_shp = theano.scan_module.scan_utils.infer_shape(self.internal_outputs,
                                                            self.internal_inputs,
                                                            shapes)

        # Clone the output shape so that shape are computed from outer inputs.
        # Note:
        # Here we can do it more simply like:
        #      ret = [theano.clone(shp, replace=repl) for shp in out_shp]
        # But  doing it multiple time could duplicate common subgraph between
        # each shape call. Theano optimizer will clean this up later, but this
        # will ask extra work to the optimizer.
        repl = dict(zip(self.internal_inputs, node.inputs))
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
                                      known_grads=dict(izip(self.internal_outputs,
                                                            output_grads)),
                                      wrt=self.internal_inputs,
                                      disconnected_inputs='ignore')

            grad_ops = []
            for g in gs:
                if g is None:
                    grad_ops.append(lambda *args: None)
                else:
                    # It is normal if some inputs are not needed in order
                    # to compute the gradient, so we ignore them.
                    grad_ops.append(OpFromGraph(self.internal_inputs + output_grads,
                                                [g],
                                                on_unused_input='ignore'))
            self.grad_ops = grad_ops

        return [go(*(inputs + output_grads)) for go in grad_ops]
'''
