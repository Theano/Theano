"""Define new Ops from existing Ops"""
from __future__ import absolute_import, print_function, division
from functools import reduce

import theano
from theano import gof
from theano.compat import izip
from theano.compile.function_module import orig_function
from theano.compile import SharedVariable, rebuild_collect_shared, optdb
from theano.gof import ops_with_inner_function
from theano.gof.graph import io_connection_pattern


class OpFromGraph(gof.Op):
    """
    class for Ops with user-defined inner graph
    """
    # NOTE: if you make a subclass of this, make sure add test for it under:
    # theano/compile/tests/test_builders.py
    def __init__(self, inputs, outputs, inline=False, grad_overrides=None, **kwargs):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list', outputs)
        for i in inputs + outputs:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                    'inputs and outputs must be Variable instances', i)
        if 'updates' in kwargs or 'givens' in kwargs:
            raise TypeError('updates and givens are not allowed here')
        self.is_inline = inline
        # To correctly support shared variables the inner fct should
        # not see them. Otherwise there is a problem with the gradient.
        self.shared_inputs = [var for var in gof.graph.inputs(outputs)
                              if isinstance(var, SharedVariable)]
        shared_vars = [var.type() for var in self.shared_inputs]

        new = rebuild_collect_shared(outputs, inputs=inputs + shared_vars,
                                     replace=dict(izip(
                                         self.shared_inputs, shared_vars)),
                                     copy_inputs_over=False)
        (local_inputs, local_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(local_inputs) == len(inputs) + len(self.shared_inputs)
        assert len(local_outputs) == len(outputs)
        assert not update_d
        assert not update_expr
        assert not shared_inputs

        self.local_inputs = local_inputs
        self.local_outputs = local_outputs
        self.inputs = inputs
        self.outputs = outputs
        self.kwargs = kwargs
        self.input_types = [inp.type for inp in inputs]
        self.output_types = [out.type for out in outputs]
        # grad_op: a functor takes form:
        #
        # def grad_op(inputs:list, ups_grads:list):
        #     return dns_grads:list
        #
        # This is used to cache gradient for subgraph
        # for __init__, just set as grad_overrides
        #
        # grad_op should be build on the 1st call to grad()
        # after which grad_op_is_cached should be True
        self.grad_op = grad_overrides
        self.grad_op_is_cached = False

    def __eq__(self, other):
        # TODO: recognize a copy
        return self is other

    def __hash__(self):
        # TODO: use internal variables in hash
        return hash(type(self))

    def grad(self, inputs, output_grads):
        if self.grad_op_is_cached:
            return self.grad_op(inputs, output_grads)

        if self.grad_op is None:
            self.grad_op = []

        # we need to convert a list into a single funtor
        if isinstance(self.grad_op, list):
            grad_op_l = self.grad_op
            if len(grad_op_l) > len(self.local_inputs):
                raise ValueError(
                    'Can override %d gradients at most, got %d' % (
                        len(self.local_inputs), len(grad_op_l)))
            if len(grad_op_l) < len(self.local_inputs):
                grad_op_l += [None] * (
                    len(self.local_inputs) - len(grad_op_l))
            wrt = [self.local_inputs[i] for i, go in
                   enumerate(grad_op_l) if not go]
            # compute non-overriding downsteam gradients from upstreams grads
            # it's normal some input may be disconnected, thus the 'ignore'
            ups_grads_d = dict(izip(self.local_outputs, output_grads))
            nat_dns_grads = iter(theano.gradient.grad(
                cost=None,
                known_grads=ups_grads_d,
                wrt=wrt,
                disconnected_inputs='ignore'))
            # combine overriding gradients
            dns_grads_l = [
                go(self.local_inputs, output_grads) if go else next(nat_dns_grads) for go in grad_op_l]
            grad_ofg = type(self)(
                inputs=self.local_inputs + output_grads,
                outputs=dns_grads_l,
                inline=self.is_inline, on_unused_input='ignore')

            def grad_op(inps, grds):
                return grad_ofg(*(list(inps) + list(grds)))
            self.grad_op = grad_op
        self.grad_op_is_cached = True
        return self.grad_op(inputs, output_grads)

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s" %
                                (type, input.type))

        apply_node = gof.Apply(
            self, list(inputs) + self.shared_inputs,
            [type() for type in self.output_types])
        apply_node.local_inputs = self.local_inputs
        apply_node.local_outputs = self.local_outputs
        return apply_node

    def connection_pattern(self, node):
        """
        Return connection pattern of subfgraph defined by inputs and outputs.

        """
        return io_connection_pattern(
            self.local_inputs, self.local_outputs)

    def infer_shape(self, node, shapes):
        out_shp = theano.scan_module.scan_utils.infer_shape(
            self.local_outputs,
            self.local_inputs,
            shapes)

        # Clone the output shape so that shape are computed from outer inputs.
        # Note:
        # Here we can do it more simply like:
        #      ret = [theano.clone(shp, replace=repl) for shp in out_shp]
        # But  doing it multiple time could duplicate common subgraph between
        # each shape call. Theano optimizer will clean this up later, but this
        # will ask extra work to the optimizer.
        repl = dict(zip(self.local_inputs, node.inputs))
        cloned = theano.clone(reduce(tuple.__add__, out_shp), replace=repl)
        ret = []
        used = 0
        for i in range(len(out_shp)):
            nb = len(out_shp[i])
            ret.append(cloned[used: used + nb])
            used += nb

        return ret

    def prepare_node(self, node, storage_map, compute_map, impl):
        if not hasattr(self, "fn") and impl == 'py':
            self.fn = orig_function(self.local_inputs,
                                    self.local_outputs,
                                    **self.kwargs)

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in zip(outputs, variables):
            # TODO: when function's output-borrowing semantics are correct,
            # we wont need this copy anymore
            output[0] = variable.copy()


@gof.local_optimizer([OpFromGraph])
def inline_ofg_expansion(node):
    """
    This optimization expands internal graph of OpFromGraph.

    Doing so can improve optimization at the cost of compilation speed.
    """
    op = node.op
    if not isinstance(op, OpFromGraph):
        return False
    if not op.is_inline:
        return False
    return theano.clone(
        op.local_outputs, {
            u: v for u, v in izip(
                node.op.local_inputs, node.inputs)})

optdb.register(
    'inline_ofg_expansion',
    gof.opt.in2out(inline_ofg_expansion),
    0.5, 'fast_compile', 'fast_run')

# Since OpFromGraph contains a Theano compiled function,
# we should let DebugMode know about it
ops_with_inner_function[OpFromGraph] = 'fn'


# API for OpFromGraph
def op_from_graph(
    inputs, outputs, inline=False, grad_overrides=None, **kwargs
):
    """
    This creates an `Op` from inputs and outputs lists of variables.
    The signature is similar to theano.function() and the resulting
    `Op`'s perform will do the same operation as::

        orig_function(inputs, outputs, **kwargs)
    Currently does not support 'updates' or 'givens' argument.

    Parameters
    ----------

    inputs: list of variables
    outputs: list of variables
    inline: bool, optional
        if True, will cause the Op's original graph being used during
        compilation, otherwise will use a pre-compiled function inside.
    grad_overrides: None | function | list of (None|function), optional
        Used to override default gradient routine.
        Overriding function(s) must take two list of variable as inputs,
        the original inputs and ups gradients
        For different `grad_overrides`:

        - `None` : will use default gradient routine.
        - function : must return list of Variable.
        - list : each function must return a single Variable. The order
            of the list must corresponds to inputs

    TODO:
        - examples for a multi-layer mlp. where?
        - __hash__, __eq__ otherwise won't merge, try
          gof.opt.is_same_graph_with_merge(op1.local_outputs, op2,
          local_outputs)
        - c_code() to remove the double overhead?
        - grad() make it support DisconnectedType and the new interface
        - check how it works with updates.
        - add test with constant as input or inside the inner graph.
        - Add support for the GPU? Probably just need an opt to remove transfer
        - Add support to pickle this Op.
        - Add support/test with random generator
        - Recursion detection to prevent Op "forkbomb", either set depth
          limit or manually check them.

    Notes
    -----
    - We support shared variables in the inner graph. This is automatic
      and invisible to the user. They can be as input to the node or in
      the inner graph.
    - We support unused inputs. This is needed for the grad.
    - `inline=True` will cause better runtime optimization at the cost
      of compilation time. Like "inline" keyword in C, this is merely a
      suggestion to compiler which is not guaranteed. Currently only
      works with "fast_compile" or "fast_run" mode.

    Examples
    --------

    Example 1:

    .. code-block:: python

        from theano import function, op_from_graph, tensor
        x, y, z = tensor.scalars('xyz')
        e = x + y * z
        op = op_from_graph([x, y, z], [e])
        # op behaves like a normal theano op
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    Example 2 with shared variable:

    .. code-block:: python

        import numpy as np
        import theano
        from theano import config, function, op_from_graph, tensor
        x, y, z = tensor.scalars('xyz')
        s = theano.shared(np.random.rand(2, 2).astype(config.floatX))
        e = x + y * z + s
        op = op_from_graph([x, y, z], [e])
        # op behaves like a normal theano op
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    Example 3 override gradient

    .. code-block:: python

        from thenao import funciton, op_from_graph, tensor, grad
        x, y, z = tensor.scalars('xyz')
        e = x + y * z
        def rescale_dy(inps, grads):
            x, y, z = inps
            g = grads
            return z*2
        op = op_from_graph(
            [x, y, z], [e], grad_overrides=[None, rescale_dy, None])
        e2 = op(x, y, z)
        dx, dy, dz = grad(e2, [x, y, z])
        fn = function([x, y, z], [dx, dy, dz])
        # the graident wrt y is now doubled
        fn(2., 3., 4.) # [1., 8., 3.]

    """
    return OpFromGraph(
        inputs, outputs, inline=inline, grad_overrides=grad_overrides, **kwargs)
