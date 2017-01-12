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
from theano.gof.utils import undef


class OpFromGraph(gof.Op):
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

    grad_overrides: None | undef | OpFromGraph instance | function \
        list of (None | undef | function), optional
        Used to override default gradient routine.
        Overriding function(s) must take two list of variable(s) as inputs,
        the original inputs and ups gradients
        For different `grad_overrides`:

        - `None` : will use default gradient routine.
        - theano.utils.undef : No gradient will be used (zero)
        - OpFromGraph instance: the OfG instance should accept inputs with same
            order and types of "inputs" and "output_grads" arguments as one would
            specify in grad() method
        - function : must return list of Variable.
        - list : each function must return a single Variable. The order
            of the list must corresponds to inputs

    rop_overrides: None | undef | OpFromGraph instance | function \
        list of (None | undef | function), optional
        Similar to grad_overrides, list order should match two list of "inputs"
        concatenated.

    **kwargs: optional
        Whenever this OfG instance is precompiled instead of inline, a call to
        theano.compile.function_module.orig_function during precompile phase
        will take the extra keyword args


    .. TODO:
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
        - Add optimizations prior to inline expansion such as removing unused
          inputs/outputs

    Notes
    -----
    - We support shared variables in the inner graph. This is automatic
      and invisible to the user. They can be as input to the node or in
      the inner graph.
    - We support unused inputs. This is needed for the grad.
    - `inline=True` will cause better runtime optimization at the cost
      of compilation time. Like "inline" keyword in C/C++, this is merely a
      suggestion to compiler which is not guaranteed. Currently only
      works with "fast_compile" or "fast_run" mode.
    - The function(s) supplied for overrding gradient/rop will be called
      only once at the first call to grad/R_op, and will be converted to
      OfG instances. Any side effect (modifying non local states) of the
      overriding function should not be relied on.

    Examples
    --------

    Example 1:

    .. code-block:: python

        from theano import function, tensor
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

    Example 3 override gradient

    .. code-block:: python

        from thenao import funciton, OpFromGraph, tensor, grad
        x, y, z = tensor.scalars('xyz')
        e = x + y * z
        def rescale_dy(inps, grads):
            x, y, z = inps
            g = grads
            return z*2
        op = OpFromGraph(
            [x, y, z], [e], grad_overrides=[None, rescale_dy, None])
        e2 = op(x, y, z)
        dx, dy, dz = grad(e2, [x, y, z])
        fn = function([x, y, z], [dx, dy, dz])
        # the graident wrt y is now doubled
        fn(2., 3., 4.) # [1., 8., 3.]

    """
    def __init__(self, inputs, outputs, inline=False, grad_overrides=None, rop_overrides=None, **kwargs):
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
        self.set_grad_overrides(grad_overrides)
        self.set_rop_overrides(rop_overrides)

    def __eq__(self, other):
        # TODO: recognize a copy
        return self is other

    def __hash__(self):
        # TODO: use internal variables in hash
        return hash(type(self))

    def _recompute_grad_op(self):
        if isinstance(self._grad_op, OpFromGraph):
            self._grad_op_is_cached = True
            return
        output_grads = [out_t() for out_t in self.output_types]
        if self._grad_op is None:
            self._grad_op = []

        # we need to convert a list/function into an OfG instance
        if isinstance(self._grad_op, list):
            goverrides_l = self._grad_op
            if len(goverrides_l) > len(self.local_inputs):
                raise ValueError(
                    'Can override %d gradients at most, got %d' % (
                        len(self.local_inputs), len(goverrides_l)),
                    self.goverrides_l)
            if len(goverrides_l) < len(self.local_inputs):
                goverrides_l += [None] * (
                    len(self.local_inputs) - len(goverrides_l))
            wrt_l = [lin for lin, gov in
                     izip(self.local_inputs, goverrides_l) if not gov]
            # compute non-overriding downsteam grads from upstreams grads
            # it's normal some input may be disconnected, thus the 'ignore'
            gdefaults = iter(theano.gradient.grad(
                cost=None,
                known_grads=dict(izip(self.local_outputs, output_grads)),
                wrt=wrt_l,
                disconnected_inputs='ignore') if wrt_l else [])
            # combine overriding gradients
            all_grads_l = []
            for inp, gov in izip(self.local_inputs, goverrides_l):
                if gov is None:
                    all_grads_l.append(next(gdefaults))
                elif gov is undef:
                    all_grads_l.append(
                        inp.zeros_like().astype(theano.config.floatX))
                else:
                    all_grads_l.append(gov(self.local_inputs, output_grads))
        elif self._grad_op is undef:
            all_grads_l = [
                inp.zeros_like().astype(theano.config.floatX)
                for inp in self.local_inputs]
        else:
            all_grads_l = self._grad_op(self.local_inputs, output_grads)
            if not isinstance(all_grads_l, (tuple, list)):
                all_grads_l = [all_grads_l]
            if len(all_grads_l) != len(self.local_inputs):
                raise ValueError(
                    'Gradient overriding function %s should return list of '
                    '%d outputs, got %d' % (
                        self._grad_op, len(self.local_inputs), len(all_grads_l)),
                    self._grad_op
                )
        self._grad_op = type(self)(
            inputs=self.local_inputs + output_grads,
            outputs=all_grads_l,
            inline=self.is_inline, on_unused_input='ignore',
        )
        self._grad_op_is_cached = True

    def _recompute_rop_op(self):
        if isinstance(self._rop_op, OpFromGraph):
            self._rop_op_is_cached = True
            return
        eval_points = [inp_t() for inp_t in self.input_types]
        if self._rop_op is None:
            self._rop_op = []

        if isinstance(self._rop_op, list):
            roverrides_l = self._rop_op
            if len(roverrides_l) > len(self.local_outputs):
                raise ValueError(
                    'Can override %d gradients at most, got %d' % (
                        len(self.local_onputs), len(roverrides_l)),
                    roverrides_l)
            if len(roverrides_l) < len(self.local_outputs):
                roverrides_l += [None] * (
                    len(self.local_outputs) - len(roverrides_l))
            # get outputs that does not have Rop override
            odefaults_l = [
                lo for lo, rov in izip(self.local_outputs, roverrides_l)
                if not rov]
            rdefaults_li = theano.gradient.Rop(
                f=odefaults_l,
                wrt=self.local_inputs,
                eval_points=eval_points
                )
            rdefaults = iter(rdefaults_li if odefaults_l else [])
            # combine overriding Rops
            all_rops_l = []
            for out, rov in izip(self.local_outputs, roverrides_l):
                if rov is None:
                    all_rops_l.append(next(rdefaults))
                elif rov is undef:
                    all_rops_l.append(
                        out.zeros_like().astype(theano.config.floatX))
                else:
                    all_rops_l.append(rov(self.local_inputs, eval_points))
        elif self._rop_op is undef:
            all_rops_l = [
                out.zeros_like().astype(theano.config.floatX)
                for out in self.local_outputs]
        else:
            all_rops_l = self._rop_op(self.local_inputs, eval_points)
            if not isinstance(all_rops_l, (tuple, list)):
                all_rops_l = [all_rops_l]
            if len(all_rops_l) != len(self.local_outputs):
                raise ValueError(
                    'Rop overriding function %s should return list of '
                    '%d outputs, got %d' % (
                        self._rop_op,
                        len(self.local_outputs),
                        len(all_rops_l)),
                    self._rop_op)
        self._rop_op = type(self)(
            inputs=self.local_inputs + eval_points,
            outputs=all_rops_l,
            inline=self.is_inline, on_unused_input='ignore')
        self._rop_op_is_cached = True

    def get_grad_op(self):
        """
        getter method for self._grad_op
        """
        if not self._grad_op_is_cached:
            self._recompute_grad_op()
        return self._grad_op

    def get_rop_op(self):
        """
        getter method for self._rop_op
        """
        if not self._rop_op_is_cached:
            self._recompute_rop_op()
        return self._rop_op

    def set_rop_overrides(self, rop_overrides):
        """
        Set R_op overrides, see help(theano.OpFromGraph) for syntax
        This will completely remove any previously set R_op overrides

        """
        self._rop_op = rop_overrides
        self._rop_op_is_cached = False

    def set_grad_overrides(self, grad_overrides):
        """
        Set gradient overrides, see help(theano.OpFromGraph) for syntax
        This will completely remove any previously set gradient overrides

        """
        self._grad_op = grad_overrides
        self._grad_op_is_cached = False

    def R_op(self, inputs, eval_points):
        if not self._rop_op_is_cached:
            self._recompute_rop_op()
        return self._rop_op(*(list(inputs) + list(eval_points)), return_list=True)

    def grad(self, inputs, output_grads):
        if not self._grad_op_is_cached:
            self._recompute_grad_op()
        return self._grad_op(*(list(inputs) + list(output_grads)), return_list=True)

    def make_node(self, *inputs):
        num_expected_inps = len(self.local_inputs) - len(self.shared_inputs)
        if len(inputs) != num_expected_inps:
            raise ValueError("Expected %d inputs, got %d" % (num_expected_inps, len(inputs)))
        inputs = [inp_t.filter_variable(inp) for inp, inp_t in izip(inputs, self.input_types)]
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
            self.fn.trust_input = True

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        assert len(variables) == len(outputs)
        for output, variable in izip(outputs, variables):
            # TODO: when function's output-borrowing semantics are correct,
            # we wont need this copy anymore
            output[0] = variable.copy()


@gof.local_optimizer([OpFromGraph])
def inline_ofg_expansion(node):
    """
    This optimization expands internal graph of OpFromGraph.
    Only performed if node.op.is_inline == True
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

# We want to run this before the first merge optimizer
# and before the first scan optimizer.
optdb.register(
    'inline_ofg_expansion',
    gof.opt.in2out(inline_ofg_expansion),
    -0.01, 'fast_compile', 'fast_run')

# Since OpFromGraph contains a Theano compiled function,
# we should let DebugMode know about it
ops_with_inner_function[OpFromGraph] = 'fn'
