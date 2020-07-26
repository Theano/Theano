"""Define new Ops from existing Ops"""
from __future__ import absolute_import, division, print_function
from functools import reduce, partial
from collections import OrderedDict

import theano
from theano import gof
from theano.compat import izip
from theano.compile.function_module import orig_function
from theano.compile import SharedVariable, rebuild_collect_shared, optdb
from theano.gof import Variable, ops_with_inner_function
from theano.gof.graph import io_connection_pattern
from theano.gof.null_type import NullType
from theano.gradient import DisconnectedType


class OpFromGraph(gof.Op):
    r"""
    This creates an ``Op`` from inputs and outputs lists of variables.
    The signature is similar to :func:`theano.function <theano.function>`
    and the resulting ``Op``'s perform will do the same operation as::

        orig_function(inputs, outputs, **kwargs)

    Currently does not support ``updates`` or ``givens`` argument.

    Parameters
    ----------

    inputs: list of :class:`Variable <theano.gof.Variable>`

    outputs: list of :class:`Variable <theano.gof.Variable>`

    inline: bool, optional
        Defaults to ``False``

        ``True`` : Cause the Op's original graph being used during
        compilation, the Op will not be visible in the compiled
        graph but rather its internal graph.

        ``False`` : will use a pre-compiled function inside.

    grad_overrides : single or list of {'default', OpFromGraph, callable, Variable with special type}, optional
        Defaults to ``'default'``.
        This argument is mutually exclusive with lop_overrides.

        ``'default'`` : Do not override, use default grad() result

        OpFromGraph instance : Override with another OpFromGraph, should
        accept inputs as the same order and types of ``inputs`` and ``output_grads``
        arguments as one would specify in grad() method.

        callable : Should take two args: ``inputs`` and ``output_grads``.
        Each argument is expected to be a list of :class:`Variable <theano.gof.Variable>`.
        Must return list of :class:`Variable <theano.gof.Variable>`.

        Variable :
            ``NullType() instance`` : Treat as non-differentiable
            ``DisconnectedType() instance`` : Treat as disconnected gradient, numerically gives zero

        list: Each OpFromGraph/callable must return a single
        :class:`Variable <theano.gof.Variable>`. Each list element corresponds to gradient of
        a specific input, length of list must be equal to number of inputs.

    lop_overrides : single or list of {'default', OpFromGraph, callable, Variable with special type}, optional
        Defaults to ``'default'``.
        This argument is mutually exclusive with ``grad_overrides``.

        ``'default'`` : Do not override, use default L_op() result

        OpFromGraph instance : Override with another OpFromGraph, should
        accept inputs as the same order and types of ``inputs``, ``outputs`` and ``output_grads``
        arguments as one would specify in grad() method.

        callable : Should take three args: ``inputs``, ``outputs`` and ``output_grads``.
        Each argument is expected to be a list of :class:`Variable <theano.gof.Variable>`.
        Must return list of :class:`Variable <theano.gof.Variable>`.

        Variable :
            ``NullType() instance`` : Treat as non-differentiable
            ``DisconnectedType() instance`` : Treat as disconnected gradient, numerically gives zero

        list: Each OpFromGraph/callable must return a single
        :class:`Variable <theano.gof.Variable>`. Each list element corresponds to gradient of
        a specific input, length of list must be equal to number of inputs.

    rop_overrides : single or list of {'default', OpFromGraph, callable, Variable with special type}, optional
        Defaults to ``default``.

        ``'default'`` : Do not override, use default R_op() result

        OpFromGraph instance : Override with another OpFromGraph, should
        accept inputs as the same order and types of ``inputs`` and ``eval_points``
        arguments as one would specify in R_op() method.

        callable : Should take two args: ``inputs`` and ``eval_points``.
        Each argument is expected to be a list of :class:`Variable <theano.gof.Variable>`.
        Must return list of :class:`Variable <theano.gof.Variable>`.

        Variable :
            ``NullType() instance`` : Treat as non-differentiable
            ``DisconnectedType() instance`` : Treat as zero since DisconnectedType is not yet supported in R_op

        list: Each OpFromGraph/callable must return a single
        :class:`Variable <theano.gof.Variable>`. Each list element corresponds
        to a specific output of R_op, length of list must be equal to number of outputs.

    connection_pattern : list of list

        If not ``None``, this will be used as the connection_pattern
        for this op.

    name : string, optional
        A name for debugging purposes

    \*\*kwargs : optional
        Check
        :func:`orig_function <theano.compile.function_module.orig_function>`
        for more arguments, only works when not inline.


    .. TODO:
        - examples for a multi-layer mlp. where?
        - __hash__, __eq__ otherwise won't merge, try
          gof.opt.is_same_graph_with_merge(op1.local_outputs, op2,
          local_outputs)
        - c_code() to remove the double overhead?
        - grad() make it support DisconnectedType and the new interface
        - add support for NullType and DisconnectedType when R_op supports them
        - check how it works with updates.
        - add test with constant as input or inside the inner graph.
        - Add support for the GPU? Probably just need an opt to remove transfer
        - Add support to pickle this Op.
        - Add support/test with random generator
        - Add optimization to removing unused inputs/outputs
        - Add optimization to work inplace on inputs when not inline

    Notes
    -----
    - We support shared variables in the inner graph. This is automatic
      and invisible to the user. They can be as input to the node or in
      the inner graph.
    - We support unused inputs. This is needed for the grad.
    - We support nested OpFromGraph.
    - ``inline=True`` will cause better runtime optimization at the cost
      of compilation time. Currently only works with ``fast_compile`` or
      ``fast_run`` mode.
    - For overriding, it's recommended to provide pure functions (no side
      effects like setting global variable) as callable(s). The callable(s)
      supplied for overriding gradient/rop will be called only once at the
      first call to grad/R_op, and will be converted to OpFromGraph instances.

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

    Example 3 override gradient

    .. code-block:: python

        from theano import function, OpFromGraph, tensor, grad
        x, y, z = tensor.scalars('xyz')
        e = x + y * z
        def rescale_dy(inps, grads):
            x, y, z = inps
            g, = grads
            return z*2
        op = OpFromGraph(
            [x, y, z], [e], grad_overrides=['default', rescale_dy, 'default']
        e2 = op(x, y, z)
        dx, dy, dz = grad(e2, [x, y, z])
        fn = function([x, y, z], [dx, dy, dz])
        # the gradient wrt y is now doubled
        fn(2., 3., 4.) # [1., 8., 3.]

    """
    TYPE_ERR_MSG = ("L_op/gradient override should be (single or list of)"
                    "'default' | OpFromGraph | callable | Variable "
                    "with NullType or DisconnectedType, got %s")
    STYPE_ERR_MSG = ('Overriding Variable instance can only have type'
                     ' of DisconnectedType or NullType, got %s')
    LOP_TYPE_ERR_MSG = 'L_op type can only be "grad" or "lop", got %s.'
    OV_INP_LEN_ERR_MSG = 'expect overrider with %d inputs, got %d'

    @staticmethod
    def _filter_grad_var(grad, inp):
        # Returns (filtered_var, overrider_var)
        # Args:
        #     grad: gradient Variable
        #     inp: the corresponding input of gradient Variable
        #
        # a grad() call could return instance of NullType() or DisconnectedType()
        # which cannot be directly used in OfG
        #
        # Since we always use an OfG instance as self._lop_op, the current
        # workaround is to "remember" the special cases of the gradient and
        # replace them after self._lop_op is called.
        #
        # This helper function changes invalid types into a filtered_var,
        # and provides a overrider_var to be replaced at grad() call
        #
        # For now, this converts NullType or DisconnectedType into zeros_like.
        # other types are unmodified: overrider_var -> None
        if isinstance(grad.type, (NullType, DisconnectedType)):
            if hasattr(inp, 'zeros_like'):
                return inp.zeros_like(), grad
            else:
                return theano.tensor.constant(0.), grad
        else:
            return grad, None

    @staticmethod
    def _filter_rop_var(inpJ, out):
        # mostly similar to _filter_grad_var
        if isinstance(inpJ.type, NullType):
            return out.zeros_like(), inpJ
        if isinstance(inpJ.type, DisconnectedType):
            # since R_op does not have DisconnectedType yet, we will just
            # make them zeros.
            return out.zeros_like(), None
        else:
            return inpJ, None

    def __init__(
        self, inputs, outputs,
        inline=False,
        lop_overrides='default',
        grad_overrides='default',
        rop_overrides='default',
        connection_pattern=None,
        name=None, **kwargs
    ):
        if not isinstance(outputs, list):
            raise TypeError('outputs must be list, got %s' % type(outputs))
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
        if lop_overrides != 'default':
            if grad_overrides != 'default':
                raise ValueError('lop_overrides and grad_overrides are mutually exclusive')
            else:
                self.set_lop_overrides(lop_overrides)
                self._lop_type = 'lop'
        elif grad_overrides != 'default':
            self.set_lop_overrides(grad_overrides)
            self._lop_type = 'grad'
        else:
            self.set_lop_overrides('default')
            self._lop_type = 'lop'
        self.set_rop_overrides(rop_overrides)

        self._connection_pattern = connection_pattern

        if name is not None:
            assert isinstance(name, str), 'name must be None or string object'
        self.name = name

    def __eq__(self, other):
        # TODO: recognize a copy
        return self is other

    def __hash__(self):
        # TODO: use internal variables in hash
        return hash(type(self))

    def __str__(self):
        name = self.__class__.__name__ if self.name is None else self.name
        is_inline = self.is_inline
        return '%(name)s{inline=%(is_inline)s}' % locals()

    @theano.change_flags(compute_test_value='off')
    def _recompute_lop_op(self):
        '''
        converts self._lop_op from user supplied form to type(self) instance

        '''
        local_inputs = self.local_inputs
        local_outputs = self.local_outputs
        inp_len = len(local_inputs)
        lop_op = self._lop_op

        if isinstance(lop_op, OpFromGraph):
            if self._lop_op_is_cached:
                return
            assert self._lop_type in ['lop', 'grad'],\
                self.LOP_TYPE_ERR_MSG % self._lop_type
            if self._lop_type == 'grad':
                needed_ninps = inp_len + len(local_outputs)
                ninps = len(lop_op.local_inputs)
                if needed_ninps != ninps:
                    raise ValueError(
                        self.OV_INP_LEN_ERR_MSG % (needed_ninps, ninps))
                # make a wrapper callable

                def lop_op(inps, grads):
                    return self._lop_op(*(inps + grads))
            elif self._lop_type == 'lop':
                # OfG can be directly used in L_op format
                needed_ninps = inp_len + 2 * len(local_outputs)
                ninps = len(lop_op.local_inputs)
                if needed_ninps != ninps:
                    raise ValueError(
                        self.OV_INP_LEN_ERR_MSG % (needed_ninps, ninps))
                self._lop_op_is_cached = True
                self._lop_op_stypes_l = [None] * inp_len
                self._lop_op.kwargs['on_unused_input'] = 'ignore'
                return

        output_grads = [out_t() for out_t in self.output_types]
        fn_grad = partial(
            theano.gradient.grad,
            cost=None,
            disconnected_inputs='ignore',
            return_disconnected='Disconnected',
            null_gradients='return',
            known_grads=OrderedDict(izip(local_outputs, output_grads)))

        assert self._lop_type in ['lop', 'grad'],\
            self.LOP_TYPE_ERR_MSG % self._lop_type
        if self._lop_type == 'lop':
            callable_args = (local_inputs, local_outputs, output_grads)
        elif self._lop_type == 'grad':
            callable_args = (local_inputs, output_grads)

        # we need to convert _lop_op into an OfG instance
        if lop_op == 'default':
            gdefaults_l = fn_grad(wrt=local_inputs)
            all_grads_l, all_grads_ov_l = izip(
                *[OpFromGraph._filter_grad_var(grad, inp) for grad, inp in izip(gdefaults_l, local_inputs)])
            all_grads_l = list(all_grads_l)
            all_grads_ov_l = list(all_grads_ov_l)
        elif isinstance(lop_op, Variable):
            if isinstance(lop_op.type, (DisconnectedType, NullType)):
                all_grads_l = [inp.zeros_like() for inp in local_inputs]
                all_grads_ov_l = [lop_op.type() for _ in range(inp_len)]
            else:
                raise ValueError(self.STYPE_ERR_MSG % lop_op.type)
        elif isinstance(lop_op, list):
            goverrides_l = lop_op
            if len(goverrides_l) != inp_len:
                raise ValueError(
                    'Need to override %d gradients, got %d' % (
                        inp_len, len(goverrides_l)), goverrides_l)
            # compute non-overriding downsteam grads from upstreams grads
            # it's normal some input may be disconnected, thus the 'ignore'
            wrt_l = [lin for lin, gov in izip(
                local_inputs, goverrides_l) if gov == 'default']
            gdefaults = iter(fn_grad(wrt=wrt_l) if wrt_l else [])
            # combine overriding gradients
            all_grads_l = []
            all_grads_ov_l = []
            for inp, fn_gov in izip(local_inputs, goverrides_l):
                if fn_gov == 'default':
                    gnext, gnext_ov = OpFromGraph._filter_grad_var(
                        next(gdefaults), inp)
                    all_grads_l.append(gnext)
                    all_grads_ov_l.append(gnext_ov)
                elif isinstance(fn_gov, Variable):
                    if isinstance(fn_gov.type, (DisconnectedType, NullType)):
                        all_grads_l.append(inp.zeros_like())
                        all_grads_ov_l.append(fn_gov.type())
                    else:
                        raise ValueError(self.STYPE_ERR_MSG % fn_gov.type)
                else:
                    if not callable(fn_gov):
                        raise TypeError(self.TYPE_ERR_MSG % fn_gov)
                    gov, gov_ov = OpFromGraph._filter_grad_var(
                        fn_gov(*callable_args), inp)
                    all_grads_l.append(gov)
                    all_grads_ov_l.append(gov_ov)
        else:
            # callable case
            if not callable(lop_op):
                raise TypeError(self.TYPE_ERR_MSG % lop_op)
            goverrides_l = lop_op(*callable_args)
            if not isinstance(goverrides_l, list):
                raise TypeError(
                    'Gradient/L_op overriding function should return a list, '
                    'got "%s"' % type(goverrides_l))
            all_grads_l, all_grads_ov_l = izip(
                *[OpFromGraph._filter_grad_var(grad, inp)
                  for grad, inp in izip(goverrides_l, local_inputs)])
            if len(all_grads_l) != len(local_inputs):
                raise ValueError(
                    'Gradient/L_op overriding function should return list of '
                    '%d outputs, got %d' % (inp_len, len(all_grads_l)))
        all_grads_l = list(all_grads_l)
        all_grads_ov_l = list(all_grads_ov_l)
        self._lop_op = type(self)(
            inputs=local_inputs + local_outputs + output_grads,
            outputs=all_grads_l,
            inline=self.is_inline,
            name=(None if self.name is None else self.name + '_' + self._lop_type),
            on_unused_input='ignore')
        self._lop_op_stypes_l = all_grads_ov_l
        self._lop_op_is_cached = True
        self._lop_type = 'lop'

    @theano.change_flags(compute_test_value='off')
    def _recompute_rop_op(self):
        '''
        converts self._rop_op from user supplied form to type(self) instance

        '''
        local_inputs = self.local_inputs
        local_outputs = self.local_outputs
        out_len = len(local_outputs)
        rop_op = self._rop_op

        if isinstance(rop_op, OpFromGraph):
            if not self._rop_op_is_cached:
                self._rop_op_is_cached = True
                self._rop_op_stypes_l = [None] * out_len
            return

        eval_points = [inp_t() for inp_t in self.input_types]
        fn_rop = partial(
            theano.gradient.Rop,
            wrt=local_inputs,
            eval_points=eval_points)
        TYPE_ERR_MSG = ("R_op overrides should be (single or list of)"
                        "OpFromGraph | 'default' | None | 0 | callable, got %s")
        STYPE_ERR_MSG = ('Overriding Variable instance can only have type'
                         ' of DisconnectedType or NullType, got %s')
        if rop_op == 'default':
            rdefaults_l = fn_rop(f=local_outputs)
            all_rops_l, all_rops_ov_l = izip(
                *[OpFromGraph._filter_rop_var(rop, out) for rop,
                  out in izip(rdefaults_l, local_outputs)])
            all_rops_l = list(all_rops_l)
            all_rops_ov_l = list(all_rops_ov_l)
        elif isinstance(rop_op, Variable):
            if isinstance(rop_op.type, NullType):
                all_rops_l = [inp.zeros_like() for inp in local_inputs]
                all_rops_ov_l = [rop_op.type() for _ in range(out_len)]
            elif isinstance(rop_op.type, DisconnectedType):
                all_rops_l = [inp.zeros_like() for inp in local_inputs]
                all_rops_ov_l = [None] * out_len
            else:
                raise ValueError(STYPE_ERR_MSG % rop_op.type)
        elif isinstance(rop_op, list):
            roverrides_l = rop_op
            if len(roverrides_l) != out_len:
                raise ValueError(
                    'Need to override %d Rop, got %d' % (
                        out_len, len(roverrides_l)), roverrides_l)
            # get outputs that does not have Rop override
            odefaults_l = [
                lo for lo, rov in izip(local_outputs, roverrides_l)
                if rov == 'default']
            rdefaults_l = fn_rop(f=odefaults_l)
            rdefaults = iter(rdefaults_l if odefaults_l else [])
            # combine overriding Rops
            all_rops_l = []
            all_rops_ov_l = []
            for out, fn_rov in izip(local_outputs, roverrides_l):
                if fn_rov == 'default':
                    rnext, rnext_ov = OpFromGraph._filter_rop_var(
                        next(rdefaults), out)
                    all_rops_l.append(rnext)
                    all_rops_ov_l.append(rnext_ov)
                elif isinstance(fn_rov, Variable):
                    if isinstance(fn_rov.type, NullType):
                        all_rops_l.append(out.zeros_like())
                        all_rops_ov_l.append(fn_rov.type())
                    if isinstance(fn_rov.type, DisconnectedType):
                        all_rops_l.append(out.zeros_like())
                        all_rops_ov_l.append(None)
                    else:
                        raise ValueError(STYPE_ERR_MSG % fn_rov.type)
                else:
                    if not callable(fn_rov):
                        raise TypeError(TYPE_ERR_MSG % fn_rov)
                    rov, rov_ov = OpFromGraph._filter_rop_var(
                        fn_rov(local_inputs, eval_points), out)
                    all_rops_l.append(rov)
                    all_rops_ov_l.append(rov_ov)
        else:
            if not callable(rop_op):
                raise TypeError(TYPE_ERR_MSG % rop_op)
            roverrides_l = rop_op(local_inputs, eval_points)
            if not isinstance(roverrides_l, list):
                raise TypeError(
                    'Rop overriding function should return a list, '
                    'got "%s"' % type(roverrides_l))
            all_rops_l, all_rops_ov_l = izip(
                *[OpFromGraph._filter_rop_var(
                    rop, out) for rop, out in izip(roverrides_l, local_outputs)])
            if len(all_rops_l) != out_len:
                raise ValueError(
                    'Rop overriding function %s should return list of '
                    '%d outputs, got %d' % (
                        self._rop_op, out_len,
                        len(all_rops_l)), rop_op)
            all_rops_l = list(all_rops_l)
            all_rops_ov_l = list(all_rops_ov_l)
        self._rop_op = type(self)(
            inputs=local_inputs + eval_points,
            outputs=all_rops_l,
            inline=self.is_inline,
            name=(None if self.name is None else self.name + '_rop'),
            on_unused_input='ignore')
        self._rop_op_stypes_l = all_rops_ov_l
        self._rop_op_is_cached = True

    def get_lop_op(self):
        """
        getter method for self._lop_op
        """
        if not self._lop_op_is_cached:
            self._recompute_lop_op()
        return self._lop_op

    def get_rop_op(self):
        """
        getter method for self._rop_op
        """
        if not self._rop_op_is_cached:
            self._recompute_rop_op()
        return self._rop_op

    def set_grad_overrides(self, grad_overrides):
        """
        Set gradient overrides, see help(theano.OpFromGraph) for syntax
        This will completely remove any previously set L_op/gradient overrides

        """
        self._lop_op = grad_overrides
        self._lop_op_is_cached = False
        self._lop_type = 'grad'
        self._lop_is_default = (grad_overrides == 'default')

    def set_lop_overrides(self, lop_overrides):
        """
        Set L_op overrides, see help(theano.OpFromGraph) for syntax
        This will completely remove any previously set L_op/gradient overrides

        """
        self._lop_op = lop_overrides
        self._lop_op_is_cached = False
        self._lop_type = 'lop'
        self._lop_is_default = (lop_overrides == 'default')

    def set_rop_overrides(self, rop_overrides):
        """
        Set R_op overrides, see help(theano.OpFromGraph) for syntax
        This will completely remove any previously set R_op overrides

        """
        self._rop_op = rop_overrides
        self._rop_op_is_cached = False
        self._rop_is_default = (rop_overrides == 'default')

    def L_op(self, inputs, outputs, output_grads):
        if not self._lop_op_is_cached:
            self._recompute_lop_op()
        inps = list(inputs) + list(outputs) + list(output_grads)
        ret_ofg_l = self._lop_op(
            *inps, return_list=True)
        ret_l = [
            ret_ofg if ov is None else ov for ret_ofg, ov in izip(
                ret_ofg_l, self._lop_op_stypes_l)]
        return ret_l

    def R_op(self, inputs, eval_points):
        if not self._rop_op_is_cached:
            self._recompute_rop_op()
        ret_ofg_l = self._rop_op(
            *(list(inputs) + list(eval_points)), return_list=True)
        ret_l = [
            ret_ofg if ov is None else ov for ret_ofg, ov in izip(
                ret_ofg_l, self._rop_op_stypes_l)]
        return ret_l

    def make_node(self, *inputs):
        num_expected_inps = len(self.local_inputs) - len(self.shared_inputs)
        if len(inputs) != num_expected_inps:
            raise ValueError(
                "Expected %d inputs, got %d" % (num_expected_inps, len(inputs)))
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
        if self._connection_pattern is not None:
            return self._connection_pattern

        inp_len = len(self.local_inputs)
        out_len = len(self.local_outputs)
        cpmat_self = io_connection_pattern(
            self.local_inputs, self.local_outputs)

        lop_op = self.get_lop_op()
        cpmat_grad = io_connection_pattern(
            lop_op.local_inputs[inp_len:],
            lop_op.local_outputs)

        # cpmat_self |= cpmat_grad.T
        # cpmat_self &= out_is_disconnected
        for i, t in enumerate(self._lop_op_stypes_l):
            if t is not None:
                if isinstance(t.type, DisconnectedType):
                    for o in range(out_len):
                        cpmat_self[i][o] = False
            for o in range(out_len):
                cpmat_self[i][o] |= cpmat_grad[o][i]

        # TODO in case DisconnectedType is implemented for R_op,
        # self._rop_op_stypes_l self._rop_op should considered for
        # connection_pattern

        return list(map(list, cpmat_self))

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
