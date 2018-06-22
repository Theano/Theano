from __future__ import absolute_import, print_function, division
""" Tensor optimizations addressing the ops in basic.py.
"""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0

from collections import defaultdict
import logging
import itertools
import operator
import sys
import time
import traceback
import warnings

import numpy as np
from six import integer_types, iteritems
from six.moves import reduce, xrange

import theano
from theano import gof
from theano.compat import izip
from theano.gof import opt, InconsistencyError, TopoOptimizer, graph
from theano.gof import Variable, Constant
from theano.gof.opt import copy_stack_trace, in2out
from theano.gof.utils import MethodNotDefined
from theano.gradient import DisconnectedType
from theano import config
from theano.tensor.elemwise import Elemwise, DimShuffle
from theano.tensor.subtensor import (get_idx_list, get_canonical_form_slice,
                                     Subtensor, IncSubtensor, make_constant,
                                     AdvancedIncSubtensor1,
                                     AdvancedIncSubtensor,
                                     AdvancedSubtensor1,
                                     advanced_subtensor,
                                     advanced_subtensor1,
                                     advanced_inc_subtensor1)
from theano.tensor.sort import TopKOp
from theano import scalar
from theano.scalar import basic
from theano.tensor import basic as T
from theano import compile  # to register the optimizer built by this file
from theano.compile.ops import Shape, Shape_i
from theano.tensor.type import (values_eq_approx_remove_inf,
                                values_eq_approx_remove_nan,
                                values_eq_approx_remove_inf_nan)

from theano.gof.opt import (Optimizer, pre_constant_merge,
                            pre_greedy_local_optimizer)
from theano.gof import toolbox
from theano.tensor.basic import (Alloc, get_scalar_constant_value, ShapeError,
                                 extract_constant, NotScalarConstantError,
                                 Reshape)
from six import StringIO

_logger = logging.getLogger('theano.tensor.opt')

# Utilities


def _fill_chain(new_out, orig_inputs):
    for i in orig_inputs:
        new_out = T.fill(i, new_out)
    return [new_out]


def encompasses_broadcastable(b1, b2):
    """

    Parameters
    ----------
    b1
        The broadcastable attribute of a tensor type.
    b2
        The broadcastable attribute of a tensor type.

    Returns
    -------
    bool
        True if the broadcastable patterns b1 and b2 are such that b2 is
        broadcasted to b1's shape and not the opposite.

    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))


def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]


def scalarconsts_rest(inputs, elemwise=True, only_process_constants=False):
    """Partition a list of variables into two kinds:
    scalar constants, and the rest."""
    consts = []
    origconsts = []
    nonconsts = []
    for i in inputs:
        try:
            v = get_scalar_constant_value(i, elemwise=elemwise,
                                          only_process_constants=only_process_constants)
            consts.append(v)
            origconsts.append(i)
        except NotScalarConstantError:
            nonconsts.append(i)
    return consts, origconsts, nonconsts


def broadcast_like(value, template, fgraph, dtype=None):
    """
    Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be cast as
    necessary.

    """
    value = T.as_tensor_variable(value)
    if value.type == template.type:
        return value
    if template not in fgraph.variables:
        raise NotImplementedError('broadcast_like currently requires the '
                                  'template Variable to be in the fgraph already')
    if dtype is None:
        dtype = template.dtype
    value = T.cast(value, dtype)
    if value.type == template.type:
        return value
    if hasattr(fgraph, 'shape_feature'):
        new_shape = fgraph.shape_feature.shape_of[template]
    else:
        new_shape = template.shape
    rval = T.alloc(value, *new_shape)
    # the template may have 1s in its shape without being broadcastable
    if rval.broadcastable != template.broadcastable:
        rval = T.unbroadcast(rval, *[i for i in xrange(rval.ndim)
                                     if rval.broadcastable[i] and
                                     not template.broadcastable[i]])
    assert rval.type.dtype == dtype

    if rval.type.broadcastable != template.broadcastable:
        raise AssertionError("rval.type.broadcastable is " +
                             str(rval.type.broadcastable) +
                             " but template.broadcastable is" +
                             str(template.broadcastable))

    return rval


class InplaceElemwiseOptimizer(Optimizer):
    """
    We parametrise it to make it work for Elemwise and GpuElemwise op.
    """
    def __init__(self, OP):
        self.op = OP

    def add_requirements(self, fgraph):
        fgraph.attach_feature(theano.gof.destroyhandler.DestroyHandler())

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = ('    ' * level)
        print(blanc, "InplaceElemwiseOptimizer ", prof['opt'].op, file=stream)
        for k in ['node_before',
                  'nb_call_replace',
                  'nb_call_validate',
                  'nb_inconsistent']:
            print(blanc, k, prof[k], file=stream)
        ndim = prof['ndim']
        if ndim:
            print(blanc, "ndim", "nb", file=stream)
            for n in sorted(ndim.keys()):
                print(blanc, n, ndim[n], file=stream)

    def apply(self, fgraph):
        """
        Usage: InplaceElemwiseOptimizer(op).optimize(fgraph)

        Attempts to replace all Broadcast ops by versions of them
        that operate inplace. It operates greedily: for each Broadcast
        Op that is encountered, for each output, tries each input to
        see if it can operate inplace on that input. If so, makes the
        change and go to the next output or Broadcast Op.

        Examples
        --------

            `x + y + z -> x += y += z`

            `(x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)`

        """
        # We should not validate too often as this takes too much time to
        # execute!
        # It is the _dfs_toposort() fct in theano/gof/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe Theano should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest optimizer is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # We execute `validate` after this number of change.
        prof = {'opt': self,
                'node_before': len(fgraph.apply_nodes),
                'nb_call_replace': 0,
                'nb_call_validate': 0,
                'nb_inconsistent': 0,
                'ndim': defaultdict(lambda: 0)}

        check_each_change = config.tensor.insert_inplace_optimizer_validate_nb
        if check_each_change == -1:
            if len(fgraph.apply_nodes) > 500:
                check_each_change = 10
            else:
                check_each_change = 1

        nb_change_no_validate = 0
        chk = fgraph.checkpoint()

        if fgraph.update_mapping:
            update_outs = [fgraph.outputs[i] for i in fgraph.update_mapping]
        else:
            update_outs = []

        protected_inputs = [
            f.protected for f in fgraph._features if
            isinstance(f, theano.compile.function_module.Supervisor)]
        protected_inputs = sum(protected_inputs, [])  # flatten the list
        protected_inputs.extend(fgraph.outputs)
        for node in list(graph.io_toposort(fgraph.inputs, fgraph.outputs)):
            op = node.op
            # gpuarray GpuElemwise inherit from Elemwise
            if not type(op) == self.op:
                continue
            # If big graph and the outputs are scalar, do not make it
            # inplace.
            if (check_each_change != 1 and
                # If multiple outputs, they must all have the same size,
                # so only check the first.
                    getattr(node.outputs[0].type, 'ndim', -1) == 0):
                continue

            if op.inplace_pattern:
                # Maybe this isn't needed anymore, but I don't want to
                # rish regression now. This case only happen if the
                # original node add already some inplace patter and we
                # still try to add more pattern.

                baseline = op.inplace_pattern
                candidate_outputs = [i for i in xrange(len(node.outputs))
                                     if i not in baseline]
                # node inputs that are Constant, already destroyed,
                # or fgraph protected inputs and fgraph outputs can't be used as
                # inplace target.
                # Remove here as faster.
                candidate_inputs = [i for i in xrange(len(node.inputs))
                                    if i not in baseline.values() and
                                    not isinstance(node.inputs[i], Constant) and
                                    # the next line should not be costly most of the time.
                                    not fgraph.has_destroyers([node.inputs[i]]) and
                                    node.inputs[i] not in protected_inputs]
            else:
                baseline = []
                candidate_outputs = list(range(len(node.outputs)))
                # node inputs that are Constant, already destroyed,
                # fgraph protected inputs and fgraph outputs can't be used as inplace
                # target.
                # Remove here as faster.
                candidate_inputs = [i for i in xrange(len(node.inputs))
                                    if not isinstance(node.inputs[i], Constant) and
                                    not fgraph.has_destroyers([node.inputs[i]]) and
                                    node.inputs[i] not in protected_inputs]

            verbose = False

            raised_warning = not verbose

            for candidate_output in candidate_outputs:

                # If the output of the node can be established as an update
                # output of the fgraph, visit the candidate_inputs in an order
                # that will improve the chances of making the node operate
                # inplace on the input it's meant to update
                candidate_out_var = node.outputs[candidate_output]
                sorted_candidate_inputs = candidate_inputs

                if candidate_out_var in update_outs:

                    # The candidate output is an update. Sort the
                    # variables in candidate_inputs in the following order:
                    # - Vars corresponding to the actual updated input
                    #   (best case scenario is for the node that procudes
                    #   an update to operate inplace on the variable to
                    #   update)
                    # - Vars computed inplace on the updates input (second
                    #   best scenario if for the node to work inplace on
                    #   a variable obtained by a chain of inplace on the
                    #   variable to update. In some cases, this will be
                    #   equivalent to operating inplace on the variable to
                    #   update)
                    # - Remaining variables
                    updated_inputs = []
                    for i, f_out in enumerate(fgraph.outputs):
                        if (f_out is candidate_out_var and i in fgraph.update_mapping):
                            updated_inp_idx = fgraph.update_mapping[i]
                            updated_inputs.append(fgraph.inputs[updated_inp_idx])

                    updated_vars = []
                    vars_from_inplace = []
                    other_vars = []
                    for inp_idx in candidate_inputs:
                        inp = node.inputs[inp_idx]
                        if inp in updated_inputs:
                            # the candidate input is the actual updated input
                            updated_vars.append(inp_idx)
                        elif (hasattr(fgraph, 'destroy_handler') and
                              inp.owner and
                              any([fgraph.destroy_handler.root_destroyer.get(up_inp, None) is inp.owner
                                   for up_inp in updated_inputs])):

                            # the candidate input is a variable computed
                            # inplace on the updated input via a sequence of
                            # one or more inplace operations
                            vars_from_inplace.append(inp_idx)
                        else:
                            other_vars.append(inp_idx)

                    sorted_candidate_inputs = (updated_vars +
                                               vars_from_inplace + other_vars)

                for candidate_input in sorted_candidate_inputs:
                    # remove inputs that don't have the same dtype as the output
                    if node.inputs[candidate_input].type != node.outputs[
                            candidate_output].type:
                        continue

                    inplace_pattern = dict(baseline)
                    inplace_pattern[candidate_output] = candidate_input
                    try:
                        if hasattr(op.scalar_op, "make_new_inplace"):
                            new_scal = op.scalar_op.make_new_inplace(
                                scalar.transfer_type(
                                    *[inplace_pattern.get(i, o.dtype)
                                      for i, o in enumerate(node.outputs)]))
                        else:
                            new_scal = op.scalar_op.__class__(
                                scalar.transfer_type(
                                    *[inplace_pattern.get(i, None)
                                      for i in xrange(len(node.outputs))]))
                        new_outputs = self.op(new_scal, inplace_pattern)(
                            *node.inputs, **dict(return_list=True))
                        new_node = new_outputs[0].owner

                        for r, new_r in zip(node.outputs, new_outputs):
                            prof['nb_call_replace'] += 1
                            fgraph.replace(r, new_r,
                                           reason="inplace_elemwise_optimizer")
                        nb_change_no_validate += 1
                        prof['ndim'][candidate_out_var.ndim] += 1
                        if nb_change_no_validate >= check_each_change:
                            prof['nb_call_validate'] += 1
                            fgraph.validate()
                            chk = fgraph.checkpoint()
                            nb_change_no_validate = 0
                    except (ValueError, InconsistencyError) as e:
                        prof['nb_inconsistent'] += 1
                        if check_each_change != 1 and not raised_warning:
                            print(("Some inplace optimization was not "
                                   "performed due to unexpected error:"),
                                  file=sys.stderr)
                            print(e, file=sys.stderr)
                            raised_warning = True
                        fgraph.revert(chk)
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new_node
                    baseline = inplace_pattern
                    break

        if nb_change_no_validate > 0:
            try:
                fgraph.validate()
            except Exception:
                if not raised_warning:
                    print(("Some inplace optimization was not "
                           "performed due to unexpected error"),
                          file=sys.stderr)
                fgraph.revert(chk)
        return prof

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s (%s)" % (
            (' ' * level), self.__class__.__name__, self.op), file=stream)
        return inplace_elemwise_optimizer

inplace_elemwise_optimizer = InplaceElemwiseOptimizer(T.Elemwise)
compile.optdb.register('inplace_elemwise_opt', inplace_elemwise_optimizer, 75,
                       'inplace_opt',  # for historic reason
                       'inplace_elemwise_optimizer',
                       'fast_run', 'inplace')


def register_useless(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_useless(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = kwargs.pop('name', None) or lopt.__name__

        compile.mode.local_useless.register(name, lopt, 'last', 'fast_run',
                                            *tags, **kwargs)
        return lopt


def register_canonicalize(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_canonicalize(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = kwargs.pop('name', None) or lopt.__name__
        compile.optdb['canonicalize'].register(name, lopt, 'fast_run',
                                               *tags, **kwargs)
        return lopt


def register_stabilize(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_stabilize(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = kwargs.pop('name', None) or lopt.__name__
        compile.optdb['stabilize'].register(name, lopt, 'fast_run',
                                            *tags, **kwargs)
        return lopt


def register_specialize(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_specialize(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = kwargs.pop('name', None) or lopt.__name__
        compile.optdb['specialize'].register(name, lopt, 'fast_run',
                                             *tags, **kwargs)
        return lopt


def register_uncanonicalize(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_uncanonicalize(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = (kwargs and kwargs.pop('name', None)) or lopt.__name__
        compile.optdb['uncanonicalize'].register(name, lopt, 'fast_run', *tags,
                                                 **kwargs)
        return lopt


def register_specialize_device(lopt, *tags, **kwargs):
    if type(lopt) == str:
        def register(inner_lopt):
            return register_specialize_device(inner_lopt, lopt, *tags, **kwargs)
        return register
    else:
        name = (kwargs and kwargs.pop('name', None)) or lopt.__name__
        compile.optdb['specialize_device'].register(name, lopt, 'fast_run', *tags,
                                                    **kwargs)
        return lopt


#####################
# Dot optimizations #
#####################

@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.Dot])
def local_0_dot_x(node):
    if not isinstance(node.op, T.Dot):
        return False

    x = node.inputs[0]
    y = node.inputs[1]
    replace = False
    try:
        if get_scalar_constant_value(x, only_process_constants=True) == 0:
            replace = True
    except NotScalarConstantError:
        pass

    try:
        if get_scalar_constant_value(y, only_process_constants=True) == 0:
            replace = True
    except NotScalarConstantError:
        pass

    if replace:
        constant_zero = T.constant(0, dtype=node.outputs[0].type.dtype)
        if x.ndim == 2 and y.ndim == 2:
            constant_zero = assert_(constant_zero,
                                    T.eq(x.shape[1], y.shape[0]))
            return [T.alloc(constant_zero, x.shape[0], y.shape[1])]
        elif x.ndim == 1 and y.ndim == 2:
            constant_zero = assert_(constant_zero,
                                    T.eq(x.shape[0], y.shape[0]))
            return [T.alloc(constant_zero, y.shape[1])]
        elif x.ndim == 2 and y.ndim == 1:
            constant_zero = assert_(constant_zero,
                                    T.eq(x.shape[1], y.shape[0]))
            return [T.alloc(constant_zero, x.shape[0])]
        elif x.ndim == 1 and y.ndim == 1:
            constant_zero = assert_(constant_zero,
                                    T.eq(x.shape[0], y.shape[0]))
            return [constant_zero]
        else:
            _logger.warning("Optimization Warning: "
                            "Optimization theano/opt.py:local_0_dot_x Found "
                            "that it could apply, but was not implemented "
                            "for dot product with these input types:\n"
                            "(%s, %s)",
                            x.type, y.type)

######################
# DimShuffle lifters #
######################


def apply_local_dimshuffle_lift(var):
    # return var
    # lift recursively
    if not var.owner:
        return var
    new = local_dimshuffle_lift.transform(var.owner)
    if new:
        return new[0]
    return var


# Checks for two types of useless dimshuffles:
#   1 - dimshuffle all dimensions in order.
#   2 - dimshuffle a broadcastable dimension.
def is_dimshuffle_useless(new_order, input):
    is_useless = True
    if len(new_order) == input.type.ndim:
        all_broadcastable_dims = [i for (i, is_broadcastable)
                                  in enumerate(input.type.broadcastable)
                                  if is_broadcastable] + ['x']
        for i in range(input.type.ndim):
            if (new_order[i] == i or
                    (i in all_broadcastable_dims and
                     new_order[i] in all_broadcastable_dims)):
                is_useless = True
            else:
                is_useless = False
                break
    else:
        is_useless = False
    return is_useless


@gof.local_optimizer([DimShuffle])
def local_dimshuffle_lift(node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)
    DimShuffle{0,1,...}(x) => x (when the dimshuffle do nothing)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.

    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    input = node.inputs[0]
    inode = input.owner
    new_order = op.new_order
    if inode and isinstance(inode.op, Elemwise) and (len(input.clients) == 1):
        # Don't use make_node to have tag.test_value set.
        new_inputs = []
        for inp in inode.inputs:
            new_inp = op.__class__(inp.type.broadcastable,
                                   op.new_order)(inp)
            new_inputs.append(apply_local_dimshuffle_lift(new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, **dict(return_list=True))
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == 'x' and 'x' or inode.op.new_order[x] for x in
                     new_order]
        input = inode.inputs[0]

    if is_dimshuffle_useless(new_order, input):
        return [input]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = op.__class__(input.type.broadcastable, new_order)(input)
        ret = apply_local_dimshuffle_lift(ret)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@gof.local_optimizer([Reshape])
def local_useless_dimshuffle_in_reshape(node):
    """
    Removes useless DimShuffle operation inside Reshape:

      reshape(vector.dimshuffle('x', 0), shp) => reshape(vector, shp)
      reshape(matrix.dimshuffle('x', 0, 'x', 1), shp) => reshape(matrix, shp)
      reshape(row.dimshuffle(1, 'x'), shp) => reshape(row, shp)
      reshape(col.dimshuffle(0), shp) => reshape(col, shp)

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False
    if not (node.inputs[0].owner is not None and
            isinstance(node.inputs[0].owner.op, DimShuffle)):
        return False

    new_order = node.inputs[0].owner.op.new_order
    input = node.inputs[0].owner.inputs[0]
    broadcastables = node.inputs[0].broadcastable
    new_order_of_nonbroadcast = []
    for i, bd in zip(new_order, broadcastables):
        if not bd:
            new_order_of_nonbroadcast.append(i)
    no_change_in_order = all(
        new_order_of_nonbroadcast[i] <= new_order_of_nonbroadcast[i + 1]
        for i in xrange(len(new_order_of_nonbroadcast) - 1))
    if no_change_in_order:
        shape = node.inputs[1]
        ret = op.__class__(node.outputs[0].ndim)(input, shape)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@gof.local_optimizer([DimShuffle])
def local_lift_transpose_through_dot(node):
    """
    dot(x,y).T -> dot(y.T, x.T)

    These optimizations "lift" (propagate towards the inputs) DimShuffle
    through dot product.  It allows to put the graph in a more standard shape,
    and to later merge consecutive DimShuffles.

    The transformation should be apply whether or not the transpose is
    inplace.  The newly-introduced transpositions are not inplace, this will
    be taken care of in a later optimization phase.

    """
    if not (isinstance(node.op, T.DimShuffle) and node.op.new_order == (1, 0)):
        return False
    if not (node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, T.Dot)):
        return False
    x, y = node.inputs[0].owner.inputs

    if x.ndim == y.ndim == 2:
        # Output is dot product of transposed inputs in reverse order
        ret = [T.dot(y.T, x.T)]

        # Copy over stack trace to output from result of dot-product
        copy_stack_trace(node.inputs[0], ret)
        return ret

register_canonicalize(local_dimshuffle_lift)
register_specialize(local_dimshuffle_lift)

######################
# Casting operations #
######################


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.TensorFromScalar])
def local_tensor_scalar_tensor(node):
    '''tensor_from_scalar(scalar_from_tensor(x)) -> x'''
    if isinstance(node.op, T.TensorFromScalar):
        s = node.inputs[0]
        if s.owner and isinstance(s.owner.op, T.ScalarFromTensor):
            t = s.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [t]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.ScalarFromTensor])
def local_scalar_tensor_scalar(node):
    '''scalar_from_tensor(tensor_from_scalar(x)) -> x'''
    if isinstance(node.op, T.ScalarFromTensor):
        t = node.inputs[0]
        if t.owner and isinstance(t.owner.op, T.TensorFromScalar):
            s = t.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [s]

#####################################
# ShapeFeature, Shape optimizations
#####################################


class MakeVector(T.Op):
    """Concatenate a number of scalars together into a vector.

    This is a simple version of stack() that introduces far less cruft
    into the graph. Should work with 0 inputs. The constant_folding
    optimization will remove it.

    """

    __props__ = ("dtype",)

    def __init__(self, dtype='int64'):
        self.dtype = dtype

    def make_node(self, *inputs):
        inputs = list(map(T.as_tensor_variable, inputs))
        if (not all(a.type == inputs[0].type for a in inputs) or
                (len(inputs) > 0 and inputs[0].dtype != self.dtype)):
            dtype = theano.scalar.upcast(self.dtype, *[i.dtype for i in inputs])
            # upcast the input to the determined dtype,
            # but don't downcast anything
            assert dtype == self.dtype, (
                "The upcast of the inputs to MakeVector should match the "
                "dtype given in __init__.")
            if not all(self.dtype == T.cast(i, dtype=dtype).dtype
                       for i in inputs):
                raise TypeError("MakeVector.make_node expected inputs"
                                " upcastable to %s. got %s" %
                                (self.dtype, str([i.dtype for i in inputs])))
            inputs = [T.cast(i, dtype=dtype) for i in inputs]
        assert all(self.dtype == a.dtype for a in inputs)
        assert all(a.ndim == 0 for a in inputs)

        if inputs:
            dtype = inputs[0].type.dtype
        else:
            dtype = self.dtype
        # bcastable = (len(inputs) == 1)
        bcastable = False
        otype = T.TensorType(broadcastable=(bcastable,), dtype=dtype)
        return T.Apply(self, inputs, [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        # not calling theano._asarray as optimization
        if (out[0] is None) or (out[0].size != len(inputs)):
            out[0] = theano._asarray(inputs, dtype=node.outputs[0].dtype)
        else:
            # assume that out has correct dtype. there is no cheap way to check
            out[0][...] = inputs

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inp, out_, sub):
        out, = out_
        # Shouldn't use PyArray_TYPE(inp[0]) for the dtype
        # when len(inp) == 0 (we need to support this case.
        # So there will be (1 * nb_dtype) + ((nb len(inp) - 1 ))
        # different c code with the following algo
        out_shape = len(inp)
        out_num = np.dtype(node.outputs[0].dtype).num
        # don't use dtype_%(out)s as when check_input=False, it isn't defined.
        out_dtype = node.outputs[0].type.dtype_specs()[1]
        if len(inp) > 0:
            assert self.dtype == node.inputs[0].dtype
            out_num = 'PyArray_TYPE(%s)' % inp[0]

        ret = """
        npy_intp dims[1];
        dims[0] = %(out_shape)s;
        if(!%(out)s || PyArray_DIMS(%(out)s)[0] != %(out_shape)s){
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject*)PyArray_EMPTY(1, dims, %(out_num)s, 0);
        }
        """ % locals()
        for idx, i in enumerate(inp):
            ret += """
            *((%(out_dtype)s *)PyArray_GETPTR1(%(out)s, %(idx)s)) = *((%(out_dtype)s *) PyArray_DATA(%(i)s));
            """ % locals()
        return ret

    def infer_shape(self, node, ishapes):
        return [(len(ishapes),)]

    def grad(self, inputs, output_gradients):
        # If the output is of an integer dtype, no gradient shall pass
        if self.dtype in theano.tensor.discrete_dtypes:
            return [ipt.zeros_like().astype(theano.config.floatX)
                    for ipt in inputs]

        grads = []
        for i, inp in enumerate(inputs):
            grads.append(output_gradients[0][i])
        return grads

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

make_vector = MakeVector()


class MakeVectorPrinter:
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            old_precedence = getattr(pstate, 'precedence', None)
            try:
                pstate.precedence = 1000
                s = [pstate.pprinter.process(input)
                     for input in r.owner.inputs]
            finally:
                pstate.precedence = old_precedence
            return "[%s]" % ", ".join(s)
        else:
            raise TypeError("Can only print make_vector.")

T.pprint.assign(MakeVector, MakeVectorPrinter())


class ShapeFeature(object):
    """Graph optimizer for removing all calls to shape().

    This optimizer replaces all Shapes and Subtensors of Shapes with
    Shape_i and MakeVector Ops.

    This optimizer has several goals:

    1. to 'lift' Shapes to as close to the inputs as possible.

    2. to infer the shape of every node in the graph in terms of the
       input shapes.

    3. remove all fills (T.second, T.fill) from the graph

    Lifting shapes as close to the inputs as possible is important for
    canonicalization because it is very bad form to have to compute
    something just to know how big it will be.  Firstly, it is a waste
    of time to compute such outputs.  But it is important to get rid
    of these outputs as early as possible in the compilation process
    because the extra computations make it appear as if many internal
    graph nodes have multiple clients.  Many optimizations refuse to
    work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is
    present, or else using a conservative default.  An Op that
    supports shape-lifting should define a infer_shape(self, node,
    input_shapes) function.  The argument input_shapes is a tuple of
    tuples... there is an interior tuple for each input to the node.
    The tuple has as many elements as dimensions.  The element in
    position i of tuple j represents the i'th shape component of the
    j'th input.  The function should return a tuple of tuples.  One
    output tuple for each node.output.  Again, the i'th element of the
    j'th output tuple represents the output[j].shape[i] of the
    function.  If an output is not a TensorType, then None should be
    returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).

    Inferring the shape of internal nodes in the graph is important
    for doing size-driven optimizations.  If we know how big various
    intermediate results will be, we can estimate the cost of many Ops
    accurately, and generate c-code that is specific [e.g. unrolled]
    to particular sizes.

    In cases where you cannot figure out the shape, raise a ShapeError.

    Notes
    -----
    Right now there is only the ConvOp that could really take
    advantage of this shape inference, but it is worth it even
    just for the ConvOp.  All that's necessary to do shape
    inference is 1) to mark shared inputs as having a particular
    shape, either via a .tag or some similar hacking; and 2) to
    add an optional In() argument to promise that inputs will
    have a certain shape (or even to have certain shapes in
    certain dimensions). We can't automatically infer the shape of
    shared variables as they can change of shape during the
    execution by default.  (NOT IMPLEMENTED YET, BUT IS IN TRAC)


    **Using Shape information in Optimizations**

    To use this shape information in OPTIMIZATIONS, use the
    ``shape_of`` dictionary.

    For example:

    .. code-block:: python

        try:
            shape_of = node.fgraph.shape_feature.shape_of
        except AttributeError:
            # This can happen when the mode doesn't include the ShapeFeature.
            return

        shape_of_output_zero = shape_of[node.output[0]]

    The ``shape_of_output_zero`` symbol will contain a tuple, whose
    elements are either integers or symbolic integers.

    TODO: check to see if the symbols are necessarily
    non-constant... or are integer literals sometimes Theano
    constants?? That would be confusing.

    """
    def get_node_infer_shape(self, node):
        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(node,
                                   [self.shape_of[r] for r in node.inputs])
        except ShapeError:
            o_shapes = self.default_infer_shape(node, [self.shape_of[r] for
                                                       r in node.inputs])
        except NotImplementedError as e:
            raise NotImplementedError(
                'Code called by infer_shape failed raising a '
                'NotImplementedError. Raising NotImplementedError to '
                'indicate that a shape cannot be computed is no longer '
                'supported, and one should now use tensor.ShapeError '
                'instead. The original exception message is: %s' % e)
        except Exception as e:
            msg = ('Failed to infer_shape from Op %s.\nInput shapes: '
                   '%s\nException encountered during infer_shape: '
                   '%s\nException message: %s\nTraceback: %s') % (
                node.op, [self.shape_of[r] for r in node.inputs],
                type(e), str(e), traceback.format_exc())
            if config.on_shape_error == "raise":
                raise Exception(msg)
            else:
                _logger.warning(msg)
            o_shapes = self.default_infer_shape(
                node, [self.shape_of[r] for r in node.inputs])

        return o_shapes

    def get_shape(self, var, idx):
        """ Optimization can call this to get the current shape_i

        It is better to call this then use directly shape_of[var][idx]
        as this method should update shape_of if needed.

        TODO: Up to now, we don't update it in all cases. Update in all cases.
        """
        r = self.shape_of[var][idx]
        if (r.owner and
                isinstance(r.owner.op, Shape_i) and
                r.owner.inputs[0] not in var.fgraph.variables):
            assert var.owner
            node = var.owner
            # recur on inputs
            for i in node.inputs:
                if getattr(i, 'ndim', None) > 0:
                    self.get_shape(i, 0)
            o_shapes = self.get_node_infer_shape(node)
            assert len(o_shapes) == len(node.outputs)

            # Only change the variables and dimensions that would introduce
            # extra computation
            for new_shps, out in zip(o_shapes, node.outputs):
                if not hasattr(out, 'ndim'):
                    continue

                merged_shps = list(self.shape_of[out])
                changed = False
                for i in range(out.ndim):
                    n_r = merged_shps[i]
                    if (n_r.owner and
                            isinstance(n_r.owner.op, Shape_i) and
                            n_r.owner.inputs[0] not in var.fgraph.variables):
                        changed = True
                        merged_shps[i] = new_shps[i]
                if changed:
                    self.set_shape(out, merged_shps, override=True)
            r = self.shape_of[var][idx]
        return r

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i."""
        if hasattr(r.type, "broadcastable") and r.type.broadcastable[i]:
            return self.lscalar_one
        else:
            # Do not call make_node for test_value
            s = Shape_i(i)(r)
            try:
                s = get_scalar_constant_value(s)
            except NotScalarConstantError:
                pass
            return s

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r, 'ndim'):
            # This happen for NoneConst.
            return None
        return tuple([self.shape_ir(i, r) for i in xrange(r.ndim)])

    def default_infer_shape(self, node, i_shapes):
        """Return a list of shape tuple or None for the outputs of node.

        This function is used for Ops that don't implement infer_shape.
        Ops that do implement infer_shape should use the i_shapes parameter,
        but this default implementation ignores it.

        """
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i, var):
        """Return a symbolic integer scalar for the shape element s_i.

        The s_i argument was produced by the infer_shape() of an Op subclass.

        var: the variable that correspond to s_i. This is just for
        error reporting.

        """
        # unpack the s_i that the Op returned
        assert s_i is not None
        if s_i == 1:
            # don't make the optimizer merge a zillion ones together
            # by always returning the same object to represent 1
            return self.lscalar_one
        if type(s_i) is float and int(s_i) == s_i:
            s_i = int(s_i)
        if (type(s_i) in integer_types or
                isinstance(s_i, np.integer) or
                (isinstance(s_i, np.ndarray) and s_i.ndim == 0)):
            # this shape is a constant
            if s_i < 0:
                msg = "There is a negative shape in the graph!"
                msg += gof.utils.get_variable_trace_string(var)
                # The rest of the pipeline don't handle correctly this
                # case.  So we have 2 choices, stop compilation or
                # consider the shape as unknow.  As we have more
                # chance to give the stack trace here then later, I
                # choose that options as it would give better error
                # message.
                raise AssertionError(msg)
            return T.constant(s_i, dtype='int64')
        if type(s_i) in (tuple, list):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known,
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)

        # s_i is x.shape[i] for some x, we change it to shape_of[x][i]
        if (s_i.owner and
                isinstance(s_i.owner.op, Subtensor) and
                s_i.owner.inputs[0].owner and
                isinstance(s_i.owner.inputs[0].owner.op, T.Shape)):
            assert s_i.ndim == 0
            assert len(s_i.owner.op.idx_list) == 1

            # The current Subtensor always put constant index in the graph.
            # This was not True in the past. So call the Subtensor function
            # that will return the right index.
            idx = get_idx_list(s_i.owner.inputs, s_i.owner.op.idx_list)
            assert len(idx) == 1
            idx = idx[0]
            try:
                i = get_scalar_constant_value(idx)
            except NotScalarConstantError:
                pass
            else:
                # Executed only if no exception was raised
                x = s_i.owner.inputs[0].owner.inputs[0]
                # x should already have been imported, and should be in shape_of.
                s_i = self.shape_of[x][i]

        if s_i.type.dtype in theano.tensor.integer_dtypes:
            if getattr(s_i.type, 'ndim', 0):
                raise TypeError('Shape element must be scalar', s_i)
            return s_i
        else:
            raise TypeError('Unsupported shape element',
                            s_i, type(s_i), getattr(s_i, 'type', None))

    def set_shape(self, r, s, override=False):
        """Assign the shape `s` to previously un-shaped variable `r`.

        Parameters
        ----------
        r : a variable
        s : None or a tuple of symbolic integers
        override : If False, it mean r is a new object in the fgraph.
            If True, it mean r is already in the fgraph and we want to
            override its shape.

        """
        if not override:
            assert r not in self.shape_of, 'r already in shape_of'
        if s is None:
            self.shape_of[r] = s
        else:
            if not isinstance(s, (tuple, list)):
                raise TypeError('shapes must be tuple/list', (r, s))

            if r.ndim != len(s):
                sio = StringIO()
                theano.printing.debugprint(r, file=sio, print_type=True)
                raise AssertionError(
                    "Something inferred a shape with %d dimensions "
                    "for a variable with %d dimensions"
                    " for the variable:\n%s" % (
                        len(s), r.ndim, sio.getvalue()))

            shape_vars = []
            for i in xrange(r.ndim):
                if (hasattr(r.type, 'broadcastable') and
                        r.type.broadcastable[i]):
                    shape_vars.append(self.lscalar_one)
                else:
                    shape_vars.append(self.unpack(s[i], r))
            assert all([not hasattr(r.type, "broadcastable") or
                        not r.type.broadcastable[i] or
                        # The two following comparison are a speed optimization
                        # But we never timed this speed optimization!
                        self.lscalar_one.equals(shape_vars[i]) or
                        self.lscalar_one.equals(
                            T.extract_constant(shape_vars[i]))
                        for i in xrange(r.ndim)])
            self.shape_of[r] = tuple(shape_vars)
            for sv in shape_vars:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def update_shape(self, r, other_r):
        """Replace shape of r by shape of other_r.

        If, on some dimensions, the shape of other_r is not informative,
        keep the shape of r on those dimensions.

        """
        # other_r should already have a shape
        assert other_r in self.shape_of, ('other_r not in shape_of', other_r)
        other_shape = self.shape_of[other_r]

        # If other_shape has no information, call is pointless.
        if other_shape is None:
            return

        if r in self.shape_of:
            r_shape = self.shape_of[r]
        else:
            # If no info is known on r's shape, use other_shape
            self.set_shape(r, other_shape)
            return
        if (other_r.owner and r.owner and
                other_r.owner.inputs == r.owner.inputs and
                other_r.owner.op == r.owner.op):
            # We are doing a merge. So the 2 shapes graph will be the
            # same.  This is only a speed optimization to call
            # ancestors() less frequently.
            return

        # Merge other_shape with r_shape, giving the priority to other_shape
        merged_shape = []
        for i, ps in enumerate(other_shape):
            if r_shape is None and other_shape:
                merged_shape.append(other_shape[i])
            elif (ps.owner and
                    isinstance(getattr(ps.owner, 'op', None), Shape_i) and
                    ps.owner.op.i == i and
                    ps.owner.inputs[0] in (r, other_r)):
                # If other_shape[i] is uninformative, use r_shape[i].
                # For now, we consider 2 cases of uninformative other_shape[i]:
                #  - Shape_i(i)(other_r);
                #  - Shape_i(i)(r).
                merged_shape.append(r_shape[i])
            elif isinstance(r_shape[i], (Constant, integer_types)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(r_shape[i])
            elif isinstance(other_shape[i], (Constant, integer_types)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(other_shape[i])
            elif other_shape[i] == r_shape[i]:
                # This mean the shape is equivalent
                # We do not want to do the ancestor check in those cases
                merged_shape.append(r_shape[i])
            elif r_shape[i] in theano.gof.graph.ancestors([other_shape[i]]):
                # Another case where we want to use r_shape[i] is when
                # other_shape[i] actually depends on r_shape[i]. In that case,
                # we do not want to substitute an expression with another that
                # is strictly more complex. Such a substitution could also lead
                # to cycles: if (in the future) r_shape[i] gets replaced by an
                # expression of other_shape[i], other_shape[i] may end up
                # depending on itself.
                merged_shape.append(r_shape[i])
            else:
                merged_shape.append(other_shape[i])
        assert all([(not hasattr(r.type, "broadcastable") or
                     not r.type.broadcastable[i] and
                     not other_r.type.broadcastable[i]) or
                    # The two following comparison are a speed optimization
                    # But we never timed this speed optimization!
                    self.lscalar_one.equals(merged_shape[i]) or
                    self.lscalar_one.equals(
                        T.extract_constant(merged_shape[i], only_process_constants=True))
                    for i in xrange(r.ndim)])
        self.shape_of[r] = tuple(merged_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def set_shape_i(self, r, i, s_i):
        '''Replace element i of shape_of[r] by s_i'''
        assert r in self.shape_of
        prev_shape = self.shape_of[r]
        # prev_shape is a tuple, so we cannot change it inplace,
        # so we build another one.
        new_shape = []
        for j, s_j in enumerate(prev_shape):
            if j == i:
                new_shape.append(self.unpack(s_i, r))
            else:
                new_shape.append(s_j)
        assert all([not hasattr(r.type, "broadcastable") or
                    not r.type.broadcastable[idx] or
                    # The two following comparison are a speed optimization
                    # But we never timed this speed optimization!
                    self.lscalar_one.equals(new_shape[idx]) or
                    self.lscalar_one.equals(T.extract_constant(new_shape[idx]))
                    for idx in xrange(r.ndim)])
        self.shape_of[r] = tuple(new_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def init_r(self, r):
        '''Register r's shape in the shape_of dictionary.'''
        if r not in self.shape_of:
            try:
                self.set_shape(r, self.shape_tuple(r))
            except AttributeError:  # XXX: where would this come from?
                self.set_shape(r, None)

    def make_vector_shape(self, r):
        return make_vector(*self.shape_of[r])

    #
    # Feature interface
    #
    #
    def on_attach(self, fgraph):
        assert not hasattr(fgraph, 'shape_feature')
        fgraph.shape_feature = self
        # Must be local to the object as otherwise we reuse the same
        # variable for multiple fgraph!
        self.lscalar_one = T.constant(1, dtype='int64')
        assert self.lscalar_one.type == T.lscalar

        self.shape_of = {}
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)

        self.scheduled = {}
        # Variable ->

        self.shape_of_reverse_index = {}
        # shape var -> graph v

        for node in fgraph.toposort():
            self.on_import(fgraph, node, reason='on_attach')

    def on_detach(self, fgraph):
        self.shape_of = {}
        self.scheduled = {}
        self.shape_of_reverse_index = {}
        del fgraph.shape_feature

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        o_shapes = self.get_node_infer_shape(node)

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints
        if len(o_shapes) != len(node.outputs):
            raise Exception(
                ('The infer_shape method for the Op "%s" returned a list ' +
                 'with the wrong number of element: len(o_shapes) = %d ' +
                 ' != len(node.outputs) = %d') % (str(node.op),
                                                  len(o_shapes),
                                                  len(node.outputs)))

        # Ensure shapes are in 'int64'. This is to make sure the assert
        # found in the `local_useless_subtensor` optimization does not fail.
        for sh_idx, sh in enumerate(o_shapes):
            if sh is None:
                continue
            if not isinstance(sh, (list, tuple)):
                raise ValueError("infer_shape of %s didn't return a list of"
                                 " list. It returned '%s'" % (str(node), str(o_shapes)))
            new_shape = []
            for i, d in enumerate(sh):
                # Note: we ignore any shape element that is not typed (i.e.,
                # does not have a 'dtype' attribute). This means there may
                # still remain int elements that are int32 on 32-bit platforms,
                # but this works with `local_useless_subtensor`, so for now we
                # keep it this way. See #266 for a better long-term fix.
                if getattr(d, 'dtype', 'int64') != 'int64':
                    assert d.dtype in theano.tensor.discrete_dtypes, (node, d.dtype)
                    assert str(d.dtype) != 'uint64', node
                    new_shape += sh[len(new_shape):i + 1]
                    if isinstance(d, T.Constant):
                        casted_d = T.constant(d.data, dtype='int64')
                    else:
                        casted_d = theano.tensor.cast(d, 'int64')
                    new_shape[i] = casted_d
            if new_shape:
                # We replace the shape with wrong dtype by the one with
                # 'int64'.
                new_shape += sh[len(new_shape):]
                o_shapes[sh_idx] = tuple(new_shape)

        for r, s in izip(node.outputs, o_shapes):
            self.set_shape(r, s)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if new_r not in self.shape_of:
            # It happen that the fgraph didn't called on_import for some
            # new_r.  This happen when new_r don't have an
            # owner(i.e. it is a constant or an input of the graph)
            # update_shape suppose that r and new_r are in shape_of.
            self.init_r(new_r)

        # This tells us that r and new_r must have the same shape if
        # we didn't know that the shapes are related, now we do.
        self.update_shape(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to
        # replace the shape_i of r with the shape of new_r.  Say that
        # r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for (shpnode, idx) in (r.clients + [(node, i)]):
            if isinstance(getattr(shpnode, 'op', None), Shape_i):
                idx = shpnode.op.i
                repl = self.shape_of[new_r][idx]
                if repl.owner is shpnode:
                    # This mean the replacement shape object is
                    # exactly the same as the current shape object. So
                    # no need for replacement. This happen for example
                    # with the InputToGpuOptimizer optimizer.
                    continue
                if (repl.owner and
                        repl.owner.inputs[0] is shpnode.inputs[0] and
                        isinstance(repl.owner.op, Shape_i) and
                        repl.owner.op.i == shpnode.op.i):
                    # The replacement is a shape_i of the same
                    # input. So no need to do this equivalent
                    # replacement.
                    continue

                if shpnode.outputs[0] in theano.gof.graph.ancestors([repl]):
                    raise InconsistencyError(
                        "This substitution would insert a cycle in the graph:"
                        "node: %s, i: %i, r: %s, new_r: %s"
                        % (node, i, r, new_r))

                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update,
        # then we should cancel it.
        unscheduled = [k for k, v in self.scheduled.items() if v == r]
        for k in unscheduled:
            del self.scheduled[k]

        # In either case, r could be in shape_of.values(), that is, r itself
        # is the shape of  something. In that case, we want to update
        # the value in shape_of, to keep it up-to-date.
        for v in self.shape_of_reverse_index.get(r, []):
            # The reverse index is only approximate. It is not updated on
            # deletion of variables, or on change_input so it might be the
            # case that there are a few extra `v`'s in it that no longer have
            # a shape of r or possibly have been deleted from shape_of
            # entirely. The important thing is that it permits to recall
            # all variables with r in their shape.
            for ii, svi in enumerate(self.shape_of.get(v, [])):
                if svi == r:
                    self.set_shape_i(v, ii, new_r)
        self.shape_of_reverse_index[r] = set()

    def same_shape(self, x, y, dim_x=None, dim_y=None):
        """Return True if we are able to assert that x and y have the
        same shape.

        dim_x and dim_y are optional. If used, they should be an index
        to compare only 1 dimension of x and y.

        """
        sx = self.shape_of[x]
        sy = self.shape_of[y]
        if sx is None or sy is None:
            return False
        if dim_x is not None:
            sx = [sx[dim_x]]
        if dim_y is not None:
            sy = [sy[dim_y]]
        assert len(sx) == len(sy)

        # We look on each dimensions we want to compare.
        # If any of them can't be asserted to be equal, return False.
        # Otherwise, we return True at the end.
        for dx, dy in zip(sx, sy):
            if dx is dy:
                continue
            # Need to try to find that they are the same shape. We
            # need to compare the full graph. It could be slow. So I
            # just implement for now the case of Shape_i.
            if not dx.owner or not dy.owner:
                return False
            if (not isinstance(dx.owner.op, Shape_i) or
                    not isinstance(dy.owner.op, Shape_i)):
                return False
            opx = dx.owner.op
            opy = dy.owner.op
            if not (opx.i == opy.i):
                return False
            # FB I'm not sure if this handle correctly constants.
            if dx.owner.inputs[0] == dy.owner.inputs[0]:
                continue
            # To be sure to cover all case, call equal_computation.
            # Can't use theano.gof.graph.is_same_graph(dx, dy)
            # As it currently expect that dx and dy aren't in a FunctionGraph
            from theano.scan_module.scan_utils import equal_computations
            if not equal_computations([dx], [dy]):
                return False
        return True


class ShapeOptimizer(Optimizer):
    """Optimizer that serves to add ShapeFeature as an fgraph feature."""
    def add_requirements(self, fgraph):
        fgraph.attach_feature(ShapeFeature())

    def apply(self, fgraph):
        pass


class UnShapeOptimizer(Optimizer):
    """Optimizer remove ShapeFeature as an fgraph feature."""
    def apply(self, fgraph):
        for feature in fgraph._features:
            if isinstance(feature, ShapeFeature):
                fgraph.remove_feature(feature)

# Register it after merge1 optimization at 0. We don't want to track
# the shape of merged node.
theano.compile.mode.optdb.register('ShapeOpt', ShapeOptimizer(),
                                   0.1, 'fast_run', 'fast_compile')
# Not enabled by default for now. Some crossentropy opt use the
# shape_feature.  They are at step 2.01. uncanonicalize is at step
# 3. After it goes to 48.5 that move to the gpu. So 10 seem resonable.
theano.compile.mode.optdb.register('UnShapeOpt', UnShapeOptimizer(),
                                   10)


def local_elemwise_alloc_op(ElemwiseOP, AllocOP, DimShuffleOP):
    def local_elemwise_alloc(node):
        """
        elemwise(alloc(x, shp), ..., y.TensorType(BROADCAST CONDITION))
          -> elemwise(x, y.TensorType(BROADCAST CONDITION))

        elemwise(dimshuffle(alloc(x, shp)),... ,y.TensorType(BROADCAST CONDITION))
          -> elemwise(x.dimshuffle(...), y.TensorType(BROADCAST CONDITION))

        BROADCAST CONDITION: the condition is that the one input that are
        not to be optimized to have the same broadcast pattern as the
        output.

        We can change the alloc by a dimshuffle as the elemwise
        already have the shape info.  The dimshuffle will be faster
        to exec.

        """
        if not isinstance(node.op, ElemwiseOP):
            return False

        if len(node.outputs) > 1:
            # Ensure all outputs have the same broadcast pattern
            # This is a supposition that I'm not sure is always true.
            assert all([o.type.broadcastable ==
                        node.outputs[0].type.broadcastable for o in
                        node.outputs[1:]])

        # The broadcast pattern of the ouptut must match the broadcast
        # pattern of at least one of the inputs.
        if not any([i.type.broadcastable ==
                    node.outputs[0].type.broadcastable for i in node.inputs]):
            return False

        def dimshuffled_alloc(i):
            return (isinstance(i.owner.op, DimShuffleOP) and
                    i.owner.inputs[0].owner and
                    isinstance(i.owner.inputs[0].owner.op, AllocOP))

        # At least one input must have an owner that is either a AllocOP or a
        # DimShuffleOP with an owner that is a AllocOP -- otherwise there is
        # nothing to optimize.
        if not any([i.owner and (isinstance(i.owner.op, AllocOP) or
                                 dimshuffled_alloc(i)) for i in node.inputs]):
            return False

        # Search for input that we can use as a baseline for the dimensions.
        assert_op_idx = -1
        for idx, i in enumerate(node.inputs):
            if i.type.broadcastable == node.outputs[0].type.broadcastable:
                # Prefer an input that is not a AllocOP nor a DimShuffleOP of a
                # AllocOP so that all allocs can be optimized.
                if not (i.owner and (isinstance(i.owner.op, AllocOP) or
                        dimshuffled_alloc(i))):
                    assert_op_idx = idx
                    break

        # It may be the case that only AllocOP and DimShuffleOP of AllocOP exist.
        if assert_op_idx < 0:
            # We want to optimize as many allocs as possible. When
            # there is more than one then do all but one.  number of
            # inputs with alloc or dimshuffle alloc
            l2 = [i for i in node.inputs
                  if (i.owner and (isinstance(i.owner.op, AllocOP) or
                      dimshuffled_alloc(i)))]
            # If only 1 alloc or dimshuffle alloc, it is the one we
            # will use for the shape. So no alloc would be removed.
            if len(l2) > 1:
                # l containt inputs with alloc or dimshuffle alloc
                # only.  Its length will always be at least one, as we
                # checked that before
                l = [idx for idx, i in enumerate(node.inputs)
                     if i.broadcastable == node.outputs[0].broadcastable]
                assert_op_idx = l[0]  # The first one is as good as any to use.
            else:
                # Nothing would be optimized!
                return False

        assert_op = node.inputs[assert_op_idx]
        cmp_op = assert_op
        new_i = []
        same_shape = node.fgraph.shape_feature.same_shape
        for i in node.inputs:
            # Remove alloc
            if (i.owner and isinstance(i.owner.op, AllocOP) and
                    i.owner.inputs[0].type != i.owner.outputs[0].type):
                # when i.owner.inputs[0].type == i.owner.outputs[0].type we
                # will remove that alloc later
                assert i.type.ndim == cmp_op.ndim
                if theano.config.experimental.local_alloc_elemwise_assert:
                    get_shape = node.fgraph.shape_feature.get_shape
                    cond = []
                    for idx in xrange(i.type.ndim):
                        if (not i.type.broadcastable[idx] and
                                not same_shape(i, cmp_op, idx, idx)):
                            i_shp = get_shape(i, idx)
                            cmp_shp = get_shape(cmp_op, idx)
                            cond.append(T.eq(i_shp, cmp_shp))
                    if cond:
                        assert_op = assert_(assert_op, *cond)
                new_i.append(i.owner.inputs[0])

            # Remove Alloc in DimShuffle
            elif i.owner and dimshuffled_alloc(i):
                assert i.type.ndim == cmp_op.type.ndim
                if theano.config.experimental.local_alloc_elemwise_assert:
                    assert_cond = [T.eq(i.shape[idx], cmp_op.shape[idx])
                                   for idx in xrange(i.type.ndim)
                                   if not i.type.broadcastable[idx] and
                                   not same_shape(i, cmp_op, idx, idx)]
                    if assert_cond:
                        assert_op = assert_(assert_op, *assert_cond)
                alloc_input = i.owner.inputs[0].owner.inputs[0]
                if alloc_input.ndim != i.owner.inputs[0].ndim:
                    # The alloc can add dimension to the value
                    # We add a dimshuffle to add them.
                    # We let later optimization merge the multiple dimshuffle
                    nb_dim_to_add = i.owner.inputs[0].ndim - alloc_input.ndim
                    alloc_input = alloc_input.dimshuffle(
                        ['x'] * nb_dim_to_add +
                        list(range(alloc_input.ndim)))

                # We need to keep the dimshuffle. It could swap axes or
                # add dimensions anywhere.
                r_i = i.owner.op(alloc_input)

                # Copy stack trace from i to new_i
                copy_stack_trace(i, r_i)
                new_i.append(r_i)
            else:
                new_i.append(i)
        new_i[assert_op_idx] = assert_op

        ret = node.op(*new_i, return_list=True)

        # Copy over stack trace from previous outputs to new outputs.
        copy_stack_trace(node.outputs, ret)
        return ret

    return local_elemwise_alloc

# TODO, global optimizer that lift the assert to the beginning of the graph.
# TODO, optimize all inputs when possible -- currently when all inputs have
# an alloc all but one is optimized.

local_elemwise_alloc = register_specialize(
    gof.local_optimizer([T.Elemwise])(
        local_elemwise_alloc_op(T.Elemwise, T.Alloc, T.DimShuffle)),
    'local_alloc_elemwise')


@gof.local_optimizer([T.Elemwise])
def local_fill_sink(node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(c, fill(a, f(b, d, e)))
    f need to be an elemwise that isn't a fill.
    """
    if (not hasattr(node, 'op') or
            not isinstance(node.op, T.Elemwise) or
            node.op == T.fill):
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == T.fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if not models:
        return False
    c = node.op(*inputs)
    for model in models:
        if model.type != c.type:
            c = T.fill(model, c)

    # The newly created node c doesn't has 'clients',
    # so this iteration is took place with node.outputs[0]
    replacements = {node.outputs[0]: c}
    for client, cl_idx in node.outputs[0].clients:
        if (hasattr(client, 'op') and
                isinstance(client.op, T.Elemwise) and
                not client.op == T.fill):
            client_inputs = client.inputs[:]
            client_inputs[cl_idx] = c
            new_client = client.op(*client_inputs)

            # Add clients to new_client
            new_client.owner.outputs[0].clients = client.outputs[0].clients
            r = local_fill_sink.transform(new_client.owner)
            if not r:
                continue
            replacements.update(r)
    return replacements

register_canonicalize(local_fill_sink)


@register_specialize
@register_stabilize
# @register_canonicalize  # We make full pass after the canonizer phase.
@gof.local_optimizer([T.fill])
def local_fill_to_alloc(node):
    """fill(s,v) -> alloc(v, shape(s))

    This is an important optimization because with the shape_to_shape_i
    optimization, the dependency on 's' is often removed.

    """
    if node.op == T.fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            rval = [v]
        elif v.type.broadcastable == node.outputs[0].type.broadcastable:
            # this is a cast
            rval = [T.cast(v, node.outputs[0].type.dtype)]
        elif r.type.broadcastable == node.outputs[0].type.broadcastable:
            # we are broadcasting v somehow, but not r
            o = broadcast_like(v, r, node.fgraph, dtype=v.dtype)
            copy_stack_trace(node.outputs[0], o)
            rval = [o]
        else:
            # we are broadcasting both v and r,
            # the output shape must be computed
            #
            # TODO: implement this case (including a test!)
            #
            #  I think the strategy should be to extend the shorter
            #  shape vector with 1s (how?) and then take the
            #  elementwise max of the two.  - how to flag an error of
            #  shape mismatch where broadcasting should be illegal?
            return
            # TODO: cut out un-necessary dimshuffles of v

        assert rval[0].type == node.outputs[0].type, (
            'rval', rval[0].type, 'orig', node.outputs[0].type, 'node',
            node,)  # theano.printing.debugprint(node.outputs[0], file='str'))
        return rval

# Register this after stabilize at 1.5 to make sure stabilize don't
# get affected by less canonicalized graph due to alloc.
compile.optdb.register('local_fill_to_alloc',
                       in2out(local_fill_to_alloc),
                       1.51, 'fast_run')
# Needed to clean some extra alloc added by local_fill_to_alloc
compile.optdb.register('local_elemwise_alloc',
                       in2out(local_elemwise_alloc),
                       1.52, 'fast_run')


@register_canonicalize("fast_compile")
@register_useless
@gof.local_optimizer([T.fill])
def local_useless_fill(node):
    """fill(s,v) -> v

    This optimization is only needed in FAST_COMPILE to make the code
    more readable. Normally, it is done by the local_fill_to_alloc
    opt.

    """
    if node.op == T.fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            # also, we don't need to copy over any stack traces here
            return [v]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@gof.local_optimizer([T.alloc])
def local_useless_alloc(node):
    """
    If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.

    """
    op = node.op
    if not isinstance(op, Alloc):
        return False

    input = node.inputs[0]
    output = node.outputs[0]

    # Check if dtype and broadcast remain the same.
    if input.type == output.type:
        # We don't need to copy over any stack traces here
        return [input]


@register_specialize
@register_stabilize
@register_canonicalize
@gof.local_optimizer([T.alloc])
def local_canonicalize_alloc(node):
    """If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed. (as local_useless_alloc)

    Also, it will canonicalize alloc by creating Dimshuffle after the
    alloc to introduce the dimensions of constant size 1.

    See https://github.com/Theano/Theano/issues/4072 to know why this
    is needed.

    """
    op = node.op
    if not isinstance(op, Alloc):
        return False

    input = node.inputs[0]
    output = node.outputs[0]

    # Check if dtype and broadcast remain the same.
    if input.type == output.type:
        # We don't need to copy over any stack traces here
        return [input]

    # Allow local_merge_alloc to do its work first
    clients = getattr(output, 'clients', [])
    for client, i in clients:
        if client != "output" and isinstance(client.op, Alloc):
            return

    # Check if alloc adds a broadcastable dimension with shape 1.

    output_shape = node.inputs[1:]
    num_dims_with_size_1_added_to_left = 0
    for i in range(len(output_shape) - input.ndim):
        if extract_constant(output_shape[i], only_process_constants=True) == 1:
            num_dims_with_size_1_added_to_left += 1
        else:
            break
    new_output_shape = output_shape[num_dims_with_size_1_added_to_left:]
    if num_dims_with_size_1_added_to_left > 0 and len(new_output_shape) >= input.ndim:
        if output.broadcastable[num_dims_with_size_1_added_to_left:] == input.broadcastable:
            inner = input
        else:
            inner = op(*([input] + new_output_shape))
        dimshuffle_new_order = (['x'] * num_dims_with_size_1_added_to_left +
                                list(xrange(len(new_output_shape))))
        return [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]


# Don't register by default.
@gof.local_optimizer([T.AllocEmpty])
def local_alloc_empty_to_zeros(node):
    """This convert AllocEmpty to Alloc of 0.

    This help investigate NaN with NanGuardMode.  Not registered by
    default. To activate it, use the Theano flag
    optimizer_including=alloc_empty_to_zeros. This also enable
    the GPU version of this optimizations.

    """
    if isinstance(node.op, T.AllocEmpty):
        return [T.zeros(node.inputs, dtype=node.outputs[0].dtype)]
compile.optdb.register('local_alloc_empty_to_zeros',
                       in2out(local_alloc_empty_to_zeros),
                       # After move to gpu and merge2, before inplace.
                       49.3,
                       'alloc_empty_to_zeros',)


@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Shape])
def local_shape_to_shape_i(node):
    if node.op == T.shape:
        # This optimization needs ShapeOpt and fgraph.shape_feature
        if not hasattr(node.fgraph, 'shape_feature'):
            return
        shape_feature = node.fgraph.shape_feature
        ret = shape_feature.make_vector_shape(node.inputs[0])

        # We need to copy over stack trace from input to output
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


# TODO: Not sure what type of node we are expecting here
@register_specialize
@register_canonicalize
@gof.local_optimizer(None)
def local_track_shape_i(node):
    try:
        shape_feature = node.fgraph.shape_feature
    except AttributeError:
        return
    if node in shape_feature.scheduled:
        # Don't unschedule node as it could be reinserted in the
        # fgraph as we don't change it in the shapefeature internal
        # structure.
        assert isinstance(node.op, Shape_i)
        replacement = shape_feature.scheduled[node]
        return [shape_feature.shape_of[replacement][node.op.i]]


@register_specialize
@register_canonicalize
@gof.local_optimizer([Subtensor])
def local_subtensor_inc_subtensor(node):
    """
    Subtensor(SetSubtensor(x, y, idx), idx) -> y

    """
    if isinstance(node.op, Subtensor):
        x = node.inputs[0]
        if not x.owner or not isinstance(x.owner.op, IncSubtensor):
            return
        if not x.owner.op.set_instead_of_inc:
            return

        if (x.owner.inputs[2:] == node.inputs[1:] and
                tuple(x.owner.op.idx_list) == tuple(node.op.idx_list)):
            out = node.outputs[0]
            y = x.owner.inputs[1]
            # If the dtypes differ, cast y into x.dtype
            if x.dtype != y.dtype:
                y = y.astype(x.dtype)
            if out.type == y.type:
                # if x[idx] and y have the same type, directly return y
                return [y]
            else:
                # The difference is related to broadcasting pattern
                assert out.broadcastable != y.broadcastable
                # We have to alloc y to the shape of x[idx]
                x_subtensor = node.op(x.owner.inputs[0], *x.owner.inputs[2:])
                return [T.alloc(y, *x_subtensor.shape)]
        else:
            return


@register_specialize
@register_canonicalize
@gof.local_optimizer([Subtensor])
def local_subtensor_remove_broadcastable_index(node):
    """
    Remove broadcastable dimension with index 0 or -1
    a[:,:,:,0] -> a.dimshuffle(0,1,2), when
        a.broadcastable = (False, False, False, True)
    a[0,:,-1,:] -> a.dimshuffle(1,3), when
        a.broadcastable = (True, False, True, False)

    """
    if isinstance(node.op, Subtensor):
        idx = node.op.idx_list
    else:
        return

    remove_dim = []
    node_inputs_idx = 1
    for dim, elem in enumerate(idx):
        if isinstance(elem, (scalar.Scalar)):
            # The idx is a Scalar, ie a Type. This means the actual index
            # is contained in node.inputs[1]
            dim_index = node.inputs[node_inputs_idx]
            if type(dim_index) == theano.scalar.basic.ScalarConstant:
                dim_index = dim_index.value
            if dim_index in [0, -1] and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
                node_inputs_idx += 1
            else:
                return
        elif isinstance(elem, slice):
            if elem != slice(None):
                return
        elif isinstance(elem, (integer_types, np.integer)):
            if elem in [0, -1] and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
        else:
            raise TypeError('case not expected')

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_specialize
@register_canonicalize('fast_compile_gpu')
@register_useless
@gof.local_optimizer([Subtensor, AdvancedSubtensor1])
def local_subtensor_make_vector(node):
    """
    Replace all subtensor(make_vector) like:
    [a,b,c][0] -> a
    [a,b,c][0:2] -> [a,b]

    Replace all AdvancedSubtensor1(make_vector) like:
    [a,b,c][[0,2]] -> [a,c]

    We can do this for constant indexes.

    """
    x = node.inputs[0]
    if not x.owner or x.owner.op != make_vector:
        return

    if isinstance(node.op, Subtensor):
        # This optimization needs ShapeOpt and fgraph.shape_feature
        try:
            idx, = node.op.idx_list
        except Exception:
            # 'how can you have multiple indexes into a shape?'
            raise

        if isinstance(idx, (scalar.Scalar, T.TensorType)):
            # The idx is a Scalar, ie a Type. This means the actual index
            # is contained in node.inputs[1]
            old_idx, idx = idx, node.inputs[1]
            assert idx.type == old_idx
    elif isinstance(node.op, AdvancedSubtensor1):
        idx = node.inputs[1]
    else:
        return

    if isinstance(idx, (integer_types, np.integer)):
        # We don't need to copy over any stack traces here
        return [x.owner.inputs[idx]]
    elif isinstance(idx, Variable):
        if idx.ndim == 0:
            # if it is a constant we can do something with it
            try:
                v = get_scalar_constant_value(idx, only_process_constants=True)
                if isinstance(v, np.integer):
                    # Python 2.4 wants to index only with Python integers
                    v = int(v)
                # We don't need to copy over any stack traces here
                try:
                    ret = [x.owner.inputs[v]]
                except IndexError:
                    raise NotScalarConstantError("Bad user graph!")
                return ret
            except NotScalarConstantError:
                pass
        elif idx.ndim == 1 and isinstance(idx, T.Constant):
            values = list(map(int, list(idx.value)))
            ret = make_vector(*[x.owner.inputs[v] for v in values])

            # Copy over stack trace from previous output to new output
            copy_stack_trace(node.outputs[0], ret)
            ret = T.patternbroadcast(ret, node.outputs[0].broadcastable)
            return [ret]
        else:
            raise TypeError('case not expected')
    elif isinstance(idx, slice):
        # it is a slice of ints and/or Variables
        # check subtensor to see if it can contain constant variables, and if
        # it can, then try to unpack them.
        try:
            const_slice = node.op.get_constant_idx(node.inputs,
                                                   allow_partial=False)[0]
            ret = make_vector(*x.owner.inputs[const_slice])
            # Copy over stack trace from previous outputs to new output
            copy_stack_trace(node.outputs, ret)
            ret = T.patternbroadcast(ret, node.outputs[0].broadcastable)
            return [ret]
        except NotScalarConstantError:
            pass
    else:
        raise TypeError('case not expected')


# TODO: the other optimization for and, or, xor, le and ge see ticket #496.

@register_useless
@register_canonicalize('fast_compile')
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_useless_elemwise(node):
    """
    eq(x, x) -> 1
    neq(x, x) -> 0
    mul(x) -> x
    add(x) -> x
    identity(x) -> x
    and(x, 1) -> x  (if x.dtype == 'bool')
    and(x, 0) -> zeros_like(x)
    or(x, 0) -> x
    or(x, 1) -> ones_like(x)  (if x.dtype == 'bool')
    xor(x, x) -> zeros_like(x)

    """
    if isinstance(node.op, T.Elemwise):
        # We call zeros_like and one_like with opt=True to generate a
        # cleaner graph.
        dtype = node.outputs[0].dtype

        if node.op.scalar_op == theano.scalar.eq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be true
                ret = T.ones_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]
        elif node.op.scalar_op == theano.scalar.neq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be false
                ret = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]

        elif node.op.scalar_op == theano.scalar.mul and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]

        elif node.op.scalar_op == theano.scalar.add and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]
        elif (node.op.scalar_op == theano.scalar.identity and
              len(node.inputs) == 1):
            return [node.inputs[0]]

        elif (isinstance(node.op.scalar_op, scalar.AND) and
              len(node.inputs) == 2):

            if isinstance(node.inputs[0], T.TensorConstant):
                const_val = T.extract_constant(node.inputs[0], only_process_constants=True)
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [T.zeros_like(node.inputs[1], dtype=dtype,
                                             opt=True)]
                    elif node.outputs[0].dtype == 'bool':
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[1].astype(node.outputs[0].dtype)]

            if isinstance(node.inputs[1], T.TensorConstant):
                const_val = T.extract_constant(node.inputs[1], only_process_constants=True)
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [T.zeros_like(node.inputs[0], dtype=dtype,
                                             opt=True)]
                    elif node.outputs[0].dtype == 'bool':
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[0].astype(node.outputs[0].dtype)]

        elif (isinstance(node.op.scalar_op, scalar.OR) and
              len(node.inputs) == 2):

            if isinstance(node.inputs[0], T.TensorConstant):
                const_val = T.extract_constant(node.inputs[0], only_process_constants=True)
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[1].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == 'bool':
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [T.ones_like(node.inputs[1], dtype=dtype,
                                            opt=True)]

            if isinstance(node.inputs[1], T.TensorConstant):
                const_val = T.extract_constant(node.inputs[1], only_process_constants=True)
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[0].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == 'bool':
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [T.ones_like(node.inputs[0], dtype=dtype,
                                            opt=True)]

        elif (isinstance(node.op.scalar_op, scalar.XOR) and
              len(node.inputs) == 2):
            if node.inputs[0] is node.inputs[1]:
                return [T.zeros_like(node.inputs[0], dtype=dtype, opt=True)]


@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_alloc_unary(node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)"""
    if isinstance(node.op, T.Elemwise) and len(node.inputs) == 1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, T.Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            # T.alloc does not preserve the stacktrace of v,
            # so we need to copy it over from x.
            copy_stack_trace(node.outputs[0], v)
            ret = T.alloc(T.cast(v, node.outputs[0].dtype), *shp)

            # T.cast does not preserve the stacktrace of x,
            # so we need to copy it over to the output.
            copy_stack_trace([node.outputs[0], a], ret)
            return [ret]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_cast_cast(node):
    """cast(cast(x, dtype1), dtype2)

    when those contrain:
    dtype1 == dtype2
    OR the base dtype is the same (int, uint, float, complex)
          and the first cast cause an upcast.

    """
    if (not isinstance(node.op, T.Elemwise) or
            not isinstance(node.op.scalar_op, scalar.Cast)):
        return
    x = node.inputs[0]
    if (not x.owner or
            not isinstance(x.owner.op, T.Elemwise) or
            not isinstance(x.owner.op.scalar_op, scalar.Cast)):
        return

    type1 = x.owner.op.scalar_op.o_type
    type2 = node.op.scalar_op.o_type
    base = x.owner.inputs[0]

    if type1 == type2:
        # We don't need to copy over any stack traces here
        return [x]

    if(is_an_upcast(base.dtype, type1.dtype)):
        # Checking for further redundancy. Eg: int8 -> int32 -> int8
        if(type2.dtype == base.dtype):
            return x.owner.inputs
        else:
            # Apply the second cast only
            v = node.op(base)
            # Copy stack trace from the output of the original cast
            copy_stack_trace(node.outputs[0], v)
            return [v]


def is_an_upcast(type1, type2):
    """Given two data types (as strings), check if converting to
    type2 from type1 constitutes an upcast.
    Differs from theano.scalar.upcast

    """
    category = {
        # The first number in the pair is the dtype (bool, uint, int, float,
        # complex). Conversion from higher to lower is never an upcast.
        # The second number roughly indicates the precision. Again, conversion
        # from higher to lower is never an upcast.

        'bool': (0, 0),
        'uint8': (1, 1), 'uint16': (1, 2), 'uint32': (1, 3), 'uint64': (1, 4),
        'int8': (2, 1), 'int16': (2, 2), 'int32': (2, 3), 'int64': (2, 4),
        'float16': (3, 1.5), 'float32': (3, 2.5), 'float64': (3, 3.5),
        'complex64': (4, 3), 'complex128': (4, 4)
    }

    cat1 = category[type1]
    cat2 = category[type2]

    if(cat2[0] >= cat1[0] and cat2[1] > cat1[1]):
        return True
    else:
        return False


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_func_inv(node):
    """
    Check for two consecutive operations that are functional inverses
    and remove them from the function graph.

    """
    inv_pairs = (
        (basic.Deg2Rad, basic.Rad2Deg),
        (basic.Cosh, basic.ArcCosh),
        (basic.Tanh, basic.ArcTanh),
        (basic.Sinh, basic.ArcSinh),
        (basic.Conj, basic.Conj),
        (basic.Neg, basic.Neg),
        (basic.Inv, basic.Inv),
    )
    x = node.inputs[0]

    if not isinstance(node.op, T.Elemwise):
        return
    if (not x.owner or not isinstance(x.owner.op, T.Elemwise)):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    for inv_pair in inv_pairs:
        if is_inverse_pair(node_op, prev_op, inv_pair):
            # We don't need to copy stack trace, because the optimization
            # is trivial and maintains the earlier stack trace
            return x.owner.inputs

    return


def is_inverse_pair(node_op, prev_op, inv_pair):
    """
    Given two consecutive operations, check if they are the
    provided pair of inverse functions.

    """
    node_is_op0 = isinstance(node_op, inv_pair[0])
    node_is_op1 = isinstance(node_op, inv_pair[1])
    prev_is_op0 = isinstance(prev_op, inv_pair[0])
    prev_is_op1 = isinstance(prev_op, inv_pair[1])

    return (node_is_op0 and prev_is_op1) or (node_is_op1 and prev_is_op0)


class Assert(T.Op):
    """
    Implements assertion in a computational graph.

    Returns the first parameter if the condition is true, otherwise, triggers
    AssertionError.

    Notes
    -----
    This Op is a debugging feature. It can be removed from the graph
    because of optimizations, and can hide some possible optimizations to
    the optimizer. Specifically, removing happens if it can be determined
    that condition will always be true. Also, the output of the Op must be
    used in the function computing the graph, but it doesn't have to be
    returned.

    Examples
    --------
    >>> import theano
    >>> T = theano.tensor
    >>> x = T.vector('x')
    >>> assert_op = T.opt.Assert()
    >>> func = theano.function([x], assert_op(x, x.size<2))

    """
    _f16_ok = True
    __props__ = ('msg',)
    view_map = {0: [0]}

    check_input = False

    def __init__(self, msg="Theano Assert failed!"):
        self.msg = msg

    def __setstate__(self, attrs):
        self.__dict__.update(attrs)
        if not hasattr(self, 'msg'):
            self.msg = "Theano Assert failed!"

    def make_node(self, value, *conds):
        if not isinstance(value, Variable):
            value = T.as_tensor_variable(value)
        cond = [T.as_tensor_variable(c) for c in conds]
        assert np.all([c.type.ndim == 0 for c in cond])
        return gof.Apply(self, [value] + cond, [value.type()])

    def perform(self, node, inputs, out_):
        out, = out_
        v = inputs[0]
        out[0] = v
        assert np.all(inputs[1:]), self.msg

    def grad(self, input, output_gradients):
        return output_gradients + [DisconnectedType()()] * (len(input) - 1)

    def connection_pattern(self, node):
        return [[1]] + [[0]] * (len(node.inputs) - 1)

    def c_code(self, node, name, inames, onames, sub):
        value = inames[0]
        out = onames[0]
        check = []
        fail = sub['fail']
        msg = self.msg.replace('"', '\\"').replace('\n', '\\n')
        for idx in xrange(len(inames) - 1):
            i = inames[idx + 1]
            dtype = node.inputs[idx + 1].dtype
            check.append('if(!((npy_%(dtype)s*)PyArray_DATA(%(i)s))[0])'
                         '{PyErr_SetString(PyExc_AssertionError,"%(msg)s");'
                         '%(fail)s}' % locals())
        check = "\n".join(check)
        return """
        %(check)s
        Py_XDECREF(%(out)s);
        %(out)s = %(value)s;
        Py_INCREF(%(value)s);
        """ % locals()

    def c_code_cache_version(self):
        return (3, 0)

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

assert_ = Assert()
# Unittest.assert_ is a deprecated name for assertTrue.
# 2to3 convert theano.tensor.opt.assert_ to theano.tensor.opt.assertTrue
# So I define a new name as a work around.
assert_op = assert_


@register_specialize
@gof.local_optimizer([Assert])
def local_remove_useless_assert(node):
    if isinstance(node.op, Assert):
        cond = []
        for c in node.inputs[1:]:
            try:
                const = get_scalar_constant_value(c)

                if 0 != const.ndim or const == 0:
                    # Should we raise an error here? How to be sure it
                    # is not catched?
                    cond.append(c)
            except NotScalarConstantError:
                cond.append(c)

        if len(cond) == 0:
            # We don't need to copy over any stack traces here
            return [node.inputs[0]]
        if len(cond) != len(node.inputs) - 1:
            ret = assert_(node.inputs[0], *cond)

            # We copy over stack trace from the output of the original assert
            copy_stack_trace(node.outputs[0], ret)
            return [ret]


@gof.local_optimizer([Assert])
def local_remove_all_assert(node):
    """An optimization disabled by default that removes all asserts from
    the graph.

    Notes
    -----
    See the :ref:`unsafe` section to know how to enable it.

    """
    if not isinstance(node.op, Assert):
        return

    # We don't need to copy over any stack traces here
    return [node.inputs[0]]
# Disabled by default
compile.optdb['canonicalize'].register('local_remove_all_assert',
                                       local_remove_all_assert,
                                       'unsafe',
                                       use_db_name_as_tag=False)
compile.optdb['stabilize'].register('local_remove_all_assert',
                                    local_remove_all_assert,
                                    'unsafe',
                                    use_db_name_as_tag=False)
compile.optdb['specialize'].register('local_remove_all_assert',
                                     local_remove_all_assert,
                                     'unsafe',
                                     use_db_name_as_tag=False)
compile.optdb['useless'].register('local_remove_all_assert',
                                  local_remove_all_assert,
                                  'unsafe',
                                  use_db_name_as_tag=False)

#######################
# Constant Canonicalization
############################


@register_canonicalize
@gof.local_optimizer([T.Elemwise])
def local_upcast_elemwise_constant_inputs(node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return
    try:
        shape_i = node.fgraph.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, T.Elemwise):
        scalar_op = node.op.scalar_op
        # print "aa", scalar_op.output_types_preference
        if (getattr(scalar_op, 'output_types_preference', None)
                in (T.scal.upgrade_to_float, T.scal.upcast_out)):
            # this is the kind of op that we can screw with the input
            # dtypes by upcasting explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        # works only for scalars
                        cval_i = get_scalar_constant_value(i,
                                                           only_process_constants=True)
                        if all(i.broadcastable):
                            new_inputs.append(T.shape_padleft(
                                T.cast(cval_i, output_dtype),
                                i.ndim))
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(
                                T.alloc(T.cast(cval_i, output_dtype),
                                        *[shape_i(d)(i)
                                          for d in xrange(i.ndim)]))
                            # print >> sys.stderr, "AAA",
                            # *[Shape_i(d)(i) for d in xrange(i.ndim)]
                    except NotScalarConstantError:
                        # for the case of a non-scalar
                        if isinstance(i, T.TensorConstant):
                            new_inputs.append(T.cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if rval[0].type != node.outputs[0].type:
                    # This can happen for example when floatX=float32
                    # and we do the true division between and int64
                    # and a constant that will get typed as int8.

                    # As this is just to allow merging more case, if
                    # the upcast don't work, we can just skip it.
                    return

                # Copy over output stacktrace from before upcasting
                copy_stack_trace(node.outputs[0], rval)
                return rval

##################
# Subtensor opts #
##################


@register_useless
@register_canonicalize
@register_specialize
@gof.local_optimizer([IncSubtensor])
def local_useless_inc_subtensor(node):
    """
    Remove IncSubtensor, when we overwrite the full inputs with the
    new value.

    """
    if not isinstance(node.op, IncSubtensor):
        return
    if node.op.set_instead_of_inc is False:
        # This is an IncSubtensor, so the init value must be zeros
        try:
            c = get_scalar_constant_value(node.inputs[0],
                                          only_process_constants=True)
            if c != 0:
                return
        except NotScalarConstantError:
            return
    if (node.inputs[0].ndim != node.inputs[1].ndim or
            node.inputs[0].broadcastable != node.inputs[1].broadcastable):
        # FB: I didn't check if this case can happen, but this opt
        # don't support it.
        return
    # We have a SetSubtensor or an IncSubtensor on zeros
    # If is this IncSubtensor useful?

    # Check that we keep all the original data.
    # Put the constant inputs in the slice.
    idx_cst = get_idx_list(node.inputs[1:], node.op.idx_list)
    if all(isinstance(e, slice) and e.start is None and
           e.stop is None and (e.step is None or T.extract_constant(e.step,
                               only_process_constants=True) == -1)
           for e in idx_cst):
        # IncSubtensor broadcast node.inputs[1] on node.inputs[0]
        # based on run time shapes, so we must check they are the same.
        if not hasattr(node.fgraph, 'shape_feature'):
            return
        if not node.fgraph.shape_feature.same_shape(node.inputs[0],
                                                    node.inputs[1]):
            return
        # There is no reverse, so we don't need a replacement.
        if all(e.step is None
               for e in node.op.idx_list):
            # They are the same shape, so we can remore this IncSubtensor
            return [node.inputs[1]]
        ret = Subtensor(node.op.idx_list)(*node.inputs[1:])
        # Copy over previous output stacktrace
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_canonicalize
@gof.local_optimizer([AdvancedIncSubtensor1])
def local_set_to_inc_subtensor(node):
    """
    AdvancedIncSubtensor1(x, x[ilist]+other, ilist, set_instead_of_inc=True) ->
    AdvancedIncSubtensor1(x, other, ilist, set_instead_of_inc=False)

    """
    if (isinstance(node.op, AdvancedIncSubtensor1) and
            node.op.set_instead_of_inc and
            node.inputs[1].owner and
            isinstance(node.inputs[1].owner.op, Elemwise) and
            isinstance(node.inputs[1].owner.op.scalar_op, scalar.Add)):
        addn = node.inputs[1].owner
        subn = None
        other = None

        if (addn.inputs[0].owner and
                isinstance(addn.inputs[0].owner.op, AdvancedSubtensor1)):
            subn = addn.inputs[0].owner
            other = addn.inputs[1]
        elif (addn.inputs[1].owner and
              isinstance(addn.inputs[1].owner.op, AdvancedSubtensor1)):
            subn = addn.inputs[1].owner
            other = addn.inputs[0]
        else:
            return
        if (subn.inputs[1] != node.inputs[2] or
                subn.inputs[0] != node.inputs[0]):
            return
        ret = advanced_inc_subtensor1(node.inputs[0], other, node.inputs[2])
        # Copy over previous output stacktrace
        # Julian: I'm not sure about this at all...
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_useless
@register_canonicalize
@register_specialize
@gof.local_optimizer([Subtensor])
def local_useless_slice(node):
    """
    Remove Subtensor of the form X[0, :] -> X[0]
    """
    if isinstance(node.op, Subtensor):
        slices = get_idx_list(node.inputs, node.op.idx_list)
        last_slice = len(slices)
        for s in slices[::-1]:
            # check if slice and then check slice indices
            if (isinstance(s, slice) and s.start is None and s.stop is None and
                    (s.step is None or T.extract_constant(s.step,
                                                          only_process_constants=True) == 1)):
                last_slice -= 1
            else:
                break
        # check if we removed something
        if last_slice < len(slices):
            subtens = Subtensor(slices[:last_slice])
            sl_ins = Subtensor.collapse(slices[:last_slice],
                                        lambda x: isinstance(x, T.Variable))
            out = subtens(node.inputs[0], *sl_ins)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, out)
            return [out]


@register_canonicalize
@register_specialize
@gof.local_optimizer([Subtensor, AdvancedSubtensor1])
def local_useless_subtensor(node):
    """
    Remove Subtensor/AdvancedSubtensor1 if it takes the full input. In the
    AdvancedSubtensor1 case, the full input is taken when the indices are
    equivalent to `arange(0, input.shape[0], 1)` using either an explicit
    list/vector or the ARange op.

    """

    # If the optimization is tried over a node that is not a part of graph before
    if not hasattr(node, 'fgraph'):
        return

    # This optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(node.fgraph, 'shape_feature'):
        return

    shape_of = node.fgraph.shape_feature.shape_of

    if isinstance(node.op, Subtensor):
        cdata = node.op.get_constant_idx(node.inputs, allow_partial=True,
                                         only_process_constants=True)
        for pos, idx in enumerate(cdata):
            if not isinstance(idx, slice):
                # If idx is not a slice, this means we remove this dimension
                # from the output, so the subtensor is not useless
                return False
            if idx.start is not None and idx.start != 0:
                # If the start of the slice is different from 0, or is a
                # variable, then we assume the subtensor is not useless
                return False
            if idx.step is not None and idx.step != 1:
                # If we are going backwards, or skipping elements, then this
                # is not a useless subtensor
                return False

        for pos, idx in enumerate(cdata):

            length_pos = shape_of[node.inputs[0]][pos]

            if isinstance(idx.stop, (integer_types, np.integer)):
                length_pos_data = sys.maxsize
                try:
                    length_pos_data = get_scalar_constant_value(length_pos,
                                                                only_process_constants=True)
                except NotScalarConstantError:
                    pass

                if idx.stop < length_pos_data:
                    return False
            elif isinstance(idx.stop, gof.Variable):
                length_pos_shape_i = idx.stop
                # length_pos is a tensor variable, but length_pos_shape_i
                # is a scalar variable. We try to see if they represent
                # the same underlying variable.
                if (length_pos_shape_i.owner and
                        isinstance(length_pos_shape_i.owner.op,
                                   T.ScalarFromTensor)):
                    length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
                elif (length_pos.owner and
                      isinstance(length_pos.owner.op, T.TensorFromScalar)):
                    length_pos = length_pos.owner.inputs[0]
                else:
                    # We did not find underlying variables of the same type
                    return False

                # The type can be different: int32 vs int64. length_pos
                # should always be int64 as that is what the shape
                # tracker keep. Subtensor accept any scalar int{8,16,32,64}
                # as index type.
                assert str(length_pos.type.dtype) == "int64"
                assert str(length_pos_shape_i.type.dtype) in ["int8", "int16",
                                                              "int32", "int64"]

                # length_pos_shape_i cannot be None
                if length_pos_shape_i != length_pos:
                    return False
            elif idx.stop is None:
                pass
            else:
                return False
    elif isinstance(node.op, AdvancedSubtensor1):
        # get length of the indexed tensor along the first axis
        try:
            length = get_scalar_constant_value(shape_of[node.inputs[0]][0],
                                               only_process_constants=True)
        except NotScalarConstantError:
            return False

        # get index (which must be a vector by definition)
        idx = node.inputs[1]

        # `idx` must be equivalent to [0,1,...,shape[0] - 1] to qualify for
        # this optimization
        if isinstance(idx, T.Constant):
            idx = idx.value
            if len(idx) != length:
                return False
            if np.any(idx != np.arange(length)):
                return False
        elif idx.owner is not None and isinstance(idx.owner.op, T.ARange):
            try:
                start, stop, step = map(lambda x: get_scalar_constant_value(x,
                                                                            only_process_constants=True),
                                        idx.owner.inputs)
            except NotScalarConstantError:
                return False

            if start != 0:
                return False
            if stop != length:
                return False
            if step != 1:
                return False
        else:
            return False
    else:
        return False

    # We don't need to copy over any stacktrace here,
    # because previous stacktrace should suffice.
    return [node.inputs[0]]


# fast_compile to allow opt subtensor(cast{float32}(make_vector))
@register_canonicalize('fast_compile')
@gof.local_optimizer([Subtensor])
def local_subtensor_lift(node):
    """
    unary(x)[idx] -> unary(x[idx])#any broadcast pattern.

    Handles the following unary ops:
    elemwise(x,...)[idx] -> elemwise(x[idx],...)
      when x,... are broadcasted scalar or not broadcasted at all
    rebroadcast(x)[idx] => rebroadcast(x[idx])

    """
    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if not u.owner or len(u.clients) > 1:
            return False

        if isinstance(u.owner.op, T.Elemwise) and len(u.owner.inputs) == 1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, x_idx)
            ret = u.owner.op(x_idx)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
            return [ret]

        if isinstance(u.owner.op, T.Elemwise):
            new_inputs = []
            if all([sum(i.type.broadcastable) == 0 for i in u.owner.inputs]):
                # There is no broadcastable in the inputs
                idx = node.inputs[1:]
                new_inputs = [node.op(i, *idx) for i in u.owner.inputs]
                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]
            elif all([sum(i.type.broadcastable) in [i.ndim, 0]
                      for i in u.owner.inputs]):
                # There is no broadcastable in the inputs or it is scalar
                idx = node.inputs[1:]
                new_inputs = []
                for i in u.owner.inputs:
                    if sum(i.type.broadcastable) == 0:
                        new_inputs.append(node.op(i, *idx))
                    else:
                        # If the subtensor remove some dims, we must
                        # lower the number of dimensions of this scalar.
                        if node.outputs[0].ndim == i.ndim:
                            new_inputs.append(i)
                        else:
                            new_inputs.append(
                                i.dimshuffle(['x'] * node.outputs[0].ndim))

                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]

        if isinstance(u.owner.op, T.Rebroadcast):
            # make sure that Rebroadcast has only 1 input
            assert len(u.owner.inputs) == 1

            # Subtensor might reduce dim., adapt broadcast pattern accordingly
            new_axis = []

            # loop through indices being subtensor-ed
            # i indexes broadcastable pattern before subtensor
            # j indexes broadcastable pattern after subtensor
            j = 0
            for (i, x) in enumerate(node.op.idx_list):
                # if its not a slice, it will reduce the dimension, should
                # not appear in the broascastable dimensions
                if isinstance(x, slice):
                    new_axis += [(j, u.broadcastable[i])]
                    j += 1
            # now keep the broadcastable pattern of all
            # items not appearing in subtensor list
            for i in xrange(len(node.op.idx_list), len(u.broadcastable)):
                new_axis += [(j, u.broadcastable[i])]
                j += 1

            subt_x = node.op(u.owner.inputs[0], *node.inputs[1:])
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs[0], subt_x)

            rbcast_subt_x = T.Rebroadcast(*new_axis)(subt_x)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], rbcast_subt_x)

            return [rbcast_subt_x]


def merge_two_slices(slice1, len1, slice2, len2):
    """
     This function merges two slices into a single slice. The code works on
     the assumption that:

     a) slice1 is actually a slice and not an index, while slice2
        can be just an index.

     b) the two slices **have been applied consecutively** on the same
        tensor

    The output slice is **not** in canonical form, but actually just a slice
    that can be applied to a tensor to produce the same output as applying
    the two consecutive slices.
    ``len1`` is the length of the tensor **before** applying the first slice,
    while ``len2`` is the length **after** applying the first slice.
    """
    list_opt = [local_abs_merge, local_mul_switch_sink,
                local_upcast_elemwise_constant_inputs,
                local_useless_switch, constant_folding]

    if type(slice1) is not slice:
        raise ValueError(('First provided slice should actually be of type'
                         'slice and not an index !'), slice1)
    sl1, reverse1 = get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = get_canonical_form_slice(slice2, len2)

    if type(sl2) is not slice:
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = T.switch(T.le(len2, 0), len1 + 1, val)
            val = T.switch(T.ge(sl2, len2), len1 + 1, val)
            val = T.switch(T.lt(sl2, 0), - len1 - 1, val)
            if sl1.step:
                val = T.switch(T.eq(sl1.step, 0), len1 + 1, val)
            val = pre_greedy_local_optimizer(list_opt, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            if config.warn.subtensor_merge_bug:
                warnings.warn((
                    'Your current code is fine, but Theano versions '
                    'prior to 0.5rc2 might have given an incorrect result. '
                    'To disable this warning, set the Theano flag '
                    'warn.subtensor_merge_bug to False.'))
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = T.switch(T.lt(reverse1, 0), n_val, p_val)
            val = T.switch(T.le(len2, 0), len1 + 1, val)
            val = T.switch(T.ge(sl2, len2), len1 + 1, val)
            val = T.switch(T.lt(sl2, 0), - len1 - 1, val)
            if sl1.step:
                val = T.switch(T.eq(sl1.step, 0), len1 + 1, val)
            val = pre_greedy_local_optimizer(list_opt, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = T.minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = T.minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = T.switch(T.and_(T.lt(pn_stop, 0),
                                  T.gt(flen, 0)),
                           -len1 - 1,
                           T.minimum(pn_stop, sl1.stop))
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = T.minimum(pn_start, sl1.stop)
        pn_start = T.maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = T.switch(T.and_(T.lt(np_stop, 0),
                                  T.gt(flen, 0)),
                           -len1 - 1,
                           T.maximum(sl1.start - 1, np_stop))
        np_start = T.maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = T.maximum(sl1.start,
                             (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = T.maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = T.switch(T.lt(reverse2 * reverse1, 0),
                         T.switch(T.lt(reverse1, 0), np_start, pn_start),
                         T.switch(T.lt(reverse1, 0), nn_start,
                                  pp_start))

        stop = T.switch(T.lt(reverse2 * reverse1, 0),
                        T.switch(T.lt(reverse1, 0), np_stop, pn_stop),
                        T.switch(T.lt(reverse1, 0), nn_stop, pp_stop))

        step = T.switch(T.lt(reverse2 * reverse1, 0), n_step, p_step)
        start = T.switch(T.le(flen, 0), 0, start)
        stop = T.switch(T.le(flen, 0), 0, stop)

        # The canonical form of the slice is pretty complicated
        # and is not simplified. We simplify it in advance here
        # as otherwise this create too many useless optimization that
        # DebugMode must check.
        start = pre_greedy_local_optimizer(list_opt, start)
        stop = pre_greedy_local_optimizer(list_opt, stop)
        step = pre_greedy_local_optimizer(list_opt, step)
        start = pre_greedy_local_optimizer(list_opt, start)
        stop = pre_greedy_local_optimizer(list_opt, stop)
        step = pre_greedy_local_optimizer(list_opt, step)

        # Pre merge constant for the same reason.
        start, stop, step = pre_constant_merge([start, stop, step])

        return slice(start, stop, step)


@register_canonicalize
@register_specialize
@gof.local_optimizer([Subtensor])
def local_subtensor_merge(node):
    """
    Refactored optimization to deal with all cases of tensor merging.
    Given a subgraph of the form Subtensor(Subtensor(u)), the optimization
    expresses all slices in a canonical form, and then merges them together.

    """

    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if u.owner and isinstance(u.owner.op, Subtensor):
            # We can merge :)
            # x actual tensor on which we are picking slices
            x = u.owner.inputs[0]
            # slices of the first applied subtensor
            slices1 = get_idx_list(u.owner.inputs, u.owner.op.idx_list)
            slices2 = get_idx_list(node.inputs, node.op.idx_list)
            # Get the shapes of the vectors !
            try:
                # try not to introduce new shape into the graph
                xshape = node.fgraph.shape_feature.shape_of[x]
                ushape = node.fgraph.shape_feature.shape_of[u]
            except AttributeError:
                # Following the suggested use of shape_feature which should
                # consider the case when the compilation mode doesn't
                # include the ShapeFeature
                xshape = x.shape
                ushape = u.shape

            merged_slices = []
            pos_2 = 0
            pos_1 = 0
            while (pos_1 < len(slices1)) and (pos_2 < len(slices2)):
                slice1 = slices1[pos_1]
                if type(slice1) is slice:
                    merged_slices.append(
                        merge_two_slices(slice1,
                                         xshape[pos_1],
                                         slices2[pos_2],
                                         ushape[pos_2]))
                    pos_2 += 1
                else:
                    merged_slices.append(slice1)
                pos_1 += 1

            if pos_2 < len(slices2):
                merged_slices += slices2[pos_2:]
            else:
                merged_slices += slices1[pos_1:]

            merged_slices = make_constant(merged_slices)
            subtens = Subtensor(merged_slices)

            sl_ins = Subtensor.collapse(
                merged_slices,
                lambda x: isinstance(x, T.Variable))
            # Do not call make_node for test_value
            out = subtens(x, *sl_ins)

            # Copy over previous output stacktrace
            # and stacktrace from previous slicing operation.
            # Why? Because, the merged slicing operation could have failed
            # because of either of the two original slicing operations
            orig_out = node.outputs[0]
            copy_stack_trace([orig_out, node.inputs[0]], out)

            # Restore original broadcastable dimensions that `subtens()` may
            # have been unable to infer again
            if out.type != orig_out.type:
                assert out.dtype == orig_out.dtype
                assert out.ndim == orig_out.ndim
                out = T.patternbroadcast(out, orig_out.broadcastable)
                copy_stack_trace([orig_out, node.inputs[0]], out)
            return [out]


@register_useless
@register_canonicalize
@register_specialize
@gof.local_optimizer([Subtensor])
def local_subtensor_of_alloc(node):
    """

    alloc(val)[x:y] -> alloc(val[...])
    alloc(val)[x:y] -> alloc(val)
    This can be seen as a lift, but it also reduce the number of computation/memory.

    """
    if not isinstance(node.op, Subtensor):
        return False
    u = node.inputs[0]
    if u.owner is None:
        return False
    if not isinstance(u.owner.op, T.Alloc):
        return False
    slices = get_idx_list(node.inputs, node.op.idx_list)
    val = u.owner.inputs[0]
    dims = u.owner.inputs[1:]
    assert len(slices) <= len(dims)

    # Number of dimensions added to val
    n_added_dims = u.ndim - val.ndim
    # Dimensions of the returned alloc
    nw_dims = []
    # Slices to take from val
    val_slices = []

    for i, (sl, dim) in enumerate(zip(slices, dims)):
        # If val was not copied over that dim,
        # we need to take the appropriate subtensor on it.
        if i >= n_added_dims:
            # We check that the corresponding val dimensions was
            # not a broadcasted dimensions.
            if (val.type.ndim > (i - n_added_dims) and
                    val.type.broadcastable[i - n_added_dims]):
                val_slices.append(slice(None))
            else:
                val_slices.append(sl)

        csl, _ = get_canonical_form_slice(sl, dim)
        if type(csl) is not slice:
            # That dimension is removed.
            pass
        else:
            nw_dim = csl.stop - csl.start

            if csl.step != 1:
                # Do not add the ceil_intdiv() graphs in the graphs
                # when this is not needed as it prevent detecting the
                # correct broadcast pattern.
                nw_dim = T.ceil_intdiv(nw_dim, csl.step)
            nw_dims += [nw_dim]

    nw_val = val[tuple(val_slices)]
    nw_dims += dims[len(slices):]
    if nw_val.ndim > len(nw_dims):
        return False
    rval = T.alloc(nw_val, *nw_dims)
    if type(rval) not in (list, tuple):
        rval = [rval]
    if rval[0].type != node.outputs[0].type:
        # It happen that the make_node() isn't able to infer the same pattern.
        # We know it is safe, so fix that.
        rval[0] = T.patternbroadcast(rval[0], node.outputs[0].broadcastable)

    return rval


@register_canonicalize
@register_stabilize
@register_specialize
@gof.local_optimizer([Subtensor])
def local_subtensor_of_dot(node):
    """
    This optimization translates T.dot(A, B)[idxs] into T.dot(A[idxs_a], B[idxs_b]),
    where idxs_a and idxs_b are defined appropriately.

    idxs_a is the first A.ndim-1 entries of idxs,
    and idxs_b is the remaining entries of idxs (if any),
    modified to skip the second-to-last dimension of B
    (because dot sums over this dimension).

    """
    if not isinstance(node.op, Subtensor):
        return
    if (not node.inputs[0].owner or
            not isinstance(node.inputs[0].owner.op, T.Dot)):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(node.inputs[0].clients) > 1:
        return

    a = node.inputs[0].owner.inputs[0]
    b = node.inputs[0].owner.inputs[1]

    idx_list = get_idx_list(node.inputs, node.op.idx_list)

    num_a_indices = min(a.ndim - 1, len(idx_list))
    a_indices = idx_list[:num_a_indices]
    b_indices = idx_list[num_a_indices:]

    # This is necessary because np.dot sums the last index of a with the second to last of b
    # so we want to skip the second-to-last index into b.
    # This wasn't necessary for a, because we just omitted the last index.
    # We skip this if b.ndim = 1, since then we just want b_sub = b, not b_sub = b[:]
    # (dot also handles b.ndim < 2 as a special case)
    if b.ndim > 1 and len(b_indices) >= b.ndim - 1:
        b_indices = (b_indices[:b.ndim - 2] +
                     (slice(None, None, None),) + b_indices[b.ndim - 2:])

    a_sub = a.__getitem__(tuple(a_indices))
    b_sub = b.__getitem__(tuple(b_indices)) if b_indices else b

    # Copy over previous output stacktrace to a_sub and b_sub,
    # because an error in the subtensor operation (e.g. an index error)
    # on either a or b must correspond to an error in the
    # subtensor operation on their dot product.
    copy_stack_trace(node.outputs[0], [a_sub, b_sub])

    # Copy over previous output stacktrace and previous dot product stacktrace,
    # because an error here may correspond to an either in either the original
    # dot product, or in the dot product after the subtensor operation.
    r = T.dot(a_sub, b_sub)
    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]


@register_canonicalize
@gof.local_optimizer([T.add])
def local_IncSubtensor_serialize(node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """
    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return (i.owner and
                isinstance(i.owner.op, (IncSubtensor,
                                        AdvancedIncSubtensor1,
                                        AdvancedIncSubtensor,)) and
                i.type == o_type and
                len(i.clients) == 1 and
                not i.owner.op.set_instead_of_inc)

    if node.op == T.add:
        o_type = node.outputs[0].type

        movable_inputs = [i for i in node.inputs if movable(i)]

        if movable_inputs:
            new_inputs = ([i for i in node.inputs if not movable(i)] +
                          [mi.owner.inputs[0] for mi in movable_inputs])
            if len(new_inputs) == 0:
                new_add = new_inputs[0]
            else:
                new_add = T.add(*new_inputs)

                # Copy over stacktrace from original output, as an error
                # (e.g. an index error) in this add operation should
                # correspond to an error in the original add operation.
                copy_stack_trace(node.outputs[0], new_add)

            # stack up the new incsubtensors
            tip = new_add
            for mi in movable_inputs:
                assert tip.type == o_type
                assert tip.type == mi.owner.inputs[0].type
                tip = mi.owner.op(tip, *mi.owner.inputs[1:])
                # Copy over stacktrace from outputs of the original
                # "movable" operation to the new operation.
                copy_stack_trace(node.outputs + mi.owner.outputs, tip)

            return [tip]

        # print incsub_inputs, [id(i.owner.inputs[0]) for i in incsub_inputs]

# We register it in a TopoOptimizer inside the canonizer EQ optimizer.
# Otherwise in some cases it was making the EQ optimizer use 45. In
# the TopoOptimizer, the EQ only use 5 passes.
compile.optdb.register('pre_local_IncSubtensor_serialize',
                       in2out(local_IncSubtensor_serialize),
                       # Just before canonizer
                       0.99, 'fast_run')


# after priority 50 Destructive inplace operations
# gemm is the first one now, at priority 70

@gof.local_optimizer([IncSubtensor], inplace=True)
def local_inplace_setsubtensor(node):
    """
    Also work for GpuIncSubtensor.

    """
    if isinstance(node.op, IncSubtensor) and not node.op.inplace:
        dta = node.op.destroyhandler_tolerate_aliased
        new_op = node.op.__class__(
            node.op.idx_list, inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            destroyhandler_tolerate_aliased=dta)
        new_node = new_op(*node.inputs)
        val = getattr(node.outputs[0].tag, 'nan_guard_mode_check', True)
        new_node.tag.nan_guard_mode_check = val

        # Copy stacktrace from original outputs to new outputs.
        # This is sensible, because the new operation is the
        # same as the old one, but now with different attributes.
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False
compile.optdb.register('local_inplace_setsubtensor',
                       TopoOptimizer(
                           local_inplace_setsubtensor,
                           failure_callback=TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([AdvancedIncSubtensor1], inplace=True)
def local_inplace_incsubtensor1(node):
    """
    Also work for GpuAdvancedIncSubtensor1.

    """
    if isinstance(node.op, AdvancedIncSubtensor1) and not node.op.inplace:
        new_op = node.op.clone_inplace()
        new_node = new_op(*node.inputs)

        # Copy stacktrace from original outputs to new outputs.
        # This is sensible, because the new operation is the
        # same as the old one, but now with different attributes.
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False
compile.optdb.register('local_inplace_incsubtensor1',
                       TopoOptimizer(
                           local_inplace_incsubtensor1,
                           failure_callback=TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG


# Register old name
@register_canonicalize("local_incsubtensor_of_allocs")
@register_stabilize("local_incsubtensor_of_allocs")
@gof.local_optimizer([IncSubtensor,
                      AdvancedIncSubtensor,
                      AdvancedIncSubtensor1])
def local_incsubtensor_of_zeros(node):
    """
    IncSubtensor(x, zeros, idx) -> x

    """
    if (isinstance(node.op, (IncSubtensor,
                             AdvancedIncSubtensor,
                             AdvancedIncSubtensor1)) and
            not node.op.set_instead_of_inc):
        x = node.inputs[0]
        y = node.inputs[1]
        try:
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            if get_scalar_constant_value(y, elemwise=False) == 0:
                # No need to copy over the stacktrace,
                # because x should already have a stacktrace
                return [x]
        except NotScalarConstantError:
            return


@register_canonicalize
@register_specialize
@gof.local_optimizer([IncSubtensor])
def local_incsubtensor_of_zeros_to_setsubtensor(node):
    """
    IncSubtensor(zeros, x, ...) -> SetSubtensor(zeros, x, ...)
    """
    if (isinstance(node.op, (IncSubtensor)) and not node.op.set_instead_of_inc):
        x = node.inputs[0]

        if isinstance(x, T.Constant) and not np.any(x.data):
            return [IncSubtensor(node.op.idx_list,
                                 node.op.inplace,
                                 set_instead_of_inc=True,
                                 destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased,
                                 )(*node.inputs)]


@register_canonicalize('local_setsubtensor_of_allocs')
@register_stabilize('local_setsubtensor_of_allocs')
@gof.local_optimizer([IncSubtensor])
def local_setsubtensor_of_constants(node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.

    """
    if isinstance(node.op, IncSubtensor) and node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]

        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        try:
            replace_x = get_scalar_constant_value(x, elemwise=False)
        except NotScalarConstantError:
            return

        try:
            replace_y = get_scalar_constant_value(y, elemwise=False)
        except NotScalarConstantError:
            return

        if replace_x == replace_y:

            # No need to copy over the stacktrace,
            # because x should already have a stacktrace
            return [x]
        else:
            return False


@register_canonicalize
@register_stabilize
@gof.local_optimizer([AdvancedSubtensor1])
def local_adv_sub1_adv_inc_sub1(node):
    """Optimize the possible AdvSub1(AdvSetSub1(...), ...).

    AdvancedSubtensor1(AdvancedSetSubtensor1(x, y, idx), idx) -> y

    Notes
    -----
    This opt add AssertOp. Otherwise, it would remove shape and
    index error. If you want to get rid of them, see the
    :ref:`unsafe_optimization` section.

    WARNING:
    A previous version of this optimization also matched
    AdvancedSubtensor1(AdvancedIncSubtensor1(0s, y, idx), idx) -> y
    This is incorrect when there are duplicate indices.
    The current version warns the user about potential past issues.

    """
    if not isinstance(node.op, AdvancedSubtensor1):
        return
    inp = node.inputs[0]
    if (not inp.owner or
            not isinstance(inp.owner.op, AdvancedIncSubtensor1)):
        return
    idx = node.inputs[1]
    idx2 = inp.owner.inputs[2]
    x = inp.owner.inputs[0]
    y = inp.owner.inputs[1]
    if idx is not idx2:
        return
    if (not inp.owner.op.set_instead_of_inc and
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            T.extract_constant(x, elemwise=False) != 0):
        return

    if not inp.owner.op.set_instead_of_inc:
        if config.warn.inc_subtensor1_opt:
            warnings.warn(
                'Your current code is fine, but Theano versions '
                'between 0.7rc1 and 0.10 (or development versions '
                'between Nov. 2014 and May 2017) '
                'might have given incorrect results. This graph has '
                'following pattern: inc_subtensor(zeros[idx], x)[idx], '
                'where idx is an array of integers. This used to be '
                'optimized to "x", which is incorrect if there are '
                'duplicated indices in idx. '
                'To disable this warning, set the Theano flag '
                'warn.inc_subtensor1_opt to False.')
        return

    cond = [T.all(T.and_(T.lt(idx, x.shape[0]), T.ge(idx, -x.shape[0])))]
    if not node.fgraph.shape_feature.same_shape(idx, y, 0, 0):
        cond.append(T.eq(idx.shape[0], y.shape[0]))
    r = Assert("Bad indexing or shapes in a AdvancedIncSubtensor1 "
               "that was optimized away")(y, *cond)
    copy_stack_trace(y, r)

    if r.dtype == node.outputs[0].dtype:
        return [r]
    # It is possible that y is upcast or downcast to x.dtype.
    # In all case, as we set or add with 0, we can just cast y.
    r2 = T.cast(r, node.outputs[0].dtype)

    # Copy over stacktrace from before casting, since
    # we don't expect problems in the casting operation,
    # and any problems in the indexing would have been spotted above.
    copy_stack_trace(r, r2)
    return [r2]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@gof.local_optimizer([IncSubtensor,
                      AdvancedIncSubtensor,
                      AdvancedIncSubtensor1])
def local_useless_inc_subtensor_alloc(node):
    """
    Replaces an [Advanced]IncSubtensor[1], whose increment is an `alloc` of
    a fully or partially broadcastable variable, by one that skips the
    intermediate `alloc` where possible.

    """
    if isinstance(node.op, (IncSubtensor,
                            AdvancedIncSubtensor,
                            AdvancedIncSubtensor1)):
        x = node.inputs[0]
        y = node.inputs[1]
        i = node.inputs[2:]

        if y.owner is not None and isinstance(y.owner.op, T.Alloc):
            # `z` is the input of the Alloc op, i.e. T.alloc(z, <shape>)
            z = y.owner.inputs[0]

            try:
                shape_feature = node.fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            shape_of = shape_feature.shape_of
            same_shape = shape_feature.same_shape

            # Get the subtensor of `x` indexed by `i` in order to compare
            # shapes later.
            if isinstance(node.op, IncSubtensor):
                xi = Subtensor(node.op.idx_list)(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = advanced_subtensor(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor1):
                xi = advanced_subtensor1(x, *i)
            else:
                raise Exception('Should never happen!')

            reason = 'local_useless_incsubtensor_alloc'

            # Add `xi` to the shape feature `fgraph`. This is important for
            # shape inference later because the variable must be part of the
            # function graph in order to call `same_shape` on it.
            if xi not in shape_of:
                shape_feature.on_import(node.fgraph, xi.owner,
                                        '%s: add `xi`' % reason)

            # `xi` may have more dimensions than `y` since the subtensor ops
            # do automatic broadcasting of the increment internally. Thus, we
            # need to make the leading implicitly broadcasted dimensions
            # explicit for shape comparison later.
            if xi.ndim > y.ndim:
                y = T.shape_padleft(y, xi.ndim - y.ndim)
                if y not in shape_of:
                    shape_feature.on_import(node.fgraph, y.owner,
                                            '%s: add `y`' % reason)

            # Build `z_broad` explicitly to include extra implicit dimensions.
            z_broad = ((True,) * (xi.ndim - z.ndim) + z.broadcastable)

            cond = [
                # The shapes of `y` and `xi` must either agree or `y` may
                # also have shape equal to 1 which may be treated as a
                # broadcastable dimension by the subtensor op.
                T.or_(T.eq(y.shape[k], 1), T.eq(y.shape[k], xi.shape[k]))
                # Loop over all dimensions.
                for k in xrange(xi.ndim)
                # We need to check the above shapes, if
                # * the pre-alloc increment `z` is broadcastable in
                # dimension `k` (if it isn't, then the shapes of `z` and
                # `y` are the same by the definition of the `Alloc` op in
                # this dimension and replacing `y` by `z` will not hide a
                # shape error), and
                # * `xi` and `y` do not have the same shape in dimension
                # `k` or we cannot infer the shape statically (if the
                # shapes of `xi` and `y` are not the same, then replacing
                # `y` by `z` will hide the shape error of `y`), and
                # * the shape of `y` is not equal to 1 or we cannot infer
                # the shape statically (if the shape of `y` is equal to
                # 1, then `y` is broadcasted by the inc_subtensor op
                # internally, so the shapes of `xi` and `y` do not need
                # to match in dimension `k`; else we need to check at
                # runtime that the shape of `y` is either 1 or the same
                # as `xi` or otherwise replacing `y` by `z` will hide a
                # shape error).
                if (z_broad[k] and
                    not same_shape(xi, y, dim_x=k, dim_y=k) and
                    shape_of[y][k] != 1)]

            if len(cond) > 0:
                msg = '`x[i]` and `y` do not have the same shape.'
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *i)
            # Copy over stacktrace from previous output, since
            # we don't expect problems when removing the intermediate
            # alloc operation and so we still want to point at the line
            # of the inc_subtensor operation.
            copy_stack_trace(node.outputs, r)

            return [r]


####################
# Rebroadcast opts #
####################

@register_useless
@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_useless_rebroadcast(node):
    """
    Remove Rebroadcast if id does not actually change the broadcasting pattern.

    """
    if isinstance(node.op, T.Rebroadcast):
        x = node.inputs[0]
        if np.all(x.broadcastable == node.outputs[0].broadcastable):
            # No broadcastable flag was modified
            # No need to copy over stack trace,
            # because x should already have a stack trace.
            return [x]
        else:
            # Keep the flags that modify something
            new_axis = {}
            for dim, bc in list(node.op.axis.items()):
                if x.broadcastable[dim] != bc:
                    new_axis[dim] = bc
            if new_axis == node.op.axis:
                # All flags are useful
                return
            else:
                r = T.Rebroadcast(*list(new_axis.items()))(x)
                # Copy over stacktrace from previous output
                copy_stack_trace(node.outputs, r)
                return [r]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_rebroadcast_lift(node):
    """
    Lifts Rebroadcast through unary Elemwise operations,
    and merges consecutive Rebroadcasts.

    Rebroadcast(Elemwise(x)) => Elemwise(Rebroadcast(x))
    Rebroadcast(Rebroadcast(x)) => Rebroadcast(x)

    """
    op = node.op
    if not isinstance(op, T.Rebroadcast):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and len(inode.inputs) == 1:
        # It may happen that `input` has no client because this optimization
        # is called from `apply_rebroadcast_opt`, which in particular is used
        # by the `unbroadcast` function before we are in the actual function
        # compilation phase.
        if hasattr(input, 'clients') and len(input.clients) == 1:
            rebroadcasted = T.Rebroadcast(*list(op.axis.items()))(
                inode.inputs[0])
            # Copy over stacktrace from previous output (after rebroadcasting)
            # to new output, because an error in the new graph right after
            # rebroadcasting must have been caused by the previous rebroadcasting.
            copy_stack_trace(node.outputs, rebroadcasted)

            rval = inode.op.make_node(rebroadcasted).outputs

            # Copy over stacktrace from previous output (after rebroadcasting)
            # and input (after elemwise operation) to new output, because an
            # error in the new graph could have been caused by either of the
            # two ops.
            copy_stack_trace(node.outputs + node.inputs, rval)

            return rval
    if inode and isinstance(inode.op, T.Rebroadcast):
        # the "axis" specification in the outer Rebroadcast overrides
        # the axis of the inner one
        axis = inode.op.axis.copy()
        axis.update(op.axis)
        iinput = inode.inputs[0]

        rval = [T.Rebroadcast(*list(axis.items()))(iinput)]

        # Copy over stacktrace from previous output (after second rebroadcast)
        # and from previous input (after first rebroadcast op) because an error in
        # the new graph could have been caused by either of the two
        # rebroadcast ops.
        copy_stack_trace(node.outputs + node.inputs, rval)
        return rval


def apply_rebroadcast_opt(rval):
    """
    Apply as many times as required the optimization local_useless_rebroadcast
    and local_rebroadcast_lift.

    Parameters
    ----------
    rval: a Variable

    Returns
    -------
    A Variable (the same if no optimization can be applied)

    """

    changed = True
    while changed and rval.owner:
        changed = False
        rval2 = theano.tensor.opt.local_useless_rebroadcast.transform(
            rval.owner)
        if rval2:
            assert len(rval2) == 1
            rval = rval2[0]
            changed = True
        if rval.owner:
            rval2 = theano.tensor.opt.local_rebroadcast_lift.transform(
                rval.owner)
            if rval2:
                assert len(rval2) == 1
                rval = rval2[0]
                changed = True
    return rval


#############
# Join opts #
#############
@register_specialize
@register_canonicalize
@register_useless
@gof.local_optimizer([T.Join])
def local_join_1(node):
    """Join(i, x) => x

    Remove Join() when only one element is joined.

    """
    if not isinstance(node.op, T.Join):
        return
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        # We don't need to copy over any stacktrace here, because the
        # input variable should already have its own stacktrace.
        return [tensors[0]]


# TODO: merge in local_useless_join
@register_useless
@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Join])
def local_join_empty(node):
    """Join(i, x, y, empty) => Join(i, x, y)

    Remove empty inputs to joins. The empty inputs can be anywhere.

    """
    if not isinstance(node.op, T.Join):
        return
    new_inputs = []
    try:
        join_idx = get_scalar_constant_value(node.inputs[0],
                                             only_process_constants=True)
    except NotScalarConstantError:
        return
    for idx in xrange(1, len(node.inputs)):
        inp = node.inputs[idx]
        # We can not use size == 0,, as this can change shape from 3,0
        # to 2,0.  This trigger DebugMode error. This happen with
        # stack(...,[]) as this add a dimshuffle on [], that add a
        # dimensions with shape 1.
        if isinstance(inp, theano.Constant) and inp.data.shape[join_idx] == 0:
            continue
        new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        if len(new_inputs) == 0:
            # T.join do not work in that case.
            # constant folding will take care of this case.
            return
        ret = T.join(node.inputs[0], *new_inputs)
        o = node.outputs[0]
        if ret.dtype != o.dtype:
            # Join can upcast some inputs
            return

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)

        if ret.type != o.type:
            assert ret.dtype == o.dtype
            assert ret.ndim == o.ndim
            ret = T.patternbroadcast(ret, node.outputs[0].broadcastable)

        # Copy over stacktrace from previous output
        # (after patternbroadcast op) for same reasons as before.
        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_specialize
@register_canonicalize
@register_useless
@gof.local_optimizer([T.Join])
def local_join_make_vector(node):
    """Join(0, make_vector1, make_vector2, ...) => Join(0, make_vector12, ...)

    Merge MakeVector inputs to Join. This can make the join completly
    disapear with the local_join_1 opt.

    """
    if not isinstance(node.op, T.Join) or node.outputs[0].ndim != 1:
        return
    new_inputs = [node.inputs[1]]
    for idx in xrange(2, len(node.inputs)):
        inp = node.inputs[idx]
        if (inp.owner and
                isinstance(inp.owner.op, MakeVector) and
                new_inputs[-1].owner and
                isinstance(new_inputs[-1].owner.op, MakeVector) and
                # MakeVector have a dtype parameter
                inp.owner.op == new_inputs[-1].owner.op):
            inps = new_inputs[-1].owner.inputs + inp.owner.inputs
            new_inputs[-1] = inp.owner.op(*inps)

            # Copy over stacktrace from previous output (after join op)
            # to new intermediate output, because an error in the intermediate
            # op must be caused by an error in the old join op.
            copy_stack_trace(node.outputs, new_inputs[-1])
        else:
            new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        ret = T.join(node.inputs[0], *new_inputs)

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)
        return [ret]


#################
#  speed/memory #
#################
@register_canonicalize
@register_specialize
@gof.local_optimizer([T.elemwise.Sum])
def local_sumsqr2dot(node):
    """
    This optimization detects T.sqr( W.dimshuffle('x',0,1) * G.dimshuffle(0,'x',1) ).sum(axis=(1,2))
     and converts this to T.dot(T.sqr(G), T.sqr(W).sum(axis=0)).
    """
    if (isinstance(node.op, T.elemwise.Sum) and
            isinstance(node.op.scalar_op, theano.scalar.basic.Add) and node.op.axis == (1, 2)):
        in1 = node.inputs[0]
        out = node.outputs[0]

        if (in1.owner and isinstance(in1.owner.op, T.Elemwise) and isinstance(in1.owner.op.scalar_op, theano.scalar.basic.Sqr)):
            in_sqr = in1.owner.inputs[0]
            if (in_sqr.owner and isinstance(in_sqr.owner.op, T.Elemwise) and
                    isinstance(in_sqr.owner.op.scalar_op, theano.scalar.basic.Mul) and len(in_sqr.owner.inputs) == 2):
                in_mul1, in_mul2 = in_sqr.owner.inputs

                if (isinstance(in_mul1.owner.op, T.elemwise.DimShuffle) and in_mul1.owner.op.new_order == ('x', 0, 1) and
                        isinstance(in_mul2.owner.op, T.elemwise.DimShuffle) and in_mul2.owner.op.new_order == (0, 'x', 1)):
                    W = in_mul1.owner.inputs[0]
                    G = in_mul2.owner.inputs[0]

                    new_out = T.dot(T.sqr(G), T.sqr(W).sum(axis=0))
                    if new_out.dtype != out.dtype:
                        new_out = T.cast(new_out, dtype=out.dtype)
                    return [new_out]


#################
# Exp stability #
#################
@register_stabilize
@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Elemwise])
def local_expm1(node):
    """
    This optimization detects exp(a)-1 and converts this to expm1(a).
    """
    if (isinstance(node.op, T.Elemwise) and
            isinstance(node.op.scalar_op, theano.scalar.basic.Sub)):
        in1, in2 = node.inputs
        out = node.outputs[0]

        if (in1.owner and isinstance(in1.owner.op, T.Elemwise) and isinstance(in1.owner.op.scalar_op, theano.scalar.basic.Exp) and
                T.extract_constant(in2, only_process_constants=False) == 1):
            in11 = in1.owner.inputs[0]
            new_out = T.expm1(in11)

            if new_out.dtype != out.dtype:
                new_out = T.cast(new_out, dtype=out.dtype)
            if new_out.type != out.type:
                return
            return [new_out]


###############
# Switch opts #
###############
@register_useless('local_remove_switch_const_cond')
@register_canonicalize('fast_compile', 'local_remove_switch_const_cond')
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_useless_switch(node):
    """
    This optimization makes the following changes in the graph:
        T.switch(cond,left,right) -->
               if cond is constant and cond == 0: right
               if cond is constant and cond != 0: left
               if left is right -> left

        T.switch(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)
    """
    if (isinstance(node.op, T.Elemwise) and
            isinstance(node.op.scalar_op, scalar.basic.Switch)):
        cond = T.extract_constant(node.inputs[0],
                                  only_process_constants=True)
        if ((type(cond) is np.ndarray and cond.ndim == 0) or
                isinstance(cond, np.number)):
            if cond == 0:
                correct_out = node.inputs[2]
            else:
                correct_out = node.inputs[1]

            if correct_out.ndim != node.outputs[0].ndim:
                # TODO: broadcast?
                return False
            if correct_out.dtype != node.outputs[0].dtype:
                out = T.cast(correct_out, node.outputs[0].dtype)
            else:
                out = correct_out

            if out.type.broadcastable != node.outputs[0].type.broadcastable:
                # We need to copy data to the new dimensions during execution

                # We should not depend on node.outputs as this would
                # make the new node depend on the old one that will
                # get optimized again. So this create a cycle.
                shps = []
                for idx, (b1, b2), in enumerate(zip(out.type.broadcastable,
                                                    node.outputs[0].type.broadcastable)):
                    if b1 == b2:
                        shps.append(out.shape[idx])
                    elif not node.inputs[1].type.broadcastable[idx]:
                        shps.append(node.inputs[1].shape[idx])
                    else:
                        shps.append(node.inputs[2].shape[idx])
                out = T.alloc(out, *shps)
            else:
                out = out

            # Copy over stacktrace from selected output to new output
            copy_stack_trace(node.outputs + correct_out, out)
            return [out]
        # if left is right -> left
        if node.inputs[1] is node.inputs[2]:
            # Note: No need to copy over stacktrace, because the input node
            # already has its own stacktrace
            if cond.type == node.inputs[1].type:
                return [node.inputs[1]]

            ret = T.fill(cond, node.inputs[1])

            # Copy over stacktrace from switch output and correct branch
            copy_stack_trace(node.outputs + node.inputs[1], ret)
            return [ret]

        # This case happens with scan.
        # Elemwise{switch}(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)
        left = node.inputs[1]
        right = node.inputs[2]
        cond_var = node.inputs[0]
        if cond_var.owner and \
           isinstance(cond_var.owner.op, T.Elemwise) and \
           isinstance(cond_var.owner.op.scalar_op, scalar.LE) and \
           cond_var.owner.inputs[0].owner and \
           isinstance(cond_var.owner.inputs[0].owner.op, Shape_i) and \
           T.extract_constant(cond_var.owner.inputs[1], only_process_constants=True) == 0 and \
           T.extract_constant(left, only_process_constants=True) == 0 and \
           right is cond_var.owner.inputs[0]:
            assert right.type == node.outputs[0].type
            # No need to copy over stacktrace, because the right input node
            # already has its own stacktrace
            return [right]
        return False
    return False


@register_specialize
@register_canonicalize
@gof.local_optimizer([T.mul])
def local_mul_switch_sink(node):
    """
    This optimization makes the following changes in the graph:
    T.mul(A,T.switch(cond,0,iff),B) -->  T.switch(cond,0,T.mul(A,B,iff))
    T.mul(A,T.switch(cond,ift,0),B) -->  T.switch(cond,T.mul(A,B,ift),0)
    A and B being several (or none) symbolic variables.
    This is useful because A and B may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    With this optimization T.grad(T.switch(...)) has the right behavior.

    Examples
    --------
      x -> f(x)
      x -> g(x)
      y = T.switch(cond,f(x),g(x))
      **without the optimization
      T.grad(y,x) -> grad(f(x),x) * grad(y,f(x)) +  grad(g(x),x) * grad(y,g(x))
      **with the optimization
      T.grad(y,x) -> switch(cond,grad(f(x),x), 0) + switch(cond,0,grad(g(x),x))
    This will be particularly useful for the lazyif because we skip
    an entire part of the graph.

    """
    if node.op != T.mul:
        return False
    for idx, i in enumerate(node.inputs):
        if i.owner and i.owner.op == T.switch:
            switch = i.owner
            try:
                if (get_scalar_constant_value(
                        switch.inputs[1], only_process_constants=True) == 0.):
                    listmul = node.inputs[:idx] + node.inputs[idx + 1:]
                    fmul = T.mul(*(listmul + [switch.inputs[2]]))

                    # Copy over stacktrace for elementwise multiplication op
                    # from previous elementwise multiplication op.
                    # An error in the multiplication (e.g. errors due to
                    # inconsistent shapes), will point to the
                    # multiplication op.
                    copy_stack_trace(node.outputs, fmul)

                    fct = [T.switch(switch.inputs[0], 0,
                                    fmul)]
                    fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                    # Copy over stacktrace for switch op from both previous
                    #  elementwise multiplication op and previous switch op,
                    # because an error in this part can be caused by either
                    # of the two previous ops.
                    copy_stack_trace(node.outputs + switch.outputs, fct)
                    return fct
            except NotScalarConstantError:
                pass
            try:
                if (get_scalar_constant_value(
                        switch.inputs[2], only_process_constants=True) == 0.):
                    listmul = node.inputs[:idx] + node.inputs[idx + 1:]
                    fmul = T.mul(*(listmul + [switch.inputs[1]]))
                    # Copy over stacktrace for elementwise multiplication op
                    # from previous elementwise multiplication op.
                    # An error in the multiplication (e.g. errors due to
                    # inconsistent shapes), will point to the
                    # multiplication op.
                    copy_stack_trace(node.outputs, fmul)

                    fct = [T.switch(switch.inputs[0],
                                    fmul, 0)]
                    fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                    # Copy over stacktrace for switch op from both previous
                    # elementwise multiplication op and previous switch op,
                    # because an error in this part can be caused by either
                    # of the two previous ops.
                    copy_stack_trace(node.outputs + switch.outputs, fct)
                    return fct
            except NotScalarConstantError:
                pass
    return False


@register_canonicalize
@gof.local_optimizer([T.true_div, T.int_div])
def local_div_switch_sink(node):
    """
    This optimization makes the following changes in the graph:
    T.div(T.switch(cond,0,iff),A) -->  T.switch(cond,0,T.div(iff,A))
    T.div(T.switch(cond,ift,0),A) -->  T.switch(cond,T.div(ift,A),0)

    A being a symbolic variable.
    This is useful because A may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    See local_mul_switch_sink for more details.

    """
    if (node.op != T.true_div and node.op != T.int_div):
        return False
    op = node.op
    if node.inputs[0].owner and node.inputs[0].owner.op == T.switch:
        switch = node.inputs[0].owner
        try:
            if get_scalar_constant_value(switch.inputs[1],
                                         only_process_constants=True) == 0.:
                fdiv = op(switch.inputs[2], node.inputs[1])
                # Copy over stacktrace for elementwise division op
                # from previous elementwise multiplication op.
                # An error in the division (e.g. errors due to
                # inconsistent shapes or division by zero),
                # will point to the new division op.
                copy_stack_trace(node.outputs, fdiv)

                fct = [T.switch(switch.inputs[0], 0,
                                fdiv)]
                fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                # Copy over stacktrace for switch op from both previous
                # elementwise division op and previous switch op,
                # because an error in this part can be caused by either
                # of the two previous ops.
                copy_stack_trace(node.outputs + switch.outputs, fct)
                return fct
        except NotScalarConstantError:
            pass
        try:
            if get_scalar_constant_value(switch.inputs[2],
                                         only_process_constants=True) == 0.:
                fdiv = op(switch.inputs[1], node.inputs[1])
                # Copy over stacktrace for elementwise division op
                # from previous elementwise multiplication op.
                # An error in the division (e.g. errors due to
                # inconsistent shapes or division by zero),
                # will point to the new division op.
                copy_stack_trace(node.outputs, fdiv)

                fct = [T.switch(switch.inputs[0],
                                fdiv, 0)]
                fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                # Copy over stacktrace for switch op from both previous
                # elementwise division op and previous switch op,
                # because an error in this part can be caused by either
                # of the two previous ops.
                copy_stack_trace(node.outputs + switch.outputs, fct)
                return fct
        except NotScalarConstantError:
            pass
    return False


# Merge add/sub/mul/div/minimum/maximum/... of switches sharing the same
# condition, to enable further simplification of their branches
# Example: switch(c, a, b) + switch(c, x, y) -> switch(c, a+x, b+y)
@register_canonicalize
@gof.local_optimizer([T.Elemwise])
def local_merge_switch_same_cond(node):
    scal = theano.scalar
    # node must be binary elemwise or add or mul
    if not isinstance(node.op, T.Elemwise) or not isinstance(
            node.op.scalar_op, (scal.BinaryScalarOp, scal.Add, scal.Mul)):
        return
    # all inputs must be switch
    if not all(s.owner and isinstance(s.owner.op, T.Elemwise) and
               isinstance(s.owner.op.scalar_op, scal.Switch)
               for s in node.inputs):
        return
    # all switch conditions must be the same
    cond = node.inputs[0].owner.inputs[0]
    if not all(s.owner.inputs[0] is cond for s in node.inputs[1:]):
        return
    # pull out switch
    return [T.switch(cond,
                     node.op(*[s.owner.inputs[1] for s in node.inputs]),
                     node.op(*[s.owner.inputs[2] for s in node.inputs]))]


#############
# Tile Opts #
#############
@register_useless
@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.Tile])
def local_useless_tile(node):
    """Tile(x, (1,)*N) -> x

    This is useless tile. (1,)*N, just mean a vector with all element
    being 1.

    """
    if isinstance(node.op, T.Tile):
        try:
            a = T.get_scalar_constant_value(node.inputs[1],
                                            only_process_constants=True)
            if a == 1:
                try:
                    l = T.get_vector_length(node.inputs[1])
                    if l == node.inputs[0].ndim:
                        # No need to copy over any stacktrace as previous
                        # input variable already has a stacktrace
                        return [node.inputs[0]]
                    elif l < node.inputs[0].ndim:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        return [node.inputs[0]]
                    else:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        x_nd = node.inputs[0].ndim
                        broad = ['x'] * (l - x_nd) + xrange(x_nd)
                        ret = node.inputs[0].dimshuffle(broad)
                        # Copy over stacktrace from previous output node,
                        # and from node before tiling operation.
                        copy_stack_trace(node.outputs + node.inputs[0], ret)
                        return [ret]
                except ValueError:
                    return
        except NotScalarConstantError:
            return


##############
# Split Opts #
##############
@register_useless
@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Split])
def local_useless_split(node):
    """ Split{n_splits=1}(x, y) -> x

    Remove Split with only 1 split.

    """
    if isinstance(node.op, T.Split):
        if node.op.len_splits == 1:
            x, axis, splits = node.inputs
            out = assert_op(x, T.eq(splits.shape[0], 1))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(node.outputs, out)
            out2 = assert_op(out, T.eq(x.shape[axis], splits[0]))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(out, out2)

            return [out2]


################
# Flatten Opts #
################
@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.Flatten])
def local_flatten_lift(node):
    """
    Flatten(UnaryElemwise(x)) -> UnaryElemwise(Flatten(x))

    This optimization is needed by optimization
    nnet/sigm.py:log1msigm_to_softplus to get applied when there is a flatten.

    """
    if (isinstance(node.op, T.Flatten) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, T.Elemwise) and
            len(node.inputs[0].owner.inputs) == 1):
        f = node.op(node.inputs[0].owner.inputs[0])

        # Copy over stacktrace from previous output node (flatten op),
        # since this is the op which may cause an error for f.
        copy_stack_trace(node.outputs, f)

        e = node.inputs[0].owner.op(f)

        # Copy over stacktrace from previous output node and from unary
        # elementwise output node since if there was an error, it would
        # probably have come from that operation.
        copy_stack_trace(node.outputs + [node.inputs[0]], e)

        return [e]

##################
# Reshape opts   #
##################


def local_reshape_chain(op):
    @gof.local_optimizer([op])
    def f(node):
        """
        Reshape(Reshape(shape1),shape2) -> Reshape(shape2)

        """
        if not opt.check_chain(node, op, op):
            return False

        # TODO: this can permit a failing program to run by eliminating
        #       the lower reshape
        rval = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])

        # Copy over stacktrace from previous output node, as any error
        # in new computational graph would have been caused by last op
        # in the old computational graph.
        copy_stack_trace(node.outputs, rval)

        # It might happen that the desired output of this node has a
        # broadcastable pattern that does not match that of 'rval'. This is
        # when originally, we were able to figure out that one of the
        # dimensions of the reshape is one, but some other transformation
        # replaced the shape by one for which this cannot be guessed.
        # We should try to figure out why we lost the information about this
        # constant value... but in the meantime, better not apply this
        # optimization.
        if rval.broadcastable == node.outputs[0].broadcastable:
            return [rval]
        else:
            return False

    return f
register_canonicalize(local_reshape_chain(T.Reshape),
                      name='local_reshape_chain')


@register_useless
@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.Reshape])
def local_useless_reshape(node):
    """
    Remove two kinds of useless reshape.

    Remove Reshape when both the input and output have a single dimension.
    Remove Reshape when reshaping to the shape of the input.

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    input = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    if input.ndim != output.ndim:
        return False

    # Simple case: both input and output have a single dimension.
    # This could hide errors if the user provides inconsistent shapes.
    if (input.ndim == 1 and output.ndim == 1 and
            input.broadcastable == output.broadcastable):
        return [input]

    # Second case: all the shapes match the input shape
    # Match Reshape(x, x.shape)
    if output_shape.owner and isinstance(output_shape.owner.op, Shape):
        shape_input = output_shape.owner.inputs[0]
        if shape_input == input:
            return [input]

    # Match Reshape(x, [x.shape[0], ..., x.shape[-1]]), accounting for
    # broadcastable and constant dimensions
    if output_shape.owner and isinstance(output_shape.owner.op, MakeVector):
        output_shape_is = output_shape.owner.inputs

        if not hasattr(node, 'fgraph'):
            shape_feature = None
        else:
            shape_feature = getattr(node.fgraph, 'shape_feature', None)

        nb_m1 = 0
        shape_match = [False] * input.ndim
        for dim in xrange(input.ndim):
            outshp_i = output_shape_is[dim]
            # Match Shape_i{dim}(input)
            if (outshp_i.owner and isinstance(outshp_i.owner.op, Shape_i) and
                    outshp_i.owner.op.i == dim and
                    outshp_i.owner.inputs[0] == input):
                shape_match[dim] = True
                continue

            # Match Shape(input)[dim]
            if (outshp_i.owner and isinstance(outshp_i.owner.op, Subtensor) and
                    len(outshp_i.owner.inputs) == 2 and
                    extract_constant(outshp_i.owner.inputs[1]) == dim):
                subtensor_inp = outshp_i.owner.inputs[0]
                if (subtensor_inp.owner and
                        isinstance(subtensor_inp.owner.op, Shape)):
                    shape_input_i = subtensor_inp.owner.inputs[0]
                    if shape_input_i == input:
                        shape_match[dim] = True
                        continue

            # Match 1 if input.broadcastable[dim] is True
            cst_outshp_i = extract_constant(outshp_i, only_process_constants=1)
            if input.broadcastable[dim] and cst_outshp_i == 1:
                shape_match[dim] = True
                continue

            # Match -1
            if cst_outshp_i == -1:
                shape_match[dim] = True
                nb_m1 += 1
                continue

            # Match shape_of[input][dim] or its constant equivalent
            if shape_feature:
                inpshp_i = shape_feature.get_shape(input, dim)
                if (inpshp_i == outshp_i or
                    (extract_constant(inpshp_i, only_process_constants=1) ==
                     extract_constant(outshp_i, only_process_constants=1))):
                    shape_match[dim] = True
                    continue

        if all(shape_match) and nb_m1 <= 1:
            return [input]

        # TODO later: if all the shapes except one match, we may want to
        # consider it useless as well, like we do in the 1-dim case.


@register_canonicalize
@gof.local_optimizer([T.Reshape])
def local_reshape_to_dimshuffle(node):
    """
    Broadcastable dimensions in Reshape are replaced with dimshuffle.

    The goal is to avoid using reshape to add or remove broadcastable
    dimensions, but use dimshuffle instead, so dimshuffles can cancel out
    or be removed later on.

    For example:
        - reshape(x, (1, n)) --> dimshuffle{x,0}(reshape(x, (n,))
        - reshape(x, (1, m, 1, n, 1, 1))
          --> dimshuffle{x,0,x,1,x,x}(reshape(x, (m, n)))
    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    input = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    dimshuffle_new_order = []
    new_output_shape = []
    index = 0  # index over the output of the new reshape
    for i in xrange(output.ndim):
        # Since output_shape is a symbolic vector, we trust extract_constant
        # to go through however it is formed to see if its i-th element is 1.
        # We need only_process_constants=False for that.
        dim = extract_constant(output_shape[i], only_process_constants=False,
                               elemwise=False)
        if dim == 1:
            dimshuffle_new_order.append('x')
        else:
            dimshuffle_new_order.append(index)
            new_output_shape.append(dim)
            index = index + 1
    if index != output.ndim:
        inner = op.__class__(len(new_output_shape))(input, new_output_shape)
        copy_stack_trace(output, inner)
        new_node = [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]
        copy_stack_trace(output, new_node)
        return new_node


@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.Reshape])
def local_reshape_lift(node):
    """
    Reshape(UnaryElemwise(x)) -> UnaryElemwise(Reshape(x))

    This optimization is needed by optimization
    nnet/sigm.py:log1msigm_to_softplus to get applied when there is a reshape.

    """
    if (isinstance(node.op, T.Reshape) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, T.Elemwise) and
            len(node.inputs[0].owner.inputs) == 1):
        r = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
        # Copy stacktrace from previous Reshape op, as an error in new
        # Reshape op could only have been caused by old one.
        copy_stack_trace(node.outputs, r)

        e = node.inputs[0].owner.op(r)
        # Copy stacktrace from both previous Reshape and UnaryElemwise op
        # because an error in new cg could have been caused by either ops.
        copy_stack_trace(node.outputs + node.inputs, e)

        # In rare case the original broadcast was (False, True), but
        # the new one is (False, False). So don't crash in that case.
        if e.type != node.outputs[0].type:
            re = T.patternbroadcast(e, node.outputs[0].broadcastable)

            # Copy over stack trace.
            # If the graph fails it is usually due to the fact that a dimension
            # that should be broadcastable does not actually have length 1,
            copy_stack_trace(e, re)
        else:
            re = e

        return [re]


##################
# Middleman cuts #
##################

register_canonicalize(gof.OpRemove(T.tensor_copy), name='remove_tensor_copy')

################
# Canonization #
################


class Canonizer(gof.LocalOptimizer):
    """
    Simplification tool. The variable is a local_optimizer. It is best used
    with a TopoOptimizer in in_to_out order.

    Usage: Canonizer(main, inverse, reciprocal, calculate)

    Parameters
    ----------
    main
        A suitable Op class that is commutative, associative and
        takes one to an arbitrary number of inputs, e.g. add or
        mul
    inverse
        An Op class such that inverse(main(x, y), y) == x
        e.g. sub or true_div
    reciprocal
        A function such that main(x, reciprocal(y)) == inverse(x, y)
        e.g. neg or inv
    calculate
        Function that takes a list of numpy.ndarray instances
        for the numerator, another list for the denumerator,
        and calculates inverse(main(\*num), main(\*denum)). It
        takes a keyword argument, aslist. If True, the value
        should be returned as a list of one element, unless
        the value is such that value = main(). In that case,
        the return value should be an empty list.

    Examples
    --------
    >>> import theano.tensor as T
    >>> from theano.tensor.opt import Canonizer
    >>> add_canonizer = Canonizer(T.add, T.sub, T.neg, \\
    ...                           lambda n, d: sum(n) - sum(d))
    >>> mul_canonizer = Canonizer(T.mul, T.true_div, T.inv, \\
    ...                           lambda n, d: prod(n) / prod(d))

    Examples of optimizations mul_canonizer can perform:

    | x / x -> 1
    | (x * y) / x -> y
    | x / y / x -> 1 / y
    | x / y / z -> x / (y * z)
    | x / (y / z) -> (x * z) / y
    | (a / b) * (b / c) * (c / d) -> a / d
    | (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
    | 2 * x / 2 -> x
    | x * y * z -> Elemwise(T.mul){x,y,z} #only one pass over the memory.
    |           !-> Elemwise(T.mul){x,Elemwise(T.mul){y,z}}

    """

    def __init__(self, main, inverse, reciprocal, calculate,
                 use_reciprocal=True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

        self.external_simplifiers = []

    def add_simplifier(self, simplifier, reason):
        self.external_simplifiers.append((reason, simplifier))

    def tracks(self):
        return [self.main, self.inverse, self.reciprocal]

    def get_num_denum(self, input):
        """
        This extract two lists, num and denum, such that the input is:
        self.inverse(self.main(\*num), self.main(\*denum)). It returns
        the two lists in a (num, denum) pair.

        For example, for main, inverse and reciprocal = \*, / and inv(),

        | input -> returned value (num, denum)

        | x*y -> ([x, y], [])
        | inv(x) -> ([], [x])
        | inv(x) * inv(y) -> ([], [x, y])
        | x*y/z -> ([x, y], [z])
        | log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        | (((a / b) * c) / d) -> ([a, c], [b, d])
        | a / (b / c) -> ([a, c], [b])
        | log(x) -> ([log(x)], [])
        | x**y -> ([x**y], [])
        | x * y * z -> ([x, y, z], [])

        """
        # This function is recursive.  The idea is that there is a
        # get_num_denum recursion in which the internal ops are all
        # one of (main, inverse, reciprocal, DimShuffle) and the
        # internal data nodes all have the dtype of the 'input'
        # argument. The leaf-Variables of the graph covered by the
        # recursion may be of any Variable type.

        if input.owner is None or input.owner.op not in [
                self.main, self.inverse, self.reciprocal]:
            if input.owner and isinstance(input.owner.op, T.DimShuffle):
                # If input is a DimShuffle of some input which does
                # something like this:

                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor
                #   with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return
                # the num/denum of its input
                dsn = input.owner    # dimshuffle node
                dsop = dsn.op        # dimshuffle op

                # the first input of the dimshuffle i.e. the ndarray to redim
                dsi0 = dsn.inputs[0]

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = (('x',) *
                                    (input.type.ndim - dsi0.type.ndim) +
                                    tuple(range(dsi0.type.ndim)))
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(input.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main,
                    # inverse or reciprocal.
                    return [input], []
            else:
                return [input], []
        num = []
        denum = []
        parent = input.owner

        # We get the (num, denum) pairs for each input
        # pairs = [self.get_num_denum(input2) if input2.type.dtype ==
        # input.type.dtype else ([input2], []) for input2 in
        # parent.inputs]
        pairs = [self.get_num_denum(input2) for input2 in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y, ...), numx, denumx, numy, denumy, ...
            # then num is concat(numx, numy, num...) and denum is
            # concat(denumx, denumy, denum...) note that main() can have any
            # number of arguments >= 0 concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is
            # concat(denumx, numy) note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        """
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(\*num),
        main(\*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        | n=0, d=0: neutral element (given by self.calculate([], []))
        |           (for example, this would be 0 if main is addition
        |           and 1 if main is multiplication)
        | n=1, d=0: num[0]
        | n=0, d=1: reciprocal(denum[0])
        | n=1, d=1: inverse(num[0], denum[0])
        | n=0, d>1: reciprocal(main(\*denum))
        | n>1, d=0: main(\*num)
        | n=1, d>1: inverse(num[0], main(\*denum))
        | n>1, d=1: inverse(main(\*num), denum[0])
        | n>1, d>1: inverse(main(\*num), main(\*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(\*num), main(\*denum))

        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return T.as_tensor_variable(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist=False)]
        if not ld:
            if ln == 1:
                # num[0] should always be a variable
                assert isinstance(num[0], gof.Variable)
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    @staticmethod
    def get_constant(v):
        """

        Returns
        -------
        object
            A numeric constant if v is a Constant or, well, a
            numeric constant. If v is a plain Variable, returns None.

        """
        if isinstance(v, Constant):
            if getattr(v.tag, 'unique_value', None) is not None:
                data = v.tag.unique_value
            else:
                data = v.data
            if data.ndim == 0:
                return data
            else:
                return None
        elif isinstance(v, Variable):
            return None
        else:
            return v

    def simplify(self, num, denum, out_type):
        """
        Shorthand for:

        .. code-block:: python

            self.simplify_constants(*self.simplify_factors(num, denum))

        """
        rval = self.simplify_constants(*self.simplify_factors(num, denum),
                                       out_type=out_type)
        for reason, simplifier in self.external_simplifiers:
            # TODO: document that 'reason' is associated with this
            #       simplification to help auditing when things go
            #       wrong
            rval = simplifier(*rval)
        return rval

    def simplify_factors(self, num, denum):
        """
        For any Variable r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        | [x], [x] -> [], []
        | [x, y], [x] -> [y], []
        | [a, b], [c, d] -> [a, b], [c, d]

        """
        ln = len(num)
        ld = len(denum)
        if (ld > 2 and ln > 2):
            # Faster version for "big" inputs.
            while True:
                s = set(num)
                # Inputs can appear multiple times
                redo = len(s) != len(num)
                inter = s.intersection(denum)
                for v in inter:
                    num.remove(v)
                    denum.remove(v)
                if not redo or not inter:
                    break
        else:
            for v in list(num):
                if v in denum:
                    num.remove(v)
                    denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum, out_type=None):
        """
        Find all constants and put them together into a single constant.

        Finds all constants in orig_num and orig_denum (using
        get_constant) and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator.

        Examples
        --------
        Let main be multiplication:

        | [2, 3, x], [] -> [6, x], []
        | [x, y, 2], [4, z] -> [0.5, x, y], [z]
        | [x, 2, y], [z, 2] -> [x, y], [z]

        """
        # Lists representing the numerator and denumerator
        num, denum = [], []

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []

        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                # We found a constant in the numerator!
                # We add it to numct
                numct.append(ct)
            else:
                num.append(v)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                denumct.append(ct)
            else:
                denum.append(v)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the
            # neutral element
            ct = self.calculate(numct, denumct, aslist=True,
                                out_type=out_type)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum
            # does it for us?
            ct = [self.calculate(numct, denumct, aslist=False,
                                 out_type=out_type)]

        # Wrapping ct in a Constant with the right dtype
        ct = [T.constant(c, dtype=out_type.dtype) for c in ct]

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct:
            # In that case we should only have one constant in `ct`.
            assert len(ct) == 1
            first_num_ct = self.get_constant(orig_num[0])
            if first_num_ct is not None and ct[0].type.values_eq(ct[0].data,
                                                                 first_num_ct):
                # This is an important trick :( if it so happens that:
                # * there's exactly one constant on the numerator and none on
                #   the denominator
                # * it's not the neutral element (ct is an empty list in that
                #   case)
                # * the constant is the same as the first argument in the
                #   numerator (we only check the first argument because the
                #   canonizer puts the computed constants first)
                # -> then we return very exactly the original num/denum.
                # If we don't do that the optimizer will just loop
                # infinitely because it will not catch on that there are
                # no changes to be made and every time it will want to
                # replace something by the same thing...
                # Note that it is important to use `values_eq` instead of
                # the == operator, to handle NaN values correctly.
                return orig_num, orig_denum

        return ct + num, denum

    def transform(self, node):
        op = node.op
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        assert len(node.outputs) == 1
        out = node.outputs[0]

        # out won't have a clients field when we didn't commit a
        # started change in the graph.  We can't do the check if we
        # want to skip it, so we force the skip it. It should be
        # reapplied later.
        if not hasattr(out, 'clients'):
            return

        # check if any of the clients of this node would be part of
        # this canonized graph...  if so, we do nothing and wait for
        # them to be transformed.
        for c, c_idx in out.clients:
            if c == 'output':
                continue
            while (isinstance(getattr(c, 'op', None), DimShuffle) and
                   len(c.outputs[0].clients) <= 1):
                c = c.outputs[0].clients[0][0]
            if getattr(c, 'op', '') in [self.main, self.inverse,
                                        self.reciprocal]:
                return False

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = self.simplify(list(orig_num), list(orig_denum), out.type)

        def same(x, y):
            return len(x) == len(y) and all(np.all(xe == ye) for xe, ye in
                                            zip(x, y))

        if same(orig_num, num) and same(orig_denum, denum):
            # We return False if there are no changes
            return False

        new = self.merge_num_denum(num, denum)
        if new.type.dtype != out.type.dtype:
            new = T.cast(new, out.type.dtype)

        assert (new.type == out.type) == (not (new.type != out.type))

        if not (new.type == out.type):
            new = _fill_chain(new, node.inputs)[0]

        if new.type == out.type:
            # This happen with test
            # theano/tensor/tests/test_opt.py:T_local_switch_sink
            new.tag.values_eq_approx = values_eq_approx_remove_inf_nan

            # We need to implement the copy over of the stacktrace.
            # See issue #5104.
            return [new]
        else:
            _logger.warning(' '.join(('CANONIZE FAILED: new, out = ',
                                      new, ',', out, 'types',
                                      new.type, ',', out.type)))
            return False

    def __str__(self):
        return getattr(self, 'name', 'Canonizer(%s, %s, %s)' % (
            self.main, self.inverse, self.reciprocal))


def mul_calculate(num, denum, aslist=False, out_type=None):
    if not num and not denum:
        # Smallest 1 possible.
        if aslist:
            return []
        else:
            return np.int8(1)

    # Make sure we do not accidentally upcast data types.
    if out_type is None:
        out_dtype = scalar.upcast(*[v.dtype for v in (num + denum)])
    else:
        out_dtype = out_type.dtype
    one = theano._asarray(1, dtype=out_dtype)

    v = reduce(np.multiply, num, one) / reduce(np.multiply, denum, one)
    if aslist:
        if np.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.true_div, T.inv, mul_calculate, False)
register_canonicalize(local_mul_canonizer, name='local_mul_canonizer')


@gof.local_optimizer([T.neg])
def local_neg_to_mul(node):
    if node.op == T.neg:
        return [T.mul(np.array(-1, dtype=node.inputs[0].dtype),
                node.inputs[0])]
register_canonicalize(local_neg_to_mul)


@register_specialize
@gof.local_optimizer([T.Sum, T.elemwise.Prod])
def local_sum_prod_mul_by_scalar(node):
    """
    sum(scalar * smth) -> scalar * sum(smth)
    sum(-smth) -> -sum(smth)

    or

    prod(scalar * smth) -> scalar ** size(smth) * prod(smth)
    prod(-smth) -> -1 ** size(smth) * prod(smth)

    """
    # TODO: if the the thing inside the Sum is a division,
    # we should get at the numerator....
    if isinstance(node.op, (T.Sum, T.elemwise.Prod)):
        node_inps, = node.inputs
        if node_inps.owner and node_inps.owner.op == T.mul:
            terms = node_inps.owner.inputs
            scalars = [t.dimshuffle() for t in terms if
                       np.all(t.type.broadcastable)]

            if len(scalars) == 0:
                # Nothing to optimize here
                return

            non_scalars = [t for t in terms if not np.all(t.broadcastable)]

            # Perform the op only on the non-scalar inputs, if applicable
            if len(non_scalars) == 0:
                new_op_input_nb_elements = 1
                new_op_output = 1
            elif len(non_scalars) == 1:
                new_op_input_nb_elements = non_scalars[0].size
                new_op_output = node.op(non_scalars[0])
            else:
                new_op_input = T.mul(*non_scalars)
                # We assume that errors always come from the prod/mul op in the
                # original computational graph, and therefore need to only
                # copy over its output stacktrace.
                copy_stack_trace(node.outputs, new_op_input)

                new_op_input_nb_elements = new_op_input.size
                new_op_output = node.op(new_op_input)

            if not len(non_scalars) == 0:
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, new_op_output)

            # If node.op is a T.elemwise.Prod, then the scalars need to be
            # raised to the power of the number of elements in the input
            # to the Prod
            if (isinstance(node.op, T.elemwise.Prod) and
                    new_op_input_nb_elements != 1):

                scalars = [s ** new_op_input_nb_elements for s in scalars]

            # Scale the output of the op by the scalars and return as
            # replacement for the original output
            mul_inputs = scalars
            if new_op_input_nb_elements != 1:
                mul_inputs.append(new_op_output)

            if len(mul_inputs) == 1:
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, mul_inputs)

                return mul_inputs
            else:
                ret = T.mul(*mul_inputs)
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, [ret] + mul_inputs)

                return [ret]

        if isinstance(node.op, T.Sum) and node_inps.owner and node_inps.owner.op == T.neg:
            s = node.op(node_inps.owner.inputs[0])
            ret = T.neg(s)
            # There are never errors in the negative op, thus
            # we need only to copy over stacktrace from previous output node to
            # the two new ops.
            copy_stack_trace(node.outputs, [s, ret])

            return [ret]


@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_elemwise_sub_zeros(node):
    """
    Elemwise{sub}(X,X) -> zeros_like(X)
    """
    if (isinstance(node.op, T.Elemwise) and
            node.op.scalar_op.nin == 2 and
            node.op.scalar_op == scalar.sub and
            node.inputs[0] == node.inputs[1]):
        res = T.zeros_like(node.inputs[0])
        # Copy over stacktrace from previous output.
        # This could help for failures due to out-of-memory.
        copy_stack_trace(node.outputs, res)
        return [res]


@register_useless
@register_specialize
@register_stabilize
@register_canonicalize
@gof.local_optimizer([T.Elemwise])
def local_useless_elemwise_comparison(node):
    """...

    :note: These cases appear in the graph generated by scan.
           These optimizations will make the graph easier to read.
    # Comparing to itself is constant
    Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    Elemwise[{minimum,maximum}](X, X) -> X

    # Comparing shape to 0 can be constant
    Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    Elemwise[maximum](0, X.shape[i]) -> X.shape[i]
    Elemwise[minimum](X.shape[i], 0) -> 0
    Elemwise[minimum](0, X.shape[i]) -> 0

    # The shape can be replaced with sum of shapes
    Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)

    # Shapes are never negative
    # Needed by Reshape.infer_shape
    Elemwise[EQ](Subtensor(Shape(x)), -N) -> Elemwise[zeros](X)

    """
    if not isinstance(node.op, T.Elemwise):
        return
    if node.op.scalar_op.nin != 2:
        return

    # We call zeros_like and one_like with opt=True to generate a
    # cleaner graph.
    dtype = node.outputs[0].dtype

    # Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    if isinstance(node.op.scalar_op, (scalar.LT, scalar.GT)) and \
       node.inputs[0] is node.inputs[1]:
        res = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    if isinstance(node.op.scalar_op, (scalar.LE, scalar.GE)) and \
       node.inputs[0] is node.inputs[1]:
        res = T.ones_like(node.inputs[0], dtype=dtype, opt=True)

        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[{minimum,maximum}](X, X) -> X
    if isinstance(node.op.scalar_op, (scalar.Minimum, scalar.Maximum)) and \
       node.inputs[0] is node.inputs[1]:
        res = node.inputs[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    if isinstance(node.op.scalar_op, scalar.LT) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Shape_i) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        res = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    if isinstance(node.op.scalar_op, scalar.GE) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Shape_i) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        res = T.ones_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    if isinstance(node.op.scalar_op, scalar.Maximum) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Shape_i) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        # No need to copy over stacktrace.
        return [node.inputs[0]]
    # Elemwise[maximum](0, X.shape[i]) -> X.shape[i]
    if isinstance(node.op.scalar_op, scalar.Maximum) and \
       T.extract_constant(node.inputs[0], only_process_constants=True) == 0 and \
       node.inputs[1].owner and \
       isinstance(node.inputs[1].owner.op, Shape_i):
        # No need to copy over stacktrace.
        return [node.inputs[1]]
    # Elemwise[minimum](X.shape[i], 0) -> 0
    if isinstance(node.op.scalar_op, scalar.Minimum) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Shape_i) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        res = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[minimum](0, X.shape[i]) -> 0
    if isinstance(node.op.scalar_op, scalar.Minimum) and \
       T.extract_constant(node.inputs[0], only_process_constants=True) == 0 and \
       node.inputs[1].owner and \
       isinstance(node.inputs[1].owner.op, Shape_i):
        res = T.zeros_like(node.inputs[1], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    if isinstance(node.op.scalar_op, scalar.LT) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Elemwise) and \
       isinstance(node.inputs[0].owner.op.scalar_op, scalar.Add) and \
       all([isinstance(var.owner and var.owner.op, Shape_i)
            for var in node.inputs[0].owner.inputs]) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        res = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)
    if isinstance(node.op.scalar_op, scalar.GE) and \
       node.inputs[0].owner and \
       isinstance(node.inputs[0].owner.op, Elemwise) and \
       isinstance(node.inputs[0].owner.op.scalar_op, scalar.Add) and \
       all([isinstance(var.owner and var.owner.op, Shape_i)
            for var in node.inputs[0].owner.inputs]) and \
       T.extract_constant(node.inputs[1], only_process_constants=True) == 0:
        res = T.ones_like(node.inputs[0], dtype=dtype, opt=True)

        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[EQ](Subtensor(Shape(x)), -N)
    # Elemwise[EQ](somegraph that only depend of shape, -N)
    # TODO: handle the case where the -N is on either side
        """
 |Elemwise{eq,no_inplace} [id B] ''
 | |Subtensor{int64} [id C] ''
 | | |Join [id D] ''
 | | | |TensorConstant{0} [id E]
 | | | |Subtensor{int64:int64:} [id F] ''
 | | | | |Shape [id G] ''
        """
    def investigate(node):
        " Return True if values will be shapes, so >= 0"
        if isinstance(node.op, (T.Shape, Shape_i)):
            return True
        elif isinstance(node.op, Subtensor) and node.inputs[0].owner:
            return investigate(node.inputs[0].owner)
        elif isinstance(node.op, T.Join):
            return all(v.owner and
                       investigate(v.owner) for v in node.inputs[1:])
        elif isinstance(node.op, MakeVector):
            return all(v.owner and
                       investigate(v.owner) for v in node.inputs)

    if (isinstance(node.op.scalar_op, scalar.EQ) and
            node.inputs[0].owner and
            investigate(node.inputs[0].owner)):
        try:
            cst = get_scalar_constant_value(node.inputs[1],
                                            only_process_constants=True)

            res = T.zeros_like(node.inputs[0], dtype=dtype, opt=True)

            if cst < 0:
                # Copy over stacktrace from previous output.
                copy_stack_trace(node.outputs, res)

                return [res]

        except NotScalarConstantError:
            pass
    return


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Sum, T.elemwise.Prod])
def local_sum_prod_div_dimshuffle(node):
    """
    sum(a / dimshuffle{...}(b), axis=l) -> sum(a, axis={...}) / b,
    if dimension l of the DimShuffle is 'x'

    or

    prod(a / dimshuffle{...}(b), axis=l) ->
    prod(a, axis={...}) / b ** a.shape[l],
    if dimension l of the DimShuffle is 'x'
    """

    # It does not make much sense now to extend it to the case where the
    # dimshuffle is in the numerator, since elemwise inversion of the
    # denominator would still be needed before the summation or production.

    if isinstance(node.op, (T.Sum, T.elemwise.Prod)):
        axis = node.op.axis
        if axis is None:
            axis = list(range(node.inputs[0].ndim))
        node_input = node.inputs[0]
        if node_input.owner and node_input.owner.op == T.true_div:
            numerator, denominator = node_input.owner.inputs

            # Old, bugged logic, reproduced here only to warn users
            if (config.warn.sum_div_dimshuffle_bug and
                    isinstance(node.op, T.Sum) and
                    numerator.owner and
                    isinstance(numerator.owner.op, T.DimShuffle)):
                # Check compatibility
                new_order = numerator.owner.op.new_order
                compatible_dims = True
                for ax in axis:
                    if len(new_order) <= ax or new_order[ax] != 'x':
                        compatible_dims = False
                        break

                if compatible_dims:
                    _logger.warn('WARNING: Your current code is fine, but'
                                 ' Theano versions between '
                                 'rev. 3bd9b789f5e8 (2010-06-16) and'
                                 ' cfc6322e5ad4 (2010-08-03) would '
                                 'have given an incorrect result. '
                                 'To disable this warning, set the Theano'
                                 ' flag warn.sum_div_dimshuffle_bug to'
                                 ' False.')

            if denominator.owner and isinstance(denominator.owner.op,
                                                T.DimShuffle):
                dimshuffle_input = denominator.owner.inputs[0]
                dimshuffle_order = denominator.owner.op.new_order

                compatible_dims = []
                incompatible_dims = []
                for ax in axis:
                    if (ax < len(dimshuffle_order) and
                            dimshuffle_order[ax] == 'x'):
                        compatible_dims.append(ax)
                    else:
                        incompatible_dims.append(ax)
                reordered_incompatible_dims = []
                for ic_ax in incompatible_dims:
                    reordered_incompatible_dims.append(
                        ic_ax - sum(
                            [1 for c_ax in compatible_dims if c_ax < ic_ax]))

                if len(compatible_dims) > 0:
                    optimized_dimshuffle_order = list(
                        ax for i, ax in enumerate(dimshuffle_order)
                        if (i not in axis) or (ax != 'x'))

                    # Removing leading 'x' (since it will be done automatically)
                    while (len(optimized_dimshuffle_order) > 0 and
                           optimized_dimshuffle_order[0] == 'x'):
                        del optimized_dimshuffle_order[0]

                    # if optimized_dimshuffle_order is sorted with
                    # not 'x', then dimshuffle is useless.
                    if all(i == e for i, e in
                           enumerate(optimized_dimshuffle_order)):
                        optimized_dimshuffle = dimshuffle_input
                    else:
                        optimized_dimshuffle = T.DimShuffle(
                            dimshuffle_input.type.broadcastable,
                            optimized_dimshuffle_order)(dimshuffle_input)

                        if (config.warn.sum_div_dimshuffle_bug and
                                isinstance(node.op, T.Sum)):
                            _logger.warn('WARNING: Your current code is fine,'
                                         ' but Theano versions between '
                                         'rev. 3bd9b789f5e8 (2010-06-16) and'
                                         ' cfc6322e5ad4 (2010-08-03) would '
                                         'have given an incorrect result. '
                                         'To disable this warning, set the'
                                         ' Theano flag '
                                         'warn.sum_div_dimshuffle_bug'
                                         ' to False.')

                    if isinstance(node.op, T.Sum):
                        op_on_compatible_dims = T.sum(
                            numerator, axis=compatible_dims)
                        rval = T.true_div(
                            op_on_compatible_dims,
                            optimized_dimshuffle)
                        if len(reordered_incompatible_dims) > 0:
                            rval = T.sum(rval,
                                         axis=reordered_incompatible_dims)
                    elif isinstance(node.op, T.elemwise.Prod):
                        op_on_compatible_dims = T.prod(
                            numerator, axis=compatible_dims)
                        dtype = numerator.dtype
                        rval = T.true_div(
                            op_on_compatible_dims,
                            (optimized_dimshuffle **
                                T.prod([numerator.shape[ax].astype(dtype)
                                        for ax in compatible_dims])))
                        if len(reordered_incompatible_dims) > 0:
                            rval = T.prod(rval,
                                          axis=reordered_incompatible_dims)
                    return [rval]


@register_canonicalize
@gof.local_optimizer([T.Sum, T.elemwise.Prod])
def local_sum_prod_all_to_none(node):
    """
    Sum{0,1,...N} -> Sum{} or
    Prod{0,1,...N} -> Prod{}

    """
    if isinstance(node.op, T.Sum) or isinstance(node.op, T.elemwise.Prod):
        opt_type = T.Sum if isinstance(node.op, T.Sum) else T.elemwise.Prod
        # if all the axes are named, then use None as a shorthand
        # this permits more merging
        if node.op.axis is None:
            return
        if set(node.op.axis) == set(range(node.inputs[0].type.ndim)):
            return [opt_type(axis=None, dtype=node.op.dtype)(node.inputs[0])]


@register_canonicalize
@gof.local_optimizer([T.Sum, T.elemwise.Prod])
def local_op_of_op(node):
    """
    Prod(Prod()) -> single Prod()
    or
    Sum(Sum()) -> single Sum()

    """
    if isinstance(node.op, T.elemwise.Prod) or isinstance(node.op, T.Sum):
        opt_type = T.Sum if isinstance(node.op, T.Sum) else T.elemwise.Prod
        node_inps, = node.inputs
        out_dtype = node.op.dtype
        # We manipulate the graph so this is done to make sure the opt
        # doesn't affect other computations.
        if len(node_inps.clients) == 1:
            if (node_inps.owner and
                    (isinstance(node_inps.owner.op, node.op.__class__))):

                # check to see either the inner or outer prod is doing a
                # product over all axis, in which case we can remove it
                if node_inps.owner.op.axis is None or node.op.axis is None:
                    return [opt_type(None, dtype=out_dtype)(
                        node_inps.owner.inputs[0])]

                # figure out which axes were in the original sum
                newaxis = list(tuple(node_inps.owner.op.axis))
                for i in node.op.axis:
                    new_i = i
                    for ii in node_inps.owner.op.axis:
                        if new_i >= ii:
                            new_i += 1
                    assert new_i not in newaxis
                    newaxis.append(new_i)

                assert len(newaxis) == len(list(node_inps.owner.op.axis) +
                                           list(node.op.axis))

                # The old bugged logic. We keep it there to generate a warning
                # when we generated bad code.
                alldims = list(range(node_inps.owner.inputs[0].type.ndim))
                alldims = [d for i, d in enumerate(alldims) if i
                           in node_inps.owner.op.axis]
                alldims = [d for i, d in enumerate(alldims)
                           if i in node.op.axis]
                newaxis_old = [i for i in
                               xrange(node_inps.owner.inputs[0].type.ndim)
                               if i not in alldims]

                if (theano.config.warn.sum_sum_bug and
                        newaxis != newaxis_old and
                        len(newaxis) == len(newaxis_old)):
                    _logger.warn(
                        "WARNING (YOUR CURRENT CODE IS FINE): Theano "
                        "versions between version 9923a40c7b7a and August "
                        "2nd, 2010 generated bugged code in this case. "
                        "This happens when there are two consecutive sums "
                        "in the graph and the intermediate sum is not "
                        "used elsewhere in the code. Some safeguard "
                        "removed some bad code, but not in all cases. You "
                        "are in one such case. To disable this warning "
                        "(that you can safely ignore since this bug has "
                        "been fixed) set the theano flag "
                        "`warn.sum_sum_bug` to False.")

                combined = opt_type(newaxis, dtype=out_dtype)
                return [combined(node_inps.owner.inputs[0])]


ALL_REDUCE = [T.elemwise.CAReduce, T.elemwise.All, T.elemwise.Any,
              T.elemwise.Sum, T.elemwise.Prod,
              T.elemwise.ProdWithoutZeros]


@register_canonicalize
@register_uncanonicalize  # Needed for MaxAndArgmax -> CAReduce
@gof.local_optimizer(ALL_REDUCE)
def local_reduce_join(node):
    """
    Reduce{scalar.op}(Join(axis=0, a, b), axis=0) -> Elemwise{scalar.op}(a, b)

    Notes
    -----
    Supported scalar.op are Maximum, Mimimum in some cases and Add and Mul in
    all cases.

    Currently we must reduce on axis 0. It is probably extensible to the case
    where we join and reduce on the same set of axis.

    """
    if (isinstance(node.op, T.CAReduce) and
            node.inputs[0].owner and
            isinstance(node.inputs[0].owner.op, T.Join)):
        join = node.inputs[0].owner
        if T.extract_constant(join.inputs[0], only_process_constants=True) != 0:
            return

        if isinstance(node.op.scalar_op, (scalar.Maximum, scalar.Minimum)):
            # Support only 2 inputs for now
            if len(join.inputs) != 3:
                return
        elif not isinstance(node.op.scalar_op, (scalar.Add, scalar.Mul)):
            return
        elif len(join.inputs) <= 2:
            # This is a useless join, that will get removed by another opt.
            return

        new_inp = []
        for inp in join.inputs[1:]:
            inp = inp.owner
            if not inp:
                return
            if (not isinstance(inp.op, DimShuffle) or
                    inp.op.new_order != ('x',) +
                    tuple(range(inp.inputs[0].ndim))):
                return
            new_inp.append(inp.inputs[0])
        ret = Elemwise(node.op.scalar_op)(*new_inp)

        if ret.dtype != node.outputs[0].dtype:
            # The reduction do something about the dtype.
            return

        reduce_axis = node.op.axis
        if reduce_axis is None:
            reduce_axis = tuple(xrange(node.inputs[0].ndim))

        # I put this warning late to don't add extra warning.
        if len(reduce_axis) != 1 or 0 not in reduce_axis:
            if theano.config.warn.reduce_join:
                warnings.warn((
                    'Your current code is fine, but Theano versions '
                    'prior to 0.7 (or this development version Sept 2014) '
                    'might have given an incorrect result for this code. '
                    'To disable this warning, set the Theano flag '
                    'warn.reduce_join to False. The problem was an '
                    'optimization, that modified the pattern '
                    '"Reduce{scalar.op}(Join(axis=0, a, b), axis=0)", '
                    'did not check the reduction axis. So if the '
                    'reduction axis was not 0, you got a wrong answer.'))
            return

        # We add the new check late to don't add extra warning.
        try:
            join_axis = get_scalar_constant_value(join.inputs[0],
                                                  only_process_constants=True)

            if join_axis != reduce_axis[0]:
                return
        except NotScalarConstantError:
            return

        return [ret]


@register_canonicalize('fast_compile', 'local_cut_useless_reduce')
@register_useless('local_cut_useless_reduce')
@gof.local_optimizer(ALL_REDUCE)
def local_useless_reduce(node):
    """Sum(a, axis=[]) -> a  """
    if isinstance(node.op, T.CAReduce):
        summed, = node.inputs
        # if reduce were doing anything, the output ndim would be reduced
        if summed.type == node.outputs[0].type:
            return [summed]


@register_canonicalize
@register_uncanonicalize
@register_specialize
@gof.local_optimizer(ALL_REDUCE)
def local_reduce_broadcastable(node):
    """Remove reduction over broadcastable dimensions."""
    if isinstance(node.op, T.CAReduce):
        reduced, = node.inputs
        odtype = node.outputs[0].dtype
        if node.op.axis is None:
            if all(reduced.broadcastable):
                return [reduced.dimshuffle().astype(odtype)]
        else:
            axis = list(node.op.axis)
            cuttable = [a for a in axis if reduced.broadcastable[a]]
            if cuttable:
                # -- we can remove some axes of summation,
                #    which simplifies the codegen for sum, especially on GPU
                new_axis = []
                pattern = []
                ii = 0
                for p in xrange(reduced.ndim):
                    if p not in cuttable:
                        if p in axis:
                            new_axis.append(ii)
                        pattern.append(p)
                        ii += 1
                new_reduced = reduced.dimshuffle(*pattern)
                if new_axis:
                    if type(node.op) == theano.tensor.elemwise.CAReduce:
                        # This happen for tensor.max(), tensor.min()
                        new_op = node.op.__class__(node.op.scalar_op,
                                                   axis=new_axis)
                    else:
                        new_op = node.op.__class__(axis=new_axis)
                    return [new_op(new_reduced)]
                else:
                    # -- in this case we can remove the reduction completely
                    return [new_reduced.astype(odtype)]


@register_specialize
@gof.local_optimizer([T.Sum, T.elemwise.Prod])
def local_opt_alloc(node):
    """
    sum(alloc(constant,shapes...)) => constant*prod(shapes)
    or
    prod(alloc(constant,shapes...)) => constant**prod(shapes)

    """
    if isinstance(node.op, T.Sum) or isinstance(node.op, T.elemwise.Prod):
        node_inps, = node.inputs
        if node_inps.owner and isinstance(node_inps.owner.op, T.Alloc):
            input = node_inps.owner.inputs[0]
            shapes = node_inps.owner.inputs[1:]
            try:
                val = get_scalar_constant_value(input,
                                                only_process_constants=True)
                assert val.size == 1
                val = val.reshape(1)[0]
                # check which type of op
                size = T.mul(*shapes)
                if input.dtype in ["float16", "float32"]:
                    # shapes are ints and normally int64.
                    # We don't want to have a float64 upcast
                    # We don't want to downcast to float16
                    # as we fear it could loose too much precision
                    # that will be amplified by the mul/pow below.
                    size = size.astype('float32')
                if (node.op.axis is None or
                        node.op.axis == tuple(range(input.ndim))):
                    if isinstance(node.op, T.Sum):
                        val = val * size
                    else:
                        val = val ** size
                    # Sum can change the input dtype (upcast or bool
                    # -> float32) by default or by user request.
                    # We can ignore the acc_dtype, as there is only 1
                    # elemwise we will do and not a sequence, so there is no
                    # accumulation of errors.
                    # So mostly, we just need to cast the output to the old
                    # dtype.
                    val = val.astype(node.outputs[0].dtype)
                    return [val]
                to_prod = [shapes[i] for i in xrange(len(shapes))
                           if i in node.op.axis]
                if to_prod:
                    size = T.mul(*to_prod)
                    if isinstance(node.op, T.Sum):
                        val *= size
                    else:
                        val = val ** size
                # See comments above.
                val = val.astype(node.outputs[0].dtype)
                return [T.alloc(val,
                                *[shapes[i] for i in xrange(len(shapes))
                                  if i not in node.op.axis])]
            except NotScalarConstantError:
                pass


@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_neg(node):
    # other specializations shouldn't put this in,
    # but sometimes they do
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.neg:
            return [node.inputs[0].owner.inputs[0]]


@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_div_neg(node):
    """
    - (-a / b) -> a / b

    Also performs - (c / b) -> ((-c) / b) when c is a scalar constant.

    """
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.true_div:
            frac = node.inputs[0]
            num, denom = frac.owner.inputs
            if num.owner and num.owner.op == T.neg:
                if len(frac.clients) == 1:
                    # No other clients of the original division
                    new_num = num.owner.inputs[0]
                    return [T.true_div(new_num, denom)]
            elif np.all(num.broadcastable) and isinstance(num, Constant):
                if len(frac.clients) == 1:
                    new_num = -num.data
                    return [T.true_div(new_num, denom)]


@gof.local_optimizer([T.mul])
def local_mul_zero(node):
    """
    As part of canonicalization, we replace multiplication by zero
    with zero.

    """
    if node.op == T.mul:
        otype = node.outputs[0].type

        for i in node.inputs:
            try:
                value = get_scalar_constant_value(i)
            except NotScalarConstantError:
                continue
            # print 'MUL by value', value, node.inputs
            if value == 0:
                # print '... returning zeros'
                return _fill_chain(theano._asarray(0, dtype=otype.dtype),
                                   node.inputs)
register_canonicalize(local_mul_zero)


@gof.local_optimizer([T.true_div])
def local_div_to_inv(node):
    if node.op == T.true_div and np.all(
            local_mul_canonizer.get_constant(node.inputs[0]) == 1.0):
        out = node.outputs[0]
        new_out = T.inv(local_mul_canonizer.merge_num_denum(node.inputs[1:],
                                                            []))
        # The ones could have forced upcasting
        if new_out.dtype != out.dtype:
            new_out = T.cast(new_out, dtype=out.dtype)
        # The ones could have forced a specific length
        if new_out.type != out.type:
            new_out = broadcast_like(new_out, out, node.fgraph)
        return [new_out]
    else:
        return False
register_specialize(local_div_to_inv)


@gof.local_optimizer([T.inv])
def local_inv_canon(node):
    if node.op == T.inv:
        return [T.pow(node.inputs[0], -1.0)]
    else:
        return False
register_canonicalize(local_inv_canon)


@gof.local_optimizer([T.pow])
def local_pow_canonicalize(node):
    if node.op == T.pow:
        cst = local_mul_canonizer.get_constant(node.inputs[1])
        if cst == 0:
            return [broadcast_like(1, node.outputs[0], node.fgraph)]
        if cst == 1:
            return [broadcast_like(node.inputs[0], node.outputs[0], node.fgraph)]
    else:
        return False
register_canonicalize(local_pow_canonicalize)


@register_specialize
@gof.local_optimizer([T.mul])
def local_mul_to_sqr(node):
    """
    x*x -> sqr(x)

    This is faster on the GPU when memory fetching is a big part of
    the computation time.

    """
    if node.op == T.mul:
        if len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [T.sqr(node.inputs[0])]


@register_canonicalize
@gof.local_optimizer([T.int_div])
def local_intdiv_by_one(node):
    """x // 1 -> x
    """
    if node.op in [T.int_div]:
        if isinstance(node.inputs[1], T.TensorConstant) and \
           np.all(node.inputs[1].value == 1):
            return [node.inputs[0].astype(node.outputs[0].dtype)]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.int_div, T.true_div])
def local_zero_div(node):
    """0 / x -> 0
    """
    if isinstance(node.op, T.Elemwise) and isinstance(
            node.op.scalar_op, (theano.scalar.IntDiv, theano.scalar.TrueDiv)):
        if local_mul_canonizer.get_constant(node.inputs[0]) == 0:
            ret = broadcast_like(0, node.outputs[0], node.fgraph)
            ret.tag.values_eq_approx = values_eq_approx_remove_nan
            return [ret]


@gof.local_optimizer([T.pow])
def local_pow_specialize(node):
    # here, we are past the point of canonicalization, so we don't want
    # to put in un-necessary fills.
    if node.op == T.pow:
        # the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable,
                                              ysym.type.broadcastable):
            rval = None

            if np.all(y == 2):
                rval = [T.sqr(xsym)]
            if np.all(y == 1):
                rval = [xsym]
            if np.all(y == 0):
                rval = [T.fill(xsym, np.asarray(1, dtype=odtype))]
            if np.all(y == 0.5):
                rval = [T.sqrt(xsym)]
            if np.all(y == -0.5):
                rval = [T.inv(T.sqrt(xsym))]
            if np.all(y == -1):
                rval = [T.inv(xsym)]
            if np.all(y == -2):
                rval = [T.inv(T.sqr(xsym))]
            if rval:
                rval[0] = T.cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (
                    rval, node.outputs)
                return rval
    else:
        return False
register_specialize(local_pow_specialize)


@register_specialize_device
@gof.local_optimizer([T.pow])
def local_pow_specialize_device(node):
    """
    This optimization is not the same on all device. We do it only on cpu here.
    """
    if node.op == T.pow:
        # the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)

        # the next line is needed to fix a strange case that I don't
        # know how to make a separate test.
        # That happen in the test_opt.py:test_log_erfc test.
        # y is a ndarray with dtype int8 and value 2,4 or 6. This make
        # the abs(y) <= 512 fail!
        # taking the value outside ndarray solve the problem.
        # it could be that in that case, numpy make the comparaison
        # into the wrong type(do in int8 that overflow.)
        if isinstance(y, np.ndarray):
            assert y.size == 1
            try:
                y = y[0]
            except IndexError:
                pass
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable,
                                              ysym.type.broadcastable):
            rval = None
            # 512 is too small for the cpu and too big for some gpu!
            if abs(y) == int(abs(y)) and abs(y) <= 512:
                pow2 = [xsym]
                pow2_scal = [theano.scalar.get_scalar_type(xsym.dtype)()]
                y_to_do = abs(y)
                for i in xrange(int(np.log2(y_to_do))):
                    pow2.append(T.sqr(pow2[i]))
                    pow2_scal.append(theano.scalar.sqr(pow2_scal[i]))
                rval1 = None
                rval1_scal = None
                while y_to_do > 0:
                    log_to_do = int(np.log2(y_to_do))
                    if rval1:
                        rval1 *= pow2[log_to_do]
                        rval1_scal *= pow2_scal[log_to_do]
                    else:
                        rval1 = pow2[log_to_do]
                        rval1_scal = pow2_scal[log_to_do]
                    y_to_do -= 2 ** log_to_do

                if abs(y) > 2:
                    # We fuse all the pow together here to make
                    # compilation faster
                    rval1 = Elemwise(
                        theano.scalar.Composite(
                            [pow2_scal[0]], [rval1_scal])).make_node(xsym)
                if y < 0:
                    rval = [T.inv(rval1)]
                else:
                    rval = [rval1]
            if rval:
                rval[0] = T.cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (
                    rval, node.outputs)
                return rval


@gof.local_optimizer([T.mul])
def local_mul_specialize(node):
    """
    Remove special-case constants from mul arguments and useless neg in inputs.

    mul(-1, x) -> neg(x)
    mul(1, x, y) -> mul(x, y)
    mul(0, ...) -> alloc(0, shapes...)

    This is not done if we would add more nodes in the graph, like with:

    mul(-1, x, y) -/-> neg(mul(x, y))

    """
    # here, we are past the point of canonicalization, so we don't
    # want to put in un-necessary fills.
    #
    # at this point [post canonicalize], mul() may have many inputs.
    if node.op == T.mul:
        # the idea here is that we have pow(x, y)
        neg = False
        new_inputs = []
        nb_neg_node = 0
        nb_cst = 0
        for input in node.inputs:
            # remove any neg arguments
            while input.owner and input.owner.op == T.neg:
                neg ^= True
                input = input.owner.inputs[0]
                nb_neg_node += 1

            # remove special case arguments of 1, -1 or 0
            y = local_mul_canonizer.get_constant(input)
            if y == 1.0:
                nb_cst += 1
            elif y == -1.0:
                nb_cst += 1
                neg ^= True  # toggles
            elif y == 0.0:
                # if we find any zero, we just return right away
                return [broadcast_like(0, node.outputs[0], node.fgraph)]
            else:
                new_inputs.append(input)

        if new_inputs != node.inputs:
            if new_inputs:
                if len(new_inputs) == 1:
                    if neg:
                        if new_inputs[0].dtype in (T.uint_dtypes + ['bool']):
                            return
                        else:
                            rval = -new_inputs[0]
                    else:
                        rval = new_inputs[0]
                else:
                    # The next case would cause a replace by an equivalent case.
                    if (neg and
                            nb_neg_node == 0 and
                            nb_cst == 1):
                        return
                    elif neg:
                        # Don't add an extra neg node as we can't
                        # fully replace this mul by a neg.
                        m1 = np.asarray(-1, dtype=node.outputs[0].dtype)
                        new_inputs = [m1] + new_inputs
                    rval = T.mul(*new_inputs)

                return [broadcast_like(rval, node.outputs[0], node.fgraph)]
            else:
                # there are no variable inputs to mul
                # N.B. this could have been constant-folded...
                if neg:
                    return [broadcast_like(-1, node.outputs[0], node.fgraph)]
                else:
                    return [broadcast_like(1, node.outputs[0], node.fgraph)]

register_specialize(local_mul_specialize)


@gof.local_optimizer([T.add])
def local_add_specialize(node):
    def fill_chain(v):
        out = _fill_chain(v, node.inputs)
        return out

    # here, we are past the point of canonicalization, so we don't want
    # to put in un-necessary fills.
    if node.op == T.add:
        new_inputs = []
        for input in node.inputs:
            try:
                y = get_scalar_constant_value(input)
            except NotScalarConstantError:
                y = input
            if np.all(y == 0.0):
                continue
            new_inputs.append(input)

        if len(new_inputs) < len(node.inputs):
            dtype = node.outputs[0].type.dtype
            if len(new_inputs) == 0:
                # we got rid of the entire expression!
                ndim = node.outputs[0].type.ndim
                # Reuse call to constant for cache()
                cst = T.constant(np.zeros((1,) * ndim, dtype=dtype))
                assert cst.type.broadcastable == (True,) * ndim
                return fill_chain(cst)

            if len(new_inputs) == 1:
                ret = fill_chain(new_inputs[0])
            else:
                ret = fill_chain(T.add(*new_inputs))
            # The dtype should not be changed. It can happen if the input
            # that was forcing upcasting was equal to 0.
            if ret[0].dtype != dtype:
                ret = [T.cast(ret[0], dtype)]
            return ret
    else:
        return False
register_specialize(local_add_specialize)

mul_canonizer = in2out(gof.LocalOptGroup(local_mul_canonizer,
                                         local_fill_sink, apply_all_opts=True),
                       name='mul_canonizer_groups')


def check_for_x_over_absX(numerators, denominators):
    """Convert x/abs(x) into sign(x). """
    # TODO: this function should dig/search through dimshuffles
    # This won't catch a dimshuffled absolute value
    for den in list(denominators):
        if (den.owner and den.owner.op == T.abs_ and
                den.owner.inputs[0] in numerators):
            if den.owner.inputs[0].type.dtype.startswith('complex'):
                # TODO: Make an Op that projects a complex number to
                #      have unit length but projects 0 to 0.  That
                #      would be a weird Op, but consistent with the
                #      special case below.  I heard there's some
                #      convention in Matlab that is similar to
                #      this... but not sure.
                pass
            else:
                denominators.remove(den)
                numerators.remove(den.owner.inputs[0])
                numerators.append(T.sgn(den.owner.inputs[0]))
    return numerators, denominators
local_mul_canonizer.add_simplifier(check_for_x_over_absX, 'X_over_absX')


@register_canonicalize
@gof.local_optimizer([T.abs_])
def local_abs_lift(node):
    """
    Move the abs toward the input.

    This is needed for check_for_x_over_absX to apply in more case.

    """
    if node.op == T.abs_ and node.inputs[0].owner:
        assert node.nin == 1
        if node.inputs[0].owner.op == T.mul:
            return [T.mul(*[T.abs_(i) for i in node.inputs[0].owner.inputs])]
        if node.inputs[0].owner.op == T.true_div:
            i = node.inputs[0].owner.inputs
            return [T.true_div(T.abs_(i[0]), T.abs_(i[1]))]


@register_specialize
@gof.local_optimizer([T.mul, T.true_div])
def local_abs_merge(node):
    """
    Merge abs generated by local_abs_lift when the canonizer don't
    need it anymore

    """
    if node.op == T.mul and sum([i.owner.op == T.abs_ for i in node.inputs
                                 if i.owner]) > 1:
        inputs = []
        for i in node.inputs:
            if i.owner and i.owner.op == T.abs_:
                inputs.append(i.owner.inputs[0])
            elif isinstance(i, Constant):
                try:
                    const = get_scalar_constant_value(i,
                                                      only_process_constants=True)
                except NotScalarConstantError:
                    return False
                if not (const >= 0).all():
                    return False
                inputs.append(i)
            else:
                return False
        return [T.abs_(T.mul(*inputs))]
    if node.op == T.true_div and sum([i.owner.op == T.abs_ for i in
                                      node.inputs if i.owner]) == 2:
        return [T.abs_(T.true_div(node.inputs[0].owner.inputs[0],
                                  node.inputs[1].owner.inputs[0]))]


@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log1p(node):
    # log(1+x) -> log1p(x)
    # log(1-x) -> log1p(-x)
    if node.op == T.log:
        log_arg, = node.inputs
        if log_arg.owner and log_arg.owner.op == T.add:
            scalars, scalar_inputs, nonconsts = scalarconsts_rest(
                log_arg.owner.inputs, only_process_constants=True)
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if scalars and np.allclose(np.sum(scalars), 1):
                if nonconsts:
                    if len(nonconsts) > 1:
                        ninp = T.add(*nonconsts)
                    else:
                        ninp = nonconsts[0]
                    if ninp.dtype != log_arg.type.dtype:
                        ninp = ninp.astype(node.outputs[0].dtype)
                    return _fill_chain(T.log1p(ninp), scalar_inputs)

        elif log_arg.owner and log_arg.owner.op == T.sub:
            one = T.extract_constant(log_arg.owner.inputs[0],
                                     only_process_constants=True)
            if one != 1:
                return
            other = log_arg.owner.inputs[1]
            if other.dtype != log_arg.dtype:
                other = other.astype(log_arg.dtype)
            return [T.log1p(T.neg(other))]


# TODO: in canonicalize, change log10 and log2 -> log
@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log_add(node):
    # log(exp(x)+exp(y))
    #
    # Suppose x >= y
    # log(exp(x) + exp(y))
    # log(exp(x) * (1 + exp(y)/exp(x)))
    # x + log(1 + exp(y)/exp(x))
    # x + log1p(exp(y)/exp(x))
    # x + log1p(exp(y-x))
    if node.op == T.log:
        z = node.inputs[0]
        if z.owner and z.owner.op == T.add:
            zi = z.owner.inputs
            if len(zi) != 2:
                # -- upgrading Maximum to handle multiple inputs wasn't trivial
                #    TODO
                # raise NotImplementedError()
                return
            pre_exp = [x.owner.inputs[0] for x in zi
                       if x.owner and x.owner.op == T.exp]
            if len(pre_exp) == len(zi):
                # all arguments to add are exp(<something>)
                max_pre = T.maximum(*pre_exp)

                ret = max_pre + T.log1p(T.exp(T.add(*[p - max_pre
                                                      for p in pre_exp])))
                ret.tag.values_eq_approx = values_eq_approx_remove_inf
                return [ret]


@gof.local_optimizer([T.log])
def local_log_sum_exp(node):
    # log(sum_i(exp(x_i))) = x_max + log(sum_i(exp(x_i - x_max)))

    if node.op != T.log:
        return

    sum_node = node.inputs[0].owner
    # If the sum has keepdims=True, there might be a dimshuffle
    if sum_node and isinstance(sum_node.op, T.DimShuffle):
        dimshuffle_op = sum_node.op
        sum_node = sum_node.inputs[0].owner
    else:
        dimshuffle_op = None

    if not sum_node or not isinstance(sum_node.op, T.Sum):
        return

    exp_node, axis = sum_node.inputs[0].owner, sum_node.op.axis
    if not exp_node or not (
            isinstance(exp_node.op, Elemwise) and
            isinstance(exp_node.op.scalar_op, scalar.Exp)):
        return

    pre_exp = exp_node.inputs[0]
    max_pre_exp = T.max(pre_exp, axis=axis)
    max_pre_exp_keepdims = T.makeKeepDims(pre_exp, max_pre_exp, axis)

    ret = (max_pre_exp +
           T.log(T.sum(T.exp(pre_exp - max_pre_exp_keepdims), axis=axis)))

    # Restore the dimshuffle op, if any.
    if dimshuffle_op:
        ret = dimshuffle_op(ret)

    return [ret]


compile.optdb.register('local_log_sum_exp',
                       in2out(local_log_sum_exp, ignore_newtrees=True),
                       1.6, 'fast_run')


def add_calculate(num, denum, aslist=False, out_type=None):
    # TODO: make sure that this function and mul_calculate are similar
    if out_type is None:
        zero = 0.0
    else:
        zero = theano._asarray(0, dtype=out_type.dtype)
    # zero = 0.0 if out_type is None else theano._asarray(0,
    # dtype=out_type.dtype)
    if out_type and out_type.dtype == 'bool':
        if len(denum) == 0:
            # NumPy 1.14 do not accept to do "bool - bool"
            v = reduce(np.add, num, zero)
        else:
            raise Exception(
                "bool subtraction not supported. This should not happen as"
                " an earlier error should have been raised")
    else:
        v = reduce(np.add, num, zero) - reduce(np.add, denum, zero)
    if aslist:
        if np.all(v == 0):
            return []
        else:
            return [v]
    return v


local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = in2out(gof.LocalOptGroup(local_add_canonizer,
                                         local_fill_sink, apply_all_opts=True),
                       name='add_canonizer_group')


register_canonicalize(local_add_canonizer, name='local_add_canonizer')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum,
                      out_type, minscore=0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    # score is number of operations saved, higher is better
    score = len(num) + div_cost * len(denum)
    new_pos_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n + num, d + denum, out_type) for (n, d)
                                            in pos_pairs]))
    new_neg_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n + num, d + denum, out_type) for (n, d)
                                            in neg_pairs]))
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs +
                                new_neg_pairs):
        # We calculate how many operations we are saving with the new
        # num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs


def attempt_distribution(factor, num, denum, out_type):
    # we try to insert each num and each denum in the factor
    # returns: changes?, new_factor, new_num, new_denum
    # if there are changes, new_num and new_denum contain all the numerators
    # and denumerators that could not be distributed in the factor
    pos, neg = local_add_canonizer.get_num_denum(factor)
    if len(pos) == 1 and not neg:
        return False, factor, num, denum
    pos_pairs = list(map(local_mul_canonizer.get_num_denum, pos))
    neg_pairs = list(map(local_mul_canonizer.get_num_denum, neg))
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs,
                                                          neg_pairs, [n], [], out_type)
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs,
                                                          neg_pairs, [], [d], out_type)
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return change, local_add_canonizer.merge_num_denum(
            list(itertools.starmap(local_mul_canonizer.merge_num_denum,
                                   pos_pairs)),
            list(itertools.starmap(local_mul_canonizer.merge_num_denum,
                                   neg_pairs))), num, denum


@register_canonicalize
@register_stabilize
@gof.local_optimizer([T.mul, T.true_div, T.inv])
def local_greedy_distributor(node):
    """
    Optimize by reducing the number of multiplications and/or divisions.

    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x
    3. There are other forms too where node is a true_div.

    The following expressions are not simplified:
    4. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.

    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    out_type = out.type
    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(
            candidate, num, denum, out_type,)

        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(
            candidate, denum, num, out_type)
        change |= _change
        new_denum.append(candidate)
    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if not (rval.type == out.type):
        # WHY DOES THIS HAPPEN?
        return False

    return [rval]


@gof.local_optimizer(None)
def constant_folding(node):
    for input in node.inputs:
        if not isinstance(input, Constant):
            return False
    # condition:  all inputs are constant
    if not node.op.do_constant_folding(node):
        # The op asks not to be constant folded.
        return False

    storage_map = dict([(i, [i.data]) for i in node.inputs])
    compute_map = dict([(i, [True]) for i in node.inputs])
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]
    impl = None
    if (hasattr(node.op, 'python_constant_folding') and
            node.op.python_constant_folding(node)):
        impl = 'py'
    thunk = node.op.make_thunk(node, storage_map, compute_map,
                               no_recycling=[], impl=impl)

    required = thunk()
    assert not required  # a node whose inputs are all provided should always
    # return successfully
    rval = []
    for output in node.outputs:
        assert compute_map[output][0], (output, storage_map[output][0])
        try:
            constant = output.type.Constant
        except AttributeError:
            constant = Constant

        v = constant(output.type, storage_map[output][0])
        copy_stack_trace(output, v)

        rval.append(v)
    return rval


topo_constant_folding = in2out(constant_folding, ignore_newtrees=True,
                               name="topo_constant_folding")
register_canonicalize(topo_constant_folding, 'fast_compile', final_opt=True)
register_uncanonicalize(topo_constant_folding, 'fast_compile', final_opt=True)
register_stabilize(topo_constant_folding, 'fast_compile', final_opt=True)
register_specialize(topo_constant_folding, 'fast_compile', final_opt=True)


def get_clients(node):
    """
    Used by erf/erfc opt to track less frequent op.

    """
    return [c for c, i in node.outputs[0].clients
            if c != "output"]


def get_clients2(node):
    """
    Used by erf/erfc opt to track less frequent op.

    """
    l = []
    for c, i in node.outputs[0].clients:
        if c != "output":
            for var in c.outputs:
                l.extend([cc for cc, ii in var.clients if cc != "output"])
    return l

# 1+erf(x)=>erfc(-x)
local_one_plus_erf = gof.PatternSub((T.add,
                                     1,
                                     (T.erf, 'x')),
                                    (T.erfc, (T.neg, 'x')),
                                    allow_multiple_clients=True,
                                    name='local_one_plus_erf',
                                    tracks=[T.erf],
                                    get_nodes=get_clients)
register_canonicalize(local_one_plus_erf)
register_stabilize(local_one_plus_erf)
register_specialize(local_one_plus_erf)

# 1-erf(x)=>erfc(x)
local_one_minus_erf = gof.PatternSub((T.sub,
                                      1,
                                      (T.erf, 'x')),
                                     (T.erfc, 'x'),
                                     allow_multiple_clients=True,
                                     name='local_one_minus_erf',)
register_canonicalize(local_one_minus_erf)
register_stabilize(local_one_minus_erf)
register_specialize(local_one_minus_erf)

local_one_minus_erf2 = gof.PatternSub((T.add,
                                      1,
                                      (T.mul, -1, (T.erf, 'x'))),
                                      (T.erfc, 'x'),
                                      allow_multiple_clients=True,
                                      name='local_one_minus_erf2')
register_canonicalize(local_one_minus_erf2)
register_stabilize(local_one_minus_erf2)
register_specialize(local_one_minus_erf2)

# 1+(-erf(x))=>erfc(x) This is a different graph then the previous as
# the canonicalize don't work completly
local_one_plus_neg_erf = gof.PatternSub((T.add,
                                         1,
                                         (T.neg, (T.erf, 'x'))),
                                        (T.erfc, 'x'),
                                        allow_multiple_clients=True,
                                        name='local_one_plus_neg_erf',
                                        tracks=[T.erf],
                                        get_nodes=get_clients2)
register_canonicalize(local_one_plus_neg_erf)
register_stabilize(local_one_plus_neg_erf)
register_specialize(local_one_plus_neg_erf)

# (-1)+erf(x) => -erfc(x) don't need erf(x)+(-1) as the canonicalize
# will put the -1 as the first argument.
local_erf_minus_one = gof.PatternSub((T.add,
                                      -1,
                                      (T.erf, 'x')),
                                     (T.neg, (T.erfc, 'x')),
                                     allow_multiple_clients=True,
                                     name='local_erf_minus_one',
                                     tracks=[T.erf],
                                     get_nodes=get_clients)
register_canonicalize(local_erf_minus_one)
register_stabilize(local_erf_minus_one)
register_specialize(local_erf_minus_one)

# 1-erfc(x) => erf(x)
local_one_minus_erfc = gof.PatternSub((T.sub,
                                       1,
                                       (T.erfc, 'x')),
                                      (T.erf, 'x'),
                                      allow_multiple_clients=True,
                                      name='local_one_minus_erfc',
                                      tracks=[T.erfc],
                                      get_nodes=get_clients)
register_canonicalize(local_one_minus_erfc)
register_stabilize(local_one_minus_erfc)
register_specialize(local_one_minus_erfc)

local_one_minus_erfc2 = gof.PatternSub((T.add,
                                        1,
                                        (T.neg, (T.erfc, 'x'))),
                                       (T.erf, 'x'),
                                       allow_multiple_clients=True,
                                       name='local_one_minus_erfc2',
                                       tracks=[T.erfc],
                                       get_nodes=get_clients2)
register_canonicalize(local_one_minus_erfc2)
register_stabilize(local_one_minus_erfc2)
register_specialize(local_one_minus_erfc2)

local_one_minus_erfc3 = gof.PatternSub((T.add,
                                        1,
                                        (T.mul, -1, (T.erfc, 'x'))),
                                       (T.erf, 'x'),
                                       allow_multiple_clients=True,
                                       name='local_one_minus_erfc3',
                                       tracks=[T.erfc],
                                       get_nodes=get_clients2)
register_canonicalize(local_one_minus_erfc3)
register_stabilize(local_one_minus_erfc3)
register_specialize(local_one_minus_erfc3)

# 1+(-erfc(x)) => erf(x) This is a different graph then the previous as
# the canonicalize don't work completly
local_one_add_neg_erfc = gof.PatternSub((T.add,
                                         1,
                                         (T.neg, (T.erfc, 'x'))),
                                        (T.erf, 'x'),
                                        allow_multiple_clients=True,
                                        name='local_one_add_neg_erfc',
                                        tracks=[T.erfc],
                                        get_nodes=get_clients2)

register_canonicalize(local_one_add_neg_erfc)
register_stabilize(local_one_add_neg_erfc)
register_specialize(local_one_add_neg_erfc)

# (-1)+erfc(-x)=>erf(x)
local_erf_neg_minus_one = gof.PatternSub((T.add,
                                          -1,
                                          (T.erfc, (T.neg, 'x'))),
                                         (T.erf, 'x'),
                                         allow_multiple_clients=True,
                                         name='local_erf_neg_minus_one',
                                         tracks=[T.erfc],
                                         get_nodes=get_clients)
register_canonicalize(local_erf_neg_minus_one)
register_stabilize(local_erf_neg_minus_one)
register_specialize(local_erf_neg_minus_one)

# (-1)+erfc(-1*x)=>erf(x)
local_erf_neg_minus_one2 = gof.PatternSub((T.add,
                                           -1,
                                           (T.erfc, (T.mul, -1, 'x'))),
                                          (T.erf, 'x'),
                                          allow_multiple_clients=True,
                                          name='local_erf_neg_minus_one2',
                                          tracks=[T.erfc],
                                          get_nodes=get_clients)
register_canonicalize(local_erf_neg_minus_one2)
register_stabilize(local_erf_neg_minus_one2)
register_specialize(local_erf_neg_minus_one2)


# Stability optimization
# log(erfc(x)) => when x>threashold,
#              -x**2-log(x)-.5*log(pi)+log(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))
# for float64: threshold=26.641747557 was choosed with:
#  [(i,numpy.log(scipy.special.erfc(numpy.asarray([i],dtype='float64'))))
#   for i in numpy.arange(26.641747557,26.6417475571,.00000000001)]
# for float32: threshold=10.0541949, [(i,numpy.log(scipy.special.erfc(
#        numpy.asarray([i],dtype='float32')))) for i in numpy.arange(
#        10.0541948,10.0541951,.0000001)]
@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log_erfc(node):
    if node.op != T.log:
        return False
    if not node.inputs[0].owner or node.inputs[0].owner.op != T.erfc:
        return False

    if hasattr(node.tag, 'local_log_erfc_applied'):
        # We use that flag to don't apply the optimization recursively
        return False
    node.tag.local_log_erfc_applied = True

    x = node.inputs[0].owner.inputs[0]
    stab_value = (-x ** 2 - T.log(x) - .5 * T.log(np.pi) +
                  T.log(1 - 1 / (2 * x ** 2) + 3 / (4 * x ** 4) -
                  15 / (8 * x ** 6)))

    if (node.outputs[0].dtype == 'float32' or
            node.outputs[0].dtype == 'float16'):
        threshold = 10.0541949
    elif node.outputs[0].dtype == 'float64':
        threshold = 26.641747557

    ret = T.switch(x < threshold, node.outputs[0], stab_value)
    ret.tag.values_eq_approx = values_eq_approx_remove_inf
    return [ret]


# Stability optimization of the grad of log(erfc(x))
# ([y*]exp(-(x**2)))/erfc(x) # The y* is optional
# ([y*]exp(x**2))/erfc(-x) => [y*](when x>threashold,
#                            sqrt(pi)*-x/(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)))
# for float64: threshold=26.63 see at the end of the fct for the explanation
# for float32: threshold=9.3 see at the end of the fct for the explanation
# TODO: remove the contraint that there are only 2 inputs to exp(x**2)
#      is the second.
# TODO: at the test point 10 in float32, there is instability in the original
#      value. The original gives -30.0, the stab -20.1 and in float64 -18.1.
#      Make it so that the test does not generate an error in that case!
@register_stabilize
@register_specialize
@gof.local_optimizer([T.true_div])
def local_grad_log_erfc_neg(node):
    if node.op != T.true_div:
        return False
    if not node.inputs[1].owner or node.inputs[1].owner.op != T.erfc:
        return False
    erfc = node.inputs[1]
    erfc_x = erfc.owner.inputs[0]
    if not node.inputs[0].owner:
        return False

    # The mul is optional.
    if node.inputs[0].owner.op != T.mul:
        mul = None
        y = []
        if not node.inputs[0].owner or node.inputs[0].owner.op != T.exp:
            return False
        exp = node.inputs[0]
    else:
        mul = node.inputs[0]
        exp = None
        for idx, inp in enumerate(mul.owner.inputs):
            if inp.owner and inp.owner.op == T.exp:
                exp = inp
                break
        if len(mul.owner.inputs) == 2:
            y = [mul.owner.inputs[1 - idx]]
        else:
            y = mul.owner.inputs[:]
            del y[idx]
    del mul
    if not exp.owner.inputs[0].owner:
        return False

    if exp.owner.inputs[0].owner.op == T.neg:
        neg = exp.owner.inputs[0]
        if (not neg.owner.inputs[0].owner or
                neg.owner.inputs[0].owner.op != T.sqr):
            return False
        sqr = neg.owner.inputs[0]
        x = sqr.owner.inputs[0]
    elif exp.owner.inputs[0].owner.op == T.mul:
        # We should compare that -(erfc_x**2) is equivalent to mul_neg.
        # There is currently no easy way to do this in the general case,
        # so we implement some common case for now.

        # In many cases the neg are replaced by mul in the graph.
        # This also allows to stabilize log(erfc(cst*x)).
        mul_neg = exp.owner.inputs[0]

        # In case that multiple mul are not fused together, we do it here.
        def check_input(inputs):
            new_inputs = []
            for i in inputs:
                if i.owner and i.owner.op == T.mul:
                    new_inputs.extend(check_input(i.owner.inputs))
                else:
                    new_inputs.append(i)
            return new_inputs
        mul_inputs = check_input(mul_neg.owner.inputs)

        # Put the constant first.
        for i in xrange(len(mul_inputs)):
            if isinstance(i, Constant):
                if i == 0:
                    break
                else:
                    tmp = mul_inputs[0]
                    mul_inputs[0] = mul_inputs[i]
                    mul_inputs[i] = tmp
                    break
        mul_neg = T.mul(*mul_inputs)

        try:
            cst2 = get_scalar_constant_value(mul_neg.owner.inputs[0],
                                             only_process_constants=True)
        except NotScalarConstantError:
            return False

        if len(mul_neg.owner.inputs) == 2:
            if (not mul_neg.owner.inputs[1].owner or
                    mul_neg.owner.inputs[1].owner.op != T.sqr):
                return False
            sqr = mul_neg.owner.inputs[1]
            x = sqr.owner.inputs[0]
        elif len(mul_neg.owner.inputs) == 3:
            if mul_neg.owner.inputs[1] is not mul_neg.owner.inputs[2]:
                return False
            x = mul_neg.owner.inputs[1]
        else:
            return False

        if cst2 != -1:
            if (not erfc_x.owner or erfc_x.owner.op != T.mul or
                    len(erfc_x.owner.inputs) != 2):
                # todo implement that case
                return False
            if erfc_x.owner.inputs[1] is not mul_neg.owner.inputs[1]:
                return False

            x = erfc_x
            try:
                cst = get_scalar_constant_value(erfc_x.owner.inputs[0],
                                                only_process_constants=True)
            except NotScalarConstantError:
                return False
            if cst2 != -cst * 2:
                return False

            # The constant is valid. Must check that the
        elif erfc_x is not x:
            return False

    else:
        return False

    if hasattr(node.tag, 'local_grad_log_erfc_neg'):
        # We use that flag to don't apply the optimization recursively
        return False

    # we move the y outside the div.
    true_div_no_mul = T.true_div(exp, erfc)
    true_div_no_mul.owner.tag.local_grad_log_erfc_neg = True

    # aaron value
    stab_value = (x * T.pow(1 - 1 / (2 * (x ** 2)) +
                  3 / (4 * (x ** 4)) - 15 / (8 * (x ** 6)), -1) *
                  T.cast(T.sqrt(np.pi), dtype=x.dtype))

    if x.dtype == 'float32' or x.dtype == 'float16':
        threshold = 9.3
        # threshold = 10.1
    elif x.dtype == 'float64':
        threshold = 26.641747557
    ret = T.switch(x < threshold, true_div_no_mul, stab_value)
    if y:
        ret = T.mul(ret, *y)
    ret.tag.values_eq_approx = values_eq_approx_remove_inf_nan
    return [ret]
    """
The libm used for the test is amdlibm
    #([y*]exp(-(x**2)))/erfc(x) # The mul is optional
#exp(x**2)/erfc(-x) => when x>threashold,
#-x*(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))*sqrt(pi) for float64:
#threshold=26.63 see below for float32: threshold=9.3 see below TODO
#remove the contraint that there are only 2 inputs to mul TODO: should
#we cast numpy.pi to x.dtype?

#float32 threshold 9.3 as the approximation is more precise at that
#point and more stable.
import numpy, scipy.special
r = numpy.arange(9,10.06,.01)

p64=[(numpy.exp(-(x**2)))/scipy.special.erfc(x) for x in r]
p32=[(numpy.exp(-(x**2)))/scipy.special.erfc(x) for x in
numpy.asarray(r,dtype='float32')]
a64=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))*numpy.sqrt(numpy.pi)
for x in r]
a32=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))
     * numpy.float32(numpy.sqrt(numpy.pi))
for x in numpy.asarray(r,dtype='float32')] for idx,(a,b,c,d,e) in
enumerate(zip(r,p64,p32,a64,a32)):print
a,b,c,d,e,c-b,e-b,numpy.absolute(c-b)<numpy.absolute(e-b)

#, show that the value don't look stable at some point before inf.
for i in xrange(1,len(p32)): print r[i], p32[i]-p32[i-1]

#float64 threshold is 26.63 the approx seam more precise at that
point.  r = numpy.arange(26.2,26.7,.001)
#scipy.special.erfc(numpy.float128(x)) don't work
#p128=[(numpy.exp(-(x**2)))/scipy.special.erfc(x)for x in
numpy.float128(r)] #those value have been computed with g++
theano/misc/erfc_stability_threshold.c && ./a.out
p128=numpy.float128(['46.47206725', '46.47383842', '46.47560959',
'46.47738076', '46.47915193', '46.48092309', '46.48269426',
'46.48446543', '46.48623660', '46.48800777', '46.48977894',
'46.49155011', '46.49332128', '46.49509245', '46.49686362',
'46.49863479', '46.50040596', '46.50217713', '46.50394830',
'46.50571947', '46.50749064', '46.50926181', '46.51103298',
'46.51280415', '46.51457532', '46.51634649', '46.51811766',
'46.51988883', '46.52166000', '46.52343118', '46.52520235',
'46.52697352', '46.52874469', '46.53051586', '46.53228703',
'46.53405820', '46.53582938', '46.53760055', '46.53937172',
'46.54114289', '46.54291407', '46.54468524', '46.54645641',
'46.54822758', '46.54999876', '46.55176993', '46.55354110',
'46.55531227', '46.55708345', '46.55885462', '46.56062579',
'46.56239697', '46.56416814', '46.56593931', '46.56771049',
'46.56948166', '46.57125283', '46.57302401', '46.57479518',
'46.57656636', '46.57833753', '46.58010871', '46.58187988',
'46.58365105', '46.58542223', '46.58719340', '46.58896458',
'46.59073575', '46.59250693', '46.59427810', '46.59604928',
'46.59782045', '46.59959163', '46.60136280', '46.60313398',
'46.60490516', '46.60667633', '46.60844751', '46.61021868',
'46.61198986', '46.61376104', '46.61553221', '46.61730339',
'46.61907456', '46.62084574', '46.62261692', '46.62438809',
'46.62615927', '46.62793045', '46.62970163', '46.63147280',
'46.63324398', '46.63501516', '46.63678633', '46.63855751',
'46.64032869', '46.64209987', '46.64387104', '46.64564222',
'46.64741340', '46.64918458', '46.65095576', '46.65272693',
'46.65449811', '46.65626929', '46.65804047', '46.65981165',
'46.66158283', '46.66335401', '46.66512519', '46.66689636',
'46.66866754', '46.67043872', '46.67220990', '46.67398108',
'46.67575226', '46.67752344', '46.67929462', '46.68106580',
'46.68283698', '46.68460816', '46.68637934', '46.68815052',
'46.68992170', '46.69169288', '46.69346406', '46.69523524',
'46.69700642', '46.69877760', '46.70054878', '46.70231997',
'46.70409115', '46.70586233', '46.70763351', '46.70940469',
'46.71117587', '46.71294705', '46.71471824', '46.71648942',
'46.71826060', '46.72003178', '46.72180296', '46.72357414',
'46.72534533', '46.72711651', '46.72888769', '46.73065887',
'46.73243006', '46.73420124', '46.73597242', '46.73774361',
'46.73951479', '46.74128597', '46.74305715', '46.74482834',
'46.74659952', '46.74837070', '46.75014189', '46.75191307',
'46.75368426', '46.75545544', '46.75722662', '46.75899781',
'46.76076899', '46.76254018', '46.76431136', '46.76608254',
'46.76785373', '46.76962491', '46.77139610', '46.77316728',
'46.77493847', '46.77670965', '46.77848084', '46.78025202',
'46.78202321', '46.78379439', '46.78556558', '46.78733677',
'46.78910795', '46.79087914', '46.79265032', '46.79442151',
'46.79619269', '46.79796388', '46.79973507', '46.80150625',
'46.80327744', '46.80504863', '46.80681981', '46.80859100',
'46.81036219', '46.81213337', '46.81390456', '46.81567575',
'46.81744693', '46.81921812', '46.82098931', '46.82276050',
'46.82453168', '46.82630287', '46.82807406', '46.82984525',
'46.83161644', '46.83338762', '46.83515881', '46.83693000',
'46.83870119', '46.84047238', '46.84224357', '46.84401475',
'46.84578594', '46.84755713', '46.84932832', '46.85109951',
'46.85287070', '46.85464189', '46.85641308', '46.85818427',
'46.85995546', '46.86172665', '46.86349784', '46.86526903',
'46.86704022', '46.86881141', '46.87058260', '46.87235379',
'46.87412498', '46.87589617', '46.87766736', '46.87943855',
'46.88120974', '46.88298093', '46.88475212', '46.88652331',
'46.88829450', '46.89006569', '46.89183688', '46.89360807',
'46.89537927', '46.89715046', '46.89892165', '46.90069284',
'46.90246403', '46.90423522', '46.90600642', '46.90777761',
'46.90954880', '46.91131999', '46.91309119', '46.91486238',
'46.91663357', '46.91840476', '46.92017596', '46.92194715',
'46.92371834', '46.92548953', '46.92726073', '46.92903192',
'46.93080311', '46.93257431', '46.93434550', '46.93611669',
'46.93788789', '46.93965908', '46.94143028', '46.94320147',
'46.94497266', '46.94674386', '46.94851505', '46.95028625',
'46.95205744', '46.95382864', '46.95559983', '46.95737103',
'46.95914222', '46.96091341', '46.96268461', '46.96445581',
'46.96622700', '46.96799820', '46.96976939', '46.97154059',
'46.97331178', '46.97508298', '46.97685417', '46.97862537',
'46.98039657', '46.98216776', '46.98393896', '46.98571015',
'46.98748135', '46.98925255', '46.99102374', '46.99279494',
'46.99456614', '46.99633733', '46.99810853', '46.99987973',
'47.00165092', '47.00342212', '47.00519332', '47.00696452',
'47.00873571', '47.01050691', '47.01227811', '47.01404931',
'47.01582050', '47.01759170', '47.01936290', '47.02113410',
'47.02290530', '47.02467649', '47.02644769', '47.02821889',
'47.02999009', '47.03176129', '47.03353249', '47.03530369',
'47.03707489', '47.03884608', '47.04061728', '47.04238848',
'47.04415968', '47.04593088', '47.04770208', '47.04947328',
'47.05124448', '47.05301568', '47.05478688', '47.05655808',
'47.05832928', '47.06010048', '47.06187168', '47.06364288',
'47.06541408', '47.06718528', '47.06895648', '47.07072768',
'47.07249888', '47.07427009', '47.07604129', '47.', '47.07958369',
'47.08135489', '47.08312609', '47.08489729', '47.08666850',
'47.08843970', '47.09021090', '47.09198210', '47.09375330',
'47.09552450', '47.09729571', '47.09906691', '47.10083811',
'47.10260931', '47.10438052', '47.10615172', '47.10792292',
'47.10969412', '47.11146533', '47.11323653', '47.11500773',
'47.11677894', '47.11855014', '47.12032134', '47.12209255',
'47.12386375', '47.12563495', '47.12740616', '47.12917736',
'47.13094857', '47.13271977', '47.13449097', '47.13626218',
'47.13803338', '47.13980459', '47.14157579', '47.14334700',
'47.14511820', '47.14688941', '47.14866061', '47.15043182',
'47.15220302', '47.15397423', '47.15574543', '47.15751664',
'47.15928784', '47.16105905', '47.16283025', '47.16460146',
'47.16637266', '47.16814387', '47.16991508', '47.17168628',
'47.17345749', '47.17522869', '47.17699990', '47.17877111',
'47.18054231', '47.18231352', '47.18408473', '47.18585593',
'47.18762714', '47.18939835', '47.19116956', '47.19294076',
'47.19471197', '47.19648318', '47.19825439', '47.20002559',
'47.20179680', '47.20356801', '47.20533922', '47.20711042',
'47.20888163', '47.21065284', '47.21242405', '47.21419526',
'47.21596647', '47.21773767', '47.21950888', '47.22128009',
'47.22305130', '47.22482251', '47.22659372', '47.22836493',
'47.23013614', '47.23190735', '47.23367855', '47.23544976',
'47.23722097', '47.23899218', '47.24076339', '47.24253460',
'47.24430581', '47.24607702', '47.24784823', '47.24961944',
'47.25139065', '47.25316186', '47.25493307', '47.25670429',
'47.25847550', '47.26024671', '47.26201792', '47.26378913',
'47.26556034', '47.26733155', '47.26910276', '47.27087397',
'47.27264518', '47.27441640', '47.27618761', '47.27795882',
'47.27973003', '47.28150124', '47.28327246', '47.28504367',
'47.28681488', '47.28858609', '47.29035730', '47.29212852',
'47.29389973', '47.29567094', '47.29744215', '47.29921337',
'47.30098458', '47.30275579', '47.30452701', '47.30629822',
'47.30806943', '47.30984065', '47.31161186', '47.31338307',
'47.31515429', '47.31692550', '47.31869671', '47.32046793',
'47.32223914', '47.32401036', '47.32578157', '47.32755278',
'47.32932400', '47.33109521', '47.33286643', '47.33463764',
'47.33640886', '47.33818007', '47.33995129', '47.34172250',
'47.34349372', '47.34526493', '47.34703615', '47.34880736',
'47.35057858', '47.35234979', '47.35412101', '47.35589223'])
p64=[(numpy.exp(-(x**2)))/scipy.special.erfc(x)for x in r]
a128=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))
      *numpy.float128(numpy.sqrt(numpy.pi))
      for x in numpy.asarray(r,dtype='float128')]
a64=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)+63/(7*x**8))**(-1))
     *numpy.sqrt(numpy.pi)
     for x in r] for a,b,c,d in zip(r,p128,p64,a64):print a,b,c,d,c-b,d-b

for i in xrange(1,len(p64)): print i, 64[i]-p64[i-1]
   """


# ###############
# # Loop fusion #
# ###############
def local_elemwise_fusion_op(OP, max_input_fct=lambda node: 32,
                             maker=None):
    """
    We parametrize it to make it work for Elemwise and GpuElemwise op.

    Parameters
    ----------
    OP
        GpuElemwise or Elemwise class (the one that we want to fuse)
    max_input_fct
        A function that returns the maximum number of inputs
        that this elemwise can take (useful for GpuElemwise).
        GPU kernel currently has a limit of 256 bytes for
        the size of all parameters passed to it. As currently
        we pass many information only by parameter, we must
        limit how many ops we fuse together to avoid busting
        that 256 limit.

        On the CPU we limit to 32 input variables
        since that is the maximum numpy support.

    """
    if maker is None:
        def maker(node, scalar_op):
            return OP(scalar_op)

    def local_fuse(node):
        """
        As part of specialization, we fuse two consecutive elemwise Ops of the
        same shape.

        For mixed dtype, we let the Composite op do the cast. It lets the C
        compiler do the cast.
        The number of dimensions is validated at call time by theano itself.

        """
        # META TODO:  PUT THESE THINGS IN TRAC, NOT TODO NOTES!!
        # TODO: use broadcast flag?

        # TODO: don't do this optimization as a localOptimizer.
        # Analyze the graph in terms of elemwise subgraphs, and then
        # replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't
        # fit within the parameter space of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs
        # have multiple clients. This can't be done with a local
        # optimiser.

        # TODO: Related: Support composites with multiple outputs

        # TODO: Use Composite to combine Elemwise and Reduce
        # operations.  We have to loop over the data anyway... might
        # as well sum it up while we're at it (this can be trickier
        # than i'm making it seound here. The data-traversal should be
        # done contiguously, and the summing-up might not be easy or
        # worthwhile if the summation axis doesn't line up with a
        # contiguous dimension)

        if type(node.op) is not OP:
            return False

        if len(node.outputs) > 1:
            # We don't support the fusion for node with multiple outputs.
            return
        inputs = []  # inputs of the new Elemwise op.
        s_inputs = []  # inputs of the new scalar op used by the Composite.
        # Inputs of the new scalar op that represents the current node.
        s_g = []

        # There is a hard limit of 256 bytes for the formal argument list to a
        # GPU kernel function.
        max_nb_input = max_input_fct(node)
        # The number of inputs to the new fused op if we do not fuse more
        # inputs.
        new_nb_input = len(node.inputs)
        # Did we fuse something?
        # Needed as we can fuse unary op that don't change the number of
        # inputs.
        # And there is a case where the inputs are the same as the current
        # node. That won't change the number of inputs of the new op.
        fused = False

        for i in node.inputs:
            do_fusion = False
            catch = False
            # Will store inputs of the fused node that are not currently inputs
            # of the node we want to create (to avoid duplicating inputs).
            tmp_input = []
            # Same as tmp_input, but for scalars.
            tmp_scalar = []

            # We should not check the number of inputs here
            # As fusing op don't always change the number of input.
            # If a variable is used as multiple into to the same node,
            # we still want to fusion. So we take the set.
            if (i.owner and
                    isinstance(i.owner.op, OP) and
                    len(set([n for n, idx in i.clients])) == 1 and
                    # Do not merge elemwise that don't have the same
                    # broadcastable pattern to don't redo duplicate
                    # computation due to broadcast.
                    i.owner.outputs[0].broadcastable ==
                    node.outputs[0].broadcastable):
                do_fusion = True
                try:
                    tmp_s_input = []
                    # we should not put duplicate input into s_inputs and inputs
                    for ii in i.owner.inputs:
                        if ii in inputs:
                            tmp_s_input.append(s_inputs[inputs.index(ii)])
                        elif ii in tmp_input:
                            tmp_s_input.append(tmp_scalar[tmp_input.index(ii)])
                        else:
                            tmp = scalar.get_scalar_type(ii.dtype).make_variable()
                            try:
                                tv = gof.op.get_test_value(ii)
                                if tv.size > 0:
                                    tmp.tag.test_value = tv.flatten()[0]
                                else:
                                    tmp.tag.test_value = tv
                            except AttributeError:
                                pass
                            tmp_s_input.append(tmp)
                            tmp_input.append(ii)
                            tmp_scalar.append(tmp_s_input[-1])
                    s_op = i.owner.op.scalar_op(*tmp_s_input,
                                                return_list=True)

                    # if the scalar_op don't have a c implementation,
                    # we skip its fusion to allow the fusion of the
                    # other ops.
                    i.owner.op.scalar_op.c_code(s_op[0].owner,
                                                "test_presence_of_c_code",
                                                ["x" for x in i.owner.inputs],
                                                ["z" for z in i.owner.outputs],
                                                {"fail": "%(fail)s"})
                except MethodNotDefined:
                    catch = True
                except NotImplementedError:
                    catch = True
                if catch:
                    _logger.info(("%s does not implement the c_code function."
                                  " As well as being potentially slow, this"
                                  " disables loop fusion of this op.") %
                                 str(i.owner.op.scalar_op))
                    do_fusion = False

            # Compute the number of inputs in case we fuse this input.
            # We subtract 1 because we replace the existing input with the new
            # inputs from `tmp_input`.
            new_nb_input_ = new_nb_input + len(tmp_input) - 1

            # If the new input is already an input of the current node, it was
            # already counted when `new_nb_input` was initialized to
            # len(node.inputs).
            # This can happen when a variable is used both by the Elemwise to
            # fuse and the current node.
            for x in tmp_input:
                if x in node.inputs:
                    new_nb_input_ -= 1

            if do_fusion and (new_nb_input_ <= max_nb_input):
                fused = True
                new_nb_input = new_nb_input_
                inputs.extend(tmp_input)
                s_inputs.extend(tmp_scalar)
                s_g.extend(s_op)
            else:
                # We must support the case where the same variable appear many
                # time in the inputs
                if inputs.count(i) == node.inputs.count(i):
                    s = s_inputs[inputs.index(i)]
                else:
                    s = scalar.get_scalar_type(i.dtype).make_variable()
                    try:
                        if theano.config.compute_test_value != 'off':
                            v = gof.op.get_test_value(i)
                            if v.size > 0:
                                s.tag.test_value = v.flatten()[0]
                    except AttributeError:
                        pass

                    inputs.append(i)
                    s_inputs.append(s)
                s_g.append(s)

        if not fused:
            return False

        if new_nb_input != len(inputs) or len(s_inputs) != len(inputs):
            raise Exception("""Something has gone wrong with the elemwise
fusion optimization. We skip this optimization. You can ignore this message,
your code will run correctly, but may be slower.""")

        s_new_out = node.op.scalar_op(*s_g, return_list=True)
        try:
            s_new_out[0].owner.op.c_code(s_new_out[0].owner,
                                         "test_presence_of_c_code",
                                         ["x" for x in s_g],
                                         ["z" for x in s_new_out],
                                         {"fail": "%(fail)s"})
        except MethodNotDefined:
            _logger.info(("%s does not implement the c_code function."
                          " As well as being potentially slow, this disables "
                          "loop fusion of this op.") % str(
                              s_new_out[0].owner.op))
            return False
        except NotImplementedError:
            _logger.info(("%s does not implement the c_code function. As well"
                          " as being potentially slow, this disables loop"
                          " fusion of this op.") % str(s_new_out[0].owner.op))
            return False

        # create the composite op.
        C = scalar.Composite(s_inputs, s_new_out)

        # create the new node.
        # Do not call make_node to have test_value
        n = maker(node, C)(*inputs).owner
        assert len(n.outputs) == 1
        assert node.outputs[0].dtype == n.outputs[0].dtype

        if len(n.inputs) > max_nb_input:
            _logger.info('loop fusion failed because Op would exceed'
                         ' kernel argument limit.')
            return False

        # we fuse as many that we can at the same time to make debug mode faster
        # debug mode will be faster as it won't test all intermediate step.
        while True:
            ret = local_fuse(n)
            if ret is not False and ret is not None:
                # print n,ret
                assert len(ret) == len(n.outputs)
                assert len(ret) == 1
                n = ret[0].owner
            else:
                break

        return n.outputs
    return local_fuse


def elemwise_max_input_fct(node):
    # The Elemwise.perform use numpy ufunc and they are limited to 31
    # inputs.
    if not theano.config.cxx:
        return 31
    return 1024


local_elemwise_fusion = local_elemwise_fusion_op(T.Elemwise,
                                                 elemwise_max_input_fct)


class FusionOptimizer(Optimizer):
    """Graph optimizer for Fusion of elemwise operations."""
    def __init__(self, local_optimizer):
        Optimizer.__init__(self)
        self.optimizer = local_optimizer

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        nb_iter = 0
        nb_replacement = 0
        nb_inconsistency_replace = 0
        time_toposort = 0
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time
        while did_something:
            t0 = time.time()
            nodelist = list(fgraph.toposort())
            time_toposort += time.time() - t0
            nodelist.reverse()
            did_something = False
            for node in nodelist:
                # Don't try to fuse node that have already been fused.
                if node in fgraph.apply_nodes:
                    new_outputs = self.optimizer(node)
                    if new_outputs:
                        assert len(new_outputs) == len(node.outputs)
                        try:
                            fgraph.replace_all_validate(
                                list(zip(node.outputs, new_outputs)),
                                reason=self.__class__.__name__)
                            did_something = True
                            nb_replacement += 1
                        except InconsistencyError:
                            nb_inconsistency_replace += 1
                            pass
            nb_iter += 1

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in iteritems(fgraph.execute_callbacks_times):
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}
        return (self, nb_iter, nb_replacement,
                nb_inconsistency_replace,
                validate_time, callback_time, callbacks_time,
                time_toposort)

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = ('    ' * level)
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(iteritems(prof[6]), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)
        print(blanc, " time_toposort", prof[7], file=stream)


def local_add_mul_fusion(node):
    """Fuse consecutive add or mul in one such node with more inputs.

    It is better to fuse add/mul that way then in a Composite node as
    this make the inner graph of the Composite smaller. This allow to
    put more computation in a Composite before hitting the max
    recusion limit when pickling Composite.

    """
    if (not isinstance(node.op, Elemwise) or
            not isinstance(node.op.scalar_op, (scalar.Add, scalar.Mul))):
        return False

    s_op = node.op.scalar_op.__class__
    new_inp = []
    fused = False
    nb_inputs = len(node.inputs)
    max_inputs = float('inf')
    if hasattr(node.op, 'max_inputs'):
        max_inputs = node.op.max_inputs(node)
    for inp in node.inputs:
        if (inp.owner and
                isinstance(inp.owner.op, Elemwise) and
                isinstance(inp.owner.op.scalar_op, s_op) and
                # Do not duplicate the operation.
                len(inp.clients) == 1 and
                (nb_inputs + len(inp.owner.inputs) - 1) <= max_inputs):
            new_inp.extend(inp.owner.inputs)
            fused = True
        else:
            new_inp.append(inp)

    # We can not compare the number of inputs as Mul and Add could have
    # 0 or 1 inputs in some corner cases.
    if fused:
        output = node.op(*new_inp)
        copy_stack_trace(node.outputs[0], output)

        # Do the recursion here to help lower the number of
        # FusionOptimizer iteration.
        if output.owner:
            output2 = local_add_mul_fusion(output.owner)
            if output2:
                return output2
        return [output]

if config.tensor.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion elemwise in fast_run")
    # Must be after gpu(48.5) and before AddDestroyHandler(49.5)
    fuse_seqopt = gof.SequenceDB()
    fuse_seqopt.register('local_add_mul_fusion',
                         FusionOptimizer(local_add_mul_fusion),
                         0, 'fast_run', 'fusion')
    fuse_seqopt.register('composite_elemwise_fusion',
                         FusionOptimizer(local_elemwise_fusion),
                         1, 'fast_run', 'fusion')
    compile.optdb.register('elemwise_fusion',
                           fuse_seqopt, 49,
                           'fast_run', 'fusion', 'local_elemwise_fusion',
                           'FusionOptimizer')
else:
    _logger.debug("not enabling optimization fusion elemwise in fast_run")
    compile.optdb.register('elemwise_fusion',
                           FusionOptimizer(local_elemwise_fusion), 49,
                           'fusion', 'local_elemwise_fusion',
                           'FusionOptimizer')


@register_canonicalize
@gof.local_optimizer([Elemwise])
def local_useless_composite(node):
    """For elemwise Composite that have multiple outputs, remove the
    outputs that are not used.

    """
    if (not isinstance(node.op, Elemwise) or
            not isinstance(node.op.scalar_op, scalar.Composite)):
        return
    comp = node.op.scalar_op
    idx = [i for i, o_extern in enumerate(node.outputs)
           if o_extern.clients]
    if len(idx) < len(node.outputs):
        new_outputs = [comp.outputs[i] for i in idx]
        c = scalar.Composite(inputs=comp.inputs,
                             outputs=new_outputs)
        e = Elemwise(scalar_op=c)(*node.inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in idx], e))

# ############################
# # Remove consider_constant #
# ############################


# Although the ops ConsiderConstant, ZeroGrad and DisconnectedGrad
# just returns the input, it should be removed from the graph to
@register_canonicalize('fast_compile')
@register_useless('fast_compile')
@gof.local_optimizer(None)
def local_view_op(node):
    if isinstance(node.op, theano.compile.ops.ViewOp):
        return node.inputs


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@gof.local_optimizer([T.Alloc])
def local_merge_alloc(node):
    # This opt takes care of several cases:
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) -> Alloc(m, x, assert(y1, y1==y2), z, w)
    if not isinstance(node.op, T.Alloc):
        return False
    if not node.inputs[0].owner or not isinstance(
            node.inputs[0].owner.op, T.Alloc):
        return False
    inputs_outer = node.inputs
    inputs_inner = node.inputs[0].owner.inputs
    dims_outer = inputs_outer[1:]
    dims_inner = inputs_inner[1:]
    dims_outer_rev = dims_outer[::-1]
    dims_inner_rev = dims_inner[::-1]
    # check if the pattern of broadcasting is matched, in the reversed ordering.
    # The reverse ordering is needed when an Alloc add an implicit new
    # broadcasted dimensions to its inputs[0]. Eg:
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    i = 0
    for dim_inner, dim_outer in zip(dims_inner_rev, dims_outer_rev):
        if dim_inner != dim_outer:
            if isinstance(dim_inner, Constant) and dim_inner.data == 1:
                pass
            else:
                dims_outer[-1 - i] = Assert(
                    "You have a shape error in your graph. To see a better"
                    " error message and a stack trace of where in your code"
                    " the error is created, use the Theano flags"
                    " optimizer=None or optimizer=fast_compile.")(
                    dim_outer, T.eq(dim_outer, dim_inner))
        i += 1
    return [T.alloc(inputs_inner[0], *dims_outer)]


@register_useless('fast_compile')
@gof.local_optimizer([TopKOp])
def local_useless_topk(node):
    """
    TopKOp generates two outputs by default
    This opt removes the useless ones

    """
    op = node.op
    if not isinstance(op, TopKOp):
        return
    if not (op.return_values and op.return_indices):
        return False

    x, k = node.inputs
    ret_val = bool(node.outputs[0].clients)
    ret_idx = bool(node.outputs[1].clients)

    if not (ret_val ^ ret_idx):
        # both true -> nothing to remove
        # both false -> let pruner handle
        return False

    old_output = node.outputs[ret_idx]
    new_output = TopKOp(
        axis=op.axis,
        sorted=op.sorted,
        idx_dtype=op.idx_dtype,
        return_values=ret_val,
        return_indices=ret_idx)(x, k)
    copy_stack_trace(node.outputs[0], new_output)
    return {old_output: new_output}
