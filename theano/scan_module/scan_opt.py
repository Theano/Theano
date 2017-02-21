"""
This module provides optimizations for scan.
The Optimization provided in this file:

local opt: remove_constants_and_unused_inputs_scan,
           constant_folding_for_scan2,
           scan_merge_inouts
           They are wrapped in in2out to create global opt.
global opt: ScanInplaceOptimizer,
            PushOutNonSeqScan,
            PushOutSeqScan,
            PushOutDot1,
            ScanMerge,
            ScanSaveMem

How the are registered:

optdb: scan_eqopt1 (.1), scan_eqopt2(1.6), scan_inplace(75)
scan_eqopt1 -> scan_seqopt1
scan_seqopt1 -> in2out(remove_constants_and_unused_inputs_scan)(1),
                PushOutNonSeqScan(2),
                PushOutSeqScan(3), PushOutDot1(4)
scan_eqopt2 -> They are all global optimizer. (in2out convert local to global).
               This is important, as the order is important and all global
               optimizer run before local optimizer in the order they where
               registered. (So don't change the order we register them!)
               If we convert to local optimizer, we must convert all of them
               to local optimizer. But:
               1) can ScanMerge be made local? Can we keep only this one
               global?
               2) ScanSaveMem assert that we remove all nodes outputs,
                  we need to keep this.
               3) It is ScanSaveMem suppose the the others ran before.
                  I added an assert at one place, but didn't looked for
                  other place.
               4) Moving this to local opt could speed up significant this opt,
                  as we pass frequently on all nodes in the graph for no
                  good reason.
               5) We register remove_constant_*  many places, as some
                  opt create them and let this one clean up the mess.
                  Doing it that way, make things simpler for those already
                  complex opt.

               in2out(constant_folding),
               in2out(remove_constants_and_unused_inputs_scan1),
               ScanMerge,
               in2out(remove_constants_and_unused_inputs_scan2),
               in2out(scan_merge_inouts),
               ScanSaveMem,
               in2out(remove_constants_and_unused_inputs_scan3)
"""
from __future__ import absolute_import, print_function, division
import logging
import copy
from sys import maxsize
from collections import OrderedDict
import numpy

import theano
from theano import tensor, scalar
from theano.tensor import opt, get_scalar_constant_value, Alloc, AllocEmpty
from theano import gof
from six import integer_types, iteritems
from six.moves import xrange
from theano.compile import optdb
from theano.compile.function_module import deep_copy_op
from theano.gof import toolbox, DestroyHandler, InconsistencyError
from theano.gof.opt import Optimizer
from theano.gof.opt import pre_constant_merge, pre_greedy_local_optimizer

from theano.scan_module import scan_op
from theano.scan_module import scan_utils
from theano.scan_module.scan_utils import equal_computations, find_up, scan_args

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin "
               "Arnaud Bergeron ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_opt')

list_opt_slice = [tensor.opt.local_abs_merge,
                  tensor.opt.local_mul_switch_sink,
                  tensor.opt.local_upcast_elemwise_constant_inputs,
                  tensor.opt.local_useless_switch,
                  tensor.opt.constant_folding]


def warning(*msg):
    _logger.warning('WARNING theano.scan: ' + ' '.join(msg))


def info(*msg):
    _logger.info('INFO theano.scan: ' + ' '.join(msg))


@gof.local_optimizer([scan_op.Scan])
def remove_constants_and_unused_inputs_scan(node):
    """
    Move constants into the inner graph, and remove unused inputs.

    Constants that are in the outer graph are represented by a free symbolic
    variable in the inner graph. If we move them into the inner graph,
    constant-folding can happen in the inner graph.
    This is applied only on sequences and non-sequences,
    not on initial states.

    """
    if not isinstance(node.op, scan_op.Scan):
        return False
    op = node.op
    # We only need to take care of sequences and other arguments
    st = op.n_seqs
    st += int(sum([len(x) for x in
                   op.tap_array[:(op.n_mit_mot + op.n_mit_sot)]]))
    st += op.n_sit_sot
    st += op.n_shared_outs

    op_ins = op.inputs
    op_outs = op.outputs

    # Corresponds to the initial states, which should stay untouched.
    # We put those variables aside, and put them back at the end.
    out_stuff_inner = op_ins[op.n_seqs:st]

    non_seqs = op_ins[st:]
    st = (op.n_seqs +
          op.n_mit_mot +
          op.n_mit_sot +
          op.n_sit_sot +
          op.n_nit_sot +
          op.n_shared_outs + 1)
    outer_non_seqs = node.inputs[st:]
    out_stuff_outer = node.inputs[1 + op.n_seqs:st]

    # To replace constants in the outer graph by clones in the inner graph
    givens = OrderedDict()
    # All the inputs of the inner graph of the new scan
    nw_inner = []
    # Same for the outer graph, initialized w/ number of steps
    nw_outer = [node.inputs[0]]

    all_ins = gof.graph.inputs(op_outs)
    for idx in xrange(op.n_seqs):
        node_inp = node.inputs[idx + 1]
        if (isinstance(node_inp, tensor.TensorConstant) and
                node_inp.tag.unique_value is not None):
            try:
                # This works if input is a constant that has all entries
                # equal
                givens[op_ins[idx]] = node_inp.clone()[0]
            except TypeError:
                pass
        elif op_ins[idx] in all_ins:
            # Check for identical other sequence
            identical_seqs = [x for x in nw_outer
                              if scan_utils.equal_computations(
                                  [x], [node_inp])]
            if identical_seqs:
                index = node.inputs.index(identical_seqs[0]) - 1
                givens[op_ins[idx]] = op_ins[index]
            else:
                nw_inner.append(op_ins[idx])
                nw_outer.append(node_inp)

    nw_n_seqs = len(nw_inner)
    # Add outputs stuff
    nw_inner += out_stuff_inner
    nw_outer += out_stuff_outer

    # Look through non sequences
    nw_inner_nonseq = []
    nw_outer_nonseq = []
    for idx, (nw_in, nw_out) in enumerate(zip(non_seqs, outer_non_seqs)):
        if isinstance(nw_out, tensor.Constant):
            givens[nw_in] = nw_out.clone()
        elif nw_in in all_ins:
            # Indices of elements of nw_outer_nonseq that are equivalent
            # to nw_out.
            identical_nonseq_idx = [
                i for (i, x) in enumerate(nw_outer_nonseq)
                if scan_utils.equal_computations([x], [nw_out])]
            if identical_nonseq_idx:
                givens[nw_in] = nw_inner_nonseq[identical_nonseq_idx[0]]
            else:
                nw_inner_nonseq.append(nw_in)
                nw_outer_nonseq.append(nw_out)

    nw_inner.extend(nw_inner_nonseq)
    nw_outer.extend(nw_outer_nonseq)

    if len(nw_inner) != len(op_ins):
        op_outs = scan_utils.clone(op_outs, replace=givens)
        nw_info = copy.deepcopy(op.info)
        nw_info['n_seqs'] = nw_n_seqs
        # DEBUG CHECK
        nwScan = scan_op.Scan(nw_inner, op_outs, nw_info)
        nw_outs = nwScan(*nw_outer, **dict(return_list=True))
        return OrderedDict([("remove", [node])] + list(zip(node.outputs, nw_outs)))
    else:
        return False


# This is a global opt for historical reason
# It should be possible to change it to a local opt.
class PushOutNonSeqScan(gof.Optimizer):
    """
    A global optimizer for pushing out the variables inside the scan that depend
    only on non-sequences.
    """

    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(gof.toolbox.ReplaceValidate())

    def apply(self, fgraph):
        nodelist = [x for x in fgraph.toposort() if isinstance(x.op,
                                                               scan_op.Scan)]
        for node in nodelist:
            self.process_node(fgraph, node)

    def process_node(self, fgraph, node):
        """
        IMPORTANT NOTE: This function uses set and dictionary data structures.
        By default they are not ordered for efficiency reasons. Take care
        and make sure of changing them with their Ordered counterparts if you
        need to iterate over these variables.

        """
        # this flag tells if there was any change during the last iterations
        clean_inputs, clean_outputs = scan_utils.reconstruct_graph(
            node.op.inputs, node.op.outputs)

        local_fgraph_topo = theano.gof.graph.io_toposort(clean_inputs,
                                                         clean_outputs)
        local_fgraph_outs_set = set(clean_outputs)
        local_fgraph_outs_map = dict([(v, k) for k, v in
                                      enumerate(clean_outputs)])

        to_remove_set = set()
        to_replace_set = set()
        to_replace_map = OrderedDict()

        def add_to_replace(y):
            to_replace_set.add(y)
            to_replace_map[y] = add_to_replace.n
            add_to_replace.n += 1
        add_to_replace.n = 0

        replace_with_in = []
        replace_with_out = []

        op = node.op
        # Construct the list of non_sequences to simplify a few things
        inner_non_seqs = op.inner_non_seqs(clean_inputs)
        inner_non_seqs_set = set(inner_non_seqs)
        inner_non_seqs_map = dict([(v, k) for k, v in
                                   enumerate(inner_non_seqs)])

        outer_non_seqs = op.outer_non_seqs(node.inputs)

        inner_seqs = op.inner_seqs(clean_inputs)
        outer_seqs = op.outer_seqs(node.inputs)

        assert len(inner_non_seqs) == len(outer_non_seqs)
        assert len(inner_seqs) == len(outer_seqs)

        for nd in local_fgraph_topo:
            if (  # we haven't already looked at this node
                    nd not in to_remove_set and
                    all([((x in inner_non_seqs_set) or
                        (x.owner in to_remove_set) or
                        isinstance(x, tensor.Constant))
                        for x in nd.inputs]) and
                    # we can do this because the assumption is that a
                    # viewOp or deepCopyOp will be just at the end of the
                    # function and not somewhere in the middle ..
                    not isinstance(nd.op, theano.compile.ViewOp) and
                    not isinstance(nd.op, theano.compile.DeepCopyOp)):

                # We have a candidate node to removable
                # Step 1. Reconstruct it on outside
                to_remove_set.add(nd)
                outside_ins = []
                for x in nd.inputs:
                    if x in inner_non_seqs_set:
                        _idx = inner_non_seqs_map[x]
                        outside_ins.append(outer_non_seqs[_idx])
                    elif x in to_replace_set:
                        outside_ins.append(replace_with_out[to_replace_map[x]])
                    elif isinstance(x, theano.Constant):
                        outside_ins.append(x.clone())
                    else:
                        raise Exception(
                            ('Error in the `scan_pushout_non_seq_'
                             'operations`. The optimization tries '
                             'to move some computation fron scan '
                             'which is not allowed to move. Report '
                             'this on theano-users list'), x)
                outside_ins = [x.type.filter_variable(y) for x, y in
                               zip(nd.inputs, outside_ins)]

                # Do not call make_node for test_value
                nw_outer_node = nd.op(*outside_ins,
                                      **dict(return_list=True))[0].owner

                # Step 2. Create variables for replacements
                for idx, y in enumerate(nd.outputs):
                    y_place_holder = scan_utils.safe_new(y, '_replace')
                    add_to_replace(y)
                    replace_with_in.append(y_place_holder)
                    assert isinstance(y, type(nw_outer_node.outputs[idx]))
                    replace_with_out.append(nw_outer_node.outputs[idx])

        # We need to check all candidate replacements and choose those that
        # make sense for us
        # Step 1. which elements of `to_replace` are used by remaining
        # components of the inner function
        clean_to_replace = []
        clean_replace_with_in = []
        clean_replace_with_out = []
        existent_nodes = [nd for nd in local_fgraph_topo
                          if nd not in to_remove_set]
        existent_nodes_set = set(existent_nodes)

        to_keep_set = set([])
        for nd in existent_nodes:
            to_keep_set.update(nd.inputs)

        for out, idx in to_replace_map.items():
            if (  # If types are different, conversion Op will be inserted,
                    # and it may trigger an infinite loop.
                    replace_with_in[idx].type == out.type and
                    out in to_keep_set and
                    out.owner not in existent_nodes_set):
                clean_to_replace.append(out)
                clean_replace_with_in.append(replace_with_in[idx])
                clean_replace_with_out.append(replace_with_out[idx])

        if len(clean_to_replace) > 0:
            # We can finally put an end to all this madness
            givens = OrderedDict()
            nw_outer = []
            nw_inner = []
            for to_repl, repl_in, repl_out in zip(clean_to_replace,
                                                  clean_replace_with_in,
                                                  clean_replace_with_out):
                if isinstance(repl_out, theano.Constant):
                    repl_in = repl_out.clone()
                else:
                    nw_inner.append(repl_in)
                    nw_outer.append(repl_out)
                givens[to_repl] = repl_in

            op_outs = scan_utils.clone(clean_outputs, replace=givens)
            op_ins = clean_inputs + nw_inner

            # Reconstruct node
            nwScan = scan_op.Scan(op_ins, op_outs, op.info)

            # Do not call make_node for test_value
            nw_node = nwScan(*(node.inputs + nw_outer),
                             **dict(return_list=True))[0].owner

            fgraph.replace_all_validate_remove(
                list(zip(node.outputs, nw_node.outputs)),
                remove=[node],
                reason='scanOp_pushout_nonseqs_ops')
            return True
        elif not to_keep_set:
            # Nothing in the inner graph should be kept
            replace_with = OrderedDict()
            for out, idx in to_replace_map.items():
                if out in local_fgraph_outs_set:
                    x = node.outputs[local_fgraph_outs_map[out]]
                    y = replace_with_out[idx]
                    shape = [shp for shp in y.shape]
                    replace_with[x] = tensor.alloc(y,
                                                   node.inputs[0],
                                                   *shape)

            # We need to add one extra dimension to the outputs
            # because the scan op expects for a tensor3, to which an
            # subtensor is applied that takes only the last element
            if replace_with:
                if len(node.outputs) == len(replace_with):
                    # Every output of the node has a replacement, the Scan
                    # node can be removed from the graph
                    fgraph.replace_all_validate_remove(
                        replace_with.items(),
                        remove=[node],
                        reason='scanOp_pushout_nonseqs_ops')
                else:
                    # The node has some outputs for which no replacement has
                    # been established. This can occur for outputs that are
                    # not produced by apply nodes (since the optimizations
                    # only visits apply nodes) such as constants or inputs
                    # passed directly as outputs. The replacements can be
                    # performed but the Scan node can't be removed at this
                    # point.
                    fgraph.replace_all_validate(
                        replace_with.items(),
                        reason='scanOp_pushout_nonseqs_ops')

        else:
            return False


# This is a global opt for historical reason
# It should be possible to change it to a local opt.
class PushOutSeqScan(gof.Optimizer):
    """
    A global optimizer for pushing out the variables inside the
    scan that depend only on constants and sequences.
    """

    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(gof.toolbox.ReplaceValidate())

    def apply(self, fgraph):
        nodelist = [x for x in fgraph.toposort()
                    if isinstance(x.op, scan_op.Scan)]
        for node in nodelist:
            self.process_node(fgraph, node)

    def process_node(self, fgraph, node):
        """
        IMPORTANT NOTE: This function uses set and dictionary data structure.
        By default they are not ordered for efficiency reasons. Take care
        and make sure of changing them to Ordered versions if you need to
        iterate over those variables.

        """
        # this flag tells if there was any change during the last iterations
        clean_inputs, clean_outputs = scan_utils.reconstruct_graph(
            node.op.inputs, node.op.outputs)

        local_fgraph_topo = theano.gof.graph.io_toposort(clean_inputs,
                                                         clean_outputs)
        local_fgraph_outs_set = set(clean_outputs)
        local_fgraph_outs_map = dict([(v, k) for k, v in
                                      enumerate(clean_outputs)])

        to_remove_set = set()
        to_replace_set = set()
        to_replace_map = OrderedDict()

        def add_to_replace(y):
            to_replace_set.add(y)
            to_replace_map[y] = add_to_replace.n
            add_to_replace.n += 1
        add_to_replace.n = 0

        replace_with_in = []
        replace_with_out = []

        op = node.op
        # Construct the list of non_sequences to simplify a few things
        inner_non_seqs = op.inner_non_seqs(clean_inputs)
        inner_non_seqs_set = set(inner_non_seqs)
        inner_non_seqs_map = dict([(v, k) for k, v in
                                   enumerate(inner_non_seqs)])

        outer_non_seqs = op.outer_non_seqs(node.inputs)
        inner_seqs = op.inner_seqs(clean_inputs)
        inner_seqs_set = set(inner_seqs)
        inner_seqs_map = dict([(v, k) for k, v in
                               enumerate(inner_seqs)])

        outer_seqs = op.outer_seqs(node.inputs)
        assert len(inner_non_seqs) == len(outer_non_seqs)
        assert len(inner_seqs) == len(outer_seqs)

        for nd in local_fgraph_topo:
            if (nd not in to_remove_set and
                all([(x in inner_non_seqs_set) or
                     (x.owner in to_remove_set) or
                     isinstance(x, tensor.Constant) or
                     (x in inner_seqs_set) for x in nd.inputs]) and
                    isinstance(nd.op, theano.tensor.Elemwise)):

                outside_ins = []
                depends_on_seqs = False

                for x in nd.inputs:
                    if x in inner_non_seqs_set:
                        _idx = inner_non_seqs_map[x]
                        outside_ins.append(outer_non_seqs[_idx])
                    elif x in inner_seqs_set:
                        outside_ins.append(outer_seqs[inner_seqs_map[x]])
                        depends_on_seqs = True
                    elif x in to_replace_set:
                        outside_ins.append(replace_with_out[
                            to_replace_map[x]])
                        depends_on_seqs = True
                    elif isinstance(x, theano.Constant):
                        outside_ins.append(x.clone())
                    else:
                        raise Exception(
                            ('Error in the `scan_pushout_seq_'
                             'operations`. The optimization tries '
                             'to move some computation fron scan '
                             'which is not allowed to move. Report '
                             'this on theano-users list'), x)

                if not depends_on_seqs:
                    # Removing this node from the inner graph of scan
                    # should be handled by the PushOutNonSeqScan
                    # optimization. The current optimization only tries
                    # to pull sequence-dependant computation out of
                    # scan.
                    continue

                to_remove_set.add(nd)

                # Do not call make_node for test_value
                nw_outer_node = nd.op(*outside_ins,
                                      **dict(return_list=True))[0].owner

                # Step 2. Create variables for replacements
                for idx, y in enumerate(nd.outputs):
                    y_place_holder = scan_utils.safe_new(y, '_replace')
                    add_to_replace(y)
                    replace_with_in.append(y_place_holder)
                    replace_with_out.append(nw_outer_node.outputs[idx])

            elif (nd not in to_remove_set and
                  isinstance(nd.op, theano.tensor.DimShuffle) and
                  (nd.inputs[0] in inner_seqs_set or
                   nd.inputs[0].owner in to_remove_set)):

                to_remove_set.add(nd)
                x = nd.inputs[0]
                if x in inner_seqs_set:
                    outside_ins = outer_seqs[inner_seqs_map[x]]
                elif x in to_replace_set:
                    outside_ins = replace_with_out[to_replace_map[x]]
                new_ord = (0,)
                for old_ord in nd.op.new_order:
                    if (old_ord == 'x'):
                        new_ord += (old_ord,)
                    else:
                        new_ord += (old_ord + 1,)
                new_outer = outside_ins.dimshuffle(new_ord)
                y = nd.outputs[0]
                y_place_holder = scan_utils.safe_new(y, '_replace')
                add_to_replace(y)
                replace_with_in.append(y_place_holder)
                replace_with_out.append(new_outer)

                if hasattr(new_outer.tag, "test_value"):
                    new_sh = new_outer.tag.test_value.shape
                    ref_sh = (outside_ins.tag.test_value.shape[0],)
                    ref_sh += nd.outputs[0].tag.test_value.shape
                    assert new_sh == ref_sh

        # We need to check all candidate replacements and choose those that
        # make sense for us
        # Step 1. which elements of `to_replace` are used by remaining
        # components of the inner function
        clean_to_replace = []
        clean_replace_with_in = []
        clean_replace_with_out = []

        existent_nodes = [nd for nd in local_fgraph_topo
                          if nd not in to_remove_set]
        existent_nodes_set = set(existent_nodes)

        to_keep_set = set([])
        for nd in existent_nodes:
            to_keep_set.update(nd.inputs)

        for out, idx in to_replace_map.items():
            if (out in to_keep_set and out.owner not in existent_nodes_set and
                # If types are different, conversion Op will be inserted,
                # and it may trigger an infinite loop.
                    replace_with_in[idx].type == out.type):

                clean_to_replace.append(out)
                clean_replace_with_in.append(replace_with_in[idx])
                clean_replace_with_out.append(replace_with_out[idx])

        if len(clean_to_replace) > 0:
            # We can finally put an end to all this madness
            givens = OrderedDict()
            nw_outer = []
            nw_inner = []
            for to_repl, repl_in, repl_out in zip(clean_to_replace,
                                                  clean_replace_with_in,
                                                  clean_replace_with_out):
                if isinstance(repl_out, theano.Constant):
                    repl_in = repl_out.clone()
                else:
                    nw_inner.append(repl_in)
                    nw_outer.append(repl_out)

                givens[to_repl] = repl_in

            op_outs = scan_utils.clone(clean_outputs, replace=givens)
            op_ins = nw_inner + clean_inputs

            # Reconstruct node
            nw_info = op.info.copy()
            nw_info['n_seqs'] += len(nw_inner)
            nwScan = scan_op.Scan(op_ins, op_outs, nw_info)
            # Do not call make_node for test_value
            nw_node = nwScan(*(node.inputs[:1] + nw_outer + node.inputs[1:]),
                             **dict(return_list=True))[0].owner

            fgraph.replace_all_validate_remove(
                list(zip(node.outputs, nw_node.outputs)),
                remove=[node],
                reason='scanOp_pushout_seqs_ops')
            return True
        elif (not to_keep_set and
              not op.as_while and
              not op.outer_mitmot(node)):
            # Nothing in the inner graph should be kept
            replace_with = OrderedDict()
            for out, idx in to_replace_map.items():
                if out in local_fgraph_outs_set:
                    x = node.outputs[local_fgraph_outs_map[out]]
                    _y = replace_with_out[idx]
                    ls = clean_outputs
                    if out in op.inner_mitsot_outs(ls):
                        odx = op.inner_mitsot_outs(ls).index(out)
                        inp = op.outer_mitsot(node)[odx]
                        st = abs(numpy.min(op.mitsot_taps()))
                        y = tensor.set_subtensor(inp[st:], _y)
                    elif out in op.inner_sitsot_outs(ls):
                        odx = op.inner_sitsot_outs(ls).index(out)
                        inp = op.outer_sitsot(node)[odx]
                        y = tensor.set_subtensor(inp[1:], _y)
                    elif out in op.inner_nitsot_outs(ls):
                        y = _y
                    else:
                        y = _y[-1]
                    replace_with[x] = y

            # We need to add one extra dimension to the outputs
            if replace_with and len(replace_with) == len(node.outputs):
                fgraph.replace_all_validate_remove(
                    list(replace_with.items()),
                    remove=[node],
                    reason='scanOp_pushout_seqs_ops')
                return True
        else:
            return False


class PushOutScanOutput(gof.Optimizer):
    """
    This is an optimization that can push operations performed
    at the end of the inner graph of scan to outside of scan.
    """

    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(gof.toolbox.ReplaceValidate())

    def apply(self, fgraph):
        # Don't perform the optimization on as_while scans. Because these scans
        # don't run for a predetermined number of steps, handling them is
        # more complicated and this optimization doesn't support it at the
        # moment.
        nodelist = [x for x in fgraph.toposort()
                    if (isinstance(x.op, scan_op.Scan) and
                        not x.op.as_while)]
        for node in nodelist:
            # Process the node as long as something gets optimized
            while node is not None:
                node = self.process_node(fgraph, node)

    def process_node(self, fgraph, node):

        op = node.op

        # Use scan_args to parse the inputs and outputs of scan for ease of
        # use
        args = scan_args(node.inputs, node.outputs,
                         op.inputs, op.outputs, op.info)

        new_scan_node = None
        clients = {}
        local_fgraph_topo = theano.gof.graph.io_toposort(args.inner_inputs,
                                                         args.inner_outputs,
                                                         clients=clients)

        for nd in local_fgraph_topo:
            if (isinstance(nd.op, theano.tensor.elemwise.Elemwise) and
                    isinstance(nd.op.scalar_op, scalar.Add) and
                    nd.out in args.inner_out_sit_sot and
                    self.inner_sitsot_only_last_step_used(nd.out, args)):

                # Ensure that one of the input to the add is the output of
                # the add from a previous iteration of the inner function
                sitsot_idx = args.inner_out_sit_sot.index(nd.out)
                if args.inner_in_sit_sot[sitsot_idx] in nd.inputs:

                    # Ensure that the other input to the add is a dot product
                    # between 2 matrices which will become a tensor3 and a
                    # matrix if pushed outside of the scan. Also make sure
                    # that the output of the Dot is ONLY used by the 'add'
                    # otherwise doing a Dot in the outer graph will only
                    # duplicate computation.

                    sitsot_in_idx = nd.inputs.index(args.inner_in_sit_sot[
                                                    sitsot_idx])

                    # 0 if sitsot_in_idx==1, 1 if sitsot_in_idx==0
                    dot_in_idx = 1 - sitsot_in_idx

                    dot_input = nd.inputs[dot_in_idx]

                    if (dot_input.owner is not None and
                        isinstance(dot_input.owner.op, theano.tensor.Dot) and
                        len(clients[dot_input]) == 1 and
                        dot_input.owner.inputs[0].ndim == 2 and
                        dot_input.owner.inputs[1].ndim == 2 and
                        self.get_outer_ndim(dot_input.owner.inputs[0], args) == 3 and
                            self.get_outer_ndim(dot_input.owner.inputs[1], args) == 3):

                        # The optimization can be be applied in this case.

                        # Move out of scan the two inputs to the Dot and
                        # perform a dot outside of scan on these two inputs
                        inner_dot_inputs = nd.inputs[dot_in_idx].owner.inputs
                        (outer_dot_inputs,
                         new_scan_node,
                         new_scan_args) = \
                            self.push_out_inner_vars(fgraph, inner_dot_inputs,
                                                     node, args)

                        # Collapse some of the dimensions of the tensors
                        # so that they become matrices. This is because a
                        # dot is usually faster on two large matrices than
                        # a bunch of small ones
                        outer_dot_inputs[0] = theano.tensor.flatten(
                            outer_dot_inputs[0].dimshuffle(1, 0, 2), outdim=2)

                        shape_input1 = theano.tensor.shape(outer_dot_inputs[1])
                        outer_dot_inputs[1] =\
                            outer_dot_inputs[1].reshape((shape_input1[0] *
                                                         shape_input1[1],
                                                         shape_input1[2]))

                        # Perform the dot on the newly obtained matrices and
                        # add the initial value
                        outer_dot_output = theano.tensor.dot(*outer_dot_inputs)
                        init_value = new_scan_args.outer_in_sit_sot[sitsot_idx][0]
                        replacement = outer_dot_output + init_value

                        # Alter the outer graph to use the output of the
                        # external Dot instead of the output of scan
                        # Modify the outer graph to add the outer Dot
                        outer_sitsot = new_scan_args.outer_out_sit_sot[sitsot_idx]
                        subtensor_node = outer_sitsot.clients[0][0]
                        outer_sitsot_last_step = subtensor_node.outputs[0]

                        fgraph.replace_all([
                            (outer_sitsot_last_step, replacement)],
                            reason="scanOp_pushout_output")

                        break
        return new_scan_node

    def inner_sitsot_only_last_step_used(self, var, scan_args):
        """
        Given a inner nit_sot output of scan, return True iff the outer
        nit_sot output has only one client and that client is a Subtensor
        instance that takes only the last step (last element along the first
        axis).

        """
        idx = scan_args.inner_out_sit_sot.index(var)
        outer_var = scan_args.outer_out_sit_sot[idx]

        if len(outer_var.clients) == 1:
            client = outer_var.clients[0][0]
            if (client != 'output' and isinstance(client.op,
                                                  theano.tensor.Subtensor)):
                lst = theano.tensor.subtensor.get_idx_list(
                    client.inputs, client.op.idx_list)
                if (len(lst) == 1 and
                        theano.tensor.extract_constant(lst[0]) == -1):
                    return True

        return False

    def get_outer_ndim(self, var, scan_args):

        # Given a variable, determine the number of dimension it would have if
        # it was pushed out of scan
        if (var in scan_args.inner_in_non_seqs or
                isinstance(var, theano.Constant)):

            outer_ndim = var.ndim
        else:
            outer_ndim = var.ndim + 1

        return outer_ndim

    def push_out_inner_vars(self, fgraph, inner_vars, old_scan_node,
                            old_scan_args):

        outer_vars = [None] * len(inner_vars)
        new_scan_node = old_scan_node
        new_scan_args = old_scan_args

        # For the inner_vars that already exist in the outer graph,
        # simply obtain a reference to them
        for idx in range(len(inner_vars)):

            var = inner_vars[idx]

            if var in old_scan_args.inner_in_seqs:
                idx_seq = old_scan_args.inner_in_seqs.index(var)
                outer_vars[idx] = old_scan_args.outer_in_seqs[idx_seq]

            elif var in old_scan_args.inner_in_non_seqs:
                idx_non_seq = old_scan_args.inner_in_non_seqs.index(var)
                outer_vars[idx] = old_scan_args.outer_in_non_seqs[idx_non_seq]

            elif isinstance(var, theano.Constant):
                outer_vars[idx] = var.clone()

            elif var in old_scan_args.inner_out_nit_sot:
                idx_nitsot = old_scan_args.inner_out_nit_sot.index(var)
                outer_vars[idx] = old_scan_args.outer_out_nit_sot[idx_nitsot]

        # For the inner_vars that don't already exist in the outer graph, add
        # them as new nitsot outputs to the scan node.
        idx_add_as_nitsots = [i for i in range(len(outer_vars))
                              if outer_vars[i] is None]
        add_as_nitsots = [inner_vars[idx] for idx in idx_add_as_nitsots]

        if len(add_as_nitsots) > 0:

            new_scan_node = self.add_nitsot_outputs(fgraph, old_scan_node,
                                                    old_scan_args,
                                                    add_as_nitsots)

            new_scan_args = scan_args(new_scan_node.inputs,
                                      new_scan_node.outputs,
                                      new_scan_node.op.inputs,
                                      new_scan_node.op.outputs,
                                      new_scan_node.op.info)

            new_outs = new_scan_args.outer_out_nit_sot[-len(add_as_nitsots):]
            for i in range(len(new_outs)):
                outer_vars[idx_add_as_nitsots[i]] = new_outs[i]

        return outer_vars, new_scan_node, new_scan_args

    def add_nitsot_outputs(self, fgraph, old_scan_node,
                           old_scan_args, new_outputs_inner):

        nb_new_outs = len(new_outputs_inner)

        # Create the initial values for the new nitsot outputs
        # (the initial value is the nb of steps to store. For a nistot,
        # it should be the number of steps performed by scan)
        new_nitsots_initial_value = [old_scan_node.inputs[0]
                                     for i in range(nb_new_outs)]

        # Create the scan_args corresponding to the new scan op to
        # create
        new_scan_args = copy.copy(old_scan_args)
        new_scan_args.inner_out_nit_sot.extend(new_outputs_inner)
        new_scan_args.outer_in_nit_sot.extend(new_nitsots_initial_value)

        # Create the scan op from the scan_args
        new_scan_op = scan_op.Scan(new_scan_args.inner_inputs,
                                   new_scan_args.inner_outputs,
                                   new_scan_args.info)

        # Create the Apply node for the scan op
        new_scan_node = new_scan_op(*new_scan_args.outer_inputs,
                                    **dict(return_list=True))[0].owner

        # Modify the outer graph to make sure the outputs of the new scan are
        # used instead of the outputs of the old scan
        new_node_new_outputs_idx = (len(old_scan_args.outer_outputs) -
                                    len(old_scan_args.outer_out_shared))

        new_node_old_outputs = (
            new_scan_node.outputs[:new_node_new_outputs_idx] +
            new_scan_node.outputs[new_node_new_outputs_idx + nb_new_outs:])

        fgraph.replace_all_validate_remove(
            list(zip(old_scan_node.outputs, new_node_old_outputs)),
            remove=[old_scan_node],
            reason='scanOp_pushout_output')

        return new_scan_node


class ScanInplaceOptimizer(Optimizer):
    """
    Graph optimizer for Scan (makes it run inplace).

    """

    def __init__(self, typeInfer=None, gpu_flag=False, gpua_flag=False):
        Optimizer.__init__(self)
        self.typeInfer = typeInfer
        self.gpu_flag = gpu_flag
        self.gpua_flag = gpua_flag

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())
        fgraph.attach_feature(DestroyHandler())

    def attempt_scan_inplace(self, fgraph, node, output_indices, alloc_ops):
        """Attempts to replace a Scan node by one which computes the specified
        outputs inplace.

        Parameters
        ----------
        fgraph : FunctionGraph
            Function graph in which to attempt the replacement
        node : Apply node
            Scan node to replace by an inplace version
        output_indices : list of integers
            Indices of the outputs to attempt to compute inplace
        alloc_ops : list of Op classes
            Classes that represent operation that allocate new memory and
            that the optimization should duplicate so it can operate inplace
            on them.
        """

        op = node.op

        info = copy.deepcopy(op.info)
        if 'destroy_map' not in info:
            info['destroy_map'] = OrderedDict()

        for out_idx in output_indices:
            info['destroy_map'][out_idx] = [out_idx + 1 + op.info['n_seqs']]

        # inputs corresponding to sequences and n_steps
        ls_begin = node.inputs[:1 + op.n_seqs]
        ls = op.outer_mitmot(node.inputs)
        ls += op.outer_mitsot(node.inputs)
        ls += op.outer_sitsot(node.inputs)
        ls_end = op.outer_shared(node.inputs)
        ls_end += op.outer_nitsot(node.inputs)
        ls_end += op.outer_non_seqs(node.inputs)

        # In `ls`, duplicate any input which has more then one client and is
        # the output of an eligible allocation op
        for i in range(len(ls)):
            inp = ls[i]
            if (len(inp.clients) > 1 and inp.owner and
                    isinstance(inp.owner.op, alloc_ops)):
                ls[i] = inp.owner.op(*inp.owner.inputs)

        n_outs = len(ls)
        for idx in xrange(n_outs):
            if ls[idx] in ls[:idx]:
                ls[idx] = deep_copy_op(ls[idx])

        inputs = ls_begin + ls + ls_end
        if self.typeInfer is None:
            typeConstructor = None
        else:
            typeConstructor = self.typeInfer(node)

        new_op = scan_op.Scan(op.inputs,
                              op.outputs,
                              info,
                              typeConstructor=typeConstructor)

        # Do not call make_node for test_value
        new_outs = new_op(*inputs, **dict(return_list=True))
        try:
            fgraph.replace_all_validate_remove(
                list(zip(node.outputs, new_outs)),
                remove=[node],
                reason='scanOp_make_inplace')
            return new_outs[0].owner
        except InconsistencyError:
            # Failed moving output to be computed inplace
            return node

    def apply(self, fgraph):

        # Depending on the values of gpu_flag and gpua_flag, get the list of
        # memory allocation ops that the optimization should be able to handle
        alloc_ops = (Alloc, AllocEmpty)
        if self.gpu_flag:
            alloc_ops += (theano.sandbox.cuda.GpuAlloc,
                          theano.sandbox.cuda.GpuAllocEmpty)
        if self.gpua_flag:
            # gpuarray might be imported but not its GpuAlloc and
            # GpuAllopEmpty ops.
            try:
                alloc_ops += (theano.gpuarray.GpuAlloc,
                              theano.gpuarray.GpuAllocEmpty)
            except:
                pass

        nodes = fgraph.toposort()[::-1]
        scan_nodes = [x for x in nodes
                      if (isinstance(x.op, scan_op.Scan) and
                          x.op.info['gpu'] == self.gpu_flag and
                          x.op.info['gpua'] == self.gpua_flag)]
        for scan_idx in xrange(len(scan_nodes)):

            # First attempt to make the Scan compute inplace every recurrent
            # output that seems like it could be computed inplace. If that
            # fails, go through these outputs individually, trying each of
            # them.
            original_node = scan_nodes[scan_idx]
            op = original_node.op
            n_outs = (op.info['n_mit_mot'] +
                      op.info['n_mit_sot'] +
                      op.info['n_sit_sot'])

            # Generate a list of outputs on which the node could potentially
            # operate inplace.
            out_indices = []
            for out_idx in range(n_outs):
                inp_idx = 1 + op.n_seqs + out_idx
                inp = original_node.inputs[inp_idx]

                # If the input is from an eligible allocation node, attempt to
                # be inplace on it, even if other nodes are modifying it
                # inplace.
                if inp.owner and isinstance(inp.owner.op, alloc_ops):
                    out_indices.append(out_idx)
                    continue

                # If the input is not from an eligible allocation node, only
                # attempt to be inplace on it if nothing else is currently
                # inplace on it.
                input_used_inplace = False
                for c in original_node.inputs[inp_idx].clients:
                    client = c[0]

                    # Get the indices of this client's inputs on which it
                    # operates inplace
                    if hasattr(client.op, 'destroy_map'):
                        # This flattens the content of destroy_map.values()
                        # which is a list of lists
                        inplace_inp_indices = sum(client.op.destroy_map.values(), [])

                        inplace_inps = [client.inputs[i] for i in inplace_inp_indices]
                        if original_node.inputs[inp_idx] in inplace_inps:
                            input_used_inplace = True
                            break

                if not input_used_inplace:
                    out_indices.append(out_idx)

            node = self.attempt_scan_inplace(fgraph, scan_nodes[scan_idx],
                                             out_indices, alloc_ops)

            if node is original_node:
                # Making the scan compute all plausible recurrent outputs
                # inplace has failed. Attempt all plausible recurrent output
                # individually.
                for pos in out_indices:
                    node = self.attempt_scan_inplace(fgraph, node, [pos],
                                                     alloc_ops)


class ScanSaveMem(gof.Optimizer):
    """
    Graph Optimizer that reduces scan memory consumption.

    """

    def __init__(self):
        gof.Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(gof.toolbox.ReplaceValidate())

    def process_node(self, fgraph, node):

        # helpful functions
        def select_min(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.minimum(x, y)

        def select_max(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return tensor.maximum(x, y)

        def sanitize(x):
            if x is None:
                return None
            else:
                return tensor.as_tensor_variable(x)

        if hasattr(fgraph, 'shape_feature'):
            shape_of = node.fgraph.shape_feature.shape_of
        else:
            # Each access to shape_of is in a try..except block in order to
            # use a default version when the variable is not in the shape_of
            # dictionary.
            shape_of = OrderedDict()
        # 1. Initialization of variables
        # Note 1) We do not actually care about outputs representing shared
        # variables (those have no intermediate values) so it is safer to
        # ignore them and not change them in any way. To simplify the
        # optimizations I construct the variable ``c_outs`` ( that counts
        # outputs up to those we care) and the list ``init_l`` which for any
        # output we care says the length of its initial state. Note that
        # defining ``init_l`` for mit_mot sequences is a bit trickier but
        # it is safe to set it to 0
        op = node.op
        c_outs = op.n_mit_mot + op.n_mit_sot + op.n_sit_sot + op.n_nit_sot

        init_l = [0 for x in xrange(op.n_mit_mot)]
        init_l += [abs(min(v)) for v in op.tap_array[op.n_mit_mot:]]
        init_l += [0 for x in xrange(op.n_nit_sot)]
        # 2. Check the clients of each output and see for how many steps
        # does scan need to run

        # This comparison checks if there is any uncounted output, which
        # can only be an output corresponding to a shared variable

        # 2.1 Initialize
        # global_nsteps is a dictionary having two fields ( 'real' deals
        # with int values, 'sym' with symbolic ones) or None
        # given that a scan op has k outputs o_1, .. o_k and each
        # output has n_j clients c_1^1, c_1^2, .. c_1^{n_1}, c_2^1, ..,
        # global_nsteps is None if any of the clients is different
        # from a subtensor or its real and sym field equal to
        # max(c_i_j.idx_list[0].stop), meaning store up to which maximal
        # index(step) for any output scan actually needs to compute
        # In other words n_steps should be equal to this maximal !
        # Note: if we have a shared variable that gets updated at every step
        # of the loop, reducing the number of steps will affect the the
        # value of the shared variable after the loop so we need not to
        # change the number of steps in that case. To do this we set
        # global_nsteps to None which is seen as a flag that nothing needs
        # to be done
        assert len(node.outputs) >= c_outs
        if len(node.outputs) == c_outs:
            global_nsteps = {'real': -1, 'sym': []}
        else:
            global_nsteps = None

        # Keeps track of the original slices that each client represent
        slices = [None for o in node.outputs]

        # A list for each output indicating how many intermediate values
        # should be stored. If negative it means none of the intermediate
        # values (i.e. the output can be removed since it is not used
        # afterwards in the computations), if 0 it means that all
        # intermediate values are required, otherwise is up to that number
        # of intermediate values
        # Note that for mit_mot outputs and shared outputs we can not change
        # the number of intermediate steps stored without affecting the
        # result of the op
        store_steps = [0 for o in xrange(op.n_mit_mot)]
        store_steps += [-1 for o in node.outputs[op.n_mit_mot:c_outs]]
        # Flag that says if an input has changed and we need to do something
        # or not
        flag_store = False

        # 2.2 Loop over the clients
        for i, out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            slices[i] = []
            for cl, _ in out.clients:

                # 2.1 outputs of the function
                # => output needs all its intermediate values
                if type(cl) == str:
                    # if the node is actually an output, then
                    # we need to store the entire thing
                    global_nsteps = None
                    slices[i] = None
                    break
                # 2.2 non-subtensor nodes
                # => output needs all its intermediate values
                elif not isinstance(cl.op, tensor.Subtensor):
                    global_nsteps = None
                    slices[i] = None
                    break
                # 2.3 subtensor nodes
                # => output might need to store just a subset of its values
                else:
                    # 2.3.1 extract idx list of subtensor
                    this_slice = tensor.get_idx_list(cl.inputs,
                                                     cl.op.idx_list)
                    if this_slice is None:
                        # if unable to extract idx_list
                        # => outputs needs all its intermediate values
                        global_nsteps = None
                        slices[i] = None
                        break

                    # 2.3.2 extract the begin/end of the first dimension
                    if i >= op.n_mit_mot:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = out.shape[0]
                    cf_slice = tensor.get_canonical_form_slice(
                        this_slice[0], length)
                    slices[i] += [(cf_slice, this_slice)]

                    if (isinstance(this_slice[0], slice) and
                            this_slice[0].stop is None):
                        global_nsteps = None
                    if isinstance(cf_slice[0], slice):
                        stop = tensor.basic.extract_constant(cf_slice[0].stop)
                    else:
                        stop = tensor.basic.extract_constant(cf_slice[0]) + 1
                    if stop == maxsize or stop == length:
                        stop = None
                    else:
                        # there is a **gotcha** here ! Namely, scan returns an
                        # array that contains the initial state of the output
                        # as well. Which means that if have a initial state of
                        # length 3, and you look for 5 steps you get an output
                        # y of length 8. If you only use y[:5], this does not
                        # mean that you only need to loop for 5 steps but
                        # actually only for 2 steps ( the first 3 are the
                        # initial state)
                        stop = stop - init_l[i]

                    # 2.3.3 we might get away with less number of steps
                    if stop is not None and global_nsteps is not None:
                        # yes if it is a tensor
                        if isinstance(stop, tensor.Variable):
                            global_nsteps['sym'] += [stop]
                        # not if it is maxsize
                        elif (type(stop) in integer_types and
                              stop == maxsize):
                            global_nsteps = None
                        # yes if it is a int k, 0 < k < maxsize
                        elif (type(stop) in integer_types and
                              global_nsteps['real'] < stop):
                            global_nsteps['real'] = stop
                        # yes if it is a int k, 0 < k < maxsize
                        elif (type(stop) in integer_types and stop > 0):
                            pass
                        # not otherwise
                        else:
                            global_nsteps = None

        # 2.3. Analyze global_nsteps to figure out for how many steps scan
        # needs to iterate
        if global_nsteps is not None:
            nw_steps = node.inputs[0]

            # there are some symbolic tensors that limit the number of
            # steps
            if len(global_nsteps['sym']) == 0:
                sym_steps = None
            else:
                sym_steps = global_nsteps['sym'][0]
                for c in global_nsteps['sym'][1:]:
                    sym_steps = tensor.maximum(sym_steps, c)

            if global_nsteps['real'] >= 0:
                real_steps = global_nsteps['real']
            else:
                real_steps = None
            nw_steps = select_min(select_max(sym_steps, real_steps),
                                  node.inputs[0])

            # Make sure the ScanSaveMem optimization never makes the new
            # number of steps to be 0 (this could happen, for instance, if
            # the optimization detects that the outputs of the Scan go through
            # subtensor nodes that end up taking no elements) because Scan with
            # 0 iterations are not supported. Make sure the new number of steps
            # is at least 1.
            nw_steps = select_max(nw_steps, 1)
        else:
            nw_steps = node.inputs[0]
            global_nsteps = None

        # 2.4 Loop over the clients again now looking just to see how many
        # intermediate steps to store
        for i, out in enumerate(node.outputs[:c_outs]):
            # look at all its clients
            for cl, _ in out.clients:
                if type(cl) == str:
                    store_steps[i] = 0
                    break
                elif not isinstance(cl.op, tensor.Subtensor):
                    store_steps[i] = 0
                    break
                else:
                    this_slice = tensor.get_idx_list(cl.inputs,
                                                     cl.op.idx_list)
                    if this_slice is None:
                        store_steps[i] = 0
                        break

                    if (isinstance(this_slice[0], slice) and
                            this_slice[0].start is None):
                        store_steps[i] = 0
                        break

                    if i > op.n_mit_mot:
                        length = node.inputs[0] + init_l[i]
                    else:
                        try:
                            length = shape_of[out][0]
                        except KeyError:
                            length = out.shape[0]
                    cf_slice = tensor.get_canonical_form_slice(
                        this_slice[0], length)

                    if isinstance(cf_slice[0], slice):
                        start = tensor.basic.extract_constant(
                            cf_slice[0].start)
                    else:
                        start = tensor.basic.extract_constant(cf_slice[0])
                    if start == 0 or store_steps[i] == 0:
                        store_steps[i] = 0
                    else:
                        # The "+ 1" is because of the memory pre-allocation
                        # mechanism used to in the Scan op to reduce overhead.
                        # To prevent aliasing between the inputs and outputs
                        # of recurrent states, it requires that the buffer be
                        # large enough to that, the new state and the oldest
                        # tap needed don't occupy the sample place in the
                        # circular buffer. For now, this only needs to be done
                        # for mitsots and sitsots (because mitmots are not
                        # currently supported by the mechanism) and only if
                        # the pre-allocation mechanism is activated.
                        prealloc_outs = theano.config.scan.allow_output_prealloc

                        first_mitsot_idx = node.op.n_mit_mot
                        last_sitsot_idx = (node.op.n_mit_mot +
                                           node.op.n_mit_sot +
                                           node.op.n_sit_sot - 1)
                        preallocable_output = (first_mitsot_idx <= i <= last_sitsot_idx)

                        if (prealloc_outs and preallocable_output):
                            pval = select_max(nw_steps - start + init_l[i],
                                              init_l[i] + 1)
                        else:
                            pval = select_max(nw_steps - start + init_l[i],
                                              init_l[i])

                        if store_steps[i] != -1:
                            pval = select_max(pval, store_steps[i])

                        # TODO: Simplify the number of steps needed.
                        # FB: This need good testing, left to later.
                        #     call get_scalar_constant_value()? it can
                        # return python/numpy scalar or numpy.ndarray
                        # currently.
                        # pval = pre_greedy_local_optimizer(list_opt_slice,
                        #                                  pval)
                        # pval = pre_constant_merge([pval])[0]
                        # if (isinstance(pval, theano.tensor.TensorConstant)
                        # and
                        #    pval.dtype.startswith('int')):
                        #    try:
                        #        pval = int(pval.data)
                        #    except Exception:
                        #        pass

                        store_steps[i] = pval
                        flag_store = True

        orphane_outs = [i for i, x in enumerate(store_steps)
                        if (type(x) is int) and (x < 0)]
        flag_store = flag_store or (len(orphane_outs) > 0)
        # 3. is there anything to change ?
        if (flag_store or global_nsteps is not None):
            # 3.1 initialize inputs for the new scan
            old_outputs = []
            nw_inputs = list(node.inputs)
            nw_inputs[0] = nw_steps

            # 3.2 check orphane outputs to see if we can eliminate any
            required, not_required = scan_utils.scan_can_remove_outs(
                node.op, orphane_outs)
            # 3.3. compose replace pairs for those nodes that need not
            # to store everything in memory ( or ar orphane and required
            # by the inner function .. )
            replaced_outs = []
            offset = 1 + op.n_seqs + op.n_mit_mot
            for idx, _val in enumerate(store_steps[op.n_mit_mot:]):
                i = idx + op.n_mit_mot
                if not(type(_val) is int and _val <= 0 and i not in required):

                    if idx + op.n_mit_mot in required:
                        val = 1
                    else:
                        val = _val
                    # If the memory for this output has been pre-allocated
                    # before going into the scan op (by an alloc node)
                    if idx < op.n_mit_sot + op.n_sit_sot:
                        # In case the input is still an alloc node, we
                        # actually have two options:
                        #   a) the input is a set_subtensor, in that case we
                        #      can replace the initial tensor by a slice,
                        #   b) it is not, and we simply take a slice of it.
                        # TODO: commit change below with Razvan
                        if (nw_inputs[offset + idx].owner and
                            isinstance(nw_inputs[offset + idx].owner.op,
                                       tensor.IncSubtensor) and
                            isinstance(
                                nw_inputs[offset + idx].owner.op.idx_list[0],
                                slice)):

                            assert isinstance(nw_inputs[offset + idx].owner.op,
                                              tensor.IncSubtensor)
                            _nw_input = nw_inputs[offset + idx].owner.inputs[1]
                            cval = tensor.as_tensor_variable(val)
                            initl = tensor.as_tensor_variable(init_l[i])
                            tmp_idx = tensor.switch(cval < initl,
                                                    cval + initl,
                                                    cval - initl)
                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                                             tmp_idx)
                            tmp = pre_constant_merge([tmp])[0]

                            nw_input = scan_utils.expand_empty(_nw_input, tmp)
                        else:
                            tmp = tensor.as_tensor_variable(val)
                            initl = tensor.as_tensor_variable(init_l[i])
                            tmp = tensor.maximum(tmp, initl)
                            tmp = pre_greedy_local_optimizer(list_opt_slice,
                                                             tmp)
                            tmp = pre_constant_merge([tmp])[0]
                            nw_input = nw_inputs[offset + idx][:tmp]

                        nw_inputs[offset + idx] = nw_input
                        replaced_outs.append(op.n_mit_mot + idx)
                        odx = op.n_mit_mot + idx
                        old_outputs += [(odx, [x[0].outputs[0] for x in
                                        node.outputs[odx].clients])]
                    # If there is no memory pre-allocated for this output
                    elif idx < op.n_mit_sot + op.n_sit_sot + op.n_nit_sot:

                        pos = (op.n_mit_mot + idx + op.n_seqs +
                               1 + op.n_shared_outs)
                        if nw_inputs[pos] == node.inputs[0]:
                            nw_inputs[pos] = val
                        odx = op.n_mit_mot + idx
                        replaced_outs.append(odx)
                        old_outputs += [(odx, [x[0].outputs[0] for x in
                                        node.outputs[odx].clients])]
            # 3.4. Recompute inputs for everything else based on the new
            # number of steps
            if global_nsteps is not None:
                for idx, val in enumerate(store_steps[op.n_mit_mot:]):
                    if val == 0:
                        # val == 0 means that we want to keep all intermediate
                        # results for that state, including the initial values.
                        if idx < op.n_mit_sot + op.n_sit_sot:
                            in_idx = offset + idx
                            # Number of steps in the initial state
                            initl = init_l[op.n_mit_mot + idx]

                            # If the initial buffer has the form
                            # inc_subtensor(zeros(...)[...], _nw_input)
                            # we want to make the zeros tensor as small as
                            # possible (nw_steps + initl), and call
                            # inc_subtensor on that instead.
                            # Otherwise, simply take 0:(nw_steps+initl).
                            if ((nw_inputs[in_idx].owner and
                                 isinstance(nw_inputs[in_idx].owner.op,
                                            tensor.IncSubtensor) and
                                 isinstance(
                                     nw_inputs[in_idx].owner.op.idx_list[0],
                                     slice))):
                                _nw_input = nw_inputs[in_idx].owner.inputs[1]
                                nw_input = scan_utils.expand_empty(_nw_input,
                                                                   nw_steps)
                                nw_inputs[in_idx] = nw_input
                            else:
                                nw_input = nw_inputs[in_idx][:(initl + nw_steps)]

                        elif idx < op.n_mit_sot + op.n_sit_sot + op.n_nit_sot:
                            in_idx = offset + idx + op.n_shared_outs
                            if nw_inputs[in_idx] == node.inputs[0]:
                                nw_inputs[in_idx] = nw_steps

            # 3.5 Remove unwanted orphane outputs
            (inps, outs, info, node_ins, compress_map) = \
                scan_utils.compress_outs(op, not_required, nw_inputs)
            inv_compress_map = OrderedDict()
            for k, v in iteritems(compress_map):
                inv_compress_map[v] = k

            node_ins = [pre_greedy_local_optimizer(list_opt_slice, x) for x in
                        node_ins]
            node_ins = pre_constant_merge(node_ins)
            # 3.6 Compose the new scan
            # TODO: currently we don't support scan with 0 step. So
            # don't create one.
            # For test, mark that savemem have optimized this node
            info['_scan_savemem_visited'] = True
            if theano.tensor.extract_constant(node_ins[0]) == 0:
                return

            # Do not call make_node for test_value
            new_outs = scan_op.Scan(inps, outs, info)(*node_ins,
                                                      **dict(return_list=True))

            old_new = []
            # 3.7 Get replace pairs for those outputs that do not change
            # the number of intermediate steps stored
            for idx, sl in enumerate(slices):
                if global_nsteps and sl is not None and store_steps[idx] == 0:
                    for hdx, cl in enumerate(node.outputs[idx].clients):
                        cnf_slice, old_slices = sl[hdx]
                        # Sanitize the nw_slice by converting ints back into
                        # constants :) I only need to do this for the first
                        # slice since that is the only slice

                        if isinstance(cnf_slice[0], slice):
                            fslice = slice(
                                sanitize(cnf_slice[0].start),
                                sanitize(cnf_slice[0].stop),
                                sanitize(cnf_slice[0].step))
                        else:
                            fslice = sanitize(cnf_slice[0])

                        nw_slice = (fslice,) + tuple(old_slices[1:])
                        nw_pos = inv_compress_map[idx]

                        subtens = tensor.Subtensor(nw_slice)
                        # slice inputs
                        sl_ins = tensor.Subtensor.collapse(
                            nw_slice,
                            lambda entry: isinstance(entry,
                                                     tensor.Variable))
                        new_o = subtens(new_outs[nw_pos], *sl_ins)
                        if new_o.ndim > 0:
                            new_o = new_o[::cnf_slice[1]]
                        replaced_outs.append(idx)
                        old_new += [(cl[0].outputs[0], new_o)]
            # 3.8. Get replace pairs for those outputs that change
            # the number of stored intermediate steps
            for pos, old_outs in old_outputs:
                if len(old_outs) > 0:
                    nw_pos = compress_map[pos]
                    for k, old in enumerate(old_outs):
                        # Get the correct slice
                        cnf_slice, old_slices = slices[pos][k]
                        if type(cnf_slice[0]) is slice:
                            start = (cnf_slice[0].start - nw_steps -
                                     init_l[pos] + store_steps[pos])
                            if (cnf_slice[0].stop is not None and
                                    cnf_slice[0].stop != maxsize):
                                stop = (cnf_slice[0].stop - nw_steps -
                                        init_l[pos] + store_steps[pos])
                            else:
                                stop = None
                            nw_slice = ((slice(sanitize(start),
                                               sanitize(stop),
                                               sanitize(cnf_slice[0].step)),) +
                                        tuple(old_slices[1:]))

                        else:
                            position = (cnf_slice[0] - nw_steps -
                                        init_l[pos] + store_steps[pos])

                            nw_slice = (sanitize(position),) + tuple(
                                old_slices[1:])
                        subtens = tensor.Subtensor(nw_slice)
                        sl_ins = tensor.Subtensor.collapse(
                            nw_slice,
                            lambda entry: isinstance(entry,
                                                     tensor.Variable))
                        new_o = subtens(new_outs[nw_pos], *sl_ins)
                        if new_o.ndim > 0:
                            new_o = new_o[::cnf_slice[1]]
                        old_new += [(old, new_o)]

            # 3.9. Get replace pairs for all other nodes
            if flag_store or global_nsteps is not None:
                for idx, o in enumerate(node.outputs):
                    if not (idx in replaced_outs) and idx not in not_required:
                        nw_pos = compress_map[idx]
                        old_new += [(o, new_outs[nw_pos])]
                # Check if the new outputs depend on the old scan node
                old_scan_is_used = [scan_utils.find_up(new.owner, node)
                                    for old, new in old_new]
                if any(old_scan_is_used):
                    return False
                remove = [old.owner for (old, new) in old_new]
                # As Fred suggested assert that also the old node is not in
                # the Graph as that will make things suboptimal
                remove.append(node)
                fgraph.replace_all_validate_remove(old_new,
                                                   remove,
                                                   reason='scanOp_save_mem')

    def apply(self, fgraph):

        nodelist = [x for x in fgraph.toposort() if isinstance(x.op,
                                                               scan_op.Scan)]
        for node in nodelist:
            self.process_node(fgraph, node)


class ScanMerge(gof.Optimizer):
    """
    Graph Optimizer that merges different scan ops.

    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(gof.toolbox.ReplaceValidate())

    def merge(self, nodes):

        if nodes[0].op.as_while:
            as_while = True
            condition = nodes[0].op.outputs[-1]
        else:
            as_while = False

        info = OrderedDict()
        info['tap_array'] = []
        info['n_seqs'] = sum([nd.op.n_seqs for nd in nodes])
        info['n_mit_mot'] = sum([nd.op.n_mit_mot for nd in nodes])
        info['n_mit_mot_outs'] = sum([nd.op.n_mit_mot_outs for nd in nodes])
        info['mit_mot_out_slices'] = []
        info['n_mit_sot'] = sum([nd.op.n_mit_sot for nd in nodes])
        info['n_sit_sot'] = sum([nd.op.n_sit_sot for nd in nodes])
        info['n_shared_outs'] = sum([nd.op.n_shared_outs for nd in nodes])
        info['n_nit_sot'] = sum([nd.op.n_nit_sot for nd in nodes])
        info['truncate_gradient'] = nodes[0].op.truncate_gradient
        info['name'] = '&'.join([nd.op.name for nd in nodes])
        info['mode'] = nodes[0].op.mode
        info['gpu'] = False
        info['as_while'] = as_while
        info['profile'] = nodes[0].op.profile
        info['allow_gc'] = nodes[0].op.allow_gc

        # We keep the inner_ins and inner_outs of each original node separated.
        # To be able to recombine them in the right order after the clone,
        # we also need to split them by types (seq, mitmot, ...).
        # On the other hand, outer_ins, outer_outs and info are held together.
        inner_ins = [[] for nd in nodes]
        outer_ins = []
        inner_outs = [[] for nd in nodes]
        outer_outs = []

        def rename(ls, suffix):
            for k in ls:
                if k.name:
                    k.name += str(suffix)
            return ls

        for idx, nd in enumerate(nodes):
            # Seq
            inner_ins[idx].append(rename(nd.op.inner_seqs(nd.op.inputs), idx))
            outer_ins += rename(nd.op.outer_seqs(nd.inputs), idx)

        for idx, nd in enumerate(nodes):
            # MitMot
            inner_ins[idx].append(
                rename(nd.op.inner_mitmot(nd.op.inputs), idx))
            inner_outs[idx].append(nd.op.inner_mitmot_outs(nd.op.outputs))
            info['tap_array'] += nd.op.mitmot_taps()
            info['mit_mot_out_slices'] += nd.op.mitmot_out_taps()
            outer_ins += rename(nd.op.outer_mitmot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitmot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # MitSot
            inner_ins[idx].append(
                rename(nd.op.inner_mitsot(nd.op.inputs), idx))
            inner_outs[idx].append(nd.op.inner_mitsot_outs(nd.op.outputs))
            info['tap_array'] += nd.op.mitsot_taps()
            outer_ins += rename(nd.op.outer_mitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_mitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # SitSot
            inner_ins[idx].append(
                rename(nd.op.inner_sitsot(nd.op.inputs), idx))
            info['tap_array'] += [[-1] for x in xrange(nd.op.n_sit_sot)]
            inner_outs[idx].append(nd.op.inner_sitsot_outs(nd.op.outputs))
            outer_ins += rename(nd.op.outer_sitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_sitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            inner_ins[idx].append(
                rename(nd.op.inner_shared(nd.op.inputs), idx))
            outer_ins += rename(nd.op.outer_shared(nd.inputs), idx)

        for idx, nd in enumerate(nodes):
            # NitSot
            inner_outs[idx].append(nd.op.inner_nitsot_outs(nd.op.outputs))
            outer_ins += rename(nd.op.outer_nitsot(nd.inputs), idx)
            outer_outs += nd.op.outer_nitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            outer_outs += nd.op.outer_shared_outs(nd.outputs)
            inner_outs[idx].append(nd.op.inner_shared_outs(nd.op.outputs))

        for idx, nd in enumerate(nodes):
            # Non Seqs
            inner_ins[idx].append(
                rename(nd.op.inner_non_seqs(nd.op.inputs), idx))
            outer_ins += rename(nd.op.outer_non_seqs(nd.inputs), idx)

        # Add back the number of steps
        outer_ins = [nodes[0].inputs[0]] + outer_ins

        if as_while:
            # add the condition, which was the one of nodes[0]
            inner_outs[0].append([condition])

        # Clone the inner graph of each node independently
        for idx, nd in enumerate(nodes):
            # concatenate all inner_ins and inner_outs of nd
            flat_inner_ins = sum(inner_ins[idx], [])
            flat_inner_outs = sum(inner_outs[idx], [])
            # clone
            flat_inner_ins, flat_inner_outs = scan_utils.reconstruct_graph(
                flat_inner_ins, flat_inner_outs)
            # split the new inner variables again in seq, mitmot, etc.
            new_inner_ins = []
            count = 0
            for nl in inner_ins[idx]:
                seq_len = len(nl)
                new_inner_ins.append(flat_inner_ins[count:(count + seq_len)])
                count += seq_len

            new_inner_outs = []
            count = 0
            for nl in inner_outs[idx]:
                seq_len = len(nl)
                new_inner_outs.append(flat_inner_outs[count:(count + seq_len)])
                count += seq_len

            inner_ins[idx] = new_inner_ins
            inner_outs[idx] = new_inner_outs

        # Flatten inner_ins and inner_outs so that all seqs are first,
        # then mitmot, etc.
        new_inner_ins = []
        new_inner_outs = []
        nb_ins_groups = len(inner_ins[0])
        nb_outs_groups = len(inner_outs[0])
        for idx, nd in enumerate(nodes):
            # All inner_ins should have the same length
            assert len(inner_ins[idx]) == nb_ins_groups

            # All inner_outs should have the same length, except if as_while,
            # in which case the first one should have one more element
            if as_while and idx > 0:
                assert len(inner_outs[idx]) == nb_outs_groups - 1
            else:
                assert len(inner_outs[idx]) == nb_outs_groups

        for gr_idx in range(nb_ins_groups):
            for idx, nd in enumerate(nodes):
                new_inner_ins += inner_ins[idx][gr_idx]

        for gr_idx in range(nb_outs_groups):
            for idx, nd in enumerate(nodes):
                if as_while and idx > 0 and gr_idx == (nb_outs_groups - 1):
                    # There is no condition on that node, skip it
                    pass
                else:
                    new_inner_outs += inner_outs[idx][gr_idx]

        new_op = scan_op.Scan(new_inner_ins, new_inner_outs, info)
        new_outs = new_op(*outer_ins)

        if not isinstance(new_outs, (list, tuple)):
            new_outs = [new_outs]

        return list(zip(outer_outs, new_outs))

    def belongs_to_set(self, node, set_nodes):
        """
        This function checks if node `node` belongs to `set_nodes`, in the
        sense that it can be merged together with every other node in
        `set_nodes`. In order for two nodes to be mergeable, they have to go
        over the same number of steps, have the same condition (if any),
        have the same value for truncate_gradient, and have the same mode.
        Questionable, we should also consider profile ?

        """
        rep = set_nodes[0]
        if (rep.op.as_while != node.op.as_while or
                node.op.truncate_gradient != rep.op.truncate_gradient or
                node.op.mode != rep.op.mode):
            return False

        nsteps = node.inputs[0]
        try:
            nsteps = int(get_scalar_constant_value(nsteps))
        except tensor.NotScalarConstantError:
            pass

        rep_nsteps = rep.inputs[0]
        try:
            rep_nsteps = int(get_scalar_constant_value(rep_nsteps))
        except tensor.NotScalarConstantError:
            pass

        # Check to see if it is an input of a different node
        for nd in set_nodes:
            if find_up(node, nd) or find_up(nd, node):
                return False

        if not node.op.as_while:
            return nsteps == rep_nsteps
        cond = node.op.outputs[-1]
        rep_cond = rep.op.outputs[-1]
        same_cond = scan_utils.equal_computations([cond], [rep_cond],
                                                  node.op.inputs,
                                                  rep.op.inputs)
        return same_cond and (nsteps == rep_nsteps)

    def apply(self, fgraph):
        # Collect all scan nodes ordered according to toposort
        scan_nodes = [nd for nd in fgraph.toposort()
                      if isinstance(nd.op, scan_op.Scan)]

        # All sets of possibly mergeable nodes
        all_sets = []

        for nd in scan_nodes:
            belongs_to_set_idx = -1
            for pos, subset in enumerate(all_sets):
                if self.belongs_to_set(nd, subset):
                    belongs_to_set_idx = pos
                    # It is possible that nd belongs to more than one subset.
                    # For instance, if we have 3 Scan nodes X, Y and Z, if Z
                    # depends on the output of X, then X and Z are incompatible
                    # and would create different subsets, but Y could be
                    # compatible with both X and Z. We choose the first one.
                    break

            if belongs_to_set_idx == -1:
                all_sets.append([nd])
            else:
                all_sets[belongs_to_set_idx].append(nd)

        for subset in all_sets:
            if len(subset) > 1:
                proposal = self.merge(subset)
                fgraph.replace_all_validate_remove(proposal,
                                                   remove=subset,
                                                   reason='scanOp_merge')


def has_duplicates(l):
    """
    Returns true if l has any duplicates (according to __eq__).

    """
    return len(set(l)) < len(l)


def make_equiv(lo, li):
    """
    Builds a dictionary of equivalences between inner inputs based on
    the equivalence of their corresponding outer inputs.

    """
    seeno = OrderedDict()
    left = []
    right = []
    for o, i in zip(lo, li):
        if o in seeno:
            left += [i]
            right += [o]
        else:
            seeno[o] = i
    return left, right


@gof.local_optimizer([scan_op.Scan])
def scan_merge_inouts(node):
    if not isinstance(node.op, scan_op.Scan):
        return False

    # Do a first pass to merge identical external inputs.
    # Equivalent inputs will be stored in inp_equiv, then a new
    # scan node created without duplicates.
    a = scan_args(node.inputs, node.outputs,
                  node.op.inputs, node.op.outputs, node.op.info)

    inp_equiv = OrderedDict()

    if has_duplicates(a.outer_in_seqs):
        new_outer_seqs = []
        new_inner_seqs = []
        for out_seq, in_seq in zip(a.outer_in_seqs, a.inner_in_seqs):
            if out_seq in new_outer_seqs:
                i = new_outer_seqs.index(out_seq)
                inp_equiv[in_seq] = new_inner_seqs[i]
            else:
                new_outer_seqs.append(out_seq)
                new_inner_seqs.append(in_seq)
        a.outer_in_seqs = new_outer_seqs
        a.inner_in_seqs = new_inner_seqs

    if has_duplicates(a.outer_in_non_seqs):
        new_outer_nseqs = []
        new_inner_nseqs = []
        for out_nseq, in_nseq in zip(a.outer_in_non_seqs, a.inner_in_non_seqs):
            if out_nseq in new_outer_nseqs:
                i = new_outer_nseqs.index(out_nseq)
                inp_equiv[in_nseq] = new_inner_nseqs[i]
            else:
                new_outer_nseqs.append(out_nseq)
                new_inner_nseqs.append(in_nseq)
        a.outer_in_non_seqs = new_outer_nseqs
        a.inner_in_non_seqs = new_inner_nseqs

    if len(inp_equiv) > 0:
        # do the replacement now. The rest will be left to ScanSaveMem
        inner_inputs = a.inner_inputs
        outer_inputs = a.outer_inputs
        info = a.info
        a_inner_outs = a.inner_outputs
        inner_outputs = scan_utils.clone(a_inner_outs, replace=inp_equiv)

        op = scan_op.Scan(inner_inputs, inner_outputs, info)
        outputs = op(*outer_inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        na = scan_args(outer_inputs, outputs, op.inputs, op.outputs, op.info)
        remove = [node]
    else:
        na = a
        remove = []

    # Now that the identical external inputs have been merged, we do a new
    # loop in order to merge external outputs that compute the same things
    # from the same inputs.
    left = []
    right = []

    if has_duplicates(na.outer_in_shared):
        _left, _right = make_equiv(na.outer_in_shared, na.inner_in_shared)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_sit_sot):
        _left, _right = make_equiv(na.outer_in_sit_sot, na.inner_in_sit_sot)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_mit_mot):
        seen = OrderedDict()
        for omm, imm, _sl in zip(na.outer_in_mit_mot,
                                 na.inner_in_mit_mot, na.mit_mot_in_slices):
            sl = tuple(_sl)
            if (omm, sl) in seen:
                simm = seen[(omm, sl)]
                left += imm
                right += simm
            else:
                seen[(omm, sl)] = imm

    if has_duplicates(na.outer_in_mit_sot):
        seen = OrderedDict()
        for oms, ims, _sl in zip(na.outer_in_mit_sot,
                                 na.inner_in_mit_sot,
                                 na.mit_sot_in_slices):
            sl = tuple(_sl)
            if (oms, sl) in seen:
                sims = seen[(oms, sl)]
                left += ims
                right += sims
            else:
                seen[(oms, sl)] = ims

    def map_out(outer_i, inner_o, outer_o, seen):
        # Return the outer input corresponding to an
        # (outer input, inner output) pair. If we see that pair for the first
        # time, return the provided outer output. If an equivalent pair had
        # already been seen, return that one instead.
        # Note that we need to check that the outer input match as well,
        # because they could have different sizes, and the corresponding
        # outer outputs cannot be merged in that case.
        for s_outer_i, s_inner_o, s_outer_o in seen:
            if (equal_computations([inner_o], [s_inner_o], left, right) and
                    outer_i == s_outer_i):
                return s_outer_o
        seen.append((outer_i, inner_o, outer_o))
        return outer_o

    seen = []

    assert len(na.outer_in_nit_sot) == len(na.inner_out_nit_sot)
    assert len(na.inner_out_nit_sot) == len(na.outer_out_nit_sot)
    na.outer_out_nit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(na.outer_in_nit_sot,
                                             na.inner_out_nit_sot,
                                             na.outer_out_nit_sot)]

    seen = []
    assert len(na.outer_in_sit_sot) == len(na.inner_out_sit_sot)
    assert len(na.inner_out_sit_sot) == len(na.outer_out_sit_sot)
    na.outer_out_sit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(na.outer_in_sit_sot,
                                             na.inner_out_sit_sot,
                                             na.outer_out_sit_sot)]

    seen = []
    assert len(na.outer_in_mit_sot) == len(na.inner_out_mit_sot)
    assert len(na.inner_out_mit_sot) == len(na.outer_out_mit_sot)
    na.outer_out_mit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(na.outer_in_mit_sot,
                                             na.inner_out_mit_sot,
                                             na.outer_out_mit_sot)]

    seen = []
    new_outer_out_mit_mot = []
    assert len(na.outer_in_mit_mot) == len(na.inner_out_mit_mot)
    assert len(na.inner_out_mit_mot) == len(na.outer_out_mit_mot)
    assert len(na.outer_out_mit_mot) == len(na.mit_mot_out_slices)
    for outer_imm, inner_omm, outer_omm, osl in zip(na.outer_in_mit_mot,
                                                    na.inner_out_mit_mot,
                                                    na.outer_out_mit_mot,
                                                    na.mit_mot_out_slices):
        for s_outer_imm, s_inner_omm, s_outer_omm, sosl in seen:
            if (osl == sosl and
                equal_computations(inner_omm, s_inner_omm, left, right) and
                    outer_imm == s_outer_imm):

                new_outer_out_mit_mot.append(s_outer_omm)
                break
        else:
            seen.append((outer_imm, inner_omm, outer_omm, osl))
            new_outer_out_mit_mot.append(outer_omm)
    na.outer_out_mit_mot = new_outer_out_mit_mot
    if remove:
        return OrderedDict([("remove", remove)] +
                           list(zip(node.outputs, na.outer_outputs)))
    return na.outer_outputs


class PushOutDot1(gof.Optimizer):
    """
    Graph optimizer for Scan(makes it run inplace).

    """

    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):

        nodes = fgraph.toposort()
        scan_nodes = [x for x in nodes if (isinstance(x.op, scan_op.Scan))]
        for node in scan_nodes:
            self.apply_opt(fgraph, node)

    def apply_opt(self, fgraph, node):
        # Replace pattern of the form
        # x[t] = x[t-1] + dot(seq[t], value)
        # with Sequence.reshape((-1, seq.shape[2])) \dot Value
        # When seq[t] is a vector/matrix  and `value` is a matrix
        # Note that this works when only you need X[-1] in the end
        # and assumes dimshuffle are applied to vectors before calling dot
        op = node.op
        sitsot_ins = op.inner_sitsot(op.inputs)
        sitsot_outs = op.inner_sitsot_outs(op.outputs)
        outer_sitsot = op.outer_sitsot_outs(node)
        seqs = op.inner_seqs(op.inputs)
        for inp, out, outer_out in zip(sitsot_ins, sitsot_outs, outer_sitsot):

            if (out.owner and
                isinstance(out.owner.op, theano.tensor.Elemwise) and
                isinstance(out.owner.op.scalar_op, theano.scalar.Add) and
                inp in out.owner.inputs and
                len(outer_out.clients) == 1 and
                not isinstance(outer_out.clients[0][0], str) and
                isinstance(outer_out.clients[0][0].op, theano.tensor.Subtensor) and
                    outer_out.clients[0][0].op.idx_list == (-1,)):

                x = out.owner.inputs[0]
                if x == inp:
                    x = out.owner.inputs[1]
                # We need to check if x is the result of an outer product
                if (x.owner and isinstance(x.owner.op, theano.tensor.Dot) and
                        x.owner.inputs[0].ndim == 2 and x.owner.inputs[1].ndim == 2):

                    # We need to check if any of the inputs are a sequence
                    inp1 = x.owner.inputs[0]
                    inp2 = x.owner.inputs[1]

                    if inp1 in seqs or inp2 in seqs:
                        new_scan_out = inp1

                        if inp1 in seqs:
                            new_scan_out = inp2
                        idx = sitsot_outs.index(out)
                        # We've found our pattern and need to construct a new
                        # scan node to replace this one. For this we need to
                        # replace the sit_sot output with a nit_sot output

                        # First let us split all arguments according to their
                        # corresponding categories

                        inner_seqs = op.inner_seqs(op.inputs)
                        outer_seqs = op.outer_seqs(node)
                        inner_mitmot = op.inner_mitmot(op.inputs)
                        outer_mitmot = op.outer_mitmot(node)
                        inner_mitmot_outs = op.inner_mitmot_outs(op.outputs)
                        inner_mitsot = op.inner_mitsot(op.inputs)
                        outer_mitsot = op.outer_mitsot(node)
                        inner_mitsot_outs = op.inner_mitsot_outs(op.outputs)
                        inner_sitsot = op.inner_sitsot(op.inputs)
                        outer_sitsot = op.outer_sitsot(node)
                        inner_sitsot_outs = op.inner_sitsot_outs(op.outputs)
                        outer_nitsot = op.outer_nitsot(node)
                        inner_nitsot_outs = op.inner_nitsot_outs(op.outputs)
                        inner_shared = op.inner_shared(op.inputs)
                        outer_shared = op.outer_shared(node)
                        inner_shared_outs = op.inner_shared_outs(op.outputs)
                        inner_non_seqs = op.inner_non_seqs(op.inputs)
                        outer_non_seqs = op.outer_non_seqs(node)

                        new_info = op.info.copy()
                        st = len(op.mitmot_taps()) + len(op.mitsot_taps())

                        new_info['tap_array'] = (
                            new_info['tap_array'][:st + idx] +
                            new_info['tap_array'][st + idx + 1:])
                        new_info['n_sit_sot'] -= 1
                        new_info['n_nit_sot'] += 1
                        inner_sitsot = (inner_sitsot[:idx] +
                                        inner_sitsot[idx + 1:])
                        outer_sitsot = (outer_sitsot[:idx] +
                                        outer_sitsot[idx + 1:])
                        inner_sitsot_outs = (inner_sitsot_outs[:idx] +
                                             inner_sitsot_outs[idx + 1:])
                        # add n_steps as the length
                        inner_nitsot_outs.append(new_scan_out)

                        _new_inner_inps = (inner_seqs +
                                           inner_mitmot +
                                           inner_mitsot +
                                           inner_sitsot +
                                           inner_shared +
                                           inner_non_seqs)
                        _new_inner_outs = (inner_mitmot_outs +
                                           inner_mitsot_outs +
                                           inner_sitsot_outs +
                                           inner_nitsot_outs +
                                           inner_shared_outs)
                        new_inner_inps, new_inner_outs =\
                            scan_utils.reconstruct_graph(_new_inner_inps,
                                                         _new_inner_outs)
                        new_op = scan_op.Scan(new_inner_inps, new_inner_outs,
                                              new_info)
                        _scan_inputs = ([node.inputs[0]] +
                                        outer_seqs +
                                        outer_mitmot +
                                        outer_mitsot +
                                        outer_sitsot +
                                        outer_shared +
                                        outer_nitsot +
                                        [node.inputs[0]] +
                                        outer_non_seqs)

                        new_outs = new_op(*_scan_inputs)
                        if type(new_outs) not in (list, tuple):
                            new_outs = [new_outs]

                        # We need now to pair correctly the new outputs
                        # with the old ones

                        outer_nitsot_outs = new_op.outer_nitsot_outs(new_outs)

                        _val = outer_nitsot_outs[-1]
                        outer_nitsot_outs = outer_nitsot_outs[:-1]
                        if inp1 in seqs:
                            _out_seq = op.outer_seqs(node)[seqs.index(inp1)]
                            # We need to clip the seq to the number of steps
                            _out_seq = _out_seq[:node.inputs[0]]
                            sh0 = _out_seq.shape[0]
                            sh1 = _out_seq.shape[1]
                            sh2 = _out_seq.shape[2]
                            out_seq = _out_seq.dimshuffle(1, 0, 2)
                            out_seq = out_seq.reshape((sh1, sh0 * sh2))
                            sh0 = _val.shape[0]
                            sh1 = _val.shape[1]
                            sh2 = _val.shape[2]

                            val = _val.reshape((sh0 * sh1, sh2))
                            new_out = tensor.dot(out_seq, val)
                        else:
                            _out_seq = op.outer_seqs(node)[seqs.index(inp2)]
                            out_seq = _out_seq.reshape(
                                (_out_seq.shape[0] * _out_seq.shape[1],
                                 _out_seq.shape[2]))

                            val = _val.dimshuffle(1, 0, 2).reshape(
                                (_val.shape[1],
                                 _val.shape[0] * _val.shape[2]))
                            new_out = tensor.dot(val, out_seq)

                        pos = node.outputs.index(outer_out)
                        old_new = list(zip(node.outputs[:pos], new_outs[:pos]))
                        old = node.outputs[pos].clients[0][0].outputs[0]
                        old_new.append((old, new_out))
                        old_new += list(zip(node.outputs[pos + 1:],
                                            new_outs[pos:]))
                        fgraph.replace_all_validate_remove(
                            old_new, remove=[node], reason='scan_pushout_dot1')


# I've added an equilibrium because later scan optimization in the sequence
# can make it such that earlier optimizations should apply. However, in
# general I do not expect the sequence to run more then once
scan_eqopt1 = theano.gof.EquilibriumDB()
scan_seqopt1 = theano.gof.SequenceDB()
scan_eqopt2 = theano.gof.EquilibriumDB()

# scan_eqopt1 before ShapeOpt at 0.1
# This is needed to don't have ShapeFeature trac old Scan that we
# don't want to reintroduce.
optdb.register('scan_eqopt1', scan_eqopt1, .05, 'fast_run', 'scan')
# We run before blas opt at 1.7 and specialize 2.0
# but after stabilize at 1.5. Should we put it before stabilize?
optdb.register('scan_eqopt2', scan_eqopt2, 1.6, 'fast_run', 'scan')
# ScanSaveMem should execute only once per node.
optdb.register('scanOp_save_mem', ScanSaveMem(), 1.61, 'fast_run', 'scan')
optdb.register('scanOp_make_inplace',
               ScanInplaceOptimizer(typeInfer=None,
                                    gpu_flag=False),
               75,
               'fast_run',
               'inplace',
               'scan')

scan_eqopt1.register(
    'all_pushout_opt', scan_seqopt1, 1, 'fast_run', 'scan')


scan_seqopt1.register('scanOp_remove_constants_and_unused_inputs0',
                      opt.in2out(remove_constants_and_unused_inputs_scan,
                                 ignore_newtrees=True),
                      1,
                      'remove_constants_and_unused_inputs_scan',
                      'fast_run',
                      'scan')


scan_seqopt1.register('scanOp_pushout_nonseqs_ops',
                      PushOutNonSeqScan(),
                      2,
                      'fast_run',
                      'scan')


scan_seqopt1.register('scanOp_pushout_seqs_ops',
                      PushOutSeqScan(),
                      3,
                      'fast_run',
                      'scan')


scan_seqopt1.register('scan_pushout_dot1',
                      PushOutDot1(),
                      4,
                      'fast_run',
                      'more_mem',
                      'scan')


scan_seqopt1.register('scanOp_pushout_output',
                      PushOutScanOutput(),
                      5,
                      'fast_run',
                      'more_mem',
                      'scan')


scan_eqopt2.register('constant_folding_for_scan2',
                     opt.in2out(tensor.opt.constant_folding,
                                ignore_newtrees=True),
                     1,
                     'fast_run',
                     'scan')


scan_eqopt2.register('scanOp_remove_constants_and_unused_inputs1',
                     opt.in2out(remove_constants_and_unused_inputs_scan,
                                ignore_newtrees=True),
                     2,
                     'remove_constants_and_unused_inputs_scan',
                     'fast_run',
                     'scan')


# after const merge but before stabilize so that we can have identity
# for equivalent nodes but we still have the chance to hoist stuff out
# of the scan later.
scan_eqopt2.register('scanOp_merge',
                     ScanMerge(),
                     4,
                     'fast_run',
                     'scan')

# After Merge optimization
scan_eqopt2.register('scanop_remove_constants_and_unused_inputs2',
                     opt.in2out(remove_constants_and_unused_inputs_scan,
                                ignore_newtrees=True),
                     5,
                     'remove_constants_and_unused_inputs_scan',
                     'fast_run',
                     'scan')

scan_eqopt2.register('scanOp_merge_inouts',
                     opt.in2out(scan_merge_inouts, ignore_newtrees=True),
                     6,
                     'scan_merge_inouts',
                     'fast_run',
                     'scan')

# After everything else
scan_eqopt2.register('scanOp_remove_constants_and_unused_inputs3',
                     opt.in2out(remove_constants_and_unused_inputs_scan,
                                ignore_newtrees=True),
                     8,
                     'remove_constants_and_unused_inputs_scan',
                     'fast_run',
                     'scan')
