"""
Defines the base class for optimizations as well as a certain
amount of useful generic optimization tools.

"""
from __future__ import absolute_import, print_function, division

from collections import deque
import copy
import logging
import pdb
import sys
import time
import warnings
import traceback

import numpy

import theano
from theano import config
from theano.compat import izip, OrderedDict
from six import string_types, iteritems, itervalues, integer_types
from six.moves import reduce
from theano.gof import graph, op, utils, unify, toolbox
from theano.gof.fg import InconsistencyError
from theano.misc.ordered_set import OrderedSet

from . import destroyhandler as dh

_logger = logging.getLogger('theano.gof.opt')
_optimizer_idx = [0]


def _list_of_nodes(fgraph):
    return list(graph.io_toposort(fgraph.inputs, fgraph.outputs))


class Optimizer(object):
    """
    WRITEME

    An L{Optimizer} can be applied to an L{FunctionGraph} to transform it.
    It can represent an optimization or in general any kind
    of transformation you could apply to an L{FunctionGraph}.

    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def __eq__(self, other):
        # added to override the  __eq__ implementation that may be inherited
        # in subclasses from other bases.
        return id(self) == id(other)

    def __neq__(self, other):
        # added to override the  __neq__ implementation that may be inherited
        # in subclasses from other bases.
        return id(self) != id(other)

    def apply(self, fgraph):
        """
        WRITEME

        Applies the optimization to the provided L{FunctionGraph}. It may
        use all the methods defined by the L{FunctionGraph}. If the
        L{Optimizer} needs to use a certain tool, such as an
        L{InstanceFinder}, it can do so in its L{add_requirements} method.

        """
        pass

    def optimize(self, fgraph, *args, **kwargs):
        """
        WRITEME

        This is meant as a shortcut to:
          opt.add_requirements(fgraph)
          opt.apply(fgraph)

        """
        self.add_requirements(fgraph)
        try:
            orig = theano.tensor.basic.constant.enable
            theano.tensor.basic.constant.enable = False
            ret = self.apply(fgraph, *args, **kwargs)
        finally:
            theano.tensor.basic.constant.enable = orig
        return ret

    def __call__(self, fgraph):
        """
        WRITEME

        Same as self.optimize(fgraph).

        """
        return self.optimize(fgraph)

    def add_requirements(self, fgraph):
        """
        WRITEME

        Add features to the fgraph that are required to apply the optimization.
        For example:
          fgraph.attach_feature(History())
          fgraph.attach_feature(MyFeature())
          etc.

        """
        pass

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print("%s%s %s id=%i" % (
            (' ' * level), self.__class__.__name__, name, id(self)), file=stream)

    @staticmethod
    def print_profile(stream, prof, level=0):
        if prof is not None:
            raise NotImplementedError(
                "The function print_profile must be overrided if the"
                " optimizer return profiling information.")


class FromFunctionOptimizer(Optimizer):
    """
    WRITEME

    """
    def __init__(self, fn, requirements=()):
        self.apply = fn
        self.requirements = requirements

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s id=%i" % (
            ' ' * level,
            str(self.apply),
            id(self)), file=stream)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __str__(self):
        return self.__name__


def optimizer(f):
    """
    Decorator for FromFunctionOptimizer.

    """
    rval = FromFunctionOptimizer(f)
    rval.__name__ = f.__name__
    return rval


def inplace_optimizer(f):
    """
    Decorator for FromFunctionOptimizer.

    """
    dh_handler = dh.DestroyHandler
    requirements = (lambda fgraph:
                    fgraph.attach_feature(dh_handler()),)
    rval = FromFunctionOptimizer(f, requirements)
    rval.__name__ = f.__name__
    return rval


class SeqOptimizer(Optimizer, list):
    # inherit from Optimizer first to get Optimizer.__hash__
    """
    WRITEME

    Takes a list of L{Optimizer} instances and applies them
    sequentially.

    """
    @staticmethod
    def warn(exc, self, optimizer):
        """
        Default failure_callback for SeqOptimizer.

        """
        _logger.error("SeqOptimizer apply %s" % str(optimizer))
        _logger.error("Traceback:")
        _logger.error(traceback.format_exc())
        if config.on_opt_error == 'raise':
            raise exc
        elif config.on_opt_error == 'pdb':
            pdb.post_mortem(sys.exc_info()[2])

    def __init__(self, *opts, **kw):
        """
        WRITEME

        """
        if len(opts) == 1 and isinstance(opts[0], (list, tuple)):
            opts = opts[0]
        self[:] = opts
        self.failure_callback = kw.pop('failure_callback', None)

    def apply(self, fgraph):
        """
        WRITEME

        Applies each L{Optimizer} in self in turn.

        """
        l = []
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            sub_validate_time = [validate_before]
        else:
            sub_validate_time = []
        callback_before = fgraph.execute_callbacks_time
        nb_node_before = len(fgraph.apply_nodes)
        sub_profs = []
        for optimizer in self:
            try:
                t0 = time.time()
                sub_prof = optimizer.optimize(fgraph)
                l.append(float(time.time() - t0))
                sub_profs.append(sub_prof)
                if fgraph.profile:
                    sub_validate_time.append(fgraph.profile.validate_time)
            except AssertionError:
                # do not catch Assertion failures
                raise
            except Exception as e:
                if self.failure_callback:
                    self.failure_callback(e, self, optimizer)
                    continue
                else:
                    raise

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
        else:
            validate_time = None
        callback_time = fgraph.execute_callbacks_time - callback_before
        return (self, l, validate_time, callback_time, nb_node_before,
                len(fgraph.apply_nodes), sub_profs, sub_validate_time)

    def __str__(self):
        return "SeqOpt(%s)" % list.__str__(self)

    def __repr__(self):
        return list.__repr__(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print("%s%s %s id=%i" % (
            (' ' * level), self.__class__.__name__, name, id(self)), file=stream)
        # This way, -1 will do all depth
        if depth != 0:
            depth -= 1
            for opt in self:
                opt.print_summary(stream, level=(level + 2), depth=depth)

    @staticmethod
    def print_profile(stream, prof, level=0):
        (opts, prof, validate_time, callback_time, nb_node_before,
         nb_node_after, sub_profs, sub_validate_time) = prof
        blanc = ('    ' * level)

        print(blanc, "SeqOptimizer", end=' ', file=stream)
        if hasattr(opts, "name"):
            print(blanc, opts.name, end=' ', file=stream)
        elif hasattr(opts, "__name__"):
            print(blanc, opts.__name__, end=' ', file=stream)
        print((" time %.3fs for %d/%d nodes"
               " before/after optimization" % (
                   sum(prof), nb_node_before, nb_node_after)), file=stream)
        print(blanc, "  %.3fs for callback" % (callback_time), file=stream)
        print(blanc, "      %.3fs for fgraph.validate()" % (validate_time), file=stream)
        if level == 0:
            print(blanc, "  time      - (name, class, index) - validate time", file=stream)
        ll = []
        for opt in opts:
            if hasattr(opt, "__name__"):
                ll.append((opt.__name__, opt.__class__.__name__,
                           opts.index(opt)))
            else:
                ll.append((opt.name, opt.__class__.__name__,
                           opts.index(opt)))
        lll = sorted(zip(prof, ll), key=lambda a: a[0])

        for (t, opt) in lll[::-1]:
            # if t < 1:
            #    continue
            if sub_validate_time:
                i = opt[-1]
                val_time = sub_validate_time[i + 1] - sub_validate_time[i]
                print(blanc, '  %.6fs - %s - %.3fs' % (
                    t, opt, val_time), file=stream)
            else:
                print(blanc, '  %.6fs - %s' % (t, opt), file=stream)

            if sub_profs[opt[-1]]:
                opts[opt[-1]].print_profile(stream, sub_profs[opt[-1]],
                                            level=level + 1)
        print(file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        """
        Merge 2 profiles returned by this cass apply() fct.

        """
        new_t = []
        new_l = []
        new_sub_profile = []
        # merge common(same object) opt
        for l in set(prof1[0]).intersection(set(prof2[0])):
            idx1 = prof1[0].index(l)
            idx2 = prof2[0].index(l)
            new_t.append(prof1[1][idx1] +
                         prof2[1][idx2])
            new_l.append(l)
            if hasattr(l, 'merge_profile'):
                assert len(prof1[6][idx1]) == len(prof2[6][idx2])
                new_sub_profile.append(l.merge_profile(prof1[6][idx1],
                                                       prof2[6][idx2]))
            else:
                new_sub_profile.append(None)

        # merge not common opt
        from six import StringIO
        for l in set(prof1[0]).symmetric_difference(set(prof2[0])):
            # The set trick above only work for the same object optimization
            # It don't work for equivalent optimization.
            # So we try to merge equivalent optimization here.
            new_l_names = [o.name for o in new_l]
            if l.name in new_l_names:
                idx = new_l_names.index(l.name)
                io1 = StringIO()
                io2 = StringIO()
                l.print_summary(io1)
                new_l[idx].print_summary(io2)
                if io1.read() == io2.read():
                    if l in prof1[0]:
                        p = prof1
                    else:
                        p = prof2
                    new_t[idx] += p[1][p[0].index(l)]
                    if hasattr(l, 'merge_profile'):
                        assert len(p[6][p[0].index(l)]) == \
                            len(new_sub_profile[idx])
                        new_sub_profile[idx] = l.merge_profile(
                            new_sub_profile[idx], p[6][p[0].index(l)])
                    else:
                        new_sub_profile[idx] = None
                continue
            if l in prof1[0]:
                p = prof1
            else:
                p = prof2
            new_t.append(p[1][p[0].index(l)])
            idx = p[0].index(l)
            new_l.append(l)
            new_sub_profile.append(p[6][idx])

        new_opt = SeqOptimizer(*new_l)
        # We need to assert based on the name as we merge also based on
        # the name.
        assert set([l.name for l in prof1[0]]).issubset(
            set([l.name for l in new_l]))
        assert set([l.name for l in prof2[0]]).issubset(
            set([l.name for l in new_l]))
        assert len(new_t) == len(new_opt) == len(new_sub_profile)
        return (new_opt, new_t, prof1[2] + prof2[2],
                prof1[3] + prof2[3],
                -1, -1, new_sub_profile, [])


class _metadict:
    """
    WRITEME

    """

    # dict that accepts unhashable keys
    # uses an associative list
    # for internal use only
    def __init__(self):
        self.d = {}
        self.l = []

    def __getitem__(self, item):
        return self.get(item, None)

    def __setitem__(self, item, value):
        try:
            self.d[item] = value
        except Exception:
            for i, (key, val) in enumerate(self.l):
                if key == item:
                    self.l[i] = (item, value)
                    return
            self.l.append((item, value))

    def __delitem__(self, item):
        try:
            if item in self.d:
                del self.d[item]
                return
        except TypeError as e:
            assert "unhashable type" in str(e)
        for i, (key, val) in enumerate(self.l):
            if key == item:
                del self.l[i]
                return
            raise KeyError(item)

    def discard(self, item):
        try:
            if item in self.d:
                del self.d[item]
                return
        except TypeError as e:
            assert "unhashable type" in str(e)
        for i, (key, val) in enumerate(self.l):
            if key == item:
                del self.l[i]
                return

    def get(self, item, default):
        try:
            return self.d[item]
        except Exception:
            for item2, value in self.l:
                try:
                    if item == item2:
                        return value
                    if item.equals(item2):
                        return value
                except Exception:
                    if item is item2:
                        return value
            else:
                return default

    def clear(self):
        self.d = {}
        self.l = []

    def __str__(self):
        return "(%s, %s)" % (self.d, self.l)


class MergeFeature(object):
    """
    Keeps track of variables in fgraph that cannot be merged together.

    That way, the MergeOptimizer can remember the result of the last merge
    pass on the fgraph.

    """
    def on_attach(self, fgraph):
        assert not hasattr(fgraph, 'merge_feature')
        fgraph.merge_feature = self

        # For constants
        self.seen_constants = set()
        # variable -> signature (for constants)
        self.const_sig = _metadict()
        # signature -> variable (for constants)
        self.const_sig_inv = _metadict()

        # For all Apply nodes
        # Set of distinct (not mergeable) nodes
        self.nodes_seen = set()
        # Ordered set of distinct (not mergeable) nodes without any input
        self.noinput_nodes = OrderedSet()

        # Each element of scheduled is a list of list of (out, new_out) pairs.
        # Each list of pairs represent the substitution needed to replace all
        # the outputs of a node with the outputs of a replacement candidate.
        # Each node can have several candidates. For instance, if "node" has
        # 2 outputs, and there are 3 replacement candidates, we will have:
        # shelf.scheduled = [
        #    [[(node.out1, cand1.out1), (node.out2, cand1.out2)],
        #     [(node.out1, cand2.out1), (node.out2, cand2.out2)],
        #     [(node.out1, cand3.out1), (node.out2, cand3.out2)]]]
        self.scheduled = []

        # List of (node, candidate) pairs, where we tried to replace node by
        # candidate, but it failed. This is used to avoid infinite loops
        # during the replacement phase.
        self.blacklist = []

        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # If inputs to node change, it is not guaranteed that it is distinct
        # from the other nodes in nodes_seen
        if node in self.nodes_seen:
            self.nodes_seen.discard(node)
            self.process_node(fgraph, node)

        # Since we are in on_change_input, node should have inputs.
        if not isinstance(node, string_types):
            assert node.inputs

        if isinstance(new_r, graph.Constant):
            self.process_constant(fgraph, new_r)

    def on_import(self, fgraph, node, reason):
        for c in node.inputs:
            if isinstance(c, graph.Constant):
                self.process_constant(fgraph, c)

        self.process_node(fgraph, node)

    def on_prune(self, fgraph, node, reason):
        self.nodes_seen.discard(node)
        if not node.inputs:
            self.noinput_nodes.discard(node)
        for c in node.inputs:
            if isinstance(c, graph.Constant) and (len(c.clients) <= 1):
                # This was the last node using this constant
                sig = self.const_sig[c]
                self.const_sig.discard(c)
                self.const_sig_inv.discard(sig)
                self.seen_constants.discard(id(c))

    def process_constant(self, fgraph, c):
        """
        Check if a constant can be merged, and queue that replacement.

        """
        if id(c) in self.seen_constants:
            return
        sig = c.merge_signature()
        other_c = self.const_sig_inv.get(sig, None)
        if other_c is not None:
            # multiple names will clobber each other..
            # we adopt convention to keep the last name
            if c.name:
                other_c.name = c.name
            self.scheduled.append([[(c, other_c, 'merge')]])
        else:
            # this is a new constant
            self.const_sig[c] = sig
            self.const_sig_inv[sig] = c
            self.seen_constants.add(id(c))

    def process_node(self, fgraph, node):
        """
        Check if a node can be merged, and queue that replacement.

        """
        if node in self.nodes_seen:
            return

        node_has_assert = False

        # These asserts ensure that the fgraph has set the clients field
        # properly.
        # The clients should at least contain `node` itself!
        if node.inputs:
            assert len(node.inputs[0].clients) > 0
            assert (node, 0) in node.inputs[0].clients

            merge_candidates = [c for (c, i) in node.inputs[0].clients
                                if c in self.nodes_seen]

            # Put all clients of Assert inputs (if exist) into merge_candidates
            # TODO: Deactivated for now as this cause cycle in the graph.
            for i in []:  # node.inputs:
                if i.owner and isinstance(i.owner.op,
                                          theano.tensor.opt.Assert):
                    node_has_assert = True
                    assert_clients = [c for (c, _) in i.owner.inputs[0].clients
                                      if c in self.nodes_seen]

                    for idx in range(len(assert_clients)):
                        client = assert_clients[idx]
                        if isinstance(i.owner.op, theano.tensor.opt.Assert):
                            for c in client.outputs[0].clients:
                                if c[0] in self.nodes_seen:
                                    assert_clients.append(c[0])

                    merge_candidates.extend(assert_clients)
        else:
            # If two nodes have no input, but perform the same operation,
            # they are not always constant-folded, so we want to merge them.
            # In that case, the candidates are all the nodes without inputs.
            merge_candidates = self.noinput_nodes

        replacement_candidates = []
        for candidate in merge_candidates:
            if candidate is node:
                continue
            if len(node.inputs) != len(candidate.inputs):
                continue

            cand_has_assert = False

            # Get input list of the candidate with assert removed
            cand_inputs_assert_removed = []
            for i in candidate.inputs:
                if i.owner and isinstance(i.owner.op,
                                          theano.tensor.opt.Assert):
                    cand_has_assert = True
                    cand_inputs_assert_removed.append(i.owner.inputs[0])
                else:
                    cand_inputs_assert_removed.append(i)

            # Get input list of the node with assert removed
            if node_has_assert:
                node_inputs_assert_removed = []
                for i in node.inputs:
                    if i.owner and isinstance(i.owner.op,
                                              theano.tensor.opt.Assert):
                        node_inputs_assert_removed.append(i.owner.inputs[0])
                    else:
                        node_inputs_assert_removed.append(i)
            else:
                node_inputs_assert_removed = node.inputs

            inputs_match = all(node_in is cand_in
                               for node_in, cand_in
                               in zip(node_inputs_assert_removed,
                                      cand_inputs_assert_removed))

            if inputs_match and node.op == candidate.op:
                if (node, candidate) in self.blacklist:
                    # They were already tried, and there was an error
                    continue

                # replace node with candidate
                if not (node_has_assert or cand_has_assert):
                    # Schedule transfer of clients from node to candidate
                    pairs = list(zip(node.outputs,
                                     candidate.outputs,
                                     ['merge'] * len(node.outputs)))

                # if the current node has assert input, it should not be
                # replaced with a candidate node which has no assert input
                elif node_has_assert and not cand_has_assert:
                    pairs = list(zip(candidate.outputs,
                                     node.outputs,
                                     ['merge'] * len(node.outputs)))
                else:
                    new_inputs = self.get_merged_assert_input(node, candidate)
                    new_node = node.op(*new_inputs)
                    pairs = list(zip(node.outputs,
                                     new_node.owner.outputs,
                                     ['new_node'] * len(node.outputs))) +\
                        list(zip(candidate.outputs,
                                 new_node.owner.outputs,
                                 ['new_node'] * len(node.outputs)))

                # transfer names
                for pair in pairs:
                    node_output, cand_output = pair[:2]
                    # clobber old name with new one
                    # it's arbitrary... one of the names has to go
                    if node_output.name:
                        cand_output.name = node_output.name

                replacement_candidates.append(pairs)

        if replacement_candidates:
            self.scheduled.append(replacement_candidates)
        else:
            self.nodes_seen.add(node)
            if not node.inputs:
                self.noinput_nodes.add(node)

    def get_merged_assert_input(self, node, candidate):
        new_inputs = []
        for node_i, cand_i in zip(node.inputs, candidate.inputs):
            # if node_i is assert
            if (node_i.owner and
                    isinstance(node_i.owner.op,
                               theano.tensor.opt.Assert)):
                # node_i is assert, cand_i is assert
                if (cand_i.owner and
                        isinstance(cand_i.owner.op,
                                   theano.tensor.opt.Assert)):
                    # Here two assert nodes are merged.
                    # Step 1. Merge conditions of both assert nodes.
                    # Step 2. Make the new assert node
                    node_cond = node_i.owner.inputs[1:]
                    cand_cond = cand_i.owner.inputs[1:]
                    new_cond = list(set(node_cond + cand_cond))
                    new_inputs.append(
                        theano.tensor.opt.assert_op(
                            node_i.owner.inputs[0],
                            *new_cond))

                # node_i is assert, cand_i is not assert
                else:
                    new_inputs.append(node_i)
            else:
                # if node_i is not an assert node, append cand_i
                new_inputs.append(cand_i)

        return new_inputs


class MergeOptimizer(Optimizer):
    """
    Merges parts of the graph that are identical and redundant.

    The basic principle is that if two Applies have ops that compare equal, and
    identical inputs, then they do not both need to be computed. The clients of
    one are transferred to the other and one of them is removed from the graph.
    This procedure is carried out in input->output order through the graph.

    The first step of merging is constant-merging, so that all clients of an
    int(1) for example, are transferred to a particular instance of int(1).

    """

    def add_requirements(self, fgraph):
        # Added by default
        # fgraph.attach_feature(toolbox.ReplaceValidate())
        if not hasattr(fgraph, 'merge_feature'):
            fgraph.attach_feature(MergeFeature())

    def apply(self, fgraph):
        # Constant and non-constant are now applied in the same phase.
        # I am not sure why, but it seems to be faster this way.
        sched = fgraph.merge_feature.scheduled
        nb_fail = 0
        t0 = time.time()
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callback_before = fgraph.execute_callbacks_time
            callbacks_before = fgraph.execute_callbacks_times.copy()

        nb_merged = 0
        nb_constant = 0
        while sched:
            pairs_list = sched.pop()
            success = True
            for pairs_ in pairs_list:
                # We must check again the equivalence, as the graph
                # can have changed. If so, doing the replacement can
                # introduce node that depend on itself.  Doing the
                # full check of such cycle everytimes is very time
                # consumming. I think this double check is faster then
                # doing the full cycle check. The full cycle check is
                # skipped by validate() if the graph don't contain
                # destroyers.
                var, candidate, merge_mode = pairs_[0]
                if merge_mode == "new_node" and hasattr(var, 'fgraph'):
                    pass
                elif (not hasattr(var, 'fgraph') or
                      not hasattr(candidate, 'fgraph')):
                    continue

                # Keep len(item) == 2 for item in pairs
                pairs = [pair[:2] for pair in pairs_]

                if var.owner and candidate.owner:
                    node = var.owner
                    candidate = candidate.owner

                    # Get input list of the candidate node with assert
                    # nodes removed
                    cand_inputs_assert_removed = []
                    for i in candidate.inputs:
                        if i.owner and isinstance(i.owner.op,
                                                  theano.tensor.opt.Assert):
                            cand_inputs_assert_removed.append(
                                i.owner.inputs[0])
                        else:
                            cand_inputs_assert_removed.append(i)

                    # Get input list of the node with assert nodes removed
                    node_inputs_assert_removed = []
                    for i in node.inputs:
                        if i.owner and isinstance(i.owner.op,
                                                  theano.tensor.opt.Assert):
                            node_inputs_assert_removed.append(
                                i.owner.inputs[0])
                        else:
                            node_inputs_assert_removed.append(i)

                    if merge_mode == "new_node":
                        inputs_match = True
                    else:
                        inputs_match = all(node_in is cand_in
                                           for node_in, cand_in in
                                           zip(node_inputs_assert_removed,
                                               cand_inputs_assert_removed))

                    # No need to compare the op again, as it don't change.
                    if not inputs_match:
                        continue

                    if hasattr(pairs[0][0].fgraph, 'destroy_handler'):
                        # If both nodes have clients that destroy
                        # them, we can't merge them.
                        clients = pairs[0][0].clients + pairs[0][1].clients
                        if sum([i in utils.flatten(c.op.destroy_map.values())
                                for c, i in clients
                                if c != 'output' and
                                hasattr(c.op, 'destroy_map')]) > 1:
                            continue

                try:
                    fgraph.replace_all_validate(pairs, 'MergeOptimizer')
                except InconsistencyError:
                    success = False
                    nb_fail += 1
                    fgraph.merge_feature.blacklist.append(
                        (pairs[0][0].owner, pairs[0][1].owner))
                if success:
                    nb_merged += len(pairs)
                    if isinstance(pairs[0][0], graph.Constant):
                        nb_constant += 1
                        # print pairs, pairs[0][0].type
                    break

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
        # clear blacklist
        fgraph.merge_feature.blacklist = []
        return (nb_fail, time.time() - t0, validate_time,
                callback_time, callbacks_time, nb_merged, nb_constant)

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def print_profile(stream, prof, level=0):
        (nb_fail, replace_time, validate_time,
         callback_time, callbacks_time, nb_merged, nb_constant) = prof

        blanc = ('    ' * level)
        print(blanc, "MergeOptimizer", file=stream)
        print(blanc, "  nb fail=%5d merged=%5d constant=%5d" % (
              nb_fail, nb_merged, nb_constant), file=stream)
        print(blanc, "  time replace=%2.2f validate=%2.2f callback=%2.2f" % (
              replace_time, validate_time, callback_time), file=stream)
        if callback_time > 1:
            print(blanc, "  callbacks_time", file=stream)
            for i in sorted(iteritems(callbacks_time), key=lambda a: a[1]):
                if i[1] > 0:
                    print(i)

    @staticmethod
    def merge_profile(prof1, prof2):
        def merge_none_number(v1, v2):
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            return v1 + v2
        nb_fail = prof1[0] + prof2[0]
        replace_time = prof1[1] + prof2[1]
        validate_time = merge_none_number(prof1[2], prof2[2])
        callback_time = merge_none_number(prof1[3], prof2[3])
        callbacks_time = merge_dict(prof1[4], prof2[4])
        nb_merged = prof1[5] + prof2[5]
        nb_constant = prof1[6] + prof2[6]
        return (nb_fail, replace_time, validate_time,
                callback_time, callbacks_time, nb_merged, nb_constant)


def is_same_graph_with_merge(var1, var2, givens=None):
    """
    Merge-based implementation of `theano.gof.graph.is_same_graph`.

    See help on `theano.gof.graph.is_same_graph` for additional documentation.

    """
    if givens is None:
        givens = {}
    # Copy variables since the MergeOptimizer will modify them.
    copied = copy.deepcopy([var1, var2, givens])
    vars = copied[0:2]
    givens = copied[2]
    # Create FunctionGraph.
    inputs = theano.gof.graph.inputs(vars)
    # The clone isn't needed as we did a deepcopy and we cloning will
    # break the mapping in givens.
    fgraph = theano.gof.fg.FunctionGraph(inputs, vars, clone=False)
    # Perform Variable substitution.
    for to_replace, replace_by in iteritems(givens):
        fgraph.replace(to_replace, replace_by)
    # Perform merge optimization.
    MergeOptimizer().optimize(fgraph)
    # When two variables perform the same computations, they will have the same
    # owner in the optimized graph.
    # We need to be careful with the special case where the owner is None,
    # which happens when the graph is made of a single Variable.
    # We also need to make sure we replace a Variable if it is present in
    # `givens`.
    vars_replaced = [givens.get(v, v) for v in vars]
    o1, o2 = [v.owner for v in vars_replaced]
    if o1 is None and o2 is None:
        # Comparing two single-Variable graphs: they are equal if they are
        # the same Variable.
        return vars_replaced[0] == vars_replaced[1]
    else:
        return o1 is o2


def pre_constant_merge(vars):
    """
    Merge constants in the subgraph used to compute nodes in `vars`.

    `vars` is a list of nodes, and we want to merge together nodes
    that are constant inputs used to compute nodes in that list.

    Notes
    -----
    This function will ignore nodes that are in an fgraph.
    It is used to pre-merge nodes generated inside an optimization,
    before it is inserted in the fgraph.
    It is useful if there are many such replacements to make,
    so that DebugMode will not check each of them.

    """
    seen_var = set()
    # signature -> variable (for constants)
    const_sig_inv = {}
    if isinstance(vars, graph.Variable):
        vars = [vars]

    def recursive_merge(var):
        if var in seen_var:
            return var
        if not hasattr(var, 'owner'):
            return var
        if var.owner and hasattr(var.owner, "fgraph"):
            return var
        seen_var.add(var)
        if isinstance(var, graph.Constant):
            sig = var.signature()
            try:
                if sig in const_sig_inv:
                    return const_sig_inv[sig]
                const_sig_inv[sig] = var
            except TypeError:  # unhashable type
                warnings.warn(
                    "We work around a problem, the following variable"
                    " signature isn't hashable. Please, report this to"
                    " theano-dev so that the better fix is done. %s" % var)
                # Some python object like slice aren't hashable. So
                # don't merge them here.
                pass
            return var
        if var.owner:
            for idx, inp in enumerate(var.owner.inputs):
                var.owner.inputs[idx] = recursive_merge(inp)
        return var

    return list(map(recursive_merge, vars))


########################
#   Local Optimizers   #
########################

class LocalOptimizer(object):
    """
    A class for node-based optimizations.

    Instances should implement the transform function,
    and be passed to configure a fgraph-based Optimizer instance.

    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def tracks(self):
        """
        Return the list of op classes that this opt applies to.

        Return None to apply to all nodes.

        """
        return None

    def transform(self, node):
        """
        Transform a subgraph whose output is `node`.

        Subclasses should implement this function so that it returns one of two
        kinds of things:

        - False to indicate that no optimization can be applied to this `node`;
          or
        - <list of variables> to use in place of `node`'s outputs in the
          greater graph.
        - dict(old variables -> new variables). A dictionary that map
          from old variables to new variables to replace.

        Parameters
        ----------
        node : an Apply instance

        """

        raise utils.MethodNotDefined("transform",
                                     type(self), self.__class__.__name__)

    def add_requirements(self, fgraph):
        """
        If this local optimization wants to add some requirements to the
        fgraph, this is the place to do it.

        """
        # Added by default
        # fgraph.attach_feature(toolbox.ReplaceValidate())
        pass

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s id=%i" % (
            (' ' * level), self.__class__.__name__, id(self)), file=stream)


class LocalMetaOptimizer(LocalOptimizer):
    """
    Base class for meta-optimizers that try a set of LocalOptimizers
    to replace a node and choose the one that executes the fastest.

    """

    def __init__(self, tracks=None, optimizers=()):
        self._tracks = tracks
        self.optimizers = list(optimizers)
        self.verbose = config.metaopt.verbose

    def register(self, optimizer):
        self.optimizers.append(optimizer)

    def tracks(self):
        return self._tracks

    def transform(self, node):
        # safety check: depending on registration, tracks may have been ignored
        if self._tracks is not None:
            if not isinstance(node.op, tuple(self._tracks)):
                return
        # first, we need to provide dummy values for all inputs
        # to the node that are not shared variables anyway
        givens = {}
        missing = set()
        for input in node.inputs:
            if isinstance(input, theano.compile.SharedVariable):
                pass
            elif hasattr(input.tag, 'test_value'):
                givens[input] = theano.shared(
                    input.type.filter(input.tag.test_value),
                    input.name,
                    broadcastable=input.broadcastable,
                    borrow=True)
            else:
                missing.add(input)
        if missing:
            givens.update(self.provide_inputs(node, missing))
            missing.difference_update(givens.keys())
        # ensure we have data for all input variables that need it
        if missing:
            if self.verbose:
                print(("%s cannot meta-optimize %s, "
                       "%d of %d input shapes unknown" %
                       (self.__class__.__name__, node, len(missing), node.nin)))
            return
        # now we can apply the different optimizations in turn,
        # compile the resulting subgraphs and time their execution
        if self.verbose:
            print(("%s meta-optimizing %s (%d choices):" %
                   (self.__class__.__name__, node, len(self.optimizers))))
        timings = []
        for opt in self.optimizers:
            outputs = opt.transform(node)
            if outputs:
                try:
                    fn = theano.function([], outputs, givens=givens,
                                         on_unused_input='ignore')
                    timing = min(self.time_call(fn) for _ in range(3))
                except Exception as e:
                    if self.verbose:
                        print("* %s: exception" % opt, e)
                    continue
                else:
                    if self.verbose:
                        print("* %s: %.5g sec" % (opt, timing))
                    timings.append((timing, outputs, opt))
            else:
                if self.verbose:
                    print("* %s: not applicable" % opt)
        # finally, we choose the fastest one
        if timings:
            timings.sort()
            if self.verbose:
                print("= %s" % timings[0][2])
            return timings[0][1]
        return

    def provide_inputs(self, node, inputs):
        """
        If implemented, returns a dictionary mapping all symbolic variables
        in ``inputs`` to SharedVariable instances of suitable dummy values.
        The ``node`` can be inspected to infer required input shapes.

        """
        raise NotImplementedError()

    def time_call(self, fn):
        start = time.time()
        fn()
        return time.time() - start


class FromFunctionLocalOptimizer(LocalOptimizer):
    """
    WRITEME

    """
    def __init__(self, fn, tracks=None, requirements=()):
        self.transform = fn
        self._tracks = tracks
        self.requirements = requirements

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def tracks(self):
        return self._tracks

    def __str__(self):
        return getattr(self, '__name__',
                       '<FromFunctionLocalOptimizer instance>')

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s id=%i" % (
            ' ' * level,
            str(self.transform),
            id(self)), file=stream)


def local_optimizer(tracks, inplace=False, requirements=()):
    def decorator(f):
        """
        WRITEME

        """
        if tracks is not None:
            if len(tracks) is 0:
                raise ValueError("Use None instead of an empty list to apply to all nodes.", f.__module__, f.__name__)
            for t in tracks:
                if not (isinstance(t, op.Op) or issubclass(t, op.PureOp)):
                    raise ValueError("Tracks are op classes or instances", f.__module__, f.__name__)
        req = requirements
        if inplace:
            dh_handler = dh.DestroyHandler
            req = tuple(requirements) + (
                lambda fgraph:
                fgraph.attach_feature(dh_handler()),)
        rval = FromFunctionLocalOptimizer(f, tracks, req)
        rval.__name__ = f.__name__
        return rval
    return decorator


class LocalOptGroup(LocalOptimizer):
    """
    WRITEME

    """

    def __init__(self, *optimizers):
        if len(optimizers) == 1 and isinstance(optimizers[0], list):
            # This happen when created by LocalGroupDB.
            optimizers = tuple(optimizers[0])
        self.opts = optimizers
        self.reentrant = any(getattr(opt, 'reentrant', True)
                             for opt in optimizers)
        self.retains_inputs = all(getattr(opt, 'retains_inputs', False)
                                  for opt in optimizers)

    def __str__(self):
        return getattr(self, '__name__',
                       ('LocalOptGroup(%s)' %
                        ','.join([str(o) for o in self.opts])))

    def tracks(self):
        t = []
        for l in self.opts:
            tt = l.tracks()
            if tt:
                t.extend(tt)
        return t

    def transform(self, node):
        for opt in self.opts:
            repl = opt.transform(node)
            if repl:
                return repl

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s id=%i" % (
            (' ' * level), self.__class__.__name__, id(self)), file=stream)
        if depth != 0:
            depth -= 1
            for lopt in self.opts:
                lopt.print_summary(stream, level=(level + 2), depth=depth)

    def add_requirements(self, fgraph):
        for opt in self.opts:
            opt.add_requirements(fgraph)


class OpSub(LocalOptimizer):
    """
    WRITEME

    Replaces the application of a certain op by the application of
    another op that takes the same inputs as what they are replacing.

    Parameters
    ----------
    op1, op2
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.

    Examples
    --------
    OpSub(add, sub) ==>
        add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))

    """

    # an OpSub does not apply to the nodes it produces
    reentrant = False
    # all the inputs of the original node are transferred to the outputs
    retains_inputs = True

    def __init__(self, op1, op2, transfer_tags=True):
        self.op1 = op1
        self.op2 = op2
        self.transfer_tags = transfer_tags

    def op_key(self):
        return self.op1

    def tracks(self):
        return [self.op1]

    def transform(self, node):
        if node.op != self.op1:
            return False
        repl = self.op2.make_node(*node.inputs)
        if self.transfer_tags:
            repl.tag = copy.copy(node.tag)
            for output, new_output in zip(node.outputs, repl.outputs):
                new_output.tag = copy.copy(output.tag)
        return repl.outputs

    def __str__(self):
        return "%s -> %s" % (self.op1, self.op2)


class OpRemove(LocalOptimizer):
    """
    WRITEME

    Removes all applications of an op by transferring each of its
    outputs to the corresponding input.

    """

    reentrant = False      # no nodes are added at all

    def __init__(self, op):
        self.op = op

    def op_key(self):
        return self.op

    def tracks(self):
        return [self.op]

    def transform(self, node):
        if node.op != self.op:
            return False
        return node.inputs

    def __str__(self):
        return "%s(x) -> x" % (self.op)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s(%s) id=%i" % (
            ' ' * level,
            self.__class__.__name__,
            str(self.op),
            id(self)), file=stream)


class PatternSub(LocalOptimizer):
    """
    WRITEME

    @todo update

    Replaces all occurrences of the input pattern by the output pattern:

    input_pattern ::= (op, <sub_pattern1>, <sub_pattern2>, ...)
    input_pattern ::= dict(pattern = <input_pattern>,
                            constraint = <constraint>)
    sub_pattern ::= input_pattern
    sub_pattern ::= string
    sub_pattern ::= a Constant instance
    sub_pattern ::= int
    sub_pattern ::= float
    constraint ::= lambda fgraph, expr: additional matching condition

    output_pattern ::= (op, <output_pattern1>, <output_pattern2>, ...)
    output_pattern ::= string
    output_pattern ::= int
    output_pattern ::= float

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant variable in the input pattern, there will be a
    match iff a constant variable with the same value and the same type
    is found in its place.

    You can add a constraint to the match by using the dict(...)  form
    described above with a 'constraint' key. The constraint must be a
    function that takes the fgraph and the current Variable that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    The constructor creates a PatternSub that replaces occurrences of
    in_pattern by occurrences of out_pattern.

    Parameters
    ----------
    in_pattern
        The input pattern that we want to replace.
    out_pattern
        The replacement pattern.
    allow_multiple_clients : bool
        If False, the pattern matching will fail if one of the subpatterns has
        more than one client.
    skip_identities_fn : TODO
    name
        Allows to override this optimizer name.
    pdb : bool
        If True, we invoke pdb when the first node in the pattern matches.
    tracks : optional
        The values that self.tracks() will return. Useful to speed up
        optimization sometimes.
    get_nodes : optional
        If you provide `tracks`, you must provide this parameter. It must be a
        function that takes the tracked node and returns a list of nodes on
        which we will try this optimizer.

    Notes
    -----
    `tracks` and `get_nodes` can be used to make this optimizer track a less
    frequent Op, so this will make this optimizer tried less frequently.

    Examples
    --------
    PatternSub((add, 'x', 'y'), (add, 'y', 'x'))
    PatternSub((multiply, 'x', 'x'), (square, 'x'))
    PatternSub((subtract, (add, 'x', 'y'), 'y'), 'x')
    PatternSub((power, 'x', Constant(double, 2.0)), (square, 'x'))
    PatternSub((boggle, {'pattern': 'x',
                         'constraint': lambda expr: expr.type == scrabble}),
               (scrabble, 'x'))
    """

    def __init__(self, in_pattern, out_pattern,
                 allow_multiple_clients=False,
                 skip_identities_fn=None, name=None, pdb=False,
                 tracks=(), get_nodes=None,
                 values_eq_approx=None):
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        self.values_eq_approx = values_eq_approx
        if isinstance(in_pattern, (list, tuple)):
            self.op = self.in_pattern[0]
        elif isinstance(in_pattern, dict):
            self.op = self.in_pattern['pattern'][0]
        else:
            raise TypeError("The pattern to search for must start with "
                            "a specific Op instance.")
        self.__doc__ = (self.__class__.__doc__ +
                        "\n\nThis instance does: " +
                        str(self) + "\n")
        self.allow_multiple_clients = allow_multiple_clients
        self.skip_identities_fn = skip_identities_fn
        if name:
            self.__name__ = name
        self.pdb = pdb
        self._tracks = tracks
        self.get_nodes = get_nodes
        if tracks != ():
            assert get_nodes

    def op_key(self):
        return self.op

    def tracks(self):
        if self._tracks != ():
            return self._tracks
        return [self.op]

    def transform(self, node, get_nodes=True):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.

        """
        if get_nodes and self.get_nodes is not None:
            for real_node in self.get_nodes(node):
                if real_node == "output":
                    continue
                ret = self.transform(real_node, get_nodes=False)
                if ret is not False and ret is not None:
                    assert len(real_node.outputs) == len(ret)
                    if self.values_eq_approx:
                        ret.tag.values_eq_approx = self.values_eq_approx
                    return dict(izip(real_node.outputs, ret))

        if node.op != self.op:
            return False
        # TODO: if we remove pdb, do this speed things up?

        def match(pattern, expr, u, allow_multiple_clients=False, pdb=False):
            # TODO move outside match
            def retry_with_equiv():
                if not self.skip_identities_fn:
                    return False
                expr_equiv = self.skip_identities_fn(expr)
                if expr_equiv is None:
                    return False
                # TODO: Not sure how to handle multiple_clients flag
                # print 'retrying match', pattern, expr_equiv
                return match(pattern, expr_equiv, u,
                             allow_multiple_clients=allow_multiple_clients)

            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if (not (expr.owner.op == pattern[0]) or
                        (not allow_multiple_clients and len(expr.clients) > 1)):
                    return retry_with_equiv()
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return retry_with_equiv()
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u, self.allow_multiple_clients)
                    if not u:
                        return False
            elif isinstance(pattern, dict):
                try:
                    real_pattern = pattern['pattern']
                except KeyError:
                    raise KeyError(
                        "Malformed pattern: %s (expected key 'pattern')"
                        % pattern)
                constraint = pattern.get('constraint', lambda expr: True)
                if constraint(expr):
                    return match(real_pattern, expr, u,
                                 pattern.get('allow_multiple_clients',
                                             allow_multiple_clients))
                else:
                    return retry_with_equiv()
            elif isinstance(pattern, string_types):
                v = unify.Var(pattern)
                if u[v] is not v and u[v] is not expr:
                    return retry_with_equiv()
                else:
                    u = u.merge(expr, v)
            elif (isinstance(pattern, (integer_types, float)) and
                    isinstance(expr, graph.Constant)):
                if numpy.all(
                        theano.tensor.constant(pattern).value == expr.value):
                    return u
                else:
                    return retry_with_equiv()
            elif (isinstance(pattern, graph.Constant) and
                    isinstance(expr, graph.Constant) and
                    pattern.equals(expr)):
                return u
            else:
                return retry_with_equiv()
            if pdb:
                import pdb
                pdb.set_trace()
            return u

        u = match(self.in_pattern, node.out, unify.Unification(), True,
                  self.pdb)
        if u:
            def build(pattern, u):
                if isinstance(pattern, (list, tuple)):
                    args = [build(p, u) for p in pattern[1:]]
                    return pattern[0](*args)
                elif isinstance(pattern, string_types):
                    return u[unify.Var(pattern)]
                elif isinstance(pattern, (integer_types, float)):
                    return pattern
                else:
                    return pattern.clone()
            p = self.out_pattern
            ret = build(p, u)
            if self.values_eq_approx:
                ret.tag.values_eq_approx = self.values_eq_approx
            return [ret]
        else:
            return False

    def __str__(self):
        if getattr(self, '__name__', None):
            return self.__name__

        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "%s(%s)" % (
                    str(pattern[0]),
                    ", ".join([pattern_to_str(p) for p in pattern[1:]]))
            elif isinstance(pattern, dict):
                return "%s subject to %s" % (
                    pattern_to_str(pattern['pattern']),
                    str(pattern.get('constraint', 'no conditions')))
            else:
                return str(pattern)
        return "%s -> %s" % (
            pattern_to_str(self.in_pattern),
            pattern_to_str(self.out_pattern))

    def __repr__(self):
        return str(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, '__name__', getattr(self, 'name', None))
        print("%s%s %s(%s, %s) id=%i" % (
            ' ' * level,
            self.__class__.__name__,
            name,
            str(self.in_pattern),
            str(self.out_pattern),
            id(self)), file=stream)


##################
#   Navigators   #
##################

# Use the following classes to apply LocalOptimizers

class Updater:
    def __init__(self, importer, pruner, chin):
        self.importer = importer
        self.pruner = pruner
        self.chin = chin

    def on_import(self, fgraph, node, reason):
        if self.importer:
            self.importer(node)

    def on_prune(self, fgraph, node, reason):
        if self.pruner:
            self.pruner(node)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if self.chin:
            self.chin(node, i, r, new_r, reason)

    def on_detach(self, fgraph):
        # To allow pickling this object
        self.importer = None
        self.pruner = None
        self.chin = None


class NavigatorOptimizer(Optimizer):
    """
    Abstract class.

    Parameters
    ----------
    local_opt
        A LocalOptimizer to apply over a FunctionGraph (or None is Ok too).
    ignore_newtrees
        - True: new subgraphs returned by an optimization is not a
          candidate for optimization.
        - False: new subgraphs returned by an optimization is a candidate
          for optimization.
        - 'auto': let the local_opt set this parameter via its 'reentrant'
          attribute.
    failure_callback
            A function that takes (exception, navigator, [(old, new),
            (old,new),...]) and we call it if there's an exception.

            If the trouble is from local_opt.transform(), the new variables
            will be 'None'.

            If the trouble is from validation (the new types don't match for
            example) then the new variables will be the ones created by
            transform().

            If this parameter is None, then exceptions are not caught here
            (raised normally).

    """
    @staticmethod
    def warn(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer: print traceback.

        """
        if config.on_opt_error != 'ignore':
            _logger.error("Optimization failure due to: %s" % str(local_opt))
            _logger.error("node: %s" % str(node))
            _logger.error("TRACEBACK:")
            _logger.error(traceback.format_exc())
        if config.on_opt_error == 'pdb':
            pdb.post_mortem(sys.exc_info()[2])
        elif isinstance(exc, AssertionError) or config.on_opt_error == 'raise':
            # We always crash on AssertionError because something may be
            # seriously wrong if such an exception is raised.
            raise exc

    @staticmethod
    def warn_inplace(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer.

        Ignore InconsistencyErrors, print traceback.

        If error during replacement repl_pairs is set. Otherwise None.

        """
        if isinstance(exc, InconsistencyError):
            return
        return NavigatorOptimizer.warn(exc, nav, repl_pairs, local_opt, node)

    @staticmethod
    def warn_ignore(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer: ignore all errors.

        """
        pass

    def __init__(self, local_opt, ignore_newtrees='auto',
                 failure_callback=None):
        self.local_opt = local_opt
        if ignore_newtrees == 'auto':
            self.ignore_newtrees = not getattr(local_opt, 'reentrant', True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def attach_updater(self, fgraph, importer, pruner, chin=None):
        """
        Install some FunctionGraph listeners to help the navigator deal with
        the ignore_trees-related functionality.

        Parameters
        ----------
        importer
            Function that will be called whenever optimizations add stuff
            to the graph.
        pruner
            Function to be called when optimizations remove stuff
            from the graph.
        chin
            "on change input" called whenever a node's inputs change.

        Returns
        -------
        object
            The FunctionGraph plugin that handles the three tasks.
            Keep this around so that you can detach later!

        """
        if self.ignore_newtrees:
            importer = None

        if importer is None and pruner is None:
            return None

        u = Updater(importer, pruner, chin)
        fgraph.attach_feature(u)
        return u

    def detach_updater(self, fgraph, u):
        """
        Undo the work of attach_updater.

        Parameters
        ----------
        u
            A return-value of attach_updater.

        Returns
        -------
        None

        """
        if u is not None:
            fgraph.remove_feature(u)

    def process_node(self, fgraph, node, lopt=None):
        """
        This function will use `lopt` to `transform` the `node`. The
        `transform` method will return either False or a list of Variables
        that are intended to replace `node.outputs`.

        If the fgraph accepts the replacement, then the optimization is
        successful, and this function returns True.

        If there are no replacement candidates or the fgraph rejects the
        replacements, this function returns False.

        Parameters
        ----------
        fgraph
            A FunctionGraph.
        node
            An Apply instance in `fgraph`
        lopt
            A LocalOptimizer instance that may have a better idea for
            how to compute node's outputs.

        Returns
        -------
        bool
            True iff the `node`'s outputs were replaced in the `fgraph`.

        """
        lopt = lopt or self.local_opt
        try:
            replacements = lopt.transform(node)
        except Exception as e:
            if self.failure_callback is not None:
                self.failure_callback(e, self,
                                      [(x, None) for x in node.outputs],
                                      lopt, node)
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False
        old_vars = node.outputs
        if isinstance(replacements, dict):
            old_vars = list(replacements.keys())
            replacements = list(replacements.values())
        elif not isinstance(replacements, (tuple, list)):
            raise TypeError('Optimizer %s gave wrong type of replacement. '
                            'Expected list or tuple. Got %s' % (
                                lopt, replacements))
        if len(old_vars) != len(replacements):
            raise ValueError('Optimizer %s gave wrong number of replacements'
                             % lopt)
        # None in the replacement mean that this variable isn't used
        # and we want to remove it
        for r, rnew in zip(old_vars, replacements):
            if rnew is None and len(r.clients) > 0:
                raise ValueError("A local optimizer tried to remove a Variable that is used")
        # If an output would be replaced by itself, no need to perform
        # the replacement
        repl_pairs = [(r, rnew) for r, rnew in zip(old_vars, replacements)
                      if rnew is not r and rnew is not None]

        if len(repl_pairs) == 0:
            return False
        try:
            fgraph.replace_all_validate(repl_pairs, reason=lopt)
            return True
        except Exception as e:
            # This means the replacements were rejected by the fgraph.
            #
            # This is not supposed to happen.  The default failure_callback
            # will print a traceback as a warning.
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs, lopt, node)
                return False
            else:
                raise

    def add_requirements(self, fgraph):
        super(NavigatorOptimizer, self).add_requirements(fgraph)
        # Added by default
        # fgraph.attach_feature(toolbox.ReplaceValidate())
        if self.local_opt:
            self.local_opt.add_requirements(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print("%s%s (%i)" % (
            (' ' * level), self.__class__.__name__, id(self)), file=stream)
        if depth != 0:
            self.local_opt.print_summary(stream, level=(level + 2),
                                         depth=(depth - 1))


class TopoOptimizer(NavigatorOptimizer):
    """
    WRITEME

    """

    def __init__(self, local_opt, order='in_to_out', ignore_newtrees=False,
                 failure_callback=None):
        if order not in ['out_to_in', 'in_to_out']:
            raise ValueError("order must be 'out_to_in' or 'in_to_out'")
        self.order = order
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees,
                                    failure_callback)

    def apply(self, fgraph, start_from=None):
        if start_from is None:
            start_from = fgraph.outputs
        callback_before = fgraph.execute_callbacks_time
        nb_nodes_start = len(fgraph.apply_nodes)
        t0 = time.time()
        q = deque(graph.io_toposort(fgraph.inputs, start_from))
        io_t = time.time() - t0

        def importer(node):
            if node is not current_node:
                q.append(node)

        def pruner(node):
            if node is not current_node:
                try:
                    q.remove(node)
                except ValueError:
                    pass

        u = self.attach_updater(fgraph, importer, pruner)
        nb = 0
        try:
            t0 = time.time()
            while q:
                if self.order == 'out_to_in':
                    node = q.pop()
                else:
                    node = q.popleft()
                current_node = node
                nb += self.process_node(fgraph, node)
            loop_t = time.time() - t0
        except Exception:
            self.detach_updater(fgraph, u)
            raise
        self.detach_updater(fgraph, u)

        callback_time = fgraph.execute_callbacks_time - callback_before
        nb_nodes_end = len(fgraph.apply_nodes)
        return (self, nb, nb_nodes_start, nb_nodes_end,
                io_t, loop_t, callback_time)

    @staticmethod
    def print_profile(stream, prof, level=0):
        (opt, nb, nb_nodes_start, nb_nodes_end,
         io_t, loop_t, callback_time) = prof

        blanc = ('    ' * level)
        print(blanc, "TopoOptimizer ",
              getattr(opt, "name", getattr(opt, "__name__", "")), file=stream)

        print(blanc, "  nb_node (start, end, changed)", (
            nb_nodes_start, nb_nodes_end, nb), file=stream)
        print(blanc, "  init io_toposort", io_t, file=stream)
        print(blanc, "  loop time", loop_t, file=stream)
        print(blanc, "  callback_time", callback_time, file=stream)

    def __str__(self):
        return getattr(self, '__name__',
                       '<TopoOptimizer instance>')


class OpKeyOptimizer(NavigatorOptimizer):
    """
    WRITEME

    """

    def __init__(self, local_opt, ignore_newtrees=False,
                 failure_callback=None):
        if not hasattr(local_opt, 'op_key'):
            raise TypeError("LocalOptimizer for OpKeyOptimizer must have "
                            "an 'op_key' method.")
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees,
                                    failure_callback)

    def apply(self, fgraph):
        op = self.local_opt.op_key()
        if isinstance(op, (list, tuple)):
            q = reduce(list.__iadd__, map(fgraph.get_nodes, op))
        else:
            q = list(fgraph.get_nodes(op))

        def importer(node):
            if node is not current_node:
                if node.op == op:
                    q.append(node)

        def pruner(node):
            if node is not current_node and node.op == op:
                try:
                    q.remove(node)
                except ValueError:
                    pass
        u = self.attach_updater(fgraph, importer, pruner)
        try:
            while q:
                node = q.pop()
                current_node = node
                self.process_node(fgraph, node)
        except Exception:
            self.detach_updater(fgraph, u)
            raise
        self.detach_updater(fgraph, u)

    def add_requirements(self, fgraph):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate(Added by default)

        """
        super(OpKeyOptimizer, self).add_requirements(fgraph)
        fgraph.attach_feature(toolbox.NodeFinder())


class ChangeTracker:
    def __init__(self):
        self.changed = False
        self.nb_imported = 0

    def on_import(self, fgraph, node, reason):
        self.nb_imported += 1
        self.changed = True

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        self.changed = True

    def reset(self):
        self.changed = False

    def on_attach(self, fgraph):
        fgraph.change_tracker = self


def merge_dict(d1, d2):
    """
    merge 2 dicts by adding the values.
    """
    d = d1.copy()
    for k, v in iteritems(d2):
        if k in d:
            d[k] += v
        else:
            d[k] = v
    return d


class EquilibriumOptimizer(NavigatorOptimizer):
    """
    Apply optimizations until equilibrium point.

    Parameters
    ----------
    optimizers : list or set
        Local or global optimizations to apply until equilibrium.
        The global optimizer will be run at the start of each iteration before
        the local optimizer.
    max_use_ratio : int or float
        Each optimizer can be applied at most (size of graph * this number)
        times.
    ignore_newtrees
        See EquilibriumDB ignore_newtrees parameter definition.
    final_optimizers
        Global optimizers that will be run after each iteration.
    cleanup_optimizers
        Global optimizers that apply a list of pre determined optimization.
        They must not traverse the graph as they are called very frequently.
        The MergeOptimizer is one example of optimization that respect this.
        They are applied after all global optimizer, then when one local optimizer is applied, then after all final optimizer.

    """

    def __init__(self,
                 optimizers,
                 failure_callback=None,
                 ignore_newtrees=True,
                 max_use_ratio=None,
                 final_optimizers=None,
                 cleanup_optimizers=None):
        super(EquilibriumOptimizer, self).__init__(
            None,
            ignore_newtrees=ignore_newtrees,
            failure_callback=failure_callback)
        self.local_optimizers_map = OrderedDict()
        self.local_optimizers_all = []
        self.global_optimizers = []
        self.final_optimizers = []
        self.cleanup_optimizers = []

        for opt in optimizers:
            if isinstance(opt, LocalOptimizer):
                if opt.tracks() is None:
                    self.local_optimizers_all.append(opt)
                else:
                    for c in opt.tracks():
                        self.local_optimizers_map.setdefault(c, []).append(opt)
            else:
                self.global_optimizers.append(opt)
        if final_optimizers:
            self.final_optimizers = final_optimizers
        if cleanup_optimizers:
            self.cleanup_optimizers = cleanup_optimizers
        self.max_use_ratio = max_use_ratio
        assert self.max_use_ratio is not None, (
            'max_use_ratio has to be a number')

    def get_local_optimizers(self):
        for opt in self.local_optimizers_all:
            yield opt
        # if repeat is not a problem we can drop the set
        s = set()
        for lopt in itervalues(self.local_optimizers_map):
            for opt in lopt:
                if opt not in s:
                    yield opt
                    s.add(opt)

    def add_requirements(self, fgraph):
        super(EquilibriumOptimizer, self).add_requirements(fgraph)
        for opt in self.get_local_optimizers():
            opt.add_requirements(fgraph)
        for opt in self.global_optimizers:
            opt.add_requirements(fgraph)
        for opt in self.final_optimizers:
            opt.add_requirements(fgraph)
        for opt in self.cleanup_optimizers:
            opt.add_requirements(fgraph)

    def apply(self, fgraph, start_from=None):
        change_tracker = ChangeTracker()
        fgraph.attach_feature(change_tracker)
        if start_from is None:
            start_from = fgraph.outputs
        else:
            for node in start_from:
                assert node in fgraph.outputs

        changed = True
        max_use_abort = False
        opt_name = None
        global_process_count = {}
        start_nb_nodes = len(fgraph.apply_nodes)
        max_nb_nodes = len(fgraph.apply_nodes)
        max_use = max_nb_nodes * self.max_use_ratio

        loop_timing = []
        loop_process_count = []
        global_opt_timing = []
        time_opts = {}
        io_toposort_timing = []
        nb_nodes = []
        node_created = {}
        global_sub_profs = []
        final_sub_profs = []
        cleanup_sub_profs = []
        for opt in (self.global_optimizers +
                    list(self.get_local_optimizers()) +
                    self.final_optimizers +
                    self.cleanup_optimizers):
            global_process_count.setdefault(opt, 0)
            time_opts.setdefault(opt, 0)
            node_created.setdefault(opt, 0)

        def apply_cleanup(profs_dict):
            changed = False
            for copt in self.cleanup_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = copt.apply(fgraph)
                time_opts[copt] += time.time() - t_opt
                profs_dict[copt].append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(copt, 0)
                    process_count[copt] += 1
                    global_process_count[copt] += 1
                    changed = True
                    node_created[copt] += change_tracker.nb_imported - nb
            return changed

        while changed and not max_use_abort:
            process_count = {}
            t0 = time.time()
            changed = False
            iter_cleanup_sub_profs = {}
            for copt in self.cleanup_optimizers:
                iter_cleanup_sub_profs[copt] = []

            # apply global optimizers
            sub_profs = []
            for gopt in self.global_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = gopt.apply(fgraph)
                time_opts[gopt] += time.time() - t_opt
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(gopt, 0)
                    process_count[gopt] += 1
                    global_process_count[gopt] += 1
                    changed = True
                    node_created[gopt] += change_tracker.nb_imported - nb
                    if global_process_count[gopt] > max_use:
                        max_use_abort = True
                        opt_name = (getattr(gopt, "name", None) or
                                    getattr(gopt, "__name__", ""))
            global_sub_profs.append(sub_profs)

            global_opt_timing.append(float(time.time() - t0))

            # apply clean up as global opt can have done changes that
            # request that
            changed |= apply_cleanup(iter_cleanup_sub_profs)

            # apply local optimizer
            topo_t0 = time.time()
            q = deque(graph.io_toposort(fgraph.inputs, start_from))
            io_toposort_timing.append(time.time() - topo_t0)

            nb_nodes.append(len(q))
            max_nb_nodes = max(max_nb_nodes, len(q))
            max_use = max_nb_nodes * self.max_use_ratio

            def importer(node):
                if node is not current_node:
                    q.append(node)

            def pruner(node):
                if node is not current_node:
                    try:
                        q.remove(node)
                    except ValueError:
                        pass

            u = self.attach_updater(fgraph, importer, pruner)
            try:
                while q:
                    node = q.pop()
                    current_node = node

                    for lopt in (self.local_optimizers_all +
                                 self.local_optimizers_map.get(type(node.op), []) +
                                 self.local_optimizers_map.get(node.op, [])):
                        nb = change_tracker.nb_imported
                        t_opt = time.time()
                        lopt_change = self.process_node(fgraph, node, lopt)
                        time_opts[lopt] += time.time() - t_opt
                        if not lopt_change:
                            continue
                        process_count.setdefault(lopt, 0)
                        process_count[lopt] += 1
                        global_process_count[lopt] += 1
                        changed = True
                        node_created[lopt] += change_tracker.nb_imported - nb
                        changed |= apply_cleanup(iter_cleanup_sub_profs)
                        if global_process_count[lopt] > max_use:
                            max_use_abort = True
                            opt_name = (getattr(lopt, "name", None) or
                                        getattr(lopt, "__name__", ""))
                        if node not in fgraph.apply_nodes:
                            # go to next node
                            break
            finally:
                self.detach_updater(fgraph, u)

            # Apply final optimizers
            sub_profs = []
            t_before_final_opt = time.time()
            for gopt in self.final_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = gopt.apply(fgraph)
                time_opts[gopt] += time.time() - t_opt
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(gopt, 0)
                    process_count[gopt] += 1
                    global_process_count[gopt] += 1
                    changed = True
                    node_created[gopt] += change_tracker.nb_imported - nb
                    if global_process_count[gopt] > max_use:
                        max_use_abort = True
                        opt_name = (getattr(gopt, "name", None) or
                                    getattr(gopt, "__name__", ""))
            final_sub_profs.append(sub_profs)

            global_opt_timing[-1] += time.time() - t_before_final_opt
            # apply clean up as final opt can have done changes that
            # request that
            changed |= apply_cleanup(iter_cleanup_sub_profs)
            # merge clean up profiles during that iteration.
            c_sub_profs = []
            for copt, sub_profs in iteritems(iter_cleanup_sub_profs):
                sub_prof = sub_profs[0]
                for s_p in sub_profs[1:]:
                    sub_prof = copt.merge_profile(sub_prof, s_p)
                c_sub_profs.append(sub_prof)
            cleanup_sub_profs.append(c_sub_profs)

            loop_process_count.append(process_count)
            loop_timing.append(float(time.time() - t0))

        end_nb_nodes = len(fgraph.apply_nodes)

        if max_use_abort:
            _logger.error("EquilibriumOptimizer max'ed out by '%s'" % opt_name +
                          ". You can safely raise the current threshold of " +
                          "%f with the theano flag 'optdb.max_use_ratio'." %
                          config.optdb.max_use_ratio)
        fgraph.remove_feature(change_tracker)
        return (self, loop_timing, loop_process_count,
                (start_nb_nodes, end_nb_nodes, max_nb_nodes),
                global_opt_timing, nb_nodes, time_opts, io_toposort_timing,
                node_created, global_sub_profs, final_sub_profs, cleanup_sub_profs)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print("%s%s %s id=%i" % (
            (' ' * level), self.__class__.__name__, name, id(self)), file=stream)
        if depth != 0:
            for lopt in self.get_local_optimizers():
                lopt.print_summary(stream, level=(level + 2),
                                   depth=(depth - 1))

    @staticmethod
    def print_profile(stream, prof, level=0):
        (opt, loop_timing, loop_process_count,
         (start_nb_nodes, end_nb_nodes, max_nb_nodes),
         global_opt_timing, nb_nodes, time_opts, io_toposort_timing,
         node_created, global_sub_profs, final_sub_profs,
         cleanup_sub_profs) = prof

        blanc = ('    ' * level)
        print(blanc, "EquilibriumOptimizer", end=' ', file=stream)
        print(blanc, getattr(opt, "name",
                             getattr(opt, "__name__", "")), file=stream)
        print(blanc, "  time %.3fs for %d passes" % (
            sum(loop_timing), len(loop_timing)), file=stream)
        print(blanc, "  nb nodes (start, end,  max) %d %d %d" % (
            start_nb_nodes, end_nb_nodes, max_nb_nodes), file=stream)
        print(blanc, "  time io_toposort %.3fs" % sum(
            io_toposort_timing), file=stream)
        s = sum([time_opts[o] for o in opt.get_local_optimizers()])
        print(blanc, "  time in local optimizers %.3fs" % s, file=stream)
        s = sum([time_opts[o] for o in opt.global_optimizers])
        print(blanc, "  time in global optimizers %.3fs" % s, file=stream)
        s = sum([time_opts[o] for o in opt.final_optimizers])
        print(blanc, "  time in final optimizers %.3fs" % s, file=stream)
        s = sum([time_opts[o] for o in opt.cleanup_optimizers])
        print(blanc, "  time in cleanup optimizers %.3fs" % s, file=stream)
        for i in range(len(loop_timing)):
            lopt = ""
            if loop_process_count[i]:
                d = list(reversed(sorted(iteritems(loop_process_count[i]),
                                         key=lambda a: a[1])))
                lopt = " ".join([str((str(k), v)) for k, v
                                 in d[:5]])
                if len(d) > 5:
                    lopt += " ..."
            print(blanc, ('  %2d - %.3fs %d (%.3fs in global opts, '
                          '%.3fs io_toposort) - %d nodes - %s' % (
                              i, loop_timing[i],
                              sum(loop_process_count[i].values()),
                              global_opt_timing[i],
                              io_toposort_timing[i], nb_nodes[i],
                              lopt)), file=stream)

        count_opt = []
        not_used = []
        not_used_time = 0
        process_count = {}
        for o in (opt.global_optimizers +
                  list(opt.get_local_optimizers()) +
                  list(opt.final_optimizers) +
                  list(opt.cleanup_optimizers)):
            process_count.setdefault(o, 0)
        for count in loop_process_count:
            for o, v in iteritems(count):
                process_count[o] += v
        for o, count in iteritems(process_count):
            if count > 0:
                count_opt.append((time_opts[o], count,
                                  node_created[o], o))
            else:
                not_used.append((time_opts[o], o))
                not_used_time += time_opts[o]

        if count_opt:
            print(blanc,
                  '  times - times applied - nb node created - name:',
                  file=stream)
            count_opt.sort()
            for (t, count, n_created, o) in count_opt[::-1]:
                print(blanc, '  %.3fs - %d - %d - %s' % (
                    t, count, n_created, o), file=stream)
            print(blanc, '  %.3fs - in %d optimization that where not used (display only those with a runtime > 0)' % (
                not_used_time, len(not_used)), file=stream)
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip opt that have 0 times, they probably wasn't even tried.
                    print(blanc + "  ", '  %.3fs - %s' % (t, o), file=stream)
            print(file=stream)
        gf_opts = [o for o in (opt.global_optimizers +
                               list(opt.final_optimizers) +
                               list(opt.cleanup_optimizers))
                   if o.print_profile.__code__ is not
                   Optimizer.print_profile.__code__]
        if not gf_opts:
            return
        print(blanc, "Global, final and clean up optimizers", file=stream)
        for i in range(len(loop_timing)):
            print(blanc, "Iter %d" % i, file=stream)
            for o, prof in zip(opt.global_optimizers, global_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(opt.final_optimizers, final_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(opt.cleanup_optimizers, cleanup_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)

    @staticmethod
    def merge_profile(prof1, prof2):
        # (opt, loop_timing, loop_process_count, max_nb_nodes,
        # global_opt_timing, nb_nodes, time_opts, io_toposort_timing) = prof1
        local_optimizers = OrderedSet(prof1[0].get_local_optimizers()).union(
            prof2[0].get_local_optimizers())
        global_optimizers = OrderedSet(prof1[0].global_optimizers).union(
            prof2[0].global_optimizers)
        if len(prof1[0].final_optimizers) > 0 or len(prof2[0].final_optimizers) > 0:
            final_optimizers = OrderedSet(prof1[0].final_optimizers).union(
                prof2[0].final_optimizers)
        else:
            final_optimizers = None
        if len(prof1[0].cleanup_optimizers) > 0 or len(prof2[0].cleanup_optimizers) > 0:
            cleanup_optimizers = OrderedSet(prof1[0].cleanup_optimizers).union(
                prof2[0].cleanup_optimizers)
        else:
            cleanup_optimizers = None
        new_opt = EquilibriumOptimizer(
            local_optimizers.union(global_optimizers),
            max_use_ratio=1,
            final_optimizers=final_optimizers,
            cleanup_optimizers=cleanup_optimizers)

        def merge_list(l1, l2):
            l = copy.copy(l1)
            for idx, nb in enumerate(l2):
                if idx < len(l):
                    l[idx] += nb
                else:
                    l.append(nb)
            return l

        loop_timing = merge_list(prof1[1], prof2[1])

        loop_process_count = list(prof1[2])
        for i in range(min(len(loop_process_count), len(prof2[2]))):
            process_count = loop_process_count[i]
            for process, count in iteritems(prof2[2][i]):
                if process in process_count:
                    process_count[process] += count
                else:
                    process_count[process] = count
        loop_process_count.extend(prof2[2][len(loop_process_count):])

        max_nb_nodes = max(prof1[3], prof2[3])

        global_opt_timing = merge_list(prof1[4], prof2[4])

        nb_nodes = merge_list(prof1[5], prof2[5])

        time_opts = merge_dict(prof1[6], prof2[6])
        io_toposort_timing = merge_list(prof1[7], prof2[7])

        assert (len(loop_timing) == len(global_opt_timing) ==
                len(io_toposort_timing) == len(nb_nodes))
        assert len(loop_timing) == max(len(prof1[1]), len(prof2[1]))

        node_created = merge_dict(prof1[8], prof2[8])
        global_sub_profs = merge_list(prof1[9], prof2[9])
        final_sub_profs = merge_list(prof1[10], prof2[10])
        cleanup_sub_profs = merge_list(prof1[10], prof2[10])
        return (new_opt,
                loop_timing,
                loop_process_count,
                max_nb_nodes,
                global_opt_timing,
                nb_nodes,
                time_opts,
                io_toposort_timing,
                node_created,
                global_sub_profs,
                final_sub_profs,
                cleanup_sub_profs)

#################
#   Utilities   #
#################


def _check_chain(r, chain):
    """
    WRITEME

    """
    chain = list(reversed(chain))
    while chain:
        elem = chain.pop()
        if elem is None:
            if r.owner is not None:
                return False
        elif r.owner is None:
            return False
        elif isinstance(elem, op.Op):
            if not r.owner.op == elem:
                return False
        else:
            try:
                if (issubclass(elem, op.Op) and
                        not isinstance(r.owner.op, elem)):
                    return False
            except TypeError:
                return False
        if chain:
            r = r.owner.inputs[chain.pop()]
    # print 'check_chain', _check_chain.n_calls
    # _check_chain.n_calls += 1

    # The return value will be used as a Boolean, but some Variables cannot
    # be used as Booleans (the results of comparisons, for instance)
    return (r is not None)
# _check_chain.n_calls = 0


def check_chain(r, *chain):
    """
    WRITEME

    """
    if isinstance(r, graph.Apply):
        r = r.outputs[0]
    return _check_chain(r, reduce(list.__iadd__, ([x, 0] for x in chain)))


def pre_greedy_local_optimizer(list_optimizations, out):
    """
    This function traverses the computation graph described by all
    ``node`` in the graph before the variable out but that are not in the
    fgraph. It applies each of the local_optimizations on the traversed graph.

    Its main use is to apply locally constant folding when generating
    the graph of the indices of a subtensor.

    We should not apply optimizations on node that are in fgraph.
    So we don't optimize node that have an attribute fgraph.

    Notes
    -----
    This doesn't do an equilibrium... So if there is optimization
    like local_upcast_elemwise_constant_inputs in the list, that
    adds additional node to the inputs of the node, it can
    be needed to call this function multiple times.

    """
    def local_recursive_function(list_opt, out, optimized_vars, depth):
        if not getattr(out, 'owner', None):
            return [out], optimized_vars
        node = out.owner

        if hasattr(node, 'fgraph'):
            return node.outputs, optimized_vars
        for idx, inp in enumerate(node.inputs):
            if inp in optimized_vars:
                nw_in = optimized_vars[inp]
            else:
                if inp.owner:
                    outs, optimized_vars = local_recursive_function(
                        list_opt,
                        inp,
                        optimized_vars,
                        depth + 1)
                    for k, v in zip(inp.owner.outputs, outs):
                        optimized_vars[k] = v
                    nw_in = outs[inp.owner.outputs.index(inp)]

                else:
                    nw_in = inp
                    optimized_vars[inp] = inp
            node.inputs[idx] = nw_in

        results = node.outputs
        for opt in list_opt:
            ret = opt.transform(node)
            if ret is not False and ret is not None:
                assert len(ret) == len(node.outputs)
                for k, v in zip(node.outputs, ret):
                    optimized_vars[k] = v
                results = ret
                if ret[0].owner:
                    node = out.owner
                else:
                    break
        return results, optimized_vars
    if out.owner:
        out_index = out.owner.outputs.index(out)
    else:
        out_index = 0
    final_outs, optimized_nodes = local_recursive_function(
        list_optimizations, out, {}, 0)
    return final_outs[out_index]
