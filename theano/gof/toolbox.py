from __future__ import absolute_import, print_function, division
from functools import partial
from collections import OrderedDict

import sys
import time
import inspect

import numpy as np
from six.moves import StringIO

import theano
from theano import config
from theano.gof import graph


class AlreadyThere(Exception):
    """
    Raised by a Feature's on_attach callback method if the FunctionGraph
    attempting to attach the feature already has a functionally identical
    feature.

    """

    pass


class ReplacementDidntRemovedError(Exception):
    """
    This exception should be thrown by replace_all_validate_remove
    when an optimization wanted to remove a Variable or a Node from
    the graph, but the replacement it gived didn't do that.

    """

    pass


class BadOptimization(Exception):
    """
    Exception: some variable and its substitute take different runtime values.

    Note: If there is only 1 parameter and it is a string, we will use
    it as the error message. This is needed when we catch, extend and
    reraise an error.

    """

    new_r = None
    """
    A `Variable` instance that took a different value from `old_r`,
    but which replaced `old_r`.

    """

    old_r = None
    """
    A `Variable` instance that was replaced by `new_r`.

    """

    old_r_val = None
    """
    The value computed for `old_r`.

    """

    new_r_val = None
    """
    The value computed for `new_r`.

    """

    reason = None
    """
    An object that indicates why old_r was turned into new_r.

    Convention is that this is the name of the optimization that
    requested the replacement.

    """

    old_graph = ""
    """
    A multiline string representation of the graph leading to
    old_r, at the time of the replacement.

    """

    new_graph = ""
    """
    A multiline string representation of the graph leading to
    new_r, at the time of the replacement.

    """

    def __init__(self, old_r, new_r=None, old_r_val=None, new_r_val=None, reason=None,
                 old_graph=None, new_graph=None):
        super(BadOptimization, self).__init__()
        self.old_r = old_r
        self.new_r = new_r
        self.old_r_val = old_r_val
        self.new_r_val = new_r_val
        self.reason = reason
        self.old_graph = old_graph
        self.new_graph = new_graph

        # To allow extending the error message of an existing error.
        self.full_err = None
        if isinstance(old_r, str):
            assert (new_r is None and old_r_val is None and new_r_val is None and
                    reason is None and old_graph is None and new_graph is None)
            self.full_err = old_r

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """
        Return a pretty multiline string representating the cause
        of the exception.

        """
        # We have a pre-made message
        if getattr(self, 'full_err', None) is not None:
            return self.full_err
        sio = StringIO()
        val_str_len_limit = 800
        print("BadOptimization Error", super(BadOptimization,
                                             self).__str__(), file=sio)
        print("  Variable: id", id(self.new_r), self.new_r, file=sio)
        print("  Op", self.new_r.owner, file=sio)
        print("  Value Type:", type(self.new_r_val), file=sio)
        try:
            ssio = StringIO()
            print("  Old Value shape, dtype, strides:", end=' ', file=ssio)
            print(self.old_r_val.shape, end=' ', file=ssio)
            print(self.old_r_val.dtype, end=' ', file=ssio)
            print(self.old_r_val.strides, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        str_old_r_val = str(self.old_r_val)
        if len(str_old_r_val) > val_str_len_limit:
            print("  Old Value: ", str(self.old_r_val)[
                :val_str_len_limit], '...', file=sio)
        else:
            print("  Old Value: ", str(self.old_r_val), file=sio)

        try:
            ssio = StringIO()
            print("  New Value shape, dtype, strides:", end=' ', file=ssio)
            print(self.new_r_val.shape, end=' ', file=ssio)
            print(self.new_r_val.dtype, end=' ', file=ssio)
            print(self.new_r_val.strides, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass
        str_new_r_val = str(self.new_r_val)
        if len(str_new_r_val) > val_str_len_limit:
            print("  New Value: ", str(self.new_r_val)[
                :val_str_len_limit], '...', file=sio)
        else:
            print("  New Value: ", str(self.new_r_val), file=sio)

        try:
            ov = np.asarray(self.old_r_val)
            nv = np.asarray(self.new_r_val)
            ssio = StringIO()
            abs_diff = np.absolute(nv - ov)
            print("  Max Abs Diff: ", np.max(abs_diff), file=ssio)
            print("  Mean Abs Diff: ", np.mean(abs_diff), file=ssio)
            print("  Median Abs Diff: ", np.median(abs_diff), file=ssio)
            print("  Std Abs Diff: ", np.std(abs_diff), file=ssio)
            arg_max_val = np.argmax(abs_diff)
            values_at_max = (nv.flatten()[arg_max_val],
                             ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)

            # N.B. the maximum(..., 1e-8) protects against div by 0 when
            #      nv == ov == 0
            reldiff = (abs_diff /
                       np.maximum(np.absolute(nv) + np.absolute(ov),
                                  1e-8))
            print("  Max Rel Diff: ", np.max(reldiff), file=ssio)
            print("  Mean Rel Diff: ", np.mean(reldiff), file=ssio)
            print("  Median Rel Diff: ", np.median(reldiff), file=ssio)
            print("  Std Rel Diff: ", np.std(reldiff), file=ssio)
            arg_max_val = np.argmax(reldiff)
            values_at_max = (nv.flatten()[arg_max_val],
                             ov.flatten()[arg_max_val])
            print("  Value at Max Diff: ", values_at_max, file=ssio)
            # only if all succeeds to we add anything to sio
            print(ssio.getvalue(), file=sio)
        except Exception:
            pass

        print("  Reason: ", str(self.reason), file=sio)
        print("  Old Graph:", file=sio)
        print(self.old_graph, file=sio)
        print("  New Graph:", file=sio)
        print(self.new_graph, file=sio)
        print("", file=sio)
        print("Hint: relax the tolerance by setting tensor.cmp_sloppy=1",
              file=sio)
        print("  or even tensor.cmp_sloppy=2 for less-strict comparison",
              file=sio)
        return sio.getvalue()


class Feature(object):
    """
    Base class for FunctionGraph extensions.

    A Feature is an object with several callbacks that are triggered
    by various operations on FunctionGraphs. It can be used to enforce
    graph properties at all stages of graph optimization.

    See Also
    --------
    theano.gof.toolbox : for common extensions.

    """

    def on_attach(self, function_graph):
        """
        Called by FunctionGraph.attach_feature, the method that attaches
        the feature to the FunctionGraph. Since this is called after the
        FunctionGraph is initially populated, this is where you should
        run checks on the initial contents of the FunctionGraph.

        The on_attach method may raise the AlreadyThere exception to cancel
        the attach operation if it detects that another Feature instance
        implementing the same functionality is already atttached to the
        FunctionGraph.

        The feature has great freedom in what it can do with the
        function_graph: it may, for example, add methods to it dynamically.

        """

    def on_detach(self, function_graph):
        """
        Called by remove_feature(feature).  Should remove any dynamically-added
        functionality that it installed into the function_graph.

        """

    def on_import(self, function_graph, node, reason):
        """
        Called whenever a node is imported into function_graph, which is
        just before the node is actually connected to the graph.
        Note: on_import is not called when the graph is created. If you
        want to detect the first nodes to be implemented to the graph,
        you should do this by implementing on_attach.

        """

    def on_prune(self, function_graph, node, reason):
        """
        Called whenever a node is pruned (removed) from the function_graph,
        after it is disconnected from the graph.

        """

    def on_change_input(self, function_graph, node, i, r, new_r, reason=None):
        """
        Called whenever node.inputs[i] is changed from r to new_r.
        At the moment the callback is done, the change has already
        taken place.

        If you raise an exception in this function, the state of the graph
        might be broken for all intents and purposes.

        """

    def orderings(self, function_graph):
        """
        Called by toposort. It should return a dictionary of
        {node: predecessors} where predecessors is a list of
        nodes that should be computed before the key node.

        If you raise an exception in this function, the state of the graph
        might be broken for all intents and purposes.

        """
        return OrderedDict()


class Bookkeeper(Feature):

    def on_attach(self, fgraph):
        """
        Called by FunctionGraph.attach_feature, the method that attaches
        the feature to the FunctionGraph. Since this is called after the
        FunctionGraph is initially populated, this is where you should
        run checks on the initial contents of the FunctionGraph.
        """
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_import(fgraph, node, "on_attach")

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_prune(fgraph, node, 'Bookkeeper.detach')


class GetCheckpoint:

    def __init__(self, history, fgraph):
        self.h = history
        self.fgraph = fgraph
        self.nb = 0

    def __call__(self):
        self.h.history[self.fgraph] = []
        self.nb += 1
        return self.nb


class LambdExtract:

    def __init__(self, fgraph, node, i, r, reason=None):
        self.fgraph = fgraph
        self.node = node
        self.i = i
        self.r = r
        self.reason = reason

    def __call__(self):
        return self.fgraph.change_input(self.node, self.i, self.r,
                                        reason=("Revert", self.reason))


class History(Feature):
    """Keep an history of changes to an FunctionGraph.

    This history can be reverted up to the last checkpoint.. We can
    revert to only 1 point in the past. This limit was added to lower
    the memory usage.

    """
    pickle_rm_attr = ["checkpoint", "revert"]

    def __init__(self):
        self.history = {}

    def on_attach(self, fgraph):
        if hasattr(fgraph, 'checkpoint') or hasattr(fgraph, 'revert'):
            raise AlreadyThere("History feature is already present or in"
                               " conflict with another plugin.")
        self.history[fgraph] = []
        # Don't call unpickle here, as ReplaceValidate.on_attach()
        # call to History.on_attach() will call the
        # ReplaceValidate.unpickle and not History.unpickle
        fgraph.checkpoint = GetCheckpoint(self, fgraph)
        fgraph.revert = partial(self.revert, fgraph)

    def unpickle(self, fgraph):
        fgraph.checkpoint = GetCheckpoint(self, fgraph)
        fgraph.revert = partial(self.revert, fgraph)

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        del fgraph.checkpoint
        del fgraph.revert
        del self.history[fgraph]

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if self.history[fgraph] is None:
            return
        h = self.history[fgraph]
        h.append(LambdExtract(fgraph, node, i, r, reason))

    def revert(self, fgraph, checkpoint):
        """
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements). A checkpoint at any
        given time can be obtained using self.checkpoint().

        """
        h = self.history[fgraph]
        self.history[fgraph] = None
        assert fgraph.checkpoint.nb == checkpoint
        while h:
            f = h.pop()
            f()
        self.history[fgraph] = h


class Validator(Feature):
    pickle_rm_attr = ["validate", "consistent"]

    def on_attach(self, fgraph):
        for attr in ('validate', 'validate_time'):
            if hasattr(fgraph, attr):
                raise AlreadyThere("Validator feature is already present or in"
                                   " conflict with another plugin.")
        # Don't call unpickle here, as ReplaceValidate.on_attach()
        # call to History.on_attach() will call the
        # ReplaceValidate.unpickle and not History.unpickle
        fgraph.validate = partial(self.validate_, fgraph)
        fgraph.consistent = partial(self.consistent_, fgraph)

    def unpickle(self, fgraph):
        fgraph.validate = partial(self.validate_, fgraph)
        fgraph.consistent = partial(self.consistent_, fgraph)

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        del fgraph.validate
        del fgraph.consistent

    def validate_(self, fgraph):
        """
        If the caller is replace_all_validate, just raise the
        exception. replace_all_validate will print out the
        verbose output. Or it has to be done here before raise.
        """
        t0 = time.time()
        try:
            ret = fgraph.execute_callbacks('validate')
        except Exception as e:
            cf = inspect.currentframe()
            uf = cf.f_back
            uf_info = inspect.getframeinfo(uf)

            # If the caller is replace_all_validate, just raise the
            # exception. replace_all_validate will print out the
            # verbose output.
            # Or it has to be done here before raise.
            if uf_info.function == 'replace_all_validate':
                raise
            else:
                verbose = uf.f_locals.get('verbose', False)
                if verbose:
                    r = uf.f_locals.get('r', "")
                    reason = uf_info.function
                    print("validate failed on node %s.\n Reason: %s, %s" %
                          (r, reason, e))
                raise
        t1 = time.time()
        if fgraph.profile:
            fgraph.profile.validate_time += t1 - t0
        return ret

    def consistent_(self, fgraph):
        try:
            fgraph.validate()
            return True
        except Exception:
            return False


class ReplaceValidate(History, Validator):
    pickle_rm_attr = (["replace_validate", "replace_all_validate",
                       "replace_all_validate_remove"] +
                      History.pickle_rm_attr + Validator.pickle_rm_attr)

    def on_attach(self, fgraph):
        for attr in ('replace_validate', 'replace_all_validate',
                     'replace_all_validate_remove'):
            if hasattr(fgraph, attr):
                raise AlreadyThere("ReplaceValidate feature is already present"
                                   " or in conflict with another plugin.")
        self._nodes_removed = set()
        self.fail_validate = False
        History.on_attach(self, fgraph)
        Validator.on_attach(self, fgraph)
        self.unpickle(fgraph)

    def unpickle(self, fgraph):
        History.unpickle(self, fgraph)
        Validator.unpickle(self, fgraph)
        fgraph.replace_validate = partial(self.replace_validate, fgraph)
        fgraph.replace_all_validate = partial(self.replace_all_validate,
                                              fgraph)
        fgraph.replace_all_validate_remove = partial(
            self.replace_all_validate_remove, fgraph)

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        History.on_detach(self, fgraph)
        Validator.on_detach(self, fgraph)
        del self._nodes_removed
        del fgraph.replace_validate
        del fgraph.replace_all_validate
        del fgraph.replace_all_validate_remove

    def replace_validate(self, fgraph, r, new_r, reason=None):
        self.replace_all_validate(fgraph, [(r, new_r)], reason=reason)

    def replace_all_validate(self, fgraph, replacements,
                             reason=None, verbose=None):
        chk = fgraph.checkpoint()
        if verbose is None:
            verbose = config.optimizer_verbose
        if config.scan.debug:
            scans = [n for n in fgraph.apply_nodes if isinstance(n.op, theano.scan_module.scan_op.Scan)]

        for r, new_r in replacements:
            try:
                fgraph.replace(r, new_r, reason=reason, verbose=False)
            except Exception as e:
                msg = str(e)
                s1 = 'The type of the replacement must be the same'
                s2 = 'does not belong to this FunctionGraph'
                s3 = 'maximum recursion depth exceeded'
                if s3 in msg:
                    # There is nothing safe we can do to recover from this.
                    # So don't revert as this raise a different error
                    # that isn't helpful.
                    e.args += (
                        "Please, report this to theano-dev mailing list."
                        " As a temporary work around, you can raise Python"
                        " stack limit with:"
                        " import sys; sys.setrecursionlimit(10000)",)
                    raise
                elif (s1 not in msg and s2 not in msg):
                    out = sys.stderr
                    print("<<!! BUG IN FGRAPH.REPLACE OR A LISTENER !!>>",
                          type(e), e, reason, file=out)
                # this might fail if the error is in a listener:
                # (fgraph.replace kinda needs better internal error handling)
                fgraph.revert(chk)
                raise
        try:
            fgraph.validate()
        except Exception as e:
            fgraph.revert(chk)
            if verbose:
                print("validate failed on node %s.\n Reason: %s, %s" % (r, reason, e))
            raise
        if config.scan.debug:
            scans2 = [n for n in fgraph.apply_nodes if isinstance(n.op, theano.scan_module.scan_op.Scan)]
            nb = len(scans)
            nb2 = len(scans2)
            if nb2 > nb:
                print("Extra scan introduced", nb, nb2, getattr(reason, 'name', reason), r, new_r)
            elif nb2 < nb:
                print("Scan removed", nb, nb2, getattr(reason, 'name', reason), r, new_r)
        if verbose:
            print(reason, r, new_r)
        # The return is needed by replace_all_validate_remove
        return chk

    def replace_all_validate_remove(self, fgraph, replacements,
                                    remove, reason=None, warn=True):
        """
        As replace_all_validate, revert the replacement if the ops
        in the list remove are still in the graph. Also print a warning.

        """
        chk = fgraph.replace_all_validate(replacements, reason)
        self._nodes_removed.update(remove)
        for rm in remove:
            if rm in fgraph.apply_nodes or rm in fgraph.variables:
                fgraph.revert(chk)
                if warn:
                    out = sys.stderr
                    print(
                        "WARNING: An optimization wanted to replace a Variable"
                        " in the graph, but the replacement for it doesn't"
                        " remove it. We disabled the optimization."
                        " Your function runs correctly, but it would be"
                        " appreciated if you submit this problem to the"
                        " mailing list theano-users so that we can fix it.",
                        file=out)
                    print(reason, replacements, file=out)
                raise ReplacementDidntRemovedError()

    def __getstate__(self):
        d = self.__dict__.copy()
        if "history" in d:
            del d["history"]
        return d

    def on_import(self, fgraph, node, reason):
        if node in self._nodes_removed:
            self.fail_validate = True

    def validate(self, fgraph):
        if self.fail_validate:
            self.fail_validate = False
            raise theano.gof.InconsistencyError("Trying to reintroduce a removed node")


class NodeFinder(Bookkeeper):

    def __init__(self):
        self.fgraph = None
        self.d = {}

    def on_attach(self, fgraph):
        if self.fgraph is not None:
            raise Exception("A NodeFinder instance can only serve one "
                            "FunctionGraph.")
        if hasattr(fgraph, 'get_nodes'):
            raise AlreadyThere("NodeFinder is already present or in conflict"
                               " with another plugin.")
        self.fgraph = fgraph
        fgraph.get_nodes = partial(self.query, fgraph)
        Bookkeeper.on_attach(self, fgraph)

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        if self.fgraph is not fgraph:
            raise Exception("This NodeFinder instance was not attached to the"
                            " provided fgraph.")
        self.fgraph = None
        del fgraph.get_nodes
        Bookkeeper.on_detach(self, fgraph)

    def on_import(self, fgraph, node, reason):
        try:
            self.d.setdefault(node.op, []).append(node)
        except TypeError:  # node.op is unhashable
            return
        except Exception as e:
            print('OFFENDING node', type(node), type(node.op), file=sys.stderr)
            try:
                print('OFFENDING node hash', hash(node.op), file=sys.stderr)
            except Exception:
                print('OFFENDING node not hashable', file=sys.stderr)
            raise e

    def on_prune(self, fgraph, node, reason):
        try:
            nodes = self.d[node.op]
        except TypeError:  # node.op is unhashable
            return
        nodes.remove(node)
        if not nodes:
            del self.d[node.op]

    def query(self, fgraph, op):
        try:
            all = self.d.get(op, [])
        except TypeError:
            raise TypeError("%s in unhashable and cannot be queried by the"
                            " optimizer" % op)
        all = list(all)
        return all


class PrintListener(Feature):

    def __init__(self, active=True):
        self.active = active

    def on_attach(self, fgraph):
        if self.active:
            print("-- attaching to: ", fgraph)

    def on_detach(self, fgraph):
        """
        Should remove any dynamically added functionality
        that it installed into the function_graph
        """
        if self.active:
            print("-- detaching from: ", fgraph)

    def on_import(self, fgraph, node, reason):
        if self.active:
            print("-- importing: %s, reason: %s" % (node, reason))

    def on_prune(self, fgraph, node, reason):
        if self.active:
            print("-- pruning: %s, reason: %s" % (node, reason))

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if self.active:
            print("-- changing (%s.inputs[%s]) from %s to %s" % (
                node, i, r, new_r))


class PreserveNames(Feature):
    """
    This preserve some variables names during optimization.

    Deprecated. We need to keep it to allow unpickling.
    """

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if r.name is not None and new_r.name is None:
            new_r.name = r.name


class PreserveVariableAttributes(Feature):
    """
    This preserve some variables attributes and tag during optimization.
    """

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if r.name is not None and new_r.name is None:
            new_r.name = r.name
        if getattr(r.tag, 'nan_guard_mode_check', False) and getattr(
                new_r.tag, 'nan_guard_mode_check', False) is False:
            new_r.tag.nan_guard_mode_check = r.tag.nan_guard_mode_check


class NoOutputFromInplace(Feature):

    def __init__(self, first_output_idx=0, last_output_idx=None):
        self.first_idx = first_output_idx
        self.last_idx = last_output_idx

    def validate(self, fgraph):
        if not hasattr(fgraph, 'destroyers'):
            return True

        outputs_to_validate = list(fgraph.outputs)[self.first_idx:
                                                   self.last_idx]

        for out in outputs_to_validate:

            if out.owner is None:
                continue

            # Validate that the node that produces the output does not produce
            # it by modifying something else inplace.
            node = out.owner
            op = node.op
            out_idx = node.outputs.index(out)
            if hasattr(op, 'destroy_map') and out_idx in op.destroy_map:
                raise theano.gof.InconsistencyError(
                    "A function graph Feature has requested (probably for ",
                    "efficiency reasons for scan) that outputs of the graph",
                    "be prevented from being the result of inplace ",
                    "operations. This has prevented output ", out, " from ",
                    "being computed by modifying another variable ",
                    "inplace.")
