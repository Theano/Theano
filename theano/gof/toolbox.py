import sys
import time

import theano
from theano import config
from theano.gof.python25 import partial
from theano.gof.python25 import OrderedDict
from theano.gof import graph


class AlreadyThere(Exception):
    """Raised by a Feature's on_attach callback method if the FunctionGraph
    attempting to attach the feature already has a functionally identical
    feature."""
    pass


class ReplacementDidntRemovedError(Exception):
    """This exception should be thrown by replace_all_validate_remove
    when an optimization wanted to remove a Variable or a Node from
    the graph, but the replacement it gived didn't do that.

    """
    pass


class Feature(object):
    """
    Base class for FunctionGraph extensions.

    A Feature is an object with several callbacks that are triggered
    by various operations on FunctionGraphs. It can be used to enforce
    graph properties at all stages of graph optimization.

    See toolbox and ext modules for common extensions.
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
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_import(fgraph, node, "on_attach")

    def on_detach(self, fgraph):
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_prune(fgraph, node, 'Bookkeeper.detach')


class GetCheckpoint:

    def __init__(self, history, fgraph):
        self.h = history
        self.fgraph = fgraph

    def __call__(self):
        return len(self.h.history[self.fgraph])


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
        checkpoint (undoes all replacements).  A checkpoint at any
        given time can be obtained using self.checkpoint().
        """
        h = self.history[fgraph]
        self.history[fgraph] = None
        while len(h) > checkpoint:
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
        del fgraph.validate
        del fgraph.consistent

    def validate_(self, fgraph):
        t0 = time.time()
        ret = fgraph.execute_callbacks('validate')
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
    pickle_rm_attr = ["replace_validate", "replace_all_validate",
                      "replace_all_validate_remove"] + \
                      History.pickle_rm_attr + Validator.pickle_rm_attr
        
    

    def on_attach(self, fgraph):
        for attr in ('replace_validate', 'replace_all_validate',
                     'replace_all_validate_remove'):
            if hasattr(fgraph, attr):
                raise AlreadyThere("ReplaceValidate feature is already present"
                                   " or in conflict with another plugin.")
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
        History.on_detach(self, fgraph)
        Validator.on_detach(self, fgraph)
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
        for r, new_r in replacements:
            try:
                fgraph.replace(r, new_r, reason=reason, verbose=False)
            except Exception, e:
                if ('The type of the replacement must be the same' not in
                    str(e) and 'does not belong to this FunctionGraph' not in str(e)):
                    out = sys.stderr
                    print >> out, "<<!! BUG IN FGRAPH.REPLACE OR A LISTENER !!>>",
                    print >> out, type(e), e, reason
                # this might fail if the error is in a listener:
                # (fgraph.replace kinda needs better internal error handling)
                fgraph.revert(chk)
                raise
        try:
            fgraph.validate()
        except Exception, e:
            fgraph.revert(chk)
            raise
        if verbose:
            print reason, r, new_r
        return chk

    def replace_all_validate_remove(self, fgraph, replacements,
                                    remove, reason=None, warn=True):
        """As replace_all_validate, revert the replacement if the ops
        in the list remove are still in the graph. It also print a warning.

        """
        chk = fgraph.replace_all_validate(replacements, reason)
        for rm in remove:
            if rm in fgraph.apply_nodes or rm in fgraph.variables:
                fgraph.revert(chk)
                if warn:
                    out = sys.stderr
                    print >> out, (
                        "WARNING: An optimization wanted to replace a Variable"
                        " in the graph, but the replacement for it doesn't"
                        " remove it. We disabled the optimization."
                        " Your function runs correctly, but it would be"
                        " appreciated if you submit this problem to the"
                        " mailing list theano-users so that we can fix it.")
                    print >> out, reason, replacements
                raise ReplacementDidntRemovedError()

    def __getstate__(self):
        d = self.__dict__.copy()
        if "history" in d:
            del d["history"]
        return d


class NodeFinder(Bookkeeper):

    def __init__(self):
        self.fgraph = None
        self.d = {}

    def on_attach(self, fgraph):
        if self.fgraph is not None:
            raise Exception("A NodeFinder instance can only serve one FunctionGraph.")
        if hasattr(fgraph, 'get_nodes'):
            raise AlreadyThere("NodeFinder is already present or in conflict"
                               " with another plugin.")
        self.fgraph = fgraph
        fgraph.get_nodes = partial(self.query, fgraph)
        Bookkeeper.on_attach(self, fgraph)

    def on_detach(self, fgraph):
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
        except Exception, e:
            print >> sys.stderr, 'OFFENDING node', type(node), type(node.op)
            try:
                print >> sys.stderr, 'OFFENDING node hash', hash(node.op)
            except Exception:
                print >> sys.stderr, 'OFFENDING node not hashable'
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
            print "-- attaching to: ", fgraph

    def on_detach(self, fgraph):
        if self.active:
            print "-- detaching from: ", fgraph

    def on_import(self, fgraph, node, reason):
        if self.active:
            print "-- importing: %s, reason: %s" % (node, reason)

    def on_prune(self, fgraph, node, reason):
        if self.active:
            print "-- pruning: %s, reason: %s" % (node, reason)

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if self.active:
            print "-- changing (%s.inputs[%s]) from %s to %s" % (
                node, i, r, new_r)


class PreserveNames(Feature):

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if r.name is not None and new_r.name is None:
            new_r.name = r.name


class NoOutputFromInplace(Feature):

    def validate(self, fgraph):
        if not hasattr(fgraph, 'destroyers'):
            return True
        for out in list(fgraph.outputs):

            if out.owner is None:
                continue

            # Validate that the node that produces the output does not produce
            # it by modifying something else inplace.
            node = out.owner
            op = node.op
            out_idx = node.outputs.index(out)
            if hasattr(op, 'destroy_map') and out_idx in op.destroy_map.keys():
                raise theano.gof.InconsistencyError(
                    "A function graph Feature has requested (probably for ",
                    "efficiency reasons for scan) that outputs of the graph",
                    "be prevented from being the result of inplace ",
                    "operations. This has prevented output ", out, " from ",
                    "being computed by modifying another variable ",
                    "inplace.")
