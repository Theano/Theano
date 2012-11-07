import sys
import time

from theano.gof.python25 import partial

import graph


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

    def on_import(self, function_graph, node):
        """
        Called whenever a node is imported into function_graph, which is
        just before the node is actually connected to the graph.
        Note: on_import is not called when the graph is created. If you
        want to detect the first nodes to be implemented to the graph,
        you should do this by implementing on_attach.
        """

    def on_prune(self, function_graph, node):
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
        return {}


class Bookkeeper(Feature):

    def on_attach(self, fgraph):
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_import(fgraph, node)

    def on_detach(self, fgraph):
        for node in graph.io_toposort(fgraph.inputs, fgraph.outputs):
            self.on_prune(fgraph, node)


class History(Feature):

    def __init__(self):
        self.history = {}

    def on_attach(self, fgraph):
        if hasattr(fgraph, 'checkpoint') or hasattr(fgraph, 'revert'):
            raise AlreadyThere("History feature is already present or in"
                               " conflict with another plugin.")
        self.history[fgraph] = []
        fgraph.checkpoint = lambda: len(self.history[fgraph])
        fgraph.revert = partial(self.revert, fgraph)

    def on_detach(self, fgraph):
        del fgraph.checkpoint
        del fgraph.revert
        del self.history[fgraph]

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if self.history[fgraph] is None:
            return
        h = self.history[fgraph]
        h.append(lambda: fgraph.change_input(node, i, r,
                                          reason=("Revert", reason)))

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

    def on_attach(self, fgraph):
        for attr in ('validate', 'validate_time'):
            if hasattr(fgraph, attr):
                raise AlreadyThere("Validator feature is already present or in"
                                   " conflict with another plugin.")

        def validate():
            t0 = time.time()
            ret = fgraph.execute_callbacks('validate')
            t1 = time.time()
            if fgraph.profile:
                fgraph.profile.validate_time += t1 - t0
            return ret

        fgraph.validate = validate

        def consistent():
            try:
                fgraph.validate()
                return True
            except Exception:
                return False
        fgraph.consistent = consistent

    def on_detach(self, fgraph):
        del fgraph.validate
        del fgraph.consistent


class ReplaceValidate(History, Validator):

    def on_attach(self, fgraph):
        History.on_attach(self, fgraph)
        Validator.on_attach(self, fgraph)
        for attr in ('replace_validate', 'replace_all_validate'):
            if hasattr(fgraph, attr):
                raise AlreadyThere("ReplaceValidate feature is already present"
                                   " or in conflict with another plugin.")
        fgraph.replace_validate = partial(self.replace_validate, fgraph)
        fgraph.replace_all_validate = partial(self.replace_all_validate, fgraph)
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

    def replace_all_validate(self, fgraph, replacements, reason=None):
        chk = fgraph.checkpoint()
        for r, new_r in replacements:
            try:
                fgraph.replace(r, new_r, reason=reason)
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


class NodeFinder(dict, Bookkeeper):

    def __init__(self):
        self.fgraph = None

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

    def on_import(self, fgraph, node):
        try:
            self.setdefault(node.op, []).append(node)
        except TypeError:  # node.op is unhashable
            return
        except Exception, e:
            print >> sys.stderr, 'OFFENDING node', type(node), type(node.op)
            try:
                print >> sys.stderr, 'OFFENDING node hash', hash(node.op)
            except Exception:
                print >> sys.stderr, 'OFFENDING node not hashable'
            raise e

    def on_prune(self, fgraph, node):
        try:
            nodes = self[node.op]
        except TypeError:  # node.op is unhashable
            return
        nodes.remove(node)
        if not nodes:
            del self[node.op]

    def query(self, fgraph, op):
        try:
            all = self.get(op, [])
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

    def on_import(self, fgraph, node):
        if self.active:
            print "-- importing: %s" % node

    def on_prune(self, fgraph, node):
        if self.active:
            print "-- pruning: %s" % node

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if self.active:
            print "-- changing (%s.inputs[%s]) from %s to %s" % (
                node, i, r, new_r)


class PreserveNames(Feature):

    def on_change_input(self, fgraph, mode, i, r, new_r, reason=None):
        if r.name is not None and new_r.name is None:
            new_r.name = r.name


