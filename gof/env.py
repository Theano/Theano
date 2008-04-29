
from copy import copy

import graph
from features import Listener, Orderings, Constraint, Tool, uniq_features
import utils
from utils import AbstractFunctionError


class InconsistencyError(Exception):
    """
    This exception is raised by Env whenever one of the listeners marks
    the graph as inconsistent.
    """
    pass


def require_set(x):
    try:
        req = x.env_require
    except AttributeError:
        req = []
    return req


class Env(graph.Graph):
    """
    An Env represents a subgraph bound by a set of input results and a
    set of output results. An op is in the subgraph iff it depends on
    the value of some of the Env's inputs _and_ some of the Env's
    outputs depend on it. A result is in the subgraph iff it is an
    input or an output of an op that is in the subgraph.

    The Env supports the replace operation which allows to replace a
    result in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in omega.

    An Env's functionality can be extended with features, which must
    be subclasses of L{Listener}, L{Constraint}, L{Orderings} or
    L{Tool}.

    Regarding inputs and orphans:

    In the context of a computation graph, the inputs and orphans are
    both results that are the source nodes of computation.  Those
    results that are named as inputs will be assumed to contain fresh.
    In other words, the backward search from outputs will stop at any
    node that has been explicitly named as an input.
    """

    ### Special ###

    def __init__(self, inputs, outputs): #, consistency_check = True):
        """
        Create an Env which operates on the subgraph bound by the inputs and outputs
        sets. If consistency_check is False, an illegal graph will be tolerated.
        """

#         self._features = {}
#         self._listeners = {}
#         self._constraints = {}
#         self._orderings = {}
#         self._tools = {}

        self._features = []

        # The inputs and outputs set bound the subgraph this Env operates on.
        self.inputs = list(inputs)
        self.outputs = list(outputs)

        # All nodes in the subgraph defined by inputs and outputs are cached in nodes
        self.nodes = set()

        # Ditto for results
        self.results = set(self.inputs)
        
        # Set of all the results that are not an output of an op in the subgraph but
        # are an input of an op in the subgraph.
        # e.g. z for inputs=(x, y) and outputs=(x + (y - z),)
        # We initialize them to the set of outputs; if an output depends on an input,
        # it will be removed from the set of orphans.
        self.orphans = set(outputs).difference(inputs)

#         for feature_class in uniq_features(features):
#             self.add_feature(feature_class, False)

        # Maps results to nodes that use them:
        # if op.inputs[i] == v then (op, i) in self._clients[v]
        self._clients = {}

        # List of functions that undo the replace operations performed.
        # e.g. to recover the initial graph one could write: for u in self.history.__reversed__(): u()
        self.history = []
        
        self.__import_r__(self.outputs)
#         for op in self.nodes():
#             self.satisfy(op)
        
#         if consistency_check:
#             self.validate()


    ### Public interface ###

    def add_output(self, output):
        "Add an output to the Env."
        self.outputs.add(output)
        self.orphans.add(output)
        self.__import_r__([output])

    def clients(self, r):
        "Set of all the (op, i) pairs such that op.inputs[i] is r."
        return self._clients.get(r, set())

    def checkpoint(self):
        """
        Returns an object that can be passed to self.revert in order to backtrack
        to a previous state.
        """
        return len(self.history)

    def consistent(self):
        """
        Returns True iff the subgraph is consistent and does not violate the
        constraints set by the listeners.
        """
        try:
            self.validate()
        except InconsistencyError:
            return False
        return True

#     def satisfy(self, x):
#         "Adds the features required by x unless they are already present."
#         for feature_class in require_set(x):
#             self.add_feature(feature_class)

    def extend(self, feature, do_import = True, validate = False):
        """
        @todo out of date
        Adds an instance of the feature_class to this env's supported
        features. If do_import is True and feature_class is a subclass
        of Listener, its on_import method will be called on all the Nodes
        already in the env.
        """
        if feature in self._features:
            return # the feature is already present
        self.__add_feature__(feature, do_import)
        if validate:
            self.validate()

    def execute_callbacks(self, name, *args):
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            fn(*args)

    def __add_feature__(self, feature, do_import):
        self._features.append(feature)
        publish = getattr(feature, 'publish', None)
        if publish is not None:
            publish()
        if do_import:
            try:
                fn = feature.on_import
            except AttributeError:
                return
            for node in self.io_toposort():
                fn(node)

    def __del_feature__(self, feature):
        try:
            del self._features[feature]
        except:
            pass
        unpublish = hasattr(feature, 'unpublish')
        if unpublish is not None:
            unpublish()            

    def get_feature(self, feature):
        idx = self._features.index(feature)
        return self._features[idx]

    def has_feature(self, feature):
        return feature in self._features

    def nclients(self, r):
        "Same as len(self.clients(r))."
        return len(self.clients(r))

    def edge(self, r):
        return r in self.inputs or r in self.orphans

    def follow(self, r):
        node = r.owner
        if self.edge(r):
            return None
        else:
            if node is None:
                raise Exception("what the fuck")
            return node.inputs

    def has_node(self, node):
        return node in self.nodes

    def replace(self, r, new_r, consistency_check = True):
        """
        This is the main interface to manipulate the subgraph in Env.
        For every op that uses r as input, makes it use new_r instead.
        This may raise an error if the new result violates type
        constraints for one of the target nodes. In that case, no
        changes are made.

        If the replacement makes the graph inconsistent and the value
        of consistency_check is True, this function will raise an
        InconsistencyError and will undo the operation, leaving the
        graph the way it was before the call to replace.

        If consistency_check is False, the replacement will succeed
        even if there is an inconsistency, unless the replacement
        violates hard constraints on the types involved.
        """
        assert r in self.results

        # Save where we are so we can backtrack
        if consistency_check:
            chk = self.checkpoint()

        # The copy is required so undo can know what clients to move back!
        clients = copy(self.clients(r))

        # Messy checks so we know what to do if we are replacing an output
        # result. Note that if v is an input result, we do nothing at all for
        # now (it's not clear what it means to replace an input result).
        was_output = False
        if r in self.outputs:
            was_output = True
            self.outputs[self.outputs.index(r)] = new_r

        was_input = False
        if r in self.inputs:
            was_input = True
            self.inputs[self.inputs.index(r)] = new_r

        # The actual replacement operation occurs here. This might raise
        # an error.
        self.__move_clients__(clients, r, new_r) # not sure how to order this wrt to adjusting the outputs

        # This function undoes the replacement.
        def undo():
            # Restore self.outputs
            if was_output:
                self.outputs[self.outputs.index(new_r)] = r

            # Restore self.inputs
            if was_input:
                self.inputs[self.inputs.index(new_r)] = r

            # Move back the clients. This should never raise an error.
            self.__move_clients__(clients, new_r, r)

        self.history.append(undo)
        
        if consistency_check:
            try:
                self.validate()
            except InconsistencyError, e:
                self.revert(chk)
                raise

    def replace_all(self, d):
        """
        For (r, new_r) in d.items(), replaces r with new_r. Checks for
        consistency at the end and raises an InconsistencyError if the
        graph is not consistent. If an error is raised, the graph is
        restored to what it was before.
        """
        chk = self.checkpoint()
        try:
            for r, new_r in d.items():
                self.replace(r, new_r, False)
        except Exception, e:
            self.revert(chk)
            raise
        try:
            self.validate()
        except InconsistencyError, e:
            self.revert(chk)
            raise

    def revert(self, checkpoint):
        """
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements).  A checkpoint at any
        given time can be obtained using self.checkpoint().
        """
        while len(self.history) > checkpoint:
            f = self.history.pop()
            f()

    def supplemental_orderings(self):
        """
        Returns a dictionary of {op: set(prerequisites)} that must
        be satisfied in addition to the order defined by the structure
        of the graph (returns orderings that not related to input/output
        relationships).
        """
        ords = {}
        for feature in self._features:
            if hasattr(feature, 'orderings'):
                for op, prereqs in feature.orderings().items():
                    ords.setdefault(op, set()).update(prereqs)
        return ords

    def toposort(self):
        """
        Returns a list of nodes in the order that they must be executed
        in order to preserve the semantics of the graph and respect
        the constraints put forward by the listeners.
        """
        ords = self.supplemental_orderings()
        order = graph.io_toposort(self.inputs, self.outputs, ords)
        return order
    
    def validate(self):
        """
        Raises an error if the graph is inconsistent.
        """
        self.execute_callbacks('validate')
#         for constraint in self._constraints.values():
#             constraint.validate()
        return True


    ### Private interface ###

    def __add_clients__(self, r, all):
        """
        r -> result
        all -> list of (op, i) pairs representing who r is an input of.

        Updates the list of clients of r with all.
        """
        self._clients.setdefault(r, set()).update(all)

    def __remove_clients__(self, r, all):
        """
        r -> result
        all -> list of (op, i) pairs representing who r is an input of.

        Removes all from the clients list of r.
        """
        if not all:
            return
        self._clients[r].difference_update(all)
        if not self._clients[r]:
            del self._clients[r]
            if r in self.orphans:
                self.orphans.remove(r)

    def __import_r_satisfy__(self, results):
        # Satisfies the owners of the results.
        for op in graph.ops(self.results, results):
            self.satisfy(op)

    def __import_r__(self, results):
        # Imports the owners of the results
        for result in results:
            owner = result.owner
            if owner:
                self.__import__(result.owner)
            if result not in self.results:
                self.results.add(result)
                self.orphans.add(result)

    def __import__(self, op):
        # We import the nodes in topological order. We only are interested
        # in new nodes, so we use all results we know of as if they were the input set.
        # (the functions in the graph module only use the input set to
        # know where to stop going down)
        new_nodes = graph.io_toposort(self.results.difference(self.orphans), op.outputs)
        
        for op in new_nodes:

            self.nodes.add(op)
            self.results.update(op.outputs)
            self.orphans.difference_update(op.outputs)
            
            for i, input in enumerate(op.inputs):
                self.__add_clients__(input, [(op, i)])
                if input not in self.results:
                    # This input is an orphan because if the op that
                    # produced it was in the subgraph, io_toposort
                    # would have placed it before, so we would have
                    # seen it (or it would already be in the graph)
                    self.orphans.add(input)
                    self.results.add(input)
            
            self.execute_callbacks('on_import', op)
#             for listener in self._listeners.values():
#                 try:
#                     listener.on_import(op)
#                 except AbstractFunctionError:
#                     pass
    __import__.E_output = 'op output in Env.inputs'

    def __prune_r__(self, results):
        # Prunes the owners of the results.
        for result in set(results):
            if result in self.inputs:
                continue
            owner = result.owner
            if owner:
                self.__prune__(owner)
#             if result in self.results:
#                 self.results.remove(result)
#                 if result in self.orphans:
#                     self.orphans.remove(result)

    def __prune__(self, op):
        # If op's outputs have no clients, removes it from the graph
        # and recursively tries to prune its inputs. If at least one
        # of the op's outputs is an output to the graph or has a client
        # then __prune__ is a no-op.
        for output in op.outputs:
            # Cannot prune an op which is an output or used somewhere
            if self.clients(output) or output in self.outputs: #output in self.outputs or self.clients(output):
                return
        if op not in self.nodes: # this can happen from replacing an orphan
            return
        self.nodes.remove(op)
        self.results.difference_update(op.outputs)
        self.execute_callbacks('on_prune', op)
#         for listener in self._listeners.values():
#             try:
#                 listener.on_prune(op)
#             except AbstractFunctionError:
#                 pass
            
        for i, input in enumerate(op.inputs):
            self.__remove_clients__(input, [(op, i)])
        self.__prune_r__(op.inputs)

    def __move_clients__(self, clients, r, new_r):

        if not (r.type == new_r.type):
            raise TypeError("Cannot move clients between Results that have different types.", r, new_r)
        
        # We import the new result in the fold
        self.__import_r__([new_r])

        for op, i in clients:
            op.inputs[i] = new_r
#         try:
#             # Try replacing the inputs
#             for op, i in clients:
#                 op.set_input(i, new_r)
#         except:
#             # Oops!
#             for op, i in clients:
#                 op.set_input(i, r)
#             self.__prune_r__([new_r])
#             raise
        self.__remove_clients__(r, clients)
        self.__add_clients__(new_r, clients)

#         # We import the new result in the fold
#         # why was this line AFTER the set_inputs???
#         # if we do it here then satisfy in import fucks up...
#         self.__import_r__([new_r])

        self.execute_callbacks('on_rewire', clients, r, new_r)
#         for listener in self._listeners.values():
#             try:
#                 listener.on_rewire(clients, r, new_r)
#             except AbstractFunctionError:
#                 pass

        # We try to get rid of the old one
        self.__prune_r__([r])

    def __str__(self):
        return "[%s]" % ", ".join(graph.as_string(self.inputs, self.outputs))

    def clone_get_equiv(self, clone_inputs = True):
        equiv = graph.clone_get_equiv(self.inputs, self.outputs, clone_inputs)
        new = self.__class__([equiv[input] for input in self.inputs],
                             [equiv[output] for output in self.outputs])
        for feature in self._features:
            new.extend(feature)
        return new, equiv

    def clone(self, clone_inputs = True):
        equiv = graph.clone_get_equiv(self.inputs, self.outputs, clone_inputs)
        new = self.__class__([equiv[input] for input in self.inputs],
                             [equiv[output] for output in self.outputs])
        for feature in self._features:
            new.extend(feature)
        try:
            new.set_equiv(equiv)
        except AttributeError:
            pass
        return new

    def __copy__(self):
        return self.clone()


