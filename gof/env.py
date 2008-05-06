
from copy import copy

import graph
##from features import Listener, Orderings, Constraint, Tool, uniq_features
import utils
from utils import AbstractFunctionError


class InconsistencyError(Exception):
    """
    This exception is raised by Env whenever one of the listeners marks
    the graph as inconsistent.
    """
    pass



class Env(object): #(graph.Graph):
    """
    An Env represents a subgraph bound by a set of input results and a
    set of output results. An op is in the subgraph iff it depends on
    the value of some of the Env's inputs _and_ some of the Env's
    outputs depend on it. A result is in the subgraph iff it is an
    input or an output of an op that is in the subgraph.

    The Env supports the replace operation which allows to replace a
    result in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in theano.
    """

    ### Special ###

    def __init__(self, inputs, outputs):
        """
        Create an Env which operates on the subgraph bound by the inputs and outputs
        sets.
        """

        self._features = []
        
        # All nodes in the subgraph defined by inputs and outputs are cached in nodes
        self.nodes = set()
        
        # Ditto for results
        self.results = set()

        self.inputs = list(inputs)
        for input in self.inputs:
            if input.owner is not None:
                raise ValueError("One of the provided inputs is the output of an already existing node. " \
                                 "If that is okay, either discard that input's owner or use graph.clone.")
            self.__setup_r__(input)
            self.results.add(input)

        self.__import_r__(outputs)
        self.outputs = outputs
        for i, output in enumerate(outputs):
            output.clients.append(('output', i))

        self.node_locks = {}
        self.result_locks = {}


    ### Setup a Result ###

    def __setup_r__(self, r):
        if hasattr(r, 'env') and r.env is not None and r.env is not self:
            raise Exception("%s is already owned by another env" % r)
        r.env = self
        r.clients = []

    def __setup_node__(self, node):
        if hasattr(node, 'env') and node.env is not self:
            raise Exception("%s is already owned by another env" % node)
        node.env = self
        node.deps = {}


    ### clients ###

    def clients(self, r):
        "Set of all the (op, i) pairs such that op.inputs[i] is r."
        return r.clients

    def __add_clients__(self, r, all):
        """
        r -> result
        all -> list of (op, i) pairs representing who r is an input of.

        Updates the list of clients of r with all.
        """
        r.clients += all

    def __remove_clients__(self, r, all, prune = True):
        """
        r -> result
        all -> list of (op, i) pairs representing who r is an input of.

        Removes all from the clients list of r.
        """
        for entry in all:
            r.clients.remove(entry)
            # remove from orphans?
        if not r.clients:
            if prune:
                self.__prune_r__([r])
                return False
            return True
        return False


    ### import ###

    def __import_r__(self, results):
        # Imports the owners of the results
        for node in set(r.owner for r in results if r.owner is not None):
            self.__import__(node)
        for r in results:
            if r.owner is None and not isinstance(r, graph.Value) and r not in self.inputs:
                raise TypeError("Undeclared input", r)
            self.results.add(r)

    def __import__(self, node, check = True):
        # We import the nodes in topological order. We only are interested
        # in new nodes, so we use all results we know of as if they were the input set.
        # (the functions in the graph module only use the input set to
        # know where to stop going down)
        new_nodes = graph.io_toposort(self.results, node.outputs)

        if check:
            for node in new_nodes:
                if hasattr(node, 'env') and node.env is not self:
                    raise Exception("%s is already owned by another env" % node)
                for r in node.inputs:
                    if hasattr(r, 'env') and r.env is not self:
                        raise Exception("%s is already owned by another env" % r)
                    if r.owner is None and not isinstance(r, graph.Value) and r not in self.inputs:
                        raise TypeError("Undeclared input", r)
        
        for node in new_nodes:
            self.__setup_node__(node)
            self.nodes.add(node)
            for output in node.outputs:
                self.__setup_r__(output)
                self.results.add(output)
            for i, input in enumerate(node.inputs):
                if input not in self.results:
                    self.__setup_r__(input)
                    self.results.add(input)
                self.__add_clients__(input, [(node, i)])
            assert node.env is self
            self.execute_callbacks('on_import', node)


    ### prune ###

    def __prune_r__(self, results):
        # Prunes the owners of the results.
        for node in set(r.owner for r in results if r.owner is not None):
            self.__prune__(node)
        for r in results:
            if not r.clients and r in self.results:
                self.results.remove(r)

    def __prune__(self, node):
        if node not in self.nodes:
            raise Exception("%s does not belong to this Env and cannot be pruned." % node)
        assert node.env is self
        # If node's outputs have no clients, removes it from the graph
        # and recursively tries to prune its inputs. If at least one
        # of the op's outputs is an output to the graph or has a client
        # then __prune__ is a no-op.
        for output in node.outputs:
            # Cannot prune an op which is an output or used somewhere
            if self.clients(output) or output in self.outputs: #output in self.outputs or self.clients(output):
                return
        self.nodes.remove(node)
        self.results.difference_update(node.outputs)
        self.execute_callbacks('on_prune', node)
        
        for i, input in enumerate(node.inputs):
            self.__remove_clients__(input, [(node, i)])
        #self.__prune_r__(node.inputs)



    ### change input ###

    def change_input(self, node, i, new_r):
        if node == 'output':
            r = self.outputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the same as the type of the original Result.", r, new_r)
            self.outputs[i] = new_r
        else:
            if node.env is not self:
                raise Exception("Cannot operate on %s because it does not belong to this Env" % node)
            r = node.inputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the same as the type of the original Result.", r, new_r)
            node.inputs[i] = new_r
        
        self.__import_r__([new_r])
        self.__add_clients__(new_r, [(node, i)])
        prune = self.__remove_clients__(r, [(node, i)], False)
        self.execute_callbacks('on_change_input', node, i, r, new_r)
        if prune:
            self.__prune_r__([r])


    ### replace ###
    
    def replace(self, r, new_r):
        """
        This is the main interface to manipulate the subgraph in Env.
        For every op that uses r as input, makes it use new_r instead.
        This may raise an error if the new result violates type
        constraints for one of the target nodes. In that case, no
        changes are made.
        """
        if r.env is not self:
            raise Exception("Cannot replace %s because it does not belong to this Env" % r)
        if not r.type == new_r.type:
            raise TypeError("The type of the replacement must be the same as the type of the original Result.", r, new_r)
        assert r in self.results

        for node, i in list(r.clients):
            assert node == 'output' and self.outputs[i] is r or node.inputs[i] is r
            self.change_input(node, i, new_r)

    def replace_all(self, pairs):
        for r, new_r in pairs:
            self.replace(r, new_r)


    ### features ###
    
    def extend(self, feature):
        """
        @todo out of date
        Adds an instance of the feature_class to this env's supported
        features. If do_import is True and feature_class is a subclass
        of Listener, its on_import method will be called on all the Nodes
        already in the env.
        """
        if feature in self._features:
            return # the feature is already present
        self._features.append(feature)
        attach = getattr(feature, 'on_attach', None)
        if attach is not None:    
            try:
                attach(self)
            except:
                self._features.pop()
                raise

    def remove_feature(self, feature):
        try:
            self._features.remove(feature)
        except:
            return
        detach = getattr(feature, 'on_detach', None)
        if detach is not None:
            detach(self)


    ### callback utils ###
    
    def execute_callbacks(self, name, *args):
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            fn(self, *args)

    def collect_callbacks(self, name, *args):
        d = {}
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            d[feature] = fn(*args)
        return d


    ### misc ###

    def toposort(self):
        env = self
        ords = {}
        for feature in env._features:
            if hasattr(feature, 'orderings'):
                for op, prereqs in feature.orderings(env).items():
                    ords.setdefault(op, []).extend(prereqs)
        order = graph.io_toposort(env.inputs, env.outputs, ords)
        return order
    
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

    def check_integrity(self):
        nodes = graph.ops(self.inputs, self.outputs)
        if self.nodes != nodes:
            missing = nodes.difference(self.nodes)
            excess = self.nodes.difference(nodes)
            raise Exception("The nodes are inappropriately cached. missing, in excess: ", missing, excess)
        for node in nodes:
            if node.env is not self:
                raise Exception("Node should belong to the env.", node)
            for i, result in enumerate(node.inputs):
                if result.env is not self:
                    raise Exception("Input of node should belong to the env.", result, (node, i))
                if (node, i) not in result.clients:
                    raise Exception("Inconsistent clients list.", (node, i), result.clients)
        results = graph.results(self.inputs, self.outputs)
        if self.results != results:
            missing = results.difference(self.results)
            excess = self.results.difference(results)
            raise Exception("The results are inappropriately cached. missing, in excess: ", missing, excess)
        for result in results:
            if result.owner is None and result not in self.inputs and not isinstance(result, graph.Value):
                raise Exception("Undeclared input.", result)
            if result.env is not self:
                raise Exception("Result should belong to the env.", result)
            for node, i in result.clients:
                if node == 'output':
                    if self.outputs[i] is not result:
                        raise Exception("Inconsistent clients list.", result, self.outputs[i])
                    continue
                if node not in nodes:
                    raise Exception("Client not in env.", result, (node, i))
                if node.inputs[i] is not result:
                    raise Exception("Inconsistent clients list.", result, node.inputs[i])

    def __str__(self):
        return "[%s]" % ", ".join(graph.as_string(self.inputs, self.outputs))









