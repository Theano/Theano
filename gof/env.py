
from copy import copy
import graph
import utils


class InconsistencyError(Exception):
    """
    This exception should be thrown by listeners to Env when the
    graph's state is invalid.
    """
    pass



class Env(utils.object2):
    """
    An Env represents a subgraph bound by a set of input results and a
    set of output results. The inputs list should contain all the inputs
    on which the outputs depend. Results of type Value or Constant are
    not counted as inputs.

    The Env supports the replace operation which allows to replace a
    result in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in theano.

    It can also be "extended" using env.extend(some_object). See the
    toolbox and ext modules for common extensions.
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
        # sets up r so it belongs to this env
        if hasattr(r, 'env') and r.env is not None and r.env is not self:
            raise Exception("%s is already owned by another env" % r)
        r.env = self
        r.clients = []

    def __setup_node__(self, node):
        # sets up node so it belongs to this env
        if hasattr(node, 'env') and node.env is not self:
            raise Exception("%s is already owned by another env" % node)
        node.env = self
        node.deps = {}


    ### clients ###

    def clients(self, r):
        "Set of all the (node, i) pairs such that node.inputs[i] is r."
        return r.clients

    def __add_clients__(self, r, new_clients):
        """
        r -> result
        new_clients -> list of (node, i) pairs such that node.inputs[i] is r.

        Updates the list of clients of r with new_clients.
        """
        r.clients += new_clients

    def __remove_clients__(self, r, clients_to_remove, prune = True):
        """
        r -> result
        clients_to_remove -> list of (op, i) pairs such that node.inputs[i] is not r anymore.

        Removes all from the clients list of r.
        """
        for entry in clients_to_remove:
            r.clients.remove(entry)
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
                        raise TypeError("An input of the graph was not provided and not given a value", r)
        
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
        """
        Changes node.inputs[i] to new_r.

        new_r.type == old_r.type must be True, where old_r is the
        current value of node.inputs[i] which we want to replace.

        For each feature that has a 'on_change_input' method, calls:
          feature.on_change_input(env, node, i, old_r, new_r)
        """
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
        For every node that uses r as input, makes it use new_r instead.
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
        Adds a feature to this env. The feature may define one
        or more of the following methods:

         - feature.on_attach(env)
            Called by extend. The feature has great freedom in what
            it can do with the env: it may, for example, add methods
            to it dynicamically.
         - feature.on_detach(env)
            Called by remove_feature(feature).
         - feature.on_import(env, node)*
            Called whenever a node is imported into env, which is
            just before the node is actually connected to the graph.
         - feature.on_prune(env, node)*
            Called whenever a node is pruned (removed) from the env,
            after it is disconnected from the graph.
         - feature.on_change_input(env, node, i, r, new_r)*
            Called whenever node.inputs[i] is changed from r to new_r.
            At the moment the callback is done, the change has already
            taken place.
         - feature.orderings(env)
            Called by toposort. It should return a dictionary of
            {node: predecessors} where predecessors is a list of
            nodes that should be computed before the key node.

        * If you raise an exception in the functions marked with an
          asterisk, the state of the graph might be inconsistent.
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
        """
        Removes the feature from the graph.

        Calls feature.on_detach(env) if an on_detach method is defined.
        """
        try:
            self._features.remove(feature)
        except:
            return
        detach = getattr(feature, 'on_detach', None)
        if detach is not None:
            detach(self)


    ### callback utils ###
    
    def execute_callbacks(self, name, *args):
        """
        Calls
          getattr(feature, name)(*args)
        for each feature which has a method called after name.
        """
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            fn(self, *args)

    def collect_callbacks(self, name, *args):
        """
        Returns a dictionary d such that:
          d[feature] == getattr(feature, name)(*args)
        For each feature which has a method called after name.
        """
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
        """
        Returns an ordering of the graph's Apply nodes such that:
          - All the nodes of the inputs of a node are before that node.
          - Satisfies the orderings provided by each feature that has
            an 'orderings' method.

        If a feature has an 'orderings' method, it will be called with
        this env as sole argument. It should return a dictionary of
        {node: predecessors} where predecessors is a list of nodes
        that should be computed before the key node.
        """
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

    def check_integrity(self):
        """
        Call this for a diagnosis if things go awry.
        """
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


    ### clone ###

    def clone(self):
        return self.clone_get_equiv()[0]

    def clone_get_equiv(self):
        g, equiv = graph.clone_get_equiv(self.inputs, self.outputs)
        e = Env([equiv[i] for i in self.inputs],
                [equiv[o] for o in self.outputs])
        for feature in self._features:
            e.extend(feature)
        return e, equiv







