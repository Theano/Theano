"""
fg.py: fg stands for FunctionGraph
Contains the FunctionGraph class and exception
types that it can raise.

"""
from __future__ import absolute_import, print_function, division
from collections import OrderedDict
import time

import theano
from theano.gof import graph
from theano.gof import utils
from theano.gof import toolbox
from theano import config

from six import iteritems, itervalues
from six.moves import StringIO
from theano.gof.utils import get_variable_trace_string
from theano.misc.ordered_set import OrderedSet
NullType = None


class CachedConstantError(Exception):
    """
    An exception thrown when we put in a FunctionGraph a Constant
    that is cached. This should not happen as the user can reuse this
    cached constant in other FunctionGraph.

    """

    pass


class InconsistencyError(Exception):
    """
    This exception should be thrown by listeners to FunctionGraph when the
    graph's state is invalid.

    """

    pass


class MissingInputError(Exception):
    """
    A symbolic input needed to compute the outputs is missing.

    """
    def __init__(self, *args, **kwargs):
        if kwargs:
            # The call to list is needed for Python 3
            assert list(kwargs.keys()) == ["variable"]
            error_msg = get_variable_trace_string(kwargs["variable"])
            if error_msg:
                args = args + (error_msg,)
        s = '\n'.join(args)  # Needed to have the new line print correctly
        Exception.__init__(self, s)


class FunctionGraph(utils.object2):
    """
    A FunctionGraph represents a subgraph bound by a set of input variables and
    a set of output variables, ie a subgraph that specifies a theano function.
    The inputs list should contain all the inputs on which the outputs depend.
    Variables of type Constant are not counted as inputs.

    The FunctionGraph supports the replace operation which allows to replace a
    variable in the subgraph by another, e.g. replace (x + x).out by (2
    * x).out. This is the basis for optimization in theano.

    This class is also responsible for verifying that a graph is valid
    (ie, all the dtypes and broadcast patterns are compatible with the
    way the the Variables are used) and for annotating the Variables with
    a .clients field that specifies which Apply nodes use the variable.
    The .clients field combined with the .owner field and the Apply nodes'
    .inputs field allows the graph to be traversed in both directions.

    It can also be extended with new features using
    FunctionGraph.attach_feature(<toolbox.Feature instance>).
    See toolbox.Feature for event types and documentation.
    Extra features allow the FunctionGraph to verify new properties of
    a graph as it is optimized.
    # TODO: are there other things features can do to the fgraph?

    Historically, the FunctionGraph was called an Env. Keep this in mind
    while reading out-of-date documentation, e-mail support threads, etc.

    The constructor creates a FunctionGraph which operates on the subgraph
    bound by the inputs and outputs sets.

    This class keeps a pointer to the inputs and outputs, and also modifies
    them.

    #TODO: document what variables are[not] set in the FunctionGraph when a
    feature is added via the constructor. How constructed is the
    FunctionGraph?

    Parameters
    ----------
    inputs
        Inputs nodes of the graph, usually declared by the user.
    outputs
        Outputs nodes of the graph.
    clone
        If true, we will clone the graph. This is useful to remove the constant
        cache problem.

    Notes
    -----
    The intermediate nodes between 'inputs' and 'outputs' are not explicitely
    passed.

    """

    def __init__(self, inputs, outputs, features=None, clone=True,
                 update_mapping=None):
        """
        Create an FunctionGraph which operates on the subgraph bound by the
        inputs and outputs sets.

        Parameters
        ----------
        inputs : list of variables
            Inputs nodes of the graph, usually declared by the user
        outputs : list of variables
            Outputs nodes of the graph.
        clone : boolean
            If true, we will clone the graph. This is useful to remove the
            constant cache problem.
        update_mapping : dictionary
            Mapping between the inputs with updates and the outputs
            corresponding to their updates.
        """

        if clone:
            inputs, outputs = graph.clone(inputs, outputs)

        self.execute_callbacks_time = 0
        self.execute_callbacks_times = {}

        if features is None:
            features = []

        # XXX: Unless I'm missing something (but there's no documentation,
        # so I probably am) this should be a set.
        self._features = []

        # All apply nodes in the subgraph defined by inputs and
        # outputs are cached in this field
        self.apply_nodes = set()

        # Ditto for variable nodes.
        # It must contain all fgraph.inputs and all apply_nodes
        # outputs even if they aren't used in the graph.
        self.variables = set()

        self.inputs = list(inputs)
        self.outputs = outputs

        for f in features:
            self.attach_feature(f)
        self.attach_feature(toolbox.ReplaceValidate())

        for input in self.inputs:
            if input.owner is not None:
                raise ValueError("One of the provided inputs is the output of"
                                 "an already existing node. "
                                 "If that is okay, either discard that "
                                 "input's owner or use graph.clone.")
            self.__setup_r__(input)
            self.variables.add(input)

        for output in outputs:
            self.__import_r__(output, reason="init")
        for i, output in enumerate(outputs):
            output.clients.append(('output', i))

        self.profile = None
        self.update_mapping = update_mapping

    def add_input(self, input):
        if input not in self.inputs:
            self.inputs.append(input)
            self.__setup_r__(input)
            self.variables.add(input)

    # Setup a Variable #
    def __setup_r__(self, r):
        # sets up r so it belongs to this fgraph
        if getattr(r, 'cached', False):
            raise CachedConstantError(
                "You manually constructed a FunctionGraph, but you passed it a"
                " graph that has a cached constant. This should not happen."
                " Clone the graph before building the FunctionGraph.")
        if (hasattr(r, 'fgraph') and
                r.fgraph is not None and
                r.fgraph is not self):
            raise Exception("%s is already owned by another fgraph" % r)
        r.fgraph = self
        r.clients = []
        # self.execute_callbacks('on_setup_variable', r)

    def __setup_node__(self, node):
        # sets up node so it belongs to this fgraph
        if hasattr(node, 'fgraph') and node.fgraph is not self:
            raise Exception("%s is already owned by another fgraph" % node)
        if (hasattr(node.op, 'view_map') and
            not all(isinstance(view, (list, tuple))
                    for view in itervalues(node.op.view_map))):
            raise Exception("Op '%s' have a bad view map '%s',"
                            " the values must be tuples or lists." % (
                                str(node.op), str(node.op.view_map)))
        if (hasattr(node.op, 'destroy_map') and
            not all(isinstance(destroy, (list, tuple))
                    for destroy in itervalues(node.op.destroy_map))):
            raise Exception("Op '%s' have a bad destroy map '%s',"
                            " the values must be tuples or lists." % (
                                str(node.op), str(node.op.destroy_map)))
        node.fgraph = self
        node.deps = {}
        # self.execute_callbacks('on_setup_node', node)

    def disown(self):
        """
        Cleans up all of this FunctionGraph's nodes and variables so they are
        not associated with this FunctionGraph anymore.

        The FunctionGraph should not be used anymore after disown is called.

        """
        for f in self._features:
            self.remove_feature(f)
        for apply_node in self.apply_nodes:
            del apply_node.fgraph
            del apply_node.deps
        for variable in self.variables:
            del variable.fgraph
            del variable.clients
        self.apply_nodes = set()
        self.variables = set()
        self.inputs = None
        self.outputs = None
        self.profile = None
        self.update_mapping = None

    # clients #
    def clients(self, r):
        """
        Set of all the (node, i) pairs such that node.inputs[i] is r.
        Told differently, a list of (node,i) such that each node have
        r as input at index i.

        """
        return r.clients

    def __add_client__(self, r, new_client):
        """
        Updates the list of clients of r with new_clients.

        Parameters
        ----------
        r
            Variable.
        new_client
            (node, i) pair such that node.inputs[i] is r.

        """
        # Ne need to do the assert as it is always True. The logic
        # that call __add_client__ is valid. When the client list is
        # long, the check it time consuming, so we don't enable it by
        # default.
        # assert not new_client in r.clients
        r.clients.append(new_client)

    def __remove_client__(self, r, client_to_remove,
                          reason=None):
        """
        Removes all from the clients list of r.

        This is the main method to remove variable or apply node from
        an FunctionGraph.

        Remove r from this fgraph if it don't have clients left. If it
        have an owner and all the outputs of the owner have no
        clients, it will be removed.

        Parameters
        ----------
        r : Variable
            The clients of r will be removed.
        client_to_remove : (op, i) pair
            (op, i) pair such that node.inputs[i] is not r anymore.

        """
        l = [(r, client_to_remove)]
        while l:
            r, client_to_remove = l.pop()
            r.clients.remove(client_to_remove)
            # entry should be uniq in r. No need to assert it as it is
            # already asserted in __add_client__.
            # assert entry not in r.clients
            if r.clients:
                continue

            # r have no more clients, so check if we need to remove it
            # and its parent.
            variable = r
            if not variable.owner:
                # A Constant or input without client. Remove it.
                self.variables.remove(variable)
                # This allow to quickly know if a var is still in the fgraph
                # or not.
                del variable.fgraph
            else:
                apply_node = variable.owner
                used = [output for output in apply_node.outputs
                        if output.clients]
                # If the apply node is not used and is not an output
                if not used:
                    if not hasattr(apply_node.tag, 'removed_by'):
                        apply_node.tag.removed_by = []
                    apply_node.tag.removed_by.append(str(reason))
                    self.apply_nodes.remove(apply_node)
                    # del apply_node.fgraph
                    self.variables.difference_update(apply_node.outputs)
                    # for var in apply_node.outputs:
                    #     del var.fgraph
                    self.execute_callbacks('on_prune', apply_node, reason)

                    for i, input in enumerate(apply_node.inputs):
                        l.append((input, (apply_node, i)))

    def __import_r__(self, variable, reason):
        """
        Import variables to this FunctionGraph and also their apply_node,
        if those nodes are not in this graph.

        Parameters:
        ----------
        reason
            reason is the name of the optimization or operation in progress.
        """
        # Imports the owners of the variables
        if variable.owner and variable.owner not in self.apply_nodes:
                self.__import__(variable.owner, reason=reason)
        elif (variable.owner is None and
                not isinstance(variable, graph.Constant) and
                variable not in self.inputs):
            global NullType
            if NullType is None:
                from .null_type import NullType
            if isinstance(variable.type, NullType):
                raise TypeError("Computation graph contains a NaN. " +
                                variable.type.why_null)
            raise MissingInputError("Undeclared input", variable=variable)
        if not getattr(variable, 'fgraph', None) is self:
            self.__setup_r__(variable)
        self.variables.add(variable)

    def __import__(self, apply_node, check=True, reason=None):
        """
        Given an apply_node, recursively search from this node to know graph,
        and then add all unknown variables and apply_nodes to this graph.
        """
        node = apply_node

        # We import the nodes in topological order. We only are interested
        # in new nodes, so we use all variables we know of as if they were the input set.
        # (the functions in the graph module only use the input set to
        # know where to stop going down)
        new_nodes = graph.io_toposort(self.variables, apply_node.outputs)

        if check:
            for node in new_nodes:
                if hasattr(node, 'fgraph') and node.fgraph is not self:
                    raise Exception("%s is already owned by another fgraph" % node)
                for r in node.inputs:
                    if hasattr(r, 'fgraph') and r.fgraph is not self:
                        raise Exception("%s is already owned by another fgraph" % r)
                    if (r.owner is None and
                            not isinstance(r, graph.Constant) and
                            r not in self.inputs):
                        # Standard error message
                        error_msg = ("Input %d of the graph (indices start "
                                     "from 0), used to compute %s, was not "
                                     "provided and not given a value. Use the "
                                     "Theano flag exception_verbosity='high', "
                                     "for more information on this error."
                                     % (node.inputs.index(r), str(node)))
                        raise MissingInputError(error_msg, variable=r)

        for node in new_nodes:
            assert node not in self.apply_nodes
            self.__setup_node__(node)
            self.apply_nodes.add(node)
            if not hasattr(node.tag, 'imported_by'):
                node.tag.imported_by = []
            node.tag.imported_by.append(str(reason))
            for output in node.outputs:
                self.__setup_r__(output)
                self.variables.add(output)
            for i, input in enumerate(node.inputs):
                if input not in self.variables:
                    self.__setup_r__(input)
                    self.variables.add(input)
                self.__add_client__(input, (node, i))
            assert node.fgraph is self
            self.execute_callbacks('on_import', node, reason)

    # change input #
    def change_input(self, node, i, new_r, reason=None):
        """
        Changes node.inputs[i] to new_r.

        new_r.type == old_r.type must be True, where old_r is the
        current value of node.inputs[i] which we want to replace.

        For each feature that has a 'on_change_input' method, calls:
        feature.on_change_input(function_graph, node, i, old_r, new_r, reason)

        """
        # TODO: ERROR HANDLING FOR LISTENERS (should it complete the change or revert it?)
        if node == 'output':
            r = self.outputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the"
                                " same as the type of the original Variable.",
                                r, new_r)
            self.outputs[i] = new_r
        else:
            if node.fgraph is not self:
                raise Exception("Cannot operate on %s because it does not"
                                " belong to this FunctionGraph" % node)
            r = node.inputs[i]
            if not r.type == new_r.type:
                raise TypeError("The type of the replacement must be the"
                                " same as the type of the original Variable.",
                                r, new_r)
            node.inputs[i] = new_r

        if r is new_r:
            return

        self.__import_r__(new_r, reason=reason)
        self.__add_client__(new_r, (node, i))
        self.__remove_client__(r, (node, i), reason=reason)
        # Precondition: the substitution is semantically valid
        # However it may introduce cycles to the graph,  in which case the
        # transaction will be reverted later.
        self.execute_callbacks('on_change_input', node, i,
                               r, new_r, reason=reason)

    # replace #
    def replace(self, r, new_r, reason=None, verbose=None):
        """
        This is the main interface to manipulate the subgraph in FunctionGraph.
        For every node that uses r as input, makes it use new_r instead.

        """
        if verbose is None:
            verbose = config.optimizer_verbose
        if verbose:
            print(reason, r, new_r)
        if hasattr(r, 'fgraph') and r.fgraph is not self:
            raise Exception("Cannot replace %s because it does not belong "
                            "to this FunctionGraph" % r, str(reason))
        if r.type != new_r.type:
            new_r2 = r.type.convert_variable(new_r)
            # We still make sure that the type converts correctly
            if new_r2 is None or new_r2.type != r.type:
                done = dict()
                used_ids = dict()
                old = theano.compile.debugmode.debugprint(
                    r, prefix='  ', depth=6,
                    file=StringIO(), done=done,
                    print_type=True,
                    used_ids=used_ids).getvalue()
                new = theano.compile.debugmode.debugprint(
                    new_r, prefix='  ', depth=6,
                    file=StringIO(), done=done,
                    print_type=True,
                    used_ids=used_ids).getvalue()
                raise toolbox.BadOptimization(
                    r, new_r, None, None, str(reason) +
                    ". The type of the replacement must be the same.", old, new)
            new_r = new_r2
        if r not in self.variables:
            # this variable isn't in the graph... don't raise an
            # exception here, just return silently because it makes it
            # easier to implement some optimizations for
            # multiple-output ops
            return

        if theano.config.compute_test_value != 'off':
            try:
                tval = theano.gof.op.get_test_value(r)
                new_tval = theano.gof.op.get_test_value(new_r)
            except AttributeError:
                pass
            else:
                tval_shape = getattr(tval, 'shape', None)
                new_tval_shape = getattr(new_tval, 'shape', None)
                if tval_shape != new_tval_shape:
                    raise AssertionError(
                        "The replacement variable has a test value with "
                        "a shape different from the original variable's "
                        "test value. Original: %s, new: %s"
                        % (tval_shape, new_tval_shape),
                        r, new_r, str(reason))

        for node, i in list(r.clients):  # copy the client list for iteration
            assert (node == 'output' and self.outputs[i] is r) or (node.inputs[i] is r)
            self.change_input(node, i, new_r, reason=reason)

        # sometimes the following is triggered.  If you understand why, please explain to James.
        # He's curious... -JB20090331
        # if len(r.clients) != 0:
        #    print >> sys.stderr, "WARNING: CLIENTS LEFT AFTER REPLACE", r, r.clients

    def replace_all(self, pairs, reason=None):
        """
        For every node that uses r as input, makes it use new_r instead

        """
        for r, new_r in pairs:
            self.replace(r, new_r, reason=reason)

    def attach_feature(self, feature):
        """
        Adds a gof.toolbox.Feature to this function_graph and triggers its
        on_attach callback.

        """
        # Filter out literally identical features
        if feature in self._features:
            return  # the feature is already present

        # Filter out functionally identical features.
        # Features may use their on_attach method to raise
        # toolbox.AlreadyThere if they detect that some
        # installed feature does the same thing already
        attach = getattr(feature, 'on_attach', None)
        if attach is not None:
            try:
                attach(self)
            except toolbox.AlreadyThere:
                return
        self.execute_callbacks_times.setdefault(feature, 0)
        # it would be nice if we could require a specific class instead of
        # a "workalike" so we could do actual error checking
        # if not isinstance(feature, toolbox.Feature):
        #    raise TypeError("Expected gof.toolbox.Feature instance, got "+\
        #            str(type(feature)))

        # Add the feature
        self._features.append(feature)

    def remove_feature(self, feature):
        """
        Removes the feature from the graph.

        Calls feature.on_detach(function_graph) if an on_detach method
        is defined.

        """
        try:
            # Why do we catch the exeception anyway?
            self._features.remove(feature)
        except ValueError:
            return
        detach = getattr(feature, 'on_detach', None)
        if detach is not None:
            detach(self)

    # callback utils #
    def execute_callbacks(self, name, *args, **kwargs):
        """Execute callbacks

        Calls `getattr(feature, name)(*args)` for each feature which has
        a method called after name.

        """
        t0 = time.time()
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                # this is safe because there is no work done inside the
                # try; the AttributeError reall must come from feature.${name}
                # not existing
                continue
            tf0 = time.time()
            fn(self, *args, **kwargs)
            self.execute_callbacks_times[feature] += time.time() - tf0
        self.execute_callbacks_time += time.time() - t0

    def collect_callbacks(self, name, *args):
        """Collects callbacks

        Returns a dictionary d such that
        `d[feature] == getattr(feature, name)(*args)`
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

    # misc #
    def toposort(self):
        """Toposort

        Return an ordering of the graph's Apply nodes such that

        * All the nodes of the inputs of a node are before that node.
        * Satisfies the orderings provided by each feature that has
          an 'orderings' method.

        If a feature has an 'orderings' method, it will be called with
        this FunctionGraph as sole argument. It should return a dictionary of
        `{node: predecessors}` where predecessors is a list of nodes that
        should be computed before the key node.
        """
        if len(self.apply_nodes) < 2:
            # optimization
            # when there are 0 or 1 nodes, no sorting is necessary
            # This special case happens a lot because the OpWiseCLinker
            # produces 1-element graphs.
            return list(self.apply_nodes)
        fg = self

        ords = self.orderings()

        order = graph.io_toposort(fg.inputs, fg.outputs, ords)

        return order

    def orderings(self):
        """
        Return dict d s.t. d[node] is a list of nodes that must be evaluated
        before node itself can be evaluated.

        This is used primarily by the destroy_handler feature to ensure that
        all clients of any destroyed inputs have already computed their outputs.

        Notes
        -----
        This only calls the orderings() fct on all features. It does not
        take care of computing dependencies by itself.

        """
        assert isinstance(self._features, list)
        all_orderings = []

        for feature in self._features:
            if hasattr(feature, 'orderings'):
                orderings = feature.orderings(self)
                if not isinstance(orderings, OrderedDict):
                    raise TypeError("Non-deterministic return value from " +
                                    str(feature.orderings) +
                                    ". Nondeterministic object is " +
                                    str(orderings))
                if len(orderings) > 0:
                    all_orderings.append(orderings)
                    for node, prereqs in iteritems(orderings):
                        if not isinstance(prereqs, (list, OrderedSet)):
                            raise TypeError(
                                "prereqs must be a type with a "
                                "deterministic iteration order, or toposort "
                                " will be non-deterministic.")
        if len(all_orderings) == 1:
            # If there is only 1 ordering, we reuse it directly.
            return all_orderings[0].copy()
        else:
            # If there is more than 1 ordering, combine them.
            ords = OrderedDict()
            for orderings in all_orderings:
                for node, prereqs in iteritems(orderings):
                    ords.setdefault(node, []).extend(prereqs)
            return ords

    def check_integrity(self):
        """
        Call this for a diagnosis if things go awry.

        """
        nodes = graph.ops(self.inputs, self.outputs)
        if self.apply_nodes != nodes:
            missing = nodes.difference(self.apply_nodes)
            excess = self.apply_nodes.difference(nodes)
            raise Exception(
                "The nodes are inappropriately cached. missing, in excess: ",
                missing, excess)
        for node in nodes:
            if node.fgraph is not self:
                raise Exception("Node should belong to the FunctionGraph.",
                                node)
            for i, variable in enumerate(node.inputs):
                if variable.fgraph is not self:
                    raise Exception(
                        "Input of node should belong to the FunctionGraph.",
                        variable, (node, i))
                if (node, i) not in variable.clients:
                    raise Exception("Inconsistent clients list.",
                                    (node, i), variable.clients)
        variables = set(graph.variables(self.inputs, self.outputs))
        if set(self.variables) != variables:
            missing = variables.difference(self.variables)
            excess = self.variables.difference(variables)
            raise Exception(
                "The variables are inappropriately cached. missing, in excess: ",
                missing, excess)
        for variable in variables:
            if (variable.owner is None and
                    variable not in self.inputs and
                    not isinstance(variable, graph.Constant)):
                raise Exception("Undeclared input.", variable)
            if variable.fgraph is not self:
                raise Exception("Variable should belong to the FunctionGraph.",
                                variable)
            for node, i in variable.clients:
                if node == 'output':
                    if self.outputs[i] is not variable:
                        raise Exception("Inconsistent clients list.",
                                        variable, self.outputs[i])
                    continue
                if node not in nodes:
                    raise Exception("Client not in FunctionGraph.",
                                    variable, (node, i))
                if node.inputs[i] is not variable:
                    raise Exception("Inconsistent clients list.",
                                    variable, node.inputs[i])

    def __str__(self):
        return "[%s]" % ", ".join(graph.as_string(self.inputs, self.outputs))

    def __repr__(self):
        return self.__str__()

    # clone #
    def clone(self, check_integrity=True):
        """
        Clone the graph and get a memo( a dict )that map old node to new node

        """
        return self.clone_get_equiv(check_integrity)[0]

    def clone_get_equiv(self, check_integrity=True, attach_feature=True):
        """Clone the graph and get a dict that maps old nodes to new ones

        Parameters:
            check_integrity: bool
                Whether to check integrity. Default is True.
            attach_feature: bool
                Whether to attach feature of origin graph to cloned graph.
                Default is True.

        Returns:
            e: FunctionGraph
                Cloned fgraph. Every node in cloned graph is cloned.
            equiv: dict
                A dict that map old node to new node.
        """
        equiv = graph.clone_get_equiv(self.inputs, self.outputs)

        if check_integrity:
            self.check_integrity()
        e = FunctionGraph([equiv[i] for i in self.inputs],
                          [equiv[o] for o in self.outputs],
                          clone=False)
        if check_integrity:
            e.check_integrity()

        if attach_feature:
            for feature in self._features:
                e.attach_feature(feature)
        return e, equiv

    def __getstate__(self):
        """
        This is needed as some features introduce instance methods.
        This is not picklable.

        """
        d = self.__dict__.copy()
        for feature in self._features:
            for attr in getattr(feature, "pickle_rm_attr", []):
                del d[attr]
        # The class Updater take fct as parameter and they are lambda function, so unpicklable.

        # execute_callbacks_times have reference to optimizer, and they can't
        # be pickled as the decorators with parameters aren't pickable.
        if "execute_callbacks_times" in d:
            del d["execute_callbacks_times"]

        return d

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        for feature in self._features:
            if hasattr(feature, "unpickle"):
                feature.unpickle(self)
