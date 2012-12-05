"""
Classes and functions for validating graphs that contain view
and inplace operations.
"""
import sys
if sys.version_info[:2] >= (2,5):
    from collections import defaultdict

# otherwise it's implemented in python25.py

import theano
import toolbox
import graph
from theano.gof.python25 import deque
from theano.gof.python25 import OrderedDict
from theano.misc.ordered_set import OrderedSet

from fg import InconsistencyError

class ProtocolError(Exception):
    """Raised when FunctionGraph calls DestroyHandler callbacks in
    an invalid way, for example, pruning or changing a node that has
    never been imported.
    """
    pass

def _contains_cycle(fgraph, orderings):
    """

    fgraph  - the FunctionGraph to check for cycles

    orderings - dictionary specifying extra dependencies besides
                 those encoded in Variable.owner / Apply.inputs

                If orderings[my_apply] == dependencies,

                then my_apply is an Apply instance,
                dependencies is a set of Apply instances,
                and every member of dependencies must be executed
                before my_apply.

                The dependencies are typically used to prevent
                inplace apply nodes from destroying their input before
                other apply nodes with the same input access it.

    Returns True if the graph contains a cycle, False otherwise.
    """

    # These are lists of Variable instances
    inputs = fgraph.inputs
    outputs = fgraph.outputs


    # this is hard-coded reimplementation of functions from graph.py
    # reason: go faster, prepare for port to C.
    # specifically, it could be replaced with a wrapper
    # around graph.io_toposort that returns True iff io_toposort raises
    # a ValueError containing the substring 'cycle'.
    # This implementation is optimized for the destroyhandler and runs
    # slightly faster than io_toposort.

    # this is performance-critical code. it is the largest single-function
    # bottleneck when compiling large graphs.

    assert isinstance(outputs, (tuple, list, deque))

    # TODO: For more speed - use a defaultdict for the orderings
    # (defaultdict runs faster than dict in the case where the key
    # is not in the dictionary, at least in CPython)

    iset = set(inputs)

    # IG: I tried converting parent_counts to use an id for the key,
    # so that the dict would do reference counting on its keys.
    # This caused a slowdown.
    # Separate benchmark tests showed that calling id is about
    # half as expensive as a dictionary access, and that the
    # dictionary also runs slower when storing ids than when
    # storing objects.


    # dict mapping an Apply or Variable instance to the number
    # of its parents (including parents imposed by orderings)
    # that haven't been visited yet
    parent_counts = {}
    # dict mapping an Apply or Variable instance to its children
    node_to_children = {}

    # visitable: A container holding all Variable and Apply instances
    # that can currently be visited according to the graph topology
    # (ie, whose parents have already been visited)
    # TODO: visitable is a fifo_queue. could this run faster if we
    # implement it as a stack rather than a deque?
    # TODO: visitable need not be a fifo_queue, any kind of container
    # that we can throw things into and take things out of quickly will
    # work. is there another kind of container that could run faster?
    # we don't care about the traversal order here as much as we do
    # in io_toposort because we aren't trying to generate an ordering
    # on the nodes
    visitable = deque()

    # IG: visitable could in principle be initialized to fgraph.inputs
    #     + fgraph.orphans... if there were an fgraph.orphans structure.
    #     I tried making one and maintaining it caused a huge slowdown.
    #     This may be because I made it a list, so it would have a
    #     deterministic iteration order, in hopes of using it to speed
    #     up toposort as well.
    #     I think since we need to scan through all variables and nodes
    #     to make parent_counts anyway, it's cheap enough to always
    #     detect orphans at cycle detection / toposort time


    # Pass through all the nodes to build visitable, parent_count, and
    # node_to_children
    for var in fgraph.variables:

        # this is faster than calling get_parents
        owner = var.owner
        if owner:
            parents = [ owner ]
        else:
            parents = []

        # variables don't appear in orderings, so we don't need to worry
        # about that here

        if parents:
            for parent in parents:
                # insert node in node_to_children[r]
                # (if r is not already in node_to_children,
                # intialize it to [])
                node_to_children.setdefault(parent, []).append(var)
            parent_counts[var] = len(parents)
        else:
            visitable.append(var)
            parent_counts[var] = 0

    for a_n in fgraph.apply_nodes:
        parents = list(a_n.inputs)
        # This is faster than conditionally extending
        # IG: I tried using a shared empty_list = [] constructed
        # outside of the for loop to avoid constructing multiple
        # lists, but this was not any faster.
        parents.extend(orderings.get(a_n, []))

        if parents:
            for parent in parents:
                # insert node in node_to_children[r]
                # (if r is not already in node_to_children,
                # intialize it to [])
                node_to_children.setdefault(parent, []).append(a_n)
            parent_counts[a_n] = len(parents)
        else:
            # an Apply with no inputs would be a weird case, but I'm
            # not sure we forbid it
            visitable.append(a_n)
            parent_counts[a_n] = 0

    # at this point,
    # parent_counts.keys() == fgraph.apply_nodes + fgraph.variables



    # Now we actually check for cycles
    # As long as there are nodes that can be visited while respecting
    # the topology, we keep visiting nodes
    # If we run out of visitable nodes and we haven't visited all nodes,
    # then there was a cycle. It blocked the traversal because some
    # node couldn't be visited until one of its descendants had been
    # visited too.
    # This is a standard cycle detection algorithm.

    visited = 0
    while visitable:
        # Since each node is inserted into the visitable queue exactly
        # once, it comes out of the queue exactly once
        # That means we can decrement its children's unvisited parent count
        # and increment the visited node count without double-counting
        node = visitable.popleft()
        visited += 1
        for client in node_to_children.get(node,[]):
            parent_counts[client] -= 1
            # If all of a node's parents have been visited,
            # it may now be visited too
            if not parent_counts[client]:
                visitable.append(client)


    return visited != len(parent_counts)

def getroot(r, view_i):
    """
    TODO: what is view_i ? based on add_impact's docstring, IG is guessing
          it might be a dictionary mapping variables to views, but what is
          a view? In these old docstrings I'm not sure if "view" always
          means "view variable" or if it also sometimes means "viewing
          pattern."
    For views: Return non-view variable which is ultimatly viewed by r.
    For non-views: return self.
    """
    try:
        return getroot(view_i[r], view_i)
    except KeyError:
        return r

def add_impact(r, view_o, impact):
    """
    In opposition to getroot, which finds the variable that is viewed *by* r, this function
    returns all the variables that are views of r.

    :param impact: is a set of variables that are views of r
    :param droot: a dictionary mapping views -> r

    TODO: this docstring is hideously wrong, the function doesn't return anything.
          has droot been renamed to view_o?
          does it add things to the impact argument instead of returning them?
          IG thinks so, based on reading the code. It looks like get_impact
          does what this docstring said this function does.
    """
    for v in view_o.get(r,[]):
        impact.add(v)
        add_impact(v, view_o, impact)

def get_impact(root, view_o):
    impact = OrderedSet()
    add_impact(root, view_o, impact)
    return impact

def fast_inplace_check(inputs):
    """ Return the variables in inputs that are posible candidate for as inputs of inplace operation

    :type inputs: list
    :param inputs: inputs Variable that you want to use as inplace destination
    """
    fgraph = inputs[0].fgraph
    protected_inputs = [f.protected for f in fgraph._features if isinstance(f,theano.compile.function_module.Supervisor)]
    protected_inputs = sum(protected_inputs,[])#flatten the list
    protected_inputs.extend(fgraph.outputs)

    inputs = [i for i in inputs if
              not isinstance(i,graph.Constant)
              and not fgraph.destroyers(i)
              and i not in protected_inputs]
    return inputs

if 0:
    # old, non-incremental version of the DestroyHandler
    class DestroyHandler(toolbox.Bookkeeper):
        """
        The DestroyHandler class detects when a graph is impossible to evaluate because of
        aliasing and destructive operations.

        Several data structures are used to do this.

        When an Op uses its view_map property to declare that an output may be aliased
        to an input, then if that output is destroyed, the input is also considering to be
        destroyed.  The view_maps of several Ops can feed into one another and form a directed graph.
        The consequence of destroying any variable in such a graph is that all variables in the graph
        must be considered to be destroyed, because they could all be refering to the same
        underlying storage.  In the current implementation, that graph is a tree, and the root of
        that tree is called the foundation.  The `droot` property of this class maps from every
        graph variable to its foundation.  The `impact` property maps backward from the foundation
        to all of the variables that depend on it. When any variable is destroyed, this class marks
        the foundation of that variable as being destroyed, with the `root_destroyer` property.
        """

        droot = {}
        """
        destroyed view + nonview variables -> foundation
        """

        impact = {}
        """
        destroyed nonview variable -> it + all views of it
        """

        root_destroyer = {}
        """
        root -> destroyer apply
        """

        def __init__(self, do_imports_on_attach=True):
            self.fgraph = None
            self.do_imports_on_attach = do_imports_on_attach

        def on_attach(self, fgraph):
            """
            When attaching to a new fgraph, check that
            1) This DestroyHandler wasn't already attached to some fgraph
               (its data structures are only set up to serve one)
            2) The FunctionGraph doesn't already have a DestroyHandler.
               This would result in it validating everything twice, causing
               compilation to be slower.

            TODO: WRITEME: what does this do besides the checks?
            """

            ####### Do the checking ###########
            already_there = False
            if self.fgraph is fgraph:
                already_there = True
            if self.fgraph not in [None, fgraph]:
                raise Exception("A DestroyHandler instance can only serve one FunctionGraph. (Matthew 6:24)")
            for attr in ('destroyers', 'destroy_handler'):
                if hasattr(fgraph, attr):
                    already_there = True

            if already_there:
                # FunctionGraph.attach_feature catches AlreadyThere and cancels the attachment
                raise toolbox.AlreadyThere("DestroyHandler feature is already present or in conflict with another plugin.")

            ####### end of checking ############

            def get_destroyers_of(r):
                droot, impact, root_destroyer = self.refresh_droot_impact()
                try:
                    return [root_destroyer[droot[r]]]
                except Exception:
                    return []

            fgraph.destroyers = get_destroyers_of
            fgraph.destroy_handler = self

            self.fgraph = fgraph
            self.destroyers = OrderedSet() #set of Apply instances with non-null destroy_map
            self.view_i = {}  # variable -> variable used in calculation
            self.view_o = {}  # variable -> set of variables that use this one as a direct input
            #clients: how many times does an apply use a given variable
            self.clients = {} # variable -> apply -> ninputs
            self.stale_droot = True

            # IG: It's unclear if this is meant to be included in deployed code. It looks like
            # it is unnecessary if FunctionGraph is working correctly, so I am commenting uses
            # of it (for speed) but leaving the commented code in place so it is easy to restore
            # for debugging purposes.
            # Note: is there anything like the C preprocessor for python? It would be useful to
            # just ifdef these things out
            # self.debug_all_apps = set()
            if self.do_imports_on_attach:
                toolbox.Bookkeeper.on_attach(self, fgraph)

        def refresh_droot_impact(self):
            if self.stale_droot:
                self.droot, self.impact, self.root_destroyer = self._build_droot_impact()
                self.stale_droot = False
            return self.droot, self.impact, self.root_destroyer

        def _build_droot_impact(self):
            droot = {}   # destroyed view + nonview variables -> foundation
            impact = {}  # destroyed nonview variable -> it + all views of it
            root_destroyer = {} # root -> destroyer apply

            for app in self.destroyers:
                for output_idx, input_idx_list in app.op.destroy_map.items():
                    if len(input_idx_list) != 1:
                        raise NotImplementedError()
                    input_idx = input_idx_list[0]
                    input = app.inputs[input_idx]
                    input_root = getroot(input, self.view_i)
                    if input_root in droot:
                        raise InconsistencyError("Multiple destroyers of %s" % input_root)
                    droot[input_root] = input_root
                    root_destroyer[input_root] = app
                    #input_impact = set([input_root])
                    #add_impact(input_root, self.view_o, input_impact)
                    input_impact = get_impact(input_root, self.view_o)
                    for v in input_impact:
                        assert v not in droot
                        droot[v] = input_root

                    impact[input_root] = input_impact
                    impact[input_root].add(input_root)

            return droot, impact, root_destroyer

        def on_detach(self, fgraph):
            if fgraph is not self.fgraph:
                raise Exception("detaching wrong fgraph", fgraph)
            del self.destroyers
            del self.view_i
            del self.view_o
            del self.clients
            del self.stale_droot
            assert self.fgraph.destroyer_handler is self
            delattr(self.fgraph, 'destroyers')
            delattr(self.fgraph, 'destroy_handler')
            self.fgraph = None

        def on_import(self, fgraph, app):
            """Add Apply instance to set which must be computed"""

            #if app in self.debug_all_apps: raise ProtocolError("double import")
            #self.debug_all_apps.add(app)
            #print 'DH IMPORT', app, id(app), id(self), len(self.debug_all_apps)

            # If it's a destructive op, add it to our watch list
            if getattr(app.op, 'destroy_map', {}):
                self.destroyers.add(app)

            # add this symbol to the forward and backward maps
            for o_idx, i_idx_list in getattr(app.op, 'view_map', {}).items():
                if len(i_idx_list) > 1:
                    raise NotImplementedError('destroying this output invalidates multiple inputs', (app.op))
                o = app.outputs[o_idx]
                i = app.inputs[i_idx_list[0]]
                self.view_i[o] = i
                self.view_o.setdefault(i, OrderedSet()).add(o)

            # update self.clients
            for i, input in enumerate(app.inputs):
                self.clients.setdefault(input, {}).setdefault(app,0)
                self.clients[input][app] += 1

            for i, output in enumerate(app.outputs):
                self.clients.setdefault(output, {})

            self.stale_droot = True

        def on_prune(self, fgraph, app):
            """Remove Apply instance from set which must be computed"""
            #if app not in self.debug_all_apps: raise ProtocolError("prune without import")
            #self.debug_all_apps.remove(app)

            #UPDATE self.clients
            for i, input in enumerate(OrderedSet(app.inputs)):
                del self.clients[input][app]

            if getattr(app.op, 'destroy_map', {}):
                self.destroyers.remove(app)

            # Note: leaving empty client dictionaries in the struct.
            # Why? It's a pain to remove them. I think they aren't doing any harm, they will be
            # deleted on_detach().

            #UPDATE self.view_i, self.view_o
            for o_idx, i_idx_list in getattr(app.op, 'view_map', {}).items():
                if len(i_idx_list) > 1:
                    #destroying this output invalidates multiple inputs
                    raise NotImplementedError()
                o = app.outputs[o_idx]
                i = app.inputs[i_idx_list[0]]

                del self.view_i[o]

                self.view_o[i].remove(o)
                if not self.view_o[i]:
                    del self.view_o[i]

            self.stale_droot = True

        def on_change_input(self, fgraph, app, i, old_r, new_r):
            """app.inputs[i] changed from old_r to new_r """
            if app == 'output':
                # app == 'output' is special key that means FunctionGraph is redefining which nodes are being
                # considered 'outputs' of the graph.
                pass
            else:
                #if app not in self.debug_all_apps: raise ProtocolError("change without import")

                #UPDATE self.clients
                self.clients[old_r][app] -= 1
                if self.clients[old_r][app] == 0:
                    del self.clients[old_r][app]

                self.clients.setdefault(new_r,{}).setdefault(app,0)
                self.clients[new_r][app] += 1

                #UPDATE self.view_i, self.view_o
                for o_idx, i_idx_list in getattr(app.op, 'view_map', {}).items():
                    if len(i_idx_list) > 1:
                        #destroying this output invalidates multiple inputs
                        raise NotImplementedError()
                    i_idx = i_idx_list[0]
                    output = app.outputs[o_idx]
                    if i_idx == i:
                        if app.inputs[i_idx] is not new_r:
                            raise ProtocolError("wrong new_r on change")

                        self.view_i[output] = new_r

                        self.view_o[old_r].remove(output)
                        if not self.view_o[old_r]:
                            del self.view_o[old_r]

                        self.view_o.setdefault(new_r, OrderedSet()).add(output)

            self.stale_droot = True

        def validate(self, fgraph):
            """Return None

            Raise InconsistencyError when
            a) orderings() raises an error
            b) orderings cannot be topologically sorted.

            """

            if self.destroyers:
                ords = self.orderings(fgraph)

                if _contains_cycle(fgraph, ords):
                    raise InconsistencyError("Dependency graph contains cycles")
            else:
                #James's Conjecture:
                #If there are no destructive ops, then there can be no cycles.
                pass
            return True

        def orderings(self, fgraph):
            """Return orderings induced by destructive operations.

            Raise InconsistencyError when
            a) attempting to destroy indestructable variable, or
            b) attempting to destroy a value multiple times, or
            c) an Apply destroys (illegally) one of its own inputs by aliasing

            """
            rval = OrderedDict()

            if self.destroyers:
                # BUILD DATA STRUCTURES
                # CHECK for multiple destructions during construction of variables

                droot, impact, __ignore = self.refresh_droot_impact()

                # check for destruction of constants
                illegal_destroy = [r for r in droot if \
                        getattr(r.tag,'indestructible', False) or \
                        isinstance(r, graph.Constant)]
                if illegal_destroy:
                    #print 'destroying illegally'
                    raise InconsistencyError("Attempting to destroy indestructible variables: %s" %
                            illegal_destroy)

                # add destroyed variable clients as computational dependencies
                for app in self.destroyers:
                    # for each destroyed input...
                    for output_idx, input_idx_list in app.op.destroy_map.items():
                        destroyed_idx = input_idx_list[0]
                        destroyed_variable = app.inputs[destroyed_idx]
                        root = droot[destroyed_variable]
                        root_impact = impact[root]
                        # we generally want to put all clients of things which depend on root
                        # as pre-requisites of app.
                        # But, app is itself one such client!
                        # App will always be a client of the node we're destroying
                        # (destroyed_variable, but the tricky thing is when it is also a client of
                        # *another variable* viewing on the root.  Generally this is illegal, (e.g.,
                        # add_inplace(x, x.T).  In some special cases though, the in-place op will
                        # actually be able to work properly with multiple destroyed inputs (e.g,
                        # add_inplace(x, x).  An Op that can still work in this case should declare
                        # so via the 'destroyhandler_tolerate_same' attribute or
                        # 'destroyhandler_tolerate_aliased' attribute.
                        #
                        # destroyhandler_tolerate_same should be a list of pairs of the form
                        # [(idx0, idx1), (idx0, idx2), ...]
                        # The first element of each pair is the input index of a destroyed
                        # variable.
                        # The second element of each pair is the index of a different input where
                        # we will permit exactly the same variable to appear.
                        # For example, add_inplace.tolerate_same might be [(0,1)] if the destroyed
                        # input is also allowed to appear as the second argument.
                        #
                        # destroyhandler_tolerate_aliased is the same sort of list of
                        # pairs.
                        # op.destroyhandler_tolerate_aliased = [(idx0, idx1)] tells the
                        # destroyhandler to IGNORE an aliasing between a destroyed
                        # input idx0 and another input idx1.
                        # This is generally a bad idea, but it is safe in some
                        # cases, such as
                        # - the op reads from the aliased idx1 before modifying idx0
                        # - the idx0 and idx1 are guaranteed not to overlap (e.g.
                        #   they are pointed at different rows of a matrix).
                        #

                        #CHECK FOR INPUT ALIASING
                        # OPT: pre-compute this on import
                        tolerate_same = getattr(app.op, 'destroyhandler_tolerate_same', [])
                        assert isinstance(tolerate_same, list)
                        tolerated = OrderedSet(idx1 for idx0, idx1 in tolerate_same
                                if idx0 == destroyed_idx)
                        tolerated.add(destroyed_idx)
                        tolerate_aliased = getattr(app.op, 'destroyhandler_tolerate_aliased', [])
                        assert isinstance(tolerate_aliased, list)
                        ignored = OrderedSet(idx1 for idx0, idx1 in tolerate_aliased
                                if idx0 == destroyed_idx)
                        #print 'tolerated', tolerated
                        #print 'ignored', ignored
                        for i, input in enumerate(app.inputs):
                            if i in ignored:
                                continue
                            if input in root_impact \
                                    and (i not in tolerated or input is not destroyed_variable):
                                raise InconsistencyError("Input aliasing: %s (%i, %i)"
                                        % (app, destroyed_idx, i))

                        # add the rule: app must be preceded by all other Apply instances that
                        # depend on destroyed_input
                        root_clients = OrderedSet()
                        for r in root_impact:
                            assert not [a for a,c in self.clients[r].items() if not c]
                            root_clients.update([a for a,c in self.clients[r].items() if c])
                        root_clients.remove(app)
                        if root_clients:
                            rval[app] = root_clients

            return rval

class DestroyHandler(toolbox.Bookkeeper):
    """
    The DestroyHandler class detects when a graph is impossible to evaluate
    because of aliasing and destructive operations.

    Several data structures are used to do this.

    An Op can use its view_map property to declare that an output may be
    aliased to an input. If that output is destroyed, the input is also
    considered to be destroyed. The view_maps of several Ops can feed into
    one another and form a directed graph. The consequence of destroying any
    variable in such a graph is that all variables in the graph must be
    considered to be destroyed, because they could all be refering to the
    same underlying storage.

    In the current implementation, that graph is a tree, and the root of that
    tree is called the foundation.

    TODO: why "in the current implementation" ? is there another implementation
          planned?
    TODO: why is the graph a tree? isn't it possible that one variable could
          be aliased to many variables? for example, don't switch and ifelse
          have to do this?

    The original DestroyHandler (if 0'ed out above) computed several data
    structures from scratch each time it was asked to validate the graph.
    Because this happens potentially thousands of times and each graph to
    validate is extremely similar to the previous one, computing the
    data structures from scratch repeatedly was wasteful and resulted in
    high compile times for large graphs.

    This implementation computes the data structures once at initialization
    and then incrementally updates them.

    It is a work in progress. The following data structures have been
    converted to use the incremental strategy:
        <none>

    The following data structures remain to be converted:
        <unknown>
    """


    def __init__(self, do_imports_on_attach=True):
        self.fgraph = None
        self.do_imports_on_attach = do_imports_on_attach

        """maps every variable in the graph to its "foundation" (deepest
        ancestor in view chain)
        TODO: change name to var_to_vroot"""
        self.droot = OrderedDict()

        """maps a variable to all variables that are indirect or direct views of it
         (including itself)
         essentially the inverse of droot
        TODO: do all variables appear in this dict, or only those that are foundations?
        TODO: do only destroyed variables go in here? one old docstring said so
        TODO: rename to x_to_views after reverse engineering what x is"""
        self.impact = OrderedDict()

        """if a var is destroyed, then this dict will map
        droot[var] to the apply node that destroyed var
        TODO: rename to vroot_to_destroyer"""
        self.root_destroyer = OrderedDict()

    def on_attach(self, fgraph):
        """
        When attaching to a new fgraph, check that
            1) This DestroyHandler wasn't already attached to some fgraph
               (its data structures are only set up to serve one)
            2) The FunctionGraph doesn't already have a DestroyHandler.
               This would result in it validating everything twice, causing
               compilation to be slower.

        Give the FunctionGraph instance:
            1) A new method "destroyers(var)"
                TODO: what does this do exactly?
            2) A new attribute, "destroy_handler"
        TODO: WRITEME: what does this do besides the checks?
        """

        ####### Do the checking ###########
        already_there = False
        if self.fgraph is fgraph:
            already_there = True
        if self.fgraph is not None:
            raise Exception("A DestroyHandler instance can only serve one FunctionGraph. (Matthew 6:24)")
        for attr in ('destroyers', 'destroy_handler'):
            if hasattr(fgraph, attr):
                already_there = True

        if already_there:
            # FunctionGraph.attach_feature catches AlreadyThere and cancels the attachment
            raise toolbox.AlreadyThere("DestroyHandler feature is already present or in conflict with another plugin.")

        ####### Annotate the FunctionGraph ############

        def get_destroyers_of(r):
            droot, impact, root_destroyer = self.refresh_droot_impact()
            try:
                return [root_destroyer[droot[r]]]
            except Exception:
                return []

        fgraph.destroyers = get_destroyers_of
        fgraph.destroy_handler = self

        self.fgraph = fgraph
        self.destroyers = OrderedSet() #set of Apply instances with non-null destroy_map
        self.view_i = OrderedDict()  # variable -> variable used in calculation
        self.view_o = OrderedDict()  # variable -> set of variables that use this one as a direct input
        #clients: how many times does an apply use a given variable
        self.clients = OrderedDict() # variable -> apply -> ninputs
        self.stale_droot = True

        self.debug_all_apps = OrderedSet()
        if self.do_imports_on_attach:
            toolbox.Bookkeeper.on_attach(self, fgraph)

    def refresh_droot_impact(self):
        """
        Makes sure self.droot, self.impact, and self.root_destroyer are
        up to date, and returns them.
        (see docstrings for these properties above)
        """
        if self.stale_droot:
            droot = OrderedDict()   # destroyed view + nonview variables -> foundation
            impact = OrderedDict()  # destroyed nonview variable -> it + all views of it
            root_destroyer = OrderedDict() # root -> destroyer apply

            for app in self.destroyers:
                for output_idx, input_idx_list in app.op.destroy_map.items():
                    if len(input_idx_list) != 1:
                        raise NotImplementedError()
                    input_idx = input_idx_list[0]
                    input = app.inputs[input_idx]
                    input_root = getroot(input, self.view_i)
                    if input_root in droot:
                        raise InconsistencyError("Multiple destroyers of %s" % input_root)
                    droot[input_root] = input_root
                    root_destroyer[input_root] = app
                    input_impact = get_impact(input_root, self.view_o)
                    for v in input_impact:
                        assert v not in droot
                        droot[v] = input_root

                    impact[input_root] = input_impact
                    impact[input_root].add(input_root)
            self.droot, self.impact, self.root_destroyer = droot, impact, root_destroyer
            self.stale_droot = False
        return self.droot, self.impact, self.root_destroyer

    def on_detach(self, fgraph):
        if fgraph is not self.fgraph:
            raise Exception("detaching wrong fgraph", fgraph)
        del self.destroyers
        del self.view_i
        del self.view_o
        del self.clients
        del self.stale_droot
        assert self.fgraph.destroyer_handler is self
        delattr(self.fgraph, 'destroyers')
        delattr(self.fgraph, 'destroy_handler')
        self.fgraph = None

    def on_import(self, fgraph, app):
        """Add Apply instance to set which must be computed"""

        if app in self.debug_all_apps: raise ProtocolError("double import")
        self.debug_all_apps.add(app)
        #print 'DH IMPORT', app, id(app), id(self), len(self.debug_all_apps)

        # If it's a destructive op, add it to our watch list
        if getattr(app.op, 'destroy_map', OrderedDict()):
            self.destroyers.add(app)

        # add this symbol to the forward and backward maps
        for o_idx, i_idx_list in getattr(app.op, 'view_map', OrderedDict()).items():
            if len(i_idx_list) > 1:
                raise NotImplementedError('destroying this output invalidates multiple inputs', (app.op))
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]
            self.view_i[o] = i
            self.view_o.setdefault(i, OrderedSet()).add(o)

        # update self.clients
        for i, input in enumerate(app.inputs):
            self.clients.setdefault(input, OrderedDict()).setdefault(app,0)
            self.clients[input][app] += 1

        for i, output in enumerate(app.outputs):
            self.clients.setdefault(output, OrderedDict())

        self.stale_droot = True

    def on_prune(self, fgraph, app):
        """Remove Apply instance from set which must be computed"""
        if app not in self.debug_all_apps: raise ProtocolError("prune without import")
        self.debug_all_apps.remove(app)

        #UPDATE self.clients
        for i, input in enumerate(OrderedSet(app.inputs)):
            del self.clients[input][app]

        if getattr(app.op, 'destroy_map', OrderedDict()):
            self.destroyers.remove(app)

        # Note: leaving empty client dictionaries in the struct.
        # Why? It's a pain to remove them. I think they aren't doing any harm, they will be
        # deleted on_detach().

        #UPDATE self.view_i, self.view_o
        for o_idx, i_idx_list in getattr(app.op, 'view_map', OrderedDict()).items():
            if len(i_idx_list) > 1:
                #destroying this output invalidates multiple inputs
                raise NotImplementedError()
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]

            del self.view_i[o]

            self.view_o[i].remove(o)
            if not self.view_o[i]:
                del self.view_o[i]

        self.stale_droot = True

    def on_change_input(self, fgraph, app, i, old_r, new_r):
        """app.inputs[i] changed from old_r to new_r """
        if app == 'output':
            # app == 'output' is special key that means FunctionGraph is redefining which nodes are being
            # considered 'outputs' of the graph.
            pass
        else:
            if app not in self.debug_all_apps: raise ProtocolError("change without import")

            #UPDATE self.clients
            self.clients[old_r][app] -= 1
            if self.clients[old_r][app] == 0:
                del self.clients[old_r][app]

            self.clients.setdefault(new_r, OrderedDict()).setdefault(app,0)
            self.clients[new_r][app] += 1

            #UPDATE self.view_i, self.view_o
            for o_idx, i_idx_list in getattr(app.op, 'view_map', OrderedDict()).items():
                if len(i_idx_list) > 1:
                    #destroying this output invalidates multiple inputs
                    raise NotImplementedError()
                i_idx = i_idx_list[0]
                output = app.outputs[o_idx]
                if i_idx == i:
                    if app.inputs[i_idx] is not new_r:
                        raise ProtocolError("wrong new_r on change")

                    self.view_i[output] = new_r

                    self.view_o[old_r].remove(output)
                    if not self.view_o[old_r]:
                        del self.view_o[old_r]

                    self.view_o.setdefault(new_r, OrderedSet()).add(output)

        self.stale_droot = True

    def validate(self, fgraph):
        """Return None

        Raise InconsistencyError when
        a) orderings() raises an error
        b) orderings cannot be topologically sorted.

        """

        if self.destroyers:
            ords = self.orderings(fgraph)

            if _contains_cycle(fgraph, ords):
                raise InconsistencyError("Dependency graph contains cycles")
        else:
            #James's Conjecture:
            #If there are no destructive ops, then there can be no cycles.
            pass
        return True

    def orderings(self, fgraph):
        """Return orderings induced by destructive operations.

        Raise InconsistencyError when
        a) attempting to destroy indestructable variable, or
        b) attempting to destroy a value multiple times, or
        c) an Apply destroys (illegally) one of its own inputs by aliasing

        """
        rval = OrderedDict()

        if self.destroyers:
            # BUILD DATA STRUCTURES
            # CHECK for multiple destructions during construction of variables

            droot, impact, __ignore = self.refresh_droot_impact()

            # check for destruction of constants
            illegal_destroy = [r for r in droot if \
                    getattr(r.tag,'indestructible', False) or \
                    isinstance(r, graph.Constant)]
            if illegal_destroy:
                raise InconsistencyError("Attempting to destroy indestructible variables: %s" %
                        illegal_destroy)

            # add destroyed variable clients as computational dependencies
            for app in self.destroyers:
                # for each destroyed input...
                for output_idx, input_idx_list in app.op.destroy_map.items():
                    destroyed_idx = input_idx_list[0]
                    destroyed_variable = app.inputs[destroyed_idx]
                    root = droot[destroyed_variable]
                    root_impact = impact[root]
                    # we generally want to put all clients of things which depend on root
                    # as pre-requisites of app.
                    # But, app is itself one such client!
                    # App will always be a client of the node we're destroying
                    # (destroyed_variable, but the tricky thing is when it is also a client of
                    # *another variable* viewing on the root.  Generally this is illegal, (e.g.,
                    # add_inplace(x, x.T).  In some special cases though, the in-place op will
                    # actually be able to work properly with multiple destroyed inputs (e.g,
                    # add_inplace(x, x).  An Op that can still work in this case should declare
                    # so via the 'destroyhandler_tolerate_same' attribute or
                    # 'destroyhandler_tolerate_aliased' attribute.
                    #
                    # destroyhandler_tolerate_same should be a list of pairs of the form
                    # [(idx0, idx1), (idx0, idx2), ...]
                    # The first element of each pair is the input index of a destroyed
                    # variable.
                    # The second element of each pair is the index of a different input where
                    # we will permit exactly the same variable to appear.
                    # For example, add_inplace.tolerate_same might be [(0,1)] if the destroyed
                    # input is also allowed to appear as the second argument.
                    #
                    # destroyhandler_tolerate_aliased is the same sort of list of
                    # pairs.
                    # op.destroyhandler_tolerate_aliased = [(idx0, idx1)] tells the
                    # destroyhandler to IGNORE an aliasing between a destroyed
                    # input idx0 and another input idx1.
                    # This is generally a bad idea, but it is safe in some
                    # cases, such as
                    # - the op reads from the aliased idx1 before modifying idx0
                    # - the idx0 and idx1 are guaranteed not to overlap (e.g.
                    #   they are pointed at different rows of a matrix).
                    #

                    #CHECK FOR INPUT ALIASING
                    # OPT: pre-compute this on import
                    tolerate_same = getattr(app.op, 'destroyhandler_tolerate_same', [])
                    assert isinstance(tolerate_same, list)
                    tolerated = OrderedSet(idx1 for idx0, idx1 in tolerate_same
                            if idx0 == destroyed_idx)
                    tolerated.add(destroyed_idx)
                    tolerate_aliased = getattr(app.op, 'destroyhandler_tolerate_aliased', [])
                    assert isinstance(tolerate_aliased, list)
                    ignored = OrderedSet(idx1 for idx0, idx1 in tolerate_aliased
                            if idx0 == destroyed_idx)
                    #print 'tolerated', tolerated
                    #print 'ignored', ignored
                    for i, input in enumerate(app.inputs):
                        if i in ignored:
                            continue
                        if input in root_impact \
                                and (i not in tolerated or input is not destroyed_variable):
                            raise InconsistencyError("Input aliasing: %s (%i, %i)"
                                    % (app, destroyed_idx, i))

                    # add the rule: app must be preceded by all other Apply instances that
                    # depend on destroyed_input
                    root_clients = OrderedSet()
                    for r in root_impact:
                        assert not [a for a, c in self.clients[r].items() if not c]
                        root_clients.update([a for a, c in self.clients[r].items() if c])
                    root_clients.remove(app)
                    if root_clients:
                        rval[app] = root_clients

        return rval
