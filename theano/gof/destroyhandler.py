"""WRITEME"""
import sys
if sys.version_info[:2] >= (2,5):
    from collections import defaultdict

# otherwise it's implemented in python25.py

import theano
import toolbox
import graph
from theano.gof.python25 import deque

from env import InconsistencyError

class ProtocolError(Exception):
    """WRITEME"""
    pass

class DestroyHandler(object):
    """WRITEME"""

    def __init__(self, do_imports_on_attach=True):
        self.map = {}
        self.do_imports_on_attach=do_imports_on_attach

    def on_attach(self, env):
        dh = self.map.setdefault(env, DestroyHandlerHelper2(do_imports_on_attach=self.do_imports_on_attach))
        dh.on_attach(env)

    def on_detach(self, env):
        self.map[env].on_detach(env)

    def on_import(self, env, op):
        self.map[env].on_import(env, op)

    def on_prune(self, env, op):
        self.map[env].on_prune(env, op)

    def on_change_input(self, env, node, i, r, new_r):
        self.map[env].on_change_input(env, node, i, r, new_r)

    def validate(self, env):
        self.map[env].validate(env)

    def orderings(self, env):
        return self.map[env].orderings(env)


def _dfs_toposort(i, r_out, orderings):
    """
    i - list of inputs
    o - list of outputs
    orderings - dict of additions to the normal inputs and outputs

    Returns nothing.  Raises exception for graph with cycles
    """
    #this is hard-coded reimplementation of functions  from graph.py
    # reason: go faster, prepare for port to C.

    assert isinstance(r_out, (tuple, list, deque))

    # TODO: For more speed - use a defaultdict for the orderings


    iset = set(i)

    if 0:
        def expand(obj):
            rval = []
            if obj not in iset:
                if isinstance(obj, graph.Variable):
                    if obj.owner:
                        rval = [obj.owner]
                if isinstance(obj, graph.Apply):
                    rval = list(obj.inputs)
                rval.extend(orderings.get(obj, []))
            else:
                assert not orderings.get(obj, [])
            return rval

    expand_cache = {}
    # reachable, clients = stack_search( deque(r_out), deps, 'dfs', True)
    start=deque(r_out)
    rval_set = set()
    rval_set.add(id(None))
    rval_list = list()
    expand_inv = {}
    sources = deque()
    while start:
        l = start.pop()# this makes the search dfs
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            if l in iset:
                assert not orderings.get(l, [])
                expand_l = []
            else:
                try:
                    if l.owner:
                        expand_l = [l.owner]
                    else:
                        expand_l = []
                except AttributeError:
                    expand_l = list(l.inputs)
                expand_l.extend(orderings.get(l, []))
            if expand_l:
                for r in expand_l:
                    expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
            else:
                sources.append(l)
            expand_cache[l] = expand_l
    assert len(rval_list) == len(rval_set)-1

    rset = set()
    rlist = []
    while sources:
        node = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in expand_inv.get(node, []):
                expand_cache[client] = [a for a in expand_cache[client] if a is not node]
                if not expand_cache[client]:
                    sources.append(client)

    if len(rlist) != len(rval_list):
        raise ValueError('graph contains cycles')

    #return [o for o in rlist if isinstance(o, graph.Apply)]




def getroot(r, view_i):
    """
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
    """
    for v in view_o.get(r,[]):
        impact.add(v)
        add_impact(v, view_o, impact)

def get_impact(root, view_o):
    impact = set()
    add_impact(root, view_o, impact)
    return impact

def fast_inplace_check(inputs):
    """ Return the variables in inputs that are posible candidate for as inputs of inplace operation

    :type inputs: list
    :param inputs: inputs Variable that you want to use as inplace destination
    """
    env = inputs[0].env
    protected_inputs = [f.protected for f in env._features if isinstance(f,theano.compile.function_module.Supervisor)]
    protected_inputs = sum(protected_inputs,[])#flatten the list
    protected_inputs.extend(env.outputs)

    inputs = [i for i in inputs if
              not isinstance(i,graph.Constant)
              and not env.destroyers(i)
              and i not in protected_inputs]
    return inputs

class DestroyHandlerHelper2(toolbox.Bookkeeper):
    """
    The DestroyHandlerHelper2 class detects when a graph is impossible to evaluate because of
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
        self.env = None
        self.do_imports_on_attach = do_imports_on_attach

    def on_attach(self, env):
        #boilerplate from old implementation
        if self.env is not None:
            raise Exception("A DestroyHandler instance can only serve one Env.")
        for attr in ('destroyers', 'destroy_handler'):
            if hasattr(env, attr):
                raise toolbox.AlreadyThere("DestroyHandler feature is already present or in conflict with another plugin.")

        def get_destroyers_of(r):
            droot, impact, root_destroyer = self.refresh_droot_impact()
            try:
                return [root_destroyer[droot[r]]]
            except Exception:
                return []

        env.destroyers = get_destroyers_of
        env.destroy_handler = self

        self.env = env
        self.destroyers = set() #set of Apply instances with non-null destroy_map
        self.view_i = {}  # variable -> variable used in calculation
        self.view_o = {}  # variable -> set of variables that use this one as a direct input
        #clients: how many times does an apply use a given variable
        self.clients = {} # variable -> apply -> ninputs
        self.stale_droot = True

        self.debug_all_apps = set()
        if self.do_imports_on_attach:
            toolbox.Bookkeeper.on_attach(self, env)

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

    def on_detach(self, env):
        if env is not self.env:
            raise Exception("detaching wrong env", env)
        del self.destroyers
        del self.view_i
        del self.view_o
        del self.clients
        del self.stale_droot
        assert self.env.destroyer_handler is self
        delattr(self.env, 'destroyers')
        delattr(self.env, 'destroy_handler')
        self.env = None

    def on_import(self, env, app):
        """Add Apply instance to set which must be computed"""

        if app in self.debug_all_apps: raise ProtocolError("double import")
        self.debug_all_apps.add(app)
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
            self.view_o.setdefault(i,set()).add(o)

        # update self.clients
        for i, input in enumerate(app.inputs):
            self.clients.setdefault(input, {}).setdefault(app,0)
            self.clients[input][app] += 1

        for i, output in enumerate(app.outputs):
            self.clients.setdefault(output, {})

        self.stale_droot = True

    def on_prune(self, env, app):
        """Remove Apply instance from set which must be computed"""
        if app not in self.debug_all_apps: raise ProtocolError("prune without import")
        self.debug_all_apps.remove(app)

        #UPDATE self.clients
        for i, input in enumerate(set(app.inputs)):
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

    def on_change_input(self, env, app, i, old_r, new_r):
        """app.inputs[i] changed from old_r to new_r """
        if app == 'output':
            # app == 'output' is special key that means Env is redefining which nodes are being
            # considered 'outputs' of the graph.
            pass
        else:
            if app not in self.debug_all_apps: raise ProtocolError("change without import")

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

                    self.view_o.setdefault(new_r,set()).add(output)

        self.stale_droot = True

    def validate(self, env):
        """Return None

        Raise InconsistencyError when
        a) orderings() raises an error
        b) orderings cannot be topologically sorted.

        """
        #print '\nVALIDATE'
        if self.destroyers:
            try:
                ords = self.orderings(env)
            except Exception, e:
                #print 'orderings failed with:', type(e), e.args
                raise
            #print 'orderings:', ords
            try:
                ### graph.io_toposort(env.inputs, env.outputs, ords)
                _dfs_toposort(env.inputs, env.outputs, ords)
            except ValueError, e:
                #print 'not passing.', ords
                if 'cycles' in str(e):
                    raise InconsistencyError("Dependency graph contains cycles")
                else:
                    raise
            #print 'passing...', ords
        else:
            #James's Conjecture:
            #If there are no destructive ops, then there can be no cycles.
            pass
        return True

    def orderings(self, env):
        """Return orderings induced by destructive operations.

        Raise InconsistencyError when
        a) attempting to destroy indestructable variable, or
        b) attempting to destroy a value multiple times, or
        c) an Apply destroys (illegally) one of its own inputs by aliasing

        """
        rval = {}

        if self.destroyers:
            # BUILD DATA STRUCTURES
            # CHECK for multiple destructions during construction of variables

            droot, impact, __ignore = self.refresh_droot_impact()
            #print "droot", droot
            #print "impact", impact
            #print "view_i", self.view_i
            #print "view_o", self.view_o

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
                    tolerated = set(idx1 for idx0, idx1 in tolerate_same
                            if idx0 == destroyed_idx)
                    tolerated.add(destroyed_idx)
                    tolerate_aliased = getattr(app.op, 'destroyhandler_tolerate_aliased', [])
                    ignored = set(idx1 for idx0, idx1 in tolerate_aliased
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
                    root_clients = set()
                    for r in root_impact:
                        assert not [a for a,c in self.clients[r].items() if not c]
                        root_clients.update([a for a,c in self.clients[r].items() if c])
                    root_clients.remove(app)
                    if root_clients:
                        rval[app] = root_clients

        return rval
