from collections import defaultdict

import toolbox
import graph

from env import InconsistencyError

class ProtocolError(Exception): pass

class DestroyHandler(toolbox.Bookkeeper):

    def __init__(self):
        self.map = {}

    def on_attach(self, env):
        dh = self.map.setdefault(env, DestroyHandlerHelper2())
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


class DestroyHandlerHelper2(toolbox.Bookkeeper):
    def __init__(self):
        self.env = None

    def on_attach(self, env):
        #boilerplate from old implementation
        if self.env is not None:
            raise Exception("A DestroyHandler instance can only serve one Env.")
        for attr in ('destroyers', 'destroy_handler'):
            if hasattr(env, attr):
                raise toolbox.AlreadyThere("DestroyHandler feature is already present or in conflict with another plugin.")

        def get_destroyers(r):
            d_of = self.get_destroyer_of(r)
            if d_of:
                return [d_of]
            else:
                return []
        env.destroyers = get_destroyers
        env.destroy_handler = self

        self.env = env
        self.destroyers = set() #set of Apply instances with non-null destroy_map
        self.view_i = {}  # result -> result
        self.view_o = {}  # result -> set of results
        #clients: how many times does an apply use a given result
        self.clients = {} # result -> apply -> ninputs  

        self.debug_all_apps = set()
        toolbox.Bookkeeper.on_attach(self, env)

    def build_droot_impact(self):
        droot = {}   # destroyed view + nonview results -> foundation
        impact = {}  # destroyed nonview result -> it + all views of it
        root_destroyer = {} # root -> destroyer apply

        for app in self.destroyers:
            for output_idx, input_idx_list in app.op.destroy_map.items():
                if len(input_idx_list) != 1:
                    raise NotImplementedError()
                input_idx = input_idx_list[0]
                input = app.inputs[input_idx]
                def getroot(r):
                    try: 
                        return getroot(self.view_i[r])
                    except KeyError:
                        return r
                input_root = getroot(input)
                if input_root in droot:
                    raise InconsistencyError("Multiple destroyers of %s" % input_root)
                droot[input_root] = input_root
                root_destroyer[input_root] = app
                impact[input_root] = set([input_root])
                def build_stuff(r):
                    for v in self.view_o.get(r,[]):
                        assert v not in droot
                        droot[v] = input_root
                        impact[input_root].add(v)
                        build_stuff(v)
                build_stuff(input_root)

        return droot, impact, root_destroyer

    def get_destroyer_of(self, r):
        droot, impact, root_destroyer = self.build_droot_impact()
        for root in impact:
            if r in impact[root]:
                return root_destroyer[root]

    def on_detach(self, env):
        if env is not self.env:
            raise Exception("detaching wrong env", env)
        del self.destroyers
        del self.view_i
        del self.view_o
        del self.clients
        assert self.env.destroyer_handler is self
        delattr(self.env, 'destroyers')
        delattr(self.env, 'destroy_handler')
        self.env = None
        
    def on_import(self, env, app):
        """Add Apply instance to set which must be computed"""

        if app in self.debug_all_apps: raise ProtocolError("double import")
        self.debug_all_apps.add(app)

        # If it's a destructive op, add it to our watch list
        if getattr(app.op, 'destroy_map', {}):
            self.destroyers.add(app)

        # add this symbol to the forward and backward maps
        for o_idx, i_idx_list in getattr(app.op, 'view_map', {}).items():
            if len(i_idx_list) > 1:
                #destroying this output invalidates multiple inputs
                raise NotImplementedError()
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

    def on_prune(self, env, app):
        """Remove Apply instance from set which must be computed"""
        if app not in self.debug_all_apps: raise ProtocolError("prune without import")
        self.debug_all_apps.remove(app)

        #UPDATE self.clients
        for i, input in enumerate(app.inputs):
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
                graph.io_toposort(env.inputs, env.outputs, ords)
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
        a) attempting to destroy indestructable result, or
        b) attempting to destroy a value multiple times, or
        c) an Apply destroys (illegally) one of its own inputs by aliasing
        
        """
        rval = {}

        if self.destroyers: 
            # BUILD DATA STRUCTURES
            # CHECK for multiple destructions during construction of variables

            droot, impact, __ignore = self.build_droot_impact()
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
                raise InconsistencyError("Attempting to destroy indestructible results: %s" %
                        illegal_destroy)

            # add destroyed result clients as computational dependencies
            for app in self.destroyers:
                # for each destroyed input...
                for output_idx, input_idx_list in app.op.destroy_map.items():
                    destroyed_idx = input_idx_list[0]
                    destroyed_result = app.inputs[destroyed_idx]
                    root = droot[destroyed_result]
                    root_impact = impact[root]
                    # we generally want to put all clients of things which depend on root
                    # as pre-requisites of app.
                    # But, app is itself one such client!
                    # App will always be a client of the node we're destroying
                    # (destroyed_result, but the tricky thing is when it is also a client of
                    # *another result* viewing on the root.  Generally this is illegal, (e.g.,
                    # add_inplace(x, x.T).  In some special cases though, the in-place op will
                    # actually be able to work properly with multiple destroyed inputs (e.g,
                    # add_inplace(x, x).  An Op that can still work in this case should declare
                    # so via the 'tolerate_same' attribute
                    #
                    # tolerate_same should be a list of pairs of the form 
                    # [(idx0, idx1), (idx0, idx2), ...]
                    # The first element of each pair is the index of a destroyed
                    # variable.
                    # The second element of each pair is the index of a different input where
                    # we will permit exactly the same variable to appear.
                    # For example, add_inplace.tolerate_same might be [(0,1)] if the destroyed
                    # input is also allowed to appear as the second argument.

                    #CHECK FOR INPUT ALIASING
                    # OPT: pre-compute this on import
                    tolerate_same = getattr(app.op, 'tolerate_same', [])
                    tolerated = set(idx1 for idx0, idx1 in tolerate_same
                            if idx0 == destroyed_idx)
                    tolerated.add(destroyed_idx)
                    #print 'tolerated', tolerated
                    for i, input in enumerate(app.inputs):
                        if input in root_impact \
                                and (i not in tolerated or input is not destroyed_result):
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

