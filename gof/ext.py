
from collections import defaultdict

import graph
import utils
import toolbox

from utils import AbstractFunctionError
from env import InconsistencyError


class DestroyHandler(toolbox.Bookkeeper):

    def __init__(self):
        self.map = {}

    def on_attach(self, env):
        dh = self.map.setdefault(env, DestroyHandlerHelper())
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




class DestroyHandlerHelper(toolbox.Bookkeeper):
    """
    This feature ensures that an env represents a consistent data flow
    when some Ops overwrite their inputs and/or provide "views" over
    some of their inputs. It does so by tracking dependencies between
    data at different stages of the graph and ensuring that
    destructive operations are performed after the destroyed data and
    all of its views have been processed.

    Examples:
     - (x += 1) + (x += 1) -> fails because the first += makes the second
       invalid
     - (a += b) + (c += a) -> succeeds but we have to do c += a first
     - (a += b) + (b += c) + (c += a) -> fails because there's a cyclical
       dependency (no possible ordering)

    This feature allows some optimizations (eg sub += for +) to be applied
    safely.

    @todo
     - x += transpose_view(x) -> fails because the input that is destroyed
       depends on an input that shares the same data
    """

    def __init__(self):
        self.env = None

    def on_attach(self, env):
        
        if self.env is not None:
            raise Exception("A DestroyHandler instance can only serve one Env.")
        for attr in ('destroyers', 'destroy_handler'):
            if hasattr(env, attr):
                raise toolbox.AlreadyThere("DestroyHandler feature is already present or in conflict with another plugin.")
        
        def __destroyers(r):
            ret = self.destroyers.get(r, {})
            ret = ret.keys()
            return ret
        env.destroyers = __destroyers
        env.destroy_handler = self

        self.env = env

        # For an Op that has a view_map, {output : input it is a view of}
        self.parent = {}

        # Reverse mapping of parent: {input : outputs that are a view of it}
        self.children = defaultdict(set)

        # {foundation : {op that destroys it : path }}
        # where foundation is a result such that (not self.parent[result])
        # and path is a sequence of results such that:
        #  * path[0] == foundation
        #  * self.parent[path[i]] == path[i-1]
        #  * path[-1] == output of the Op that is the Destroyer
        self.destroyers = {}

        # Cache for the paths
        self.paths = {}

        ### if any of dups, cycles or illegal is not empty, the env is inconsistent
        # Set of results that are destroyed more than once.
        self.dups = set()
        # Set of sequences of results that represent a dependency cycle, i.e.
        # [a, ... b, ... c, ... a] if our graph is ((a += b) + (b += c) + (c += a))
        self.cycles = set()
        # Set of results that have one Op that destroys them but have been marked
        # indestructible by the user.
        self.illegal = set()
        
        self.seen = set()
        
        toolbox.Bookkeeper.on_attach(self, env)

    def on_detach(self, env):
        del self.parent
        del self.children
        del self.destroyers
        del self.paths
        del self.dups
        del self.cycles
        del self.illegal
        del self.seen
        self.env = None

    def __path__(self, r):
        """
        Returns a path from r to the result that it is ultimately
        a view of, i.e. path such that:
         - path[-1] == r
         - path[i] == parent[path[i+1]]
         - parent[path[0]] == None
        """
        path = self.paths.get(r, None)
        if path:
            return path
        rval = [r]
        r = self.parent.get(r, None) ### ???
        while r:
            rval.append(r)
            r = self.parent.get(r, None)
        rval.reverse()
        for i, x in enumerate(rval):
            self.paths[x] = rval[0:i+1]
        return rval

    def __views__(self, r):
        """
        Returns the set of results (inclusive) such that all the
        results in the set are views of r, directly or indirectly.
        """
        children = self.children[r]
        if not children:
            return [r]
        else:
            rval = [r]
            for child in children:
                rval += self.__views__(child)
        return utils.uniq(rval)

    def __users__(self, r):
        """
        Returns the outputs of all the ops that use r or a view
        of r. In other words, for all ops that have an input that
        is r or a view of r, adds their outputs to the set that
        is returned.
        """
        views = self.__views__(r)
        rval = [] # set()
        for view in views:
            for node, i in view.clients: #self.env.clients(view):
                if node != 'output':
                    rval += node.outputs
        return utils.uniq(rval)

    def __pre__(self, op):
        """
        Returns all results that must be computed prior to computing
        this node.
        """
        rval = set()
        if op is None:
            return rval
        keep_going = False
        for input in op.inputs:
            # Get the basic result the input is a view of.
            foundation = self.__path__(input)[0]
            destroyers = self.destroyers.get(foundation, set())
            if destroyers:
                keep_going = True
            # Is this op destroying the foundation? If yes,
            # all users of the foundation must be computed before
            # we overwrite its contents.
            if op in destroyers:
                users = self.__users__(foundation)
                rval.update(users)
        rval.update(op.inputs) # obviously
        rval.difference_update(op.outputs) # this op's outputs will always be in the users
        return rval

    def __detect_cycles_helper__(self, r, seq):
        """
        Does a depth-first search to find cycles in the graph of
        computation given a directed connection from an op to
        its __pre__ set.
        @type seq: sequence
        @param seq: nodes visited up to now
        @param r: current node
        If r is found in seq, we have a cycle and it is added to
        the set of cycles.
        """
        if r in seq:
            self.cycles.add(tuple(seq[seq.index(r):]))
            return
        pre = self.__pre__(r.owner)
        for r2 in pre:
            self.__detect_cycles_helper__(r2, seq + [r])

    def __detect_cycles__(self, start, just_remove=False):
        """
        Tries to find a cycle containing any of the users of
        start. Prior to doing, we remove all existing cycles
        containing an user of start from the cycles set. If
        just_remove is True, we return immediately after removing the
        cycles.
        """
        users = set(self.__users__(start))
        users.add(start)
        for user in users:
            for cycle in set(self.cycles):
                if user in cycle:
                    self.cycles.remove(cycle)
        if just_remove:
            return
        for user in users:
            self.__detect_cycles_helper__(user, [])

    def get_maps(self, node):
        """
        @return: (vmap, dmap) where:
         - vmap -> {output : [inputs output is a view of]}
         - dmap -> {output : [inputs that are destroyed by the node
                              (and presumably returned as that output)]}
        """
        try: _vmap = node.op.view_map
        except AttributeError, AbstractFunctionError: _vmap = {}
        try: _dmap = node.op.destroy_map
        except AttributeError, AbstractFunctionError: _dmap = {}
        vmap = {}
        for oidx, iidxs in _vmap.items():
            if oidx < 0 or oidx >= node.nout:
                raise ValueError("In %s.view_map: output index out of range" % node.op, oidx, _vmap)
            if any(iidx < 0 or iidx >= node.nin for iidx in iidxs):
                raise ValueError("In %s.view_map: input index out of range" % node.op, iidxs, _vmap)
            vmap[node.outputs[oidx]] = [node.inputs[iidx] for iidx in iidxs]
        dmap = {}
        for oidx, iidxs in _dmap.items():
            if oidx < 0 or oidx >= node.nout:
                raise ValueError("In %s.destroy_map: output index out of range" % node.op, oidx, _dmap)
            if any(iidx < 0 or iidx >= node.nin for iidx in iidxs):
                raise ValueError("In %s.destroy_map: input index out of range" % node.op, iidxs, _dmap)
            dmap[node.outputs[oidx]] = [node.inputs[iidx] for iidx in iidxs]
        return vmap, dmap

    def on_import(self, env, op):
        """
        Recomputes the dependencies and search for inconsistencies given
        that we just added an node to the env.
        """
        
        self.seen.add(op)
        view_map, destroy_map = self.get_maps(op)

        for input in op.inputs:
            self.children.setdefault(input, set())

        for i, output in enumerate(op.outputs):
            views = view_map.get(output, None)
            destroyed = destroy_map.get(output, None)

            if destroyed:
                for input in destroyed:
                    path = self.__path__(input)
                    self.__add_destroyer__(path + [output])

            elif views:
                if len(views) > 1:
                    # This is a limitation of DestroyHandler
                    # TODO: lift it (requires changes everywhere)
                    raise Exception("Output is a view of too many inputs.")
                self.parent[output] = views[0]
                for input in views:
                    self.children[input].add(output)

            self.children[output] = set()

        for output in op.outputs:
            # output has no users and is not in any cycle because it
            # is new. We must however check for cycles from the output
            # eg if we are importing F in F(a += b, a) we will obtain
            # the following cycle: [F.out, +=.out, F.out] because __pre__
            # of +=.out, since it is destructive, must contains all the
            # users of a including F.out. A cycle not involving F.out
            # cannot occur.
            self.__detect_cycles_helper__(output, [])

            
    def on_prune(self, env, op):
        """
        Recomputes the dependencies and searches for inconsistencies to remove
        given that we just removed a node to the env.
        """

        view_map, destroy_map = self.get_maps(op)
        
        if destroy_map:
            # Clean up self.destroyers considering that this op is gone.
            destroyers = []
            for i, input in enumerate(op.inputs):
                destroyers.append(self.destroyers.get(self.__path__(input)[0], {}))
            for destroyer in destroyers:
                path = destroyer.get(op, [])
                if path:
                    self.__remove_destroyer__(path)
                    
        if view_map:
            # Clean the children of the inputs if this Op was a view of any of them.
            for i, input in enumerate(op.inputs):
                self.children[input].difference_update(op.outputs)

        for output in op.outputs:
            try:
                del self.paths[output]
            except:
                pass
            # True means that we are just removing cycles pertaining to this output
            # including cycles involving the users of the output (since there should
            # be no more users after the op is pruned).
            # No new cycles can be added by removing a node.
            self.__detect_cycles__(output, True)

        # Clean up parents and children
        for i, output in enumerate(op.outputs):
            try:
                self.parent[output]
                del self.parent[output]
            except:
                pass
            del self.children[output]
            
        self.seen.remove(op)


    def __add_destroyer__(self, path):
        """
        Processes the information that path[0] is destroyed by path[-1].owner.
        """
        
        foundation = path[0]
        target = path[-1]

        node = target.owner

        destroyers = self.destroyers.setdefault(foundation, {})
        path = destroyers.setdefault(node, path)

#         for foundation, destroyers in self.destroyers.items():
#             for op in destroyers.keys():
#                 ords.setdefault(op, set()).update([user.owner for user in self.__users__(foundation) if user not in op.outputs])
        
        if len(destroyers) > 1:
            self.dups.add(foundation)

        # results marked 'indestructible' must not be destroyed.
        if getattr(foundation.tag, 'indestructible', False) or isinstance(foundation, graph.Constant):
            self.illegal.add(foundation)


    def __remove_destroyer__(self, path):
        """
        Processes the information that path[0] is no longer destroyed by path[-1].owner.
        """

        foundation = path[0]
        target = path[-1]
        node = target.owner

        destroyers = self.destroyers[foundation]
        del destroyers[node]
        
        if not destroyers:
            if foundation in self.illegal:
                self.illegal.remove(foundation)
            del self.destroyers[foundation]
        elif len(destroyers) == 1 and foundation in self.dups:
            self.dups.remove(foundation)


    def on_change_input(self, env, node, i, r, new_r):
        if node != 'output':
            self.on_rewire(env, [(node, i)], r, new_r)

    
    def on_rewire(self, env, clients, r_1, r_2):
        """
        Recomputes the dependencies and searches for inconsistencies to remove
        given that all the clients are moved from r_1 to r_2, clients being
        a list of (node, i) pairs such that node.inputs[i] used to be r_1 and is
        now r_2.
        """
        path_1 = self.__path__(r_1)
        path_2 = self.__path__(r_2)

        # All the affected results one level below the replacement.
        prev = set()
        for op, i in clients:
            prev.update(op.outputs)

        # Here we look at what destroys r_1, directly or indirectly. Since we
        # replace r_1, we must adjust the destroyers. Each destroyer has a path,
        # as described in __path__ and __add_destroyer__. Here is the logic to
        # adjust a path that contains r_1 at index idx and r_prev at index idx+1.
        #  * idx == len(path)-1: do nothing
        #  * r_prev not in prev: do nothing
        #  * else: concatenate path_2 to the part of the path before r_1.
        foundation = path_1[0]
        destroyers = self.destroyers.get(foundation, {}).items()
        for op, path in destroyers:
            if r_1 in path:
                idx = path.index(r_1)
                if idx == len(path)-1 or path[idx+1] not in prev:
                    continue
                self.__remove_destroyer__(path)
                self.__add_destroyer__(path_2 + path[idx+1:])

        # Clean up parents and children
        for op, i in clients:
            view_map, _ = self.get_maps(op)
            for output, inputs in view_map.items():
                if r_2 in inputs:
                    assert self.parent.get(output, None) == r_1
                    self.parent[output] = r_2
                    self.children[r_1].remove(output)
                    self.children[r_2].add(output)
                    for view in self.__views__(r_1):
                        try:
                            del self.paths[view]
                        except:
                            pass
                    for view in self.__views__(r_2):
                        try:
                            del self.paths[view]
                        except:
                            pass

        # Recompute the cycles from both r_1 and r_2.
        self.__detect_cycles__(r_1) # we should really just remove the cycles that have r_1 and a result in prev just before
        self.children.setdefault(r_2, set())
        self.__detect_cycles__(r_2)

    def validate(self, env):
        """
        Raises an L{InconsistencyError} on any of the following conditions:
         - Some results are destroyed by more than one L{Op}
         - There is a cycle of preconditions
         - An L{Op} attempts to destroy an indestructible result.
        """
        if self.dups:
            raise InconsistencyError("The following values are destroyed more than once: %s" % self.dups)
        elif self.cycles:
            raise InconsistencyError("There are cycles: %s" % self.cycles)
        elif self.illegal:
            raise InconsistencyError("Attempting to destroy indestructible results: %s" % self.illegal)
        else:
            return True

    def orderings(self, env):
        """
        Returns a dict of {node : set(nodes that must be computed before it)} according
        to L{DestroyHandler}.
        In particular, all the users of a destroyed result have priority over the
        L{Op} that destroys the result.
        """
        self.validate(env)
        ords = {}
        for foundation, destroyers in self.destroyers.items():
            for op in destroyers.keys():
                ords.setdefault(op, set()).update([user.owner for user in self.__users__(foundation) if user not in op.outputs])
        return ords



def view_roots(r):
    """
    Utility function that returns the leaves of a search through
    consecutive view_map()s.
    """
    owner = r.owner
    if owner is not None:
        try:
            view_map = owner.op.view_map
            view_map = dict([(owner.outputs[o], i) for o, i in view_map.items()])
        except AttributeError:
            return [r]
        if r in view_map:
            answer = []
            for i in view_map[r]:
                answer += view_roots(owner.inputs[i])
            return answer
        else:
            return [r]
    else:
        return [r]

