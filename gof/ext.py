
from features import Listener, Constraint, Orderings, Tool
from utils import AbstractFunctionError

from copy import copy
from env import InconsistencyError


__all__ = ['Destroyer',
           'Viewer',
           'DestroyHandler',
           ]



class DestroyHandler(Listener, Constraint, Orderings, Tool):
    
    def __init__(self, env):
        self.parent = {}
        self.children = {}
        self.destroyers = {}
        self.paths = {}
        self.dups = set()
        self.cycles = set()
        self.illegal = set()
        self.env = env
        self.seen = set()
        for input in env.inputs:
            self.children[input] = set()

    def publish(self):
        def __destroyers():
            ret = self.destroyers.get(foundation, set())
            ret = ret.keys()
            return ret
        self.env.destroyers = __destroyers

    def __path__(self, r):
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
        children = self.children[r]
        if not children:
            return set([r])
        else:
            rval = set([r])
            for child in children:
                rval.update(self.__views__(child))
        return rval

    def __users__(self, r):
        views = self.__views__(r)
        rval = set()
        for view in views:
            for op, i in self.env.clients(view):
                if op in self.seen:
                    rval.update(op.outputs)
        return rval

    def __pre__(self, op):
        rval = set()
        if op is None:
            return rval
        keep_going = False
        for input in op.inputs:
            foundation = self.__path__(input)[0]
            destroyers = self.destroyers.get(foundation, set())
            if destroyers:
                keep_going = True
            if op in destroyers:
                users = self.__users__(foundation)
                rval.update(users)
        rval.update(op.inputs)
        rval.difference_update(op.outputs)
        return rval

    def __detect_cycles_helper__(self, r, seq):
        if r in seq:
            self.cycles.add(tuple(seq[seq.index(r):]))
            return
        pre = self.__pre__(r.owner)
        for r2 in pre:
            self.__detect_cycles_helper__(r2, seq + [r])

    def __detect_cycles__(self, start, just_remove=False):
        users = self.__users__(start)
        users.add(start)
        for user in users:
            for cycle in copy(self.cycles):
                if user in cycle:
                    self.cycles.remove(cycle)
        if just_remove:
            return
        for user in users:
            self.__detect_cycles_helper__(user, [])

    def get_maps(self, op):
        try: vmap = op.view_map()
        except AttributeError, AbstractFunctionError: vmap = {}
        try: dmap = op.destroy_map()
        except AttributeError, AbstractFunctionError: dmap = {}
        return vmap, dmap

    def on_import(self, op):
        
        self.seen.add(op)
        view_map, destroy_map = self.get_maps(op)

        for i, output in enumerate(op.outputs):
            views = view_map.get(output, None)
            destroyed = destroy_map.get(output, None)
            
            if destroyed:
                for input in destroyed:
                    path = self.__path__(input)
                    self.__add_destroyer__(path + [output])

            elif views:
                if len(views) > 1:
                    raise Exception("Output is a view of too many inputs.")
                self.parent[output] = views[0]
                for input in views:
                    self.children[input].add(output)

            self.children[output] = set()

        for output in op.outputs:
            self.__detect_cycles__(output)

            
    def on_prune(self, op):
        view_map, destroy_map = self.get_maps(op)
        
        if destroy_map:
            destroyers = []
            for i, input in enumerate(op.inputs):
                destroyers.append(self.destroyers.get(self.__path__(input)[0], {}))
            for destroyer in destroyers:
                path = destroyer.get(op, [])
                if path:
                    self.__remove_destroyer__(path)
                    
        if view_map:
            for i, input in enumerate(op.inputs):
                self.children[input].difference_update(op.outputs)

        for output in op.outputs:
            try:
                del self.paths[output]
            except:
                pass
            self.__detect_cycles__(output, True)

        for i, output in enumerate(op.outputs):
            try:
                self.parent[output]
                del self.parent[output]
            except:
                pass
            del self.children[output]
            
        self.seen.remove(op)


    def __add_destroyer__(self, path):
        foundation = path[0]
        target = path[-1]

        op = target.owner

        destroyers = self.destroyers.setdefault(foundation, {})
        path = destroyers.setdefault(op, path)

        if len(destroyers) > 1:
            self.dups.add(foundation)

        if getattr(foundation, 'indestructible', False):
            self.illegal.add(foundation)


    def __remove_destroyer__(self, path):
        foundation = path[0]
        target = path[-1]
        op = target.owner

        destroyers = self.destroyers[foundation]
        del destroyers[op]
        
        if not destroyers:
            if foundation in self.illegal:
                self.illegal.remove(foundation)
            del self.destroyers[foundation]
        elif len(destroyers) == 1 and foundation in self.dups:
            self.dups.remove(foundation)


    def on_rewire(self, clients, r_1, r_2):
        path_1 = self.__path__(r_1)
        path_2 = self.__path__(r_2)

        prev = set()
        for op, i in clients:
            prev.update(op.outputs)
        
        foundation = path_1[0]
        destroyers = self.destroyers.get(foundation, {}).items()
        for op, path in destroyers:
            if r_1 in path:
                idx = path.index(r_1)
                self.__remove_destroyer__(path)
                if not (idx > 0 and path[idx - 1] in prev):
                    continue
                index = path.index(r_1)
                new_path = path_2 + path[index+1:]
                self.__add_destroyer__(new_path)

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

        self.__detect_cycles__(r_1)
        self.__detect_cycles__(r_2)

    def validate(self):
        if self.dups:
            raise InconsistencyError("The following values are destroyed more than once: %s" % self.dups)
        elif self.cycles:
            raise InconsistencyError("There are cycles: %s" % self.cycles)
        elif self.illegal:
            raise InconsistencyError("Attempting to destroy indestructible results: %s" % self.illegal)
        else:
            return True

    def orderings(self):
        ords = {}
        for foundation, destroyers in self.destroyers.items():
            for op in destroyers.keys():
                ords.setdefault(op, set()).update([user.owner for user in self.__users__(foundation) if user not in op.outputs])
        return ords


class Destroyer:

    def destroyed_inputs(self):
        raise AbstractFunctionError()

    def destroy_map(self):
        # compatibility
        return {self.out: self.destroyed_inputs()}
    
    __env_require__ = DestroyHandler



class Viewer:

    def view_map(self):
        raise AbstractFunctionError()

    def view_roots(self, output):
        def helper(r):
            """Return the leaves of a search through consecutive view_map()s"""
            owner = r.owner
            if owner is not None:
                try:
                    view_map = owner.view_map()
                except AttributeError, AbstractFunctionError:
                    return []
                if r in view_map:
                    answer = []
                    for r2 in view_map[r]:
                        answer.extend(helper(r2))
                    return answer
                else:
                    return [r]
            else:
                return [r]
        return helper(output)

