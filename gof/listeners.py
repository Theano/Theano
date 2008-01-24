
from copy import copy

import graph
from env import Env, EnvListener



class PrintListener(EnvListener):

    def __init__(self, env, active = True):
        self.env = env
        self.active = active
        if active:
            print "-- initializing"

    def on_import(self, op):
        if self.active:
            print "-- importing: %s" % graph.as_string(self.env.inputs, op.outputs)

    def on_prune(self, op):
        if self.active:
            print "-- pruning: %s" % graph.as_string(self.env.inputs, op.outputs)

    def on_rewire(self, clients, v, new_v):
        if self.active:
            if v.owner is None:
                vg = v.name
            else:
                vg = graph.as_string(self.env.inputs, v.owner.outputs)
            if new_v.owner is None:
                new_vg = new_v.name
            else:
                new_vg = graph.as_string(self.env.inputs, new_v.owner.outputs)
            print "-- moving from %s to %s" % (vg, new_vg)



class ChangeListener(EnvListener):

    def __init__(self, env):
        self.change = False

    def on_import(self, op):
        self.change = True

    def on_prune(self, op):
        self.change = True

    def on_replace(self, v, new_v):
        self.change = True

    def __call__(self, value = "get"):
        if value == "get":
            return self.change
        else:
            self.change = value



class InstanceFinder(EnvListener, dict):

    def __init__(self, env):
        self.env = env

    def all_bases(self, cls):
        rval = set(cls)
        for base in cls.__bases__:
            rval.add(self.all_bases(base))
        return [cls for cls in rval if issubclass(cls, Op)]

    def on_import(self, op):
        for base in self.all_bases(op.__class__):
            self.setdefault(base, set()).add(op)

    def on_prune(self, op):
        for base in self.all_bases(op.__class__):
            self[base].remove(op)
            if not self[base]:
                del self[base]

    def __query__(self, cls):
        all = [x for x in self.get(cls, [])]
        while all:
            next = all.pop()
            if next in self.env.ops():
                yield next

    def query(self, cls):
        return self.__query__(cls)




# class GraphOrder(EnvListener, dict):

#     def init(self, graph):
#         self.graph = graph

#     def __adjust__(self, op, minimum):
#         if not op or self[op] >= minimum:
#             return
#         self[op] = minimum
#         for output in op.outputs:
#             for client, i in output.clients:
#                 self.__adjust__(client, minimum + 1)

#     def on_import(self, op):
#         order = 0
#         for input in op.inputs:
#             if input not in self.graph.inputs:
#                 order = max(order, self[input.owner] + 1)
#         self[op] = order

#     def on_prune(self, op):
#         del self[op]

#     def on_replace(self, v, new_v):
#         self.__adjust__(new_v.owner, self.get(v.owner, 0))



class SuperFinder(EnvListener, dict):

    def __init__(self, env, props):
        self.env = env
        self.props = props

    def on_import(self, op):
        for prop, value in self.props(op).items():
            self.setdefault(prop, {}).setdefault(value, set()).add(op)

    def on_prune(self, op):
        for prop, value in self.props(op).items():
            self[prop][value].remove(op)
            if len(self[prop][value]) == 0:
                del self[prop][value]
                if len(self[prop]) == 0:
                    del self[prop]

    def __query__(self, order, template):
        all = []
        for prop, value in template.items():
            all += [x for x in self.get(prop, {}).get(value, set())]
        # If not None, the order option requires the order listener to be included in the env under the name 'order'
        if order == 'o->i':
            all.sort(lambda op1, op2: self.env.order[op1].__cmp__(self.env.order[op2]))
        elif order == 'i->o':
            all.sort(lambda op1, op2: self.env.order[op2].__cmp__(self.env.order[op1]))
        while all:
            next = all.pop()
            if next in self.env.ops():
                yield next

    def query(self, **template):
        return self.__query__(None, template)

    def query_downstream(self, **template):
        return self.__query__('i->o', template)

    def query_upstream(self, **template):
        return self.__query__('o->i', template)



# class DupListener(EnvListener):

#     def __init__(self, env):
#         self.to_cid = {}
#         self.to_obj = {}
#         self.env = env
#         for i, input in enumerate(env.inputs):
#             self.to_cid[input] = i
#             self.to_obj[i] = input

#     def init(self, env):
#         self.env = env
#         for i, input in enumerate(env.inputs):
#             self.to_cid[input] = i
#             self.to_obj[i] = input

#     def on_import(self, op):
#         cid = (op.__class__, tuple([self.to_cid[input] for input in op.inputs]))
#         self.to_cid[op] = cid
#         self.to_obj.setdefault(cid, op)
#         for i, output in enumerate(op.outputs):
#             ocid = (i, cid)
#             self.to_cid[output] = ocid
#             self.to_obj.setdefault(ocid, output)

#     def on_prune(self, op):
#         # we don't delete anything
#         return

#     def apply(self, env):
#         if env is not self.env:
#             raise Exception("The DupListener merge optimization can only apply to the env it is listening to.")
#         def fn(op):
#             op2 = self.to_obj[self.to_cid[op]]
#             if op is not op2:
#                 return [(o, o2) for o, o2 in zip(op.outputs, op2.outputs)]
#         env.walk_from_outputs(fn)

#     def __call__(self):
#         self.apply(self.env)



class DestroyHandler(EnvListener):
    
    def __init__(self, env):
        self.parent = {}
        self.children = {}
        self.destroyers = {}
        self.paths = {}
        self.dups = set()
        self.cycles = set()
        self.env = env
        for input in env.inputs:
            self.parent[input] = None
            self.children[input] = set()

    def __path__(self, r):
        path = self.paths.get(r, None)
        if path:
            return path
        rval = [r]
        r = self.parent[r]
        while r:
            rval.append(r)
            r = self.parent[r]
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
        if not keep_going:
            return set()
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
        return getattr(op, 'view_map', lambda x:{})(), \
               getattr(op, 'destroy_map', lambda x:{})()

    def on_import(self, op):
        view_map, destroy_map = self.get_maps(op)

        for input in op.inputs:
            self.parent.setdefault(input, None)
        
        for i, output in enumerate(op.outputs):
            views = view_map.get(output, None)
            destroyed = destroy_map.get(output, None)
            
            if destroyed:
                self.parent[output] = None
                for input in destroyed:
                    path = self.__path__(input)
                    self.__add_destroyer__(path + [output])

            elif views:
                if len(inputs) > 1:
                    raise Exception("Output is a view of too many inputs.")
                self.parent[output] = inputs[0]
                for input in views:
                    self.children[input].add(output)

            else:
                self.parent[output] = None

            self.children[output] = set()

        for output in op.outputs:
            self.__detect_cycles__(output)

#         if destroy_map:
#             print "op: ", op
#             print "ord: ", [str(x) for x in self.orderings()[op]]
#             print

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
            del self.parent[output]
            del self.children[output]


    def __add_destroyer__(self, path):
        foundation = path[0]
        target = path[-1]

        op = target.owner

        destroyers = self.destroyers.setdefault(foundation, {})
        path = destroyers.setdefault(op, path)

        if len(destroyers) > 1:
            self.dups.add(foundation)


    def __remove_destroyer__(self, path):
        foundation = path[0]
        target = path[-1]
        op = target.owner

        destroyers = self.destroyers[foundation]
        del destroyers[op]
        
        if not destroyers:
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
                if r_1 in inputs:
                    assert self.parent[output] == r_1
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
        else:
            return True

    def orderings(self):
        ords = {}
        for foundation, destroyers in self.destroyers.items():
            for op in destroyers.keys():
                ords.setdefault(op, set()).update([user.owner for user in self.__users__(foundation) if user not in op.outputs])
        return ords








