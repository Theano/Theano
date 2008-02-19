
from copy import copy
from op import Op
from lib import DummyOp
from result import Result
from features import Listener, Constraint, Orderings
from env import InconsistencyError
from utils import ClsInit
import graph


#TODO: move mark_outputs_as_destroyed to the place that uses this function
#TODO: move Return to where it is used.
__all__ = ['DestroyHandler', 'IONames', 'mark_outputs_as_destroyed']


class IONames:
    """
    Requires assigning a name to each of this Op's inputs and outputs.
    """

    __metaclass__ = ClsInit

    input_names = ()
    output_names = ()
    
    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        for names in ['input_names', 'output_names']:
            if names in dct:
                x = getattr(cls, names)
                if isinstance(x, str):
                    x = [x,]
                    setattr(cls, names, x)
                if isinstance(x, (list, tuple)):
                    x = [a for a in x]
                    setattr(cls, names, x)
                    for i, varname in enumerate(x):
                        if not isinstance(varname, str) or hasattr(cls, varname) or varname in ['inputs', 'outputs']:
                            raise TypeError("In %s: '%s' is not a valid input or output name" % (cls.__name__, varname))
                        # Set an attribute for the variable so we can do op.x to return the input or output named "x".
                        setattr(cls, varname,
                                property(lambda op, type=names.replace('_name', ''), index=i:
                                         getattr(op, type)[index]))
                else:
                    print 'ERROR: Class variable %s::%s is neither list, tuple, or string' % (name, names)
                    raise TypeError, str(names)
            else:
                setattr(cls, names, ())

#     def __init__(self, inputs, outputs, use_self_setters = False):
#         assert len(inputs) == len(self.input_names)
#         assert len(outputs) == len(self.output_names)
#         Op.__init__(self, inputs, outputs, use_self_setters)

    def __validate__(self):
        assert len(self.inputs) == len(self.input_names)
        assert len(self.outputs) == len(self.output_names)
                
    @classmethod
    def n_inputs(cls):
        return len(cls.input_names)
        
    @classmethod
    def n_outputs(cls):
        return len(cls.output_names)
        
    def get_by_name(self, name):
        """
        Returns the input or output which corresponds to the given name.
        """
        if name in self.input_names:
            return self.input_names[self.input_names.index(name)]
        elif name in self.output_names:
            return self.output_names[self.output_names.index(name)]
        else:
            raise AttributeError("No such input or output name for %s: %s" % (self.__class__.__name__, name))



class DestroyHandler(Listener, Constraint, Orderings):
    
    def __init__(self, env):
        self.parent = {}
        self.children = {}
        self.destroyers = {}
        self.paths = {}
        self.dups = set()
        self.cycles = set()
        self.env = env
        for input in env.inputs:
#            self.parent[input] = None
            self.children[input] = set()

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
#         if not keep_going:
#             return set()
        rval.update(op.inputs)
        rval.difference_update(op.outputs)
        return rval

    def __detect_cycles_helper__(self, r, seq):
#        print "!! ", r, seq
        if r in seq:
            self.cycles.add(tuple(seq[seq.index(r):]))
            return
        pre = self.__pre__(r.owner)
        for r2 in pre:
            self.__detect_cycles_helper__(r2, seq + [r])

    def __detect_cycles__(self, start, just_remove=False):
#        print "!!! ", start
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
        vmap = getattr(op, 'view_map',{})
        dmap = getattr(op, 'destoy_map', {})
        return vmap, dmap

    def on_import(self, op):
        view_map, destroy_map = self.get_maps(op)

#         for input in op.inputs:
#             self.parent.setdefault(input, None)
        
        for i, output in enumerate(op.outputs):
            views = view_map.get(output, None)
            destroyed = destroy_map.get(output, None)
            
            if destroyed:
#                self.parent[output] = None
                if isinstance(destroyed, Result):
                    destroyed = [destroyed]
                for input in destroyed:
                    path = self.__path__(input)
                    self.__add_destroyer__(path + [output])

            elif views:
                if isinstance(views, Result):
                    views = [views]
                if len(views) > 1: #views was inputs before?
                    raise Exception("Output is a view of too many inputs.")
                self.parent[output] = views[0]
                for input in views:
                    self.children[input].add(output)

#            else:
#                self.parent[output] = None

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
            try:
                del self.parent[output]
            except:
                pass
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
        else:
            return True

    def orderings(self):
        ords = {}
        for foundation, destroyers in self.destroyers.items():
            for op in destroyers.keys():
                ords.setdefault(op, set()).update([user.owner for user in self.__users__(foundation) if user not in op.outputs])
        return ords


class Return(DummyOp):
    """
    Dummy op which represents the action of returning its input
    value to an end user. It "destroys" its input to prevent any
    other Op to overwrite it.
    """
    def destroy_map(self): return {self.out:[self.inputs[0]]}


def mark_outputs_as_destroyed(outputs):
    return [Return(output).out for output in outputs]

