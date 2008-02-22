
from op import Op
from result import is_result, ResultBase
from utils import ClsInit, Keyword, AbstractFunctionError
import opt
import env
import features
import ext


__all__ = [ 'UNDEFINED',
           'current_mode',
           'set_mode',
           'build_mode',
           'eval_mode',
           'build_eval_mode',
           'pop_mode',
           'DummyOp',
           'DummyRemover',
           'PythonOp',
           'PythonOpt',
           'make_static']


UNDEFINED = Keyword("UNDEFINED", False)

def make_static(cls, fname):
    f = getattr(cls, fname)
    if hasattr(f, 'im_func'):
        f = f.im_func
    setattr(cls, fname, staticmethod(f))

def compute_from(nodes, history):
    """Recursively evaluate each node (in a quick & dirty way).

    history (aka inputs) is a set of nodes that need not be [re]computed.

    TODO: make this more correct by building a little graph and executing it.
    The current implementation doesn't take into account any ordering
    constraints imposed by destructors, for example.
    """
    def compute_recursive(node):
        if node and (node not in history):
            if hasattr(node, 'owner'):  #node is storage
                compute_recursive(node.owner)
            else:                       #node is op
                for input in node.inputs:
                    compute_recursive(input)
                node.perform()
            history.add(node)
    for n in nodes:
        compute_recursive(n)

def compute(*nodes):
    """Recursively evaluate each node (in a quick & dirty way)."""
    compute_from(nodes, set())

class ForbidConstantOverwrite(features.Listener, features.Constraint):

    def __init__(self, env):
        self.env = env
        self.bad = set()

    def root_inputs(self, input):
        owner = input.owner
        view_map = owner.view_map()
        if input in view_map:
            answer = []
            for input2 in view_map[input]:
                answer += owner.root_inputs(input2)
            return answer
        else:
            return [input]

    def on_import(self, op):
        for output, inputs in op.destroy_map().items():
            for input in inputs:
                for root_input in self.root_inputs(input):
                    if getattr(root_input, 'constant', False):
                        self.bad.add(op)
                        return

    def on_prune(self, op):
        if op in self.bad:
            self.bad.remove(op)

    def on_rewire(self, clients, r, new_r):
        for op, i in clients:
            self.on_prune(op)
            self.on_import(op)
        
    def validate(self):
        if self.bad:
            raise env.InconsistencyError("The following ops overwrite a constant value: %s" % self.bad)
        else:
            return True



class DestroyHandler(features.Listener, features.Constraint, features.Orderings):
    
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
                if is_result(destroyed):
                    destroyed = [destroyed]
                for input in destroyed:
                    path = self.__path__(input)
                    self.__add_destroyer__(path + [output])

            elif views:
                if is_result(views):
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


class PythonOp(Op):
    
    __metaclass__ = ClsInit

    __require__ = DestroyHandler
    
    nout = 1

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make impl a static method
        cls.set_impl(cls.impl)
        make_static(cls, 'specs')
    
    def __new__(cls, *inputs, **kwargs):
        op = Op.__new__(cls)
        op.__init__(*inputs)
        mode = kwargs.get('mode', None) or current_mode()
        if mode == 'eval':
            op.perform()
            if op.nout == 1:
                return op.out.data
            else:
                return [output.data for output in op.outputs]
        elif mode == 'build_eval':
            op.perform()
        if op.nout == 1:
            return op.out
        else:
            return op.outputs
    
    def __init__(self, *inputs):
        Op.__init__(self, inputs, self.gen_outputs())

    def __validate__(self):
        return all([ is_result(i) for i in self.inputs])
    
    def gen_outputs(self):
        raise NotImplementedError()
    
    def view_map(self): return {}

    def destroy_map(self): return {}

    def root_inputs(self, input):
        owner = input.owner
        if owner:
            view_map = owner.view_map()
            if input in view_map:
                answer = []
                for input2 in view_map[input]:
                    answer += owner.root_inputs(input2)
                return answer
            else:
                return [input]
        else:
            return [input]

    def input_is_up_to_date(self, input):
        answer = True
        for input in self.root_inputs(input):
            answer &= input.up_to_date
        return answer

    def input_is_constant(self, input):
        answer = False
        for input in self.root_inputs(input):
            answer |= input.constant
        return answer

    def check_input(self, input):
        if input.data is None:
            raise ValueError("Uncomputed input: %s in %s" % (input, self))
        if not self.input_is_up_to_date(input):
            raise ValueError("Input is out of date: %s in %s" % (input, self))

    def perform(self):
        exc = set()
        for output, inputs in self.destroy_map().items():
            exc.update(inputs)
            for input in inputs:
                if self.input_is_constant(input):
                    raise ValueError("Input is constant: %s" % input)
        for input in exc:
            self.check_input(input)
            input.up_to_date = False
        for input in self.inputs:
            if input not in exc:
                self.check_input(input)
        if 0:
            #J- why is this try catch here? Leftover debug?
            try:
                results = self._impl()
            except Exception, e:
                print "Error in %s: %s" % (self, e)
                raise
        else:
            results = self._impl()
        if self.nout == 1:
            self.out.set_value(results)
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value(result)

    def _perform(self):
        results = self._impl()
        if self.nout == 1:
            self.out.set_value(results)
#            self.outputs[0].data = results
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value(result)

    def _perform_inplace(self):
        results = self._impl()
        if self.nout == 1:
            self.out.set_value_inplace(results)
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value_inplace(result)


    def _impl(self):
        return self.impl(*[input.data for input in self.inputs])

    @classmethod
    def set_impl(cls, impl):
        make_static(cls, 'impl')
    
    def impl(*args):
        raise NotImplementedError("This op has no implementation.")

    def _specs(self):
        try:
            return self.specs(*[input.spec for input in self.inputs])
        except NotImplementedError:
            raise NotImplementedError("%s cannot infer the specs of its outputs" % self.__class__.__name__)

    def specs(*inputs):
        raise NotImplementedError

    def refresh(self, except_list = []):
        for input in self.inputs:
            input.refresh()
        change = self._propagate_specs()
        if change:
            self.alloc(except_list)
        return change
    
    def _propagate_specs(self):
        specs = self._specs()
        if self.nout == 1:
            specs = [specs]
        change = False
        for output, spec in zip(self.outputs, specs):
            if output.spec != spec:
                output.spec = spec
                change = True
        return change
    
    def alloc(self, except_list = []):
        for output in self.outputs:
            if output not in except_list:
                output.alloc()
    
    __require__ = ForbidConstantOverwrite

    def __copy__(self):
        """
        Copies the inputs list shallowly and copies all the outputs
        because of the one owner per output restriction.
        """
#         new_inputs = copy(op.inputs)
#         # We copy the outputs because they are tied to a single Op.
#         new_outputs = [copy(output) for output in op.outputs]
        build_mode()
        op = self.__class__(*self.inputs)
        pop_mode()
#         op._inputs = new_inputs
#         op._outputs = new_outputs
#         for i, output in enumerate(op.outputs):
#             # We adjust _owner and _index manually since the copies
#             # point to the previous op (self).
#             output._owner = op
#             output._index = i
        if isinstance(op, (list, tuple)):
            return op[0].owner
        return op.owner

__mode__ = ['build_eval']


def current_mode():
    return __mode__[-1]

def set_mode(mode):
    __mode__.append(mode)

def build_mode():
    set_mode('build')

def eval_mode():
    set_mode('eval')

def build_eval_mode():
    set_mode('build_eval')

def pop_mode():
    if len(__mode__) == 1:
        raise Exception("There's only one mode left on the stack.")
    else:
        __mode__.pop()



class PythonOpt(opt.Optimizer):

    def __init__(self, opt):
        self.opt = opt
    
    def optimize(self, env):
        build_mode()
        self.opt.optimize(env)
        pop_mode()



class DummyOp(Op):
    
    def __init__(self, input):
        Op.__init__(self, [input], [ResultBase()])
        
    def thunk(self):
        return lambda:None


DummyRemover = opt.OpRemover(DummyOp)

