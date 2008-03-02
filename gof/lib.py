from copy import copy

from op import Op
from result import is_result, ResultBase
from utils import ClsInit, Keyword, AbstractFunctionError
import opt
import env
import features
import ext

from python25 import all

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
                if node.destroy_map():
                    raise ValueError('compute_from() does not work on nodes with destroy_maps')
                for input in node.inputs:
                    compute_recursive(input)
                node.perform()
            history.add(node)
    for n in nodes:
        compute_recursive(n)

def compute(*nodes):
    """Recursively evaluate each node (in a quick & dirty way)."""
    compute_from(nodes, set())

def root_inputs(input):
    """Return the leaves of a search through consecutive view_map()s"""
    owner = input.owner
    if owner:
        view_map = owner.view_map()
        if input in view_map:
            answer = []
            for input2 in view_map[input]:
                answer.extend(root_inputs(input2))
            return answer
        else:
            return [input]
    else:
        return [input]

class ForbidConstantOverwrite(features.Listener, features.Constraint):

    def __init__(self, env):
        self.env = env
        self.bad = set()

    def on_import(self, op):
        for output, inputs in op.destroy_map().items():
            for input in inputs:
                for root_input in root_inputs(input):
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


class NewPythonOp(Op):

    __env_require__ = DestroyHandler, ForbidConstantOverwrite

    def view_map(self):
        return {}

    def destroy_map(self):
        return {}

class PythonOp(NewPythonOp):
    
    __metaclass__ = ClsInit

    nout = 1

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make impl a static method
        cls.set_impl(cls.impl)
    
    def __new__(cls, *inputs, **kwargs):
        op = NewPythonOp.__new__(cls)
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
        NewPythonOp.__init__(self, inputs, self.gen_outputs())

    def __validate__(self):
        return all([is_result(i) for i in self.inputs])
    
    def gen_outputs(self):
        raise AbstractFunctionError()

    def check_input(self, input):
        def input_is_up_to_date(input):
            answer = True
            for input in root_inputs(input):
                answer &= input.up_to_date
            return answer
        if input.data is None:
            raise ValueError("Uncomputed input: %s in %s" % (input, self))
        if not input_is_up_to_date(input):
            raise ValueError("Input is out of date: %s in %s" % (input, self))

    def perform(self):
        def input_is_constant(input):
            answer = False
            for input in root_inputs(input):
                answer |= input.constant
            return answer
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
            self.out.data = results
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.data = result

    def _perform(self):
        results = self._impl()
        if self.nout == 1:
            self.out.data = results
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.data = result

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



class DummyOp(NewPythonOp):
    
    def __init__(self, input):
        Op.__init__(self, [input], [ResultBase()])
        
    def thunk(self):
        return lambda:None


DummyRemover = opt.OpRemover(DummyOp)


if 0:
    class RefreshableOp(NewPythonOp):

        def _specs(self):
            try:
                return self.specs(*[input.spec for input in self.inputs])
            except NotImplementedError:
                raise NotImplementedError("%s cannot infer the specs of its outputs" % self.__class__.__name__)

        def specs(*inputs):
            raise NotImplementedError

        def refresh(self):
            """Update and allocate outputs if necessary"""

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
