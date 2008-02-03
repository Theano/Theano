
from op import Op
from result import Result #, HolderResult
from utils import ClsInit, Keyword
import opt
import env
import features
import ext


__all__ = ['UNCOMPUTED',
           'UNDEFINED',
           'current_mode',
           'set_mode',
           'build_mode',
           'eval_mode',
           'build_eval_mode',
           'pop_mode',
           'PythonR',
           'DummyOp',
           'DummyRemover',
           'PythonOp',
           'PythonOpt',
           'make_static']


UNCOMPUTED = Keyword("UNCOMPUTED", False)
UNDEFINED = Keyword("UNDEFINED", False)


def make_static(cls, fname):
    f = getattr(cls, fname)
    if hasattr(f, 'im_func'):
        f = f.im_func
    setattr(cls, fname, staticmethod(f))


class ForbidConstantOverwrite(features.Listener, features.Constraint):

    def __init__(self, env):
        self.env = env
        self.bad = set()

    def root_inputs(self, input):
        owner = input.owner
        if owner and isinstance(owner, ext.Viewer):
            view_map = owner.view_map()
            if input in view_map:
                answer = []
                for input2 in view_map[input]:
                    answer += owner.root_inputs(input2)
                return answer
        else:
            return [input]

    def on_import(self, op):
        if isinstance(op, ext.Destroyer):
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



class PythonR(Result):

    __slots__ = ['data', 'spec', 'constant', 'up_to_date']
    
    def __init__(self, x = UNCOMPUTED, constant = False):
        self.constant = False
        self.set_value(x)
        self.constant = constant
        self.up_to_date = True
        self.spec = None
        
    def set_value(self, value):
        if self.constant:
            raise Exception("This Result is a constant. Its value cannot be changed.")
        if value is None or value is UNCOMPUTED:
            self.data = UNCOMPUTED
        elif isinstance(value, PythonR):
            self.set_value(value.data)
        else:
            self.data = value
        self.up_to_date = True
        self.refresh()

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def refresh(self):
        self.spec = id(self.data)

    def alloc(self):
        raise TypeError("Cannot allocate following this specification.")

    def compute(self):
        """Overrides Op.compute(). Only recurses if self.data is UNCOMPUTED"""
        if self.data is UNCOMPUTED:
            self.owner.compute()


class PythonOp(Op):
    
    __metaclass__ = ClsInit
    
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
        for input in self.inputs:
            assert isinstance(input, PythonR)
    
    def gen_outputs(self):
        return [PythonR() for i in xrange(self.nout)]

    def root_inputs(self, input):
        owner = input.owner
        if owner and isinstance(owner, ext.Viewer):
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
        if input.data is UNCOMPUTED:
            raise ValueError("Uncomputed input: %s in %s" % (input, self))
        if not self.input_is_up_to_date(input):
            raise ValueError("Input is out of date: %s in %s" % (input, self))

    def perform(self):
        exc = set()
        if isinstance(self, ext.Destroyer):
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
        try:
            results = self._impl()
        except Exception, e:
            print "Error in %s: %s" % (self, e)
            raise
        if self.nout == 1:
            self.out.set_value(results)
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value(result)

    def _perform(self):
        results = self._impl()
        if self.nout == 1:
            #self.out.set_value(results)
            self.outputs[0].data = results
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value(result)

    def _perform_like_c(self):
        results = self._impl()
        if self.nout == 1:
            self.outputs[0].data[:] = results
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.data[:] = result


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
        Op.__init__(self, [input], [Result()])
        
    def thunk(self):
        return lambda:None


DummyRemover = opt.OpRemover(DummyOp)

