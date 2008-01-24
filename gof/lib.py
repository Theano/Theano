
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
           'COp',
           'DualImplOp']


UNCOMPUTED = Keyword("UNCOMPUTED", False)
UNDEFINED = Keyword("UNDEFINED", False)


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

    __slots__ = ['data', 'constant', 'up_to_date']
    
    def __init__(self, x = None, constant = False):
        self.constant = False
        self.set_value(x)
        self.constant = constant
        self.up_to_date = True
        
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

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def perform(self):
        if self.owner:
            self.owner.perform()

    def compute(self):
        if self.owner:
            self.owner.compute()


class PythonOp(Op):
    
    __metaclass__ = ClsInit
    __mode__ = ['build_eval']
    
    nout = 1

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make impl a static method
        impl = cls.impl
        if hasattr(cls.impl, 'im_func'):
            impl = impl.im_func
        cls.impl = staticmethod(impl)
    
    def __new__(cls, *inputs, **kwargs):
        op = Op.__new__(cls)
        op.__init__(*inputs)
        mode = kwargs.get('mode', None) or cls.current_mode()
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

    @classmethod
    def current_mode(cls):
        return cls.__mode__[-1]

    @classmethod
    def set_mode(cls, mode):
        cls.__mode__.append(mode)

    @classmethod
    def build_mode(cls):
        cls.set_mode('build')

    @classmethod
    def eval_mode(cls):
        cls.set_mode('eval')

    @classmethod
    def build_eval_mode(cls):
        cls.set_mode('build_eval')

    @classmethod
    def pop_mode(cls):
        if len(cls.__mode__) == 1:
            raise Exception("There's only one mode left on the stack.")
        else:
            cls.__mode__.pop()
    
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

#     def input_is_up_to_date(self, input):
#         if not input.up_to_date:
#             return False
#         owner = input.owner
#         if owner and isinstance(owner, ext.Viewer):
#             view_map = owner.view_map()
#             if input in view_map:
#                 answer = True
#                 for input2 in view_map[input]:
#                     answer &= owner.input_is_up_to_date(input2)
#                 return answer
#         return True

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
            self.out.set_value(results)
        else:
            assert self.nout == len(results)
            for result, output in zip(results, self.outputs):
                output.set_value(result)

    def compute(self):
        for input in self.inputs:
            if input.data is UNCOMPUTED:
                if input.owner:
                    input.owner.compute()
                else:
                    raise Exception("Uncomputed input: %s in %s" % (input, self))
        self.perform()

    def _impl(self):
        return self.impl(*[input.data for input in self.inputs])
    
    def impl(*args):
        raise NotImplementedError("This op has no implementation.")

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





current_mode = PythonOp.current_mode
set_mode = PythonOp.set_mode
build_mode = PythonOp.build_mode
eval_mode = PythonOp.eval_mode
build_eval_mode = PythonOp.build_eval_mode
pop_mode = PythonOp.pop_mode


class PythonOpt(opt.Optimizer):

    def __init__(self, opt):
        self.opt = opt
    
    def optimize(self, env):
        PythonOp.build_mode()
        self.opt.optimize(env)
        PythonOp.pop_mode()



class DummyOp(Op):
    
    def __init__(self, input):
        Op.__init__(self, [input], [Result()])
        
    def thunk(self):
        return lambda:None


DummyRemover = opt.OpRemover(DummyOp)


# literals_db = {}

# def literal(x):
#     if x in literals_db:
#         return literals_db.get(x)
#     else:
#         ret = PythonR(x, constant = True)
#         liberals_db[x] = ret
#         return ret


















    
class COp(Op):

    def thunk(self):
        cc.compile([self])

    def c_libs(self):
        return []

    def c_imports(self):
        return []

    def c_impl(self):
        raise NotImplementedError("Provide the operation's behavior here.")


class DualImplOp(PythonOp, COp):

    language = 'c'
    supported_languages = 'c', 'python'

    def thunk(self, language = None):
        """
        Returns a thunk that does the operation on the inputs and stores the
        results in the outputs. The language parameter defaults to self.language
        and determines which implementation to use.
        """
        if not language:
            language = self.language
        if language == 'c':
            return COp.thunk(self)
        elif language == 'python':
            return PythonOp.thunk(self)
        elif language == 'all':
            return [self.thunk(lang) for lang in self.supported_languages]
        else:
            raise ValueError("language should be any of %s or 'all', not '%s'" % (self.supported_languages, language))

    def compare_implementations(self,
                                samples,
                                setter = lambda res, v: res.set_value(v),
                                cmp = lambda x, y: x == y):
        """
        Compares the different implementations of this operation on a
        list of input values to verify that they behave the same. The
        input values are put in the Result instances using the setter
        function (defaults to set_value). The output lists are
        compared using the cmp predicate (defaults to ==).
        """
        for sample in samples:
            for input, v in zip(self.inputs, sample):
                input.set_value(v)
            self.thunk('python')()
            
            # we must copy the outputs because they will be overwritten
            results_py = [copy(output).extract() for output in self.outputs]

            # we redo the assignment because the Op might be destructive,
            # in which case the inputs might not be correct anymore
            for input, v in zip(self.inputs, sample):
                input.set_value(v)
            self.thunk('c')()
            results_c = [copy(output).extract() for output in self.outputs]

            assert cmp(results_py, results_c)



