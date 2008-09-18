"""Convenient driver of graph construction, optimization, and linking."""


import numpy
import gof
import sys
from copy import copy
import tensor_opt


def check_equal(x, y):
    x, y = x[0], y[0]
    if isinstance(x, numpy.ndarray) or isinstance(y, numpy.ndarray):
        if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
            raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
    else:
        if x != y:
                raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})

def infer_reuse_pattern(env, outputs_to_disown):
    do_not_reuse = list()
    seen = set()
    def walk(r):
        if r.owner is None or r in seen:
            return
        seen.add(r)
        do_not_reuse.append(r)
        node = r.owner
        op = node.op
        dmap = op.destroy_map if hasattr(op, 'destroy_map') else {}
        vmap = op.view_map if hasattr(op, 'view_map') else {}
        for l in dmap.values() + vmap.values():
            for i in l:
                walk(node.inputs[i])
    for output in outputs_to_disown:
        walk(output)
    return do_not_reuse

predefined_linkers = {
    'py'   : gof.PerformLinker(),
    'c'    : gof.CLinker(),
    'c|py' : gof.OpWiseCLinker(),
    'c&py' : gof.DualLinker(checker = check_equal)
    }

default_linker = 'c|py'


predefined_optimizers = {
    None    : lambda env: None,
    'merge' : gof.MergeOptimizer(),
    'math'  : gof.MergeOptMerge(tensor_opt.math_optimizer)
    }

default_optimizer = 'merge'


class Mode(object):
    
    def __init__(self, linker = default_linker, optimizer = default_optimizer):
        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, str) or linker is None:
            linker = predefined_linkers[linker]
        self.linker = linker
        if isinstance(optimizer, str) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        self.optimizer = optimizer

    def __str__(self):
        return "Mode(linker = %s, optimizer = %s)" % (self.provided_linker, self.provided_optimizer)


predefined_modes = {
    'SANITY_CHECK'            : Mode('c&py', 'math'),
    'FAST_COMPILE'            : Mode('py', None),
    'FAST_RUN'                : Mode('c|py', None), #'math'),
    'EXPENSIVE_OPTIMIZATIONS' : Mode('c|py', 'math')
    }

default_mode = 'FAST_RUN'




class SymbolicInput(object):

    def __init__(self, result, name=None, update=None, mutable=None, autoname=True):
        """
        result: a Result instance. 
            This will be assigned a value before running the function,
            not computed from its owner.
        
        name: Any type. (If autoname=True, defaults to result.name). 
            If name is a valid Python identifier, this input can be set by kwarg, and its value
            can be accessed by self.<name>.

        update: Result instance
            value (see previous) will be replaced with this expression result after each function call.
            If update is None, the update will be the default value of the input.
            has_default must be True.

        mutable: Bool (default: False if update is None, True if update is not None)
            True: permit the compiled function to modify the python object being passed as the input
            False: do not permit the compiled function to modify the python object being passed as the input.

        autoname: Bool (default: True)
            See the name option.
        """
        self.result = result
        self.name = result.name if (autoname and name is None) else name
        #self.has_default = has_default if (has_default is not None) else (update is not None)
        self.update = update # if (update or not value) else gof.Constant(result.type, value)
        self.mutable = mutable if (mutable is not None) else (update is not None)
#         if self.update is not None and not self.has_default:
#             raise ValueError("If update is not None, the input must accept a default value.")

    def __str__(self):
        if self.update:
            return "In(%s -> %s)" % (self.result, self.update)
        else:
            return "In(%s)" % self.result

    def __repr__(self):
        return str(self)


class SymbolicInputKit(object):

    def __init__(self, name):
        self.name = name
        self.sinputs = []
        self.results = []

    def add_input(self, sinput):
        self.sinputs.append(sinput)
        self.results.append(sinput.result)

    def distribute(self, value, indices, containers):
        raise NotImplementedError

    def complete(self, inputs):
        ret = []
        for input in inputs:
            try:
                i = self.results.index(input)
                ret.append((i, self.sinputs[i]))
            except ValueError:
                pass
        ret.sort()
        return zip(*ret)


class SymbolicOutput(object):
    
    def __init__(self, result, borrow=False):
        """
        borrow: set this to True to indicate that a reference to
                function's internal storage may be returned. A value
                returned for this output might be clobbered by running
                the function again, but the function might be faster.
        """
        self.result = result
        self.borrow = borrow

Out = SymbolicOutput



class Supervisor:

    def __init__(self, protected):
        self.protected = list(protected)

    def validate(self, env):
        if not hasattr(env, 'destroyers'):
            return True
        for r in self.protected + list(env.outputs):
            if env.destroyers(r):
                raise gof.InconsistencyError("Trying to destroy a protected Result.")


def std_env(input_specs, output_specs, accept_inplace = False):
    orig_inputs = [spec.result for spec in input_specs]
    updates = [spec.update for spec in input_specs if spec.update]
    orig_outputs = [spec.result for spec in output_specs] + updates

    inputs, outputs = gof.graph.clone(orig_inputs, orig_outputs)
    env = gof.env.Env(inputs, outputs)

    for node in env.nodes:
        if getattr(node.op, 'destroy_map', None):
            if not accept_inplace:
                raise TypeError("Graph must not contain inplace operations", node)
            else:
                env.extend(gof.DestroyHandler())
                break

    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not spec.mutable))
    return env, map(SymbolicOutput, updates)


class FunctionMaker(object):

    @staticmethod
    def wrap_in(input):
        if isinstance(input, (SymbolicInput, SymbolicInputKit)):
            return input
        elif isinstance(input, gof.Result):
            return SymbolicInput(input)
        elif isinstance(input, (list, tuple)):
            if len(input) == 2:
                return SymbolicInput(input[0], update = input[1])
            else:
                raise TypeError("Expected two elements in the list or tuple.", input)
        else:
            raise TypeError("Unknown input type:", type(input), input)

    @staticmethod
    def expand_in(input, inputs):
        if isinstance(input, SymbolicInputKit):
            return input.complete(inputs)
        elif isinstance(input, SymbolicInput):
            return [None, [input]]

    @staticmethod
    def wrap_out(output):
        if isinstance(output, SymbolicOutput):
            return output
        elif isinstance(output, gof.Result):
            return SymbolicOutput(output)
        else:
            raise TypeError("Unknown output type:", type(output), output)

    def __init__(self, inputs, outputs, mode = 'FAST_RUN', accept_inplace = True):

        # Handle the case where inputs and/or outputs is a single Result (not in a list)
        unpack_single = False
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.result for o in outputs])
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices])

        # make the env
        env, additional_outputs = std_env(expanded_inputs, outputs, accept_inplace)
        self.env = env

        mode = predefined_modes.get(mode, mode)
        optimizer, linker = mode.optimizer, copy(mode.linker)

        # optimize the env
        optimizer(env)

        # initialize the linker
        if not hasattr(linker, 'accept'):
            raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
                             "or one of %s" % predefined_linkers.keys())

        no_borrow = [output for output, spec in zip(env.outputs, outputs+additional_outputs) if not spec.borrow]
        if not no_borrow:
            self.linker = linker.accept(env)
        else:
            self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, no_borrow))
        
        self.indices = indices
        self.inputs = inputs
        self.expanded_inputs = expanded_inputs
        self.outputs = outputs
        self.unpack_single = unpack_single

    def create(self, defaults = [], profiler = None):
        input_storage = []
        _defaults = []
        for (input, indices, results), default in zip(self.indices, defaults):
            if isinstance(default, gof.Filter):
                if indices is not None:
                    raise TypeError("Cannot take a Filter instance as default for a SymbolicInputKit.")
                input_storage.append(default.storage)
                continue
            if indices is None:
                input_storage.append([None])
            else:
                input_storage += [[None] for i in indices]
            if isinstance(input, SymbolicInputKit):
                if default is None:
                    _defaults.append((True, True, None))
                else:
                    _defaults.append((False, False, default))
            elif input.update is not None:
                if default is None:
                    raise ValueError("A default (initial) value is required for an input which can update itself.")
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    _defaults.append((True, False, None))
                else:
                    _defaults.append((False, True, default))
        defaults = _defaults

        # Get a function instance
        if profiler is None:
            # some linkers may not support profilers, so we avoid passing the option altogether
            _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage)
        else:
            _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage,
                                                 profiler = profiler)
        fn = Function(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self, profiler)
        return fn


from functools import partial

class Function(object):

    def __init__(self, fn, input_storage, output_storage, indices, outputs, defaults, unpack_single, maker, profiler):

        self.fn = fn
        self.input_storage = input_storage
        self.output_storage = output_storage

        containers = list(self.input_storage)
        finder = {}
        inv_finder = {}
        defaults_layer = []

        def distribute(indices, cs, value):
            input.distribute(value, indices, cs)
            for c in cs:
                c.provided = True
        def set(c, v):
            c.data = v

        setters = []
        for i, ((input, indices, sinputs), (required, refeed, value)) in enumerate(zip(indices, defaults)):
            if indices is None:
                c = containers[0]
                if value:
                    c.data = value
                c.required = required
                c.provided = False
                finder[i] = c
                finder[input.result] = c
                finder[input.name] = c
                inv_finder[c] = input
                setters.append(partial(set, c))
                containers[:1] = []
            else:
                cs = containers[:len(indices)]
                input.distribute(value, indices, cs)
                f = partial(distribute, indices, cs)
                finder[i] = f
                finder[input] = f
                finder[input.name] = f
                setters.append(f)
                for c, sin in zip(cs, sinputs):
                    finder[sin.result] = c
                    finder[sin.name] = c
                    inv_finder[c] = input
                    c.required = required
                    c.provided = False
                containers[:len(indices)] = []

        self.finder = finder
        self.inv_finder = inv_finder
        self.indices = indices
        self.outputs = outputs
        self.defaults = defaults
        self.unpack_single = unpack_single
        self.maker = maker
        self.profiler = profiler

        # this class is important in overriding the square-bracket notation:
        #     fn.value[x]
        # self reference is available via the closure on the class
        class ValueAttribute(object):
            def __getitem__(self, item):
                s = finder[item]
                if isinstance(s, gof.Filter):
                    return s.value
                else:
                    raise NotImplementedError
            def __setitem__(self, item, value):
                s = finder[item]
                if isinstance(s, gof.Filter):
                    s.value = value
                    s.provided = True
                else:
                    s(value)

        # this class is important in overriding the square-bracket notation:
        #     fn.container[x]
        # self reference is available via the closure on the class
        class ContainerAttribute(object):
            def __getitem__(self, item):
                return finder[item]

        self._value = ValueAttribute()
        self._container = ContainerAttribute()


    def __getitem__(self, item):
        return self.value[item]

    def __setitem__(self, item, value):
        self.value[item] = value
    
    def __copy__(self):
        defaults = list(self.defaults)
        for i, default in enumerate(defaults):
            default[0] = self[i]
        return self.maker.create(defaults, self.profiler)

    def __call__(self, *args, **kwargs):
        for c in self.input_storage:
            c.provided = False
        for i, (required, refeed, value) in enumerate(self.defaults):
            if refeed:
                self[i] = value
        for i, arg in enumerate(args):
            self[i] = arg
        for k, arg in enumerate(kwargs):
            self[k] = arg
        for c in self.input_storage:
            if c.required and not c.provided:
                raise TypeError("Missing required input: ", self.inv_finder[c].result)
        self.fn()
        outputs = [x.data for x in self.output_storage]
        for input, storage in reversed(zip(self.maker.expanded_inputs, self.input_storage)):
            if input.update:
                storage.data = outputs.pop()
        if self.unpack_single and len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


    value = property(
            lambda self: self._value,
            None, #not settable
            doc="""TODOC""")
    container = property(
            lambda self: self._container,
            None,
            doc="""TODOC""")



#         self.fn = fn
#         self.inputs = inputs
#         self.outputs = outputs
#         self.input_storage = input_storage
#         self.output_storage = output_storage

#         self.finder = dict([(input.name, storage) for input, storage in zip(inputs, input_storage)])
#         self.finder.update(dict([(input, storage) for input, storage in zip(inputs, input_storage)]))
#         self.finder.update(dict([(i, storage) for i, storage in enumerate(input_storage)]))
#         self.inv_finder = dict([(storage, (input, input.name, i))
#                                 for i, (input, storage) in enumerate(zip(inputs, input_storage))])

#         # this class is important in overriding the square-bracket notation:
#         #     fn.value[x]
#         # self reference is available via the closure on the class
#         class ValueAttribute(object):

#             def __getitem__(self, item):
#                 return self.finder[item].value

#             def __setitem__(self, item, value):
#                 self.finder[item].value = value

#         # this class is important in overriding the square-bracket notation:
#         #     fn.container[x]
#         # self reference is available via the closure on the class
#         class ContainerAttribute(object):

#             def __getitem__(self, item):
#                 return self.finder[item]

#             def __setitem__(self, item, new):
#                 orig = self.finder[item]
#                 input, name, i = self.inv_finder.pop(orig)
#                 self.inv_finder[new] = input, name, i
#                 self.finder[input] = new
#                 self.finder[name] = new
#                 self.finder[i] = new

#         self._value = ValueAttribute()
#         self._container = ContainerAttribute()

#     def __getitem__(self, item):
#         return self.value[item]

#     def __setitem__(self, item, value):
#         self.value[item] = value
    
#     def __copy__(self):
#         raise NotImplementedError()

#     def __call__(self, *args, **kwargs):
#         for i, arg in enumerate(args):
#             self[i] = arg



def function(input, output, mode='FAST_RUN', autoname_input=True):

    # create a subclass of Function for the given arguments.
    class F(Function):
        pass

    if autoname_input:
        # name the inputs according to the autoname feature
        raise NotImplementedError()

    # add all input names as properties of F
    # if there is a name collision:
    # - raise ValueError
    # - return message suggesting to rename or disable autoname (if it was enabled)
    raise NotImplementedError()

    # return an instance of this new Function subclass
    return F(input, output, mode)











# class State(object):
#     def __init__(self, variable, new_state = None):
#         self.variable = variable
#         if new_state is None:
#             self.new_state = variable
#         else:
#             self.new_state = new_state

# class StateContainer(object):
#     def __init__(self, data):
#         self.data = data

# def env_with_state(normal_inputs, normal_outputs, states, accept_inplace = False):
#     state_inputs = [s.variable for s in states]
#     state_outputs = [s.new_state for s in states]
#     inputs = normal_inputs + state_inputs
#     outputs = normal_outputs + state_outputs
#     inputs, outputs = gof.graph.clone(inputs, outputs)
#     env = gof.env.Env(inputs, outputs)
#     for node in env.nodes:
#         if getattr(node.op, 'destroy_map', None):
#             if not accept_inplace:
#                 raise TypeError("Graph must not contain inplace operations", node)
#             else:
#                 env.extend(gof.DestroyHandler())
#                 break
#     env.extend(Supervisor(normal_inputs))
#     return env

# def function_with_state(fn, state_containers, unpack_single = True):
#     n = len(state_containers)
#     nin = len(fn.inputs)
#     nout = len(fn.outputs)
#     if n == 0:
#         if unpack_single and nin == 1:
#             return lambda *inputs: fn(*inputs)[0]
#         else:
#             return fn
#     def f(*inputs):
#         results = fn(*(list(inputs) + [c.data for c in state_containers]))
#         for c, d in zip(state_containers, results[-n:]):
#             c.data = d
#         results = results[:-n]
#         if unpack_single and len(results) == 1:
#             return results[0]
#         else:
#             return results







# class FunctionFactory:

#     def __init__(self,
#                  inputs,
#                  outputs,
#                  states = [],
#                  linker = default_linker,
#                  optimizer = default_optimizer,
#                  borrow_outputs = False,
#                  accept_inplace = False):

#         self.states = states
#         inputs, outputs = list(inputs), list(outputs)

#         # Error checking
#         for r in inputs + outputs:
#             if not isinstance(r, gof.Result):
#                 raise TypeError("All inputs and outputs to FunctionFactory should be Result instances. Received:", type(r), r)
#         for state in states:
#             if not isinstance(state, State):
#                 raise TypeError("All states must be State instances", type(state), state)
#         if len(inputs) != len(set(inputs)):
#             print >>sys.stderr, "Warning: duplicate inputs"

#         # make the env
#         env = env_with_state(inputs, outputs, states, accept_inplace)
#         self.env = env

#         # optimize the env
#         optimizer = predefined_optimizers.get(optimizer, optimizer)
#         optimizer(env)

#         # initialize the linker
#         linker = copy(predefined_linkers.get(linker, linker))
#         if not hasattr(linker, 'accept'):
#             raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
#                              "or one of %s" % predefined_linkers.keys())

#         if borrow_outputs:
#             self.linker = linker.accept(env)
#         else:
#             self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, env.outputs))


#     def create(self, 
#                states = [], 
#                profiler = None, 
#                unpack_single = True, 
#                strict = 'if_destroyed'):

#         # Error checking
#         if strict not in [True, False, 'if_destroyed']:
#             raise ValueError("'strict' parameter of create should be one of [True, False, 'if_destroyed']")
#         if len(states) != len(self.states):
#             raise ValueError("not the right number of state initializers (expected %i, got %i)" % (len(self.states), len(states)))

#         # Get a function instance
#         if profiler is None:
#             # some linkers may not support profilers, so we avoid passing the option altogether
#             _fn = self.linker.make_function(unpack_single = False)
#         else:
#             _fn  = self.linker.make_function(unpack_single = False,
#                                              profiler = profiler)
#         fn = function_with_state(_fn, states, unpack_single)

#         # Make the inputs strict accordingly to the specified policy
#         for env_input, fn_input in zip(self.env.inputs, _fn.inputs):
#             if strict is True or (strict == 'if_destroyed' and self.env.destroyers(env_input)):
#                 fn_input.strict = True
#         return fn



# def function(inputs,
#              outputs,
#              states = [],
#              linker = default_linker,
#              optimizer = default_optimizer,
#              borrow_outputs = False,
#              accept_inplace = False,
#              profiler = None,
#              unpack_single = True,
#              strict = 'if_destroyed'):
    
#     ff = FunctionFactory(inputs,
#                          outputs,
#                          states = [s[0] for s in states],
#                          linker = linker,
#                          optimizer = optimizer,
#                          borrow_outputs = borrow_outputs)
#     return ff.create(states = [s[1] for s in states],
#                      profiler = profiler,
#                      unpack_single = unpack_single,
#                      strict = strict)






























































































# import numpy
# import gof
# import sys
# from copy import copy

# #TODO: put together some default optimizations (TRAC #67)

# def exec_py_opt(inputs, outputs, features=[]):
#     """Return an optimized graph running purely python implementations"""
#     return Function(intputs, outputs, features, exec_py_opt.optimizer, gof.link.PerformLinker(), False)
# exec_py_opt.optimizer = None

# def exec_opt(inputs, outputs, features=[]):
#     """Return a fast implementation"""
#     return Function(intputs, outputs, features, exec_opt.optimizer, gof.link.PerformLinker(), False)
# exec_opt.optimizer = None

# class _DefaultOptimizer(object):
#     #const = gof.opt.ConstantFinder()
#     merge = gof.opt.MergeOptimizer()
#     def __call__(self, env):
#         #self.const(env)
#         self.merge(env)
# default_optimizer = _DefaultOptimizer()
        
# def _mark_indestructible(results):
#     for r in results:
#         r.tag.indestructible = True

# # def linker_cls_python_and_c(env, **kwargs):
# #     """Use this as the linker_cls argument to Function.__init__ to compare
# #     python and C implementations"""

# def check_equal(x, y):
#     x, y = x[0], y[0]
#     if isinstance(x, numpy.ndarray) or isinstance(y, numpy.ndarray):
#         if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
#             raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
#     else:
#         if x != y:
#                 raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})

# #     return gof.DualLinker(checker, **kwargs).accept(env)


# def infer_reuse_pattern(env, outputs_to_disown):
#     do_not_reuse = list()
#     seen = set()
#     def walk(r):
#         if r.owner is None or r in seen:
#             return
#         seen.add(r)
#         do_not_reuse.append(r)
#         node = r.owner
#         op = node.op
#         dmap = op.destroy_map if hasattr(op, 'destroy_map') else {}
#         vmap = op.view_map if hasattr(op, 'view_map') else {}
#         for l in dmap.values() + vmap.values():
#             for i in l:
#                 walk(node.inputs[i])
#     for output in outputs_to_disown:
#         walk(output)
#     return do_not_reuse


# def cloned_env(inputs, outputs):
#     inputs, outputs = gof.graph.clone(inputs, outputs)
#     env = gof.env.Env(inputs, outputs)
#     return env

# def std_env(inputs, outputs, disown_inputs = False,
#             use_destroy_handler = True):
#     inputs, outputs = gof.graph.clone(inputs, outputs)
#     _mark_indestructible(outputs)
#     env = gof.env.Env(inputs, outputs)
#     if use_destroy_handler:
#         env.extend(gof.DestroyHandler())
#     env.extend(gof.ReplaceValidate())
#     env.validate()
#     for input in inputs:
#         input.destroyed_by_user = use_destroy_handler and len(env.destroyers(input)) != 0
#         if not input.destroyed_by_user and not disown_inputs:
#             # prevent optimizations from destroying the inputs
#             input.tag.indestructible = True
#     return env

# def std_opt(env):
#     pass


# predefined_linkers = {
#     'py'   : gof.PerformLinker(),
#     'c'    : gof.CLinker(),
#     'c|py' : gof.OpWiseCLinker(),
#     'c&py' : gof.DualLinker(checker = check_equal)
#     }

# class FunctionFactory:

#     def __init__(self, inputs, outputs, linker = 'py', optimizer = std_opt, borrow_outputs = False, disown_inputs = False,
#                  use_destroy_handler = True):
#         if len(inputs) != len(set(inputs)):
#             print >>sys.stderr, "Warning: duplicate inputs"
#         for r in list(inputs) + list(outputs):
#             if not isinstance(r, gof.Result):
#                 raise TypeError("All inputs and outputs to FunctionFactory should be Result instances. Received:", type(r), r)
#         env = std_env(inputs, outputs, disown_inputs = disown_inputs,
#                       use_destroy_handler = use_destroy_handler)
#         if None is not optimizer:
#             optimizer(env)
#         env.validate()
#         self.env = env
#         linker = copy(predefined_linkers.get(linker, linker))
#         if not hasattr(linker, 'accept'):
#             raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
#                              "or one of ['py', 'c', 'c|py', 'c&py']")
#         if borrow_outputs:
#             self.linker = linker.accept(env)
#         else:
#             self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, env.outputs))
            
            
#     def create(self, profiler = None, unpack_single = True, strict = 'if_destroyed'):
#         if strict not in [True, False, 'if_destroyed']:
#             raise ValueError("'strict' parameter of create should be one of [True, False, 'if_destroyed']")
#         if profiler is None:
#             fn = self.linker.make_function(unpack_single=unpack_single)
#         else:
#             fn  = self.linker.make_function(unpack_single=unpack_single,
#                                             profiler=profiler)
#         for env_input, fn_input in zip(self.env.inputs, fn.inputs):
#             if strict is True or (env_input.destroyed_by_user and strict == 'if_destroyed'):
#                 fn_input.strict = True
#         return fn

#     def partial(self, *first, **kwargs):
#         fn = self.create(**kwargs)
#         return lambda *last: fn(*(first + last))


# def function(inputs,
#              outputs,
#              linker = 'py',
#              optimizer = std_opt,
#              borrow_outputs = False,
#              disown_inputs = False,
#              profiler = None,
#              unpack_single = True,
#              strict = 'if_destroyed',
#              use_destroy_handler = True):
#     ff = FunctionFactory(inputs,
#                          outputs,
#                          linker = linker,
#                          optimizer = optimizer,
#                          borrow_outputs = borrow_outputs,
#                          disown_inputs = disown_inputs,
#                          use_destroy_handler = use_destroy_handler)
#     return ff.create(profiler = profiler,
#                      unpack_single = unpack_single,
#                      strict = strict)


# def eval_outputs(outputs, **kwargs):
#     return function([], outputs, **kwargs)()


# _fcache = {} # it would be nice to use weakref.WeakKeyDictionary()

# def fast_compute(*outputs):
#     if outputs in _fcache:
#         f = _fcache[outputs]
#     else:
#         f = function([], outputs, linker = 'c')
#         _fcache[outputs] = f
#     return f()




# class OpFromGraph(gof.Op):
#     """
#     This create an L{Op} from a list of input results and a list of output
#     results.

#     The signature is the same as the signature of L{FunctionFactory}
#     and/or function and the resulting L{Op}'s perform will do the same
#     operation as::
#       function(inputs, outputs, **kwargs)

#     Take note that the following options, if provided, must take the
#     value(s) listed below:
#       unpack_single = False
#       borrow_outputs = False

#     OpFromGraph takes an additional input, grad_depth. If grad_depth
#     is n, OpFromGraph will make special Ops for gradients up to the
#     nth level, allowing the user to differentiate this op up to n
#     times. The parameter defaults to 1. If grad_depth == 0, the op
#     will not be differentiable.

#     Example:
#       x, y, z = tensor.scalars('xyz')
#       e = x + y * z
#       op = OpFromGraph([x, y, z], [e], linker='c')
#       # op behaves like a normal theano op
#       e2 = op(x, y, z) + op(z, y, x)
#       fn = function([x, y, z], [e2])
#     """
    
#     def __init__(self, inputs, outputs, grad_depth = 1, **kwargs):
#         if kwargs.get('borrow_outputs') or kwargs.get('unpack_single'):
#             raise ValueError('The borrow_outputs and unpack_single options cannot be True')
#         kwargs['unpack_single'] = False
#         kwargs['borrow_outputs'] = False
#         self.fn = function(inputs, outputs, **kwargs)
#         self.inputs = inputs
#         self.outputs = outputs
#         self.input_types = [input.type for input in inputs]
#         self.output_types = [output.type for output in outputs]
#         if grad_depth > 0:
#             import gradient as G
#             output_grads = [t() for t in self.output_types]
#             gd = G.grad_sources_inputs(zip(self.outputs, output_grads), self.inputs)
#             gs = map(gd.get, self.inputs)
#             self.grad_ops = []
#             for g in gs:
#                 if g is None:
#                     self.grad_ops.append(lambda *args: None)
#                 else:
#                     self.grad_ops.append(OpFromGraph(inputs + output_grads,
#                                                      [g],
#                                                      grad_depth = grad_depth - 1))

#     def make_node(self, *inputs):
#         for input, type in zip(inputs, self.input_types):
#             if not type == input.type:
#                 raise TypeError("Wrong type, expected %s but got %s" % type, input.type)
#         return gof.Apply(self,
#                          inputs,
#                          [type() for type in self.output_types])

#     def perform(self, node, inputs, outputs):
#         results = self.fn(*inputs)
#         for output, result in zip(outputs, results):
#             output[0] = result

#     def grad(self, inputs, output_grads):
#         if hasattr(self, 'grad_ops'):
#             return [go(*(inputs + output_grads)) for go in self.grad_ops]
#         else:
#             raise NotImplementedError


# # class Container(object):
# #     def __init__(self, r, value, strict = False):
# #         self.r = r
# #         self.type = r.type
# #         self._value = value
# #         self.strict = strict
# #     def __get(self):
# #         return self._value
# #     def __set(self, value):
# #         try:
# #             if self.strict:
# #                 self._value = self.type.filter(value, strict = True)
# #             else:
# #                 self._value = self.type.filter(value)
# #         except:
# #             raise_with_op(self.r)
# #     value = property(__get, __set)
# #     def __str__(self):
# #         return "<" + str(self.value) + ">"
# #     def __repr__(self):
# #         return "<" + repr(self.value) + ">"
# # #     def put(self, filters):
# # #         filter = filters.popleft()
# # #         filter.data = self.value
# # #         return [filter], filters



# #########################aaaaaaaaaaa



# # class State:
# #     def __init__(self, init, next = None):
# #         self.init = init
# #         self.next = next


# # class StateFunctionFactory(Function):

# #     def __init__(self, inputs, outputs, states, **kwargs):
# #         states_
        
# #         inputs = [state.init for state in states] + inputs
# #         outputs = [state.next for ]

    


# # class Function:
# #     """
# #     An 'executable' compiled from a graph

# #     This class is meant to be used as a function: the idea is to use
# #     __call__(*args) and it will compute your graph's function on the args and
# #     return the value(s) corresponding to the output(s).
    
# #     @ivar fn: the return value of L{linker.make_function}(False)

# #     Additional Attributes if keep_locals == True
# #     inputs - inputs in the env
# #     outputs - outputs in the env
# #     features - features to add to the env
# #     linker_cls - the linker class
# #     linker - the linker allocated from env
# #     env - The env passed to the linker

# #     @note: B{Re: Memory ownership, aliasing, re-use:}
# #     That the objects returned by L{Function.__call__}(self, *args) are owned
# #     by self, and that in general these outputs might be overwritten (in-place)
# #     by subsequent calls to L{self.__call__}(*args).  Why?  This behaviour is
# #     necessary for inplace operations to work, and L{Function}'s linker might re-use
# #     memory from one execution to the next in order to make each execution faster.

# #     """
# #     def __init__(self, inputs, outputs,
# #             features = [],
# #             optimizer = default_optimizer,
# #             linker_cls = gof.link.PerformLinker,
# #             profiler = None,
# #             unpack_single = True,
# #             except_unreachable_input = True,
# #             keep_locals = True):
# #         """
# #         Copy the graph, optimize, and link it.

# #         @param inputs: a list of results to be this function's inputs
# #         @param outputs: a list of results to be this function's outputs
# #         @param features: features to add to the env
# #         @param optimizer: an optimizer to apply to the copied graph, before linking
# #         @param linker_cls: a callable that takes an env and returns a Linker
# #         @param profiler: a L{Profiler} for the produced function (only valid if the
# #                    linker_cls's make_function takes a profiler argument)
# #         @param unpack_single: unpack return value lists of length 1. @see: L{Linker.make_function}
# #         @param keep_locals: add the local variables from __init__ to the class
# #         """

# #         _mark_indestructible(outputs)

# #         if len(inputs) != len(set(inputs)):
# #             raise Exception('duplicate inputs')
# #         if len(outputs) != len(set(outputs)):
# #             raise Exception('duplicate outputs')

# #         #evaluate the orphans, and put these values into the clone of the env

# #         orphans = list(gof.graph.results_and_orphans(inputs, outputs,
# #             except_unreachable_input=except_unreachable_input)[1])
# #         orphan_data = eval_outputs(orphans, unpack_single=False)

# #         #print 'orphans', orphans

# #         #print 'ops', gof.graph.ops(inputs, outputs)
# #         env = gof.env.Env(inputs, outputs)

# #         #print 'orphans in env', env.orphans()

# #         env, equiv = env.clone_get_equiv(clone_inputs=True)
# #         for feature in features:
# #             env.extend(feature(env))
# #         env.extend(gof.DestroyHandler(env))

# #         #print 'orphans after clone', env.orphans()

# #         for d, o in zip(orphan_data, [equiv[orphan] for orphan in orphans]):
# #             #print 'assigning orphan value', d
# #             #o.data = d
# #             new_o = gof.Constant(o.type, d)
# #             env.replace(o, new_o)
# #             assert new_o in env.orphans

# #         # optimize and link the cloned env
# #         if None is not optimizer:
# #             optimizer(env)

# #         linker = linker_cls(env)

# #         if keep_locals:# useful flag for debugging!
# #             self.__dict__.update(locals())

# #         if profiler is None:
# #             self.fn  = linker.make_function(unpack_single=unpack_single)
# #         else:
# #             self.fn  = linker.make_function(unpack_single=unpack_single,
# #                                             profiler=profiler)
# #         self.inputs = env.inputs
# #         self.outputs = env.outputs
# #         self.features = features
# #         self.optimizer = optimizer
# #         self.linker_cls = linker_cls
# #         self.profiler = profiler
# #         self.unpack_single = unpack_single
# #         self.except_unreachable_input = except_unreachable_input
# #         self.keep_locals = keep_locals

# #     def __call__(self, *args):
# #         return self.fn(*args)


# # def eval_outputs(outputs,
# #         features = [],
# #         optimizer = None,
# #         linker_cls = gof.link.PerformLinker,
# #         unpack_single = True,
# #         keep_locals = True):

# #     if len(outputs) == 0:
# #         #print 'returning with no inputs'
# #         if unpack_single:
# #             return None
# #         else:
# #             return []

# #     inputs = gof.graph.inputs(outputs)
# #     if any(not isinstance(input, gof.Constant) for input in inputs):
# #         raise TypeError("Cannot evaluate outputs because some of the leaves are not Constant.", outputs)
# #     in_data = [i.data for i in inputs]
# #     #print 'in_data = ', in_data
# #     if len(inputs) != len(in_data):
# #         raise Exception('some input data is unknown')

# #     env = gof.env.Env(inputs, outputs)
# #     env.replace_all(dict([(i, i.type()) for i in inputs]))
# #     env = env.clone(clone_inputs=True)

# #     _mark_indestructible(env.outputs)
# #     if None is not optimizer:
# #         optimizer(env)
# #     linker = linker_cls(env)
# #     fn = linker.make_function(unpack_single=unpack_single)
# #     rval = fn(*in_data)
# #     return rval


# # StateFunction([x, y], [e], (w, w + lr * bla()))




# # class _Function:

# #     def __init__(self,
# #                  inputs,
# #                  outputs,
# #                  optimizer,
# #                  linker_type = 'py',
# #                  unpack_single = True,
# #                  except_unreachable_input = True,
# #                  disposable_inputs = [],
# #                  borrow_outputs = []):




# #         _mark_indestructible(outputs)

# #         if len(inputs) != len(set(inputs)):
# #             raise Exception('duplicate inputs')
# #         if len(outputs) != len(set(outputs)):
# #             raise Exception('duplicate outputs')

# #         orphans = list(gof.graph.results_and_orphans(inputs, outputs,
# #             except_unreachable_input=except_unreachable_input)[1])
# #         orphan_data = eval_outputs(orphans, unpack_single=False)

# #         env = gof.env.Env(inputs, outputs, features + [gof.EquivTool], consistency_check = True)

# #         env = env.clone(clone_inputs=True)

# #         for d, o in zip(orphan_data, [env.equiv(orphan) for orphan in orphans]):
# #             o.data = d

# #         # optimize and link the cloned env
# #         if None is not optimizer:
# #             optimizer(env)

# #         linker = linker_cls(env)

# #         if keep_locals:# useful flag for debugging!
# #             self.__dict__.update(locals())

# #         if profiler is None:
# #             self.fn  = linker.make_function(inplace=True,
# #                                             unpack_single=unpack_single)
# #         else:
# #             self.fn  = linker.make_function(inplace=True,
# #                                             unpack_single=unpack_single,
# #                                             profiler=profiler)
# #         self.inputs = env.inputs
# #         self.outputs = env.outputs
# #         self.features = features
# #         self.optimizer = optimizer
# #         self.linker_cls = linker_cls
# #         self.profiler = profiler
# #         self.unpack_single = unpack_single
# #         self.except_unreachable_input = except_unreachable_input
# #         self.keep_locals = keep_locals

# #     def __call__(self, *args):
# #         return self.fn(*args)

# #     def __copy__(self):
# #         return Function(self.inputs, self.outputs,
# #                         features = self.features,
# #                         optimizer = self.optimizer,
# #                         linker_cls = self.linker_cls,
# #                         profiler = self.profiler,
# #                         unpack_single = self.unpack_single,
# #                         except_unreachable_input = self.except_unreachable_input,
# #                         keep_locals = self.keep_locals)












# # class StateFunction:

# #     def __init__(self, inputs, outputs, *states):
# #         in_states, out_states = zip(*states)
# #         env = 

    
