"""Convenient driver of graph construction, optimization, and linking."""

import copy_reg
import cPickle

from functools import partial

import numpy
from .. import gof
import sys
from copy import copy

from mode import *
from io import *


def infer_reuse_pattern(env, outputs_to_disown):
    """
    Given an env and a list of results, returns the list of all
    results which may share the same underlying data storage as any of
    the specified results. Used internally by function, FunctionMaker.

    This list is also refered to as no_recycling sometimes.
    """
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


class Supervisor:
    """
    Listener for Env events which makes sure that no operation overwrites the
    contents of protected Results. The outputs of the Env are protected by default.
    """

    def __init__(self, protected):
        self.protected = list(protected)

    def validate(self, env):
        if not hasattr(env, 'destroyers'):
            return True
        for r in self.protected + list(env.outputs):
            if env.destroyers(r):
                raise gof.InconsistencyError("Trying to destroy a protected Result.", r)


def std_env(input_specs, output_specs, accept_inplace = False):
    """
    Makes an Env corresponding to the input specs and the output
    specs.  Any SymbolicInput in the input_specs, if its update field
    is not None, will add an output to the Env corresponding to that
    update. The return value is the Env as well as a list of
    SymbolicOutput instances corresponding to the updates.

    If accept_inplace is False, the graph will be checked for inplace
    operations and an exception will be raised if it has any. If
    accept_inplace is True, a DestroyHandler will be added to the Env
    if there are any inplace operations.

    The returned Env is a clone of the graph between the provided
    inputs and outputs.
    """
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

    # We need to protect all immutable inputs from inplace operations.
    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not (spec.mutable or (hasattr(env, 'destroyers') and env.destroyers(input)))))
    return env, map(SymbolicOutput, updates)


###
### Function
###

DUPLICATE = ['DUPLICATE'] # unique id object used as a placeholder for duplicate entries
class Function(object):
    """
    Type of the functions returned by theano.function or theano.FunctionMaker.create.
    """

    def __init__(self, fn, input_storage, output_storage, indices, outputs, defaults, unpack_single, maker):
        """
        fn -> a function returned by some linker's make_thunk method
        input_storage -> list of Container instances used by fn to fetch the inputs
        output_storage -> list of Container instances used by fn to store the outputs in
        indices -> list of (SymbolicInput|SymbolicInputKit, indices, [SymbolicInput,...]), one tuple for each input
        defaults -> list of (required (bool), refeed (bool), value), one tuple for each input
            required -> whether this input is required or optional
            refeed -> whether this input's contents must be reverted to value after each call or not
            value -> the initial or default value of the input
        unpack_single -> if the function has one output and unpack_single is True, return that output. Else,
            return [output].
        maker -> FunctionMaker instance used to make this Function (used for copy)
        """

        self.fn = fn
        self.input_storage = input_storage
        self.output_storage = output_storage
        self.indices = indices

        containers = list(self.input_storage)
        finder = {}
        inv_finder = {}

        def distribute(indices, cs, value):
            input.distribute(value, indices, cs)
            for c in cs:
                c.provided += 1
        def assign(c, v):
            c.data = v

        setters = []
        # Initialize the storage
        for i, ((input, indices, sinputs), (required, refeed, value)) in enumerate(zip(self.indices, defaults)):
            if indices is None: # this is true iff input is not a SymbolicInputKit
                c = containers[0]  #containers is being used as a stack. Here we pop off the next one.
                if input.strict:
                    c.strict = True
                if value is not None:
                    # always initialize the storage
                    c.data = value
                c.required = required
                c.provided = 0 # this is a count of how many times the input has been provided (reinitialized to 0 on __call__)
                # We set an entry in finder for:
                # - the index of the input
                # - the result instance the input is based on
                # - the name of the input
                # All entries map to the container or to DUPLICATE if an ambiguity is detected
                finder[i] = c
                finder[input.result] = c
                finder[input.name] = c if input.name not in finder else DUPLICATE
                # inv_finder maps the container to the input (useful for one error message)
                inv_finder[c] = input
                setters.append(partial(assign, c))
                containers[:1] = []
            else:
                # The input is a SymbolicInputKit, so we take as many containers as the Kit provides inputs
                cs = containers[:len(indices)]
                # distribute does the initialization of the containers
                input.distribute(value, indices, cs)
                f = partial(distribute, indices, cs)
                # Like before, we set a finder entry for the kit. Note that
                # we are not mapping to a container but to a function which
                # can reinitialize all the containers
                finder[i] = f
                finder[input] = f
                finder[input.name] = f if input.name not in finder else DUPLICATE
                setters.append(f)
                # For each input in the kit and its corresponding container, we put an entry in finder.
                # This allows the user to micro-manage elements of the kit if need be.
                # All containers inherit the required field and have their own "provided" counter
                for c, sin in zip(cs, sinputs):
                    finder[sin.result] = c
                    finder[sin.name] = c
                    finder[sin.name] = c if sin.name not in finder else DUPLICATE
                    inv_finder[c] = input
                    c.required = required
                    c.provided = 0
                containers[:len(indices)] = []

        self.finder = finder
        self.inv_finder = inv_finder
        self.outputs = outputs
        self.defaults = defaults
        self.unpack_single = unpack_single
        self.maker = maker

        # this class is important in overriding the square-bracket notation:
        #     fn.value[x]
        # self reference is available via the closure on the class
        class ValueAttribute(object):
            def __getitem__(self, item):
                try:
                    s = finder[item]
                except KeyError:
                    raise TypeError("Unknown input or state: %s" % item)
                if s is DUPLICATE:
                    raise TypeError("Ambiguous name: %s - please check the names of the inputs of your function for duplicates." % item)
                if isinstance(s, gof.Container):
                    return s.value
                else:
                    raise NotImplementedError
            def __setitem__(self, item, value):
                try:
                    s = finder[item]
                except KeyError:
                    raise TypeError("Unknown input or state: %s" % item)
                if s is DUPLICATE:
                    raise TypeError("Ambiguous name: %s - please check the names of the inputs of your function for duplicates." % item)
                if isinstance(s, gof.Container):
                    s.value = value
                    s.provided += 1
                else:
                    s(value)

        # this class is important in overriding the square-bracket notation:
        #     fn.container[x]
        # self reference is available via the closure on the class
        class ContainerAttribute(object):
            def __getitem__(self, item):
                return finder[item]
            # You cannot set the container

        self._value = ValueAttribute()
        self._container = ContainerAttribute()

    def __getitem__(self, item):
        return self.value[item]

    def __setitem__(self, item, value):
        self.value[item] = value
        
    
    def __copy__(self):
        defaults = [default for _1, _2, default in self.defaults]
        cpy = self.maker.create(defaults, trustme = True)
        for (input,_1,_2), here, there in zip(self.indices, self.input_storage, cpy.input_storage):
            if input.mutable and here is not None:
                there.data = copy(here.data)
            else:
                there.data = here.data
        return cpy

    def __call__(self, *args, **kwargs):
        # Reinitialize each container's 'provided' counter
        for c in self.input_storage:
            c.provided = 0
        # Set positional arguments
        for i, arg in enumerate(args):
            self[i] = arg
        # Set keyword arguments
        for k, arg in kwargs.iteritems():
            self[k] = arg
        # Check if inputs are missing or if inputs were set more than once
        for c in self.input_storage:
            if c.required and not c.provided:
                raise TypeError("Missing required input: %s" % getattr(self.inv_finder[c], 'result', self.inv_finder[c]))
            if c.provided > 1:
                raise TypeError("Multiple values for input: %s" % getattr(self.inv_finder[c], 'result', self.inv_finder[c]))
        # Do the actual work
        self.fn()
        outputs = [x.data for x in self.output_storage]
        # Update the inputs that have an update function
        for input, storage in reversed(zip(self.maker.expanded_inputs, self.input_storage)):
            if input.update:
                storage.data = outputs.pop()
        # Put default values back in the storage
        for i, (required, refeed, value) in enumerate(self.defaults):
            if refeed:
                self[i] = value
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

# pickling/deepcopy support for Function

def _pickle_Function(f):
    ins = list(f.input_storage)
    defaults = []
    for (input, indices, inputs), (required, refeed, default) in zip(f.indices, f.defaults):
        if isinstance(input, SymbolicInputKit):
            li = len(indices)
            if not default:
                defaults.append(ins[:li])
            else:
                defaults.append(default)
            ins[:li] = []
        else:
            defaults.append(ins[0])
            del ins[0]
    return (_constructor_Function, (f.maker, defaults, [x.data for x in f.input_storage]))

def _constructor_Function(maker, defaults, data):
    f = maker.create(defaults, trustme = True)
    for container, x in zip(f.input_storage, data):
        container.data = x
    return f

copy_reg.pickle(Function, _pickle_Function)



###
### SanityCheckFunction
###

class SanityCheckFunction(Function):

    def __init__(self, others, check_equal, *args, **kwargs):
        super(SanityCheckFunction, self).__init__(*args, **kwargs)
        self.others = others
        self.check_equal = check_equal

    def __setitem__(self, item, value):
        super(SanityCheckFunction, self).__setitem__(item, value)
        for fn in self.others:
            fn[item] = value

    def __call__(self, *args, **kwargs):
        results = super(SanityCheckFunction, self).__call__(*args, **kwargs)
        all_outputs = [copy(c.value) for c in self.output_storage] # we keep a copy to make sure it's not overwritten
        for fn in self.others:
            fn(*args, **kwargs)
            # This checks all output storage (this includes state variables that we updated)
            # This is ok because the results of a call stick around in their storage
            for i, (r1, c2) in enumerate(zip(all_outputs, fn.output_storage)):
                r2 = c2.value
                if not self.check_equal(r1, r2):
                    name = c2.name
                    raise ValueError("Result #%i%s using %s and %s differs."
                                     % (i,
                                        " (%s)" % name if name else "",
                                        self.maker.mode,
                                        fn.maker.mode),
                                     r1, r2)
        return results



###
### FunctionMaker
###

NODEFAULT = ['NODEFAULT']
class FunctionMaker(object):

    @staticmethod
    def wrap_in(input):
        if isinstance(input, (SymbolicInput, SymbolicInputKit)):
            return input
        elif isinstance(input, gof.Result):
            # r -> SymbolicInput(result=r)
            return SymbolicInput(input)
        elif isinstance(input, (list, tuple)):
            # (r, u) -> SymbolicInput(result=r, update=u)
            if len(input) == 2:
                return SymbolicInput(input[0], update = input[1])
            else:
                raise TypeError("Expected two elements in the list or tuple.", input)
        else:
            raise TypeError("Unknown input type:", type(input), input)

    @staticmethod
    def expand_in(sinput, rinputs):
        # For SymbolicInputKits, this extracts a list of SymbolicInput instances
        # and corresponding indices such that these SymbolicInputs are representative
        # of some of the Result instances in inputs.
        # For SymbolicInput, this returns None as the list of indices and a list with
        # just the SymbolicInput.
        if isinstance(sinput, SymbolicInputKit):
            return sinput.complete(rinputs)
        elif isinstance(sinput, SymbolicInput):
            return [None, [sinput]]

    @staticmethod
    def wrap_out(output):
        if isinstance(output, SymbolicOutput):
            return output
        elif isinstance(output, gof.Result):
            return SymbolicOutput(output)
        else:
            raise TypeError("Unknown output type:", type(output), output)

    def __init__(self, inputs, outputs, mode = 'FAST_RUN', accept_inplace = False, function_builder = Function):
        """
        Create a FunctionMaker for the specified inputs, outputs and mode.

        @param inputs: a list of SymbolicInput instances
        @param outputs: a list of SymbolicOutput instances
                   outputs may also be a single Result (not a list), in which
                   case the functions produced by FunctionMaker will return
                   their output value directly
        @param mode: a Mode instance telling FunctionMaker how to optimize and link
        @param accept_inplace: True iff it is acceptable to have inplace operations
                          in the graph from the inputs to the outputs
        """


        # Handle the case where inputs and/or outputs is a single Result (not in a list)
        unpack_single = False
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.result for o in outputs] + [i.update for i in inputs if getattr(i, 'update', False)])
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices], [])

        # make the env
        env, additional_outputs = std_env(expanded_inputs, outputs, accept_inplace)
        self.env = env

        # Fetch the mode and then the optimizer and linker
        mode = predefined_modes.get(mode, mode)
        optimizer, linker = mode.optimizer, copy(mode.linker)

        # optimize the env
        optimizer(env)

        # initialize the linker
        if not hasattr(linker, 'accept'):
            raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
                             "or one of %s" % predefined_linkers.keys())

        #the 'no_borrow' outputs are the ones for which that we can't return the internal storage pointer.
        no_borrow = [output for output, spec in zip(env.outputs, outputs+additional_outputs) if not spec.borrow]
        if no_borrow:
            self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, no_borrow))
        else:
            self.linker = linker.accept(env)
        
        self.indices = indices
        self.inputs = inputs
        self.expanded_inputs = expanded_inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.mode = mode
        self.accept_inplace = accept_inplace
        self.function_builder = function_builder

    def create(self, defaults = None, trustme = False):
        """
        Create a function.

        defaults -> a list matching the inputs list and providing default values
                    if the default for an input is None, then that input is a
                    required input. For an input with an update, the default
                    acts as initialization.
        trustme -> disables some exceptions, used internally
        """
        if defaults is None:
            defaults = [None]*len(self.inputs)
        input_storage = [] # list of independent one-element lists, will be passed to the linker
        _defaults = []

        # The following loop is to fill in the input_storage and _defaults lists.
        for (input, indices, subinputs), default in zip(self.indices, defaults):
            __default = default

            if isinstance(default, gof.Container):
                # If the default is a gof.Container, this means we want to share
                # the same storage. This is done by appending default.storage
                # to input_storage
                if indices is not None:
                    raise TypeError("Cannot take a Container instance as default for a SymbolicInputKit.")
                input_storage.append(default.storage)
                default = None
                required = False
            elif isinstance(input, SymbolicInputKit):
                # If the input is a SymbolicInputKit, it represents more than
                # one storage unit. The indices and subinputs lists represent which
                # of the kit's inputs are active in this graph, so we make as many
                # storage units as needed
                if isinstance(default, (list, tuple)) \
                        and all(isinstance(x, gof.Container) for x in default):
                    if len(default) == len(indices):
                        input_storage += [x.storage for x in default]
                    elif len(default) > len(indices):
                        input_storage += [default[i].storage for i in indices]
                    else:
                        raise ValueError('Not enough storage for SymbolicInputKit', input, indices, default)
                    default = NODEFAULT
                else:
                    input_storage += [[None] for i in indices]
            else:
                # Normal case: one new, independent storage unit
                input_storage.append([None])

            # Filling _defaults. Each entry is a tuple of three elements:
            # (required, refeed, value)
            # - required means that the user must provide a value when calling the function
            # - refeed means that we want to put the default back in the storage after each function call
            # - value is the value that will be put in the storage initially

            # Even though a SymbolicInputKit represents more than one input,
            # we still only have one entry for the defaults list.
            if isinstance(input, SymbolicInputKit):
                if default is NODEFAULT:
                    _defaults.append((False, False, None))
                elif default is None:
                    _defaults.append((True, True, None))
                else:
                    _defaults.append((False, False, default))
            elif input.update is not None:
                # If the input has an update, then (logically) it is not required since
                # it is just a parameter and of course we don't want to refeed the default
                # back into the storage as it would defeat the point of updating it. We
                # always do this policy.
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        # This might catch some bugs early
                        raise ValueError("A default (initial) value is required for an input which can update itself.", input)
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        # No default, so this is a required input. Nothing to feed back, initial value is None.
                        _defaults.append((True, False, None))
                else:
                    # Default value. It is not required, but we want to put it back into the storage
                    # everytime so it behaves like most programming languages' default values
                    _defaults.append((False, True, default))
        defaults = _defaults

        # Get a function instance
        _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage)
        fn = self.function_builder(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self)
        return fn


def _pickle_FunctionMaker(fm):
    return (_constructor_FunctionMaker, (fm.inputs, fm.outputs[0] if fm.unpack_single else fm.outputs, fm.mode, fm.accept_inplace))

def _constructor_FunctionMaker(*args):
    return FunctionMaker(*args)

copy_reg.pickle(FunctionMaker, _pickle_FunctionMaker)


def _pickle_slice(s):
    return (slice, (s.start, s.stop, s.step))

copy_reg.pickle(slice, _pickle_slice)



__checkers = []

def check_equal(x, y):
    for checker in __checkers:
        try:
            return checker(x, y)
        except:
            continue
    return x == y
    #raise Exception('No checker for equality between %s and %s' % (x, y))

def register_checker(checker):
    __checkers.insert(0, checker)




def function(inputs, outputs, mode='FAST_RUN', accept_inplace = False):
    """
    Return a function calculating the outputs from the inputs.

    inputs -> list of SymbolicInput or In instances
    outputs -> a SymbolicOutput or a list of SymbolicOutput or Out instances
      The return value of the returned function will match the format of this
      argument (either the value itself or a list of one or more return values)
    mode -> a descriptive string or a Mode instance; descriptive strings can be one of:
      * SANITY_CHECK TODO: NotImplemented
      * FAST_COMPILE (apply only optimization that are fast to apply)
      * FAST_RUN (default) (optimize without too much time)
      * EXPENSIVE_OPTIMIZATION TODO: NotImplemented
    accept_inplace -> True iff the graph can contain inplace operations
      prior to the optimization phase (default is False)

    Every element of the input list will be upgraded to an In instance if necessary,
    using the following rules:

    * a Result instance r will be upgraded like In(r)
    * a tuple (name, r) will be In(r, name=name)
    * a tuple (r, val) will be In(r, value=value, autoname=True)
    * a tuple ((r,up), val) will be In(r, value=value, update=up, autoname=True)
    * a tuple (name, r, val) will be In(r, name=name, value=value)
    * a tuple (name, (r,up), val) will be In(r, name=name, value=val, update=up, autoname=True)

    Similarly, every element of the output list will be upgraded to an
    Out instance if necessary:

    * a Result instance r will be upgraded like Out(r)
    """

    def wrap_in(input):
        if isinstance(input, (SymbolicInput, SymbolicInputKit)):
            return input
        elif isinstance(input, gof.Result):
            return In(input)
        elif isinstance(input, (list, tuple)):
            orig = input
            if not input:
                raise TypeError("Nonsensical input specification: %s" % input)
            if isinstance(input[0], str):
                name = input[0]
                input = input[1:]
            else:
                name = None
            if isinstance(input[0], (list, tuple)):
                if len(input[0]) != 2 or len(input) != 2:
                    raise TypeError("Invalid input syntax: %s (check documentation or use an In instance)" % orig)
                (result, update), value = input
            elif isinstance(input[0], gof.Result):
                if len(input) == 1:
                    result, update, value = input[0], None, None
                elif len(input) == 2:
                    (result, value), update = input, None
                else:
                    raise TypeError("Invalid input syntax: %s (check documentation or use an In instance)" % orig)
            elif isinstance(input[0], (SymbolicInput, SymbolicInputKit)):
                if len(input) == 1:
                    return input[0]
                elif len(input) == 2:
                    input, value = input
                    if name is not None: input.name = name
                    input.value = value
                    return input
            else:
                raise TypeError("The input specification is not valid: %s" % input)

            if not isinstance(result, gof.Result):
                raise TypeError("Unknown input type: %s, expected Result instance" % type(result), result)
            if update is not None and not isinstance(update, gof.Result):
                raise TypeError("Unknown update type: %s, expected Result instance" % type(update), update)
            if value is not None and isinstance(value, (gof.Result, SymbolicInput)):
                raise TypeError("The value for input %s should not be a Result or SymbolicInput instance (got: %s)" % (result, value))

            return In(result, name=name, value=value, update=update)
        else:
            raise TypeError("Unknown input type: %s, expected Result instance" % type(input), input)

    def wrap_out(output):
        if isinstance(output, SymbolicOutput):
            return output
        elif isinstance(output, gof.Result):
            return SymbolicOutput(output)
        else:
            raise TypeError("Unknown output type: %s (%s)" % (type(output), output))

    inputs = map(wrap_in, inputs)
    outputs = map(wrap_out, outputs) if isinstance(outputs, (list, tuple)) else wrap_out(outputs)

    defaults = [getattr(input, 'value', None) for input in inputs]

    if isinstance(mode, (list, tuple)): # "mode comparison" semantics
        if not mode:
            raise ValueError("Please provide at least one mode.")
        elif len(mode) == 1:
            fn = FunctionMaker(inputs, outputs, mode[0], accept_inplace = accept_inplace).create(defaults)
        else:              
            #return a different kind of function
            def dup_defaults():
                return [copy(default.value) if isinstance(default, gof.Container) else copy(default)
                        for default in defaults]
            makers = [FunctionMaker(inputs, outputs, m, accept_inplace = accept_inplace) for m in mode[1:]]
            fns = [maker.create(dup_defaults(), trustme = True) for maker in makers]
            builder = partial(SanityCheckFunction, fns, check_equal)
            maker1 = FunctionMaker(inputs, outputs, mode[0], accept_inplace = accept_inplace, function_builder = builder)
            fn = maker1.create(defaults)
    else:
        fn = FunctionMaker(inputs, outputs, mode, accept_inplace = accept_inplace).create(defaults)

    return fn



