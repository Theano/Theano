"""Convenient driver of graph construction, optimization, and linking."""

import copy_reg
import cPickle

from functools import partial


import numpy
import gof
import sys
from copy import copy

def check_equal(x, y):
    """
    Returns True iff x[0] and y[0] are equal (checks the dtype and
    shape if x and y are numpy.ndarray instances). Used internally.
    """
    x, y = x[0], y[0]
    if isinstance(x, numpy.ndarray) or isinstance(y, numpy.ndarray):
        if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
            raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
    else:
        if x != y:
            raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})

def infer_reuse_pattern(env, outputs_to_disown):
    """
    Given an env and a list of results, returns the list of all
    results which may share the same underlying data storage as any of
    the specified results. Used internally by function, FunctionMaker.
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

# If a string is passed as the linker argument in the constructor for
# Mode, it will be used as the key to retrieve the real linker in this
# dictionary
predefined_linkers = {
    'py'   : gof.PerformLinker(),
    'c'    : gof.CLinker(),
    'c|py' : gof.OpWiseCLinker(),
    'c&py' : gof.DualLinker(checker = check_equal)
    }

default_linker = 'c|py'

def register_linker(name, linker):
    """Add a `Linker` which can be referred to by `name` in `Mode`."""
    if name in predefined_linkers:
        raise ValueError('Linker name already taken: %s' % name)
    predefined_linkers[name] = linker


# If a string is passed as the optimizer argument in the constructor
# for Mode, it will be used as the key to retrieve the real optimizer
# in this dictionary
predefined_optimizers = {
    None    : lambda env: None,
    'merge' : gof.MergeOptimizer(),
    }
default_optimizer = 'merge'

def register_optimizer(name, opt):
    """Add a `Optimizer` which can be referred to by `name` in `Mode`."""
    if name in predefined_optimizers:
        raise ValueError('Optimizer name already taken: %s' % name)
    predefined_optimizers[name] = opt


class Mode(object):
    """
    The Mode represents a way to optimize and then link a computation
    graph.

     * optimizer -> a structure of type Optimizer. An Optimizer may
       simplify the math, put similar computations together, improve
       numerical stability and various other improvements.
     * linker -> a structure of type Linker. A Linker decides which
       implementations to use (C or Python, for example) and how to
       string them together to perform the computation.

    See predefined_linkers, predefined_optimizers and also
    predefined_modes.
    """
    
    def __init__(self, linker = default_linker, optimizer = default_optimizer):
        self.__setstate__((linker, optimizer))

    def __getstate__(self):
        return (self.provided_linker, self.provided_optimizer)

    def __setstate__(self, (linker, optimizer)):
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

# If a string is passed as the mode argument in function or
# FunctionMaker, the Mode will be taken from this dictionary using the
# string as the key
predefined_modes = {'FAST_COMPILE': Mode('py', 'merge')} 
default_mode = 'FAST_COMPILE'

def register_mode(name, mode):
    """Add a `Mode` which can be referred to by `name` in `function`."""
    if name in predefined_modes:
        raise ValueError('Mode name already taken: %s' % name)
    predefined_modes[name] = mode



class SymbolicInput(object):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    result: a Result instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to result.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    update: Result instance (default: None)
        value (see previous) will be replaced with this expression result after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.
    """

    def __init__(self, result, name=None, update=None, mutable=None, strict=False, autoname=True):
        self.result = result
        self.name = result.name if (autoname and name is None) else name
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string! (got: %s)" % self.name)
        self.update = update
        self.mutable = mutable if (mutable is not None) else (update is not None)
        self.strict = strict

    def __str__(self):
        if self.update:
            return "In(%s -> %s)" % (self.result, self.update)
        else:
            return "In(%s)" % self.result

    def __repr__(self):
        return str(self)


class SymbolicInputKit(object):
    """
    Represents a group ("kit") of SymbolicInputs. If fed into function or
    FunctionMaker, only the inputs which are needed to compile the function
    properly will be taken.

    A SymbolicInputKit provides the distribute function in order to set or
    initialize several inputs from a single value. Specialized Kits should
    override it.
    """

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError('naem must be a string (got: %s)' % name)
        self.name = name
        self.sinputs = []
        self.results = []

    def add_input(self, sinput):
        """
        Add a SymbolicInput to this SymbolicInputKit. It will be given the
        next available index.
        """
        self.sinputs.append(sinput)
        self.results.append(sinput.result)

    def distribute(self, value, indices, containers):
        """
        Given a list of indices corresponding to SymbolicInputs in this kit
        as well as a corresponding list of containers, initialize all the
        containers using the provided value.
        """
        raise NotImplementedError

    def complete(self, inputs):
        """
        Given inputs (a list of Result instances), checks through all
        the SymbolicInputs in the kit and return a sorted list of
        indices and a list of their corresponding SymbolicInputs such
        that each of them represents some result in the inputs list.

        Not all the provided inputs will have a corresponding
        SymbolicInput in the kit.
        """
        ret = []
        for input in inputs:
            try:
                i = self.results.index(input)
                ret.append((i, self.sinputs[i]))
            except ValueError:
                pass
        ret.sort()
        return zip(*ret)


class In(SymbolicInput):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    result: a Result instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to result.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    value: Any type.
        The initial/default value for this input. If update is None, this input acts just like
        an argument with a default value in Python. If update is not None, changes to this
        value will "stick around", whether due to an update or a user's explicit action.

    update: Result instance (default: None)
        value (see previous) will be replaced with this expression result after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.
    """
    def __init__(self, result, name=None, value=None, update=None, mutable=None, strict=False, autoname=True):
        super(In, self).__init__(result, name, update, mutable, strict, autoname)
        self.value = value


class SymbolicOutput(object):
    """
    Represents a symbolic output for use with function or FunctionMaker.

    borrow: set this to True to indicate that a reference to
            function's internal storage may be returned. A value
            returned for this output might be clobbered by running
            the function again, but the function might be faster.
    """
    
    def __init__(self, result, borrow=False):
        self.result = result
        self.borrow = borrow

Out = SymbolicOutput



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
                raise gof.InconsistencyError("Trying to destroy a protected Result.")


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
    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not spec.mutable))
    return env, map(SymbolicOutput, updates)


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

    def __init__(self, inputs, outputs, mode = 'FAST_RUN', accept_inplace = False):
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
        _inputs = gof.graph.inputs([o.result for o in outputs])
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
        self.mode = mode
        self.accept_inplace = accept_inplace

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

            # If the default is a gof.Container, this means we want to share
            # the same storage. This is done by appending default.storage
            # to input_storage
            if isinstance(default, gof.Container):
                if indices is not None:
                    raise TypeError("Cannot take a Container instance as default for a SymbolicInputKit.")
                input_storage.append(default.storage)
                default = None
            # If the input is a SymbolicInputKit, it represents more than
            # one storage unit. The indices and subinputs lists represent which
            # of the kit's inputs are active in this graph, so we make as many
            # storage units as needed
            elif isinstance(input, SymbolicInputKit):
                input_storage += [[None] for i in indices]
            # Normal case: one new, independent storage unit
            else:
                input_storage.append([None])

            # Filling _defaults. Each entry is a tuple of three elements:
            # (required, refeed, value)
            # - required means that the user must provide a value when calling the function
            # - refeed means that we want to put the default back in the storage after each function call
            # - value is the value that will be put in the storage initially

            # Even though a SymbolicInputKit represents more than one input,
            # we still only have one entry for the defaults list.
            if isinstance(input, SymbolicInputKit):
                if default is None:
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
                        _defaults.append((False, False, default))
                    else:
                        # This might catch some bugs early
                        raise ValueError("A default (initial) value is required for an input which can update itself.", input)
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    # No default, so this is a required input. Nothing to feed back, initial value is None.
                    _defaults.append((True, False, None))
                else:
                    # Default value. It is not required, but we want to put it back into the storage
                    # everytime so it behaves like most programming languages' default values
                    _defaults.append((False, True, default))
        defaults = _defaults

        # Get a function instance
        _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage)
        fn = Function(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self)
        return fn


def _pickle_FunctionMaker(fm):
    return (_constructor_FunctionMaker, (fm.inputs, fm.outputs, fm.mode, fm.accept_inplace))

def _constructor_FunctionMaker(*args):
    return FunctionMaker(*args)

copy_reg.pickle(FunctionMaker, _pickle_FunctionMaker)


def _pickle_slice(s):
    return (slice, (s.start, s.stop, s.step))

copy_reg.pickle(slice, _pickle_slice)




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
        def set(c, v):
            c.data = v

        setters = []
        # Initialize the storage
        for i, ((input, indices, sinputs), (required, refeed, value)) in enumerate(zip(self.indices, defaults)):
            if indices is None: # this is true iff input is not a SymbolicInputKit
                c = containers[0]
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
                setters.append(partial(set, c))
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
                raise TypeError("Missing required input: %s" % self.inv_finder[c].result)
            if c.provided > 1:
                raise TypeError("Multiple values for input: %s" % self.inv_finder[c].result)
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


def _pickle_Function(f):
    ins = list(f.input_storage)
    defaults = []
    for (input, indices, inputs), (required, refeed, default) in zip(f.indices, f.defaults):
        if isinstance(input, SymbolicInputKit):
            defaults.append(default)
            ins[:len(indices)] = []
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


def function(inputs, outputs, mode='FAST_RUN', accept_inplace = False):
    """
    Return a function calculating the outputs from the inputs.

    inputs -> list of SymbolicInput or In instances
    outputs -> a SymbolicOutput or a list of SymbolicOutput or Out instances
      The return value of the returned function will match the format of this
      argument (either the value itself or a list of one or more return values)
    mode -> a descriptive string or a Mode instance; descriptive strings can be one of:
      * SANITY_CHECK
      * FAST_COMPILE
      * FAST_RUN (default)
      * EXPENSIVE_OPTIMIZATION
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
                        
            return In(result, name=name, value=value, update=update)
        else:
            raise TypeError("Unknown input type:", type(input), input)

    def wrap_out(output):
        if isinstance(output, SymbolicOutput):
            return output
        elif isinstance(output, gof.Result):
            return SymbolicOutput(output)
        else:
            raise TypeError("Unknown output type: %s (%s)" % (type(output), output))

    inputs = map(wrap_in, inputs)
    outputs = map(wrap_out, outputs) if isinstance(outputs, (list, tuple)) else wrap_out(outputs)

    fn = FunctionMaker(inputs, outputs, mode, accept_inplace = accept_inplace).create([getattr(input, 'value', None) for input in inputs])

    return fn






class OpFromGraph(gof.Op):
    """
    This create an L{Op} from a list of input results and a list of output
    results.

    The signature is the same as the signature of L{FunctionFactory}
    and/or function and the resulting L{Op}'s perform will do the same
    operation as::
      function(inputs, outputs, **kwargs)

    Take note that the following options, if provided, must take the
    value(s) listed below:
      unpack_single = False
      borrow_outputs = False

    OpFromGraph takes an additional input, grad_depth. If grad_depth
    is n, OpFromGraph will make special Ops for gradients up to the
    nth level, allowing the user to differentiate this op up to n
    times. The parameter defaults to 1. If grad_depth == 0, the op
    will not be differentiable.

    Example:
      x, y, z = tensor.scalars('xyz')
      e = x + y * z
      op = OpFromGraph([x, y, z], [e], linker='c')
      # op behaves like a normal theano op
      e2 = op(x, y, z) + op(z, y, x)
      fn = function([x, y, z], [e2])
    """
    
    def __init__(self, inputs, outputs, grad_depth = 1, **kwargs):
        self.fn = function(inputs, outputs, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self.input_types = [input.type for input in inputs]
        self.output_types = [output.type for output in outputs]
        if grad_depth > 0:
            import gradient as G
            output_grads = [t() for t in self.output_types]
            gd = G.grad_sources_inputs(zip(self.outputs, output_grads), self.inputs)
            gs = map(gd.get, self.inputs)
            self.grad_ops = []
            for g in gs:
                if g is None:
                    self.grad_ops.append(lambda *args: None)
                else:
                    self.grad_ops.append(OpFromGraph(inputs + output_grads,
                                                     [g],
                                                     grad_depth = grad_depth - 1))

    def make_node(self, *inputs):
        for input, type in zip(inputs, self.input_types):
            if not type == input.type:
                raise TypeError("Wrong type, expected %s but got %s" % type, input.type)
        return gof.Apply(self,
                         inputs,
                         [type() for type in self.output_types])

    def perform(self, node, inputs, outputs):
        results = self.fn(*inputs)
        for output, result in zip(outputs, results):
            output[0] = result

    def grad(self, inputs, output_grads):
        if hasattr(self, 'grad_ops'):
            return [go(*(inputs + output_grads)) for go in self.grad_ops]
        else:
            raise NotImplementedError

