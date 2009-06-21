"""Driver of graph construction, optimization, and linking.

"""
__docformat__ = "restructuredtext en"

import copy_reg
import cPickle

from functools import partial

import numpy
from .. import gof
import sys
import copy

import mode as mode_module
from io import *

def infer_reuse_pattern(env, outputs_to_disown):
    """
    Given an env and a list of variables, returns the list of all
    variables which may share the same underlying data storage as any of
    the specified variables. Used internally by function, FunctionMaker.

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
    contents of protected Variables. The outputs of the Env are protected by default.
    """

    def __init__(self, protected):
        self.protected = list(protected)

    def validate(self, env):
        if not hasattr(env, 'destroyers'):
            return True
        for r in self.protected + list(env.outputs):
            if env.destroyers(r):
                raise gof.InconsistencyError("Trying to destroy a protected Variable.", r)


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
    orig_inputs = [spec.variable for spec in input_specs]
    updates = [spec.update for spec in input_specs if spec.update]
    orig_outputs = [spec.variable for spec in output_specs] + updates

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

class AliasedMemoryError(Exception):
    """Memory is aliased that should not be"""
    pass


###
### Function
###

DUPLICATE = ['DUPLICATE'] # unique id object used as a placeholder for duplicate entries
class Function(object):
    """
    Type of the functions returned by theano.function or theano.FunctionMaker.create.


    `Function` is the callable object that does computation.  It has the storage of inputs and
    outputs, performs the packing and unpacking of inputs and return values.  It implements the
    square-bracket indexing so that you can look up the value of a symbolic node.

    Functions are copyable via {{{fn.copy()}}} and {{{copy.copy(fn)}}}.
    When a function is copied, this instance is duplicated.  Contrast with self.maker
    (instance of `FunctionMaker`) that is shared between copies.
    The meaning of copying a function is that the containers and their current values will all be duplicated.
    This requires that mutable inputs be copied, whereas immutable inputs may be shared between copies.



    A Function instance is hashable, on the basis of its memory address (its id).

    A Function instance is only equal to itself.
    
    A Function instance may be serialized using the `pickle` or `cPickle` modules.
    This will save all default inputs, the graph, and *** to the pickle file (WRITEME).

    """

    pickle_aliased_memory_strategy = 'warn'
    """How to deal with pickling finding aliased storage.

    Meaningful settings are: 'ignore', 'warn', 'raise'

    If the value is 'warn', then a message will be printed to stderr if aliased storage is
    dectected during pickle.dump.

    If the value is 'raise', then an AliasedMemoryError will be raised if aliased storage is
    detected during pickle.dump.
    
    """

    input_storage = None
    """list of Container instances"""

    output_storage = None
    """list of Container instances"""

    indices = None
    """list of (SymbolicInput|SymbolicInputKit, indices, [SymbolicInput,...]), one tuple for
    each input

    The first tuple element is the SymbolicInput object for the corresponding function input.

    The second and third tuple elements are used only by Kits, which are deprecated.
    """

    defaults = None
    """ list of 3-tuples, one 3-tuple for each input.

    Tuple element 0: Bool:  Is this input required at each function call?
    Tuple element 1: Bool:  Should this inputs value be reverted after each call?
    Tuple element 2: Any:  The value associated with this input.
    """

    unpack_single = None
    """Bool: for outputs lists of length 1, should the 0'th element be returned directly?"""

    return_none = None
    """Bool: whether the function should return None or not"""

    maker = None
    """FunctionMaker instance"""

    fn = None
    """a function that evaluates the graph.  Typically a linker's make_thunk method created this
    function."""

    finder = None
    """Dictionary mapping several kinds of things to containers.
    
    We set an entry in finder for:

    - the index of the input

    - the variable instance the input is based on

    - the name of the input

    All entries map to the container or to DUPLICATE if an ambiguity is detected
    """

    inv_finder = None
    """Dict. Reverse lookup of `finder`.

    It maps container -> SymbolicInput
    """

    def __init__(self, fn, input_storage, output_storage, indices, outputs, defaults, unpack_single, return_none, maker):
        """
        Initialize attributes. create finder, inv_finder.
        """

        self.fn = fn
        self.input_storage = input_storage
        self.output_storage = output_storage
        self.indices = indices
        self.outputs = outputs
        self.defaults = defaults
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.maker = maker

        # we'll be popping stuff off this `containers` object.  It's a copy
        containers = list(self.input_storage) 
        finder = {}
        inv_finder = {}

        def distribute(indices, cs, value):
            input.distribute(value, indices, cs)
            for c in cs:
                c.provided += 1
        #def assign(c, v):
            #c.data = v

        #setters = []
        # Initialize the storage
        for i, ((input, indices, sinputs), (required, refeed, value)) in enumerate(zip(self.indices, defaults)):
            if indices is None: # this is true iff input is not a SymbolicInputKit
                c = containers[0]  #containers is being used as a stack. Here we pop off the next one.
                if input.strict:
                    c.strict = True

                # Whether the default value will be directly accessible within
                # the function's container (c.copy_from_container = None), or
                # if the function has its own container and thus needs to copy
                # the default value at each call (c.copy_from_container =
                # pointer towards it).
                # Shared containers are only used for implicit inputs (so that
                # there is no risk of overwriting their content with a user-
                # provided value).
                c.copy_from_container = None
                if value is not None:
                    # Always initialize the storage.
                    if isinstance(value, gof.Container):
                        # There is no point in obtaining the current value
                        # stored in the container, since:
                        # - for an implicit input, the container is shared
                        # - for a non implicit input, the value may change
                        # the function is called.
                        if not input.implicit:
                            c.copy_from_container = value
                        else:
                            # Safety check: the container will be shared, so
                            # there should be no need to refeed the default
                            # value.
                            assert not refeed
                    else:
                        c.value = value
                c.required = required
                c.implicit = input.implicit
                c.provided = 0 # this is a count of how many times the input has been provided (reinitialized to 0 on __call__)
                finder[i] = c
                finder[input.variable] = c
                finder[input.name] = c if input.name not in finder else DUPLICATE
                # inv_finder maps the container to the input (useful for one error message)
                inv_finder[c] = input
                #setters.append(partial(assign, c))
                containers[:1] = []
            else:
                # TODO The following code may need to do something to handle
                # implicit inputs.

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
                #setters.append(f)
                # For each input in the kit and its corresponding container, we put an entry in finder.
                # This allows the user to micro-manage elements of the kit if need be.
                # All containers inherit the required field and have their own "provided" counter
                for c, sin in zip(cs, sinputs):
                    finder[sin.variable] = c
                    finder[sin.name] = c
                    finder[sin.name] = c if sin.name not in finder else DUPLICATE
                    inv_finder[c] = input
                    c.required = required
                    c.provided = 0
                containers[:len(indices)] = []

        self.finder = finder
        self.inv_finder = inv_finder

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
            def __contains__(self, item):
                return finder.__contains__(item)

        # this class is important in overriding the square-bracket notation:
        #     fn.container[x]
        # self reference is available via the closure on the class
        class ContainerAttribute(object):
            def __getitem__(self, item):
                return finder[item]
            def __contains__(self, item):
                return finder.__contains__(item)
            # You cannot set the container

        self._value = ValueAttribute()
        self._container = ContainerAttribute()

    def __contains__(self, item):
        return self.value.__contains__(item)

    def __getitem__(self, item):
        return self.value[item]

    def __setitem__(self, item, value):
        self.value[item] = value
        
    
    def __copy__(self):
        defaults = [default for _1, _2, default in self.defaults]
        cpy = self.maker.create(defaults, trustme = True)
        for (input,_1,_2), here, there in zip(self.indices, self.input_storage, cpy.input_storage):
            if input.mutable and here is not None:
                there.data = copy.copy(here.data)
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

        # Check if inputs are missing, or if inputs were set more than once, or
        # if we tried to provide inputs that are supposed to be implicit.
        # Also initialize default values that are obtained from an external
        # container. This is required because this container's value may be
        # modified between function calls.
        # Other types of default values should not need to be re-initialized:
        # - shared containers are updated automatically
        # - default values defined directly by their value are re-fed into the
        # input storage after a function call, and any modification possibly
        # made to them (for mutable types) will be reflected there as well.
        for c in self.input_storage:
            if c.required and not c.provided:
                raise TypeError("Missing required input: %s" % getattr(self.inv_finder[c], 'variable', self.inv_finder[c]))
            if c.provided > 1:
                raise TypeError("Multiple values for input: %s" % getattr(self.inv_finder[c], 'variable', self.inv_finder[c]))
            if c.implicit and c.provided > 0:
                raise TypeError('Tried to provide value for implicit input: %s'
                        % getattr(self.inv_finder[c], 'variable',
                            self.inv_finder[c]))
            if c.provided == 0 and c.copy_from_container is not None:
                # Copy default value from another (non shared) container.
                # Safety check, may be removed in the future.
                assert not c.implicit
                c.value = c.copy_from_container.value
                # TODO Would it be better to use self[..] = value?

        # Do the actual work
        self.fn()

        # Retrieve the values that were computed
        outputs = [x.data for x in self.output_storage]

        #remove internal references to required inputs
        #these can't be re-used anyway
        for x in self.input_storage:
            if c.required:
                c.storage[0] = None

        # if we are allowing garbage collection, remove the input and output reference from the internal
        # storage cells
        if getattr(self.fn, 'allow_gc', False):
            assert len(self.output_storage) == len(self.maker.env.outputs)
            for o_container, o_variable in zip(self.output_storage, self.maker.env.outputs):
                if o_variable.owner is not None:
                    # this node is the variable of computation
                    # WARNING: This circumvents the 'readonly' attribute in x
                    o_container.storage[0] = None

        # Update the inputs that have an update function
        for input, storage in reversed(zip(self.maker.expanded_inputs, self.input_storage)):
            if input.update is not None:
                storage.data = outputs.pop()

        # Put default values back in the storage
        for i, (required, refeed, value) in enumerate(self.defaults):
            if refeed:
                if isinstance(value, gof.Container):
                    value = value.storage[0]
                self[i] = value

        if self.return_none:
            return None
        elif self.unpack_single and len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    value = property(
        lambda self: self._value,
        None, # this property itself is not settable
        doc="""dictionary-like access to the values associated with Variables""")
    container = property(
        lambda self: self._container,
        None, # this property itself is not settable
        doc="""dictionary-like access to the containers associated with Variables""")

# pickling/deepcopy support for Function

def _pickle_Function(f):
    #copy of the input storage list
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

    inputs_data = [x.data for x in f.input_storage]

    # HACK to detect aliased storage.
    # aliased relationships will not be preserved across the pickle operation
    if not (f.pickle_aliased_memory_strategy == 'ignore'):
        all_data = defaults + inputs_data
        for i, d_i in enumerate(all_data):
            for j, d_j in enumerate(all_data):
                if (i < j) and isinstance(d_i, numpy.ndarray) and isinstance(d_j, numpy.ndarray):
                    if numpy.may_share_memory(d_i, d_j):
                        if f.pickle_aliased_memory_strategy == 'warn':
                            print >> sys.stderr, ('WARNING: '
                                    'aliased relationship between Function arguments '
                                    'will not be preserved by un-pickling operation')
                            #print >> sys.stderr, d_i, d_j, id(d_i), id(d_j)
                        else:
                            raise AliasedMemoryError(d_i, d_j)

    rval = (_constructor_Function, (f.maker, defaults, inputs_data))
    return rval

def _constructor_Function(maker, defaults, data):
    f = maker.create(defaults, trustme = True)
    assert len(f.input_storage) == len(data)
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

        for fn in self.others:
            for stor1, stor2 in zip(self.input_storage, fn.input_storage):
                stor2.value = copy.copy(stor1.value)

        variables = super(SanityCheckFunction, self).__call__(*args, **kwargs)

        all_outputs = [copy.copy(c.value) for c in self.output_storage] # we keep a copy to make sure it's not overwritten
        for fn in self.others:
            fn(*args, **kwargs)

            for i, (c1, c2, input) in enumerate(zip(self.input_storage, fn.input_storage, self.maker.inputs)):
                if not input.mutable:
                    if not self.check_equal(c1.value, c2.value):
                        name = c2.name
                        raise ValueError("Input #%i%s using %s and %s differs."
                                         % (i,
                                            " (%s)" % name if name else "",
                                            self.maker.mode,
                                            fn.maker.mode),
                                         c1.value, c2.value)

            # This checks all output storage (this includes state variables that we updated)
            # This is ok because the variables of a call stick around in their storage
            for i, (r1, c2) in enumerate(zip(all_outputs, fn.output_storage)):
                r2 = c2.value
                if not self.check_equal(r1, r2):
                    name = c2.name
                    raise ValueError("Variable #%i%s using %s and %s differs."
                                     % (i,
                                        " (%s)" % name if name else "",
                                        self.maker.mode,
                                        fn.maker.mode),
                                     r1, r2)
        return variables



###
### FunctionMaker
###

NODEFAULT = ['NODEFAULT']
class FunctionMaker(object):
    """`FunctionMaker` is the class to `create` `Function` instances.
    
    This class has the env, the optimizer, and the linker.  When copying a `Function`, there is
    no need to duplicate the `FunctionMaker` instance.  Deepcopy still copies both, which can
    variable in re-compilation.

    """

    @staticmethod
    def wrap_in(input):
        if isinstance(input, (SymbolicInput, SymbolicInputKit)):
            return input
        elif isinstance(input, gof.Variable):
            # r -> SymbolicInput(variable=r)
            return SymbolicInput(input)
        elif isinstance(input, (list, tuple)):
            # (r, u) -> SymbolicInput(variable=r, update=u)
            if len(input) == 2:
                return SymbolicInput(input[0], update = input[1])
            else:
                raise TypeError("Expected two elements in the list or tuple.", input)
        else:
            raise TypeError("Unknown input type: %s (%s), expected Variable instance", type(input), input)

    @staticmethod
    def expand_in(sinput, rinputs):
        # For SymbolicInputKits, this extracts a list of SymbolicInput instances
        # and corresponding indices such that these SymbolicInputs are representative
        # of some of the Variable instances in inputs.
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
        elif isinstance(output, gof.Variable):
            return SymbolicOutput(output)
        else:
            raise TypeError("Unknown output type: %s (%s)", type(output), output)

    def __init__(self, inputs, outputs, 
            mode = None, accept_inplace = False, function_builder = Function):
        """
        :type inputs: a list of SymbolicInput instances

        :type outputs: a list of SymbolicOutput instances
                    outputs may also be a single Variable (not a list), in which
                    case the functions produced by FunctionMaker will return
                    their output value directly

        :param mode: a Mode instance telling FunctionMaker how to optimize and link.  None
        means to use the `default_mode`.

        :param accept_inplace: True iff it is acceptable to have inplace operations
                    in the graph from the inputs to the outputs
        """

        mode = mode if mode is not None else mode_module.default_mode

        # Handle the case where inputs and/or outputs is a single Variable (not in a list)
        unpack_single = False
        return_none = False
        if outputs is None:
            return_none = True
            outputs = []
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.variable for o in outputs] + [i.update for i in inputs if getattr(i, 'update', False)])
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices], [])

        # make the env
        env, additional_outputs = std_env(expanded_inputs, outputs, accept_inplace)
        self.env = env

        # Fetch the mode and then the optimizer and linker
        mode = mode_module.predefined_modes.get(mode, mode)
        optimizer, linker = mode.optimizer, copy.copy(mode.linker)

        # optimize the env
        optimizer(env)

        # initialize the linker
        if not hasattr(linker, 'accept'):
            raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
                             "or one of %s" % mode_module.predefined_linkers.keys())

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
        self.return_none = return_none
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
            # Replace any default value given as a variable by its container.
            # Note that this makes sense only in the context of shared variables,
            # but for now we avoid dealing directly with them to avoid dependency
            # on the shared variables work-in-progress repository.
            if isinstance(default, gof.Variable):
                default = default.container

            __default = default

            if isinstance(default, gof.Container) and input.implicit:
                # If the default is a gof.Container and it is an implicit
                # input, this means we want to share the same storage. This is
                # done by appending default.storage to input_storage
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
                        and all(isinstance(x, gof.Container) for x in default) \
                        and input.implicit:
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
                    if (trustme or (isinstance(__default, gof.Container)
                        and input.implicit)):
                        _defaults.append((False, False, None))
                    else:
                        # This might catch some bugs early
                        raise ValueError("A default (initial) value is required for an input which can update itself.", input)
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    if (trustme or (isinstance(__default, gof.Container)
                        and input.implicit)):
                        _defaults.append((False, False, None))
                    else:
                        # No default, so this is a required input. Nothing to feed back, initial value is None.
                        _defaults.append((True, False, None))
                else:
                    # Default value. It is not required, but we want to put it back into the storage
                    # everytime so it behaves like most programming languages' default values.
                    # Note (OD): why is it not required? If it was not put back
                    # into the storage, then the default value may be incorrect
                    # on subsequent calls. Thus, setting 'refeed' to True seems
                    # very important here.
                    _defaults.append((False, True, default))
        defaults = _defaults

        # Get a function instance
        _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage)
        fn = self.function_builder(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self.return_none, self)
        return fn


def _pickle_FunctionMaker(fm):
    outputs = None if fm.return_none else (fm.outputs[0] if fm.unpack_single else fm.outputs)
    rval = (_constructor_FunctionMaker, (fm.inputs, outputs, fm.mode, fm.accept_inplace))
    return rval

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




def function(inputs, outputs, mode=None, accept_inplace = False):
    """
    Return a function calculating the outputs from the inputs.

    :param inputs: list of `SymbolicInput` or `In` instances

    :param outputs: a SymbolicOutput or a list of `SymbolicOutput` or `Out` instances.   The return
        value of the returned function will match the format of this argument (either the value
        itself or a list of one or more return values)

    :param mode: a descriptive string or a Mode instance. (Default of None means to use
    `mode.default_mode` (See below for descriptive string list).
    
    Currently, the library provides the following mode strings:

     - SANITY_CHECK TODO: NotImplemented

     - FAST_COMPILE (apply only optimization that are fast to apply)

     - FAST_RUN (default) (optimize without too much time)

     - EXPENSIVE_OPTIMIZATION TODO: NotImplemented

    :param accept_inplace:  True iff the graph can contain inplace operations prior to the
    optimization phase (default is False)

    Every element of the input list will be upgraded to an `In` instance if necessary,
    using the rules implemented by the `convert_function_input` function.

    Similarly, every element of the output list will be upgraded to an
    `Out` instance if necessary:

    * a `Variable` instance r will be upgraded like `Out`(r)


    Random Numbers
    --------------

    If your computation involves random numbers, then you have to pass the `RandomKit` as an
    input argument.  That RandomKit must have a name to be able to seed the generator.  To seed
    the generator, use the `__setitem__` method: 

    ..code-block: python
    
        f[<kitname>] = seed   #re-seed the elements of a RandomKit

    """
    mode = mode if mode is not None else mode_module.default_mode


    inputs = map(convert_function_input, inputs)
    if outputs is not None:
        outputs = map(FunctionMaker.wrap_out, outputs) if isinstance(outputs, (list, tuple)) else FunctionMaker.wrap_out(outputs)

    defaults = [getattr(input, 'value', None) for input in inputs]

    mode = mode_module.predefined_modes.get(mode, mode)
    if isinstance(mode, (list, tuple)): # "mode comparison" semantics
        if not mode:
            raise ValueError("Please provide at least one mode.")
        elif len(mode) == 1:
            fn = FunctionMaker(inputs, outputs, mode[0], accept_inplace = accept_inplace).create(defaults)
        else:              
            #return a different kind of function
            def dup_defaults():
                # TODO This may need to be changed to use containers as defaults.
                return [copy.copy(default.value) if isinstance(default, gof.Container) else
                        copy.copy(default)
                        for default in defaults]
            makers = [FunctionMaker(inputs, outputs, m, accept_inplace = accept_inplace) for m in mode[1:]]
            fns = [maker.create(dup_defaults(), trustme = True) for maker in makers]
            builder = partial(SanityCheckFunction, fns, check_equal)
            maker1 = FunctionMaker(inputs, outputs, mode[0], accept_inplace = accept_inplace, function_builder = builder)
            fn = maker1.create(defaults)
    else:
        Maker = getattr(mode, 'function_maker', FunctionMaker)
        fn = Maker(inputs, outputs, mode, accept_inplace = accept_inplace).create(defaults)

    return fn



def convert_function_input(input):
    """
    Upgrade a input shortcut to an In instance.
    
    The rules for upgrading are as follows:

    - a `Variable` instance r will be upgraded like `In`(r)

    - a tuple (name, r) will be `In`(r, name=name)
    
    - a tuple (r, val) will be `In`(r, value=value, autoname=True)

    - a tuple ((r,up), val) will be `In`(r, value=value, update=up, autoname=True)

    - a tuple (name, r, val) will be `In`(r, name=name, value=value)

    - a tuple (name, (r,up), val) will be `In`(r, name=name, value=val, update=up, autoname=True)


    """
    if isinstance(input, (SymbolicInput, SymbolicInputKit)):
        return input
    elif isinstance(input, gof.Constant):
        raise TypeError('A Constant instance is not a legal function input', input)
    elif isinstance(input, gof.Variable):
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
            (variable, update), value = input
        elif isinstance(input[0], gof.Variable):
            if len(input) == 1:
                variable, update, value = input[0], None, None
            elif len(input) == 2:
                (variable, value), update = input, None
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

        if not isinstance(variable, gof.Variable):
            raise TypeError("Unknown input type: %s, expected Variable instance" % type(variable), variable)
        if update is not None and not isinstance(update, gof.Variable):
            raise TypeError("Unknown update type: %s, expected Variable instance" % type(update), update)
        if value is not None and isinstance(value, (gof.Variable, SymbolicInput)):
            raise TypeError("The value for input %s should not be a Variable or SymbolicInput instance (got: %s)" % (variable, value))

        return In(variable, name=name, value=value, update=update)
    else:
        raise TypeError("Unknown input type: %s, expected Variable instance" % type(input), input)

