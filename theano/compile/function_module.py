"""Driver of graph construction, optimization, and linking.

"""
__docformat__ = "restructuredtext en"

import copy
import copy_reg
import cPickle
import itertools
import time

import numpy

import theano
from theano import gof
from theano.gof.python25 import partial
import mode as mode_module
from io import In, SymbolicInput, SymbolicInputKit, SymbolicOutput
from theano.configdefaults import config

import logging
_logger = logging.getLogger('theano.compile.function_module')


def alias_root(v):
    """Return the variable to which v is aliased by view_maps and destroy_maps"""
    if v.owner is None: return v
    vmap = getattr(v.owner.op, 'view_map', {})
    dmap = getattr(v.owner.op, 'destroy_map', {})
    outpos = v.owner.outputs.index(v)
    v_views = vmap.get(outpos, []) + dmap.get(outpos, [])
    if len(v_views) > 1:
        raise NotImplementedError()
    elif v_views:
        return alias_root(v.owner.inputs[v_views[0]])
    else:
        return v


def view_tree_set(v, treeset):
    """Add to `treeset` all variables that are views of v, given that v is not a view"""
    treeset.add(v)
    for cl, v_input_pos_to_cl in v.clients:
        if cl == 'output':
            continue
        vmap = getattr(cl.op, 'view_map', {})
        dmap = getattr(cl.op, 'destroy_map', {})
        for opos, iposlist in vmap.items() + dmap.items():
            if v_input_pos_to_cl in iposlist:
                if cl.outputs[opos] not in treeset:
                    view_tree_set(cl.outputs[opos], treeset)


def infer_reuse_pattern(env, outputs_to_disown):
    """
    Given an env and a list of variables, returns the list or set of all variables which may
    share the same underlying data storage as any of the specified variables. Used internally
    by function, FunctionMaker.

    This list (or set) is also refered to as no_recycling sometimes, especially by linker code.
    """
    rval = set()
    for o in outputs_to_disown:
        view_tree_set(alias_root(o), rval)
    # remove from rval all of the inputs, constants, values.
    rval = set(r for r in rval if r.owner is not None)

    return rval


def env_updated_vars(env, expanded_inputs):
    """
    Reconstruct the full "updates" dictionary, mapping from Env input
    variables to the env outputs that will replace their values.

    :rtype: dict variable -> variable
    """
    updated_vars = {}
    potential_values = list(env.outputs)  # copy the list
    if len(expanded_inputs) != len(env.inputs):
        raise ValueError('expanded_inputs must match len(env.inputs)')
    for e_input, ivar in reversed(zip(expanded_inputs, env.inputs)):
        if e_input.update is not None:
            updated_vars[ivar] = potential_values.pop()
    return updated_vars


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
                raise TypeError("Graph must not contain inplace operations", node, node.op)
            else:
                env.extend(gof.DestroyHandler())
                break

    # We need to protect all immutable inputs from inplace operations.
    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not (spec.mutable or (hasattr(env, 'destroyers') and env.destroyers(input)))))

    # If named nodes are replaced, keep the name
    env.extend(gof.toolbox.PreserveNames())
    return env, map(SymbolicOutput, updates)

class AliasedMemoryError(Exception):
    """Memory is aliased that should not be"""
    pass


###
### Function
###

def register_DeepCopyOp_c_code(typ, code):
    """ Tell DeepCopyOp how to generate C code for a Theano Type

    :param typ: A Theano type. It must be the Theano class itself and not an
                instance of the class.
    :param code: C code that deep copies the Theano type 'typ'.
                 Use %(iname)s and %(oname)s for the input and output C
                 variable names respectively.
    """
    DeepCopyOp.c_codes[typ] = code


class DeepCopyOp(theano.gof.Op):
    c_codes = {}  # Theano Type, code

    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, x):
        return theano.gof.Apply(self, [x], [x.type()])

    def perform( self, node, args, outs):
        if hasattr(args[0],'copy'):
            #when args[0] is a an ndarray of 0 dimensions,
            #this return a numpy.dtype and not an ndarray
            #So when the args have a copy attribute we use it
            #as this don't have this problem
            outs[0][0] = args[0].copy()
        else:
            outs[0][0] = copy.deepcopy(args[0])

    def c_code_cache_version(self):
        return (1)

    def c_code(self, node, name, inames, onames, sub):
        iname = inames[0]
        oname = onames[0]
        fail = sub['fail']
        if isinstance(node.inputs[0].type, theano.tensor.TensorType):
            return """
        Py_XDECREF(%(oname)s);

        %(oname)s = (PyArrayObject*)PyArray_NewCopy(%(iname)s,NPY_ANYORDER);

        if (!%(oname)s)
        {
            PyErr_SetString(PyExc_ValueError, "DeepCopyOp: the copy failed!");
            %(fail)s;
        }

        """%locals()
        elif node.inputs[0].type.__class__ in self.c_codes:
            return self.c_codes[node.inputs[0].type.__class__] % locals()
        else:
            super(DeepCopyOp, self).c_code(node, name, inames, onames, sub)


class ViewOp(theano.gof.Op):
    def __init__(self):
        self.view_map={0:[0]}

    def __str__(self):
        return self.__class__.__name__

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, x):
        return theano.gof.Apply(self, [x], [x.type()])

    def perform( self, node, args, outs):
        outs[0][0] = args[0]

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, args, g_outs):
        return g_outs

deep_copy_op = DeepCopyOp()
view_op      = ViewOp()



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
        self.profile = None # reassigned in FunctionMaker.create

        # We will be popping stuff off this `containers` object.  It is a copy.
        containers = list(self.input_storage)
        finder = {}
        inv_finder = {}

        def distribute(indices, cs, value):
            input.distribute(value, indices, cs)
            for c in cs:
                c.provided += 1
        #def assign(c, v):
            #c.data = v

        # Store the list of names of named inputs.
        named_inputs = []
        # Count the number of un-named inputs.
        n_unnamed_inputs = 0

        #setters = []
        # Initialize the storage
        # this loop works by modifying the elements (as variable c) of self.input_storage inplace.
        for i, ((input, indices, sinputs), (required, refeed, value)) in enumerate(zip(self.indices, defaults)):
            if indices is None: # this is true iff input is not a SymbolicInputKit
                c = containers[0]  #containers is being used as a stack. Here we pop off the next one.
                c.strict = getattr(input, 'strict', False)
                c.allow_downcast = getattr(input, 'allow_downcast', None)

                if value is not None:
                    # Always initialize the storage.
                    if isinstance(value, gof.Container):
                        # There is no point in obtaining the current value
                        # stored in the container, since the container is
                        # shared.
                        # For safety, we make sure 'refeed' is False, since
                        # there is no need to refeed the defaullt value.
                        assert not refeed
                    else:
                        c.value = value
                c.required = required
                c.implicit = input.implicit
                c.provided = 0 # this is a count of how many times the input has been provided (reinitialized to 0 on __call__)
                finder[i] = c
                finder[input.variable] = c
                if input.name not in finder:
                    finder[input.name] = c
                else:
                    finder[input.name] = DUPLICATE
                if input.name is None:
                    n_unnamed_inputs += 1
                else:
                    named_inputs.append(input.name)
                #backport
                #finder[input.name] = c if input.name not in finder else DUPLICATE
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
                if input.name not in finder:
                    finder[input.name] = f
                else:
                    finder[input.name] = DUPLICATE
                #backport
                #finder[input.name] = f if input.name not in finder else DUPLICATE
                #setters.append(f)
                # For each input in the kit and its corresponding container, we put an entry in finder.
                # This allows the user to micro-manage elements of the kit if need be.
                # All containers inherit the required field and have their own "provided" counter
                for c, sin in zip(cs, sinputs):
                    finder[sin.variable] = c
                    finder[sin.name] = c
                    if sin.name not in finder:
                        finder[sin.name] = c
                    else:
                        finder[sin.name] = DUPLICATE
                    #backport
                    #finder[sin.name] = c if sin.name not in finder else DUPLICATE
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
                    raise TypeError("Unknown input or state: %s" % str(item))
                if s is DUPLICATE:
                    raise TypeError("Ambiguous name: %s - please check the names "\
                        "of the inputs of your function for duplicates." % str(item))
                if isinstance(s, gof.Container):
                    return s.value
                else:
                    raise NotImplementedError
            def __setitem__(self, item, value):
                try:
                    s = finder[item]
                except KeyError:
                    # Print informative error message.
                    msg = get_info_on_inputs(named_inputs, n_unnamed_inputs)
                    raise TypeError("Unknown input or state: %s. %s" % (str(item), msg))
                if s is DUPLICATE:
                    raise TypeError("Ambiguous name: %s - please check the names "\
                        "of the inputs of your function for duplicates." % str(item))
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

        # Compute self.n_returned_outputs.
        # This is used only when fn.need_update_inputs is False
        # because we're using one of the VM objects and it is
        # putting updates back into the input containers all by itself.
        assert len(self.maker.expanded_inputs) == len(self.input_storage)
        self.n_returned_outputs = len(self.output_storage)
        for input in self.maker.expanded_inputs:
            if input.update is not None:
                self.n_returned_outputs -= 1

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
        profile = self.profile
        t0 = time.time()

        # Reinitialize each container's 'provided' counter
        for c in self.input_storage:
            c.provided = 0

        if len(args)+len(kwargs)>len(self.input_storage):
            raise TypeError("Too many parameter passed to theano function")

        # Set positional arguments
        i = 0
        for arg in args:
            #TODO: provide a Param option for skipping the filter if we
            #      really want speed.
            s = self.input_storage[i]
            # see this emails for a discuation about None as input
            # https://groups.google.com/group/theano-dev/browse_thread/thread/920a5e904e8a8525/4f1b311a28fc27e5
            if arg is None:
                s.storage[0] = arg
            else:
                try:
                    s.storage[0] = s.type.filter(arg, strict=s.strict,
                            allow_downcast=s.allow_downcast)

                except Exception, e:
                    function_name="theano function"
                    if self.name:
                        function_name += 'with name "'+self.name+'" '
                    #end if
                    e.args = tuple(["Bad input argument to " + function_name +
                                    " at index %d(0-based)" % i] + list(e.args))
                    raise
                #end except
            #end if
            s.provided += 1
            i+=1


        # Set keyword arguments
        if kwargs:  # for speed, skip the iteritems for empty kwargs
            for k, arg in kwargs.iteritems():
                self[k] = arg

        if (not hasattr(self, '_check_for_aliased_inputs') or
            self._check_for_aliased_inputs):
            ## Collect aliased inputs among the storage space
            args_share_memory = []
            for i in xrange(len(self.input_storage)):
                i_var = self.maker.inputs[i].variable
                i_val = self.input_storage[i].storage[0]
                if hasattr( i_var.type, 'may_share_memory'):
                    is_aliased = False
                    for j in xrange(len(args_share_memory)):

                        group_j = itertools.izip(
                            [self.maker.inputs[k].variable for k
                             in args_share_memory[j]],
                            [self.input_storage[k].storage[0] for k
                             in args_share_memory[j]])
                        if numpy.any([ (var.type is i_var.type and
                                        var.type.may_share_memory(val,i_val)
                                       ) for (var,val) in group_j]):

                            is_aliased = True
                            args_share_memory[j].append(i)
                            break

                    if not is_aliased:
                        args_share_memory.append([i])

                # Check for groups of more than one argument that share memory
                for group in args_share_memory:
                    if len(group) > 1:
                        # see if any of these arguments are mutable
                        mutable = numpy.any([(self.maker.inputs[idx].mutable or
                                             self.maker.inputs[idx].borrow )
                                             for idx in group ])
                        # copy all but the first
                        for idx in group[1:]:
                            self.input_storage[i].storage[0] = copy.copy(
                                self.input_storage[i].storage[0])




        # Check if inputs are missing, or if inputs were set more than once, or
        # if we tried to provide inputs that are supposed to be implicit.
        for c in self.input_storage:
            if c.required and not c.provided:
                raise TypeError("Missing required input: %s" % getattr(self.inv_finder[c], 'variable', self.inv_finder[c]))
            if c.provided > 1:
                raise TypeError("Multiple values for input: %s" % getattr(self.inv_finder[c], 'variable', self.inv_finder[c]))
            if c.implicit and c.provided > 0:
                raise TypeError('Tried to provide value for implicit input: %s'
                        % getattr(self.inv_finder[c], 'variable',
                            self.inv_finder[c]))

        # Do the actual work
        t0_fn = time.time()
        try:
            outputs = self.fn()
        except Exception:
            if hasattr(self.fn, 'position_of_error'):
                # this is a new vm-provided function
                # the C VM needs this because the exception manipulation
                # done by raise_with_op is not implemented in C.
                gof.vm.raise_with_op(self.fn.nodes[self.fn.position_of_error])
            else:
                # old-style linkers raise their own exceptions
                raise

        dt_fn = time.time() - t0_fn
        self.maker.mode.fn_time += dt_fn
        if profile:
            profile.vm_call_time += dt_fn

        # Retrieve the values that were computed
        if outputs is None:
            outputs = [x.data for x in self.output_storage]
        assert len(outputs) == len(self.output_storage)

        # Remove internal references to required inputs.
        # These cannot be re-used anyway.
        for c in self.input_storage:
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

        if getattr(self.fn, 'need_update_inputs', True):
            # Update the inputs that have an update function
            for input, storage in reversed(zip(self.maker.expanded_inputs, self.input_storage)):
                if input.update is not None:
                    storage.data = outputs.pop()
        else:
            outputs = outputs[:self.n_returned_outputs]

        # Put default values back in the storage
        for i, (required, refeed, value) in enumerate(self.defaults):
            if refeed:
                if isinstance(value, gof.Container):
                    value = value.storage[0]
                self[i] = value
        #
        # NOTE: This logic needs to be replicated in
        #       scan.
        #       grep for 'PROFILE_CODE'
        #

        dt_call = time.time() - t0
        self.maker.mode.call_time += dt_call
        if profile:
            profile.fct_callcount += 1
            profile.fct_call_time += dt_call
            if hasattr(self.fn, 'update_profile'):
                self.fn.update_profile(profile)

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
    input_storage = []

    for (input, indices, inputs), (required, refeed, default) in zip(f.indices, f.defaults):
        if isinstance(input, SymbolicInputKit):
            li = len(indices)
            if not default:
                input_storage.append(ins[:li])
            else:
                input_storage.append(default)
            ins[:li] = []
        else:
            input_storage.append(ins[0])
            del ins[0]

    inputs_data = [x.data for x in f.input_storage]

    # HACK to detect aliased storage.
    # This is here because aliased relationships are not [currently] preserved across the pickle operation
    if not (f.pickle_aliased_memory_strategy == 'ignore'):
        all_data = input_storage + inputs_data # addition here means list append
        for i, d_i in enumerate(all_data):
            for j, d_j in enumerate(all_data):
                if (i < j) and isinstance(d_i, numpy.ndarray) and isinstance(d_j, numpy.ndarray):
                    if numpy.may_share_memory(d_i, d_j):
                        if f.pickle_aliased_memory_strategy == 'warn':
                            _logger.warning(('aliased relationship between'
                                    ' Function arguments %s, %s'
                                    ' will not be preserved by un-pickling'
                                    ' operation') %(str(d_i), str(d_j)))
                        else:
                            raise AliasedMemoryError(d_i, d_j)

    rval = (_constructor_Function, (f.maker, input_storage, inputs_data))
    return rval

def _constructor_Function(maker, input_storage, inputs_data):
    f = maker.create(input_storage, trustme = True)
    assert len(f.input_storage) == len(inputs_data)
    for container, x in zip(f.input_storage, inputs_data):
        assert (container.data is x) or \
            (isinstance(x, numpy.ndarray) and (container.data == x).all()) or \
            (container.data == x)
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
        # DEPRECATED?  Is this just for DualLinker?

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
                        if name:
                            the_name = name
                        else:
                            the_name = ""
                        raise ValueError("Input #%i%s using %s and %s differs."
                                         % (i,
                                            #backport
                                            #" (%s)" % name if name else "",
                                            " (%s)" % the_name,
                                            self.maker.mode,
                                            fn.maker.mode),
                                         c1.value, c2.value)

            # This checks all output storage (this includes state variables that we updated)
            # This is ok because the variables of a call stick around in their storage
            for i, (r1, c2) in enumerate(zip(all_outputs, fn.output_storage)):
                r2 = c2.value
                if not self.check_equal(r1, r2):
                    name = c2.name
                    if name:
                        the_name = name
                    else:
                        the_name = ""
                    raise ValueError("Variable #%i%s using %s and %s differs."
                                     % (i,
                                        #backport
                                        #" (%s)" % name if name else "",
                                        " (%s)" % the_name,
                                        self.maker.mode,
                                        fn.maker.mode),
                                     r1, r2)
        return variables



###
### FunctionMaker
###

def insert_deepcopy(env, wrapped_inputs, wrapped_outputs):
    """
    Insert deepcopy in the env to break aliasing of outputs
    """
    # This loop was inserted to remove aliasing between outputs when they all
    # evaluete to the same value. Originally it was OK for outputs to be aliased,
    # but some of the outputs can be shared variables, and is not good for shared
    # variables to be aliased. It might be possible to optimize this by making sure
    # there is no aliasing only between shared variables.

    # If some outputs are constant, we add deep copy to respect the memory contract

    # We don't insert deep copy when the output.borrow is True for all conserned outputs.

    assert len(wrapped_inputs) == len(env.inputs)
    assert len(wrapped_outputs) == len(env.outputs)
    reason = "insert_deepcopy"
    updated_env_inputs = [env_i for i, env_i in zip(wrapped_inputs, env.inputs) if getattr(i, 'update', False)]

    # We can't use env.inputs as this don't include Constant Value.
    all_graph_inputs = gof.graph.inputs(env.outputs)

    for i in xrange(len(env.outputs)):
        views_of_output_i = set()
        view_tree_set(alias_root(env.outputs[i]), views_of_output_i)
        copied = False
        # do not allow outputs to be aliased
        for j in xrange(i+1, len(env.outputs)):
            # We could don't put deep copy if both outputs have borrow==True
            # and not(wrapped_outputs[i].borrow and wrapped_outputs[j].borrow):
            if env.outputs[j] in views_of_output_i:
                if wrapped_outputs[i].borrow and wrapped_outputs[j].borrow:
                    env.change_input('output',i, view_op(env.outputs[i]),
                                     reason=reason)
                else:
                    env.change_input('output', i, deep_copy_op(env.outputs[i]),
                                     reason=reason)
                copied = True
                break

        if not copied:
            for input_j in all_graph_inputs:
                # do not allow outputs to be aliased to an inputs (j), unless
                # a) that j'th input has been 'destroyed' by e.g. in-place computations
                # b) that j'th input is a shared variable that is also being updated
                if hasattr(env,'get_destroyers_of') and env.get_destroyers_of(input_j):
                    continue
                if input_j in updated_env_inputs:
                    continue
                if input_j in views_of_output_i:
                    # We don't put deep_copy_op if the input and the output have borrow==True
                    if input_j in env.inputs:
                        j = env.inputs.index(input_j)
                        if wrapped_outputs[i].borrow and wrapped_inputs[j].borrow:
                            env.change_input('output',i, view_op(env.outputs[i]),
                                             reason="insert_deepcopy")
                            break
                        else:
                            env.change_input('output', i, deep_copy_op(env.outputs[i]),
                                             reason="insert_deepcopy")
                            break
                    elif wrapped_outputs[i].borrow:
                        env.change_input('output',i, view_op(env.outputs[i]),
                                         reason="insert_deepcopy")
                        break
                    else:
                        env.change_input('output', i, deep_copy_op(env.outputs[i]),
                                         reason="insert_deepcopy")
                        break

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
            mode = None, accept_inplace = False, function_builder = Function,
            profile=None):
        """
        :type inputs: a list of SymbolicInput instances

        :type outputs: a list of SymbolicOutput instances
                    outputs may also be a single Variable (not a list), in which
                    case the functions produced by FunctionMaker will return
                    their output value directly

        :param mode: a Mode instance telling FunctionMaker how to optimize and link.  None
        means to use the `config.mode`.

        :param accept_inplace: True iff it is acceptable to have inplace operations
                    in the graph from the inputs to the outputs
        """
        mode = mode_module.get_mode(mode)

        # figure out which profile object to use (if any)
        # to help with forward-porting ProfileMode,
        # we allow ProfileMode to provide a ProfileStats object
        # using this somewhat awkward mechanism.
        mode_profile = getattr(mode, 'profile', None)
        if (profile is not None) and (mode_profile is not None):
            raise TypeError(
                    'profile passed via both "mode" and "profile" arguments')
        self.profile = profile = profile or mode_profile

        # Handle the case where inputs and/or outputs is a single Variable (not in a list)
        self.orig_outputs = outputs
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
        #import pudb; pudb.set_trace()
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.variable for o in outputs] + [i.update
            for i in inputs if getattr(i, 'update', False)])

        #TODO: REMOVE THIS CRUFT - it's complicated for SymbolicInputKits
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices], [])
        assert expanded_inputs == inputs  #JB - I added this to make sure we could delete above

        # make the env (copies the graph, creates NEW INPUT AND OUTPUT VARIABLES)
        env, additional_outputs = std_env(expanded_inputs, outputs, accept_inplace)
        self.env = env

        # Fetch the optimizer and linker
        optimizer, linker = mode.optimizer, copy.copy(mode.linker)

        # optimize the env
        compute_test_value_orig = theano.config.compute_test_value
        add_stack_trace_on_call = gof.Op.add_stack_trace_on_call
        try:
            theano.config.compute_test_value = "off"
            gof.Op.add_stack_trace_on_call = False
            start_optimizer = time.time()
            optimizer(env)
            end_optimizer = time.time()

            opt_time = end_optimizer - start_optimizer
            mode.optimizer_time += opt_time
            if profile:
                profile.optimizer_time += opt_time
            _logger.debug('Optimizing took %f seconds', opt_time)

            #Add deep copy to respect the memory interface
            insert_deepcopy(env, inputs, outputs+additional_outputs)
        finally:
            theano.config.compute_test_value = compute_test_value_orig
            gof.Op.add_stack_trace_on_call = add_stack_trace_on_call

        # initialize the linker
        if not hasattr(linker, 'accept'):
            raise ValueError("'linker' parameter of FunctionFactory should be a Linker with an accept method " \
                             "or one of %s" % mode_module.predefined_linkers.keys())

        #the 'no_borrow' outputs are the ones for which that we can't return the internal storage pointer.
        assert len(env.outputs) == len(outputs+additional_outputs)
        no_borrow = [output for output, spec in zip(env.outputs, outputs+additional_outputs) if not spec.borrow]
        if no_borrow:
            self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, no_borrow))
        else:
            self.linker = linker.accept(env)

        if hasattr(linker, 'accept_var_updates'):
            # hacky thing so VMLinker knows about updates
            self.linker.accept_var_updates(
                    env_updated_vars(env, expanded_inputs))

        self.indices = indices
        self.inputs = inputs
        self.expanded_inputs = expanded_inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.mode = mode
        self.accept_inplace = accept_inplace
        self.function_builder = function_builder

        self.required = [(i.value == None) for i in self.inputs]
        self.refeed = [
                (i.value != None and not isinstance(i.value, gof.Container) and i.update == None)
                    for i in self.inputs]

    def create(self, input_storage=None, trustme=False):
        """
        Create a function.

        input_storage -> a list matching the inputs list and providing default values
                    if the default for an input is None, then that input is a
                    required input. For an input with an update, the default
                    acts as initialization.
        trustme -> disables some exceptions, used internally
        """
        if input_storage is None:
            input_storage = [None]*len(self.inputs)
        input_storage_lists = [] # list of independent one-element lists, will be passed to the linker
        defaults = []

        # The following loop is to fill in the input_storage_lists and defaults lists.
        assert len(self.indices) == len(input_storage)
        for i, ((input, indices, subinputs), input_storage_i) in enumerate(zip(self.indices, input_storage)):
            # Replace any default value given as a variable by its container.
            # Note that this makes sense only in the context of shared variables,
            # but for now we avoid dealing directly with them to avoid dependency
            # on the shared variables work-in-progress repository.
            if isinstance(input_storage_i, gof.Variable):
                input_storage_i = input_storage_i.container

            if isinstance(input_storage_i, gof.Container):
                # If the default is a gof.Container, this means we want to
                # share the same storage. This is done by appending
                # input_storage_i.storage to input_storage_lists.
                if indices is not None:
                    raise TypeError("Cannot take a Container instance as default for a SymbolicInputKit.")
                input_storage_lists.append(input_storage_i.storage)
                defaults.append((self.required[i],
                    self.refeed[i],
                    input_storage_i.storage[0]))
            else:
                # Normal case: one new, independent storage unit
                input_storage_lists.append([input_storage_i])
                defaults.append((self.required[i], self.refeed[i], input_storage_i))


        # Get a function instance
        start_linker = time.time()
        _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage_lists)
        end_linker = time.time()

        linker_time = end_linker - start_linker
        _logger.debug('Linker took %f seconds', linker_time)
        self.mode.linker_time += linker_time
        if self.profile:
            self.profile.linker_time += linker_time
            _fn.time_thunks = self.profile.flag_time_thunks

        fn = self.function_builder(_fn, _i, _o, self.indices, self.outputs,
                defaults, self.unpack_single, self.return_none, self)
        fn.profile = self.profile
        return fn

def _pickle_FunctionMaker(self):
    kwargs = dict(
                inputs = self.inputs,
                outputs = self.orig_outputs,
                mode = self.mode,
                accept_inplace = self.accept_inplace,
                function_builder = self.function_builder,
                profile = self.profile,
                )
    return (_constructor_FunctionMaker, (kwargs,))

def _constructor_FunctionMaker(kwargs):
    return FunctionMaker(**kwargs)

copy_reg.pickle(FunctionMaker, _pickle_FunctionMaker)



try:
    # Pickle of slice is implemented on python 2.6.  To enabled be
    # compatible with python 2.4, we implement pickling of slice
    # ourself.
    cPickle.dumps(slice(0, 10, 100))
except TypeError:
    # This slice pickle implementation seam backward and forward compatible.
    def _pickle_slice(s):
        return (slice, (s.start, s.stop, s.step))
    copy_reg.pickle(slice, _pickle_slice)


__checkers = []

def check_equal(x, y):
    for checker in __checkers:
        try:
            return checker(x, y)
        except Exception:
            continue
    return x == y
    #raise Exception('No checker for equality between %s and %s' % (x, y))

def register_checker(checker):
    __checkers.insert(0, checker)

def orig_function(inputs, outputs, mode=None, accept_inplace = False, name=None, profile=None):
    """
    Return a Function that will calculate the outputs from the inputs.

    :param inputs: list of `SymbolicInput` or `In` instances

    :param outputs: a SymbolicOutput or a list of `SymbolicOutput` or `Out` instances.   The return
        value of the returned function will match the format of this argument (either the value
        itself or a list of one or more return values)

    :param mode: a descriptive string or a Mode instance. (Default of None means to use
    `config.mode` (See below for descriptive string list).

    :param name: an optional name for this fct. If used, the profile mode will print the time spent in this fct.

    Currently, the library provides the following mode strings:

     - FAST_RUN (default) (optimize without too much time)

     - FAST_COMPILE (minimal optimization)

     - PROFILE_MODE : allow to print a profile mode with mode.print_summary

     - DEBUG_MODE : verify many internal conditions that are normally assumed (SLOW)

    :param accept_inplace:  True iff the graph can contain inplace operations prior to the
    optimization phase (default is False)

    :param profile: None or ProfileStats instance

    """

    #Every element of the input list will be upgraded to an `In` instance if necessary,
    #using the rules implemented by the `convert_function_input` function.

    #Similarly, every element of the output list will be upgraded to an
    #`Out` instance if necessary:

    t1 = time.time()
    mode = mode_module.get_mode(mode)

    inputs = map(convert_function_input, inputs)
    if outputs is not None:
        if isinstance(outputs, (list, tuple)):
            outputs = map(FunctionMaker.wrap_out, outputs)
        else:
            outputs = FunctionMaker.wrap_out(outputs)

    defaults = [getattr(input, 'value', None) for input in inputs]

    if isinstance(mode, (list, tuple)): # "mode comparison" semantics
        _logger.warning('Passing multiple modes is deprecated (20091019)')
        if not mode:
            raise ValueError("Please provide at least one mode.")
        elif len(mode) == 1:
            fn = FunctionMaker(
                    inputs,
                    outputs,
                    mode[0],
                    accept_inplace = accept_inplace,
                    profile=profile).create(
                            defaults)
        else:
            if profile:
                raise NotImplementedError('profiling not implemented in this kind of mode')
            #return a different kind of function
            def dup_defaults():
                # TODO This may need to be changed to use containers as defaults.
                retval = []
                for default in defaults:
                    if isinstance(default, gof.Container):
                        retval +=[copy.copy(default.value)]
                    else:
                        retval +=[copy.copy(default)]
                return retval
                #backport
                #return [copy.copy(default.value) if isinstance(default, gof.Container) else
                #        copy.copy(default)
                #        for default in defaults]
            makers = [FunctionMaker(inputs, outputs, m, accept_inplace = accept_inplace) for m in mode[1:]]
            fns = [maker.create(dup_defaults(), trustme = True) for maker in makers]
            builder = partial(SanityCheckFunction, fns, check_equal)
            maker1 = FunctionMaker(inputs, outputs, mode[0], accept_inplace = accept_inplace, function_builder = builder)
            fn = maker1.create(defaults)
    else:
        Maker = getattr(mode, 'function_maker', FunctionMaker)
        fn = Maker(inputs,
                outputs,
                mode,
                accept_inplace = accept_inplace,
                profile=profile).create(
                        defaults)

    t2 = time.time()
    if profile:
        profile.compile_time+=t2-t1

    fn.name = name
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
        if isinstance(input[0], basestring):
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


def get_info_on_inputs(named_inputs, n_unnamed_inputs):
    """Return a human-readable description of named and un-named inputs."""
    n_named_inputs = len(named_inputs)
    def get_plural(n):
        if n > 1:
            return 's'
        else:
            return ''
    if n_named_inputs == 0:
        if n_unnamed_inputs == 0:
            msg = 'The function is supposed to have no input.'
        else:
            if n_unnamed_inputs == 1:
                msg = ("The function has a single input variable which has no "
                        "name, and thus cannot be assigned through a keyword"
                        " argument (use 'name=...' in a Variable's "
                        "constructor to give it a name).")
            else:
                # Use plural.
                msg = ("The function has %s inputs, but none of them is named,"
                        " and thus they cannot be assigned through keyword "
                        "arguments (use 'name=...' in a Variable's "
                        "constructor to give it a name)." % n_unnamed_inputs)
    else:
        if n_unnamed_inputs == 0:
            msg = ("The function has %s named input%s (%s)." % (
                n_named_inputs, get_plural(n_named_inputs),
                ', '.join(named_inputs)))
        else:
            msg = ("The function has %s named input%s (%s), and %s unnamed "
                    "input%s which thus cannot be accessed through keyword "
                    "argument%s (use 'name=...' in a variable's constructor "
                    "to give it a name)." % (
                    n_named_inputs, get_plural(n_named_inputs),
                    ', '.join(named_inputs), n_unnamed_inputs,
                    get_plural(n_unnamed_inputs),
                    get_plural(n_unnamed_inputs)))
    return msg
