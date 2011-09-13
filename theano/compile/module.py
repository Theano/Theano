"""Classes implementing Theano's Module system.

For design notes, see doc/advanced/module.txt

"""

__docformat__ = "restructuredtext en"

from theano import gof
from theano.printing import pprint
import io, sys

from theano.gof.python25 import any, all, defaultdict, partial

from itertools import chain

import function_module as F
import mode as get_mode


def name_join(*args):
    """
    Creates a string representation for the given names:
    join('a', 'b', 'c') => 'a.b.c'
    """
    return ".".join(arg for arg in args if arg)

def name_split(sym, n=-1):
    """
    Gets the names from their joined representation
    split('a.b.c') => ['a', 'b', 'c']
    Returns the n first names, if n==-1 returns all of them.
    split('a.b.c',1) => ['a', 'b.c']
    """
    return sym.split('.', n)

class AllocationError(Exception):
    """
    Exception raised when a Variable has no associated storage.
    """
    pass

class Component(object):
    """
    Base class for the various kinds of components which are not
    structural but may be meaningfully used in structures (Member,
    Method, etc.)
    """

    def __init__(self):
        self.__dict__['_name'] = ''
        self.__dict__['parent'] = None

    def allocate(self, memo):
        """
        Populates the memo dictionary with gof.Variable -> io.In
        pairings. The value field of the In instance should contain a
        gof.Container instance. The memo dictionary is meant to tell
        the build method of Components where the values associated to
        certain variables are stored and how they should behave if they
        are implicit inputs to a Method (needed to compute its
        output(s) but not in the inputs or updates lists).
        """
        raise NotImplementedError

    def build(self, mode, memo):
        """
        Makes an instance of this Component using the mode provided
        and taking the containers in the memo dictionary.

        A Component which builds nothing, such as External, may return
        None.

        The return value of this function will show up in the Module graph produced by make().
        """
        raise NotImplementedError()

    def make_no_init(self, mode=None):
        """
        Allocates the necessary containers using allocate() and uses
        build() with the provided mode to make an instance which will
        be returned.  The initialize() method of the instance will not
        be called.
        """
        if mode is None:
            mode = get_mode.get_default_mode()
        memo = {}
        self.allocate(memo)
        rval = self.build(mode, memo)
        return rval

    def make(self, *args, **kwargs):
        """
        Allocates the necessary containers using allocate() and uses
        build() to make an instance which will be returned. The
        initialize() method of the instance will be called with the
        arguments and the keyword arguments. If 'mode' is in the
        keyword arguments it will be passed to build().
        """
        mode = kwargs.pop('mode', get_mode.get_default_mode())
        rval = self.make_no_init(mode)
        if hasattr(rval, 'initialize'):
            rval.initialize(*args, **kwargs)
        return rval

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__

    def pretty(self, **kwargs):
        """
        Returns a pretty representation of this Component, suitable
        for reading.
        """
        raise NotImplementedError

    def __get_name__(self):
        """
        Getter for self.name
        """
        return self._name

    def __set_name__(self, name):
        """
        Setter for self.name
        """
        self._name = name

    name = property(lambda self: self.__get_name__(),
                    lambda self, value: self.__set_name__(value),
                    "Contains the name of this Component")



class _RComponent(Component):
    """
    Base class for a Component wrapping a Variable. For internal use.
    """

    def __init__(self, r):
        super(_RComponent, self).__init__()
        self.r = r
        # If self.owns_name is True, then the name of the variable
        # may be adjusted when the name of the Component is. Else,
        # the variable will always keep its original name. The component
        # will only be allowed to own a variable's name if it has no
        # original name to begin with. This allows the user to opt out
        # of the automatic naming scheme if he or she wants to. It is
        # also usually the case that a Variable used in more than one
        # Component should only retain the first name it gets.
        self.owns_name = r.name is None

    def __set_name__(self, name):
        super(_RComponent, self).__set_name__(name)
        if self.owns_name:
            self.r.name = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.r)

    def pretty(self, **kwargs):
        rval = '%s :: %s' % (self.__class__.__name__, self.r.type)
        return rval


class External(_RComponent):
    """
    External represents a Variable which comes from somewhere else
    (another module) or is a temporary calculation.
    """

    def allocate(self, memo):
        # nothing to allocate
        return None

    def build(self, mode, memo):
        """
        Builds nothing.
        """
        return None

    def pretty(self, **kwargs):
        rval = super(External, self).pretty()
        if self.r.owner:
            rval += '\n= %s' % (pprint(self.r, dict(target = self.r)))
        return rval

class Member(_RComponent):
    """
    Member represents a Variable which is a state of a Composite. That
    Variable will be accessible from a built Composite and it is
    possible to do updates on Members.

    Member builds a gof.Container.
    """

    def allocate(self, memo):
        """
        If the memo does not have a Container associated to this
        Member's Variable, instantiates one and sets it in the memo.
        """
        r = self.r
        if memo and r in memo:
            return memo[r]
        assert isinstance(r, gof.Variable)
        rval = gof.Container(r, storage = [getattr(r, 'data', None)],
                readonly=isinstance(r, gof.Constant))
        memo[r] = io.In(variable=r,
                        value=rval,
                        mutable=False)
        return memo[r]

    def build(self, mode, memo):
        """
        Returns the Container associated to this Member's Variable.
        """
        return memo[self.r].value

class Method(Component):
    """
    Method is a declaration of a function. It contains inputs,
    outputs and updates. If the Method is part of a Composite
    which holds references to Members, the Method may use them
    without declaring them in the inputs, outputs or updates list.

    inputs, outputs or updates may be strings. In that case, they
    will be resolved in the Composite which is the parent of this
    Method.

    Method builds a Function (same structure as a call to
    theano.function)
    """

    inputs = []
    """function inputs (see `compile.function`)

    If Module members are named explicitly in this list, then they will not use shared storage.
    Storage must be provided either via an `io.In` value argument, or at the point of the
    function call.
    """

    outputs=None
    """function outputs (see `compile.function`)"""

    updates = {}
    """update expressions for module members

    If this method should update the shared storage value for a Module member, then the
    update expression must be given in this dictionary.


    Keys in this dictionary must be members of the module graph--variables for which this Method
    will use the shared storage.

    The value associated with each key should be a Variable (or a string that can be resolved to
    a Variable) representing the computation of a new value for this shared storage after
    each function call.

    """

    mode=None
    """This will override the Module compilation mode for this Method"""

    def __init__(self, inputs, outputs, updates = {}, mode=None):
        """Initialize attributes
        :param inputs: value for `Method.inputs`

        :param outputs: value for `Method.outputs`

        :param updates: value for `Method.updates`

        :param mode: value for `Method.mode`

        :type inputs: list of (str or `Variable` or `io.In`)

        :type outputs: None or str or `Variable` or `io.Out` or list of (str or `Variable` or
        `io.Out`)

        :type updates: dict of `Variable` or str -> `Variable` or str

        :type mode: None or any mode accepted by `compile.function`

        """
        super(Method, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.updates = dict(updates)
        self.mode = mode

    def resolve_all(self):
        """Convert all inputs, outputs, and updates specified as strings to Variables.

        This works by searching the attribute list of the Module to which this Method is bound.
        """
        def resolve_variable(x, passthrough=(gof.Variable,)):
            if isinstance(x, passthrough):
                return x
            elif isinstance(x, _RComponent):
                return x.r
            else:
                raise Exception('The following thing is not of the following types', x,
                        passthrough + (_RComponent,))
                # return self.resolve(x).r

        def resolve_inputs():
            if isinstance(self.inputs, (io.In, gof.Variable, basestring)):
                inputs = [self.inputs]
            else:
                inputs = list(self.inputs)
            self.inputs = [resolve_variable(input,
                passthrough=(gof.Variable, io.In)) for input in inputs]

        def resolve_outputs():
            if isinstance(self.outputs, (io.Out, gof.Variable, basestring, type(None))):
                output = self.outputs
                self.outputs = resolve_variable(output,
                    passthrough=(gof.Variable, io.Out, type(None)))
            else:
                outputs = list(self.outputs)
                self.outputs = [resolve_variable(output,
                    passthrough=(gof.Variable, io.Out)) for output in outputs]

        def resolve_updates():
            updates = self.updates
            self.updates = {}
            for k, v in updates.iteritems():
                k, v = resolve_variable(k), resolve_variable(v)
                self.updates[k] = v

        resolve_inputs()
        resolve_outputs()
        resolve_updates()

    def allocate(self, memo):
        """
        Method allocates nothing.
        """
        return None

    def build(self, mode, memo, allocate_all = False):
        """Compile a function for this Method.

        :param allocate_all: if True, storage will be
        allocated for all needed Variables even if there is no
        associated storage for them in the memo. If allocate_all is
        False, storage will only be allocated for Variables that are
        reachable from the inputs list.

        :returns: a function that implements this method
        :rtype: `Function` instance

        """
        if self in memo:
            return memo[self]

        self.resolve_all() # resolve all so we don't have to mess with strings
        def get_storage(r, require=False):
            # If require is True, we can only get storage from the memo.
            try:
                return memo[r]
            except KeyError:
                if require:
                    raise AllocationError('There is no storage associated to %s used by %s = %s.'
                                          ' Verify that it is indeed a Member of the'
                                          ' enclosing module or of one of its submodules.' % (r, self.name, self))
                else:
                    return io.In(variable=r,
                            value=gof.Container(r,
                                storage=[getattr(r, 'data', None)],
                                readonly=(isinstance(r, gof.Constant))),
                            mutable=False)
        inputs = self.inputs

        # Deal with explicit inputs
        inputs = []
        for input in self.inputs:
            if type(input) is io.In:
                inputs.append(input)
            elif isinstance(input, gof.Variable):
                input_in = io.In(
                        variable=input,
                        mutable=False)
                inputs.append(input_in)
            else:
                raise TypeError(input, type(input))

        # Deal with updates to shared storage
        for k, v in self.updates.iteritems():
            assert isinstance(k, gof.Variable)
            if isinstance(k, gof.Constant):
                raise TypeError('Module Constants cannot be updated', k)
            assert isinstance(v, gof.Variable)

            #identify an input for variable k
            input_k = None
            for input in inputs:
                if input.variable == k:
                    input_k = input

            #print 'METHOD UPDATE', k, v, input_k
            if input_k is None:
                # this is an implicit input,
                # use shared storage
                input_k = io.In(
                        variable=k,
                        update=v,
                        value=get_storage(k, not allocate_all).value,
                        mutable=True,
                        implicit = True)
                inputs.append(input_k)
            else:
                raise ValueError(('Variable listed in both inputs and updates.'
                    ' Use inputs to use your own storage, use updates to '
                    'work on module-shared storage'), k)

        # Deal with module inputs that are not updated

        outputs = self.outputs
        _inputs = [x.variable for x in inputs]
        # Grab the variables that are not accessible from either the inputs or the updates.
        if outputs is None:
            outputs_list = []
        else:
            if isinstance(outputs, (list, tuple)):
                outputs_list = list(outputs)
            else:
                outputs_list = [outputs]

        #backport
        #outputs_list = [] if outputs is None else (list(outputs) if isinstance(outputs, (list, tuple)) else [outputs])

        outputs_variable_list = []
        for o in outputs_list:
            if isinstance(o, io.Out):
                outputs_variable_list += [o.variable]
            else:
                outputs_variable_list += [o]

        #backport
        #outputs_variable_list = [o.variable if isinstance(o, io.Out) else o for o in outputs_list]
        for input in gof.graph.inputs(outputs_variable_list
                                      + [x.update for x in inputs if getattr(x, 'update', False)],
                                      blockers = _inputs):
            if input not in _inputs:
                # Add this input to the inputs; we require that storage already exists for them,
                # but otherwise they are immutable.
                if isinstance(input, gof.Value): # and not isinstance(input, gof.Constant):
                    #input might be Value or Constant
                    storage = get_storage(input)

                    assert type(storage) is io.In
                    container = storage.value
                    #the user is allowed to change this value between function calls if it isn't a constant
                    assert container.readonly == (isinstance(input, gof.Constant))
                    #the function is not allowed to change this value
                    assert storage.mutable == False
                else:
                    storage = get_storage(input, not allocate_all)

                # Declare as an implicit input.
                # TODO Note from OD: is this dangerous? (in case this storage
                # is shared, and would sometimes need to be implicit, sometimes
                # not).
                storage.implicit = True

                assert type(storage) is io.In
                inputs.append(storage)

        if self.mode is None:
            effective_mode = mode
        else:
            effective_mode = self.mode

        #backport
        #effective_mode = mode if self.mode is None else self.mode
        rval = F.orig_function(inputs, outputs, effective_mode)
        memo[self] = rval
        return rval

    def pretty(self, **kwargs):
        self.resolve_all()
        if self.inputs:
            rval = 'inputs: %s\n' % ", ".join(map(str, self.inputs))
        else:
            rval = ''
        if isinstance(self.outputs, (list, tuple)):
            inputs, outputs, updates = self.inputs, self.outputs
        else:
            inputs, outputs, updates =  [self.outputs], self.updates

        #backport
        #inputs, outputs, updates = self.inputs, self.outputs if isinstance(self.outputs, (list, tuple)) else [self.outputs], self.updates

        # If mode is in kwargs, prints the optimized version of the method
        mode = kwargs.pop('mode', None)
        if mode:
            f = self.build(mode, {}, True)
            einputs, eoutputs = f.maker.env.inputs, f.maker.env.outputs
            updates = dict(((k, v) for k, v in zip(einputs[len(inputs):], eoutputs[len(outputs):])))
            inputs, outputs = einputs[:len(inputs)], eoutputs[:len(outputs)]
        rval += pprint(inputs, outputs, updates, False)
        return rval

    def __str__(self):
        if self.updates:
            sep = "; "
        else:
            sep = ""
        return "Method(%s -> %s%s%s)" % \
            (self.inputs,
             self.outputs,
             sep,
             #backport
             #"; " if self.updates else "",
             ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))

    def __call__(self, *args, **kwargs):
        raise TypeError("'Method' object is not callable"
                "  (Hint: compile your module first.  See Component.make())")

class CompositeInstance(object):
    """
    Generic type which various Composite subclasses are intended to
    build.
    """

    def __init__(self, component, __items__):
        # The Component that built this CompositeInstance
        self.__dict__['component'] = component
        # Some data structure indexable using []
        self.__dict__['__items__'] = __items__

    def __getitem__(self, item):
        x = self.__items__[item]
        # For practical reasons, if the item is a Container, we
        # return its contents.
        if isinstance(x, gof.Container):
            return x.value
        return x

    def __setitem__(self, item, value):
        x = self.__items__[item]
        if isinstance(x, gof.Container):
            # If the item is a Container, we set its value
            x.value = value
        elif hasattr(x, 'initialize'):
            # If the item has an initialize() method, we use
            # it with the value as argument
            x.initialize(value)
        else:
            ##self.__items__[item] = value
            raise KeyError('Cannot set item %s' % item)

class Composite(Component):
    """
    Composite represents a structure that contains Components.
    """

    def resolve(self, name):
        raise NotImplementedError

    def components(self):
        """
        Returns all components.
        """
        raise NotImplementedError

    def components_map(self):
        """
        Returns (key, value) pairs corresponding to each component.
        """
        raise NotImplementedError

    def flat_components(self, include_self = False):
        """
        Generator that yields each component in a flattened hierarchy
        of composites and components. If include_self is True, the
        list will include the Composite instances, else it will only
        yield the list of leaves.
        """
        if include_self:
            yield self
        for component in self.components():
            if isinstance(component, Composite):
                for x in component.flat_components(include_self):
                    yield x
            else:
                yield component

    def flat_components_map(self, include_self = False, path = []):
        """
        Generator that yields (path, component) pairs in a flattened
        hierarchy of composites and components, where path is a
        sequence of keys such that
          component is self[path[0]][path[1]]...

        If include_self is True, the list will include the Composite
        instances, else it will only yield the list of leaves.
        """
        if include_self:
            yield path, self
        for name, component in self.components_map():
            path2 = path + [name]
            if isinstance(component, Composite):
                for fullpath, x in component.flat_components_map(include_self, path2):
                    yield fullpath, x
            else:
                yield path2, component

    def allocate(self, memo):
        """
        Does allocation for each component in the composite.
        """
        for member in self.components():
            member.allocate(memo)

    def get(self, item):
        """
        Get the Component associated to the key.
        """
        raise NotImplementedError

    def set(self, item, value):
        """
        Set the Component associated to the key.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        # Uses get() internally
        print 'COMPOSITE GETITEM', item
        x = self.get(item)
        if isinstance(x, (External, Member)):
            return x.r
        return x

    def __setitem__(self, item, value):
        # Uses set() internally
        self.set(item, value)

    def __iter__(self):
        retval = []
        for c in self.components():
            if isinstance(c, (External, Member)):
                retval += [c.r]
            else:
                retval += [c]

        return retval
        #backport
        #return (c.r if isinstance(c, (External, Member)) else c for c in self.components())



class ComponentListInstance(CompositeInstance):

    def __str__(self):
        return '[%s]' % ', '.join(map(str, self.__items__))

    def __len__(self):
        return len(self.__items__)

    def initialize(self, init):
        for i, initv in enumerate(init):
            self[i] = initv

class ComponentList(Composite):
    """
    ComponentList represents a sequence of Component. It builds a
    ComponentListInstance.
    """

    def __init__(self, *_components):
        super(ComponentList, self).__init__()
        if len(_components) == 1 and isinstance(_components[0], (list, tuple)):
            _components = _components[0]
        self._components = []
        for c in _components:
            if not isinstance(c, Component):
                raise TypeError(c, type(c))
            self.append(c)

    def components(self):
        return self._components

    def components_map(self):
        return enumerate(self._components)

    def build(self, mode, memo):
        if self in memo:
            return memo[self]
        builds = [c.build(mode, memo) for c in self._components]
        rval = ComponentListInstance(self, builds)
        memo[self] = rval
        return rval

    def get(self, item):
        return self._components[item]

    def set(self, item, value):
        if isinstance(value, gof.Variable):
            value = Member(value)
        elif not isinstance(value, Component):
            raise TypeError('ComponentList may only contain Components.', value, type(value))
        #value = value.bind(self, str(item))
        value.name = name_join(self.name, str(item))
        self._components[item] = value

    def append(self, c):
        if isinstance(c, gof.Variable):
            c = Member(c)
        elif not isinstance(c, Component):
            raise TypeError('ComponentList may only contain Components.', c, type(c))
        self._components.append(c)

    def extend(self, other):
        for o in other:
            self.append(o)

    def __add__(self, other):
        if isinstance(other, (list, tuple)):
            return ComponentList(self._components + map(wrap,other))
        elif isinstance(other, ComponentList):
            return ComponentList(self._components + other._components)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (list, tuple)):
            return ComponentList(map(wrap,other) + self._components)
        elif isinstance(other, ComponentList):
            return ComponentList(other._components + self._components)
        else:
            return NotImplemented

    def __str__(self):
        return str(self._components)

    def __len__(self):
        return len(self._components)

    def pretty(self, **kwargs):
        cr = '\n    ' #if header else '\n'
        strings = []
        #if header:
        #    rval += "ComponentList:"
        for i, c in self.components_map():
            strings.append('%i:%s%s' % (i, cr, c.pretty(**kwargs).replace('\n', cr)))
            #rval += cr + '%i -> %s' % (i, c.pretty(header = True, **kwargs).replace('\n', cr))
        return '\n'.join(strings)

    def __set_name__(self, name):
        super(ComponentList, self).__set_name__(name)
        for i, member in enumerate(self._components):
            member.name = '%s.%i' % (name, i)


def default_initialize(self, init = {}, **kwinit):
    for k, initv in dict(init, **kwinit).iteritems():
        self[k] = initv

class ComponentDictInstanceNoInit(CompositeInstance):
    """Component Instance that allows new items to be added"""
    def __setitem__(self, item, value):
        if item not in self.__items__:
            # Set it if it's not there
            # TODO: is this needed here? move to ModuleInstance?
            self.__items__[item] = value
        else:
            super(ComponentDictInstanceNoInit, self).__setitem__(item, value)

    def __str__(self):
        strings = []
        for k, v in sorted(self.__items__.iteritems()):
            if isinstance(v, gof.Container):
                v = v.value
            if not k.startswith('_') and not callable(v) and not k in getattr(self, '__hide__', []):
                pre = '%s: ' % k
                strings.append('%s%s' % (pre, str(v).replace('\n', '\n' + ' '*len(pre))))
        return '{%s}' % '\n'.join(strings).replace('\n', '\n ')


class ComponentDictInstance(ComponentDictInstanceNoInit):
    """
    ComponentDictInstance is meant to be instantiated by ComponentDict.
    """

    def initialize(self, init={}, **kwinit):
        for k, initv in dict(init, **kwinit).iteritems():
            self[k] = initv



class ComponentDict(Composite):
    InstanceType = ComponentDictInstance # Type used by build() to make the instance

    def __init__(self, components = {}, **kwcomponents):
        super(ComponentDict, self).__init__()
        components = dict(components, **kwcomponents)
        for val in components.itervalues():
            if not isinstance(val, Component):
                raise TypeError(val, type(val))

        self.__dict__['_components'] = components

    def components(self):
        return self._components.itervalues()

    def components_map(self):
        return self._components.iteritems()

    def build(self, mode, memo):
        if self in memo:
            return self[memo]
        inst = self.InstanceType(self, {})
        for name, c in self._components.iteritems():
            x = c.build(mode, memo)
            if x is not None:
                inst[name] = x
        memo[self] = inst
        return inst

    def get(self, item):
        return self._components[item]

    def set(self, item, value):
        if not isinstance(value, Component):
            msg = """
            ComponentDict may only contain Components.
            (Hint: maybe value here needs to be wrapped, see theano.compile.module.register_wrapper.)"""
            raise TypeError(msg, value, type(value))
        #value = value.bind(self, item)
        value.name = name_join(self.name, str(item))
        self._components[item] = value

    def pretty(self, **kwargs):
        cr = '\n    ' #if header else '\n'
        strings = []
#         if header:
#             rval += "ComponentDict:"
        for name, component in self.components_map():
            if name.startswith('_'):
                continue
            strings.append('%s:%s%s' % (name, cr, component.pretty(**kwargs).replace('\n', cr)))
        strings.sort()
        return '\n'.join(strings)

    def __str__(self):
        return self.__class__.__name__+"(%s)" % ', '.join(x for x in sorted(map(str, self._components)) if x[0] != '_')

    def __set_name__(self, name):
        super(ComponentDict, self).__set_name__(name)
        for mname, member in self._components.iteritems():
            member.name = '%s.%s' % (name, mname)





__autowrappers = []

def register_wrapper(condition, wrapper):
    """
    :type condition: function x -> bool

    :param condition: this function should return True iff `wrapper` can sensibly turn x into a
    Component.

    :type wrapper: function x -> Component

    :param wrapper: this function should convert `x` into an instance of a Component subclass.
    """
    __autowrappers.append((condition, wrapper))

def wrapper(x):
    """Returns a wrapper function appropriate for `x`
    Returns None if not appropriate wrapper is found
    """
    for condition, wrap_fn in __autowrappers:
        if condition(x):
            return wrap_fn
    return None

def wrap(x):
    """
    Wraps `x` in a `Component`. Wrappers can be registered using
    `register_wrapper` to allow wrapping more types.

    It is necessary for Module attributes to be wrappable.
    A Module with an attribute that is not wrappable as a Component, will cause
    `Component.make` to fail.

    """
    w = wrapper(x)
    if w is not None:
        return w(x)
    else:
        return x

def dict_wrap(d):
    d_copy = {}
    for k,v in d.iteritems():
        d_copy[k]=wrap(v)
    return d_copy

# Component -> itself
register_wrapper(lambda x: isinstance(x, Component),
                 lambda x: x)

# Variable -> Member
register_wrapper(lambda x: isinstance(x, gof.Variable) and not x.owner,
                 lambda x: Member(x))

# Variable -> External
register_wrapper(lambda x: isinstance(x, gof.Variable) and x.owner,
                 lambda x: External(x))

# [[Variable1], {Variable2}, Variable3...] -> ComponentList(Member(Variable1), Member(Variable2), ...)
register_wrapper(lambda x: isinstance(x, (list, tuple)) \
                     and all(wrapper(r) is not None for r in x),
                 lambda x: ComponentList(*map(wrap, x)))

#{ "name1":{Component,Variable,list,tuple,dict},...} -> ComponentDict({Component,Variable,list,tuple,dict},...)
register_wrapper(lambda x: isinstance(x, dict) \
                     and all(wrapper(r) is not None for r in x.itervalues()),
                 lambda x: ComponentDict(dict_wrap(x)))

class Curry:
    def __init__(self, obj, name, arg):
        self.obj = obj
        self.name = name
        self.meth = getattr(self.obj, self.name)
        self.arg = arg
    def __call__(self, *args, **kwargs):
        return self.meth(self.arg, *args, **kwargs)
    def __getstate__(self):
        return [self.obj, self.name, self.arg]
    def __setstate__(self, state):
        self.obj, self.name, self.arg = state
        self.meth = getattr(self.obj, self.name)


class ModuleInstance(ComponentDictInstanceNoInit):
    """
    WRITEME

    :note: ModuleInstance is meant to be instantiated by Module. This differs
    from ComponentDictInstance on a key point, which is that getattr
    does a similar thing to getitem.

    :note: ModuleInstance is compatible for use as ComponentDict.InstanceType.
    """

    def __getattr__(self, attr):
        if attr == '__items__' and '__items__' not in self.__dict__:
            self.__dict__['__items__'] = {}
        try:
            return self[attr]
        except KeyError:
            raise AttributeError('%s has no %s attribute.' % (self.__class__, attr))

    def __setattr__(self, attr, value):
        try:
            self[attr] = value
        except KeyError:
            self.__dict__[attr] = value

class Module(ComponentDict):
    """WRITEME

    You should inherit from Module with the members will be other Modules or Components.  To
    make more specialized elements of a Module graph, consider inheriting from Component
    directly.
    """
    InstanceType = ModuleInstance # By default, we use build ModuleInstance

    def __init__(self, *args, **kw):
        super(Module, self).__init__(*args, **kw)
        self.__dict__["local_attr"]={}
        self.__dict__["_components"]={}


    def __wrapper__(self, x):
        """
        This function is called whenever x is set as an attribute of
        the Module.
        """
        return wrap(x)

    def __setattr__(self, attr, value):
        # a is a new attribute
        # we will use the local_attr dict to store it
        v = self.unpack_member_and_external(value)

        # this __setattr__ function overrides property.__set__, so we don't worry about a
        # setter here
        self.__dict__[attr] = v
        self.__dict__["local_attr"][attr] = v

    @staticmethod
    def unpack_member_and_external(v):
        if isinstance(v, Member):
            print >> sys.stderr, ("WARNING: assignment of Member "
                    "object %s (either directly or indirectly) to Module "
                    "is deprecated.  Just use Variable." % v)
            return v.r
        elif isinstance(v, External):
            print >> sys.stderr, ("WARNING: assignment of External "
                    "object %s (either directly or indirectly) to Module "
                    "is deprecated.  Just use Variable." % v)
            return v.r
        elif isinstance(v, (gof.Variable,Method,Module)):
            return v
        elif isinstance(v,(int,bool)):
            return v
        elif isinstance(v, (list)):
            return map(Module.unpack_member_and_external,v)
        elif isinstance(v, (tuple)):
            return tuple(map(Module.unpack_member_and_external,v))
        elif isinstance(v,dict):
            v_copy = dict()
            for k,vv in v.iteritems():
                v_copy[k]=Module.unpack_member_and_external(vv)
            return v
        else:
#                raise NotImplementedError
#                print "WARNING: unknow:",v
            return v

    def old__getattr__(self, attr):
        if attr == '_components' and '_components' not in self.__dict__:
            self.__dict__['_components'] = {}
        try:
            rval = self.__dict__["local_attr"][attr]
        except KeyError:
            raise AttributeError('%s has no %s attribute.' % (self.__class__, attr))
        return rval

    def old__setattr__(self, attr, value):
        """
        """
        if attr in ('parent', '_components'):
            self.__dict__[attr] = value
            return
        self.__dict__["local_attr"][attr] = self.unpack_member_and_external(value)

    def build(self, mode, memo):
        if self in memo:
            return memo[self]
        for k,v in self.local_attr.iteritems():
            self.__setattr__(k,v)
        inst = super(Module, self).build(mode, memo)
        if not isinstance(inst, ModuleInstance):
            raise TypeError('The InstanceType of a Module should inherit from ModuleInstance',
                    (self, type(inst)))
        for methodname in dir(self):
            # Any method with a name like '_instance_XXX' is added to
            # the object built under the name obj.XXX
            if methodname.startswith('_instance_'):
                new_methodname = methodname[len('_instance_'):]
                if hasattr(inst, new_methodname):
                    print >> sys.stderr, "WARNING: not overriding already-defined method",
                    print >> sys.stderr, getattr(inst, new_methodname),
                    print >> sys.stderr, "with",
                    print >> sys.stderr, getattr(self, methodname)
                else:
                    curried = Curry(self, methodname, inst)
                    # setattr doesn't work here because we overrode __setattr__
                    # setattr(inst, new_methodname, curried)
                    inst.__dict__[new_methodname] = curried
                    assert getattr(inst, new_methodname) == curried
                    #print 'ADDING METHOD', method, 'to', id(inst), new_methodname, getattr(inst, new_methodname)
        memo[self] = inst
        return inst

    def _instance_initialize(self, inst, init = {}, **kwinit):
        """
        Default initialization method.
        """
        for name, value in chain(init.iteritems(), kwinit.iteritems()):
            inst[name] = value

    def make_module_instance(self, *args, **kwargs):
        """
        Module's __setattr__ method hides all members under local_attr. This
        method iterates over those elements and wraps them so they can be used
        in a computation graph. The "wrapped" members are then set as object
        attributes accessible through the dotted notation syntax (<module_name>
        <dot> <member_name>). Submodules are handled recursively.
        """

        # Function to go through member lists and dictionaries recursively,
        # to look for submodules on which make_module_instance needs to be called
        def recurse(v):
            if isinstance(v,list):
                iter = enumerate(v)
            else:
                iter = v.iteritems()
            #backport
            #iter = enumerate(v) if isinstance(v,list) else v.iteritems()
            for sk,sv in iter:
                if isinstance(sv,(list,dict)):
                    sv = recurse(sv)
                elif isinstance(sv,Module):
                    sv = sv.make_module_instance(args,kwargs)
                v[sk] = sv
            return v

        for k,v in self.local_attr.iteritems():
            if isinstance(v,Module):
                v = v.make_module_instance(args,kwargs)
                self[k] = self.__wrapper__(v)
            elif isinstance(v,Method):
                self.__setitem__(k,v)
            else:
                # iterate through lists and dictionaries to wrap submodules
                if isinstance(v,(list,dict)):
                    self[k] = self.__wrapper__(recurse(v))
                try:
                    self[k] = self.__wrapper__(v)
                except Exception:
                    if isinstance(v, Component):
                        raise
                    else:
                        self.__dict__[k] = v
        return self

    def make(self, *args, **kwargs):
        """
        Allocates the necessary containers using allocate() and uses
        build() to make an instance which will be returned. The
        initialize() method of the instance will be called with the
        arguments and the keyword arguments. If 'mode' is in the
        keyword arguments it will be passed to build().
        """
        self.make_module_instance(args,kwargs)

        mode = kwargs.pop('mode', get_mode.get_default_mode())
        rval = self.make_no_init(mode)
        if hasattr(rval, 'initialize'):
            rval.initialize(*args, **kwargs)
        return rval

    def __str__(self):
        return self.__class__.__name__+"(%s)" % ', '.join(x for x in sorted(map(str, self.local_attr)) if x[0] != '_')

    def __get_name__(self):
        """
        Getter for self.name
        """
        return self._name

    def __set_name__(self, name):
        """
        Setter for self.name
        """
        self._name = name

    name = property(lambda self: self.__get_name__(),
                    lambda self, value: self.__set_name__(value),
                    "Contains the name of this Component")

FancyModule = Module
FancyModuleInstance = ModuleInstance


def func_to_mod(f):
    """
    Creates a dummy module, with external member variables for  the input
    parameters required by the function f, and a member output defined as:
        output <= f(**kwinit)
    """
    def make(**kwinit):
        m = Module()
        outputs = f(**kwinit)
        if isinstance(outputs, list):
            for i,o in enumerate(outputs):
                setattr(m, 'output%(i)i', o)
        else:
            m.output = outputs

        return m
    return make
