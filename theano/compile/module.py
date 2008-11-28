
from .. import gof
from ..printing import pprint
from collections import defaultdict
from itertools import chain
from functools import partial
from copy import copy
import io
import function_module as F


def join(*args):
    """
    Creates a string representation for the given names:
    join('a', 'b', 'c') => 'a.b.c'
    """
    return ".".join(arg for arg in args if arg)

def split(sym, n=-1):
    """
    Gets the names from their joined representation
    split('a.b.c') => ['a', 'b', 'c']
    Returns the n first names, if n==-1 returns all of them.
    split('a.b.c',1) => ['a', 'b.c']
    """
    return sym.split('.', n)

def canonicalize(name):
    """
    Splits the name and converts each name to the
    right type (e.g. "2" -> 2)
    [Fred: why we return the right type? Why int only?]
    """
    if isinstance(name, str):
        name = split(name)
    def convert(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return x
    return map(convert, name)


class AllocationError(Exception):
    """
    Exception raised when a Result has no associated storage.
    """
    pass

class BindError(Exception):
    """
    Exception raised when a Component is already bound and we try to
    bound it again.
    see Component.bind() help for more information.
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

    def bind(self, parent, name, dup_ok=True):
        """
        Marks this component as belonging to the parent (the parent is
        typically a Composite instance). The component can be accessed
        through the parent with the specified name. If dup_ok is True
        and that this Component is already bound, a duplicate of the
        component will be made using the dup() method and the
        duplicate will be bound instead of this Component. If dup_ok
        is False and this Component is already bound, a BindError wil
        be raised.

        bind() returns the Component instance which has been bound to
        the parent. For an unbound instance, this will usually be
        self.
        """
        if self.bound():
            if dup_ok:
                try:
                    return self.dup().bind(parent, name, False)
                except BindError, e:
                    e.args = (e.args[0] +
                              ' ; This seems to have been caused by an implementation of dup'
                              ' that keeps the previous binding (%s)' % self.dup,) + e.args[1:]
                    raise
            else:
                raise BindError("%s is already bound to %s as %s" % (self, self.parent, self.name))
        self.parent = parent
        self.name = join(parent.name, name)
        return self

    def bound(self):
        """
        Returns True if this Component instance is bound to a
        Composite.
        """
        return self.parent is not None

    def allocate(self, memo):
        """
        Populates the memo dictionary with gof.Result -> io.In
        pairings. The value field of the In instance should contain a
        gof.Container instance. The memo dictionary is meant to tell
        the build method of Components where the values associated to
        certain results are stored and how they should behave if they
        are implicit inputs to a Method (needed to compute its
        output(s) but not in the inputs or updates lists).
        """
        raise NotImplementedError

    def build(self, mode, memo):
        """
        Makes an instance of this Component using the mode provided
        and taking the containers in the memo dictionary.

        A Component which builds nothing, such as External or
        Temporary, may return None.
        """
        raise NotImplementedError

    def make_no_init(self, mode='FAST_COMPILE'):
        """
        Allocates the necessary containers using allocate() and uses
        build() with the provided mode to make an instance which will
        be returned.  The initialize() method of the instance will not
        be called.
        """
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
        mode = kwargs.pop('mode', 'FAST_COMPILE')
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

    def dup(self):
        """
        Returns a Component identical to this one, but which is not
        bound to anything and does not retain the original's name.

        This is useful to make Components that are slight variations
        of another or to have Components that behave identically but
        are accessed in different ways.
        """
        raise NotImplementedError()

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
    Base class for a Component wrapping a Result. For internal use.
    """

    def __init__(self, r):
        super(_RComponent, self).__init__()
        self.r = r
        # If self.owns_name is True, then the name of the result
        # may be adjusted when the name of the Component is. Else,
        # the result will always keep its original name. The component
        # will only be allowed to own a result's name if it has no
        # original name to begin with. This allows the user to opt out
        # of the automatic naming scheme if he or she wants to. It is
        # also usually the case that a Result used in more than one
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

    def dup(self):
        return self.__class__(self.r)


class External(_RComponent):
    """
    External represents a Result which comes from somewhere else
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
    Member represents a Result which is a state of a Composite. That
    Result will be accessible from a built Composite and it is
    possible to do updates on Members.

    Member builds a gof.Container.
    """

    def allocate(self, memo):
        """
        If the memo does not have a Container associated to this
        Member's Result, instantiates one and sets it in the memo.
        """
        r = self.r
        if memo and r in memo:
            return memo[r]
        rval = gof.Container(r, storage = [getattr(r, 'data', None)])
        memo[r] = io.In(result = r, value = rval, mutable = False)
        return memo[r]

    def build(self, mode, memo):
        """
        Returns the Container associated to this Member's Result.
        """
        return memo[self.r].value



class Method(Component):

    def __init__(self, inputs, outputs, updates = {}, kits = [], **kwupdates):
        """
        Method is a declaration of a function. It contains inputs,
        outputs and updates. If the Method is part of a Composite
        which holds references to Members, the Method may use them
        without declaring them in the inputs, outputs or updates list.

        [TODO: remove references to kits, for they are not really
        needed anymore]

        inputs, outputs or updates may be strings. In that case, they
        will be resolved in the Composite which is the parent of this
        Method.

        Method builds a Function (same structure as a call to
        theano.function)
        """
        super(Method, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.updates = dict(updates, **kwupdates)
        self.kits = list(kits)

    def bind(self, parent, name, dup_ok=True):
        rval = super(Method, self).bind(parent, name, dup_ok=dup_ok)
        rval.resolve_all()
        return rval

    def resolve(self, name):
        """
        Resolves the name of an input or output in the parent.
        """
        if not self.bound():
            raise ValueError('Trying to resolve a name on an unbound Method.')
        result = self.parent.resolve(name)
        if not hasattr(result, 'r'):
            raise TypeError('Expected a Component with subtype Member or External.')
        return result

    def resolve_result(self, x):
        if isinstance(x, gof.Result):
            return x
        elif isinstance(x, _RComponent):
            return x.r
        else:
            return self.resolve(x).r

    def resolve_all(self):
        """
        Resolves all inputs, outputs and updates that were given as
        strings so that the fields contain the corresponding Result
        instances instead.
        """
        if isinstance(self.inputs, (gof.Result, str)):
            inputs = [self.inputs]
        else:
            inputs = list(self.inputs)
        self.inputs = [self.resolve_result(input) for input in inputs]
        if isinstance(self.outputs, (list, tuple, ComponentList)):
            self.outputs = [self.resolve_result(output) for output in self.outputs]
        else:
            self.outputs = self.resolve_result(self.outputs)
        updates = self.updates
        self.updates = {}
        for k, v in updates.iteritems():
            k, v = self.resolve_result(k), self.resolve_result(v)
            self.updates[k] = v

    def allocate(self, memo):
        """
        Method allocates nothing.
        """
        return None

    def build(self, mode, memo, allocate_all = False):
        """
        Produces a function. If allocate_all is True, storage will be
        allocated for all needed Results, even if there is no
        associated storage for them in the memo. If allocate_all is
        False, storage will only be allocated for Results that are
        reachable from the inputs list.
        """
        self.resolve_all() # resolve all so we don't have to mess with strings
        def get_storage(r, require = False):
            # If require is True, we can only get storage from the memo.
            try:
                return memo[r]
            except KeyError:
                if require:
                    raise AllocationError('There is no storage associated to %s used by %s.'
                                          ' Verify that it is indeed a Member of the'
                                          ' enclosing module or of one of its submodules.' % (r, self))
                else:
                    return io.In(result = r, value = gof.Container(r, storage = [None]), mutable = False)
        # Wrap the inputs in In instances. TODO: allow the inputs to _be_ In instances
        inputs = self.inputs
        inputs = [io.In(result = input,
                        value = get_storage(input).value,
                        mutable = False)
                  for input in inputs]
        # Add the members to update to the inputs. TODO: see above
        inputs += [io.In(result = k,
                         update = v,
                         value = get_storage(k, not allocate_all).value,
                         mutable = True,
                         strict = True)
                   for k, v in self.updates.iteritems()]
        outputs = self.outputs
        _inputs = [x.result for x in inputs]
        # Grab the results that are not accessible from either the inputs or the updates.
        for input in gof.graph.inputs((list(outputs) if isinstance(outputs, (list, tuple)) else [outputs])
                                      + [x.update for x in inputs if getattr(x, 'update', False)],
                                      blockers = _inputs):
            if input not in _inputs and not isinstance(input, gof.Value):
                # Add this input to the inputs; we require that storage already exists for them,
                # but otherwise they are immutable.
                storage = get_storage(input, not allocate_all)
                inputs.append(storage)

        return F.function(inputs, outputs, mode)

    def pretty(self, **kwargs):
        self.resolve_all()
        if self.inputs:
            rval = 'inputs: %s\n' % ", ".join(map(str, self.inputs))
        else:
            rval = ''
        inputs, outputs, updates = self.inputs, self.outputs if isinstance(self.outputs, (list, tuple)) else [self.outputs], self.updates
        
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
        return "Method(%s -> %s%s%s)" % \
            (self.inputs,
             self.outputs,
             "; " if self.updates else "",
             ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))

    def dup(self):
        self.resolve_all()
        return self.__class__(list(self.inputs),
                              list(self.outputs) if isinstance(self.outputs, list) else self.outputs,
                              dict(self.updates),
                              list(self.kits))

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
        x = self.get(item)
        if isinstance(x, (External, Member)):
            return x.r
        return x
        
    def __setitem__(self, item, value):
        # Uses set() internally
        self.set(item, value)

    def __iter__(self):
        return (c.r if isinstance(c, (External, Member)) else c for c in self.components())



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
            self.append(c)

    def resolve(self, name):
        # resolves # to the #th number in the list
        # resolves name string to parent.resolve(name)
        # TODO: eliminate canonicalize
        name = canonicalize(name)
        try:
            item = self.get(name[0])
        except TypeError:
            # if name[0] is not a number, we check in the parent
            if not self.bound():
                raise TypeError('Cannot resolve a non-integer name on an unbound ComponentList.')
            return self.parent.resolve(name)
        if len(name) > 1:
            # TODO: eliminate
            return item.resolve(name[1:])
        return item

    def components(self):
        return self._components

    def components_map(self):
        return enumerate(self._components)

    def build(self, mode, memo):
        builds = [c.build(mode, memo) for c in self._components]
        return ComponentListInstance(self, builds)

    def get(self, item):
        return self._components[item]

    def set(self, item, value):
        if isinstance(value, gof.Result):
            value = Member(value)
        elif not isinstance(value, Component):
            raise TypeError('ComponentList may only contain Components.', value, type(value))
        value = value.bind(self, str(item))
        self._components[item] = value

    def append(self, c):
        if isinstance(c, gof.Result):
            c = Member(c)
        elif not isinstance(c, Component):
            raise TypeError('ComponentList may only contain Components.', c, type(c))
        self._components.append(c)

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

    def dup(self):
        return self.__class__(*[c.dup() for c in self._components])


def default_initialize(self, init = {}, **kwinit):
    for k, initv in dict(init, **kwinit).iteritems():
        self[k] = initv

class ComponentDictInstance(CompositeInstance):
    """
    ComponentDictInstance is meant to be instantiated by ComponentDict.
    """

    def __setitem__(self, item, value):
        if item not in self.__items__:
            # Set it if it's not there
            # TODO: is this needed here? move to ModuleInstance?
            self.__items__[item] = value
            return
        super(ComponentDictInstance, self).__setitem__(item, value)
    
    def __str__(self):
        strings = []
        for k, v in sorted(self.__items__.iteritems()):
            if isinstance(v, gof.Container):
                v = v.value
            if not k.startswith('_') and not callable(v) and not k in getattr(self, '__hide__', []):
                pre = '%s: ' % k
                strings.append('%s%s' % (pre, str(v).replace('\n', '\n' + ' '*len(pre))))
        return '{%s}' % '\n'.join(strings).replace('\n', '\n ')


class ComponentDict(Composite):
    InstanceType = ComponentDictInstance # Type used by build() to make the instance

    def __init__(self, components = {}, **kwcomponents):
        super(ComponentDict, self).__init__()
        components = dict(components, **kwcomponents)
        self.__dict__['_components'] = components

    def resolve(self, name):
        name = canonicalize(name)
        item = self.get(name[0])
        if len(name) > 1:
            return item.resolve(name[1:])
        return item

    def components(self):
        return self._components.itervalues()

    def components_map(self):
        return self._components.iteritems()

    def build(self, mode, memo):
        inst = self.InstanceType(self, {})
        for name, c in self._components.iteritems():
            x = c.build(mode, memo)
            if x is not None:
                inst[name] = x
        return inst

    def get(self, item):
        return self._components[item]

    def set(self, item, value):
        if not isinstance(value, Component):
            raise TypeError('ComponentDict may only contain Components.', value, type(value))
        value = value.bind(self, item)
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
        return "ComponentDict(%s)" % ', '.join(x for x in sorted(map(str, self._components)) if x[0] != '_')

    def __set_name__(self, name):
        super(ComponentDict, self).__set_name__(name)
        for mname, member in self._components.iteritems():
            member.name = '%s.%s' % (name, mname)





__autowrappers = []

def register_wrapper(condition, wrapper):
    __autowrappers.append((condition, wrapper))

def wrap(x):
    """
    Wraps x in a Component. Wrappers can be registered using
    register_wrapper to allow wrapping more types.
    """
    if isinstance(x, Component):
        return x
    for condition, wrapper in __autowrappers:
        if condition(x):
            return wrapper(x)
    return x

# Result -> External
register_wrapper(lambda x: isinstance(x, gof.Result),
                 lambda x: External(x))

# [Component1, Component2, ...] -> ComponentList(Component1, Component2, ...)
register_wrapper(lambda x: isinstance(x, (list, tuple)) and all(isinstance(r, Component) for r in x),
                 lambda x: ComponentList(*x))

# [Result1, Result2, ...] -> ComponentList(Member(Result1), Member(Result2), ...)
register_wrapper(lambda x: isinstance(x, (list, tuple)) \
                     and all(isinstance(r, gof.Result) and not r.owner for r in x),
                 lambda x: ComponentList(*map(Member, x)))


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
    

class ModuleInstance(ComponentDictInstance):
    """
    ModuleInstance is meant to be instantiated by Module. This differs
    from ComponentDictInstance on a key point, which is that getattr
    does a similar thing to getitem.

    ModuleInstance is compatible for use as ComponentDict.InstanceType.
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
    InstanceType = ModuleInstance # By default, we use build ModuleInstance
    
    def __wrapper__(self, x):
        """
        This function is called whenever x is set as an attribute of
        the Module.
        """
        return wrap(x)

    def __getattr__(self, attr):
        if attr == '_components' and '_components' not in self.__dict__:
            self.__dict__['_components'] = {}
        try:
            rval = self[attr]
        except KeyError:
            raise AttributeError('%s has no %s attribute.' % (self.__class__, attr))
        if isinstance(rval, (External, Member)):
            # Special treatment for External and Member, so that
            # the user may use them to build graphs more easily.
            return rval.r
        return rval

    def __setattr__(self, attr, value):
        if attr in ('parent', '_components'):
            self.__dict__[attr] = value
            return
        elif attr == 'name':
            self.__set_name__(value)
            return
            
        value = self.__wrapper__(value)
        try:
            self[attr] = value
        except:
            if isinstance(value, Component):
                raise
            else:
                self.__dict__[attr] = value

    def build(self, mode, memo):
        inst = super(Module, self).build(mode, memo)
        for method in dir(self):
            # Any method with a name like '_instance_XXX' is added to
            # the object built under the name obj.XXX
            if method.startswith('_instance_'):
                setattr(inst, method[10:], Curry(self, method, inst))
        return inst

    def _instance_initialize(self, inst, init = {}, **kwinit):
        """
        Default initialization method.
        """
        for name, value in chain(init.iteritems(), kwinit.iteritems()):
            inst[name] = value

FancyModule = Module
FancyModuleInstance = ModuleInstance


class KitComponent(Component):
    """
    Represents a SymbolicInputKit (see io.py).
    """
    
    def __init__(self, kit):
        super(KitComponent, self).__init__()
        self.kit = kit

    def allocate(self, memo):
        """
        Allocates a Container for each input in the kit. Sets a key in
        the memo that maps the SymbolicInputKit to the list of
        Containers.
        """
        for input in self.kit.sinputs:
            r = input.result
            if r not in memo:
                input = copy(input)
                input.value = gof.Container(r, storage = [None])
                memo[r] = input

    def build(self, mode, memo):
        return [memo[i.result].value for i in self.kit.sinputs]







