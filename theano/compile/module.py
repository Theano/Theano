
from .. import gof
from ..printing import pprint
from collections import defaultdict
from itertools import chain
from functools import partial
from copy import copy
import io
import function_module as F


def join(*args):
    return ".".join(arg for arg in args if arg)
def split(sym, n=-1):
    return sym.split('.', n)

def canonicalize(name):
    if isinstance(name, str):
        name = split(name)
    def convert(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return x
    return map(convert, name)


class AllocationError(Exception):
    pass

class BindError(Exception):
    pass

class Component(object):

    def __init__(self):
        self.__dict__['_name'] = ''
        self.__dict__['parent'] = None

    def bind(self, parent, name, dup_ok=True):
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
        return self.parent is not None

    def allocate(self, memo):
        raise NotImplementedError

    def build(self, mode, memo):
        raise NotImplementedError

    def make_no_init(self, mode='FAST_COMPILE'):
        memo = {}
        self.allocate(memo)
        rval = self.build(mode, memo)
        return rval

    def make(self, *args, **kwargs):
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
        raise NotImplementedError

    def __get_name__(self):
        return self._name

    def __set_name__(self, name):
        self._name = name

    name = property(lambda self: self.__get_name__(),
                    lambda self, value: self.__set_name__(value))



class _RComponent(Component):

    def __init__(self, r):
        super(_RComponent, self).__init__()
        self.r = r
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

    def allocate(self, memo):
        # nothing to allocate
        return None

    def build(self, mode, memo):
        return None

    def pretty(self, **kwargs):
        rval = super(External, self).pretty()
        if self.r.owner:
            rval += '\n= %s' % (pprint(self.r, dict(target = self.r)))
        return rval



class Member(_RComponent):

    def allocate(self, memo):
        r = self.r
        if memo and r in memo:
            return memo[r]
        rval = gof.Container(r, storage = [None])
        memo[r] = rval
        return rval

    def build(self, mode, memo):
        return memo[self.r]



class Method(Component):

    def __init__(self, inputs, outputs, updates = {}, kits = [], **kwupdates):
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
        if not self.bound():
            raise ValueError('Trying to resolve a name on an unbound Method.')
        result = self.parent.resolve(name)
        if not hasattr(result, 'r'):
            raise TypeError('Expected a Component with subtype Member or External.')
        return result

    def resolve_result(self, x):
        if isinstance(x, gof.Result):
            return x
        else:
            return self.resolve(x).r

    def resolve_all(self):
        if not isinstance(self.inputs, (list, tuple)):
            inputs = [self.inputs]
        else:
            inputs = self.inputs
        self.inputs = [self.resolve_result(input) for input in inputs]
        if isinstance(self.outputs, (list, tuple)):
            self.outputs = [self.resolve_result(output) for output in self.outputs]
        else:
            self.outputs = self.resolve_result(self.outputs)
        updates = self.updates
        self.updates = {}
        for k, v in updates.iteritems():
            k, v = self.resolve_result(k), self.resolve_result(v)
            self.updates[k] = v

    def allocate(self, memo):
        return None

    def build(self, mode, memo, allocate_all = False):
        self.resolve_all()
        def get_storage(r, require = False):
            try:
                return memo[r]
            except KeyError:
                if require:
                    raise AllocationError('There is no storage associated to %s used by %s.'
                                          ' Verify that it is indeed a Member of the'
                                          ' enclosing module or of one of its submodules.' % (r, self))
                else:
                    return gof.Container(r, storage = [None])
        inputs = self.inputs
        inputs = [io.In(result = input,
                        value = get_storage(input),
                        mutable = False)
                  for input in inputs]
        inputs += [io.In(result = k,
                         update = v,
                         value = get_storage(k, not allocate_all),
                         mutable = True,
                         strict = True)
                   for k, v in self.updates.iteritems()]
        outputs = self.outputs
        _inputs = [x.result for x in inputs]
        for input in gof.graph.inputs((list(outputs) if isinstance(outputs, (list, tuple)) else [outputs])
                                      + [x.update for x in inputs if getattr(x, 'update', False)],
                                      blockers = _inputs):
            if input not in _inputs and not isinstance(input, gof.Value):
                inputs += [io.In(result = input,
                                 value = get_storage(input, not allocate_all),
                                 mutable = False)]
        inputs += [(kit, get_storage(kit, not allocate_all)) for kit in self.kits]
        return F.function(inputs, outputs, mode)

    def pretty(self, **kwargs):
        self.resolve_all()
        if self.inputs:
            rval = 'inputs: %s\n' % ", ".join(map(str, self.inputs))
        else:
            rval = ''
        mode = kwargs.pop('mode', None)
        inputs, outputs, updates = self.inputs, self.outputs if isinstance(self.outputs, (list, tuple)) else [self.outputs], self.updates
        if mode:
            f = self.build(mode, {}, True)
            einputs, eoutputs = f.maker.env.inputs, f.maker.env.outputs
            updates = dict(((k, v) for k, v in zip(einputs[len(inputs):], eoutputs[len(outputs):])))
            inputs, outputs = einputs[:len(inputs)], eoutputs[:len(outputs)]
#             nin = len(inputs)
#             nout = len(outputs)
#             k, v = zip(*updates.items()) if updates else ((), ())
#             nup = len(k)
#             eff_in = tuple(inputs) + tuple(k)
#             eff_out = tuple(outputs) + tuple(v)
#             supp_in = tuple(gof.graph.inputs(eff_out))
#             env = gof.Env(*gof.graph.clone(eff_in + supp_in,
#                                            eff_out))
#             sup = F.Supervisor(set(env.inputs).difference(env.inputs[len(inputs):len(eff_in)]))
#             env.extend(sup)
#             mode.optimizer.optimize(env)
#             inputs, outputs, updates = env.inputs[:nin], env.outputs[:nout], dict(zip(env.inputs[nin:], env.outputs[nout:]))
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

    def __init__(self, component, __items__):
        self.__dict__['component'] = component
        self.__dict__['__items__'] = __items__

    def __getitem__(self, item):
        x = self.__items__[item]
        if isinstance(x, gof.Container):
            return x.value
        return x

    def __setitem__(self, item, value):
        x = self.__items__[item]
        if isinstance(x, gof.Container):
            x.value = value
        elif hasattr(x, 'initialize'):
            x.initialize(value)
        else:
            ##self.__items__[item] = value
            raise KeyError('Cannot set item %s' % item)

class Composite(Component):

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
        if include_self:
            yield self
        for component in self.components():
            if isinstance(component, Composite):
                for x in component.flat_components(include_self):
                    yield x
            else:
                yield component

    def flat_components_map(self, include_self = False, path = []):
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
        for member in self.components():
            member.allocate(memo)

    def get(self, item):
        raise NotImplementedError
        
    def set(self, item, value):
        raise NotImplementedError

    def __getitem__(self, item):
        x = self.get(item)
        if isinstance(x, (External, Member)):
            return x.r
        return x
        
    def __setitem__(self, item, value):
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

    def __init__(self, *_components):
        super(ComponentList, self).__init__()
        if len(_components) == 1 and isinstance(_components[0], (list, tuple)):
            _components = _components[0]
        self._components = []
        for c in _components:
            self.append(c)

    def resolve(self, name):
        name = canonicalize(name)
        try:
            item = self.get(name[0])
        except TypeError:
            # if name[0] is not a number, we check in the parent
            if not self.bound():
                raise TypeError('Cannot resolve a non-integer name on an unbound ComponentList.')
            return self.parent.resolve(name)
        if len(name) > 1:
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


class ModuleInstance(CompositeInstance):

    def __setitem__(self, item, value):
        if item not in self.__items__:
            self.__items__[item] = value
            return
        super(ModuleInstance, self).__setitem__(item, value)
    
    def __str__(self):
        strings = []
        for k, v in sorted(self.__items__.iteritems()):
            if isinstance(v, gof.Container):
                v = v.value
            if not k.startswith('_') and not callable(v) and not k in getattr(self, '__hide__', []):
                pre = '%s: ' % k
                strings.append('%s%s' % (pre, str(v).replace('\n', '\n' + ' '*len(pre))))
        return '{%s}' % '\n'.join(strings).replace('\n', '\n ')


class Module(Composite):
    InstanceType = ModuleInstance

    def __init__(self, components = {}, **kwcomponents):
        super(Module, self).__init__()
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
            raise TypeError('Module may only contain Components.', value, type(value))
        value = value.bind(self, item)
        self._components[item] = value

    def pretty(self, **kwargs):
        cr = '\n    ' #if header else '\n'
        strings = []
#         if header:
#             rval += "Module:"
        for name, component in self.components_map():
            if name.startswith('_'):
                continue
            strings.append('%s:%s%s' % (name, cr, component.pretty(**kwargs).replace('\n', cr)))
        strings.sort()
        return '\n'.join(strings)

    def __str__(self):
        return "Module(%s)" % ', '.join(x for x in sorted(map(str, self._components)) if x[0] != '_')

    def __set_name__(self, name):
        super(Module, self).__set_name__(name)
        for mname, member in self._components.iteritems():
            member.name = '%s.%s' % (name, mname)





__autowrappers = []

def register_wrapper(condition, wrapper):
    __autowrappers.append((condition, wrapper))

def wrap(x):
    if isinstance(x, Component):
        return x
    for condition, wrapper in __autowrappers:
        if condition(x):
            return wrapper(x)
    return x

register_wrapper(lambda x: isinstance(x, gof.Result),
                 lambda x: External(x))

register_wrapper(lambda x: isinstance(x, (list, tuple)) and all(isinstance(r, Component) for r in x),
                 lambda x: ComponentList(*x))

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
    

class FancyModuleInstance(ModuleInstance):

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

class FancyModule(Module):
    InstanceType = FancyModuleInstance
    
    def __wrapper__(self, x):
        return wrap(x)

    def __getattr__(self, attr):
        if attr == '_components' and '_components' not in self.__dict__:
            self.__dict__['_components'] = {}
        try:
            rval = self[attr]
        except KeyError:
            raise AttributeError('%s has no %s attribute.' % (self.__class__, attr))
        if isinstance(rval, (External, Member)):
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
        inst = super(FancyModule, self).build(mode, memo)
        for method in dir(self):
            if method.startswith('_instance_'):
                setattr(inst, method[10:], Curry(self, method, inst))
        return inst

    def _instance_initialize(self, inst, init = {}, **kwinit):
        for name, value in chain(init.iteritems(), kwinit.iteritems()):
            inst[name] = value



class KitComponent(Component):
    
    def __init__(self, kit):
        super(KitComponent, self).__init__()
        self.kit = kit

    def allocate(self, memo):
        kit = self.kit
        if kit in memo:
            return memo[kit]
        containers = []
        for input in kit.sinputs:
            r = input.result
            if r not in memo:
                memo[r] = gof.Container(r, storage = [None])
            containers.append(memo[r])
            #containers.append(gof.Container(r, storage = [None]))
        memo[kit] = containers
        return containers

    def build(self, mode, memo):
        return memo[self.kit]







