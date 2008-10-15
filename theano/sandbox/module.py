
import theano
from theano import gof, compile
from collections import defaultdict
from itertools import chain
from functools import partial
from theano.gof.utils import scratchpad
from copy import copy



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
        except ValueError:
            return x
    return map(convert, name)


class AllocationError(Exception):
    pass



class Component(object):

    def __init__(self):
        self.__dict__['_name'] = ''
        self.__dict__['parent'] = None

    def bind(self, parent, name):
        if self.bound():
            raise Exception("%s is already bound to %s as %s" % (self, self.parent, self.name))
        self.parent = parent
        self.name = join(parent.name, name)

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

    def pretty(self):
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

    def pretty(self):
        rval = '%s :: %s' % (self.__class__.__name__, self.r.type)
        return rval



class External(_RComponent):

    def allocate(self, memo):
        # nothing to allocate
        return None

    def build(self, mode, memo):
        return None

    def pretty(self):
        rval = super(External, self).pretty()
        if self.r.owner:
            rval += '\n= %s' % (pprint.pp2.process(self.r, dict(target = self.r)))
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



from theano.sandbox import pprint
class Method(Component):

    def __init__(self, inputs, outputs, updates = {}, **kwupdates):
        super(Method, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.updates = dict(updates, **kwupdates)
        self.kits = []

    def bind(self, parent, name):
        super(Method, self).bind(parent, name)
        self.resolve_all()

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

    def build(self, mode, memo):
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
        inputs = [compile.In(result = input,
                             value = get_storage(input))
                  for input in inputs]
        inputs += [compile.In(result = k,
                              update = v,
                              value = get_storage(k, True),
                              strict = True)
                   for k, v in self.updates.iteritems()]
        outputs = self.outputs
        _inputs = [x.result for x in inputs]
        for input in gof.graph.inputs(outputs if isinstance(outputs, (list, tuple)) else [outputs]
                                      + [x.update for x in inputs if getattr(x, 'update', False)]):
            if input not in _inputs and not isinstance(input, gof.Value):
                inputs += [compile.In(result = input,
                                      value = get_storage(input, True))]
        inputs += [(kit, get_storage(kit, True)) for kit in self.kits]
        return compile.function(inputs, outputs, mode)

    def pretty(self, header = True, **kwargs):
        self.resolve_all()
#         cr = '\n    ' if header else '\n'
#         rval = ''
#         if header:
#             rval += "Method(%s):" % ", ".join(map(str, self.inputs))
        if self.inputs:
            rval = 'inputs: %s\n' % ", ".join(map(str, self.inputs))
        else:
            rval = ''
        rval += pprint.pp.process_graph(self.inputs, self.outputs, self.updates, False)
        return rval

    def __str__(self):
        return "Method(%s -> %s%s%s)" % \
            (self.inputs,
             self.outputs,
             "; " if self.updates else "",
             ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))



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
            raise TypeError('Cannot set item %s' % item)

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
                for name, x in component.flat_components_map(include_self, path2):
                    yield path2, x
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
        value.bind(self, str(item))
        self._components[item] = value

    def append(self, c):
        if isinstance(c, gof.Result):
            c = Member(c)
        elif not isinstance(c, Component):
            raise TypeError('ComponentList may only contain Components.', c, type(c))
        self._components.append(c)        

    def __str__(self):
        return str(self._components)

    def pretty(self, header = True, **kwargs):
        cr = '\n    ' #if header else '\n'
        strings = []
        #if header:
        #    rval += "ComponentList:"
        for i, c in self.components_map():
            strings.append('%i:%s%s' % (i, cr, c.pretty().replace('\n', cr)))
            #rval += cr + '%i -> %s' % (i, c.pretty(header = True, **kwargs).replace('\n', cr))
        return '\n'.join(strings)

    def __set_name__(self, name):
        super(ComponentList, self).__set_name__(name)
        for i, member in enumerate(self._components):
            member.name = '%s.%i' % (name, i)



class ModuleInstance(CompositeInstance):
    __hide__ = []

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
            if not k.startswith('_') and not callable(v) and not k in self.__hide__:
                pre = '%s: ' % k
                strings.append('%s%s' % (pre, str(v).replace('\n', '\n' + ' '*len(pre))))
        return '{%s}' % '\n'.join(strings).replace('\n', '\n ')


class Module(Composite):
    __instance_type__ = ModuleInstance

    def __init__(self, components = {}, **kwcomponents):
        super(Module, self).__init__()
        components = dict(components, **kwcomponents)
        self._components = components

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
        inst = self.__instance_type__(self, {})
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
        value.bind(self, item)
        self._components[item] = value

    def pretty(self, header = True, **kwargs):
        cr = '\n    ' #if header else '\n'
        strings = []
#         if header:
#             rval += "Module:"
        for name, component in self.components_map():
            if name.startswith('_'):
                continue
            strings.append('%s:%s%s' % (name, cr, component.pretty().replace('\n', cr)))
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
        self.meth(self.arg, *args, **kwargs)
    def __getstate__(self):
        return [self.obj, self.name, self.arg]
    def __setstate__(self, state):
        self.obj, self.name, self.arg = state
        self.meth = getattr(self.obj, self.name)
    

class FancyModuleInstance(ModuleInstance):

    def __getattr__(self, attr):
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
    __instance_type__ = FancyModuleInstance
    
    def __wrapper__(self, x):
        return wrap(x)

    def __getattr__(self, attr):
        try:
            rval = self[attr]
        except KeyError:
            raise AttributeError('%s has no %s attribute.' % (self.__class__, attr))
        if isinstance(rval, (External, Member)):
            return rval.r
        return rval

    def __setattr__(self, attr, value):
        if attr == 'parent':
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


from .. import tensor as T
class RModule(FancyModule):

    def __init__(self, components = {}, **kwcomponents):
        super(RModule, self).__init__(components, **kwcomponents)
        self.random = T.RandomKit('rkit')
        self._components['_rkit'] = KitComponent(self.random)

    def __wrapper__(self, x):
        x = wrap(x)
        if isinstance(x, Method):
            x.kits += [self.random]
        return x

    def _instance_seed(self, inst, seed, recursive = True):
        if recursive:
            for path, c in self.flat_components_map(True):
                if isinstance(c, RModule):
                    inst2 = inst
                    for name in path:
                        inst2 = inst2[name]
                    c._rkit.kit.distribute(seed, xrange(len(inst._rkit)), inst2._rkit)
        else:
            self._rkit.kit.distribute(seed, xrange(len(inst._rkit)), inst._rkit)







