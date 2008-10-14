
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

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        self._name = ''
        self.parent = None
        return self

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

    def make(self, mode = 'FAST_RUN', init = None, **kwinit):
        memo = {}
        self.allocate(memo)
        rval = self.build(mode, memo)
        if init and kwinit:
            rval.initialize(init, **kwinit)
        elif init:
            rval.initialize(init)
        elif kwinit:
            rval.initialize(**kwinit)
        return rval

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__

    def __get_name__(self):
        return self._name

    def __set_name__(self, name):
        self._name = name

    name = property(lambda self: self.__get_name__(),
                    lambda self, value: self.__set_name__(value))



class External(Component):

    def __init__(self, r):
        self.r = r
        self.owns_name = r.name is None

    def allocate(self, memo):
        # nothing to allocate
        return None

    def build(self, mode, memo):
        return None

    def __set_name__(self, name):
        super(External, self).__set_name__(name)
        if self.owns_name:
            self.r.name = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.r)



# class MemberInstance:
    
#     def __init__(self, storage):
#         self.storage = storage

#     def get(self):
#         return self.storage.value

#     def set(self, value):
#         self.storage.value = value

class Member(Component):

    def __init__(self, r):
        self.r = r

    def allocate(self, memo):
        r = self.r
        if memo and r in memo:
            return memo[r]
        rval = gof.Container(r, storage = [None])
        memo[r] = rval
        return rval

    def build(self, mode, memo):
        return memo[self.r]
        #return MemberInstance(memo[self.r])

    def __set_name__(self, name):
        super(Member, self).__set_name__(name)
        self.r.name = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.r)



class Method(Component):

    def __init__(self, inputs, outputs, updates = {}, **kwupdates):
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
        if isinstance(x, gof.Result):# or \
                #isinstance(x, compile.SymbolicInput) or \
                #isinstance(x, compile.SymbolicInputKit):
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
        else:
            x.initialize(value)
        #raise TypeError('Cannot set this item: %s' % item)

    def initialize(self, init):
        for i, initv in enumerate(init):
            self[i] = initv

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



class ComponentList(Composite):

    def __init__(self, *_components):
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
        return CompositeInstance(self, builds)

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

    def __set_name__(self, name):
        super(ComponentList, self).__set_name__(name)
        for i, member in enumerate(self._components):
            member.name = '%s.%i' % (name, i)



class ModuleInstance(CompositeInstance):

    def __setitem__(self, item, value):
        if item not in self.__items__:
            self.__items__[item] = value
            return
        super(ModuleInstance, self).__setitem__(item, value)

    def initialize(self, init = {}, **kwinit):
        for name, value in chain(init.iteritems(), kwinit.iteritems()):
            self[name] = value

class Module(Composite):
    __instance_type__ = ModuleInstance

    def __init__(self, components = {}, **kwcomponents):
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

register_wrapper(lambda x: isinstance(x, list) and all(isinstance(r, Component) for r in x),
                 lambda x: ComponentList(*x))

register_wrapper(lambda x: isinstance(x, list) \
                     and all(isinstance(r, gof.Result) and not r.owner for r in x),
                 lambda x: ComponentList(*map(Member, x)))


class FancyModuleInstance(ModuleInstance):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        try:
            self[attr] = value
        except:
            self.__dict__[attr] = value

class FancyModule(Module):
    __instance_type__ = FancyModuleInstance
    
    def __wrapper__(self, x):
        return wrap(x)

    def __getattr__(self, attr):
        rval = self[attr]
        if isinstance(rval, (External, Member)):
            return rval.r
        return rval

    def __setattr__(self, attr, value):
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
                setattr(inst, method[10:], partial(getattr(self, method), inst))
        return inst

    def _instance_initialize(self, inst, init = {}, **kwinit):
        for name, value in chain(init.iteritems(), kwinit.iteritems()):
            inst[name] = value



class KitComponent(Component):
    
    def __init__(self, kit):
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

#         kit = self.kit
#         if kit in memo:
#             return memo[kit]
#         rval = [gof.Container(input.result,
#                               storage = [None])
#                 for input in kit.sinputs]
#         memo[kit] = rval
#         return rval

    def build(self, mode, memo):
        return memo[self.kit]


from theano import tensor as T
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








from theano import tensor as T

x, y = T.scalars('xy')
s = T.scalar()
s1, s2, s3 = T.scalars('s1', 's2', 's3')
#rterm = T.random.random_integers(T.shape(s), 100, 1000)

# print T.random.sinputs

# f = compile.function([x, 
#                       ((s, s + x + rterm), 10),
#                       (T.random, 10)],
#                      s + x)


# print f[s]
# print f(10)
# print f[s]



mod = RModule()
mod.s = Member(s)
#mod.list = ComponentList(Member(s1), Member(s2))
#mod.list = [Member(s1), Member(s2)]
mod.list = [s1, s2]
mod.inc = Method(x, s + x,
                 s = mod.s + x + mod.random.random_integers((), 100, 1000))
mod.dec = Method(x, s - x,
                 s = s - x)
mod.sadd = Method([], s1 + mod.list[1])

m = mod.random.normal([], 1., 1.)
mod.test1 = Method([], m)
mod.test2 = Method([], m)

mod.whatever = 123

print mod._components

inst = mod.make(s = 2, list = [900, 9000])

inst.seed(10)
print inst.test1()
print inst.test1()
print inst.test2()
inst.seed(10)
print inst.test1()
print inst.test2()

print inst.s
inst.seed(10)
inst.inc(3)
print inst.s
inst.dec(4)
print inst.s

print inst.list[0]
print inst.list[1]

inst.list = [1, 2]

print inst.sadd()
inst.initialize(list = [10, -17])
print inst.sadd()





































# class KlassComponent(object):
#     _name = ""
    
#     def bind(self, klass, name):
#         if self.bound():
#             raise Exception("%s is already bound to %s as %s" % (self, self.klass, self.name))
#         self.klass = klass
#         self.name = join(klass.name, name)

#     def bound(self):
#         return hasattr(self, 'klass')

#     def __repr__(self):
#         return str(self)

#     def __str__(self):
#         return self.__class__.__name__

#     def __get_name__(self):
#         return self._name

#     def __set_name__(self, name):
#         self._name = name

#     name = property(lambda self: self.__get_name__(),
#                     lambda self, value: self.__set_name__(value))



# class KlassResult(KlassComponent):
    
#     def __init__(self, r):
#         self.r = r

#     def __set_name__(self, name):
#         super(KlassResult, self).__set_name__(name)
#         self.r.name = name

#     def __str__(self):
#         return "%s(%s)" % (self.__class__.__name__, self.r)



# class KlassMember(KlassResult):

#     def __init__(self, r):
#         if r.owner:
#             raise ValueError("A KlassMember must not be the result of a previous computation.")
#         super(KlassMember, self).__init__(r)



# class KlassMethod(KlassComponent):

#     def __init__(self, inputs, outputs, updates = {}, **kwupdates):
#         if not isinstance(inputs, (list, tuple)):
#             inputs = [inputs]
#         self.inputs = inputs
#         self.outputs = outputs
#         self.updates = dict(updates, **kwupdates)

#     def bind(self, klass, name):
#         super(KlassMethod, self).bind(klass, name)
#         self.inputs = [klass.resolve(i, KlassResult).r for i in self.inputs]
#         self.outputs = [klass.resolve(o, KlassResult).r for o in self.outputs] \
#             if isinstance(self.outputs, (list, tuple)) \
#             else klass.resolve(self.outputs, KlassResult).r
#         updates = self.updates
#         self.updates = {}
#         self.extend(updates)

#     def extend(self, updates = {}, **kwupdates):
#         if not hasattr(self, 'klass'):
#             self.updates.update(updates)
#             self.updates.update(kwupdates)
#         else:
#             for k, v in chain(updates.iteritems(), kwupdates.iteritems()):
#                 k, v = self.klass.resolve(k, KlassMember), self.klass.resolve(v, KlassResult)
#                 self.updates[k.r] = v.r

#     def __str__(self):
#         return "KlassMethod(%s -> %s%s%s)" % \
#             (self.inputs,
#              self.outputs,
#              "; " if self.updates else "",
#              ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))



# class Klass(KlassComponent):

#     def __new__(cls, *args, **kwargs):
#         self = object.__new__(cls)
#         self.__dict__['__components__'] = {}
#         self.__dict__['_name'] = ""
#         self.__dict__['__components_list__'] = []
#         self.__dict__['__component_names__'] = []
#         return self

#     ###
#     ### Access to the klass members and methods
#     ###

#     def resolve(self, symbol, filter = None):
#         if isinstance(symbol, gof.Result):
#             if not filter or filter is KlassResult:
#                 return KlassResult(symbol)
#             for component in self.__components_list__:
#                 if isinstance(component, Klass):
#                     try:
#                         return component.resolve(symbol, filter)
#                     except:
#                         continue
#                 if isinstance(component, KlassResult) and component.r is symbol:
#                     if filter and not isinstance(component, filter):
#                         raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
#                                         % (filter.__name__, symbol, self, type(component).__name__))
#                     return KlassResult(symbol)
#             raise ValueError('%s is not part of this klass or any of its inner klasses. Please add it to the structure before you use it.' % symbol)
#         elif isinstance(symbol, str):
#             sp = split(symbol, 1)
#             if len(sp) == 1:
#                 try:
#                     result = self.__components__[symbol]
#                 except KeyError:
#                     raise AttributeError('Could not resolve symbol %s in klass %s' % (symbol, self))
#                 if filter and not isinstance(result, filter):
#                     raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
#                                     % (filter.__name__, symbol, self, type(result).__name__))
#                 return result
#             else:
#                 sp0, spr = sp
#                 klass = self.__components__[sp0]
#                 if not isinstance(klass, Klass):
#                     raise TypeError('Could not get subattribute %s of %s' % (spr, klass))
#                 return klass.resolve(spr, filter)
#         else:
#             raise TypeError('resolve takes a string or Result argument, not %s' % symbol)

#     def members(self, as_results = False):
#         filtered = [x for x in self.__components_list__ if isinstance(x, KlassMember)]
#         if as_results:
#             return [x.r for x in filtered]
#         else:
#             return filtered

#     def methods(self):
#         filtered = [x for x in self.__components_list__ if isinstance(x, KlassMethod)]
#         return filtered

#     def member_klasses(self):
#         filtered = [x for x in self.__components_list__ if isinstance(x, Klass)]
#         return filtered

#     ###
#     ### Make
#     ###

#     def __make__(self, mode, stor = None):
#         if stor is None:
#             stor = scratchpad()
#             self.initialize_storage(stor)

#         members = []
#         methods = []
#         rval = KlassInstance()
#         for component, name in zip(self.__components_list__, self.__component_names__):
#             if isinstance(component, KlassMember):
#                 container = getattr(stor, name)
#                 members.append((component, container))
#                 rval.__finder__[name] = container
#             elif isinstance(component, Klass):
#                 inner, inner_members = component.__make__(mode, getattr(stor, name))
#                 rval.__dict__[name] = inner
#                 members += inner_members
#             elif isinstance(component, KlassMethod):
#                 methods.append(component)

#         for method in methods:
#             inputs = list(method.inputs)
#             for (component, container) in members:
#                 r = component.r
#                 update = method.updates.get(component.r, component.r)
#                 inputs.append(theano.In(result = r,
#                                         update = update,
#                                         value = container,
#                                         name = r.name and split(r.name)[-1],
#                                         mutable = True,
#                                         strict = True))
#             fn = theano.function(inputs,
#                                  method.outputs,
#                                  mode = mode)
#             rval.__dict__[split(method.name)[-1]] = fn

#         return rval, members

#     def make(self, mode = 'FAST_RUN', **init):
#         rval = self.__make__(mode)[0]
#         self.initialize(rval, **init)
#         return rval

#     ###
#     ### Instance setup and initialization
#     ###

#     def initialize_storage(self, stor):
#         if not hasattr(stor, '__mapping__'):
#             stor.__mapping__ = {}
#         mapping = stor.__mapping__
#         for name, component in self.__components__.iteritems():
#             if isinstance(component, Klass):
#                 sp = scratchpad()
#                 setattr(stor, name, sp)
#                 sp.__mapping__ = mapping
#                 component.initialize_storage(sp)
#             elif isinstance(component, KlassMember):
#                 r = component.r
#                 if r in mapping:
#                     container = mapping[r]
#                 else:
#                     container = gof.Container(r.type,
#                                               name = name,
#                                               storage = [None])
#                     mapping[r] = container
#                 setattr(stor, name, container)

#     def initialize(self, inst, **init):
#         for k, v in init.iteritems():
#             inst[k] = v

#     ###
#     ### Magic methods and witchcraft
#     ###

#     def __setattr__(self, attr, value):
#         if attr == 'name':
#             self.__set_name__(value)
#             return
#         elif attr in ['_name', 'klass']:
#             self.__dict__[attr] = value
#             return
#         if isinstance(value, gof.Result):
#             value = KlassResult(value)
#         if isinstance(value, KlassComponent):
#             value.bind(self, attr)
#         else:
#             self.__dict__[attr] = value
#             return
#         self.__components__[attr] = value
#         self.__components_list__.append(value)
#         self.__component_names__.append(attr)
#         if isinstance(value, KlassResult):
#             value = value.r
#         self.__dict__[attr] = value

#     def __set_name__(self, name):
#         orig = self.name
#         super(Klass, self).__set_name__(name)
#         for component in self.__components__.itervalues():
#             if orig:
#                 component.name = join(name, component.name[len(orig):])
#             else:
#                 component.name = join(name, component.name)

#     def __str__(self):
#         n = len(self.name)
#         if n: n += 1
#         member_names = ", ".join(x.name[n:] for x in self.members())
#         if member_names: member_names = "members: " + member_names
#         method_names = ", ".join(x.name[n:] for x in self.methods())
#         if method_names: method_names = "methods: " + method_names
#         klass_names = ", ".join(x.name[n:] for x in self.member_klasses())
#         if klass_names: klass_names = "inner: " + klass_names
#         return "Klass(%s)" % "; ".join(x for x in [self.name, member_names, method_names, klass_names] if x)



# class KlassInstance(object):

#     def __init__(self):
#         self.__dict__['__finder__'] = {}

#     def __getitem__(self, attr):
#         if isinstance(attr, str):
#             attr = split(attr, 1)
#             if len(attr) == 1:
#                 return self.__finder__[attr[0]].value
#             else:
#                 return getattr(self, attr[0])[attr[1]]
#         else:
#             raise TypeError('Can only get an item via string format: %s' % attr)

#     def __setitem__(self, attr, value):
#         if isinstance(attr, str):
#             attr = split(attr, 1)
#             if len(attr) == 1:
#                 self.__finder__[attr[0]].value = value
#             else:
#                 getattr(self, attr[0])[attr[1]] = value
#         else:
#             raise TypeError('Can only set an item via string format: %s' % attr)

#     def __getattr__(self, attr):
#         return self[attr]

#     def __setattr__(self, attr, value):
#         self[attr] = value






















# # class Method(Component):

# #     __priority__ = -1

# #     def __init__(self, inputs, outputs, updates = {}, **kwupdates):
# #         if not isinstance(inputs, (list, tuple)):
# #             inputs = [inputs]
# #         self._inputs = inputs
# #         self._outputs = outputs
# #         self._updates = dict(updates, **kwupdates)

# #     def bind(self, parent, name):
# #         super(Method, self).bind(parent, name)
# #         self.inputs = self.inputs
# #         self.outputs = self.outputs
# #         self.updates = self.updates

# #     def resolve(self, name):
# #         if not self.bound():
# #             raise ValueError('Trying to resolve a name on an unbound Method.')
# #         result = self.parent.resolve(name)
# #         if not hasattr(result, 'r'):
# #             raise TypeError('Expected a Component with subtype Member or External.')
# #         return result

# #     def resolve_result(self, x):
# #         if isinstance(x, str):
# #             return self.resolve(x).r
# #         else:
# #             return x

# #     def set_inputs(self, inputs):
# #         if not isinstance(outputs, (list, tuple)):
# #             inputs = [inputs]
# #         if not self.bound():
# #             self._inputs = inputs
# #             return
# #         self._inputs = [self.resolve_result(input) for input in inputs]

# #     def set_input(self, i, input):
# #         if not self.bound():
# #             self._inputs[i] = input
# #             return
# #         self._inputs[i] = self.resolve(input).r \
# #             if isinstance(input, str) \
# #             else input

# #     def set_outputs(self, outputs):
# #         if not self.bound():
# #             self._outputs = outputs
# #             return
# #         if isinstance(outputs, (list, tuple)):
# #             self._outputs = [self.resolve_result(output) for output in outputs]
# #         else:
# #             self._outputs = self.resolve_result(outputs)

# #     def set_output(self, i, output):
# #         if not self.bound():
# #             self._outputs[i] = output
# #             return
# #         if not isinstance(self.outputs, (list, tuple)) and i == 0:
# #             self.set_outputs(output)
# #         else:
# #             self._outputs[i] = self.resolve_result(output)

# #     def set_updates(self, updates):
# #         self._updates = {}
# #         self.add_updates(updates)

# #     def add_updates(self, updates = {}, **kwupdates):
# #         if not self.bound():
# #             self.updates.update(updates)
# #             self.updates.update(kwupdates)
# #         else:
# #             for k, v in chain(updates.iteritems(), kwupdates.iteritems()):
# #                 k, v = self.resolve_result(k), self.resolve_result(v)
# #                 self.updates[k] = v

# #     inputs = property(lambda self: self._inputs, set_inputs)
# #     outputs = property(lambda self: self._outputs, set_outputs)
# #     updates = property(lambda self: self._updates, set_updates)

# #     def __str__(self):
# #         return "Method(%s -> %s%s%s)" % \
# #             (self.inputs,
# #              self.outputs,
# #              "; " if self.updates else "",
# #              ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))





# #         self.inputs = [parent.resolve(i).r for i in self.inputs]
# #         self.outputs = [parent.resolve(o).r for o in self.outputs] \
# #             if isinstance(self.outputs, (list, tuple)) \
# #             else parent.resolve(self.outputs, KlassResult).r
# #         updates = self.updates
# #         self.updates = {}
# #         self.extend(updates)
