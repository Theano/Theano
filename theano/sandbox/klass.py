
import theano
from theano import gof
from collections import defaultdict
from itertools import chain
from theano.gof.utils import scratchpad
from copy import copy


def join(*args):
    return ".".join(arg for arg in args if arg)
def split(sym, n=-1):
    return sym.split('.', n)



class KlassComponent(object):
    _name = ""
    
    def bind(self, klass, name):
        if self.bound():
            raise Exception("%s is already bound to %s as %s" % (self, self.klass, self.name))
        self.klass = klass
        self.name = join(klass.name, name)

    def bound(self):
        return hasattr(self, 'klass')

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



class KlassResult(KlassComponent):
    
    def __init__(self, r):
        self.r = r

    def __set_name__(self, name):
        super(KlassResult, self).__set_name__(name)
        self.r.name = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.r)



class KlassMember(KlassResult):

    def __init__(self, r):
        if r.owner:
            raise ValueError("A KlassMember must not be the result of a previous computation.")
        super(KlassMember, self).__init__(r)



class KlassMethod(KlassComponent):

    def __init__(self, inputs, outputs, updates = {}, **kwupdates):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        self.inputs = inputs
        self.outputs = outputs
        self.updates = dict(updates, **kwupdates)

    def bind(self, klass, name):
        super(KlassMethod, self).bind(klass, name)
        self.inputs = [klass.resolve(i, KlassResult).r for i in self.inputs]
        self.outputs = [klass.resolve(o, KlassResult).r for o in self.outputs] \
            if isinstance(self.outputs, (list, tuple)) \
            else klass.resolve(self.outputs, KlassResult).r
        updates = self.updates
        self.updates = {}
        self.extend(updates)

    def extend(self, updates = {}, **kwupdates):
        if not hasattr(self, 'klass'):
            self.updates.update(updates)
            self.updates.update(kwupdates)
        else:
            for k, v in chain(updates.iteritems(), kwupdates.iteritems()):
                k, v = self.klass.resolve(k, KlassMember), self.klass.resolve(v, KlassResult)
                self.updates[k.r] = v.r

    def __str__(self):
        return "KlassMethod(%s -> %s%s%s)" % \
            (self.inputs,
             self.outputs,
             "; " if self.updates else "",
             ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))



class Klass(KlassComponent):

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        self.__dict__['__components__'] = {}
        self.__dict__['_name'] = ""
        self.__dict__['__components_list__'] = []
        self.__dict__['__component_names__'] = []
        return self

    ###
    ### Access to the klass members and methods
    ###

    def resolve(self, symbol, filter = None):
        if isinstance(symbol, gof.Result):
            if not filter or filter is KlassResult:
                return KlassResult(symbol)
            for component in self.__components_list__:
                if isinstance(component, Klass):
                    try:
                        return component.resolve(symbol, filter)
                    except:
                        continue
                if isinstance(component, KlassResult) and component.r is symbol:
                    if filter and not isinstance(component, filter):
                        raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
                                        % (filter.__name__, symbol, self, type(component).__name__))
                    return KlassResult(symbol)
            raise ValueError('%s is not part of this klass or any of its inner klasses. Please add it to the structure before you use it.' % symbol)
        elif isinstance(symbol, str):
            sp = split(symbol, 1)
            if len(sp) == 1:
                try:
                    result = self.__components__[symbol]
                except KeyError:
                    raise AttributeError('Could not resolve symbol %s in klass %s' % (symbol, self))
                if filter and not isinstance(result, filter):
                    raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
                                    % (filter.__name__, symbol, self, type(result).__name__))
                return result
            else:
                sp0, spr = sp
                klass = self.__components__[sp0]
                if not isinstance(klass, Klass):
                    raise TypeError('Could not get subattribute %s of %s' % (spr, klass))
                return klass.resolve(spr, filter)
        else:
            raise TypeError('resolve takes a string or Result argument, not %s' % symbol)

    def members(self, as_results = False):
        filtered = [x for x in self.__components_list__ if isinstance(x, KlassMember)]
        if as_results:
            return [x.r for x in filtered]
        else:
            return filtered

    def methods(self):
        filtered = [x for x in self.__components_list__ if isinstance(x, KlassMethod)]
        return filtered

    def member_klasses(self):
        filtered = [x for x in self.__components_list__ if isinstance(x, Klass)]
        return filtered

    ###
    ### Make
    ###

    def __make__(self, mode, stor = None):
        if stor is None:
            stor = scratchpad()
            self.initialize_storage(stor)

        members = []
        methods = []
        rval = KlassInstance()
        for component, name in zip(self.__components_list__, self.__component_names__):
            if isinstance(component, KlassMember):
                container = getattr(stor, name)
                members.append((component, container))
                rval.__finder__[name] = container
            elif isinstance(component, Klass):
                inner, inner_members = component.__make__(mode, getattr(stor, name))
                rval.__dict__[name] = inner
                members += inner_members
            elif isinstance(component, KlassMethod):
                methods.append(component)

        for method in methods:
            inputs = list(method.inputs)
            for (component, container) in members:
                r = component.r
                update = method.updates.get(component.r, component.r)
                inputs.append(theano.In(result = r,
                                        update = update,
                                        value = container,
                                        name = r.name and split(r.name)[-1],
                                        mutable = True,
                                        strict = True))
            fn = theano.function(inputs,
                                 method.outputs,
                                 mode = mode)
            rval.__dict__[split(method.name)[-1]] = fn

        return rval, members

    def make(self, mode = 'FAST_RUN', **init):
        rval = self.__make__(mode)[0]
        self.initialize(rval, **init)
        return rval

    ###
    ### Instance setup and initialization
    ###

    def initialize_storage(self, stor):
        if not hasattr(stor, '__mapping__'):
            stor.__mapping__ = {}
        mapping = stor.__mapping__
        for name, component in self.__components__.iteritems():
            if isinstance(component, Klass):
                sp = scratchpad()
                setattr(stor, name, sp)
                sp.__mapping__ = mapping
                component.initialize_storage(sp)
            elif isinstance(component, KlassMember):
                r = component.r
                if r in mapping:
                    container = mapping[r]
                else:
                    container = gof.Container(r.type,
                                              name = name,
                                              storage = [None])
                    mapping[r] = container
                setattr(stor, name, container)

    def initialize(self, inst, **init):
        for k, v in init.iteritems():
            inst[k] = v

    ###
    ### Magic methods and witchcraft
    ###

    def __setattr__(self, attr, value):
        if attr == 'name':
            self.__set_name__(value)
            return
        elif attr in ['_name', 'klass']:
            self.__dict__[attr] = value
            return
        if isinstance(value, gof.Result):
            value = KlassResult(value)
        if isinstance(value, KlassComponent):
            value.bind(self, attr)
        else:
            self.__dict__[attr] = value
            return
        self.__components__[attr] = value
        self.__components_list__.append(value)
        self.__component_names__.append(attr)
        if isinstance(value, KlassResult):
            value = value.r
        self.__dict__[attr] = value

    def __set_name__(self, name):
        orig = self.name
        super(Klass, self).__set_name__(name)
        for component in self.__components__.itervalues():
            if orig:
                component.name = join(name, component.name[len(orig):])
            else:
                component.name = join(name, component.name)

    def __str__(self):
        n = len(self.name)
        if n: n += 1
        member_names = ", ".join(x.name[n:] for x in self.members())
        if member_names: member_names = "members: " + member_names
        method_names = ", ".join(x.name[n:] for x in self.methods())
        if method_names: method_names = "methods: " + method_names
        klass_names = ", ".join(x.name[n:] for x in self.member_klasses())
        if klass_names: klass_names = "inner: " + klass_names
        return "Klass(%s)" % "; ".join(x for x in [self.name, member_names, method_names, klass_names] if x)



class KlassInstance(object):

    def __init__(self):
        self.__dict__['__finder__'] = {}

    def __getitem__(self, attr):
        if isinstance(attr, str):
            attr = split(attr, 1)
            if len(attr) == 1:
                return self.__finder__[attr[0]].value
            else:
                return getattr(self, attr[0])[attr[1]]
        else:
            raise TypeError('Can only get an item via string format: %s' % attr)

    def __setitem__(self, attr, value):
        if isinstance(attr, str):
            attr = split(attr, 1)
            if len(attr) == 1:
                self.__finder__[attr[0]].value = value
            else:
                getattr(self, attr[0])[attr[1]] = value
        else:
            raise TypeError('Can only set an item via string format: %s' % attr)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


