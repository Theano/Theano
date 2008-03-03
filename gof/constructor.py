

from utils import AbstractFunctionError



class Dispatcher(list):

    all_dispatchers = {}

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.all_dispatchers[name] = self

    def __call__(self, *inputs, **opts):
        for candidate in self:
            try:
                return candidate(*inputs, **opts)
            except TypeError:
                continue
        if opts:
            s = " with options %s" % opts
        else:
            s = ""
        raise OmegaTypeError("No candidate found for %s(%s) %s" \
                             % (self.name,
                                ", ".join([input.__class__.__name__ for input in inputs]),
                                s))

    def add_handler(self, x):
        self.insert(0, x)

    def fallback_handler(self, x):
        self.append(x)




class Allocator:

    def __init__(self, fn):
        self.fn = fn


class IdentityAllocator(Allocator):

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)



class Constructor(dict):

    def __init__(self, allocator):
        self._allocator = allocator

    def add_from_module(self, module):
        for symbol in dir(module):
            if symbol[:2] == '__': continue
            obj = getattr(module, symbol)
            try:
                self[symbol] = self._allocator(obj)
            except TypeError:
                pass

    def add_module(self, module, module_name = None):
        if module_name is None:
            module_name = module.__name__
        d = Constructor(self._allocator)
        d.add_from_module(module)
        self[module_name] = d

    def update(self, d, can_fail = False):
        for name, fn in d.items():
            self.add(name, fn, can_fail)

    def add(self, name, fn, can_fail = True):
        if isinstance(fn, Constructor):
            self[name] = fn
        else:
            try:
                self[name] = self._allocator(fn)
            except TypeError:
                if can_fail:
                    raise

    def __getattr__(self, attr):
        return self[attr]



