

from utils import AbstractFunctionError


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




    
# class Constructor:

#     def __init__(self):
#         pass
#     def add_module(self, module, module_name, accept=lambda x:issubclass(x, cf.base)):
#         dct = {}
#         for symbol in dir(module):
#             if symbol[:2] == '__': continue
#             obj = getattr(module, symbol)
#             if accept(obj): dct[symbol] = Allocator(obj)
#         class Dummy:pass
#         self.__dict__[module_name] = Dummy()
#         self.__dict__[module_name].__dict__.update(dct)
#     def add_from_module(self, module, accept=lambda x:issubclass(x, cf.base)):
#         for symbol in dir(module):
#             if symbol[:2] == '__': continue
#             obj = getattr(module, symbol)
#             #print 'considering', symbol, obj
#             if accept(obj): self.__dict__[symbol] = Allocator(obj)
#     def add_globals_from_module(self, module, accept=lambda x:issubclass(x, cf.base)):
#         for symbol in dir(module):
#             if symbol[:2] == '__': continue
#             obj = getattr(module, symbol)
#             #print 'considering', symbol, obj
#             if accept(obj):
#                 if hasattr(globals(), symbol):
#                     print 'Warning, overwriting global variable: %s' % symbol
#                 globals()[symbol] = Allocator(obj)
                

# if __name__=='__main__':

#     c = Constructor()
#     c.add_module(cf,'cf')
#     aa,bb = c.cf.A(), c.cf.B()
#     print aa,bb
#     c.add_from_module(cf)
#     a,b = c.A(), c.B()
#     print a,b

#     c.add_globals_from_module(cf)
#     d,e = A(), B()
#     print d,e


