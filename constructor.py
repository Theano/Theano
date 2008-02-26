import unittest
import constructor_fodder as cf

class Allocator:
    def __init__(self, cls, ctor):
        self.cls = cls
        self.ctor = ctor
    def __call__(self, *args, **kwargs):
        rval = self.cls.__new__(self.cls, *args, **kwargs)
        rval.__init__(*args, **kwargs)
        return rval

class ModeOpAllocator:
    def __init__(self, cls, ctor):
        self.cls = cls
        self.ctor = ctor
    def __call__(self, *args, **kwargs):
        op = self.cls.__new__(self.cls, *args, **kwargs)
        op.__init__(*args, **kwargs)
        mode = self.ctor.mode()
        if mode == 'eval':
            op.perform()
            if op.nout == 1:
                return op.out.data
            else:
                return [output.data for output in op.outputs]
        elif mode == 'build_eval':
            op.perform()
        if op.nout == 1:
            return op.out
        else:
            return op.outputs

class Constructor:

    def __init__(self):
        pass
    def add_module(self, module, module_name, accept=lambda x:issubclass(x, cf.base)):
        dct = {}
        for symbol in dir(module):
            if symbol[:2] == '__': continue
            obj = getattr(module, symbol)
            if accept(obj): dct[symbol] = Allocator(obj)
        class Dummy:pass
        self.__dict__[module_name] = Dummy()
        self.__dict__[module_name].__dict__.update(dct)
    def add_from_module(self, module, accept=lambda x:issubclass(x, cf.base)):
        for symbol in dir(module):
            if symbol[:2] == '__': continue
            obj = getattr(module, symbol)
            #print 'considering', symbol, obj
            if accept(obj): self.__dict__[symbol] = Allocator(obj)
    def add_globals_from_module(self, module, accept=lambda x:issubclass(x, cf.base)):
        for symbol in dir(module):
            if symbol[:2] == '__': continue
            obj = getattr(module, symbol)
            #print 'considering', symbol, obj
            if accept(obj):
                if hasattr(globals(), symbol):
                    print 'Warning, overwriting global variable: %s' % symbol
                globals()[symbol] = Allocator(obj)
                

if __name__=='__main__':

    c = Constructor()
    c.add_module(cf,'cf')
    aa,bb = c.cf.A(), c.cf.B()
    print aa,bb
    c.add_from_module(cf)
    a,b = c.A(), c.B()
    print a,b

    c.add_globals_from_module(cf)
    d,e = A(), B()
    print d,e
