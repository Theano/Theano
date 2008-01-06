
import gof
import numpy

from copy import copy

# __all__ = ['set_mode', 'get_mode', 'NumpyR', 'NumpyOp']


_mode = 'eval'

def set_mode(mode):
    global _mode
    _mode = mode

def get_mode():
    return _mode

def start_build():
    set_mode('build')

def end_build():
    set_mode('eval')

def build(f, *args, **kwargs):
    start_build()
    r = f(*args, **kwargs)
    end_build()
    return r



# class Proxy(object):

#     __slots__ = ['_obj']
    
#     def __init__(self, obj = None):
#         self._obj = obj

#     def __getattribute__(self, attr):
#         if attr in ['__class__', '_obj']:
#             return object.__getattribute__(self, attr)
#         return getattr(self._obj, attr)

#     def __setattr__(self, attr, value):
#         if attr in ['_obj']:
#             object.__setattr__(self, attr, value)
#         else:
#             setattr(self._obj, attr, value)

#     def __delattr__(self, attr):
#         delattr(self._obj, attr)

#     def __str__(self):
#         return str(self._obj)

#     def __iadd__(self, y):
#         newobj = self._obj.__iadd__(y)
#         if isinstance(newobj, Proxy):
#             newobj = newobj._obj
#         self._obj = newobj
#         return self



# class NumpyR(numpy.ndarray):

#     __add__  = wrapper(numpy.ndarray.__add__)
#     __radd__ = wrapper(numpy.ndarray.__radd__)



class IViewer(gof.ext.Viewer):
    _v_map = {}

    def view_map(self):
        rval = {}
        for i, o in self._v_map.items():
            rval[self.inputs[i]] = self.outputs[o]
        return rval


class IDestroyer(gof.ext.Destroyer):
    _d_map = {}

    def destroy_map(self):
        rval = {}
        for i, o in self._d_map.items():
            rval[self.inputs[i]] = self.outputs[o]
        return rval


class wrapper:

    __slots__ = ['f', 'opclass']
    
    def __init__(self, f, vmap = None, dmap = None):
        self.f = f

        if not callable(f):
            raise TypeError("Can only wrap a callable.")

        bases = [NumpyOp]
        if vmap:
            bases.append(IViewer)
        if dmap:
            bases.append(IDestroyer)
        
        if hasattr(f, '__module__'):
            mod = f.__module__
        elif hasattr(f, '__objclass__'):
            mod = f.__objclass__.__name__
        else:
            mod = ""

        Wrapper = type("Op:" + mod + ":" + f.__name__, tuple(bases), {})
        if vmap: Wrapper._v_map = vmap
        if dmap: Wrapper._d_map = dmap
        
        def thunk(self):
            def ret():
                self.outputs[0].storage = f(*[input.storage for input in self.inputs])
            return ret

        Wrapper.thunk = thunk #.impl = staticmethod(impl)

        self.opclass = Wrapper

    
    def __call__(self, *args):
        op = self.opclass(*args)
        if _mode == 'eval':
            op.thunk()()
#        outputs = [Proxy(o) for o in op.outputs]
        outputs = copy(op.outputs)
        if op.nout == 1:
            return outputs[0]
        else:
            return outputs

        
#         if '_mode' in kwargs:
#             mode = kwargs['_mode']
#             del kwargs['_mode']
#         else:
#             mode = _mode
        
#         if mode == 'eval':
#             return f(*[to_ndarray(arg) for arg in args])
#         elif mode == 'build':
#             o = self.op(*args)
            
#         elif mode == 'build_eval':
#             return f(*args, **kwargs)



class NumpyR(gof.HolderResult):

    def __init__(self, a = None):
        if a is None:
            self.storage = None
        elif isinstance(a, numpy.ndarray):
            self.storage = a
        else:
            raise TypeError("NumpyR constructor expects a ndarray instance.")

    def __add__(self, y):
        return add(self, y)

    def __mul__(self, y):
        return dot(self, y)

    def __iadd__(self, y):
        return iadd(self, y)
    
    def __str__(self):
        return str(self.storage)

    def __repr__(self):
        return repr(self.storage)



def to_numpyr(x):
    if isinstance(x, NumpyR):
        return x
    elif isinstance(x, NumpyOp):
        return x.out
#    elif isinstance(x, Proxy):
#        return to_numpyr(x._obj)
    elif isinstance(x, numpy.ndarray):
        return NumpyR(x)
    else:
        raise TypeError("%s cannot be converted to or encapsulated in a NumpyR instance." % x)



class NumpyOp(gof.Op):

    nout = 1
    
    def __init__(self, *args):
        inputs = [to_numpyr(arg) for arg in args]
        outputs = [NumpyR() for i in xrange(self.nout)]
        gof.Op.__init__(self, inputs, outputs)

#     def __validate__(self):
#         for input in self.inputs:
#             assert isinstance(input, NumpyR)
#         for output in self.outputs:
#             assert isinstance(output, NumpyR)


add = wrapper(numpy.ndarray.__add__)
iadd = wrapper(numpy.ndarray.__iadd__, None, {0: 0})

dot = wrapper(numpy.dot)

