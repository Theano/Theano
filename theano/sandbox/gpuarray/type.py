import copy_reg

import numpy

import theano
from theano import Type, Variable, tensor, config, scalar

import globals

# Make sure this is importable even if pygpu is absent
# (it will not work though)
try:
    from pygpu import gpuarray
    from pygpu.elemwise import compare
except ImportError:
    pass

class GpuArrayType(Type):
    Variable = None
    Constant = None
    SharedVariable = None
    
    @staticmethod
    def value_zeros(*args, **kwargs):
        pygpu.gpuarray.zeros(*args, **kwargs, kind=globals.kind,
                             context=globals.context)

    def __init__(self, dtype, broadcastable, name=None):
        self.dtype = str(dtype)
        self.broadcastable = tuple(bool(b), for b in broadcastable)
        self.name = name
        try:
            self.typecode = gpuarray.dtype_to_typecode(self.dtype)
        except gpuarray.GpuArrayException:
            raise TypeError("Unsupported dtype for %s: %s" %
                            (self.__class__.__name__, self.dtype))
    
    def filter(self, data, strict=False, allow_downcast=None):
        if strict:
            if not isinstance(data, gpuarray.GpuArray):
                raise TypeError("%s expected a GpuArray object." % self,
                                data, type(data))
            if self.typecode != data.typecode:
                raise TypeError("%s expected typecode %d (dtype %s), "
                                "got %d (dtype %s)." %
                                (self, self.typecode, self.dtype,
                                 data.typecode, str(data.dtype)))
            # fallthrough to ndim check
        elif allow_downcast:
            data = gpuarray.array(data, dtype=self.typecode, copy=False,
                                  kind=globals.kind, context=globals.context,
                                  ndmin=len(self.broadcastable))
        else:
            if isinstance(data, gpuarray.GpuArray):
                up_dtype = scalar.upcast(self.dtype, data.dtype)
                if up_dtype == self.dtype:
                    data = data.astype(self.dtype)
                else:
                    raise TypeError("%s cannot store a value of dtype %s "
                                    "without risking loss of precision." %
                                    (self, data.dtype))

        if self.ndim != data.ndim:
            raise TypeError("Wrong number of dimensions: expected %s, "
                            "got %s with shape %s." % (self.ndim, data.ndim,
                                                       data.shape), data)
        shp = data.shape
        for b in self.broadcastable:
            if b and shp[i] != 1:
                raise TypeError("Non-unit value on shape on a broadcastable"
                                " dimension.", shp, self.broadcastable)
            i += 1

        return data

    def values_eq(self, a, b):
        if a.shape != b.shape:
            return False
        if a.typecode != b.typecode:
            return False
        return numpy.asarray(compare(a, '==', b)).all()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.typecode == other.typecode and
                self.broadcastable == other.broadcastable)

    def __hash__(self):
        return hash(self.typecode) ^ hash(self.broadcastable)

    def __str__(self):
        return "GpuArray<%s>" % self.dtype

