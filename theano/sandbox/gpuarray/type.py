import numpy

import theano
from theano import Type, Variable, Constant, tensor, config, scalar
from theano.compile import SharedVariable

# Make sure this is importable even if pygpu is absent
# (it will not work though)
try:
    import pygpu
    from pygpu import gpuarray
    from pygpu.elemwise import compare
    from basic_ops import host_from_gpu, gpu_from_host
except ImportError:
    pass

class GpuArrayType(Type):
    def value_zeros(self, shape):
        return pygpu.gpuarray.zeros(shape, dtype=self.typecode, kind=self.kind,
                                    context=self.context)

    def __init__(self, dtype, broadcastable, kind=None, context=None,
                 name=None):
        import globals
        if kind is None:
            kind = globals.kind
        if context is None:
            context = globals.context
        # In case this was not provided and no global value is available
        if kind is None:
            raise RuntimeError("pygpu is not initialized")
        self.dtype = str(dtype)
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.ndim = len(self.broadcastable)
        self.kind = kind
        self.context = context
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
            if self.kind != data.kind:
                raise TypeError("kind of GpuArray does not match")
            if self.context != data.context:
                raise TypeError("context of GpuArray differs")
            if self.typecode != data.typecode:
                raise TypeError("%s expected typecode %d (dtype %s), "
                                "got %d (dtype %s)." %
                                (self, self.typecode, self.dtype,
                                 data.typecode, str(data.dtype)))
            # fallthrough to ndim check
        elif allow_downcast:
            data = gpuarray.array(data, dtype=self.typecode, copy=False,
                                  kind=self.kind, context=self.context,
                                  ndmin=len(self.broadcastable))
        else:
            up_dtype = scalar.upcast(self.dtype, data.dtype)
            if up_dtype == self.dtype:
                data = gpuarray.array(data, dtype=self.typecode, copy=False,
                                      kind=self.kind, context=self.context)
            else:
                raise TypeError("%s cannot store a value of dtype %s "
                                "without risking loss of precision." %
                                (self, data.dtype))

        if self.ndim != data.ndim:
            raise TypeError("Wrong number of dimensions: expected %s, "
                            "got %s with shape %s." % (self.ndim, data.ndim,
                                                       data.shape), data)
        shp = data.shape
        for i, b in enumerate(self.broadcastable):
            if b and shp[i] != 1:
                raise TypeError("Non-unit value on shape on a broadcastable"
                                " dimension.", shp, self.broadcastable)
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
                self.broadcastable == other.broadcastable and
                self.kind == other.kind and
                self.context == other.context)

    def __hash__(self):
        return (hash(self.typecode) ^ hash(self.broadcastable) ^
                hash(self.kind) ^ hash(self.context))

    def __str__(self):
        return "GpuArray<%s>" % self.dtype



class _operators(tensor.basic._tensor_py_operators):
    def _as_TensorVariable(self):
        return host_from_gpu(self)

    def _as_GpuArrayVariable(self):
        return self

    dtype = property(lambda s: s.type.dtype)
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)


class GpuArrayVariable(_operators, Variable):
    pass


GpuArrayType.Variable = GpuArrayVariable


class GpuArraySignature(tensor.basic.TensorConstantSignature):
    pass  # might do something better if we can run the sum on the
          # GPU, but for now this will suffice.


class GpuArrayConstant(_operators, Constant):
    def signature(self):
        return GpuArraySignature((self.type, numpy.asarray(self.data)))

    def __str__(self):
        if self.name is not None:
            return self.name
        return "GpuArrayConstant{%s}" % numpy.asarray(self.data)


GpuArrayType.Constant = GpuArrayConstant


class GpuArraySharedVariable(_operators, SharedVariable):
    def get_value(self, borrow=False, return_internal_type=False):
        if return_internal_type:
            if borrow:
                return self.container.value
            else:
                return self.container.value.copy()
        else:
            return numpy.asarray(self.container.value)

    def set_value(self, value, borrow=False):
        self.container.value = pygpu.gpuarray.array(value, copy=(not borrow))

    def __getitem__(self, *args):
        return _operators.__getitem__(self, *args)


GpuArrayType.SharedVariable = GpuArraySharedVariable


def gpuarray_shared_constructor(value, name=None, strict=False,
                                allow_downcast=None, borrow=False,
                                broadcastable=None, kind=None, context=None):
    """SharedVariable constructor for GpuArrayType"""
    if not isinstance(value, (numpy.ndarray, pygpu.gpuarray.GpuArray)):
        raise TypeError('ndarray or GpuArray required')

    if broadcastable is None:
        broadcastable = (False,) * value.ndim
    type = GpuArrayType(value.dtype, broadcastable, kind=kind, context=context)
    deviceval = pygpu.gpuarray.array(value, copy=(not borrow), kind=type.kind,
                                     context=type.context)
    return GpuArraySharedVariable(type=type, value=deviceval, name=name,
                                  strict=strict)
