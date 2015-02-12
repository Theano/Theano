import numpy

import theano
from theano.tensor.var import _tensor_py_operators
from theano import Type, Variable, Constant, tensor, config, scalar
from theano.compile import SharedVariable

# Make sure this is importable even if pygpu is absent
# (it will not work though)
try:
    import pygpu
    from pygpu import gpuarray
    from pygpu.elemwise import compare, elemwise2
except ImportError:
    pass

_context_reg = {}

def reg_context(name, ctx):
    """
    Register a context by mapping it to a name.
    
    The context must be of type `GpuContext` and the name can be
    anything hashable (but is usually a string).  Only one context can
    be registered per name and the second registration for a given
    name will raise an error.

    :param name: an hashable object
    :param ctx: a GpuContext instance
    """
    if name in _context_reg:
        raise ValueError("context name %s is already defined" % (name,))
    if not isinstance(ctx, gpuarray.GpuContext):
        raise TypeError("context is not GpuContext")
    _context_reg[name] = ctx


def get_context(ref):
    """
    Retrive the context associated with a name.

    Return the context object mapped to `ref` that was previously
    register through :func:`reg_context`.  Trying to get the context
    for an unregistered `ref` will raise a exception.

    :param ref: an hashable object
    """
    if not ref in _context_reg:
        raise ValueError("context name %s not defined" %(ref,))
    return _context_reg[ref]


def _name_for_ctx(ctx):
    for k, v in _context_reg:
        if v == ctx:
            return k
        raise ValueError('context is not registered')

# This is a private method for use by the tests only
def _unreg_context(name):
    del _context_reg[name]


class GpuArrayType(Type):
    def __init__(self, dtype, broadcastable, name=None, context=None):
        # In case this was not provided and no global value is available
        self.dtype = str(dtype)
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.ndim = len(self.broadcastable)
        self.name = name
        self.context = context
        # Make sure it is a valid and registered context
        get_context(context)
        try:
            self.typecode = gpuarray.dtype_to_typecode(self.dtype)
        except gpuarray.GpuArrayException:
            raise TypeError("Unsupported dtype for %s: %s" %
                            (self.__class__.__name__, self.dtype))

    def clone(self, dtype=None, broadcastable=None):
        if dtype is None:
            dtype = self.dtype
        if broadcastable is None:
            broadcastable = self.broadcastable
        return self.__class__(dtype=dtype, broadcastable=broadcastable,
                              name=self.name, context=self.context)

    # This is a property to keep the type pickleable
    @property
    def real_context(self):
        return get_context(self.context)

    def __str__(self):
        return "GpuArrayType(%s, %s)" % (self.dtype, self.broadcastable)

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
            if self.real_context != data.context:
                raise TypeError("data context does not match type context")
            # fallthrough to ndim check
        elif allow_downcast:
            data = gpuarray.array(data, dtype=self.typecode, copy=False,
                                  ndmin=len(self.broadcastable),
                                  context=self.real_context)
        else:
            # allow ints/lists/numpy-like objects
            if not hasattr(data, 'dtype'):
                data = numpy.asarray(data)
            up_dtype = scalar.upcast(self.dtype, data.dtype)
            if up_dtype == self.dtype:
                data = gpuarray.array(data, dtype=self.dtype, copy=False,
                                      context=self.real_context)
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

    def filter_variable(self, other):
        if hasattr(other, '_as_GpuArrayVariable'):
            other = other._as_GpuArrayVariable(self.context)

        if not isinstance(other, Variable):
            other = self.Constant(type=self, data=other)

        if other.type == self:
            return other

        if not isinstance(other.type, tensor.TensorType):
            raise TypeError('Incompatible type', (self, other.type))
        if (other.type.dtype != self.dtype):
            raise TypeError('Incompatible dtype', (self.dtype,
                                                   other.type.dtype))
        if other.type.ndim != self.ndim:
            raise TypeError('Incompatible number of dimensions.'
                            ' Expected %d, got %d.' % (self.ndim, other.ndim))
        if other.type.broadcastable != self.broadcastable:
            raise TypeError('Incompatible broadcastable dimensions.'
                            ' Expected %s, got %s.' %
                            (str(other.type.broadcastable),
                             str(self.broadcastable)))

        from .basic_ops import GpuFromHost
        return GpuFromHost(self.context)(other)

    @staticmethod
    def values_eq(a, b):
        if a.shape != b.shape:
            return False
        if a.typecode != b.typecode:
            return False
        return numpy.asarray(compare(a, '==', b)).all()

    @staticmethod
    def values_eq_approx(a, b,
                         allow_remove_inf=False, allow_remove_nan=False,
                         rtol=None, atol=None):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if 'int' in str(a.dtype):
            return GpuArrayType.values_eq(a, b)
        else:
            if allow_remove_inf or allow_remove_nan:
                raise NotImplementedError(
                    "GpuArrayType.values_eq_approx() don't implemented the"
                    " allow_remove_inf and allow_remove_nan parameter")
            narrow = 'float32', 'complex64'
            if (str(a.dtype) in narrow) or (str(b.dtype) in narrow):
                atol_ = theano.tensor.basic.float32_atol
                rtol_ = theano.tensor.basic.float32_rtol
            else:
                atol_ = theano.tensor.basic.float64_atol
                rtol_ = theano.tensor.basic.float64_rtol
            if rtol is not None:
                rtol_ = rtol
            if atol is not None:
                atol_ = atol
            res = elemwise2(a, '', b, a, odtype=numpy.dtype('bool'),
                            op_tmpl="res[i] = (fabs(%%(a)s - %%(b)s) <"
                            "(%(atol_)s + %(rtol_)s * fabs(%%(b)s)))" %
                            locals())
            return numpy.asarray(res).all()

    def value_zeros(self, shape):
        return pygpu.gpuarray.zeros(shape, dtype=self.typecode,
                                    context=self.real_context)

    def make_variable(self, name=None):
        return self.Variable(self, name=name)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.typecode == other.typecode and
                self.broadcastable == other.broadcastable and
                self.context == other.context)

    def replace_variable(self, var):
        if (type(self) == type(var.type) and
            self.typecode == var.type.typecode and
            self.context == var.type.context and
            self.ndim == var.type.ndim and
            all(sb == ob or ob for sb, ob in zip(self.broadcastable,
                                                 var.type.broadcastable))):
            return theano.tensor.patternbroadcast(var, self.broadcastable)

    def __hash__(self):
        return hash((type(self), self.typecode, self.broadcastable,
                     self.context))

    def __str__(self):
        return "GpuArray<%s,%s>" % (self.context, self.dtype)

    def dtype_specs(self):
        """Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.
        """
        # TODO: add more type correspondances for e.g. int32, int64, float32,
        # complex64, etc.
        try:
            return {
                'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                'uint8': (int, 'npy_uint8', 'NPY_UINT8'),
                'int8': (int, 'npy_int8', 'NPY_INT8'),
                'uint16': (int, 'npy_uint16', 'NPY_UINT16'),
                'int16': (int, 'npy_int16', 'NPY_INT16'),
                'uint32': (int, 'npy_uint32', 'NPY_UINT32'),
                'int32': (int, 'npy_int32', 'NPY_INT32'),
                'uint64': (int, 'npy_uint64', 'NPY_UINT64'),
                'int64': (int, 'npy_int64', 'NPY_INT64'),
                'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')
                }[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" %
                            (self.__class__.__name__, self.dtype))

    def get_shape_info(self, obj):
        return obj.shape

    def get_size(self, shape_info):
        if shape_info:
            return numpy.prod(shape_info) * numpy.dtype(self.dtype).itemsize
        else:
            return numpy.dtype(self.dtype).itemsize

    def c_declare(self, name, sub, check_input=True):
        return """
        PyGpuArrayObject *%(name)s;
        """ % locals()

    def c_init(self, name, sub):
        return "%s = NULL;" % (name,)

    def c_extract(self, name, sub, check_input=True):
        # TODO I don't check broadcast stuff for now.
        return """
        %(name)s = NULL;
        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "expected a GpuArray, not None");
            %(fail)s
        }
        /* First check if we are the base type exactly (the most common case),
           then do the full subclass check if needed. */
        if (py_%(name)s->ob_type != &PyGpuArrayType &&
            !PyObject_TypeCheck(py_%(name)s, &PyGpuArrayType)) {
            PyErr_SetString(PyExc_ValueError, "expected a GpuArray");
            %(fail)s
        }
        %(name)s = (PyGpuArrayObject *)py_%(name)s;
        Py_INCREF(%(name)s);
        """ % {'name': name, 'fail': sub['fail']}

    def c_cleanup(self, name, sub):
        return "Py_XDECREF(%(name)s); %(name)s = NULL;" % {'name': name}

    def c_sync(self, name, sub):
        return """
        if (!%(name)s) {
            Py_XDECREF(py_%(name)s);
            Py_INCREF(Py_None);
            py_%(name)s = Py_None;
        } else if ((void *)py_%(name)s != (void *)%(name)s) {
            Py_XDECREF(py_%(name)s);
            py_%(name)s = (PyObject *)%(name)s;
            Py_INCREF(py_%(name)s);
        }
        """ % {'name': name}

    def c_init_code(self):
        # We don't actually need the numpy API except in
        # HostFromGpu and GpuFromHost and those case will be covered
        # by the TensorType parameter
        return ['import_pygpu__gpuarray();']

    def c_headers(self):
        # We need arrayobject for the PyArrayDescr struct def
        # (even if we just use a pointer to it in a function def)
        return ['<gpuarray/array.h>', '<gpuarray/kernel.h>', '<gpuarray/error.h>',
                '<gpuarray/buffer_blas.h>', '<numpy/arrayobject.h>',
                '<gpuarray_api.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), numpy.get_include()]

    def c_libraries(self):
        return ['gpuarray']

    def c_code_cache_version(self):
        ver = pygpu.gpuarray.api_version()
        # we only use the major version since the minor revision are
        # API-compatible.
        return (1, ver[0])

    def may_share_memory(self, a, b):
        return (isinstance(a, gpuarray.GpuArray) and
                isinstance(b, gpuarray.GpuArray) and
                gpuarray.may_share_memory(a, b))


class _operators(_tensor_py_operators):
    def _as_TensorVariable(self):
        from .basic_ops import host_from_gpu
        return host_from_gpu(self)

    def _as_GpuArrayVariable(self, ctx):
        from .basic_ops import AnyContext, GpuFromGpu
        if ctx is AnyContext or self.type.context == ctx:
            return self
        else:
            return GpuFromGpu(ctx)(self)


class GpuArrayVariable(_operators, Variable):
    pass


GpuArrayType.Variable = GpuArrayVariable


class GpuArraySignature(tensor.TensorConstantSignature):
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
        self.container.value = pygpu.gpuarray.array(value, copy=(not borrow),
                                                    context=self.type.real_context)

    def __getitem__(self, *args):
        return _operators.__getitem__(self, *args)


GpuArrayType.SharedVariable = GpuArraySharedVariable


def gpuarray_shared_constructor(value, name=None, strict=False,
                                allow_downcast=None, borrow=False,
                                broadcastable=None, context=None):
    """SharedVariable constructor for GpuArrayType"""
    if not isinstance(value, (numpy.ndarray, pygpu.gpuarray.GpuArray)):
        raise TypeError('ndarray or GpuArray required')

    if context is None and isinstance(value, gpuarray.GpuArray):
        context = _name_for_ctx(value.context)

    if broadcastable is None:
        broadcastable = (False,) * value.ndim
    type = GpuArrayType(value.dtype, broadcastable, context=context)
    deviceval = pygpu.gpuarray.array(value, copy=(not borrow),
                                     context=type.real_context)
    return GpuArraySharedVariable(type=type, value=deviceval, name=name,
                                  strict=strict)

theano.compile.register_view_op_c_code(GpuArrayType, """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
""", version=(0,))

# Register GpuArrayType C code for Shape Op.
theano.compile.register_shape_c_code(
    GpuArrayType,
    """
    npy_intp shape[] = {%(iname)s->ga.nd};
    if(%(oname)s == NULL || (PyArray_DIMS(%(oname)s)[0] != shape[0]))
    {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INT64);
    }
    for(int i=0;i<shape[0];i++)
    {
        ((npy_int64*)PyArray_GETPTR1(%(oname)s, i))[0] = %(iname)s->ga.dimensions[i];
    }
    """,
    version=1)

theano.compile.register_shape_i_c_code(
    GpuArrayType,
    """
    if(!%(oname)s)
        %(oname)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0] =
                              %(iname)s->ga.dimensions[%(i)s];
    """,
    """
    if (%(i)s>=%(iname)s->ga.nd){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        %(fail)s
    }
    """,
    version=(1,))

theano.compile.register_deep_copy_op_c_code(GpuArrayType, """
    Py_XDECREF(%(oname)s);
    %(oname)s = pygpu_copy(%(iname)s, GA_ANY_ORDER);
    if (!%(oname)s) { %(fail)s }
""", version=(5,))

theano.compile.register_rebroadcast_c_code(
    GpuArrayType,
    """
    if(%(iname)s->ga.dimensions[%(axis)s] != 1){
        PyErr_Format(PyExc_ValueError,
            "Dimension %(axis)s in Rebroadcast's input was"
            " supposed to be 1 (got %%d instead)",
            %(iname)s->ga.dimensions[%(axis)s]);
        %(fail)s
    }
    """,
    version=1)

theano.compile.register_specify_shape_c_code(
    GpuArrayType,
    """
        if (PyGpuArray_NDIM(%(iname)s) != PyArray_DIMS(%(shape)s)[0]) {
            PyErr_Format(PyExc_AssertionError,
                         "SpecifyShape: vector of shape has %%d elements,"
                         " but the input has %%d dimensions.",
                         PyGpuArray_NDIM(%(iname)s),
                         PyArray_DIMS(%(shape)s)[0]);
            %(fail)s;
        }
        for(int i = 0; i < PyGpuArray_NDIM(%(iname)s); i++){
            dtype_%(shape)s shp = ((dtype_%(shape)s*)PyArray_GETPTR1(%(shape)s,
                                                                     i))[0];
            if (PyGpuArray_DIMS(%(iname)s)[i] != shp) {
                PyErr_Format(PyExc_AssertionError,
                             "SpecifyShape: dim %%d of input has shape %%d,"
                             " expected %%d.",
                             i, PyGpuArray_DIMS(%(iname)s)[i],
                             shp);
                %(fail)s;
            }
        }
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_XINCREF(%(oname)s);
    """,
    version=1,
    c_support_code_apply='#include <numpy_compat.h>')


class GpuContextType(Type):
    def filter(self, data, strict=False, allow_downcast=None):
        # up until here context are still just names
        data = get_context(data)
        if not isinstance(data, gpuarray.GpuContext):
            raise TypeError('context is not a GpuContext')
        return data

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    @staticmethod
    def values_eq(a, b):
        return a == b

    def c_declare(self, name, sub, check_input=True):
        return "PyGpuContextObject *%s;" % (name,)

    def c_init(self, name, sub):
        return "%s = NULL;" % (name,)

    def c_extract(self, name, sub, check_input=True):
        if check_input:
            res = """
if (!PyObject_TypeCheck(py_%(name)s, &PyGpuContextType)) {
  PyErr_SetString(PyExc_TypeError, "expected a GpuContext");
  %(fail)s
}
""" % dict(name=name, fail=sub['fail'])
        else:
            res = ""
        return res + """
%(name)s = (PyGpuContextObject *)py_%(name)s;
Py_INCREF(%(name)s);
""" % dict(name=name)

    def c_cleanup(self, name, sub):
        return "Py_XDECREF(%(name)s); %(name)s = NULL;" % dict(name=name)

    # c_sync is intentionally not declared to prevent normal usage

    def c_init_code(self):
        return ['import_pygpu__gpuarray();']

    def c_headers(self):
        return ['<gpuarray_api.h>']

    def c_header_dirs(self):
        return [pygpu.get_include()]

    def c_code_cache_version(self):
        ver = pygpu.gpuarray.api_version()
        return (0, ver[0])

    # Variable, Contstant, ... not declared

gpu_context_type = GpuContextType()
