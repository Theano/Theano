from __future__ import absolute_import, print_function, division
import os
import numpy
from six import StringIO

import theano
from theano import Type, Variable, config, tensor
from theano.tensor.var import _tensor_py_operators

try:
    import mkl_ndarray
    import mkl_ndarray.mkl_ndarray as mkl

    try:
        mkl_ndarray.__file__
    except AttributeError:
        from theano.gof.cmodule import get_lib_extension
        mkl_ndarray.__file__ = os.path.join(mkl_ndarray.__path__._path[0], 'mkl_ndarray.' + get_lib_extension())
except ImportError:
    mkl = None


class _operators(_tensor_py_operators):
    dtype = property(lambda s: s.type.dtype)
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)


class MKLNdarrayVariable(_operators, Variable):
    """
    See Also
    --------
    Variable

    """
    pass


class MKLNdarrayType(Type):
    """
    The type that represents an array on CPU with MKL supoorts.

    The `dtype` indicates what scalar data type the elements of
    variables of this type will be.

    The `broadcastable` indicates whether each dimension is broadcastable
    or not (to be broadcastable a dimension must always be of length 1)

    The `context_name` is the name of the context. For similar interface
    with the new back-end for GPU.

    `name` for the type that will be used in printouts.

    """
    Variable = MKLNdarrayVariable
    Constant = None
    SharedVariable = None
    broadcastable = None

    ndim = property(lambda self: len(self.broadcastable), doc='number of dimensions')

    def __init__(self, dtype, broadcastable, name=None):
        self.dtype = str(dtype)
        if self.dtype == 'floatX':
            self.dtype = config.floatX

        if self.dtype not in ('float32', 'float64'):
            raise TypeError('%s only supports float32/float64 for now.'
                            'Tried using dtype %s for variable %s' %
                            (self.__class__.__name__, dtype, name))

        self.typenum = 11 if dtype is 'float32' else 12
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.name = name
        self.dtype_specs()

    def clone(self, dtype=None, broadcastable=None):
        if broadcastable is None:
            broadcastable = self.broadcastable

        if dtype is None:
            dtype = self.dtype

        return self.__class__(dtype=dtype, broadcastable=broadcastable, name=self.name)

    def filter(self, data, strict=False, allow_downcast=None):
        """
        Since it is not support currently to cast layout from FP32 to FP64, we
        don't do up or down cast in this function for now. It means that it is
        always strict for all input data.

        """
        # TODO: we will add up and down cast code here when the low level APIs
        # support that.
        if not isinstance(data, mkl.MKLNdarray):
            raise TypeError('%s expected a MKLNdarray object.' % self,
                            data, type(data))

        if data.dtype != self.dtype:
            raise TypeError(('%s expected a MKLNdarray object with'
                            'dtype=%s (got %s).') % (self, self.dtype, data.dtype))

        if data.ndim != self.ndim:
            raise TypeError('Wrong number of dimensions: expected %s,'
                            'got %s with shape %s.' % (self.ndim, data.ndim, data.shape))

        shp = data.shape
        for i, b in enumerate(self.broadcastable):
            if b and shp[i] != 1:
                raise TypeError('None-unit value on shape on a broadcastable'
                                ' dimension.', shp, self.broadcastable)

        return data

    def filter_variable(self, other, allow_convert=True):
        raise NotImplementedError('MKLNdarrayType filter_variable')

    @staticmethod
    def values_eq(a, b):
        return tensor.TensorType.values_eq(numpy.asarray(a), numpy.asarray(b))

    @staticmethod
    def values_eq_approx(a, b, allow_remove_inf=False, allow_remove_nan=False,
                         rtol=None, atol=None):
        return tensor.TensorType.values_eq_approx(
            numpy.asarray(a),
            numpy.asarray(b),
            allow_remove_inf=allow_remove_inf,
            allow_remove_nan=allow_remove_nan,
            rtol=rtol, atol=atol
        )

    def dtype_specs(self):
        """
        Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.

        """
        try:
            return {'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                    'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                    'uint8': (int, 'npy_uint8', 'NPY_UINT8'),
                    'int8': (int, 'npy_int8', 'NPY_INT8'),
                    'uint16': (int, 'npy_uint16', 'NPY_UINT16'),
                    'int16': (int, 'npy_int16', 'NPY_INT16'),
                    'uint32': (int, 'npy_uint32', 'NPY_UINT32'),
                    'int32': (int, 'npy_int32', 'NPY_INT32'),
                    'uint64': (int, 'npy_uint64', 'NPY_UINT64'),
                    'int64': (int, 'npy_int64', 'NPY_INT64'),
                    'complex128': (complex, 'theano_complex128',
                                   'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64',
                                  'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" %
                            (self.__class__.__name__, self.dtype))

    def __eq__(self, other):
        """
        Compare True if other is the same kind of MKLNdarrayType.

        """
        return (type(self) == type(other) and
                self.dtype == other.dtype and
                self.broadcastable == other.broadcastable)

    def __hash__(self):
        """
        Hash equal for same kinds of MKLNdarrayType.

        """
        return hash((type(self), self.dtype, self.broadcastable))

    def make_variable(self, name=None):
        """
        Return a 'MKLNdarrayVariable' with this type.

        Parameters
        ----------
        name: str
            A pretty name to identify this 'Variable' when printing and
            debugging.

        """
        return self.Variable(self, name=name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            named_broadcastable = {tuple(): 'scalar',
                                   (False,): 'vector',
                                   (False, True): 'col',
                                   (True, False): 'row',
                                   (False, False): 'matix'}
            if b in named_broadcastable:
                bcast = named_broadcastable[b]
            elif any(b):
                bcast = str(b)
            else:
                bcast = '%iD' % len(b)
            return 'MKLNdarrayType<%s>(%s, %s)' % (self.context_name, self.dtype, bcast)

    def __repr__(self):
        return str(self)

    def c_declare(self, name, sub, check_input=True):
        """
        Declare variables which will be used in C code.
        """
        return """
        MKLNdarray * %(name)s;
        """ % locals()

    def c_init(self, name, sub):
        """
        Init variables declared in c_declare.
        """
        return """
        %(name)s = NULL;
        """ % locals()

    def c_extract(self, name, sub, check_input=True):
        sio = StringIO()
        fail = sub['fail']
        nd = self.ndim
        print("""
        assert (py_%(name)s->ob_refcnt >= 2);

        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "expected a MKLNdarray, not None");
            %(fail)s
        }

        if (MKLNdarray_Check(py_%(name)s)) {
            %(name)s = (MKLNdarray*)py_%(name)s;
            assert (%(name)s);
            Py_INCREF(%(name)s);
        }
        """ % locals(), file=sio)

        return sio.getvalue()

    def c_cleanup(self, name, sub):
        """
        Add cleanup code here.
        """
        return """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
            %(name)s = NULL;
        }
        """ % locals()

    def c_sync(self, name, sub):
        """
        When the computations are done, transfer the variables from the C
        structure we put them in to the destination Python object. This will
        only be called for the outputs.
        """
        return """
        if (NULL == %(name)s) {
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
            Py_INCREF(py_%(name)s);
        } else {
            if (py_%(name)s != (PyObject*)%(name)s) {
                Py_XDECREF(py_%(name)s);
                py_%(name)s = (PyObject*)%(name)s;
                Py_INCREF(py_%(name)s);
            }
            assert(py_%(name)s->ob_refcnt);
        }
        """ % locals()

    def c_headers(self):
        return ['mkl_ndarray.h']

    def c_header_dirs(self):
        ret = [os.path.dirname(mkl_ndarray.__file__), os.path.dirname(__file__)]
        return ret

    def c_lib_dirs(self):
        ret = [os.path.dirname(mkl_ndarray.__file__)]
        return ret

    def c_libraries(self):
        return ['mkl_ndarray']

    def c_code_cache_version(self):
        return (1, 0, 1)

    def c_compile_args(self):
        return ['-Wl,-R' + os.path.dirname(mkl_ndarray.__file__)]

    def get_shape_info(self, obj):
        return obj.shape

    def get_size(self, shape_info):
        """
        If shape_info is given, return memory size of all data,
        else size of single element is returned.
        """
        if shape_info:
            return numpy.prod(shape_info) * numpy.dtype(self.dtype).itemsize
        else:
            return numpy.dtype(self.dtype).itemsize


# expandable_types: can add an extra dimension and for which Scan can deal with.
# TODO: further check with MKLNdarrayType to support that.
# theano.compile.ops.expandable_types += (MKLNdarrayType,)

# Register C code for ViewOp on CudaNdarrayType
theano.compile.register_view_op_c_code(
    MKLNdarrayType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    version=(1,))

theano.compile.register_shape_c_code(
    MKLNdarrayType,
    """
    npy_intp shape[] = {MKLNdarray_NDIM(%(iname)s)};
    if (%(oname)s == NULL || (PyArray_DIMS(%(oname)s)[0] != shape[0])) {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*)PyArray_SimpleNew(1, shape, NPY_INT64);
    }

    for (int i=0; i<shape[0]; i++) {
        ((npy_int64*)PyArray_GETPTR1(%(oname)s, i))[0] = MKLNdarray_DIMS(%(iname)s)[i];
    }
    """,
    version=(1,))

theano.compile.register_shape_i_c_code(
    MKLNdarrayType,
    """
    if (!%(oname)s)
        $(oname)s = (PyArrayObject*)PyArray_ZEROS(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0] = MKLNdarray_DIMS(%(iname)s)[%(i)s]
    """,
    """
    if (%(i)s >= MKLNdarray_NDIM(%(iname)s)) {
        PyErr_SetString(PyExc_TypeError, "Number of dimensions lower than expected");
        %(fail)s;
    }
    """,
    version=(1,))

theano.compile.register_deep_copy_op_c_code(
    MKLNdarrayType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = (MKLNdarray*)MKLNdarray_Copy(%(iname)s);

    if (!%(oname)s) {
        PyErr_SetString(PyExc_RuntimeError,
                        "DeepCopyOp: copy failed");
        %(fail)s;
    }
    """,
    version=(1,))


theano.compile.register_specify_shape_c_code(
    MKLNdarrayType,
    """
    if (MKLNdarray_NDIM(%(iname)s) != PyArray_DIMS(%(shape)s)[0]) {
        PyErr_Format(PyExc_AssertionError,
                     "SpecifyShape: vector of shape has %%d elements,"
                     " but the input has %%d dimensions.",
                     PyArray_DIMS(%(shape)s)[0],
                     MKLNdarray_NDIM(%(iname)s));
        %(fail)s;
    }

    for (int i = 0; i < MKLNdarray_NDIM(%(iname)s); i++) {
        dtype_%(shape)s shp = ((dtype_%(shape)s*)PyArray_GETPTR1(%(shape)s,
                                                                 i))[0];
        if (MKLNdarray_DIMS(%(iname)s)[i] != shp) {
            PyErr_Format(PyExc_AssertionError,
                         "SpecifyShape: dim %%d of input has shape %%d,"
                         " expected %%d.",
                         i, MKLNdarray_DIMS(%(iname)s)[i],
                         shp);
            %(fail)s;
        }
    }

    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    version=(1,))
