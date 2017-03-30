from __future__ import absolute_import, print_function, division
import os
import numpy
import theano
from theano import Type, Variable
from theano import tensor
from six import StringIO

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


class _operators(tensor.basic._tensor_py_operators):
    dtype = property(lambda s: s.type.dtype)
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)


class MKLNdarrayVariable(_operators, Variable):
    pass


class MKLNdarrayType(Type):
    context_num = 'mkl'
    Variable = MKLNdarrayVariable
    Constant = None
    SharedVariable = None
    ndim = None
    broadcastable = None

    def __init__(self, broadcastable, name=None, dtype=None):
        if dtype not in ('float32', 'float64'):
            raise TypeError('%s only supports dtype float32/float64 for now. \
                            Tried sing dtype float32 for variable %s' % (self.__class__.__name__, name))
        self.dtype = dtype
        self.typenum = 11 if dtype is 'float32' else 12
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.name = name
        self.dtype_specs()  # error checking is done there

    def clone(self, dtype=None, broadcastable=None):
        if broadcastable is None:
            broadcastable = self.broadcastable
        return self.__class__(broadcastable, name=self.name, dtype=dtype)

    def filter(self, data, strict=False, allow_downcast=None):
        # return self.filter_inplace(data, None, strict=strict, allow_downcast=allow_downcast)
        return data

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a Variable into a MKLNdarrayType, if compatible.

        This Variable should either already be a MKLNdarrayType, or be
        a TensorType. It has to have the right number of dimensions,
        broadcastable pattern, and dtype.

        """
        raise NotImplementedError('MKLNdarrayType filter_variable')

    @staticmethod
    def values_eq(a, b):
        raise NotImplementedError('MKLNdarray values_eq')

    def dtype_specs(self):
        """
        Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.

        """
        # TODO: add more type correspondances for e.g. int32, int64, float32,
        # complex64, etc.
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
                other.broadcastable == self.broadcastable)

    def __hash__(self):
        """
        Hash equal for same kinds of MKLNdarrayType.

        """
        return hash(type(self)) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc='number of dimensions')

    def make_variable(self, name=None):
        """
        Return a 'TensorVariable' of this type.

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
            if not numpy.any(b):
                s = '%iD' % len(b)
            else:
                s = str(b)

            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matix'}.get(b, s)
            return 'MKLNdarrayType(%s, %s)' % (str(self.dtype), bcast)

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

    def c_extract(self, name, sub, check_input=True,
                  check_broadcast=True):
        sio = StringIO()
        fail = sub['fail']
        nd = self.ndim
        print("""
        assert (py_%(name)s->ob_refcnt >= 2);

        if (MKLNdarray_Check(py_%(name)s))
        {
            %(name)s = (MKLNdarray*)py_%(name)s;
            assert (%(name)s);
            Py_INCREF(py_%(name)s);
        }
        """ % locals(), file=sio)

        return sio.getvalue()

    def c_cleanup(self, name, sub):
        """
        Add cleanup code here.
        """
        return """
        if (%(name)s)
        {
            Py_XDECREF(%(name)s);
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
            // failure: sync None to storage
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
            Py_INCREF(py_%(name)s);
        }
        else
        {
            if (py_%(name)s != (PyObject*)%(name)s)
            {
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

    def c_support_code(self):
        return ''

    def c_code_cache_version(self):
        return (1, 0, 0)

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


theano.compile.ops.expandable_types += (MKLNdarrayType,)

# Register C code for ViewOp on CudaNdarrayType
theano.compile.register_view_op_c_code(
    MKLNdarrayType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    version=1)
