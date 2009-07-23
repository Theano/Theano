import sys, os
import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor

import cuda_ndarray

from .type_support import filter as type_support_filter

from .nvcc_compiler import nvcc_module_compile_str

class CudaNdarrayType(Type):

    typenum = 11 # Until hardware improves, this class deals with floats.

    dtype = 'float32'

    Variable = None
    """ This will be set to the Variable type corresponding to this class.

    That variable type is `CudaNdarrayVariable` defined in the ``var.py`` file beside this one.

    :note: 
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding ``Variable = CudaNdarrayVariable``.
    """

    Constant = None
    """ This will be set to `CudaNdarrayConstant` defined in ``var.py``

    :note: 
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding this class. 
    """

    SharedVariable = None
    """ This will be set to `CudaNdarraySharedVariable` defined in ``var.py``

    :note: 
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding this class. 
    """

    def __init__(self, broadcastable, name=None):
        self.broadcastable = tuple(broadcastable)
        self.name = name
        self.dtype_specs() # error checking is done there

    def filter(self, data, strict=False):
        return type_support_filter(data, self.broadcastable, strict)

    @staticmethod
    def values_eq_approx(a, b):
        return tensor.TensorType.values_eq_approx(numpy.asarray(a), numpy.asarray(b))

    def dtype_specs(self):
        """Return a tuple (python type, c type, numpy typenum) that corresponds to
        self.dtype.
        
        This function is used internally as part of C code generation.
        """
        #TODO: add more type correspondances for e.g. int32, int64, float32,
        #complex64, etc.
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
                    'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    def __eq__(self, other):
        """Compare True iff other is the same kind of CudaNdarrayType"""
        return type(self) == type(other) and other.broadcastable == self.broadcastable

    def __hash__(self):
        """Hash equal for same kinds of CudaNdarrayType"""
        return hash(type(self)) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc = "number of dimensions")
    """Number of dimensions

    This read-only property is the preferred way to get the number of dimensions
    of a `CudaNdarrayType`.
    
    """

    def make_variable(self, name = None):
        """Return a `TensorVariable` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Variable` when printing and debugging

        """
        return self.Variable(self, name = name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            #bcast = str(self.broadcastable)
            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matrix'}.get(b, "%iD" % len(b) if not any(b) else str(b))
            return "CudaNdarrayType(%s, %s)" % (str(self.dtype), bcast)

    def __repr__(self):
        return str(self)
        #"CudaNdarrayType{%s, %s}" % (str(self.dtype), str(self.broadcastable))

    def c_declare(self, name, sub):
        ndim = self.ndim
        c_typename = self.dtype_specs()[1]
        return """ CudaNdarray * cnda_%(name)s;""" %locals()

    def c_init(self, name, sub):
        return "cnda_%(name)s = NULL;" % locals()

    def c_extract(self, name, sub):
        return """
        if (CudaNdarray_Check(py_%(name)s))
        {
            cnda_%(name)s = (CudaNdarray*)py_%(name)s;
            Py_INCREF(py_%(name)s);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
            cnda_%(name)s = NULL;
            %(fail)s;
        }
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_cleanup(self, name, sub):
        return """
        //std::cerr << "cleanup " << py_%(name)s << "\\n";
        Py_XDECREF(py_%(name)s);
        """ % locals()

    def c_sync(self, name, sub):
        """Override `CLinkerOp.c_sync` """
        return """
        //std::cerr << "sync\\n";
        if (NULL == cnda_%(name)s) {  
            // failure: sync None to storage
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
            Py_INCREF(py_%(name)s);
        }
        else
        {
            if (py_%(name)s != (PyObject*)cnda_%(name)s)
            {
                Py_XDECREF(py_%(name)s);
                py_%(name)s = (PyObject*)cnda_%(name)s;
                Py_INCREF(py_%(name)s);
            }
            assert(py_%(name)s->ob_refcnt);
        }
        """ % locals()

    def c_headers(self):
        """Override `CLinkerOp.c_headers` """
        return ['cuda_ndarray.cuh']

    def c_header_dirs(self):
        """Override `CLinkerOp.c_headers` """
        return [os.path.dirname(cuda_ndarray.__file__),
                os.path.join(os.getenv("CUDA_ROOT"),'include')]

    def c_lib_dirs(self):
        return [os.path.dirname(cuda_ndarray.__file__),
                os.path.join(os.getenv("CUDA_ROOT"),'lib')]

    def c_libraries(self):
        return ['cuda_ndarray', 'cudart']

    def c_support_code(cls):
        return ""

    def c_code_cache_version(self):
        return () #do not cache this stuff until it matures


    def c_compiler(self): return nvcc_module_compile_str







