
import copy
import utils
from utils import AbstractFunctionError, object2
from graph import Result
import traceback


########
# Type #
########

class Type(object2):

    def filter(self, data, strict = False):
        """
        Return data or an appropriately wrapped data. Raise an
        exception if the data is not of an acceptable type.

        If strict is True, the data returned must be the same
        as the data passed as an argument. If it is False, filter
        may cast it to the appropriate type.
        """
        raise AbstractFunctionError()
    
    def make_result(self, name = None):
        r = Result(self, name = name)
        return r
    
    def __call__(self, name = None):
        r = self.make_result(name)
        r.tag.trace = traceback.extract_stack()[:-1]
        return r
        
    def c_is_simple(self):
        """
        A hint to tell the compiler that this type is a builtin C
        type or a small struct and that its memory footprint is
        negligible.
        """
        return False

    def c_literal(self, data):
        raise AbstractFunctionError()
    
    def c_declare(self, name, sub):
        """
        Declares variables that will be instantiated by L{c_extract}.
        """
        raise AbstractFunctionError()

    def c_init(self, name, sub):
        raise AbstractFunctionError()

    def c_extract(self, name, sub):
        """
        The code returned from this function must be templated using
        "%(name)s", representing the name that the caller wants to
        call this L{Result}. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".

        @todo: Point out that template filling (via sub) is now performed
        by this function. --jpt
        """
        raise AbstractFunctionError()
    
    def c_cleanup(self, name, sub):
        """
        This returns C code that should deallocate whatever
        L{c_extract} allocated or decrease the reference counts. Do
        not decrease py_%(name)s's reference count.
        """
        raise AbstractFunctionError()

    def c_sync(self, name, sub):
        """
        The code returned from this function must be templated using "%(name)s",
        representing the name that the caller wants to call this Result.
        The returned code may set "py_%(name)s" to a PyObject* and that PyObject*
        will be accessible from Python via result.data. Do not forget to adjust
        reference counts if "py_%(name)s" is changed from its original value.
        """
        raise AbstractFunctionError()

    def c_compile_args(self):
        """
        Return a list of compile args recommended to manipulate this L{Result}.
        """
        raise AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this L{Result}.
        """
        raise AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this L{Result}.

        For example: return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH.  No option is provided for an Op to provide an
        extra library directory because this would change the linking path for
        other Ops in a potentially disasterous way.
        """
        raise AbstractFunctionError()

    def c_support_code(self):
        """
        Return utility code for use by this L{Result} or L{Op}s manipulating this
        L{Result}.
        """
        raise AbstractFunctionError()


class SingletonType(Type):
    __instance = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = Type.__new__(cls)
        return cls.__instance
    def __str__(self):
        return self.__class__.__name__


class Generic(SingletonType):
    """
    Represents a generic Python object.
    """
    
    def filter(self, data, strict = False):
        return data

    def c_declare(self, name, sub):
        return """
        PyObject* %(name)s;
        """ % locals()

    def c_extract(self, name, sub):
        return """
        Py_XINCREF(py_%(name)s);
        %(name)s = py_%(name)s;
        """ % locals()
    
    def c_cleanup(self, name, sub):
        return """
        Py_XDECREF(%(name)s);
        """ % locals()

    def c_sync(self, name, sub):
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = %(name)s;
        Py_XINCREF(py_%(name)s);
        """ % locals()

generic = Generic()


class PropertiedType(Type):

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

