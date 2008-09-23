"""Defines the `Type` class."""

__docformat__ = "restructuredtext en"

import copy
import utils
from utils import AbstractFunctionError, object2
from graph import Result
import traceback


########
# Type #
########

class CLinkerType(object):
    """Interface specification for Types that can be arguments to a `CLinkerOp`.

    A CLinkerType instance is mainly reponsible  for providing the C code that
    interfaces python objects with a C `CLinkerOp` implementation.

    See WRITEME for a general overview of code generation by `CLinker`.

    """

    def c_is_simple(self):
        """Optional: Return True for small or builtin C types.

        A hint to tell the compiler that this type is a builtin C type or a
        small struct and that its memory footprint is negligible.  Simple
        objects may be passed on the stack.

        """
        return False

    def c_literal(self, data):
        """Optional: WRITEME

        :Parameters:
         - `data`: WRITEME
            WRITEME

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method
        
        """
        raise AbstractFunctionError()
    
    def c_declare(self, name, sub):
        """Required: Return c code to declare variables that will be
        instantiated by `c_extract`.

        Example: WRITEME

        :Parameters:
         - `name`: WRITEME
            WRITEME
         - `sub`: WRITEME
            WRITEME

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method
        """
        raise AbstractFunctionError()

    def c_extract(self, name, sub):
        """Required: Return c code to extract a PyObject * instance.

        The code returned from this function must be templated using
        "%(name)s", representing the name that the caller wants to
        call this `Result`. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".

        @todo: Point out that template filling (via sub) is now performed
        by this function. --jpt

        WRITEME

        :Parameters:
         - `name`: WRITEME
            WRITEME
         - `sub`: WRITEME
            WRITEME

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()
    
    def c_cleanup(self, name, sub):
        """Optional: Return c code to clean up after `c_extract`.

        This returns C code that should deallocate whatever `c_extract`
        allocated or decrease the reference counts. Do not decrease
        py_%(name)s's reference count.

        WRITEME

        :Parameters:
         - `name`: WRITEME
            WRITEME
         - `sub`: WRITEME
            WRITEME

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

    def c_sync(self, name, sub):
        """Required: Return c code to pack C types back into a PyObject.

        The code returned from this function must be templated using "%(name)s",
        representing the name that the caller wants to call this Result.  The
        returned code may set "py_%(name)s" to a PyObject* and that PyObject*
        will be accessible from Python via result.data. Do not forget to adjust
        reference counts if "py_%(name)s" is changed from its original value.

        :Parameters:
         - `name`: WRITEME
            WRITEME
         - `sub`: WRITEME
            WRITEME

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

    def c_compile_args(self):
        """Optional: Return a list of compile args recommended to compile the
        code returned by other methods in this class.

        WRITEME: example of formatting for -I, -L, -f args.

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

    def c_headers(self):
        """Optional: Return a list of header files required by code returned by
        this class.

        WRITEME: example of local file, standard file.

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

    def c_libraries(self):
        """Optional: Return a list of libraries required by code returned by
        this class.

        For example: return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH.  No option is provided for an Op to provide an
        extra library directory because this would change the linking path for
        other Ops in a potentially disasterous way.

        QUESTION: What about via the c_compile_args? a -L option is allowed no?

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

    def c_support_code(self):
        """Optional: Return utility code for use by a `Result` or `Op` to be
        included at global scope prior to the rest of the code for this class.

        QUESTION: How many times will this support code be emitted for a graph
        with many instances of the same type?

        :Exceptions:
         - `AbstractFunctionError`: Subclass does not implement this method

        """
        raise AbstractFunctionError()

class PureType(object):
    """Interface specification for result type instances.

    A Type instance is mainly reponsible for two things:

    - creating `Result` instances (conventionally, `__call__` does this), and

    - filtering a value assigned to a `Result` so that the value conforms to restrictions
      imposed by the type (also known as casting, this is done by `filter`),

    """

    def filter(self, data, strict = False):
        """Required: Return data or an appropriately wrapped/converted data.
        
        Subclass implementation should raise a TypeError exception if the data is not of an
        acceptable type.

        If strict is True, the data returned must be the same as the data passed as an
        argument. If it is False, filter may cast it to an appropriate type.

        :Exceptions:
         - `AbstractFunctionError`: subclass doesn't implement this function.

        """
        raise AbstractFunctionError()
    
    def make_result(self, name = None):
        """Return a new `Result` instance of Type `self`.

        :Parameters:
         - `name`: None or str
            A pretty string for printing and debugging.

        """
        r = Result(self, name = name)
        return r
    
    def __call__(self, name = None):
        """Return a new `Result` instance of Type `self`.

        :Parameters:
         - `name`: None or str
            A pretty string for printing and debugging.

        """
        r = self.make_result(name)
        r.tag.trace = traceback.extract_stack()[:-1]
        return r


_nothing = """
       """


class Type(object2, PureType, CLinkerType):
    """Convenience wrapper combining `PureType` and `CLinkerType`.

    Theano comes with several subclasses of such as:

    - `Generic`: for any python type

    - `Tensor`: for numpy.ndarray

    - `Sparse`: for scipy.sparse

    But you are encouraged to write your own, as described in WRITEME.

    The following following code illustrates the use of a Type instance, here tensor.fvector:

    .. python::
        # Declare a symbolic floating-point vector using __call__
        b = tensor.fvector()

        # Create a second Result with the same Type instance
        c = tensor.fvector()

    Whenever you create a symbolic variable in theano (technically, `Result`) it will contain a
    reference to a Type instance.  That reference is typically constant during the lifetime of
    the Result.  Many variables can refer to a single Type instance, as do b and c above.  The
    Type instance defines the kind of value which might end up in that variable when executing
    a `Function`.  In this sense, theano is like a strongly-typed language because the types
    are included in the graph before the values.  In our example above, b is a Result which is
    guaranteed to corresond to a numpy.ndarray of rank 1 when we try to do some computations
    with it.

    Many `Op` instances will raise an exception if they are applied to inputs with incorrect
    types.  Type references are also useful to do type-checking in pattern-based optimizations.

    """

class SingletonType(Type):
    """WRITEME"""
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

    This class implements the `PureType` and `CLinkerType` interfaces for generic PyObject
    instances. 

    EXAMPLE of what this means, or when you would use this type.

    WRITEME
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
    """WRITEME"""

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

