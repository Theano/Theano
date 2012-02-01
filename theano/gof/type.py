"""WRITEME Defines the `Type` class."""

__docformat__ = "restructuredtext en"

import copy
import utils
from utils import MethodNotDefined, object2
import graph
from theano import config

########
# Type #
########
from theano.gof.op import CLinkerObject

class CLinkerType(CLinkerObject):
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
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise MethodNotDefined("c_literal", type(self), self.__class__.__name__)

    def c_declare(self, name, sub):
        """Required: Return c code to declare variables that will be
        instantiated by `c_extract`.

        Example:
        .. code-block: python

            return "PyObject ** addr_of_%(name)s;"

        :param name: the name of the ``PyObject *`` pointer that will  the value for this Type

        :type name: string

        :param sub: a dictionary of special codes.  Most importantly sub['fail'].  See CLinker
        for more info on `sub` and ``fail``.

        :type sub: dict string -> string

        :note: It is important to include the `name` inside of variables which are declared
        here, so that name collisions do not occur in the source file that is generated.

        :note: The variable called ``name`` is not necessarily defined yet where this code is
        inserted.  This code might be inserted to create class variables for example, whereas
        the variable ``name`` might only exist inside certain functions in that class.

        :todo: Why should variable declaration fail?  Is it even allowed to?

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method
        """
        raise MethodNotDefined()

    def c_init(self, name, sub):
        """Required: Return c code to initialize the variables that were declared by
        self.c_declare()

        Example:
        .. code-block: python

            return "addr_of_%(name)s = NULL;"

        :note: The variable called ``name`` is not necessarily defined yet where this code is
        inserted.  This code might be inserted in a class constructor for example, whereas
        the variable ``name`` might only exist inside certain functions in that class.

        :todo: Why should variable initialization fail?  Is it even allowed to?
        """
        raise MethodNotDefined("c_init", type(self), self.__class__.__name__)

    def c_extract(self, name, sub):
        """Required: Return c code to extract a PyObject * instance.

        The code returned from this function must be templated using
        ``%(name)s``, representing the name that the caller wants to
        call this `Variable`. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".

        :todo: Point out that template filling (via sub) is now performed
        by this function. --jpt


        Example:
        .. code-block: python

            return "if (py_%(name)s == Py_None)" + \\\
                        addr_of_%(name)s = &py_%(name)s;" + \\\
                   "else" + \\\
                   { PyErr_SetString(PyExc_ValueError, 'was expecting None'); %(fail)s;}"


        :param name: the name of the ``PyObject *`` pointer that will  the value for this Type

        :type name: string

        :param sub: a dictionary of special codes.  Most importantly sub['fail'].  See CLinker
        for more info on `sub` and ``fail``.

        :type sub: dict string -> string

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise MethodNotDefined("c_extract", type(self), self.__class__.__name__)

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
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise MethodNotDefined()

    def c_sync(self, name, sub):
        """Required: Return c code to pack C types back into a PyObject.

        The code returned from this function must be templated using "%(name)s",
        representing the name that the caller wants to call this Variable.  The
        returned code may set "py_%(name)s" to a PyObject* and that PyObject*
        will be accessible from Python via variable.data. Do not forget to adjust
        reference counts if "py_%(name)s" is changed from its original value.

        :Parameters:
         - `name`: WRITEME
            WRITEME
         - `sub`: WRITEME
            WRITEME

        :Exceptions:
         - `MethodNotDefined`: Subclass does not implement this method

        """
        raise MethodNotDefined("c_sync", type(self), self.__class__.__name__)

    def c_code_cache_version(self):
        """Return a tuple of integers indicating the version of this Type.

        An empty tuple indicates an 'unversioned' Type that will not be cached between processes.

        The cache mechanism may erase cached modules that have been superceded by newer
        versions.  See `ModuleCache` for details.

        """
        return ()



class PureType(object):
    """Interface specification for variable type instances.

    A :term:`Type` instance is mainly reponsible for two things:

    - creating `Variable` instances (conventionally, `__call__` does this), and

    - filtering a value assigned to a `Variable` so that the value conforms to restrictions
      imposed by the type (also known as casting, this is done by `filter`),

    """

    Variable = graph.Variable #the type that will be created by call to make_variable.
    Constant = graph.Constant #the type that will be created by call to make_constant

    def filter(self, data, strict=False, allow_downcast=None):
        """Required: Return data or an appropriately wrapped/converted data.

        Subclass implementation should raise a TypeError exception if the data is not of an
        acceptable type.

        If strict is True, the data returned must be the same as the
        data passed as an argument. If it is False, and allow_downcast
        is True, filter may cast it to an appropriate type. If
        allow_downcast is False, filter may only upcast it, not lose
        precision. If allow_downcast is None (default), the behaviour can be
        Type-dependent, but for now it means only Python floats can be
        downcasted, and only to floatX scalars.

        :Exceptions:
         - `MethodNotDefined`: subclass doesn't implement this function.

        """
        raise MethodNotDefined("filter", type(self), self.__class__.__name__)

    # If filter_inplace is defined, it will be called instead of
    # filter() This is to allow reusing the old allocated memory. As
    # of this writing this is used only when we transfer new data to a
    # shared variable on the gpu.  

    #def filter_inplace(value, storage, strict=False, allow_downcast=None)

    def filter_variable(self, other):
        """Convert a symbolic variable into this Type, if compatible.

        For the moment, the only Types compatible with one another are
        TensorType and CudaNdarrayType, provided they have the same
        number of dimensions, same broadcasting pattern, and same dtype.

        If Types are not compatible, a TypeError should be raised.
        """
        if not isinstance(other, graph.Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type != self:
            raise TypeError(
                    'Cannot convert Type %(othertype)s '
                    '(of Variable %(other)s) into Type %(self)s. '
                    'You can try to manually convert %(other)s into a %(self)s.'
                    % dict(
                        othertype=other.type,
                        other=other,
                        self=self)
                    )
        return other

    def is_valid_value(self, a):
        """Required: Return True for any python object `a` that would be a legal value for a Variable of this Type"""
        try:
            self.filter(a, strict=True)
            return True
        except (TypeError, ValueError):
            return False

    def value_validity_msg(self, a):
        """Optional: return a message explaining the output of is_valid_value"""
        return "none"

    def make_variable(self, name = None):
        """Return a new `Variable` instance of Type `self`.

        :Parameters:
         - `name`: None or str
            A pretty string for printing and debugging.

        """
        return self.Variable(self, name = name)

    def make_constant(self, value, name=None):
        return self.Constant(type=self, data=value, name=name)


    def __call__(self, name = None):
        """Return a new `Variable` instance of Type `self`.

        :Parameters:
         - `name`: None or str
            A pretty string for printing and debugging.

        """
        return utils.add_tag_trace(self.make_variable(name))

    def values_eq(self, a, b):
        """
        Return True if a and b can be considered exactly equal.

        a and b are assumed to be valid values of this Type.
        """
        return a == b

    def values_eq_approx(self, a, b):
        """
        Return True if a and b can be considered approximately equal.

        :param a: a potential value for a Variable of this Type.

        :param b: a potential value for a Variable of this Type.

        :rtype: Bool

        This function is used by theano debugging tools to decide
        whether two values are equivalent, admitting a certain amount
        of numerical instability.  For example, for floating-point
        numbers this function should be an approximate comparison.

        By default, this does an exact comparison.
        """
        return self.values_eq(a, b)


_nothing = """
       """


class Type(object2, PureType, CLinkerType):
    """Convenience wrapper combining `PureType` and `CLinkerType`.

    Theano comes with several subclasses of such as:

    - `Generic`: for any python type

    - `TensorType`: for numpy.ndarray

    - `SparseType`: for scipy.sparse

    But you are encouraged to write your own, as described in WRITEME.

    The following following code illustrates the use of a Type instance, here tensor.fvector:

    .. python::
        # Declare a symbolic floating-point vector using __call__
        b = tensor.fvector()

        # Create a second Variable with the same Type instance
        c = tensor.fvector()

    Whenever you create a symbolic variable in theano (technically, `Variable`) it will contain a
    reference to a Type instance.  That reference is typically constant during the lifetime of
    the Variable.  Many variables can refer to a single Type instance, as do b and c above.  The
    Type instance defines the kind of value which might end up in that variable when executing
    a `Function`.  In this sense, theano is like a strongly-typed language because the types
    are included in the graph before the values.  In our example above, b is a Variable which is
    guaranteed to correspond to a numpy.ndarray of rank 1 when we try to do some computations
    with it.

    Many `Op` instances will raise an exception if they are applied to inputs with incorrect
    types.  Type references are also useful to do type-checking in pattern-based optimizations.

    """


class SingletonType(Type):
    """Convenient Base class for a Type subclass with no attributes

    It saves having to implement __eq__ and __hash__
    """
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

    def filter(self, data, strict=False, allow_downcast=None):
        return data

    def is_valid_value(self, a):
        return True

    def c_declare(self, name, sub):
        return """
        PyObject* %(name)s;
        """ % locals()

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        """ % locals()

    def c_extract(self, name, sub):
        return """
        Py_INCREF(py_%(name)s);
        %(name)s = py_%(name)s;
        """ % locals()

    def c_cleanup(self, name, sub):
        return """
        Py_XDECREF(%(name)s);
        """ % locals()

    def c_sync(self, name, sub):
        return """
        assert(py_%(name)s->ob_refcnt > 1);
        Py_DECREF(py_%(name)s);
        py_%(name)s = %(name)s ? %(name)s : Py_None;
        Py_INCREF(py_%(name)s);
        """ % locals()

    def __str__(self):
        return self.__class__.__name__

generic = Generic()
