"""
WRITEME

Defines the `Type` class.

"""
from __future__ import absolute_import, print_function, division

import ctypes
import platform

from six import string_types

import re
import theano
from theano.gof import graph, utils
from theano.gof.utils import MethodNotDefined, object2
from theano import change_flags

########
# Type #
########
from theano.gof.op import CLinkerObject, Op

__docformat__ = "restructuredtext en"


class CLinkerType(CLinkerObject):
    """
    Interface specification for Types that can be arguments to a `CLinkerOp`.

    A CLinkerType instance is mainly responsible  for providing the C code that
    interfaces python objects with a C `CLinkerOp` implementation.

    See WRITEME for a general overview of code generation by `CLinker`.

    """

    def c_element_type(self):
        """
        Optional: Return the name of the primitive C type of items into variables
        handled by this type.

        e.g:

         - For ``TensorType(dtype='int64', ...)``: should return ``"npy_int64"``.
         - For ``GpuArrayType(dtype='int32', ...)``: should return ``"ga_int"``.

        """
        raise MethodNotDefined("c_element_type", type(self), self.__class__.__name__)

    def c_is_simple(self):
        """
        Optional: Return True for small or builtin C types.

        A hint to tell the compiler that this type is a builtin C type or a
        small struct and that its memory footprint is negligible. Simple
        objects may be passed on the stack.

        """
        return False

    def c_literal(self, data):
        """
        Optional: WRITEME

        Parameters
        ----------
        data : WRITEME
            WRITEME

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_literal", type(self),
                               self.__class__.__name__)

    def c_declare(self, name, sub, check_input=True):
        """
        Required: Return c code to declare variables that will be
        instantiated by `c_extract`.

        Parameters
        ----------
        name: str
            The name of the ``PyObject *`` pointer that will
            the value for this Type
        sub: dict string -> string
            a dictionary of special codes.  Most importantly
            sub['fail']. See CLinker for more info on `sub` and ``fail``.

        Notes
        -----
        It is important to include the `name` inside of variables which
        are declared here, so that name collisions do not occur in the
        source file that is generated.

        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted to
        create class variables for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable declaration fail?  Is it even allowed to?

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        Examples
        --------
        .. code-block: python

            return "PyObject ** addr_of_%(name)s;"

        """
        raise MethodNotDefined()

    def c_init(self, name, sub):
        """
        Required: Return c code to initialize the variables that were declared
        by self.c_declare().

        Notes
        -----
        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted in a
        class constructor for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable initialization fail?  Is it even allowed to?

        Examples
        --------
        .. code-block: python

            return "addr_of_%(name)s = NULL;"

        """
        raise MethodNotDefined("c_init", type(self), self.__class__.__name__)

    def c_extract(self, name, sub, check_input=True):
        """
        Required: Return c code to extract a PyObject * instance.

        The code returned from this function must be templated using
        ``%(name)s``, representing the name that the caller wants to
        call this `Variable`. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".

        TODO: Point out that template filling (via sub) is now performed
              by this function. --jpt

        Parameters
        ----------
        name : str
            The name of the ``PyObject *`` pointer that will
            store the value for this Type.
        sub : dict string -> string
            A dictionary of special codes. Most importantly
            sub['fail']. See CLinker for more info on `sub` and ``fail``.

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        Examples
        --------
        .. code-block: python

            return "if (py_%(name)s == Py_None)" + \\\
                        addr_of_%(name)s = &py_%(name)s;" + \\\
                   "else" + \\\
                   { PyErr_SetString(PyExc_ValueError, \\\
                        'was expecting None'); %(fail)s;}"

        """
        raise MethodNotDefined("c_extract", type(self),
                               self.__class__.__name__)

    def c_extract_out(self, name, sub, check_input=True):
        """
        Optional: C code to extract a PyObject * instance.

        Unlike c_extract, c_extract_out has to accept Py_None,
        meaning that the variable should be left uninitialized.

        """
        return """
        if (py_%(name)s == Py_None)
        {
            %(c_init_code)s
        }
        else
        {
            %(c_extract_code)s
        }
        """ % dict(
            name=name,
            c_init_code=self.c_init(name, sub),
            c_extract_code=self.c_extract(name, sub, check_input))

    def c_cleanup(self, name, sub):
        """
        Return C code to clean up after `c_extract`.

        This returns C code that should deallocate whatever `c_extract`
        allocated or decrease the reference counts. Do not decrease
        py_%(name)s's reference count.

        WRITEME

        Parameters
        ----------
        name : WRITEME
            WRITEME
        sub : WRITEME
            WRITEME

        Raises
        ------
         MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined()

    def c_sync(self, name, sub):
        """
        Required: Return C code to pack C types back into a PyObject.

        The code returned from this function must be templated using
        "%(name)s", representing the name that the caller wants to
        call this Variable. The returned code may set "py_%(name)s"
        to a PyObject* and that PyObject* will be accessible from
        Python via variable.data. Do not forget to adjust reference
        counts if "py_%(name)s" is changed from its original value.

        Parameters
        ----------
        name : WRITEME
            WRITEME
        sub : WRITEME
            WRITEME

        Raises
        ------
        MethodNotDefined
            Subclass does not implement this method.

        """
        raise MethodNotDefined("c_sync", type(self), self.__class__.__name__)

    def c_code_cache_version(self):
        """
        Return a tuple of integers indicating the version of this Type.

        An empty tuple indicates an 'unversioned' Type that will not
        be cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions. See `ModuleCache` for details.

        """
        return ()


class PureType(object):
    """
    Interface specification for variable type instances.

    A :term:`Type` instance is mainly responsible for two things:

    - creating `Variable` instances (conventionally, `__call__` does this), and

    - filtering a value assigned to a `Variable` so that the value
      conforms to restrictions imposed by the type (also known as
      casting, this is done by `filter`).

    """

    # the type that will be created by call to make_variable.
    Variable = graph.Variable

    # the type that will be created by call to make_constant
    Constant = graph.Constant

    def filter(self, data, strict=False, allow_downcast=None):
        """
        Required: Return data or an appropriately wrapped/converted data.

        Subclass implementation should raise a TypeError exception if
        the data is not of an acceptable type.

        If strict is True, the data returned must be the same as the
        data passed as an argument. If it is False, and allow_downcast
        is True, filter may cast it to an appropriate type. If
        allow_downcast is False, filter may only upcast it, not lose
        precision. If allow_downcast is None (default), the behaviour can be
        Type-dependent, but for now it means only Python floats can be
        downcasted, and only to floatX scalars.

        Raises
        ------
        MethodNotDefined
            Subclass doesn't implement this function.

        """
        raise MethodNotDefined("filter", type(self), self.__class__.__name__)

    # If filter_inplace is defined, it will be called instead of
    # filter() This is to allow reusing the old allocated memory. As
    # of this writing this is used only when we transfer new data to a
    # shared variable on the gpu.

    # def filter_inplace(value, storage, strict=False, allow_downcast=None)

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a symbolic variable into this Type, if compatible.

        For the moment, the only Types compatible with one another are
        TensorType and GpuArrayType, provided they have the same
        number of dimensions, same broadcasting pattern, and same
        dtype.

        If Types are not compatible, a TypeError should be raised.

        """
        if not isinstance(other, graph.Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type != self and allow_convert:
            other2 = self.convert_variable(other)
            if other2 is not None:
                return other2

        if other.type != self:
            raise TypeError(
                'Cannot convert Type %(othertype)s '
                '(of Variable %(other)s) into Type %(self)s. '
                'You can try to manually convert %(other)s into a %(self)s.'
                % dict(othertype=other.type, other=other, self=self))
        return other

    def convert_variable(self, var):
        """
        Patch variable so that its type will match self, if possible.

        If the variable can't be converted, this should return None.

        The conversion can only happen if the following implication is
        true for all possible `val`.

          self.is_valid_value(val) => var.type.is_valid_value(val)

        For the majority of types this means that you can only have
        non-broadcastable dimensions become broadcastable and not the
        inverse.

        The default is to not convert anything which is always safe.

        """
        return None

    def is_valid_value(self, a):
        """
        Required: Return True for any python object `a` that would be a
        legal value for a Variable of this Type.

        """
        try:
            self.filter(a, strict=True)
            return True
        except (TypeError, ValueError):
            return False

    def value_validity_msg(self, a):
        """
        Optional: Return a message explaining the output of
        is_valid_value.

        """
        return "none"

    def make_variable(self, name=None):
        """
        Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
            A pretty string for printing and debugging.

        """
        return self.Variable(self, name=name)

    def make_constant(self, value, name=None):
        return self.Constant(type=self, data=value, name=name)

    def __call__(self, name=None):
        """
        Return a new `Variable` instance of Type `self`.

        Parameters
        ----------
        name : None or str
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

        This function is used by theano debugging tools to decide
        whether two values are equivalent, admitting a certain amount
        of numerical instability. For example, for floating-point
        numbers this function should be an approximate comparison.

        By default, this does an exact comparison.

        Parameters
        ----------
        a
            A potential value for a Variable of this Type.

        b
            A potential value for a Variable of this Type.

        Returns
        -------
        bool

        """
        return self.values_eq(a, b)

#    def get_shape_info(self, obj):
        """
        Optional function. See TensorType().get_shape_info for definition.

        """

#    def get_size(self, shape_info):
        """
        Optional function. See TensorType().get_size for definition.

        """

_nothing = """
       """


class Type(object2, PureType, CLinkerType):
    """
    Convenience wrapper combining `PureType` and `CLinkerType`.

    Theano comes with several subclasses of such as:

    - `Generic`: for any python type

    - `TensorType`: for numpy.ndarray

    - `SparseType`: for scipy.sparse

    But you are encouraged to write your own, as described in WRITEME.

    The following code illustrates the use of a Type instance,
    here tensor.fvector:

    .. code-block:: python

        # Declare a symbolic floating-point vector using __call__
        b = tensor.fvector()

        # Create a second Variable with the same Type instance
        c = tensor.fvector()

    Whenever you create a symbolic variable in theano (technically,
    `Variable`) it will contain a reference to a Type instance. That
    reference is typically constant during the lifetime of the
    Variable.  Many variables can refer to a single Type instance, as
    do b and c above.  The Type instance defines the kind of value
    which might end up in that variable when executing a `Function`.
    In this sense, theano is like a strongly-typed language because
    the types are included in the graph before the values.  In our
    example above, b is a Variable which is guaranteed to correspond
    to a numpy.ndarray of rank 1 when we try to do some computations
    with it.

    Many `Op` instances will raise an exception if they are applied to
    inputs with incorrect types.  Type references are also useful to
    do type-checking in pattern-based optimizations.

    """


class SingletonType(Type):
    """
    Convenient Base class for a Type subclass with no attributes.

    It saves having to implement __eq__ and __hash__.

    """

    __instance = None

    def __new__(cls):
        # If sub-subclass of SingletonType don't redeclare __instance
        # when we look for it, we will find it in the subclass.  We
        # don't want that, so we check the class.  When we add one, we
        # add one only to the current class, so all is working
        # correctly.
        if cls.__instance is None or not isinstance(cls.__instance, cls):
            cls.__instance = Type.__new__(cls)
        return cls.__instance

    def __str__(self):
        return self.__class__.__name__

    # even if we try to make a singleton, this do not always work.  So
    # we compare the type. See test_type_other.test_none_Constant for
    # an exmple. So we need to implement __eq__ and __hash__
    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is type(other):
            return True
        return False

    def __hash__(self):
        return hash(type(self))


class Generic(SingletonType):
    """
    Represents a generic Python object.

    This class implements the `PureType` and `CLinkerType` interfaces
    for generic PyObject instances.

    EXAMPLE of what this means, or when you would use this type.

    WRITEME

    """

    def filter(self, data, strict=False, allow_downcast=None):
        return data

    def is_valid_value(self, a):
        return True

    def c_declare(self, name, sub, check_input=True):
        return """
        PyObject* %(name)s;
        """ % locals()

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        """ % locals()

    def c_extract(self, name, sub, check_input=True):
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

    def c_code_cache_version(self):
        return (1,)

    def __str__(self):
        return self.__class__.__name__

generic = Generic()

_cdata_type = None

if platform.python_implementation() != 'PyPy':
    _cdata_type = ctypes.py_object.from_address(
        ctypes.addressof(ctypes.pythonapi.PyCapsule_Type)).value


class _make_cdata(Op):
    __props__ = ('rtype',)

    def __init__(self, rtype):
        assert isinstance(rtype, CDataType)
        self.rtype = rtype

    def do_constant_folding(self, node):
        return False

    def make_node(self, val):
        from theano.scalar import as_scalar
        from theano import Apply

        val = as_scalar(val).astype('uint64')
        return Apply(self, [val], [self.rtype()])

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        %(out)s = (%(ctype)s)%(inp)s;
        """ % dict(ctype=self.rtype.ctype, out=outputs[0], inp=inputs[0])

    def c_code_cache_version(self):
        if self.rtype.version is None:
            return None
        return (0, self.rtype.version)


class CDataType(Type):
    """
    Represents opaque C data to be passed around. The intent is to
    ease passing arbitrary data between ops C code.

    The constructor builds a type made to represent a C pointer in theano.

    Parameters
    ----------
    ctype
        The type of the pointer (complete with the `*`).

    freefunc
        A function to call to free the pointer. This function must
        have a `void` return and take a single pointer argument.

    version
        The version to use in Theano cache system.
    """
    __props__ = ('ctype', 'freefunc', 'headers', 'header_dirs',
                 'libraries', 'lib_dirs', 'extra_support_code',
                 'compile_args', 'version')

    def __init__(self, ctype, freefunc=None, headers=(), header_dirs=(),
                 libraries=(), lib_dirs=(), compile_args=(),
                 extra_support_code="", version=None):
        assert isinstance(ctype, string_types)
        self.ctype = ctype
        if freefunc is not None:
            assert isinstance(freefunc, string_types)
        self.freefunc = freefunc
        self.headers = tuple(headers)
        self.header_dirs = tuple(header_dirs)
        self.libraries = tuple(libraries)
        self.lib_dirs = tuple(lib_dirs)
        self.compile_args = tuple(compile_args)
        self.extra_support_code = extra_support_code
        self._fn = None
        self.version = version

    def filter(self, data, strict=False, allow_downcast=None):
        # We ignore this type-check (_cdata_type is None) in PyPy
        # because this type is not exposed to us.
        if data is not None and _cdata_type is not None:
            if not isinstance(data, _cdata_type):
                raise TypeError("expected None or a PyCapsule")

        return data

    def _get_func(self):
        """
        Return a function that makes a value from an integer.

        The integer value is assumed to be a valid pointer for the
        type and no check is done to ensure that.
        """
        from theano.scalar import get_scalar_type

        if self._fn is None:
            with change_flags(compute_test_value='off'):
                v = get_scalar_type('int64')()
                self._fn = theano.function([v], _make_cdata(self)(v),
                                           mode=theano.Mode(optimizer=None),
                                           profile=False)
        return self._fn

    def make_value(self, ptr):
        """
        Make a value of this type.

        Parameters
        ----------
        ptr : int
            Integer representation of a valid pointer value

        """
        return self._get_func()(ptr)

    def c_declare(self, name, sub, check_input=True):
        return """
        %(ctype)s %(name)s;
        """ % dict(ctype=self.ctype, name=name)

    def c_init(self, name, sub):
        return "%(name)s = NULL;" % dict(name=name)

    def c_extract(self, name, sub, check_input=True):
        return """
  %(name)s = (%(ctype)s)PyCapsule_GetPointer(py_%(name)s, NULL);
  if (%(name)s == NULL) %(fail)s
        """ % dict(name=name, ctype=self.ctype, fail=sub['fail'])

    def c_support_code(self):
        return """
void _capsule_destructor(PyObject *o) {
    void *d = PyCapsule_GetContext(o);
    void *p = PyCapsule_GetPointer(o, NULL);
    void (*f)(void *) = (void (*)(void *))d;
    if (f != NULL) f(p);
}
""" + self.extra_support_code

    def c_sync(self, name, sub):
        freefunc = self.freefunc
        if freefunc is None:
            freefunc = "NULL"
        s = """
Py_XDECREF(py_%(name)s);
if (%(name)s == NULL) {
  py_%(name)s = Py_None;
  Py_INCREF(py_%(name)s);
} else {
  py_%(name)s = PyCapsule_New((void *)%(name)s, NULL,
                              _capsule_destructor);
  if (py_%(name)s != NULL) {
    if (PyCapsule_SetContext(py_%(name)s, (void *)%(freefunc)s) != 0) {
      /* This won't trigger a call to freefunc since it could not be
         set. The error case below will do it. */
      Py_DECREF(py_%(name)s);
      /* Signal the error */
      py_%(name)s = NULL;
    }
  }
}"""
        if self.freefunc is not None:
            s += """
if (py_%(name)s == NULL) { %(freefunc)s(%(name)s); }
"""
        return s % dict(name=name, freefunc=freefunc)

    def c_cleanup(self, name, sub):
        # No need to do anything here since the CObject/Capsule will
        # free the data for us when released.
        return ""

    def c_headers(self):
        return self.headers

    def c_header_dirs(self):
        return self.header_dirs

    def c_libraries(self):
        return self.libraries

    def c_lib_dirs(self):
        return self.lib_dirs

    def c_compile_args(self):
        return self.compile_args

    def c_code_cache_version(self):
        v = (3, )
        if self.version is not None:
            v = v + (self.version,)
        return v

    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.ctype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        if not hasattr(self, 'headers'):
            self.headers = ()
            self.header_dirs = ()
            self.libraries = ()
            self.lib_dirs = ()
        if not hasattr(self, 'version'):
            self.version = None


class CDataTypeConstant(graph.Constant):
    def merge_signature(self):
        # We don't want to merge constants that don't point to the
        # same object.
        return id(self.data)

    def signature(self):
        # There is no way to put the data in the signature, so we
        # don't even try
        return (self.type,)

CDataType.Constant = CDataTypeConstant


class EnumType(Type, dict):
    """
    Main subclasses:
     - :class:`EnumList`
     - :class:`CEnumType`

    Op parameter class that allows to create enumerations of constant values.

     - Constants are available as object attributes in Python code and as macro-defined constants in C code.
     - Constants can be floating values, integers, or booleans (automatically converted to integers).
     - Constants name must start with a capital letter and contain capital letters, underscores or digits.
     - A constant can have an alias, and then be available through both constant name and constant alias.

    **Example**

    .. code-block:: python

        enum = EnumType(CONSTANT_1=1, CONSTANT_2=2.5, CONSTANT_3=False, CONSTANT_4=True)
        print (enum.CONSTANT_1, enum.CONSTANT_2, enum.CONSTANT_3, enum.CONSTANT_4)
        # will print 1 2.5 0 1

    In C code:

    .. code-block:: c

        int constant_1 = CONSTANT_1;
        double constant_2 = CONSTANT_2;
        int constant_3 = CONSTANT_3; // constant_3 == 0
        int constant_4 = CONSTANT_4; // constant_4 == 1

    You can also specify a C type for the op param. Default C type is ``double``.

    .. code-block:: python

        enum = EnumType(CONSTANT_1=0, CONSTANT_2=1, CONSTANT_3=2, ctype='size_t')
        # In C code, the Op param will then be a ``size_t``.

    .. note::

        You can also specify a C name (``cname``) or the current enumeration. This C name may be
        used to name functions related to that specific enumeration, e.g. for debugging
        purposes. Default C name is the C type (with any sequence of spaces replaced with
        an underscore). If you want to debug and your C type is quite generic (e.g.
        ``int`` or ``double``), we recommend you specify a C name.

        C name must be a valid C identifier.

        .. code-block:: python

            enum = EnumType(CONSTANT_1=0, CONSTANT_2=1, CONSTANT_3=2,
                            ctype='size_t', cname='MyEnumName')

    **Example with aliases**

    When creating an enum, you can give some aliases to specific constants while keeping other constants without aliases.
    An alias must be a string, and there is currently no string format constraints.

    To give an alias to a constant in the EnumType constructor, use the following key-value syntax::

        constant_name=(constant_alias, constant_value)

    You can then retrieve a constant from an alias with method ``EnumType.fromalias()``.

    Aliases are intended to be used in Python code only (only constants names are available in C code).
    Especially, an alias will be recognized by ``Enumtype.filter()`` method with non-strict filtering,
    allowing a maximum flexibility for converting strings to numeric constants available in Python and C code.

    .. code-block:: python

        from theano.gof import EnumType

        # You can remark that constant 'C' does not have an alias.
        enum = EnumType(A=('alpha', 1), B=('beta', 2), C=3, D=('delta', 4))

        # Constants are all directly available by name.
        print(enum.A, enum.B, enum.C, enum.D)

        # But we can also now get some constants by alias.
        a = enum.fromalias('alpha')
        b = enum.fromalias('beta')
        d = enum.fromalias('delta')

        # If method fromalias() receives an unknown alias,
        # it will looks for a constant with this alias
        # as exact constant name.
        c = enum.fromalias('C') # will get enum.C

        # An alias defined in an EnumType will be correctly converted with non-strict filtering.
        value = enum.filter('delta', strict=False)
        # value now contaisn enum.D, ie. 4.

    .. note::

        This Type (and subclasses) is not complete and should never be used for regular graph operations.

    """

    def __init_ctype(self, ctype):
        # C type may be a list of keywords, e.g. "unsigned long long".
        # We should check each part.
        ctype_parts = ctype.split()
        if not all(re.match('^[A-Za-z_][A-Za-z0-9_]*$', el) for el in ctype_parts):
            raise TypeError('%s: invalid C type.' % type(self).__name__)
        self.ctype = ' '.join(ctype_parts)

    def __init_cname(self, cname):
        if not re.match('^[A-Za-z_][A-Za-z0-9_]*$', cname):
            raise TypeError("%s: invalid C name." % type(self).__name__)
        self.cname = cname

    def __init__(self, **kwargs):
        self.__init_ctype(kwargs.pop('ctype', 'double'))
        self.__init_cname(kwargs.pop('cname', self.ctype.replace(' ', '_')))
        self.aliases = dict()
        for k in kwargs:
            if re.match('^[A-Z][A-Z0-9_]*$', k) is None:
                raise AttributeError('%s: invalid enum name: "%s". '
                                     'Only capital letters, underscores and digits '
                                     'are allowed.' % (type(self).__name__, k))
            if isinstance(kwargs[k], (list, tuple)):
                if len(kwargs[k]) != 2:
                    raise TypeError('%s: when using a tuple to define a constant, your tuple should contain 2 values: '
                                    'constant alias followed by constant value.' % type(self).__name__)
                alias, value = kwargs[k]
                if not isinstance(alias, str):
                    raise TypeError('%s: constant alias should be a string, got "%s".'
                                    % (type(self).__name__, alias))
                if alias == k:
                    raise TypeError("%s: it's useless to create an alias "
                                    "with the same name as its associated constant." % type(self).__name__)
                if alias in self.aliases:
                    raise TypeError('%s: consant alias "%s" already used.' % (type(self).__name__, alias))
                self.aliases[alias] = k
                kwargs[k] = value
            if isinstance(kwargs[k], bool):
                kwargs[k] = int(kwargs[k])
            elif not isinstance(kwargs[k], (int, float)):
                raise TypeError('%s: constant "%s": expected integer or floating value, got "%s".'
                                % (type(self).__name__, k, type(kwargs[k]).__name__))
        if [a for a in self.aliases if a in self]:
            raise TypeError("%s: some aliases have same names as constants." % type(self).__name__)
        super(EnumType, self).__init__(**kwargs)

    def fromalias(self, alias):
        """
        Get a constant value by its alias.
        If there is not such alias in this enum, look for a constant
        with this alias as constant name.
        """
        return self[self.aliases[alias]] if alias in self.aliases else self[alias]

    def has_alias(self, alias):
        """
        return True if and only if this enum has this alias.
        """
        return alias in self.aliases

    def get_aliases(self):
        """
        Return the sorted tuple of all aliases in this enumeration.
        """
        return tuple(sorted(self.aliases.keys()))

    def __repr__(self):
        names_to_aliases = {constant_name: '' for constant_name in self}
        for alias in self.aliases:
            names_to_aliases[self.aliases[alias]] = '(%s)' % alias
        return '%s<%s>(%s)' % (type(self).__name__, self.ctype,
                               ', '.join('%s%s:%s' % (k, names_to_aliases[k], self[k]) for k in sorted(self.keys())))

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return Type.__getattr__(self, key)

    def __setattr__(self, key, value):
        if key in self:
            raise NotImplementedError('constant values are immutable.')
        Type.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        raise NotImplementedError('constant values are immutable.')

    def __delitem__(self, key):
        raise NotImplementedError('constant values are immutable.')

    def __hash__(self):
        # All values are Python basic types, then easy to hash.
        return hash((type(self), self.ctype) +
                    tuple((k, self[k]) for k in sorted(self.keys())) +
                    tuple((a, self.aliases[a]) for a in sorted(self.aliases.keys())))

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ctype == other.ctype and
                len(self) == len(other) and
                len(self.aliases) == len(other.aliases) and
                all(k in other for k in self) and
                all(a in other.aliases for a in self.aliases) and
                all(self[k] == other[k] for k in self) and
                all(self.aliases[a] == other.aliases[a] for a in self.aliases))

    # EnumType should be used to create constants available in both Python and C code.
    # However, for convenience, we make sure EnumType can have a value, like other common types,
    # such that it could be used as-is as an op param.
    # C type of value is defined in self.ctype.

    def filter(self, data, strict=False, allow_downcast=None):
        if not strict:
            if isinstance(data, bool):
                data = int(data)
            elif isinstance(data, str):
                # We now accept strings as data values.
                # Strings should be a constant alias or a constant name.
                data = self.fromalias(data)
        assert data in self.values()
        return data

    def values_eq(self, a, b):
        return a == b

    def values_eq_approx(self, a, b):
        # For an enum, it does not have a meaning to be approx equal.
        return self.values_eq(a, b)

    pyint_compat_code = """
    #if PY_MAJOR_VERSION >= 3
        #ifndef PyInt_Check
            #define PyInt_Check PyLong_Check
        #endif
        #ifndef PyInt_AsLong
            #define PyInt_AsLong PyLong_AsLong
        #endif
    #endif
    """

    def c_to_string(self):
        """
        Return code for a C function that will convert an enumeration value
        to a string representation. The function prototype is:

        .. code-block:: c

            int theano_enum_to_string_<cname>(<ctype> value, char* output_string);

        Where ``ctype`` and ``cname`` are the C type and the C name of current Theano enumeration.

        ``output_string`` should be large enough to contain the longest name in this enumeration.

        If given value is unknown, the C function sets a Python ValueError exception and returns a non-zero.

        This C function may be useful to retrieve some runtime informations.
        It is available in C code when theano flag ``config.cmodule.debug`` is set to ``True``.
        """
        return """
        #ifdef DEBUG
        int theano_enum_to_string_%(cname)s(%(ctype)s in, char* out) {
            int ret = 0;
            switch(in) {
                %(cases)s
                default:
                    PyErr_SetString(PyExc_ValueError, "%(classname)s:  unknown enum value.");
                    ret = -1;
                    break;
            }
            return ret;
        }
        #endif
        """ % dict(cname=self.cname, ctype=self.ctype,
                   classname=type(self).__name__,
                   cases=''.join("""
                   case %(name)s: sprintf(out, "%(name)s"); break;
                   """ % dict(name=name) for name in self))

    def c_support_code(self):
        return (
            self.pyint_compat_code +
            ''.join("""
            #define %s %s
            """ % (k, str(self[k])) for k in sorted(self.keys())) +
            self.c_to_string()
        )

    def c_declare(self, name, sub, check_input=True):
        return """%(ctype)s %(name)s;""" % dict(ctype=self.ctype, name=name)

    def c_init(self, name, sub):
        return "%(name)s = (%(ctype)s)0;" % dict(name=name, ctype=self.ctype)

    def c_cleanup(self, name, sub):
        return ""

    def c_extract(self, name, sub, check_input=True):
        return """
        if (PyInt_Check(py_%(name)s)) {
            %(name)s = (%(ctype)s)PyInt_AsLong(py_%(name)s);
        } else {
            %(name)s = (%(ctype)s)PyFloat_AsDouble(py_%(name)s);
        }
        if (PyErr_Occurred()) {
            %(fail)s
        }
        """ % dict(ctype=self.ctype, name=name, fail=sub['fail'])

    def c_code_cache_version(self):
        return (2, self.ctype, self.cname, tuple(self.items()))


class EnumList(EnumType):
    """
    **Inherit from**:
     - :class:`EnumType`

    Op parameter class that allows to create enumeration of constant values.
    Same as :class:`EnumType`, but automatically gives an unique integer value for each constant in a list of
    constants names (constant at index ``i`` in the list will receive value ``i``,
    with ``i`` from ``0`` to ``len(constants) - 1``).

    Example::

        enum = EnumList('CONSTANT_1', 'CONSTANT_2', 'CONSTANT_3', 'CONSTANT_4', 'CONSTANT_5')
        print (enum.CONSTANT_1, enum.CONSTANT_2, enum.CONSTANT_3, enum.CONSTANT_4, enum.CONSTANT_5)
        # will print: 0 1 2 3 4

    Like :class:`EnumType`, you can also define the C type and a C name for the op param.
    Default C type is ``int``::

        enum = EnumList('CONSTANT_1', 'CONSTANT_2', 'CONSTANT_3', 'CONSTANT_4', ctype='unsigned int')

    Like :class:`EnumType`, you can also add an alias to a constant, by replacing the only constant name
    (e.g. ``'CONSTANT_NAME'``) by a couple with constant name first and constant alias second
    (e.g. ``('CONSTANT_NAME', 'constant_alias')``).

    .. code-block:: python

        enum = EnumList(('A', 'alpha'), ('B', 'beta'), 'C', 'D', 'E', 'F', ('G', 'gamma'))

    See test class :class:`theano.gof.tests.test_types.TestOpEnumList` for a working example.

    """

    def __init__(self, *args, **kwargs):
        assert len(kwargs) in (0, 1, 2), (type(self).__name__ +
                                          ': expected 0 to 2 extra parameters ("ctype", "cname").')
        ctype = kwargs.pop('ctype', 'int')
        cname = kwargs.pop('cname', None)

        for arg_rank, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                if len(arg) != 2:
                    raise TypeError('%s: when using a tuple to define a constant, your tuple should contain 2 values: '
                                    'constant name followed by constant alias.' % type(self).__name__)
                constant_name, constant_alias = arg
                if not isinstance(constant_alias, str):
                    raise TypeError('%s: constant alias should be a string, got "%s".'
                                    % (type(self).__name__, constant_alias))
                constant_value = (constant_alias, arg_rank)
            else:
                constant_name = arg
                constant_value = arg_rank
            if not isinstance(constant_name, str):
                raise TypeError('%s: constant name should be a string, got "%s".'
                                % (type(self).__name__, constant_name))
            if constant_name in kwargs:
                raise TypeError('%s: constant name already used ("%s").' % (type(self).__name__, constant_name))
            kwargs[constant_name] = constant_value

        kwargs.update(ctype=ctype)
        if cname is not None:
            kwargs.update(cname=cname)
        super(EnumList, self).__init__(**kwargs)


class CEnumType(EnumList):
    """
    **Inherit from**:
     - :class:`EnumList`

    Op parameter class that allows to create enumeration of constant values that represent C-defined constants.

     - Constant should have same names as in C.
     - In Python, constants will have arbitrary-defined values.
       They should be used only for choices, not for its values.
     - In C code, the real values defined in C will be used.
       They could be used either for choices or for its real values.

    Like :class:`EnumList`, you can also define the C type and a C name for the op param.
    Default C type is ``int``.

    .. code-block:: python

        enum = CEnumType('CONSTANT_CNAME_1', 'CONSTANT_CNAME_2', 'CONSTANT_CNAME_3', ctype='long')

    Like :class:`EnumList`, you can also add an alias to a constant, with same syntax as in :class:`EnumList`.

    See test :func:`theano.gof.tests.test_types.TestEnumTypes.test_op_with_cenumtype` for a working example.

    .. note::

        Be sure C constants are available in your C code. If they come from a C header, consider implementing
        ``c_headers()`` and ``c_header_dirs()`` in the Op class where you use CEnumType as op parameter type.

    """

    def c_support_code(self):
        return self.pyint_compat_code + self.c_to_string()

    def c_extract(self, name, sub, check_input=True):
        swapped_dict = dict((v, k) for (k, v) in self.items())
        # swapped_dict's keys are integers.

        return """
        switch(PyInt_AsLong(py_%(name)s)) {
            %(cases)s
            default:
                PyErr_SetString(PyExc_ValueError, "CEnumType: invalid value to map to C constants.");
                {%(fail)s}
                break;
        }
        """ % dict(name=name,
                   cases=''.join("""
                   case %(i)d: %(name)s = %(constant_cname)s; break;
                   """ % dict(i=i, name=name, constant_cname=swapped_dict[i]) for i in sorted(swapped_dict.keys())),
                   fail=sub['fail'])

    def c_code_cache_version(self):
        return (1, super(CEnumType, self).c_code_cache_version())
