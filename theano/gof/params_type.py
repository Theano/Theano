"""
Module for wrapping many Op parameters into one object available in both Python and C code.

The module provides the main public class :class:`ParamsType` that allows to bundle many Theano types
into one parameter type, and an internal convenient class :class:`Params` which will be automatically
used to create a Params object that is compatible with the ParamsType defined.

The Params object will be available in both Python code (as a standard Python object) and C code
(as a specific struct with parameters as struct fields). To be fully-available in C code, Theano
types wrapped into a ParamsType must provide a C interface (e.g. TensorType, Scalar, GpuArrayType,
or your own type. See :ref:`extending_op_params` for more details).

Example of usage
----------------

Importation:

.. code-block:: python

    # Import ParamsType class.
    from theano.gof import ParamsType

    # If you want to use a tensor and a scalar as parameters,
    # you should import required Theano types.
    from theano.tensor import TensorType
    from theano.scalar import Scalar

In your Op sub-class:

.. code-block:: python

    params_type = ParamsType(attr1=TensorType('int32', (False, False)), attr2=Scalar('float64'))

If your op contains attributes ``attr1`` **and** ``attr2``, the default ``op.get_params()``
implementation will automatically try to look for it and generate an appropriate Params object.
Attributes must be compatible with the corresponding types defined into the ParamsType
(we will try to convert and downcast if needed). In this example, ``your_op.attr1``
should be a matrix of integers, and ``your_op.attr2``
should be a real number (integer or floating value).

.. code-block:: python

    def __init__(value_attr1, value_attr2):
        self.attr1 = value_attr1
        self.attr2 = value_attr2

In ``perform()`` implementation (with params named ``param``):

.. code-block:: python

    matrix_param = param.attr1
    number_param = param.attr2

In ``c_code()`` implementation (with ``param = sub['params']``):

.. code-block:: c

    PyArrayObject* matrix = param->attr1;
    npy_float64    number = param->attr2;
    /* You won't need to free them or whatever else. */


See :class:`QuadraticOpFunc` and :class:`QuadraticCOpFunc` in ``theano/gof/tests/test_params_type.py``
for complete working examples.

Combining ParamsType with Theano enumeration types
--------------------------------------------------

Theano provide some enumeration types that allow to create constant primitive values (integer and floating values)
available in both Python and C code. See :class:`theano.gof.type.EnumType` and its subclasses for more details.

If your ParamsType contains Theano enumeration types, then constants defined inside these
enumerations will be directly available as ParamsType attributes.

**Example**::

    from theano.gof import ParamsType, EnumType, EnumList

    wrapper = ParamsType(enum1=EnumList('CONSTANT_1', 'CONSTANT_2', 'CONSTANT_3'),
                         enum2=EnumType(PI=3.14, EPSILON=0.001))

    # Each enum constant is available as a wrapper attribute:
    print(wrapper.CONSTANT_1, wrapper.CONSTANT_2, wrapper.CONSTANT_3,
          wrapper.PI, wrapper.EPSILON)

    # For convenience, you can also look for a constant by name with
    # ``ParamsType.get_enum()`` method.
    pi = wrapper.get_enum('PI')
    epsilon = wrapper.get_enum('EPSILON')
    constant_2 = wrapper.get_enum('CONSTANT_2')
    print(pi, epsilon, constant_2)

This implies that a ParamsType cannot contain different enum types with common enum names::

    # Following line will raise an error,
    # as there is a "CONSTANT_1" defined both in enum1 and enum2.
    wrapper = ParamsType(enum1=EnumList('CONSTANT_1', 'CONSTANT_2'),
                         enum2=EnumType(CONSTANT_1=0, CONSTANT_3=5))

If your enum types contain constant aliases, you can retrive them from ParamsType
with ``ParamsType.enum_from_alias(alias)`` method (see :class:`theano.gof.type.EnumType`
for more info about enumeration aliases).

.. code-block:: python

    wrapper = ParamsType(enum1=EnumList('A', ('B', 'beta'), 'C'),
                         enum2=EnumList(('D', 'delta'), 'E', 'F'))
    b1 = wrapper.B
    b2 = wrapper.get_enum('B')
    b3 = wrapper.enum_from_alias('beta')
    assert b1 == b2 == b3

"""

from __future__ import absolute_import, print_function, division
import re
import hashlib
from theano.gof.utils import MethodNotDefined, c_cpp_keywords
from theano.gof import Type, EnumType


class Params(dict):
    """
    Internal convenient class to wrap many Python objects into one
    (this class is not safe as the hash method does not check if values are effectively hashable).

    **Example:**

    .. code-block:: python

        from theano.gof import ParamsType, Params
        from theano.scalar import Scalar
        # You must create a ParamsType first:
        params_type = ParamsType(attr1=Scalar('int32'),
                                 key2=Scalar('float32'),
                                 field3=Scalar('int64'))
        # Then you can create a Params object with
        # the params type defined above and values for attributes.
        params = Params(params_type, attr1=1, key2=2.0, field3=3)
        print(params.attr1, params.key2, params.field3)
        d = dict(attr1=1, key2=2.5, field3=-1)
        params2 = Params(params_type, **d)
        print(params2.attr1, params2.key2, params2.field3)

    """

    def __init__(self, params_type, **kwargs):
        if not isinstance(params_type, ParamsType):
            raise TypeError('Params: 1st constructor argument should be a ParamsType.')
        for field in params_type.fields:
            if field not in kwargs:
                raise TypeError('Params: ParamsType attribute "%s" not in Params args.' % field)
        super(Params, self).__init__(**kwargs)
        self.__dict__.update(__params_type__=params_type, __signatures__=None)

    def __repr__(self):
        return 'Params(%s)' % ', '.join([('%s:%s:%s' % (k, type(self[k]).__name__, self[k])) for k in sorted(self.keys())])

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('Params: attribute "%s" does not exist.' % key)
        return self[key]

    def __setattr__(self, key, value):
        raise NotImplementedError('Params is immutable')

    def __setitem__(self, key, value):
        raise NotImplementedError('Params is immutable')

    def __delitem__(self, key):
        raise NotImplementedError('Params is immutable')

    def __hash__(self):
        # As values are immutable, we can save data signatures the first time
        # to not regenerate them in future hash() calls.
        if self.__signatures__ is None:
            # NB: For writing, we must bypass setattr() which is always called by default by Python.
            self.__dict__['__signatures__'] = tuple(
                # NB: Params object should have been already filtered.
                self.__params_type__.types[i].make_constant(self[self.__params_type__.fields[i]]).signature()
                for i in range(self.__params_type__.length)
            )
        return hash((type(self), self.__params_type__) + self.__signatures__)

    def __eq__(self, other):
        return (type(self) == type(other) and self.__params_type__ == other.__params_type__ and all(
            # NB: Params object should have been already filtered.
            self.__params_type__.types[i].values_eq(self[self.__params_type__.fields[i]], other[self.__params_type__.fields[i]])
            for i in range(self.__params_type__.length)
        ))

    def __ne__(self, other):
        return not self.__eq__(other)


class ParamsType(Type):
    """
    This class can create a struct of Theano types (like TensorType, GpuArrayType, etc.)
    to be used as a convenience op parameter wrapping many data.

    ParamsType constructor takes key-value args.
    Key will be the name of the attribute in the struct.
    Value is the Theano type of this attribute, ie. an instance of (a subclass of) :class:`Type`
    (eg. ``TensorType('int64', (False,))``).

    In a Python code any attribute named ``key`` will be available via::

        structObject.key

    In a C code, any attribute named ``key`` will be available via:

    .. code-block:: c

        structObject->key;

    .. note::

        This Type is not complete and should never be used for regular graph operations.

    """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            raise ValueError('Cannot create ParamsType from empty data.')

        for attribute_name in kwargs:
            if re.match('^[A-Za-z_][A-Za-z0-9_]*$', attribute_name) is None:
                raise AttributeError('ParamsType: attribute "%s" should be a valid identifier.' % attribute_name)
            if attribute_name in c_cpp_keywords:
                raise SyntaxError('ParamsType: "%s" is a potential C/C++ keyword and should not be used as attribute name.'
                                  % attribute_name)
            type_instance = kwargs[attribute_name]
            type_name = type_instance.__class__.__name__
            if not isinstance(type_instance, Type):
                raise TypeError('ParamsType: attribute "%s" should inherit from Theano Type, got "%s".'
                                % (attribute_name, type_name))

        self.length = len(kwargs)
        self.fields = tuple(sorted(kwargs.keys()))
        self.types = tuple(kwargs[field] for field in self.fields)
        self.name = self.generate_struct_name()

        self.__const_to_enum = {}
        self.__alias_to_enum = {}
        enum_types = [t for t in self.types if isinstance(t, EnumType)]
        if enum_types:
            # We don't want same enum names in different enum types.
            if sum(len(t) for t in enum_types) != len(set(k for t in enum_types for k in t)):
                raise AttributeError('ParamsType: found different enum types with common constants names.')
            # We don't want same aliases in different enum types.
            if sum(len(t.aliases) for t in enum_types) != len(set(alias for t in enum_types for alias in t.aliases)):
                raise AttributeError('ParamsType: found different enum types with common constants aliases.')
            # We don't want aliases that have same names as some constants.
            all_enums = {e for t in enum_types for e in t}
            all_aliases = {a for t in enum_types for a in t.aliases}
            if [a for a in all_aliases if a in all_enums]:
                raise AttributeError('ParamsType: found aliases that have same names as constants.')
            # We map each enum name to the enum type in which it is defined.
            # We will then use this dict to find enum value when looking for enum name in ParamsType object directly.
            self.__const_to_enum = {enum_name: enum_type for enum_type in enum_types for enum_name in enum_type}
            self.__alias_to_enum = {alias: enum_type for enum_type in enum_types for alias in enum_type.aliases}

    def __setstate__(self, state):
        # NB:
        # I have overridden __getattr__ to make enum constants available through
        # the ParamsType when it contains enum types. To do that, I use some internal
        # attributes: self.__const_to_enum and self.__alias_to_enum. These attributes
        # are normally found by Python without need to call getattr(), but when the
        # ParamsType is unpickled, it seems gettatr() may be called at a point before
        # __const_to_enum or __alias_to_enum are unpickled, so that gettatr() can't find
        # those attributes, and then loop infinitely.
        # For this reason, I must add this trivial implementation of __setstate__()
        # to avoid errors when unpickling.
        self.__dict__.update(state)

    def __getattr__(self, key):
        # Now we can access value of each enum defined inside enum types wrapped into the current ParamsType.
        if key in self.__const_to_enum:
            return self.__const_to_enum[key][key]
        return super(ParamsType, self).__getattr__(self, key)

    def __repr__(self):
        return 'ParamsType<%s>' % ', '.join([('%s:%s' % (self.fields[i], self.types[i])) for i in range(self.length)])

    def __eq__(self, other):
        return (type(self) == type(other) and self.fields == other.fields and self.types == other.types)

    def __hash__(self):
        return hash((type(self),) + self.fields + self.types)

    def generate_struct_name(self):
        # This method tries to generate an unique name for the current instance.
        # This name is intended to be used as struct name in C code and as constant
        # definition to check if a similar ParamsType has already been created
        # (see c_support_code() below).
        fields_string = ','.join(self.fields).encode('utf-8')
        types_string = ','.join(str(t) for t in self.types).encode('utf-8')
        fields_hex = hashlib.sha256(fields_string).hexdigest()
        types_hex = hashlib.sha256(types_string).hexdigest()
        return '_Params_%s_%s' % (fields_hex, types_hex)

    def has_type(self, theano_type):
        """
        Return True if current ParamsType contains the specified Theano type.

        """
        return theano_type in self.types

    def get_type(self, field_name):
        """
        Return the Theano type associated to the given field name
        in the current ParamsType.

        """
        return self.types[self.fields.index(field_name)]

    def get_field(self, theano_type):
        """
        Return the name (string) of the first field associated to
        the given Theano type. Fields are sorted in lexicographic
        order. Raise an exception if this Theano type is not
        in the current ParamsType.

        This method is intended to be used to retrieve a field name
        when we know that current ParamsType contains the given
        Theano type only once.

        """
        return self.fields[self.types.index(theano_type)]

    def get_enum(self, key):
        """
        Look for a constant named ``key`` in the Theano enumeration types
        wrapped into current ParamsType. Return value of the constant found,
        or raise an exception if either the constant is not found or
        current wrapper does not contain any Theano enumeration type.

        **Example**::

            from theano.gof import ParamsType, EnumType, EnumList
            from theano.scalar import Scalar

            wrapper = ParamsType(scalar=Scalar('int32'),
                                 letters=EnumType(A=1, B=2, C=3),
                                 digits=EnumList('ZERO', 'ONE', 'TWO'))
            print(wrapper.get_enum('C'))  # 3
            print(wrapper.get_enum('TWO'))  # 2

            # You can also directly do:
            print(wrapper.C)
            print(wrapper.TWO)

        """
        return self.__const_to_enum[key][key]

    def enum_from_alias(self, alias):
        """
        Look for a constant that has alias ``alias`` in the Theano enumeration types
        wrapped into current ParamsType. Return value of the constant found,
        or raise an exception if either

        1. there is no constant with this alias,
        2. there is no constant which name is this alias, or
        3. current wrapper does not contain any Theano enumeration type.

        **Example**::

            from theano.gof import ParamsType, EnumType, EnumList
            from theano.scalar import Scalar

            wrapper = ParamsType(scalar=Scalar('int32'),
                                 letters=EnumType(A=(1, 'alpha'), B=(2, 'beta'), C=3),
                                 digits=EnumList(('ZERO', 'nothing'), ('ONE', 'unit'), ('TWO', 'couple')))
            print(wrapper.get_enum('C'))  # 3
            print(wrapper.get_enum('TWO'))  # 2
            print(wrapper.enum_from_alias('alpha')) # 1
            print(wrapper.enum_from_alias('nothing')) # 0

            # For the following, alias 'C' is not defined, so the method looks for
            # a constant named 'C', and finds it.
            print(wrapper.enum_from_alias('C')) # 3

        .. note::

            Unlike with constant names, you can **NOT** access constants values directly with aliases through
            ParamsType (ie. you can't write ``wrapper.alpha``). You **must** use ``wrapper.enum_from_alias()``
            method to do that.

        """
        return self.__alias_to_enum[alias].fromalias(alias) if alias in self.__alias_to_enum else self.__const_to_enum[alias][alias]

    def get_params(self, *objects, **kwargs):
        """
        Convenient method to extract fields values from a list of Python objects and key-value args,
        and wrap them into a :class:`Params` object compatible with current ParamsType.

        For each field defined in the current ParamsType, a value for this field
        is looked for in the given objects attributes (looking for attributes with this field name)
        and key-values args (looking for a key equal to this field name), from left to right
        (first object, then, ..., then last object, then key-value args), replacing a previous
        field value found with any value found in next step, so that only the last field value
        found is retained.

        Fields values given in objects and kwargs must be compatible with types
        associated to corresponding fields in current ParamsType.

        **Example**::

            import numpy
            from theano.gof import ParamsType
            from theano.tensor import dmatrix
            from theano.scalar import Scalar

            class MyObject:
                def __init__(self):
                    self.a = 10
                    self.b = numpy.asarray([[1, 2, 3], [4, 5, 6]])

            params_type = ParamsType(a=Scalar('int32'), b=dmatrix, c=Scalar('bool'))

            o = MyObject()
            value_for_c = False

            # Value for c can't be retrieved from o, so we add a value for that field in kwargs.
            params = params_type.get_params(o, c=value_for_c)
            # params.a contains 10
            # params.b contains [[1, 2, 3], [4, 5, 6]]
            # params.c contains value_for_c
            print(params)

        """
        fields_values = dict()
        # We collect fields values from given objects.
        # If a field is present in many objects, only the field in the last object will be retained.
        for obj in objects:
            for field in self.fields:
                try:
                    fields_values[field] = getattr(obj, field)
                except Exception:
                    pass
        # We then collect fields values from given kwargs.
        # A field value in kwargs will replace any previous value collected from objects for this field.
        for field in self.fields:
            if field in kwargs:
                fields_values[field] = kwargs[field]
        # Then we filter the fields values and we create the Params object.
        filtered = {self.fields[i]: self.types[i].filter(fields_values[self.fields[i]], strict=False, allow_downcast=True)
                    for i in range(self.length)}
        return Params(self, **filtered)

    def extended(self, **kwargs):
        """
        Return a copy of current ParamsType
        extended with attributes given in kwargs.
        New attributes must follow same rules as in
        ParamsType constructor.

        """
        self_to_dict = {self.fields[i]: self.types[i] for i in range(self.length)}
        self_to_dict.update(kwargs)
        return ParamsType(**self_to_dict)

    # Returns a Params object with expected attributes or (in strict mode) checks that data has expected attributes.
    def filter(self, data, strict=False, allow_downcast=None):
        if strict and not isinstance(data, Params):
            raise TypeError('%s: strict mode: data should be an instance of Params.' % self)
        # Filter data attributes to check if they respect the ParamsType's contract.
        filtered = {self.fields[i]: self.types[i].filter(getattr(data, self.fields[i]), strict, allow_downcast)
                    for i in range(self.length)}
        return data if (strict or isinstance(data, Params)) else Params(self, **filtered)

    def values_eq(self, a, b):
        return all(self.types[i].values_eq(getattr(a, self.fields[i]), getattr(b, self.fields[i]))
                   for i in range(self.length))

    def values_eq_approx(self, a, b):
        return all(self.types[i].values_eq_approx(getattr(a, self.fields[i]), getattr(b, self.fields[i]))
                   for i in range(self.length))

    def c_compile_args(self, c_compiler):
        c_compile_args_list = []
        for _type in self.types:
            try:
                try:
                    c_compile_args_list.extend(_type.c_compile_args(c_compiler))
                except TypeError:
                    c_compile_args_list.extend(_type.c_compile_args())
            except MethodNotDefined:
                pass
        return c_compile_args_list

    def c_no_compile_args(self, c_compiler):
        c_no_compile_args_list = []
        for _type in self.types:
            try:
                try:
                    c_no_compile_args_list.extend(_type.c_no_compile_args(c_compiler))
                except TypeError:
                    c_no_compile_args_list.extend(_type.c_no_compile_args())
            except MethodNotDefined:
                pass
        return c_no_compile_args_list

    def c_headers(self, c_compiler):
        c_headers_list = []
        for _type in self.types:
            try:
                try:
                    c_headers_list.extend(_type.c_headers(c_compiler))
                except TypeError:
                    c_headers_list.extend(_type.c_headers())
            except MethodNotDefined:
                pass
        return c_headers_list

    def c_libraries(self, c_compiler):
        c_libraries_list = []
        for _type in self.types:
            try:
                try:
                    c_libraries_list.extend(_type.c_libraries(c_compiler))
                except TypeError:
                    c_libraries_list.extend(_type.c_libraries())
            except MethodNotDefined:
                pass
        return c_libraries_list

    def c_header_dirs(self):
        c_header_dirs_list = []
        for _type in self.types:
            try:
                c_header_dirs_list.extend(_type.c_header_dirs())
            except MethodNotDefined:
                pass
        return c_header_dirs_list

    def c_lib_dirs(self):
        c_lib_dirs_list = []
        for _type in self.types:
            try:
                c_lib_dirs_list.extend(_type.c_lib_dirs())
            except MethodNotDefined:
                pass
        return c_lib_dirs_list

    def c_init_code(self):
        c_init_code_list = []
        for _type in self.types:
            try:
                c_init_code_list.extend(_type.c_init_code())
            except MethodNotDefined:
                pass
        return c_init_code_list

    def c_support_code(self):
        sub = {'fail': '{this->setErrorOccurred(); return;}'}
        struct_name = self.name
        struct_name_defined = struct_name.upper()
        c_support_code_set = set()
        c_declare_list = []
        c_init_list = []
        c_cleanup_list = []
        c_extract_list = []
        for attribute_name, type_instance in zip(self.fields, self.types):

            try:
                # c_support_code() may return a code string or a list of code strings.
                support_code = type_instance.c_support_code()
                if not isinstance(support_code, list):
                    support_code = [support_code]
                c_support_code_set.update(support_code)
            except MethodNotDefined:
                pass

            c_declare_list.append(type_instance.c_declare(attribute_name, sub))

            c_init_list.append(type_instance.c_init(attribute_name, sub))

            c_cleanup_list.append(type_instance.c_cleanup(attribute_name, sub))

            c_extract_list.append("""
            void extract_%(attribute_name)s(PyObject* py_%(attribute_name)s) {
                %(extract_code)s
            }
            """ % {
                'attribute_name': attribute_name,
                'extract_code': type_instance.c_extract(attribute_name, sub)
            })

        struct_declare = '\n'.join(c_declare_list)
        struct_init = '\n'.join(c_init_list)
        struct_cleanup = '\n'.join(c_cleanup_list)
        struct_extract = '\n\n'.join(c_extract_list)
        struct_extract_method = """
        void extract(PyObject* object, int field_pos) {
            switch(field_pos) {
                // Extraction cases.
                %s
                // Default case.
                default:
                    PyErr_Format(PyExc_TypeError, "ParamsType: no extraction defined for a field %%d.", field_pos);
                    this->setErrorOccurred();
                    break;
            }
        }
        """ % ('\n'.join(
            [('case %d: extract_%s(object); break;' % (i, self.fields[i])) for i in range(self.length)])
        )
        final_struct_code = """
        /** ParamsType %(struct_name)s **/
        #ifndef %(struct_name_defined)s
        #define %(struct_name_defined)s
        struct %(struct_name)s {
            /* Attributes, */
            int %(struct_name)s_error;
            %(struct_declare)s

            /* Constructor. */
            %(struct_name)s() {
                %(struct_name)s_error = 0;
                %(struct_init)s
            }

            /* Destructor. */
            ~%(struct_name)s() {
                // cleanup() is defined below.
                cleanup();
            }

            /* Cleanup method. */
            void cleanup() {
                %(struct_cleanup)s
            }

            /* Extraction methods. */
            %(struct_extract)s

            /* Extract method. */
            %(struct_extract_method)s

            /* Other methods. */
            void setErrorOccurred() {
                ++%(struct_name)s_error;
            }
            int errorOccurred() {
                return %(struct_name)s_error;
            }
        };
        #endif
        /** End ParamsType %(struct_name)s **/
        """ % dict(struct_name_defined=struct_name_defined, struct_name=struct_name, struct_declare=struct_declare,
                   struct_init=struct_init, struct_cleanup=struct_cleanup, struct_extract=struct_extract,
                   struct_extract_method=struct_extract_method)

        return list(sorted(list(c_support_code_set))) + [final_struct_code]

    def c_code_cache_version(self):
        return ((3,), tuple(t.c_code_cache_version() for t in self.types))

    # As this struct has constructor and destructor, it could be instanciated on stack,
    # but current implementations of C ops will then pass the instance by value at functions,
    # so it's better to work directly with pointers.

    def c_declare(self, name, sub, check_input=True):
        return """
        %(struct_name)s* %(name)s;
        """ % dict(struct_name=self.name, name=name)

    def c_init(self, name, sub):
        # NB: It seems c_init() is not called for an op param.
        # So the real initialization is done at top of c_extract.
        return """
        %(name)s = NULL;
        """ % dict(name=name)

    def c_cleanup(self, name, sub):
        return """
        delete %(name)s;
        %(name)s = NULL;
        """ % dict(name=name)

    def c_extract(self, name, sub, check_input=True):
        return """
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        %(name)s = new %(struct_name)s;

        { // This need a separate namespace for Clinker
        const char* fields[] = {%(fields_list)s};
        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "ParamsType: expected an object, not None.");
            %(fail)s
        }
        for (int i = 0; i < %(length)s; ++i) {
            PyObject* o = PyDict_GetItemString(py_%(name)s, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "ParamsType: missing expected attribute \\"%%s\\" in object.", fields[i]);
                %(fail)s
            }
            %(name)s->extract(o, i);
            if (%(name)s->errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\\nParamsType: error when extracting value for attribute \\"%%s\\".\\n", fields[i]);
                %(fail)s
            }
        }
        }
        """ % dict(name=name, struct_name=self.name, length=self.length, fail=sub['fail'],
                   fields_list='"%s"' % '", "'.join(self.fields))
