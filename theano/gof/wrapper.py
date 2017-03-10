"""
Module for wrapping many Theano variables into one C struct for op params.

This module contains two classes:

 - :class:`Wrapper`: main class to define the op params type.
 - :class:`Wrap`: internal convenient class to create an object that is compatible with Wrapper-defined op params.

Example of usage
----------------

Importation:

.. code-block:: python

    from theano.gof import Wrapper

In an op you create:

.. code-block:: python

    from theano.tensor import TensorType, dmatrix
    params_type = Wrapper(attr1=TensorType('int32', (False, False)), attr2=dmatrix)

If your op contains props ``attr1`` *and* ``attr2``, the default ``op.get_params()`` implementation
will automatically try to look for it and generate an appropriate wrapped struct.
Props must be compatible with the corresponding types defined into the Wrapper
(we will try to convert and downcast if needed).

.. code-block:: python

    __props__ = ('attr1', 'attr2')
    def __init__(value_attr1, value_attr2):
        self.attr1 = value_attr1
        self.attr2 = value_attr2

In ``perform()`` implementation (with params named ``param``):

.. code-block:: python

    var1 = param.attr1
    var2 = param.attr2

In ``c_code()`` implementation (with ``param = sub['params']``):

.. code-block:: c

    PyArrayObject* attr1 = param->attr1;
    PyArrayObject* attr2 = param->attr2;
    /* You won't need to free them or whatever else. */


See :class:`QuadraticOpFunc` and :class:`QuadraticCOpFunc` in ``theano/gof/tests/test_wrapper.py``
for complete working examples.

"""

from __future__ import absolute_import, print_function, division
import re
import hashlib
from theano.gof.utils import MethodNotDefined, c_cpp_keywords
from theano.gof import Type


class Wrap(dict):
    """
    Internal convenient class to wrap many Python objects into one
    (this class is not safe as the hash method does not check if values are effectively hashable).

    **Example:**

    .. code-block:: python

        from theano.gof import Wrapper, Wrap
        from theano.scalar import Scalar
        # You must create a Wrapper first:
        wp = Wrapper(attr1=Scalar('int32'), key2=Scalar('float32'), field3=Scalar('int64'))
        # Then you can create a Wrap with the wrapper defined above and values for attributes.
        w = Wrap(wp, attr1=1, key2=2.0, field3=3)
        print(w.attr1, w.key2, w.field3)
        d = dict(attr1=1, key2=2, field3=-1)
        w2 = Wrap(wp, **d)
        print(w2.attr1, w2.key2, w2.field3)

    """

    def __init__(self, wrapper, **kwargs):
        if not isinstance(wrapper, Wrapper):
            raise TypeError('Wrap: 1st constructor argument should be a Wrapper.')
        for field in wrapper.fields:
            if field not in kwargs:
                raise TypeError('Wrap: Wrapper attribute "%s" not in Wrap args.' % field)
        super(Wrap, self).__init__(**kwargs)
        self.__dict__.update(wrapper=wrapper)

    def __repr__(self):
        return 'Wrap(%s)' % ', '.join([('%s:%s' % (k, type(self[k]))) for k in sorted(self.keys())])

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('Wrap: attribute "%s" does not exist.' % key)
        return self[key]

    def __setattr__(self, key, value):
        raise NotImplementedError('Wrap is immutable')

    def __setitem__(self, key, value):
        raise NotImplementedError('Wrap is immutable')

    def __delitem__(self, key):
        raise NotImplementedError('Wrap is immutable')

    def __hash__(self):
        return hash((type(self), self.wrapper) + tuple(
            # NB: Wrapped data should have been already filtered.
            self.wrapper.types[i].make_constant(self[self.wrapper.fields[i]]).signature()
            for i in range(self.wrapper.length)
        ))

    def __eq__(self, other):
        return (type(self) == type(other) and self.wrapper == other.wrapper and all(
            # NB: Wrapped data should have been already filtered.
            self.wrapper.types[i].values_eq(self[self.wrapper.fields[i]], other[self.wrapper.fields[i]])
            for i in range(self.wrapper.length)
        ))

    def __ne__(self, other):
        return not self.__eq__(other)


class Wrapper(Type):
    """
    This class can create a struct of Theano types (like TensorType, GpuArrayType, etc.)
    to be used as a convenience op parameter wrapping many data.

    Wrapper constructor takes key-value args.
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
            raise ValueError('Cannot create Wrapper from empty data.')

        for attribute_name in kwargs:
            if re.match('^[A-Za-z_][A-Za-z0-9_]*$', attribute_name) is None:
                raise AttributeError('Wrapper: attribute "%s" should be a valid identifier.' % attribute_name)
            if attribute_name in c_cpp_keywords:
                raise SyntaxError('Wrapper: "%s" is a potential C/C++ keyword and should not be used as attribute name.'
                                  % attribute_name)
            type_instance = kwargs[attribute_name]
            type_name = type_instance.__class__.__name__
            if not isinstance(type_instance, Type):
                raise TypeError('Wrapper: attribute "%s" should inherit from Theano Type, got "%s".'
                                % (attribute_name, type_name))

        self.length = len(kwargs)
        self.fields = tuple(sorted(kwargs.keys()))
        self.types = tuple(kwargs[field] for field in self.fields)
        self.name = self.generate_struct_name()

    def __repr__(self):
        return 'Wrapper<%s>' % ', '.join([('%s:%s' % (self.fields[i], self.types[i])) for i in range(self.length)])

    def __eq__(self, other):
        return (type(self) == type(other) and self.fields == other.fields and self.types == other.types)

    def __hash__(self):
        return hash((type(self),) + self.fields + self.types)

    def generate_struct_name(self):
        # This method tries to generate an unique name for the current instance.
        # This name is intended to be used as struct name in C code and as constant
        # definition to check if a similar Wrapper has already been created
        # (see c_support_code() below).
        fields_string = ','.join(self.fields).encode('utf-8')
        types_string = ','.join(str(t) for t in self.types).encode('utf-8')
        fields_hex = hashlib.md5(fields_string).hexdigest()
        types_hex = hashlib.md5(types_string).hexdigest()
        return '_wrapper_%s_%s' % (fields_hex, types_hex)

    def wrap_data(self, data, strict, allow_downcast):
        # Try to wrap data. Raise an exception if data does not respect the Wrapper's contract.
        wrap_instance = dict()
        for i in range(self.length):
            wrap_instance[self.fields[i]] = self.types[i].filter(getattr(data, self.fields[i]), strict, allow_downcast)
        return data if strict else Wrap(self, **wrap_instance)

    # Returns a wrapped object with expected attributes or (in strict mode) checks that data has expected attributes.
    def filter(self, data, strict=False, allow_downcast=None):
        if strict and not isinstance(data, Wrap):
            raise TypeError('%s: strict mode: data should be an instance of Wrap.' % self)
        return self.wrap_data(data, strict, allow_downcast)

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
        sub = {'fail': '{this->setErrorOccurred(); this->cleanup(); return;}'}
        struct_name = self.name
        struct_name_defined = struct_name.upper()
        struct_declare = ''
        struct_init = ''
        struct_cleanup = ''
        struct_extract = ''
        c_declare_list = []
        c_init_list = []
        c_cleanup_list = []
        c_extract_list = []
        for attribute_name, type_instance in zip(self.fields, self.types):

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
                    PyErr_Format(PyExc_TypeError, "Wrapper: no extraction defined for a field %%d.", field_pos);
                    this->setErrorOccurred();
                    this->cleanup();
                    break;
            }
        }
        """ % ('\n'.join(
            [('case %d: extract_%s(object); break;' % (i, self.fields[i])) for i in range(self.length)])
        )
        return """
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
        """ % locals()

    def c_code_cache_version(self):
        return (1, 5)

    # As this struct has constructor and destructor, it could be instanciated on stack,
    # but current implementations of C ops will then pass the instance by value at functions,
    # so it's better to work directly with pointers.

    def c_declare(self, name, sub, check_input=True):
        struct_name = self.name
        return """
        %(struct_name)s* %(name)s;
        """ % locals()

    def c_init(self, name, sub):
        # NB: It seems c_init() is not called for an op param.
        # So the real initialization is done at top of c_extract.
        return """
        %(nams)s = NULL;
        """ % locals()

    def c_cleanup(self, name, sub):
        return """
        delete %(name)s;
        %(name)s = NULL;
        """ % locals()

    def c_extract(self, name, sub, check_input=True):
        struct_name = self.name
        fail = sub['fail']
        length = self.length
        fields_list = '"%s"' % '", "'.join(self.fields)
        return """
        /* Seems c_init() is not called for a op param. So I call `new` here. */
        %(name)s = new %(struct_name)s;

        const char* fields[] = {%(fields_list)s};
        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "Wrapper: expected an object, not None.");
            %(fail)s
        }
        for (int i = 0; i < %(length)s; ++i) {
            PyObject* o = PyDict_GetItemString(py_%(name)s, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "Wrapper: missing expected attribute \\"%%s\\" in object.", fields[i]);
                %(fail)s
            }
            %(name)s->extract(o, i);
            if (%(name)s->errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\\nWrapper: error when extracting value for attribute \\"%%s\\".\\n", fields[i]);
                %(fail)s
            }
        }
        """ % locals()
