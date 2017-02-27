"""
Module for wrapping many Theano variables into one struct for op params.

This module contains two classes:
    - Wrapper: class to define the op params type.
    - Wrap: internal convenient class to create an object that is compatible with a Wrapper-defined op params.

Example of usage:

    Importation:

        from theano.gof.wrapper import Wrapper

    In an op you create:

        params_type = Wrapper(attr1=TensorType('int32', (False, False)), attr2=TensorType('float64', (True, False)))

    If your op contains props `attr1` AND `attr2`, the op.get_params() method will
    automatically try to look for it and generate an appropriate wrapped struct.
    The props must be able to pass the filtering (not strict, downcasting allowed)
    of corresponding types defined into Wrapper.

        __props__ = ('attr1', 'attr2')
        def __init__(value_attr1, value_attr2):
            self.attr1 = value_attr1
            self.attr2 = value_attr2

    In perform() implementation (with params named `param`):

        var1 = param.attr1
        var2 = param.attr2

    In c_code() implementation (with `param = sub['params']`):

        PyArrayObject* attr1 = param.attr1;
        PyArrayObject* attr2 = param.attr2;
        /* You won't need to free them or whatever else. */


See `theano/gof/tests/test_wrapper.py` for a complete working example.

"""

from __future__ import absolute_import, print_function, division
import re
import hashlib
import numpy
from theano.gof.utils import MethodNotDefined
from theano.gof import Type
from theano.tensor.utils import hash_from_ndarray

# NB: Maybe we should check if an attribute name is a C/C++ keyword, and raise an error if so.
# These are some lists of C/C++ keywords:
# http://fr.cppreference.com/w/cpp/keyword
# http://fr.cppreference.com/w/c/keyword


class Wrap(object):
    """
    Internal convenient class to wrap many Python objects into one
    (this class is not safe as the hash method does not check if values are effectively hashable).

    Example:
        >>> w = Wrap(attr1=1, attr2=2.0, attri='3')
        >>> print(w.attr1, w.attr2, w.attri)
        >>> d = dict(a=1, b=2, c='test')
        >>> w2 = Wrap(**d)
        >>> print(w2.a, w2.b, w2.c)

    """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            raise TypeError('Wrap: cannot wrap empty data.')
        # We want to use only the params provided in kwargs to hash the object,
        # so I prefer to put them into a separate attribute (self.data) instead
        # of directly in self.__dict__, to avoid confusion with builtin fields.
        super(Wrap, self).__setattr__('data', kwargs)

    def __repr__(self):
        return 'Wrap(%s)' % ', '.join([('%s:%s' % (k, type(self.data[k]))) for k in sorted(self.data.keys())])

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError('Wrap: attribute "%s" does not exist.' % key)
        return self.data[key]

    def __setattr__(self, key, value):
        if key not in self.data:
            raise AttributeError('Wrap: attribute "%s" does not exist.' % key)
        self.data[key] = value

    def __hash__(self):
        keys = sorted(self.data.keys())
        types = []
        attributes = []
        for k in keys:
            types += (type(self.data[k]),)
            if isinstance(self.data[k], numpy.ndarray):
                # Note: hash_from_ndarray returns a string, so the hash is not yet complete
                # (__hash__ must return an integer).
                attributes += (hash_from_ndarray(self.data[k]),)
            else:
                # No checking, data should be hashable.
                attributes += (self.data[k],)
        return hash((type(self),) + tuple(keys) + tuple(types) + tuple(attributes))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for k in self.data:
            if (k not in other.data or
                    not isinstance(self.data[k], type(other.data[k])) or
                    not isinstance(other.data[k], type(self.data[k]))):
                return False
            if isinstance(self.data[k], numpy.ndarray):
                if not numpy.allclose(self.data[k], other.data[k]):
                    return False
            elif self.data[k] != other.data[k]:
                return False
        return True


class Wrapper(Type):
    """
    This class can create a struct of Theano types (like TensorType, GpuArrayType, etc.)
    to be used as a convenience op parameter wrapping many data.

    Wrapper constructor takes key-value args.
    Key will be the name of the attribute in the struct.
    Value is the Theano type of this attribute, ie. an instance of (a subclass of) Type
    (eg. TensorType('int64', (False,))).

    In a Python code any attribute named `key` will be available via:
        structObject.key

    In a C code, attributes created to represent an instance of the type associated to `key` will be available via:
        structObject.key
        structObject.dtype_key # e.g. from TensorType C code.
        structObject.other_attribute_named_from_key
        etc.

    """

    def __init__(self, **kwargs):
        if len(kwargs) == 0:
            raise ValueError('Cannot create Wrapper from empty data.')

        for attribute_name in kwargs:
            if re.match('^[A-Za-z_][A-Za-z0-9_]*$', attribute_name) is None:
                raise SyntaxError('Wrapper: attribute "%s" should be a valid identifier.' % attribute_name)
            type_instance = kwargs[attribute_name]
            type_name = type_instance.__class__.__name__
            if not isinstance(type_instance, Type):
                raise TypeError('Wrapper: attribute "%s" should inherit from theano Type, got "%s".'
                                % (attribute_name, type_name))

        self.length = len(kwargs)
        self.fields = tuple(sorted(kwargs.keys()))
        self.types = tuple(kwargs[field] for field in self.fields)
        self.name = self.generate_struct_name()

    def __repr__(self):
        return 'Wrapper<%s>' % ', '.join([('%s:%s' % (self.fields[i], self.types[i])) for i in range(self.length)])

    def __eq__(self, other):
        # To be checked.
        return (type(self) == type(other) and self.fields == other.fields and self.types == other.types)

    def __hash__(self):
        return hash((type(self),) + self.fields + self.types)

    def generate_struct_name(self):
        """"
        This method tries to generate an unique name for the current instance.
        This name is intended to be used as struct name in C code and as constant
        definition to check if a similar Wrapper has already been created
        (see c_support_code() below).
        """
        fields_string = ','.join(self.fields).encode('utf-8')
        types_string = ','.join(str(t) for t in self.types).encode('utf-8')
        fields_hex = hashlib.md5(fields_string).hexdigest()
        types_hex = hashlib.md5(types_string).hexdigest()
        return '_wrapper_%s_%s' % (fields_hex, types_hex)

    def check_that_values_are_compatible(self, data, strict, allow_downcast):
        wrap_instance = dict()
        for i in range(self.length):
            wrap_instance[self.fields[i]] = self.types[i].filter(getattr(data, self.fields[i]), strict, allow_downcast)
        return data if strict else Wrap(**wrap_instance)

    # Returns a wrapped object with expected attributes or (in strict mode) checks that data has expected attributes.
    def filter(self, data, strict=False, allow_downcast=None):
        if isinstance(data, dict):
            if strict:
                raise TypeError('%s: strict mode: data should be an object, not a dict.' % self)
            data = Wrap(**data)
        return self.check_that_values_are_compatible(data, strict, allow_downcast)

    def values_eq(self, a, b):
        # We check that a and b have expected attributes and strict values.
        a = self.filter(a, strict=True)
        b = self.filter(b, strict=True)
        # Then we compare.
        for i in range(self.length):
            if not self.types[i].values_eq(getattr(a, self.fields[i]), getattr(b, self.fields[i])):
                return False
        return True

    def values_eq_approx(self, a, b):
        # We check, wrap and round a and b if necessary.
        a = self.filter(a, strict=False, allow_downcast=True)
        b = self.filter(b, strict=False, allow_downcast=True)
        # Then we compare.
        for i in range(self.length):
            if not self.types[i].values_eq_approx(getattr(a, self.fields[i]), getattr(b, self.fields[i])):
                return False
        return True

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
        return (1, 3)

    def c_declare(self, name, sub, check_input=True):
        struct_name = self.name
        return """
        %(struct_name)s %(name)s;
        """ % locals()

    # c_init() and c_cleanup() are useless if we create the struct
    # on stack, as struct class has constructor and destructor.

    def c_init(self, name, sub):
        return ""

    def c_cleanup(self, name, sub):
        return ""

    def c_extract(self, name, sub, check_input=True):
        fail = sub['fail']
        length = self.length
        fields_list = '"%s"' % '", "'.join(self.fields)
        return """
        const char* fields[] = {%(fields_list)s};
        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "Wrapper: expected an object, not None.");
            %(fail)s
        }
        for (int i = 0; i < %(length)s; ++i) {
            PyObject* o = PyObject_GetAttrString(py_%(name)s, fields[i]);
            if (o == NULL) {
                PyErr_Format(PyExc_TypeError, "Wrapper: missing expected attribute \\"%%s\\" in object.", fields[i]);
                %(fail)s
            }
            %(name)s.extract(o, i);
            if (%(name)s.errorOccurred()) {
                /* The extract code from attribute type should have already raised a Python exception,
                 * so we just print the attribute name in stderr. */
                fprintf(stderr, "\\nWrapper: error when extracting value for attribute \\"%%s\\".\\n", fields[i]);
                %(fail)s
            }
        }
        """ % locals()
