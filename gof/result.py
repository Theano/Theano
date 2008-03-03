
"""
Contains the Result class, which is the base interface for a
value that is the input or the output of an Op.

"""

from utils import AbstractFunctionError


__all__ = ['ResultBase',
           'StateError',
           'Empty',
           'Allocated',
           'Computed',
           ]


class StateError(Exception):
    """The state of the Result is a problem"""


# ResultBase state keywords
class Empty : """Memory has not been allocated"""
class Allocated: """Memory has been allocated, contents are not the owner's output."""
class Computed : """Memory has been allocated, contents are the owner's output."""


############################
# Result
############################

class ResultBase(object):
    """Base class for storing Op inputs and outputs

    Attributes:
    _role - None or (owner, index) #or BrokenLink
    _data - anything
    state - one of (Empty, Allocated, Computed)
    name - string

    Properties:
    role - (rw)
    owner - (ro)
    index - (ro)
    data - (rw) : calls data_filter when setting

    Methods:
    alloc() - create storage in data, suitable for use by C ops. 
                (calls data_alloc)

    Abstract Methods:
    data_filter
    data_alloc
    
    """

    __slots__ = ['_role', '_data', 'state', '_name']

    def __init__(self, role=None, data=None, name=None):
        self._role = role
        self._data = [None]
        self.state = Empty
        self.__set_data(data)
        self.name = name

    #
    # role 
    #

    def __get_role(self):
        return self._role

    def __set_role(self, role):
        owner, index = role
        if self._role is not None:
            # this is either an error or a no-op
            _owner, _index = self._role
            if _owner is not owner:
                raise ValueError("Result %s already has an owner." % self)
            if _index != index:
                raise ValueError("Result %s was already mapped to a different index." % self)
            return # because _owner is owner and _index == index
        self._role = role

    role = property(__get_role, __set_role)

    #
    # owner
    #

    def __get_owner(self):
        if self._role is None: return None
        return self._role[0]

    owner = property(__get_owner, 
            doc = "Op of which this Result is an output, or None if role is None")

    #
    # index
    #

    def __get_index(self):
        if self._role is None: return None
        return self._role[1]

    index = property(__get_index,
                doc = "position of self in owner's outputs, or None if role is None")


    # 
    # data
    # 

    def __get_data(self):
        return self._data[0]

    def __set_data(self, data):
        if data is self._data[0]:
            return
        if data is None:
            self._data[0] = None
            self.state = Empty
            return
        try:
            self.validate(data)
        except AbstractFunctionError:
            pass
        self._data[0] = data
        self.state = Computed
        
    data = property(__get_data, __set_data,
                    doc = "The storage associated with this result")

    def validate(self, data):
        """(abstract) Raise an exception if the data is not of an
        acceptable type.

        If a subclass overrides this function, __set_data will use
        it to check that the argument can be used properly. This gives
        a subclass the opportunity to ensure that the contents of
        self._data remain sensible.
        
        """
        raise AbstractFunctionError()


#     #
#     # alloc
#     #

#     def alloc(self):
#         """Create self.data from data_alloc, and set state to Allocated

#         Graph routines like the linker will ask Ops to allocate outputs.  The
#         Ops, in turn, usually call this function.  Results that are involved in
#         destroy maps and view maps are exceptions to the usual case.
#         """
#         self.data = self.data_alloc()  #might raise exception
#         self.state = Allocated

#     def data_alloc(self):
#         """(abstract) Return an appropriate _data based on self.

#         If a subclass overrides this function, then that overriding
#         implementation will be used in alloc() to produce a data object.
        
#         """
#         raise AbstractFunctionError()


    #
    # C code generators
    #

    def c_declare(self):
        """
        Declares variables that will be instantiated by c_data_extract.
        """
        raise AbstractFunctionError()

    def c_extract(self):
        get_from_list = """
        PyObject* py_%(name)s = PyList_GET_ITEM(%(name)s_storage, 0);
        Py_XINCREF(py_%(name)s);
        """
        return get_from_list + self.c_data_extract()

    def c_data_extract(self):
        """
#         The code returned from this function must be templated using
#         "%(name)s", representing the name that the caller wants to
#         call this Result. The Python object self.data is in a
#         variable called "py_%(name)s" and this code must declare a
#         variable named "%(name)s" of type "%(type)s" where "%(type)s"
#         will be replaced by the return value of
#         self.c_type(). Additional variables and typedefs may not be
#         produced. If the data is improper, set an appropriate error
#         message and insert "%(fail)s".
        """
        raise AbstractFunctionError()

    def c_cleanup(self):
        decref = """
        Py_XDECREF(py_%(name)s);
        """
        return self.c_data_cleanup() + decref
    
    def c_data_cleanup(self):
        """
        This returns C code that should deallocate whatever
        c_data_extract allocated or decrease the reference counts. Do
        not decrease py_%(name)s's reference count.

        Note: EITHER c_cleanup OR c_sync will be called.
        """
        raise AbstractFunctionError()
    
    def c_sync(self):
        set_in_list = """
        PyList_SET_ITEM(%(name)s_storage, 0, py_%(name)s);
        Py_XDECREF(py_%(name)s);
        """
        return self.c_data_sync() + set_in_list

    def c_data_sync(self):
        """
        The code returned from this function must be templated using "%(name)s",
        representing the name that the caller wants to call this Result.
        The returned code may set "py_%(name)s" to a PyObject* and that PyObject*
        will be accessible from Python via result.data. Do not forget to adjust
        reference counts if "py_%(name)s" is changed from its original value!
        """
        raise AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this Result.
        """
        return []

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this Result.
        """
        return []

    def c_support(self):
        """
        Return utility code for use by this Result or Ops manipulating this
        Result.
        """
        raise AbstractFunctionError()
    
    #
    # name
    #

    def __get_name(self):
        if self._name:
            return self._name
        elif self._role:
            return "%s.%i" % (self.owner.__class__, self.owner.outputs.index(self))
        else:
            return None
    def __set_name(self, name):
        if name is not None and not isinstance(name, str):
            raise TypeError("Name is expected to be a string, or None.")
        self._name = name

    name = property(__get_name, __set_name,
                    doc = "Name of the Result.")


    #
    # String representation
    #

    def __str__(self):
        name = self.name
        if name:
            if self.state is Computed:
                return name + ":" + str(self.data)
            else:
                return name
        elif self.state is Computed:
            return str(self.data)
        else:
            return "<?>"

    def __repr__(self):
        return self.name or "<?>"

    
    #
    # same properties
    #

    def same_properties(self, other):
        raise AbstractFunction()


    
    #################
    # NumpyR Compatibility
    #
    up_to_date = property(lambda self: True)
    def refresh(self): pass
    def set_owner(self, owner, idx):
        self.role = (owner, idx)
    def set_value(self, value):
        self.data = value #may raise exception


#     def c_data_extract(self):
#         return """
#         PyArrayObject* %%(name)s;
#         if (py_%%(name)s == Py_None)
#             %%(name)s = NULL;
#         else
#             %%(name)s = (PyArrayObject*)(py_%%(name)s);
#         typedef %(dtype)s %%(name)s_dtype;
#         """ % dict(dtype = self.dtype)

#     def c_data_sync(self):
#         return """
#         if (!%(name)s) {
#             Py_XDECREF(py_%(name));
#             py_%(name)s = Py_None;
#         }
#         else if ((void*)py_%(name)s != (void*)%(name)s) {
#             Py_XDECREF(py_%(name));
#             py_%(name)s = (PyObject*)%(name)s;
#         }
#         """

