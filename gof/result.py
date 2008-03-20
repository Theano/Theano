"""
Contains the Result class, which is the base interface for a
value that is the input or the output of an Op.

"""

import copy
from utils import AbstractFunctionError


__all__ = ['ResultBase',
           'PythonResult',
           'StateError',
           'Empty',
           'Allocated',
           'Computed',
           ]


### CLEANUP - DO WE REALLY EVEN THE STATE ANYMORE? ###

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

    Abstract Methods:
    data_filter
    """

    __slots__ = ['_role', '_data', 'state', '_name']

    def __init__(self, role=None, name=None):
        self._role = role
        self._data = [None]
        self.state = Empty
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
        """
        Filters the data provided and sets the result in the storage.
        """
        if data is self._data[0]:
            return
        if data is None:
            self._data[0] = None
            self.state = Empty
            return
        try:
            data = self.filter(data)
        except AbstractFunctionError:
            pass
        self._data[0] = data
        self.state = Computed
        
    data = property(__get_data, __set_data,
                    doc = "The storage associated with this result")

    def filter(self, data):
        """
        Raise an exception if the data is not of an acceptable type.

        If a subclass overrides this function, __set_data will use it
        to check that the argument can be used properly. This gives a
        subclass the opportunity to ensure that the contents of
        self._data remain sensible.

        Returns data or an appropriately wrapped data.
        """
        raise AbstractFunctionError()


    #
    # C code generators
    #

    def c_is_simple(self):
        """
        A hint to tell the compiler that this type is a builtin C
        type or a small struct and that its memory footprint is
        negligible.
        """
        return False
    
    def c_declare(self):
        """
        Declares variables that will be instantiated by c_data_extract.
        """
        raise AbstractFunctionError()

    def c_extract(self):
        """
        The code returned from this function must be templated using
        "%(name)s", representing the name that the caller wants to
        call this Result. The Python object self.data is in a
        variable called "py_%(name)s" and this code must set the
        variables declared by c_declare to something representative
        of py_%(name)s. If the data is improper, set an appropriate
        exception and insert "%(fail)s".
        """
        raise AbstractFunctionError()
    
    def c_cleanup(self):
        """
        This returns C code that should deallocate whatever
        c_data_extract allocated or decrease the reference counts. Do
        not decrease py_%(name)s's reference count.
        """
        raise AbstractFunctionError()

    def c_sync(self):
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
        Return a list of compile args recommended to manipulate this Result.
        """
        raise AbstractFunctionError()

    def c_headers(self):
        """
        Return a list of header files that must be included from C to manipulate
        this Result.
        """
        raise AbstractFunctionError()

    def c_libraries(self):
        """
        Return a list of libraries to link against to manipulate this Result.
        """
        raise AbstractFunctionError()

    def c_support_code(self):
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
        """Return bool; True iff all properties are equal (ignores contents, role)"""
        raise AbstractFunction()


    def __copy__(self):
        """Create a new instance of self.__class__ with role None, independent data"""
        raise AbstractFunctionError()


class PythonResult(ResultBase):
    """
    Represents a generic Python object. The object is available
    through %(name)s.
    """
    
    def c_declare(self):
        return """
        PyObject* %(name)s;
        """

    def c_extract(self):
        return """
        Py_XINCREF(py_%(name)s);
        %(name)s = py_%(name)s;
        """
    
    def c_cleanup(self):
        return """
        Py_XDECREF(%(name)s);
        """

    def c_sync(self):
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = %(name)s;
        Py_XINCREF(py_%(name)s);
        """
    
    def same_properties(self, other):
        return False

    def __copy__(self):
        rval = PythonResult(None, self.name)
        rval.data = copy.copy(self.data)
        return rval

    


