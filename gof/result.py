
"""
Contains the Result class, which is the base interface for a
value that is the input or the output of an Op.

"""

import unittest

from err import GofError
from utils import AbstractFunctionError


__all__ = ['is_result', 'ResultBase', 'BrokenLink', 'BrokenLinkError' ]


class BrokenLink:
    """
    This is placed as the owner of a Result that was replaced by
    another Result.
    """

    __slots__ = ['owner', 'index']

    def __init__(self, owner, index):
        self.owner = owner
        self.index = index

    def __nonzero__(self):
        return False


class BrokenLinkError(GofError):
    """
    """
    pass


# ResultBase state keywords
class Empty : """Memory has not been allocated"""
class Allocated: """Memory has been allocated, contents are not the owner's output."""
class Computed : """Memory has been allocated, contents are the owner's output."""


############################
# Result
############################

def is_result(obj):
    """Return True iff obj provides the interface of a Result"""
    attr_list = 'owner',
    return all([hasattr(obj, attr) for attr in attr_list])

class ResultBase(object):
    """Base class for storing Op inputs and outputs

    Attributes:
    _role - None or (owner, index) or BrokenLink
    _data - anything
    constant - Boolean
    state - one of (Empty, Allocated, Computed)

    Properties:
    role - (rw)
    owner - (ro)
    index - (ro)
    data - (rw) : calls data_filter when setting
    replaced - (rw) : True iff _role is BrokenLink

    Methods:
    alloc() - create storage in data, suitable for use by C ops. 
                (calls data_alloc)

    Abstract Methods:
    data_filter
    data_alloc


    Notes (from previous implementation):

    A Result instance should be immutable: indeed, if some aspect of a
    Result is changed, operations that use it might suddenly become
    invalid. Instead, a new Result instance should be instanciated
    with the correct properties and the invalidate method should be
    called on the Result which is replaced (this will make its owner a
    BrokenLink instance, which behaves like False in conditional
    expressions).
    
    """
    class BrokenLink:
        """The owner of a Result that was replaced by another Result"""
        __slots__ = ['old_role']
        def __init__(self, role): self.old_role = role
        def __nonzero__(self): return False

    class BrokenLinkError(Exception):
        """Exception thrown when an owner is a BrokenLink"""

    class AbstractFunction(Exception):
        """Exception thrown when an abstract function is called"""

    __slots__ = ['_role', 'constant', '_data', 'state']

    def __init__(self, role=None, data=None, constant=False):
        self._role = role
        self.constant = constant
        if data is None: #None is not filtered
            self._data = None
            self.state = Empty
        else:
            try:
                self._data = self.data_filter(data)
            except ResultBase.AbstractFunction:
                self._data = data
            self.state = Computed

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
        if self.replaced: raise ResultBase.BrokenLinkError()
        return self._role[0]

    owner = property(__get_owner, 
            doc = "Op of which this Result is an output, or None if role is None")

    #
    # index
    #

    def __get_index(self):
        if self._role is None: return None
        if self.replaced: raise ResultBase.BrokenLinkError()
        return self._role[1]

    index = property(__get_index,
                doc = "position of self in owner's outputs, or None if role is None")


    # 
    # data
    # 

    def __get_data(self):
        return self._data

    def __set_data(self, data):
        if self.replaced: raise ResultBase.BrokenLinkError()
        if self.constant: raise Exception('cannot set constant ResultBase')
        try:
            self._data = self.data_filter(data)
        except ResultBase.AbstractFunction: #use default behaviour
            self._data = data
        self.state = Computed

    data = property(__get_data, __set_data,
            doc = "The storage associated with this result")

    def data_filter(self, data):
        """(abstract) Return an appropriate _data based on data.

        If a subclass overrides this function, then that overriding
        implementation will be used in __set_data to map the argument to
        self._data.  This gives a subclass the opportunity to ensure that
        the contents of self._data remain sensible.
        
        """
        raise ResultBase.AbstractFunction()

    #
    # alloc
    #

    def alloc(self):
        """Create self.data from data_alloc, and set state to Allocated"""
        self.data = self.data_alloc()  #might raise exception
        self.state = Allocated

    def data_alloc(self):
        """(abstract) Return an appropriate _data based on self.

        If a subclass overrides this function, then that overriding
        implementation will be used in alloc() to produce a data object.
        
        """
        raise ResultBase.AbstractFunction()


    #
    # replaced
    #

    def __get_replaced(self):
        return isinstance(self._role, ResultBase.BrokenLink)

    def __set_replaced(self, replace):
        if replace == self.replaced: return
        if replace:
            self._role = ResultBase.BrokenLink(self._role)
        else:
            self._role = self._role.old_role

    replaced = property(__get_replaced, __set_replaced, doc = "has this Result been replaced?")



    #################
    # NumpyR Compatibility
    #
    up_to_date = property(lambda self: True)
    def refresh(self): pass
    def set_owner(self, owner, idx):
        self.role = (owner, idx)
    def set_value(self, value):
        self.data = value #may raise exception

class _test_ResultBase(unittest.TestCase):
    def test_0(self):
        r = ResultBase()


if __name__ == '__main__':
    unittest.main()

