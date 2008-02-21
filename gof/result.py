
"""
Contains the Result class, which is the base interface for a
value that is the input or the output of an Op.

"""


from err import GofError
from utils import AbstractFunctionError


__all__ = ['is_result', 'Result', 'BrokenLink', 'BrokenLinkError']


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



############################
# Result
############################

def is_result(obj):
    """Return True iff obj provides the interface of a Result"""
    attr_list = 'owner',
    return all([hasattr(obj, attr) for attr in attr_list])

class Result(object):
    """Storage node for data in a graph of Op instances.

    Attributes:
    owner - represents the Op which computes this Result. Contains either None
        or an instance of Op.
    index - the index of this Result in owner.outputs.

    Methods:
    - 

    Notes:

    Result has no __init__ or __new__ routine. It is the Op's
    responsibility to set the owner field of its results.

    The Result class is abstract. It must be subclassed to support the
    types of data needed for computation.

    A Result instance should be immutable: indeed, if some aspect of a
    Result is changed, operations that use it might suddenly become
    invalid. Instead, a new Result instance should be instanciated
    with the correct properties and the invalidate method should be
    called on the Result which is replaced (this will make its owner a
    BrokenLink instance, which behaves like False in conditional
    expressions).
    """
    
    __slots__ = ['_owner', '_index']
    
    def get_owner(self):
        if not hasattr(self, '_owner'):
            self._owner = None
        return self._owner

    owner = property(get_owner, 
            doc = "The Op of which this Result is an output or None if there is no such Op.")

    def set_owner(self, owner, index):
        if self.owner is not None:
            if self.owner is not owner:
                raise ValueError("Result %s already has an owner." % self)
            elif self.index != index:
                raise ValueError("Result %s was already mapped to a different index." % self)
        self._owner = owner
        self._index = index

    def invalidate(self):
        if self.owner is None:
            raise Exception("Cannot invalidate a Result instance with no owner.")
        elif not isinstance(self.owner, BrokenLink):
            self._owner = BrokenLink(self._owner, self._index)
            del self._index

    def revalidate(self):
        if isinstance(self.owner, BrokenLink):
            owner, index = self._owner.owner, self._owner.index
            self._owner = owner
            self._index = index

    def perform(self):
        """Calls self.owner.perform() if self.owner exists.

        This is a mutually recursive function with gof.op.Op

        """
        if self.owner:
            self.owner.perform()
    

#     def extract(self):
#         """
#         Returns a representation of this datum for use in Op.impl.
#         Successive calls to extract should always return the same object.
#         """
#         raise NotImplementedError

#     def sync(self):
#         """
#         After calling Op.impl, synchronizes the Result instance with the
#         new contents of the storage. This might usually not be necessary.
#         """
#         raise NotImplementedError

#     def c_libs(self):
#         """
#         Returns a list of libraries that must be included to work with
#         this Result.
#         """
#         raise NotImplementedError

#     def c_imports(self):
#         """
#         Returns a list of strings representing headers to import when
#         building a C interface that uses this Result.
#         """
#         raise NotImplementedError

#     def c_declare(self):
#         """
#         Returns code which declares and initializes a C variable in
#         which this Result can be held.
#         """
#         raise NotImplementedError

#     def pyo_to_c(self):
#         raise NotImplementedError

#     def c_to_pyo(self):
#         raise NotImplementedError




############################
# Utilities
############################

# class SelfContainedResult(Result):
#     """
#     This represents a Result which acts as its own data container. It
#     is recommended to subclass this if you wish to be able to use the
#     Result in normal computations as well as working with a graph
#     representation.
#     """
    
# #     def extract(self):
# #         """Returns self."""
# #         return self

# #     def sync(self):
# #         """Does nothing."""
# #         pass



# class HolderResult(Result):
#     """
#     HolderResult adds a 'data' slot which is meant to contain the
#     object used by the Op implementation. It is recommended to subclass
#     this if you want to be able to use the exact same object at
#     different points in a computation.
#     """

#     __slots__ = ['data']

# #     def extract(self):
# #         """Returns self.data."""
# #         return self.data

# #     def sync(self):
# #         """
# #         Does nothing. Override if you have additional fields or
# #         functionality in your subclass which need to be computed from
# #         the data.
# #         """
# #         pass






