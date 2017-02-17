from __future__ import absolute_import, print_function, division
from theano.gof.type import Type


class NullType(Type):
    """
    A type that allows no values.

    Used to represent expressions
    that are undefined, either because they do not exist mathematically
    or because the code to generate the expression has not been
    implemented yet.

    Parameters
    ----------
    why_null : str
        A string explaining why this variable can't take on any values.

    """

    def __init__(self, why_null='(no explanation given)'):
        self.why_null = why_null

    def filter(self, data, strict=False, allow_downcast=None):
        raise ValueError("No values may be assigned to a NullType")

    def filter_variable(self, other, allow_convert=True):
        raise ValueError("No values may be assigned to a NullType")

    def may_share_memory(a, b):
        return False

    def values_eq(a, b, force_same_dtype=True):
        raise ValueError("NullType has no values to compare")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'NullType'
null_type = NullType()
