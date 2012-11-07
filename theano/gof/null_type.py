from theano.gof.type import Type


class NullType(Type):
    """

    A type that allows no values. Used to represent expressions
    that are undefined, either because they do not exist mathematically
    or because the code to generate the expression has not been
    implemented yet.

    """

    def __init__(self, why_null='(no explanation given)'):
        """
            why_null: A string explaining why this variable
                    can't take on any values
        """
        self.why_null = why_null

    def filter(self, data, strict=False, allow_downcast=None):
        raise ValueError("No values may be assigned to a NullType")

    def filter_variable(self, other):
        raise ValueError("No values may be assigned to a NullType")

    def may_share_memory(a, b):
        return False

    def values_eq(a, b, force_same_dtype=True):
        raise ValueError("NullType has no values to compare")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self, other):
        return hash(type(self))
