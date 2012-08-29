from type import Type
from graph import Variable

class NullType(Type):
    def __init__(self, why_null = '(no explanation given)'):
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



class NullVariable(Variable):

    pass
