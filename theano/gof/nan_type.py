from type import Type
from graph import Variable

class NaNType(Type):
    def __init__(self, why_nan = '(no explanation given)'):
        """
            why_nan: A string explaining why this variable is NaN
        """
        self.why_nan = why_nan

    def filter(self, data, strict=False, allow_downcast=None):
       raise
    def filter_variable(self, other):
       raise
    def may_share_memory(a, b):
       return False
    def values_eq(a, b, force_same_dtype=True):
       raise



class NaNVariable(Variable):

    pass
