
from core import Numpy2, omega_op


def input(x):
    #static member initialization
    if not hasattr(input, 'float_dtype'):
        input.float_dtype = 'float64'
        input.int_dtype = 'int64'
        input.NN = Numpy2

    if isinstance(x, numpy.ndarray):
        #return NumpyR(x)
        return input.NN(data=x)
    elif isinstance(x, int):
        z = numpy.zeros((), dtype = input.int_dtype)
        z += x
        return input.NN(data=z)
    elif isinstance(x, float):
        z = numpy.zeros((), dtype = input.float_dtype)
        z += x
        return input.NN(data=z)
    elif is_result(x):
        raise TypeError("%s is already a result." % x)
    else:
        return ResultBase(data=x)

def wrap(x):
    if isinstance(x, Numpy2):
        return x
    #elif isinstance(x, NumpyR):
        #return x
    elif is_result(x):
        return x
    elif isinstance(x, omega_op):
        return x.out
    else:
        return literal(x)

def literal(x):
    """Return a ResultValue instance wrapping a literal."""
    def _hashable(x):
        try:
            x in {}
            return True
        except TypeError: # x is unhashable
            return False

    #static member initialization
    if not hasattr(literal, 'hdb'): 
        literal.hdb = {}
        literal.udb = {}

    if _hashable(x):
        db = literal.hdb
        key = (type(x),x)
    else:
        db = literal.udb
        key = (id(x),)

    if key in db:
        return db[key]
    else:
        rval = input(x)
        rval.constant = True
        db[key] = rval
        return rval


