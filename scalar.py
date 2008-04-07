
import numpy

from copy import copy
import inspect

from gof import ResultBase, GuardedOp, utils


def as_scalar(x, name = None):
    if isinstance(x, float):
        s = Scalar('float64', name = name)
        s.data = x
        return s
    if isinstance(x, Scalar):
        return x


class Scalar(ResultBase):

    def __init__(self, dtype, name = None):
        ResultBase.__init__(self, role = None, name = name)
        self.dtype = dtype

    def __get_constant(self):
        return self._constant

    def __set_constant(self, value):
        if value:
            self.indestructible = True
        self._constant = value

    constant = property(__get_constant, __set_constant)
        
    def filter(self, data):
        py_type = self.dtype_specs()[0]
        return py_type(data)

    def same_properties(self, other):
        return other.dtype == self.dtype

#     def mergeable(self, other):
#         return getattr(self, 'constant', False) \
#             and getattr(other, 'constant', False) \
#             and self.data == other.data

    def dtype_specs(self):
        return {'float64': (float, 'double', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble')}[self.dtype]

    def c_declare(self, name, sub):
        return """
        %(dtype)s %(name)s;
        typedef %(dtype)s %(name)s_dtype;
        """ % dict(name = name, dtype = self.dtype_specs()[1])

    def c_init(self, name, sub):
        return """
        %(name)s = 0;
        """ % locals()
    
    def c_extract(self, name, sub):
        specs = self.dtype_specs()
        return """
        if (!%(check)s(py_%(name)s))
            %(fail)s
        %(name)s = (%(dtype)s)%(conv)s(py_%(name)s);
        """ % dict(sub,
                   name = name,
                   dtype = specs[1],
                   check = specs[2],
                   conv = specs[3])
    
    def c_sync(self, name, sub):
        specs = self.dtype_specs()
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = %(conv)s((%(dtype)s)%(name)s);
        if (!py_%(name)s)
            py_%(name)s = Py_None;
        """ % dict(name = name,
                   dtype = specs[1],
                   conv = specs[4])

    def c_cleanup(self, name, sub):
        return ""

    def __copy__(self):
        """
        Return a copy of this instance (with its own attributes)
        """
        cpy = self.__class__(self.dtype, self.name)
        cpy.data = self.data
        return cpy



class ScalarMixedOp(GuardedOp):

    nin = -1
    nout = 1
    
    def __init__(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)" \
                                    % (self.__class__.__name__, len(inputs), self.nin))

        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]
        o_dtypes = utils.from_return_values(self.propagate_dtypes(*i_dtypes))

        self.inputs = inputs
        self.outputs = [Scalar(dtype) for dtype in o_dtypes]

    def propagate_dtypes(self, *inputs):
        raise AbstractFunctionError()
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def grad(self, inputs, output_gradients):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])

#     def c_var_names(self):
#         (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
#         inames = utils.from_return_values(inames)
#         onames = utils.from_return_values(onames)
#         return [inames, onames]

#     def c_code(self):
#         return self.c_impl(self.inputs, self.outputs)
        

def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)


class PureScalarOp(ScalarMixedOp):

    cast_method = lambda self, *args: upcast(*args)
    
    def propagate_dtypes(self, *i_dtypes):
        for dtype in i_dtypes:
            if dtype is None:
                raise TypeError("Expected a Scalar.")
        return self.cast_method(*i_dtypes)


class UnaryScalarOp(PureScalarOp):
    nin = 1

class BinaryScalarOp(PureScalarOp):
    nin = 2




