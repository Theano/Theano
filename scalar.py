
import numpy

from copy import copy

from gof import ResultBase, GuardedOp, utils


def as_scalar(x, name = None):
    if isinstance(x, float):
        s = Scalar('float64', name = name)
        s.data = x
        return s
    if isinstance(x, Scalar):
        return x


class Scalar(ResultBase):

    def __init__(self, dtype, name=None):
        self.dtype = dtype
        self.constant = False
        ResultBase.__init__(self, role = None, data = None, name = name)

    def __get_constant(self):
        return self._constant

    def __set_constant(self, value):
        if value:
            self.indestructible = True
        self.constant = value

    constant = property(__get_constant, __set_constant)
        
    def validate(self, data):
        py_type = self.py_type()
        if not isinstance(data, py_type):
            raise TypeError("Expected %s instance." % py_type)

    def same_properties(self, other):
        return other.dtype == self.dtype

    def mergeable(self, other):
        return getattr(self, 'constant', False) \
            and getattr(other, 'constant', False) \
            and self.data == other.data
    
    def py_type(self):
        return {'float64': float}[self.dtype]
        
    def c_type(self):
        return {'float64': 'double'}[self.dtype]
        
    def c_from(self):
        return {'float64': 'PyFloat_FromDouble'}[self.dtype]
        
    def c_as(self):
        return {'float64': 'PyFloat_AsDouble'}[self.dtype]

    def c_declare(self):
        return """
        %(dtype)s* %%(name)s;
        typedef %(dtype)s %%(name)s_dtype;
        """ % dict(dtype = self.c_type())

    def c_data_extract(self):
        return """
        %%(name)s = (%(dtype)s)%(conv)s(py_%%(name)s);
        if (!%%(name)s)
            %%(fail)s
        """ % dict(dtype = self.c_type(),
                   conv = self.c_as())
    
    def c_data_sync(self):
        return """
        Py_XDECREF(py_%%(name)s);
        py_%%(name)s = %(conv)s((%(dtype)s)%%(name)s);
        if (!py_%%(name)s)
            py_%%(name)s = Py_None;
        """ % dict(dtype = self.c_type(),
                   conv = self.c_as())

    def c_data_cleanup(self):
        return ""

    def c_headers(self):
        return []

    def c_libraries(self):
        return []



class ScalarMixedOp(GuardedOp):

    nin = -1
    nout = 1
    
    def __init__(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
                    % (self, len(inputs), self.nin)

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




