
import numpy
from copy import copy

from gof import ResultBase
from gof import Op


class NumpyR(ResultBase):

    def __init__(self, dtype, nd, name=None):
        self.nd = nd
        self.dtype = dtype
        ResultBase.__init__(self, role = None, data = None, name = name)

    def validate(self, data):
        if not isinstance(data, numpy.ndarray):
            raise TypeError("Expected ndarray instance.")
        elif not len(data.shape) == self.nd:
            raise TypeError("Expected ndarray with %i dimensions." % self.nd)
        elif not str(data.dtype) == self.dtype:
            raise TypeError("Expected ndarray with data type %i." % self.dtype)

    
#     def to_c_type(self, dtype):
#         if dtype == "float64":
#             return "double"
#         else:
#             raise TypeError("Cannot translate dtype to C.")
        
    def c_declare(self):
        return """
        PyArrayObject* %%(name)s;
        typedef %(dtype)s %%(name)s_dtype;
        """ % dict(dtype = self.to_c_type(self.dtype))

    def c_data_extract(self):
        return """
        if (py_%(name)s == Py_None)
            %(name)s = NULL;
        else
            %(name)s = (PyArrayObject*)(py_%(name)s);
        """

    def c_data_cleanup(self):
        return ""
    
    def c_data_sync(self):
        return """
        if (!%(name)s) {
            Py_XDECREF(py_%(name));
            py_%(name)s = Py_None;
        }
        else if ((void*)py_%(name)s != (void*)%(name)s) {
            Py_XDECREF(py_%(name));
            py_%(name)s = (PyObject*)%(name)s;
        }
        """

    def c_headers(self):
        return []

    def c_libraries(self):
        return []

    def __copy__(self):
        cpy = self.__class__(self.dtype, self.nd, self.name)
        cpy.data = copy(self.data)
        return cpy



def TheanoOp(Op):

    nin = -1
    nout = 1
    
    def __init__(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
                    % (self, len(inputs), self.nin)

        i_nds = [getattr(input, 'nd', None) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]

        o_nds = self.propagate_nd(*i_nds)
        o_dtypes = self.propagate_dtypes(*i_dtypes)
        
        return [NumpyR(nd, dtype) for nd, dtype in zip(o_nds, o_dtypes)]

    def propagate_nds(self, *inputs):
        raise AbstractFunctionError()

    def propagate_dtypes(self, *inputs):
        raise AbstractFunctionError()
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])




