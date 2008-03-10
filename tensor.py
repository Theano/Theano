
import numpy
from copy import copy

from gof import ResultBase
from gof import Op


def tensor(data, name = None):
    return Tensor(data.dtype, [0]*len(data.shape), data, name)

def _broadcastable_pattern(pattern):
    def factory(data = None, name = None):
        if data: assert len(data.shape) == len(pattern)
        return Tensor(data.dtype, pattern, data, name)

matrix = _broadcastable_pattern([0, 0])
row = _broadcastable_pattern([1, 0])
col = _broadcastable_pattern([0, 1])


class Tensor(ResultBase):

    def __init__(self, dtype, broadcastable, data=None, name=None):
        self.broadcastable = broadcastable
        self.dtype = dtype
        ResultBase.__init__(self, role = None, data = None, name = name)

    def filter(self, data):
        arr = numpy.asarray(data, dtype = self.dtype)
        for b, s in zip(self.broadcastable, arr.shape):
            assert not b or s == 1
        return arr
        
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
        """
        Returns a copy of this Tensor. If there is data stored inside it, it is also copied.
        """
        cpy = self.__class__(self.dtype, self.broadcastable, None, self.name)
        cpy.data = copy(self.data)
        return cpy



def TensorOp(Op):

    nin = -1
    nout = 1
    
    def __init__(self, *inputs):

        def wrap_as_tensor(x):
            if isinstance(x, Tensor):
                return x
            else:
                return Tensor(x)

        inputs = map(wrap_as_tensor, inputs)
        
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
                    % (self, len(inputs), self.nin)

        i_broadcastables = [getattr(input, 'broadcastable', None) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]

        o_broadcastables = utils.from_return_values(self.propagate_broadcastable(*i_broadcastables))
        o_dtypes = utils.from_return_values(self.propagate_dtype(*i_dtypes))

        self.inputs = inputs
        self.outputs = [Tensor(dtype, broadcastable) for broadcastable, dtype in zip(o_broadcastables, o_dtypes)]

    def propagate_broadcastable(self, *inputs):
        raise AbstractFunctionError()

    def propagate_dtype(self, *inputs):
        raise AbstractFunctionError()
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])




