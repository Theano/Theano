
import numpy
from copy import copy
import inspect
from gof import ResultBase, Op, utils



def tensor(data, name = None):
    data = numpy.asarray(data)
    return Tensor(data.dtype, [0]*len(data.shape), data, name)

def _broadcastable_pattern(pattern):
    def factory(data = None, name = None):
        if data: assert len(data.shape) == len(pattern)
        return Tensor(data.dtype, pattern, data, name)
    return factory

matrix = _broadcastable_pattern([0, 0])
row = _broadcastable_pattern([1, 0])
col = _broadcastable_pattern([0, 1])


class Tensor(ResultBase):

    def __init__(self, dtype=None, broadcastable=None, data=None, name=None, constant=False):
        if dtype is None or broadcastable is None:
            if data is None:
                raise TypeError("Provide non-None data to complete the dtype and broadcastable flags.")
            data = numpy.asarray(data)
            dtype = data.dtype
            if constant:
                broadcastable = [1*(x == 1) for x in data.shape]
            else:
                broadcastable = [0] * len(data.shape)
        self.broadcastable = broadcastable
        self.dtype = str(dtype)
        self.constant = constant
        ResultBase.__init__(self, role = None, data = data, name = name)

    def __get_constant(self):
        return self._constant

    def __set_constant(self, value):
        if value:
            self.indestructible = True
        self._constant = value

    constant = property(__get_constant, __set_constant)

    def filter(self, data):
        arr = numpy.asarray(data, dtype = self.dtype)
        for b, s in zip(self.broadcastable, arr.shape):
            assert not b or s == 1
        return arr

    def dtype_specs(self):
        return {'float64': (float, 'double')}[self.dtype]
            
    def c_declare(self):
        return """
        PyArrayObject* %%(name)s;
        typedef %(dtype)s %%(name)s_dtype;
        """ % dict(dtype = self.dtype_specs()[1])

    def c_init(self):
        return """
        %(name)s = NULL;
        """

    def c_extract(self):
        return """
        if (py_%(name)s == Py_None) {
            %(name)s = NULL;
        }
        else if (!PyArray_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_ValueError, "expected an ndarray");
            %(fail)s
        }
        else {
            %(name)s = (PyArrayObject*)(py_%(name)s);
            Py_XINCREF(%(name)s);
        }
        """

    def c_cleanup(self):
        return """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
            for (int i = 0; i < PyArray_REFCOUNT(%(name)s); i++) {
                printf("X");
            }
            printf("Y\\n");
        }
        """
    
    def c_sync(self):
        return """
        if (!%(name)s) {
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
        }
        else if ((void*)py_%(name)s != (void*)%(name)s) {
            Py_XDECREF(py_%(name)s);
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


