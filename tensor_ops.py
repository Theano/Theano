
from tensor import *
from gof import Op, utils


def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)

class TensorOp(Op):

    nin = -1
    nout = 1

    cast_method = lambda self, *args: upcast(*args)
    
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
    
    def propagate_dtype(self, *i_dtypes):
        for dtype in i_dtypes:
            if dtype is None:
                raise TypeError("Expected a Tensor.")
        return self.cast_method(*i_dtypes)
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])
    
    def c_var_names(self):
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        inames = utils.from_return_values(inames)
        onames = utils.from_return_values(onames)
        return [inames, onames]
    
    def c_code(self):
        return self.c_impl(self.inputs, self.outputs)

    def c_impl(self, inputs, outputs):
        raise AbstractFunctionError()

        

class UnaryTensorOp(TensorOp):
    nin = 1

class BinaryTensorOp(TensorOp):
    nin = 2



class Transpose(UnaryTensorOp):

    def propagate_broadcastable(self, x):
        x2 = copy(x)
        x2.reverse()
        return x2

    def impl(self, x):
        return x.T

    def c_impl(self, x, z):
        return """
        PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
        if (PyArray_REFCOUNT(transposed) == 1) {
            printf("lala\\n");
        }
        if (%(z)s) {
            Py_XDECREF(%(z)s);
        }
        %(z)s = transposed;
        """


from gof import modes
modes.make_constructors(globals())

