"""A simple class to store ndarray data """

from gof import ResultBase, Op, utils
import numpy
from copy import copy


###########################
# BaseTensor Class
###########################

class BaseTensor(ResultBase):
    """ResultBase to store numpy.ndarray or equivalent via .data
    
    Attributes:
    _dtype - numpy dtype string such as 'int64' or 'float64' (among others)
    _broadcastable - tuple of ints in  (0,1) saying which dimensions of this
        tensor are guaranteed to be 1, and up for broadcasting

    Properties:
    dtype - read-only access to _dtype, which should not be changed
    broadcastable - read-only access to _broadcastable, which should not be changed

    This class does not implement python operators and has no dependencies
    on the Ops that use it.
    """

    def __init__(self, dtype, broadcastable, role=None, name=None):
        """Initialize a Tensor"""

        # data is not given here. This may seem a bit strange, but when data was
        # an argument, it made sense to use *either* the given dtype,
        # broadcastable, or override them from the fields of data. This makes
        # the function ugly, especially because it isn't obvious how to set
        # broadcastable from data.  
        #
        # The only clean option I could think of, when passing a data arg was to 
        # require the broadcastable field to be given.  Since broadcastable is
        # the argument that is awkward to construct, I decided to put all this
        # into the tensor(data,...) function below, which is like a second
        # constructor that works with an ndarray.
        ResultBase.__init__(self, role=role, name=name)
        self._dtype = str(dtype)
        self.dtype_specs() # this is just for error checking
        self._broadcastable = tuple(broadcastable)

    ######################
    # ResultBase interface
    ######################

    # 
    # filter
    #
    def filter(self, arr):
        """cast to an ndarray and ensure arr has correct rank, shape"""
        if not isinstance(arr, numpy.ndarray):
            arr = numpy.asarray(arr, dtype = self.dtype)
        if len(self.broadcastable) != len(arr.shape):
            raise ValueError(BaseTensor.filter.E_rank,
                    self.broadcastable,
                    arr.shape,
                    self.owner)
        for b, s in zip(self.broadcastable, arr.shape):
            if b and (s != 1):
                raise ValueError(BaseTensor.filter.E_shape)
        return arr
    # these strings are here so that tests can use them
    filter.E_rank = 'wrong rank'
    filter.E_shape = 'non-unit size on broadcastable dimension'

    #
    # type information
    #
    def dtype_specs(self):
        """Return python - C type correspondance tuple for self.data

        Return a tuple (python type, c type, numpy typenum) that corresponds to
        self.dtype.  It is for use in C code generation.
        """
        #TODO: add more type correspondances for e.g. int32, int64, float32,
        #complex64, etc.
        try:
            return {'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                    'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                    'int8': (int, 'npy_int8', 'NPY_INT8'),
                    'int16': (int, 'npy_int16', 'NPY_INT16'),
                    'int32': (int, 'npy_int32', 'NPY_INT32'),
                    'int64': (int, 'npy_int64', 'NPY_INT64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for BaseTensor: %s" % self.dtype)

    #
    # Hash for constant folding
    #
    def hash(self):
        if self.data:
            return (BaseTensor, self.dtype, self.broadcastable, self.data.data[:])
        else:
            return (BaseTensor, self.dtype, self.broadcastable, None)
            
    #
    # C codegen stubs
    #
    def c_declare(self):
        return """
        PyArrayObject* %%(name)s;
        int type_num_%%(name)s;
        typedef %(dtype)s dtype_%%(name)s;
        """ % dict(dtype = self.dtype_specs()[1])

    def c_init(self):
        return """
        %%(name)s = NULL;
        type_num_%%(name)s = %(type_num)s;
        """ % dict(type_num = self.dtype_specs()[2])

    def c_extract(self):
        return """
        %%(name)s = NULL;
        type_num_%%(name)s = %(type_num)s;
        if (py_%%(name)s == Py_None) {
            // We can either fail here or set %%(name)s to NULL and rely on Ops using
            // tensors to handle the NULL case, but if they fail to do so they'll end up
            // with nasty segfaults, so this is public service.
            PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
            %%(fail)s
            //%%(name)s = NULL;
        }
        else if (!PyArray_Check(py_%%(name)s)) {
            PyErr_SetString(PyExc_ValueError, "expected an ndarray");
            %%(fail)s
        }
        else if (((PyArrayObject*)py_%%(name)s)->descr->type_num != %(type_num)s) {
            PyErr_SetString(PyExc_ValueError, "expected %(type_num)s");
            %%(fail)s
        }
        else {
            %%(name)s = (PyArrayObject*)(py_%%(name)s);
            Py_XINCREF(%%(name)s);
        }
        """ % dict(type_num = self.dtype_specs()[2])

    def c_cleanup(self):
        return """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
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
            Py_XINCREF(py_%(name)s);
        }
        """

    def c_headers(self):
        return []

    def c_libraries(self):
        return []

    def c_support_code(cls):
        operator_template = """
            me operator %(op)s(me y) {
                me ret;
                ret.real = this->real %(op)s y.real;
                ret.imag = this->imag %(op)s y.imag;
                return ret;
            }
        """        
        template = """
        struct theano_complex%(nbits)s : public npy_complex%(nbits)s
        {
            typedef theano_complex%(nbits)s me;
            typedef npy_complex%(nbits)s base;

            %(operators)s
        };
        """
        d = dict(operators = "\n".join([operator_template % dict(op=op) for op in ["+", "-", "*", "/"]]))
        return template % dict(d, nbits = 64) + template % dict(d, nbits = 128)


    ############################
    # Tensor specific attributes
    ############################

    dtype = property(lambda self: self._dtype)
    broadcastable = property(lambda self: self._broadcastable)

    ############################
    # Cloning facilities
    ############################

    def __copy__(self):
        return self.clone(True)
    
    def clone(self, transfer_data = False):
        """Return a copy of this instance (with its own attributes)
        
        If transfer_data is True, a copy of self.data is assigned to the copy's
        data property, otherwise the copy's data is left as None.
        """
        cpy = self.__class__(self.dtype, self.broadcastable, None, self.name)
        if transfer_data:
            cpy.data = copy(self.data)
        return cpy




class BaseTensorOp(Op):
    """
    A basic Op subclass that can be used to make Ops that operate on Tensors.
    It is not mandatory to inherit from this class, but it is practical.

    BasicTensorOp is parametrized as follows:
     * nin: number of inputs
     * nout: number of outputs
     * out_tensor_class: BaseTensor subclass used to instantiate the outputs
     * input_wrapper: returns a Tensor from its argument
     * propagate_dtype: returns a list of dtypes corresponding to the
           output dtypes from a list of input dtypes (if an input is
           not a Tensor, the passed value will be None)
     * propagate_broadcastable: returns a list of tuples corresponding to
           the output broadcastable flags from the input broadcastable
           flags  (if an input is not a Tensor, the passed value will be
           None).
    """
    
    nin = -1 # nin == -1 means: arbitrary number of inputs
    nout = 1
    
    out_tensor_class = BaseTensor

    @classmethod
    def input_wrapper(cls, obj):
        """
        Returns a Result from an arbitrary-typed input, if possible.
        """
        if isinstance(obj, BaseResult):
            return obj
        else:
            raise TypeError("Expected a Result instance.")

    def __init__(self, *inputs):
        inputs = map(self.input_wrapper, inputs)
        
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
                    % (self, len(inputs), self.nin)

        i_broadcastables = [getattr(input, 'broadcastable', None) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]

        o_broadcastables = utils.from_return_values(self.propagate_broadcastable(*i_broadcastables))
        o_dtypes = utils.from_return_values(self.propagate_dtype(*i_dtypes))

        self.inputs = inputs
        self.outputs = [self.out_tensor_class(dtype, broadcastable) for broadcastable, dtype in zip(o_broadcastables, o_dtypes)]

    def propagate_broadcastable(self, *inputs):
        raise AbstractFunctionError()
    
    def propagate_dtype(self, *i_dtypes):
        rval = set([dtype for dtype in i_dtypes if dtype is not None])
        if len(rval) == 0:
            raise ValueError("Cannot infer the dtypes of the outputs with no Tensor inputs.")
        elif len(rval) > 1:
            raise ValueError("The dtypes of all inputs should be identical.")
        return [rval.pop()] * self.nout



