"""A simple class to store L{numpy.ndarray} data """

from gof import Result, Op, utils, AbstractFunctionError
import numpy
from copy import copy


###########################
# BaseTensor Class
###########################

class BaseTensor(Result):
    """
    L{Result} to store L{numpy.ndarray} or equivalent via .data
    
    @type _dtype: numpy dtype string such as 'int64' or 'float64' (among others)
    @type _broadcastable: tuple or list or array of boolean values, whose length
      is the number of dimensions of the contained L{ndarray}.
    @ivar _broadcastable: Each element of the broadcastable vector tells us
      something about the corresponding dimension:
        - False means the dimension can be anything.
        - True means  the dimension must be 1. Also, this dimension will be considered
          for L{broadcasting}, as described and implemented in Numpy.

    Properties:
    dtype - read-only access to _dtype, which should not be changed
    broadcastable - read-only access to _broadcastable, which should not be changed

    This class does not implement python operators and has no dependencies
    on the L{Op}s that use it.

    @todo At some point we should document a glossary, such as terms like
    broadcasting and shape.
    """

    def __init__(self, dtype, broadcastable, name=None):
        """Initialize a L{BaseTensor}

        @note: This does not actually allocate any data.
        """

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
        Result.__init__(self, role=None, name=name)
        self._dtype = str(dtype)
        self.dtype_specs() # this is just for error checking
        self._broadcastable = tuple(broadcastable)

    ######################
    # Result interface
    ######################

    # 
    # filter
    #
    def filter(self, arr):
        """Cast to an L{numpy.ndarray} and ensure arr has correct rank and shape."""
        if not (isinstance(arr, numpy.ndarray) \
                and arr.dtype==self.dtype):
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
        L{self.dtype}.  It is for use in C code generation.
        """
        #TODO: add more type correspondances for e.g. int32, int64, float32,
        #complex64, etc.
        try:
            return {'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                    'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                    'int8': (int, 'npy_int8', 'NPY_INT8'),
                    'int16': (int, 'npy_int16', 'NPY_INT16'),
                    'int32': (int, 'npy_int32', 'NPY_INT32'),
                    'int64': (int, 'npy_int64', 'NPY_INT64'),
                    'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    #
    # Description for constant folding
    #
    def desc(self):
        """
        Returns a hashable description of this L{BaseTensor}.
        """
        if self.data is not None:
            return (BaseTensor, self.dtype, self.broadcastable, self.data.data[:])
        else:
            return (BaseTensor, self.dtype, self.broadcastable, None)
            
    #
    # C codegen stubs
    #
    def c_declare(self, name, sub):
        return """
        PyArrayObject* %(name)s;
        int type_num_%(name)s;
        typedef %(dtype)s dtype_%(name)s;
        """ % dict(sub, name = name, dtype = self.dtype_specs()[1])

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        type_num_%(name)s = %(type_num)s;
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_extract(self, name, sub):
        return """
        %(name)s = NULL;
        type_num_%(name)s = %(type_num)s;
        if (py_%(name)s == Py_None) {
            // We can either fail here or set %(name)s to NULL and rely on Ops using
            // tensors to handle the NULL case, but if they fail to do so they'll end up
            // with nasty segfaults, so this is public service.
            PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
            %(fail)s
            //%(name)s = NULL;
        }
        else if (!PyArray_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_ValueError, "expected an ndarray");
            %(fail)s
        }
        else if (((PyArrayObject*)py_%(name)s)->descr->type_num != %(type_num)s) {
            PyErr_SetString(PyExc_ValueError, "expected %(type_num)s");
            %(fail)s
        }
        else {
            %(name)s = (PyArrayObject*)(py_%(name)s);
            Py_XINCREF(%(name)s);
        }
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_cleanup(self, name, sub):
        return """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
        }
        """ % locals()
    
    def c_sync(self, name, sub):
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
        """ % locals()

    def c_headers(self):
        return []

    def c_libraries(self):
        return []

    def c_support_code(cls):
        template = """
        struct theano_complex%(nbits)s : public npy_complex%(nbits)s
        {
            typedef theano_complex%(nbits)s complex_type;
            typedef npy_float%(half_nbits)s scalar_type;

            complex_type operator +(complex_type y) {
                complex_type ret;
                ret.real = this->real + y.real;
                ret.imag = this->imag + y.imag;
                return ret;
            }
            complex_type operator -(complex_type y) {
                complex_type ret;
                ret.real = this->real - y.real;
                ret.imag = this->imag - y.imag;
                return ret;
            }
            complex_type operator *(complex_type y) {
                complex_type ret;
                ret.real = this->real * y.real - this->imag * y.imag;
                ret.imag = this->real * y.imag + this->imag * y.real;
                return ret;
            }
            complex_type operator /(complex_type y) {
                complex_type ret;
                scalar_type y_norm_square = y.real * y.real + y.imag * y.imag;
                ret.real = (this->real * y.real + this->imag * y.imag) / y_norm_square;
                ret.imag = (this->imag * y.real - this->real * y.imag) / y_norm_square;
                return ret;
            }
        };
        """
        return template % dict(nbits = 64, half_nbits = 32) + template % dict(nbits = 128, half_nbits = 64)
        # todo: use C templating


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
        cpy = self.__class__(self.dtype, self.broadcastable, self.name)
        if transfer_data:
            cpy.data = copy(self.data)
        return cpy




class BaseTensorOp(Op):
    """
    A basic L{Op} subclass that can be used to make L{Op}s that operate on L{Tensor}s.
    It is not mandatory to inherit from this class, but it is practical.

    @ivar nin: number of inputs
    @ivar nout: number of outputs
    @ivar out_tensor_class: L{BaseTensor} subclass used to instantiate the outputs

     - input_wrapper: returns a L{Tensor} from its argument
     - propagate_dtype: returns a list of dtypes corresponding to the
     output dtypes from a list of input dtypes (if an input is not a
     L{Tensor}, the passed value will be None)
     - propagate_broadcastable: returns a list of tuples corresponding
     to the output broadcastable flags from the input broadcastable flags
     (if an input is not a L{Tensor}, the passed value will be None).
    """
    
    nin = -1 # nin == -1 means: arbitrary number of inputs
    nout = 1
    
    out_tensor_class = BaseTensor

    @classmethod
    def input_wrapper(cls, obj):
        """
        Returns a L{Result} from an arbitrary-typed input, if possible.
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



