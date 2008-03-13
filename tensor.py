"""A ResultBase to store numpy.ndarray with basic accompanying Ops"""

import numpy
from copy import copy
import inspect
from gof import ResultBase, Op, utils, Destroyer, Viewer

###########################
# Tensor Class
###########################

class Tensor(ResultBase):
    """ResultBase to store numpy.ndarray or equivalent via .data
    
    Attributes:
    _dtype - numpy dtype string such as 'int64' or 'float64' (among others)
    _broadcastable - tuple of ints in  (0,1) saying which dimensions of this
        tensor are guaranteed to be 1, and up for broadcasting

    Properties:
    dtype - read-only access to _dtype, which should not be changed
    broadcastable - read-only access to _broadcastable, which should not be changed

    Operators:
    - most numeric operators are overloaded to return Ops that *would* perform
      the corresponding calculation

    
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
        self._broadcastable = tuple(broadcastable)

    ######################
    # ResultBase interface
    ######################

    # 
    # filter
    #
    def filter(self, arr):
        if not isinstance(arr, numpy.ndarray):
            arr = numpy.asarray(arr, dtype = self.dtype)
        if len(self.broadcastable) != len(arr.shape):
            raise ValueError(Tensor.filter.E_rank)
        for b, s in zip(self.broadcastable, arr.shape):
            if b and (s != 1):
                raise ValueError(Tensor.filter.E_shape)
        return arr
    # these strings are here so that tests can use them
    filter.E_rank = 'wrong rank'
    filter.E_shape = 'non-unit size on broadcastable dimension'

    #
    # type information  : Olivier what does this mean?
    #
    def dtype_specs(self):
        return {'float64': (float, 'double')}[self.dtype]
            
    #
    # C codegen stubs
    #
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
        %(name)s = NULL;
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
            Py_XINCREF(py_%(name)s);
        }
        """

    def c_headers(self):
        return []

    def c_libraries(self):
        return []


    ############################
    # Tensor specific attributes
    #
    ############################

    dtype = property(lambda self: self._dtype)
    broadcastable = property(lambda self: self._broadcastable)

    # STDLIB
    def __copy__(self):
        """
        Returns a copy of this Tensor. If there is data stored inside it, it is also copied.
        """
        cpy = self.__class__(self.dtype, self.broadcastable, None, self.name)
        cpy.data = copy(self.data)
        return cpy

    #UNARY
    def __abs__(self): return Abs(self).out
    def __neg__(self): return Neg(self).out

    #CASTS
    def __int__(self): return AsInt(self).out
    def __float__(self): return AsInt(self).out
    def __complex__(self): return AsComplex(self).out

    #COMPARISONS
    def __lt__(self,other): return lt(self, other)
    def __le__(self,other): return le(self, other)
    def __gt__(self,other): return gt(self, other)
    def __ge__(self,other): return ge(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): return add(self,other)
    def __sub__(self,other): return sub(self,other)
    def __mul__(self,other): return mul(self,other)
    def __div__(self,other): return div(self,other)
    def __pow__(self,other): return pow(self,other)

    #ARITHMETIC - INPLACE
    def __iadd__(self,other): return add_inplace(self,other)
    def __isub__(self,other): return sub_inplace(self,other)
    def __imul__(self,other): return mul_inplace(self,other)
    def __idiv__(self,other): return div_inplace(self,other)
    def __ipow__(self,other): return pow_inplace(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rpow__(self,other): return pow(other,self)

    #TRANSPOSE
    def __get_T(self):
        return tensor_copy(transpose(self))
    T = property(__get_T)

    #SLICING
    def __getitem__(self, key): raise NotImplementedError()
    def __getslice__(self, key): raise NotImplementedError()

# alternate Tensor constructor
def tensor(data, broadcastable=None, role=None, name=None):
    """Return a Tensor containing given data"""
    data = numpy.asarray(data)
    if broadcastable is None:
        broadcastable = [s==1 for s in data.shape]
    elif broadcastable in [0, 1]:
        broadcastable = [broadcastable] *  len(data.shape)
    rval = Tensor(data.dtype, broadcastable, role, name)
    rval.data = data # will raise if broadcastable was mis-specified
    return rval


############################
# Supporting Ops
############################

def _scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    """a decorator for operators before broadcasting works properly"""
    def f(x, y):
        def as_tensor(obj):
            if isinstance(obj, Tensor):
                return obj
            else:
                return tensor(obj)
        x, y = as_tensor(x), as_tensor(y)
        if 0 not in y.broadcastable:
            return scalar_f(x, y)
        if 0 not in x.broadcastable:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f

class _Op(Op):
    """A convenient base for the ops in this file"""
    nin = -1
    nout = 1

    def __init__(self, *inputs):

        def as_tensor(obj):
            if isinstance(obj, Tensor):
                return obj
            else:
                return tensor(obj)
        inputs = map(as_tensor, inputs)
        
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
        def upcast(dtype, *dtypes):
            z = numpy.zeros((), dtype = dtype)
            for dtype in dtypes:
                z = z + numpy.zeros((), dtype = dtype)
            return str(z.dtype)
        for dtype in i_dtypes:
            if dtype is None:
                raise TypeError("Expected a Tensor.")
        rval = upcast(*i_dtypes)
        return rval
    
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

class _Unary:
    nin = 1

class _Binary:
    nin = 2

def _assert_same_shapes(x, *rest):
    """Ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)"""
    shape = x.shape
    for other in rest:
        if other.shape != shape:
            raise _assert_same_shapes.E_shape
_assert_same_shapes.E_shape = ValueError("The dimensions of the inputs do not match.")

def _assert_tensor_scalar(x, a):
    """ensure that the second input is a scalar"""
    if numpy.product(a.shape) != 1:
        raise ValueError("The second argument must be a scalar.")

class _Elemwise(_Op):

    @staticmethod
    def extract_name(name):
        if name.endswith("_i"):
            return name[:-2]
        else:
            return name
    
    @staticmethod
    def is_loop_var(name):
        return name.endswith("_i")
    
    def c_var_names(self):
        cls = self.__class__
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
        spec = ([cls.extract_name(name) for name in inames],
                [cls.extract_name(name) for name in onames])
        return spec

    def loop_variables(self):
        cls = self.__class__
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
        return ([cls.extract_name(name) for name in inames if cls.is_loop_var(name)],
                [cls.extract_name(name) for name in onames if cls.is_loop_var(name)])
    
    def propagate_broadcastable(self, *inputs):
        inames, onames = self.c_var_names()
        iloop, oloop = self.loop_variables()
        if oloop != onames:
            raise Exception(\
                    "Cannot infer broadcastable for non-loop variable(s) %s" \
                    % set(onames).difference(oloop), self.__class__)
        all_bcast = [broadcastable for broadcastable, iname in zip(inputs, inames) if iname in iloop]
        ret = []
        for arr in zip(*all_bcast):
            if 0 in arr:
                ret.append(0)
            else:
                ret.append(1)
        return [ret] * self.nout

    @classmethod
    def inplace_version(cls):
        class Ret(cls, Destroyer):
            def destroy_list(self):
                return self.inputs[0]
        return Ret

    def c_init(self, inputs, outputs):
        pass

    def c_foreach(self, inputs, outputs):
        pass

    def c_finalize(self, inputs, outputs):
        pass


class TensorScalarOp(_Elemwise):
    def c_var_names(self):
        return (['x', '_a'], ['z', ])
    def loop_variables(self):
        return (['x', ], ['z', ])
    def c_init((x, _a), (z, )):
        return """
        if (PyArray_SIZE(_a) != 1) {
            PyErr_SetString(PyExc_ValueError, \"The size of the scalar argument is not 1.\");
        }
        _a_dtype a = ((_a_dtype*)PyArray_DATA(_a))[0];
        """
    def _c_foreach(self):
        return "z_i = %s;" % self.c_expr

def constructor(op_cls):
    def f(*args, **kwargs):
        op = op_cls(*args, **kwargs)
        if len(op.outputs) > 1:
            return op.outputs
        else:
            return op.outputs[0]
    return f

##########################
# Unary Operations
##########################

class Abs(_Elemwise):
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, x, gz):
        return gz * Sgn(x).out #TODO: handle the corner case (get it? pun?)
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = abs(x_i);"
#Constructor not necessary because builtin abs() does this

class Neg(_Elemwise):
    def impl(self, x):
        return -x
    def grad(self, x, gz):
        return -gz
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = -x_i;"
#Constructor not necessary because unary '-' does this

class Sgn(_Elemwise):
    def impl(self, x):
        return numpy.abs(x) / x
    def grad(self, x, gz):
        return [None]
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = x_i/abs(x_i);" # TODO: C use copysign
sgn = constructor(Sgn)

class Sum(_Elemwise):
    def impl(self, x):
        return numpy.sum(x)
    def grad(self, x, gz):
        return fill(x, gz)
    def propagate_broadcastable(self, *inputs):
        return [()]
    def c_init(self, (x, ), (sum, )):
        return "sum_dtype* sump = ((sum_dtype*)PyArray_DATA(sum)); sump[0] = 0;"
    def c_foreach(self, (x_i, ), (sum, )):
        return "sump[0] += x_i;"
sum = constructor(Sum)

class Fill(_Elemwise):
    def impl(self, model, value):
        return (model * 0) + value #TODO: we can probably do better than this
    def grad(self, (model, value), gz):
        return None, sum(gz)
    def c_init(self, (model, value), (z, )):
        return "value_dtype value0 = ((value_dtype*)PyArray_DATA(value))[0];"
    def c_foreach(self, (model_i, value), (z_i, )):
        return "z_i = value0;"
fill = constructor(Fill)


class TensorCopy(_Elemwise):
    def impl(self, x):
        return numpy.array(x)
    def grad(self, x, gz):
        return gz
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = x_i;"
tensor_copy = constructor(TensorCopy)

if 0:
    ##########################
    # View Operations
    ##########################

    class transpose(_Op, Viewer):
        def view_map(self):
            return {self.out: [self.inputs[0]]}
        def impl(self, x):
            return x.T
        def grad(self, x, gz):
            return transpose_copy(gz)
        def propagate_broadcastable(self, x):
            rval = list(x)
            rval.reverse()
            return [rval]
        
        def c_impl(self, x, z):
            return """
            PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
            if (%(z)s) {
                Py_XDECREF(%(z)s);
            }
            %(z)s = transposed;
            """

    class Subtensor(_Op, Viewer):
        def view_map(self): 
            return {self.out: [self.inputs[0]]}
        def impl(x, item): 
            rval = x.__getitem__(item)
            #print 'get_slice running', rval
            return rval
        def grad(x, gz):
            # - option: allocate a potentially large matrix of zeros, and fill in
            # the appropriate elements from gz
            # - option: return a sparse matrix
            # - option: return gz, but think about how to include a special addition
            # function that uses a matching view over the original data
            raise NotImplemented 


if 0:
    ##########################
    # Arithmetic : Add
    ##########################

    # Elemwise #
    class add_elemwise(_Elemwise):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            return x + y
        def grad(self, (x, y), gz):
            return gz, gz
        def c_foreach(self, (x_i, y_i), (z_i, )):
            return "z_i = x_i + y_i;"

    class add_elemwise_inplace(add_elemwise.inplace_version()):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            x += y
            return x

    # Scalar #
    class add_scalar(TensorScalarOp):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            return x + a
        def grad(self, (x, a), gz):
            return gz, sum(gz)
        c_expr = "x_i + a"

    class add_scalar_inplace(add_scalar.inplace_version()):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            x += a
            return x

    add = _scalar_switch(add_elemwise, add_scalar, add_scalar)
    add_inplace = _scalar_switch(add_elemwise_inplace, add_scalar_inplace)


if 0:
    ##########################
    # Arithmetic : Sub
    ##########################

    # Elemwise #
    class SubElemwise(_Elemwise):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            return x - y
        def grad(self, (x, y), gz):
            return gz, -gz
        def c_foreach(self, (x_i, y_i), (z_i, )):
            return "z_i = x_i - y_i;"

    class SubElemwiseInplace(SubElemwise.inplace_version()):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            x -= y
            return x

    # Scalar #
    def sub_scalar_r(x, a):
        return add_scalar(x, -a)

    def sub_scalar_l(x, a):
        return add_scalar(-x, a)

    def sub_scalar_rinplace(x, a):
        return add_scalar_inplace(x, -a)

    sub = _scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
    sub_inplace = _scalar_switch(sub_elemwise_inplace, sub_scalar_rinplace)

if 1:
    ##########################
    # Arithmetic : Mul
    ##########################

    # Elemwise #
    class MulElemwise(_Elemwise):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            return x * y
        def grad(self, (x, y), gz):
            return mul(y, gz), mul(x, gz)
        def c_foreach(self, (x_i, y_i), (z_i, )):
            return "z_i = x_i * y_i;"
    mul_elemwise = constructor(MulElemwise)

    class MulElemwiseInplace(MulElemwise.inplace_version()):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            x *= y
            return x
    mul_elemwise_inplace = constructor(MulElemwiseInplace)

    # Scalar #
    class Scale(TensorScalarOp):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            return x * a
        def grad(self, (x, a), gz):
            return scale(a, gz), sum(mul_elemwise(x, gz))
        c_expr = "x_i * a"
    scale = constructor(Scale)

    class ScaleInplace(Scale.inplace_version()):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            x *= a
            return x
    scale_inplace = constructor(ScaleInplace)

    mul = _scalar_switch(mul_elemwise, scale, scale)
    mul_inplace = _scalar_switch(mul_elemwise_inplace, scale_inplace)


if 0:
    ##########################
    # Arithmetic : Div
    ##########################

    # Elemwise #
    class DivElemwise(_Elemwise):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            return x / y
        def grad(self, (x, y), gz):
            return div(gz, y), -div(mul(x, gz), sqr(y))
        def c_foreach(self, (x_i, y_i), (z_i, )):
            return "z_i = x_i / y_i;"

    class DivElemwiseInplace(DivElemwise.inplace_version()):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            x /= y
            return x

    # Scalar #
    def div_scalar_r(x, a):
        return scale(x, inv_elemwise(a))

    def div_scalar_l(x, a):
        return scale(inv_elemwise(x), a)

    def div_scalar_rinplace(x, a):
        return scale_inplace(x, inv_elemwise(a))

    div = _scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
    div_inplace = _scalar_switch(div_elemwise_inplace, div_scalar_rinplace)




if 0:
    ##########################
    # Arithmetic : Pow
    ##########################

    # Elemwise #

    class PowElemwise(_Elemwise):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            return x ** y
        def grad(self, (x, s), gz):
            gx = gz * s * (pow_elemwise(x, s-1.0))
            gs = gz * log(x) * pow_elemwise(x, s)
            return gx, gs
        def c_foreach(self, (x_i, s_i), (z_i, )):
            return "z_i = pow(x_i, s_i)"

    class PowElemwiseInplace(PowElemwise.inplace_version()):
        def impl(self, x, y):
            _assert_same_shapes(x, y)
            x **= y
            return x

    # Scalar #
    class PowScalarL(TensorScalarOp):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            return a ** x
        def grad(self, (x, s), gz):
            gx = sum(gz * s * pow_scalar_l(add_scalar(s,-1.0), x))
            gs = scale(mul(gz, pow_scalar_l(s, x)), log(x))
            return gx, gs
        c_expr = "pow(a, x_i)"

    class PowScalarR(TensorScalarOp):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            return x ** a
        def grad(self, (x, s), gz):
            gx = scale(mul_elemwise(gz,pow_scalar_r(x, add_scalar(s,-1.0))), s)
            gs = sum(mul_elemwise(mul_elemwise(gz, pow_scalar_r(x,s)), log(x)))
            return gx, gs
        c_expr = "pow(x_i, a)"

    class PowScalarRInplace(PowScalarR.inplace_version()):
        def impl(self, x, a):
            _assert_tensor_scalar(x, a)
            x **= a
            return x

    pow = _scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
    pow_inplace = _scalar_switch(pow_elemwise_inplace, pow_scalar_rinplace)


if 0:
    ##########################
    # Comparisons 
    ##########################

    # Less-than
    class lt_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class lt_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Less-than or equal
    class le_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class le_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Greater-than or equal
    class gt_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class gt_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Greater-than or equal
    class ge_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class ge_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()




if 0:
    def _broadcastable_pattern(pattern):
        def factory(data = None, name = None, dtype=None):
            if data: 
                assert len(data.shape) == len(pattern)
                if dtype is not None:
                    assert dtype is data.dtype
                dtype = data.dtype
                rval = Tensor(dtype, pattern, name)
                rval.data = data
            else:
                rval = Tensor(dtype, pattern, name)
            return  rval
        return factory

    row = _broadcastable_pattern([1, 0])
    col = _broadcastable_pattern([0, 1])
    matrix = _broadcastable_pattern([0, 0])

if 0: #old __init__ code
    """Create a Tensor

    If data is given:
        - constant defaults to True
        - if dtype is given, it must match data.dtype
            - otherwise: default is data.dtype
        - if broadcastable is given, len(broadcastable) must match len(data.shape)
            - otherwise: if it is constant, it defaults to 1 where shape[i]==1
            - if it is not constant, it defaults to 0s

    If data is not given:
        - constant defaults to False
    """
    if dtype is None or broadcastable is None:
        if data is None:
            raise TypeError("Provide non-None data to complete the dtype and broadcastable flags.")
        data = numpy.asarray(data)
        if constant is None:
            constant = True
        dtype = data.dtype
        if constant:
            broadcastable = [1*(x == 1) for x in data.shape]
        else:
            broadcastable = [0] * len(data.shape)

if 0:
    def tensor__new__(cls, *args, **kwargs):
        """__new__ is overloaded to handle the special form Tensor(x) when x is
        a Tensor or an Op whose default output is a Tensor.  In these cases, the
        argument x is returned, and a new Tensor is not created.
        """
        if len(args) == 1:
            a = args[0]

        t = super(Tensor, cls).__new__(cls, *args, **kwargs)
        t.__init__(*args, **kwargs)
        return t


