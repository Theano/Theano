
from tensor import *
from gof import Op, utils


def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)

def wrap_as_tensor(x):
    if isinstance(x, Tensor):
        return x
    else:
        return Tensor(data=x, constant=True)

class TensorOp(Op):

    nin = -1
    nout = 1

    cast_method = lambda self, *args: upcast(*args)
    
    def __init__(self, *inputs):

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
        //if (PyArray_REFCOUNT(transposed) == 1) {
        //    printf("lala\\n");
        //}
        //if (%(z)s) {
        //    Py_XDECREF(%(z)s);
        //}
        %(z)s = transposed;
        Py_XINCREF(%(z)s);
        """


from gof import modes
modes.make_constructors(globals())



def scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    def f(x, y):
        x, y = wrap_as_tensor(x), wrap_as_tensor(y)
        if 0 not in y.broadcastable:
            return scalar_f(x, y)
        if 0 not in x.broadcastable:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f

# Wrapper to ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)
def assert_same_shapes(x, *rest):
    shape = x.shape
    for other in rest:
        if other.shape != shape:
            raise ValueError("The dimensions of the inputs do not match.")

# Wrapper to ensure that the last input to impl is a scalar
def assert_tensor_scalar(x, a):
    if numpy.product(a.shape) != 1:
        raise ValueError("The second argument must be a scalar.")



class tensor_scalar_op(elemwise):
    @classmethod
    def variable_names(cls):
        return (['x', '_a'], ['z', ])
    @classmethod
    def loop_variables(cls):
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



#########
## Add ##
#########

# Elemwise #
class AddElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x + y
    def grad(self, (x, y), gz):
        return gz, gz
    def c_foreach((x_i, y_i), (z_i, )):
        return "z_i = x_i + y_i;"

class AddElemwiseInplace(AddElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x += y
        return x

# Scalar #
class AddScalar(TensorScalarOp):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        return x + a
    def grad(self, (x, a), gz):
        return gz, sum(gz)
    c_expr = "x_i + a"

AddScalarInplace = add_scalar.inplace_version()
add_scalar_inplace.set_impl(tensor_scalar_impl(numpy.ndarray.__iadd__))


class twice(elemwise):
    def impl(x):
        return 2.0 * x
    def grad(x, gz):
        return scale(gz, 2.0)
    def c_foreach((x_i, ), (z_i, )):
        "z_i = x_i + x_i;"

twice_inplace = twice.inplace_version()


## Subtraction ##

class sub_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__sub__)
    def grad(x, y, gz):
        return gz, -gz
    def c_foreach((x_i, y_i), (z_i, )):
        return "z_i = x_i - y_i;"

sub_elemwise_inplace = sub_elemwise.inplace_version()
sub_elemwise_inplace.set_impl(assert_same_shapes(numpy.ndarray.__isub__))

def sub_scalar_r(x, a):
    return add_scalar(x, -a)

def sub_scalar_l(x, a):
    return add_scalar(-x, a)

def sub_scalar_r_inplace(x, a):
    return add_scalar_inplace(x, -a)


## Element-wise multiplication ##

class mul_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__mul__)
    def grad(x, y, gz):
        return mul(y, gz), mul(x, gz)
    def c_foreach((x_i, y_i), (z_i, )):
        return "z_i = x_i * y_i;"

mul_elemwise_inplace = mul_elemwise.inplace_version()
mul_elemwise_inplace.set_impl(assert_same_shapes(numpy.ndarray.__imul__))


class scale(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__mul__)
    def grad(x, a, gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))
    c_expr = "x_i * a"

scale_inplace = scale.inplace_version()
scale_inplace.set_impl(tensor_scalar_impl(numpy.ndarray.__imul__))


class sqr(elemwise):
    def impl(x):
        return x * x
    def grad(x, gz):
        return scale(mul_elemwise(x, gz), 2.0)
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = x_i * x_i;"

isqr = sqr.inplace_version()
isqr.set_impl(lambda x: x.__imul__(x))



class sqrt(elemwise):
    impl = numpy.sqrt
    def grad(x, gz):
        return scale(div(gz, sqrt(x)), 0.5)
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = pow(x_i, 0.5);"

isqrt = sqrt.inplace_version()
isqrt.set_impl(lambda x: x.__ipow__(0.5))



## Element-wise division ##

class div_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__div__)
    def grad(x, y, gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))
    def c_foreach((x_i, y_i), (z_i, )):
        return "z_i = x_i / y_i;"

div_elemwise_inplace = div_elemwise.inplace_version()
div_elemwise_inplace.set_impl(assert_same_shapes(numpy.ndarray.__idiv__))

def div_scalar_r(x, a):
    return scale(x, inv_elemwise(a))

def div_scalar_l(x, a):
    return scale(inv_elemwise(x), a)

def div_scalar_r_inplace(x, a):
    return scale_inplace(x, inv_elemwise(a))



## Scaling ##

class scale(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__mul__)
    def grad(x, a, gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))
    c_expr = "x_i * a"

scale_inplace = scale.inplace_version()
scale_inplace.set_impl(tensor_scalar_impl(numpy.ndarray.__imul__))


class neg(elemwise):
    impl = numpy.ndarray.__neg__
    def grad(x, gz):
        return -gz
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = -x_i;"

neg_inplace = neg.inplace_version()
neg_inplace.set_impl(lambda x: x.__imul__(-1))


class inv_elemwise(elemwise):
    impl = lambda x: 1 / x
    def grad(x, gz):
        return -gz
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = 1 / x_i;"

inv_elemwise_inplace = inv_elemwise.inplace_version()


## Dot product ##

class dot(omega_op):
    @staticmethod
    def _output_shape(xshape, yshape):
        if len(xshape) == 0: # x is a scalar
            shape = yshape
        else:
            if len(yshape) >= 2: #y is a matrix or tensor
                assert xshape[-1] == yshape[-2]
                shape = tuple(xshape[:-1]+ yshape[:-2]+yshape[-1:])
            elif len(yshape)==1: #y is vector
                assert xshape[-1] == yshape[-1]
                shape = tuple(xshape[:-1])
            else:                #y is a scalar
                shape = xshape
        return shape

    impl = numpy.dot
    def grad(x, y, gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)
    def refresh(self, alloc=False):
        x,y = self.inputs
        shape = self._output_shape(x.shape, y.shape)
        dtype = upcast(x.dtype, y.dtype)
        if self.out.data is not None \
                and self.out.shape == shape \
                and self.out.dtype == dtype:
                    return  #everything is ok
        if alloc or self.out.data is not None: #data should be allocated
            self.out.data = None
            self.out.shape = shape
            self.out.dtype = dtype
            self.out.alloc()
        else:
            self.out.shape = shape
            self.out.dtype = dtype
    def c_support_code(self):
        return blas.cblas_header_text()
    def c_libs(self):
        return blas.ldflags()
    def c_impl((_x, _y), (_z, )):
        return blas.gemm_code('', '1.0', '0.0')



## Transposition ##

class transpose(omega_op):
    def view_map(self): return {self.out: [self.inputs[0]]}
    impl = numpy.transpose
    def grad(x, gz):
        return transpose_copy(gz)
    def refresh_shape(self):
        rval = list(self.inputs[0].shape)
        rval.reverse()
        return rval
    def refresh_dtype(self):
        return  self.inputs[0].dtype
    def c_impl((x, ), (xt, )):
        return """
        const int l = x->nd;
        // The user must ensure that all references to
        //xt->data go through xt, or there's going to be trouble..
        int refcheck = 0;

          if (x == xt)
            {
              return -1;
            }
          if (refcheck)
            {
              int refcnt =  PyArray_REFCOUNT(xt);
                if ((refcnt > 2)  // you might think this should be 1.. but this works
                    //|| (xt->base != NULL)
                    || (xt->weakreflist != NULL))
                  {
                    PyErr_SetString(PyExc_ValueError,
                                        "cannot resize an array that has "\\
                                        "been referenced or is referencing\\n"\\
                                        "another array in this way.  Use the "\\
                                        "resize function");
                    return -2;
                  }
            }

        if (xt->nd != x->nd)
        {
            // this technique comes from PyArray_Resize()
            npy_intp * dimptr = (npy_intp*)PyDimMem_RENEW(xt->dimensions, 2 * x->nd);
            if (!dimptr)
            {
                  PyErr_NoMemory();
                  return 1;
            }
            xt->nd = x->nd;
            xt->dimensions = dimptr;
            xt->strides = dimptr + x->nd;
        }
        //copy x's dimensions and strides
        for (int i = 0; i < l; ++i)
        {
            xt->dimensions[i] = x->dimensions[l-i-1];
            xt->strides[i] = x->strides[l-i-1];
        }

        // point directly at b's type descriptor
        Py_INCREF(x->descr);
        Py_DECREF(xt->descr);
        xt->descr = x->descr;

        // name x as a base of xt, increment its refcount
        if ( xt->base != (PyObject*)x)
        {
          Py_INCREF(x);
          if ((xt->base) && (xt->base != Py_None)) 
            {
              Py_DECREF(xt->base);
            }
          xt->base = (PyObject*)x;
        }
    
        // mark xt as not owning its data
        if (PyArray_CHKFLAGS(xt, NPY_OWNDATA))
          {
            PyDataMem_FREE(xt->data);
            xt->flags &= ~NPY_OWNDATA;
          }
        xt->data = x->data;

        // this function is described in 
        // ~/zzz.NOBACKUP/pub/src/numpy-1.0.3.1/numpy/core/src/arrayobject.c:1890
        PyArray_UpdateFlags(xt, NPY_CONTIGUOUS|NPY_FORTRAN|NPY_ALIGNED|NPY_WRITEABLE); 

        /*
          TODO
          What should be done with the weakreflist ?
        */
    """

def transpose_copy(x):
    return array_copy(transpose(x))


## Copy ##

class array_copy(elemwise):
    impl = numpy.array
    grad = lambda x, gz: gz
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = x_i;"


## Power ##

class sqr(elemwise):
    def impl(x):
        return x * x
    def grad(x, gz):
        return scale(mul_elemwise(x, gz), 2.0)
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = x_i * x_i;"

sqr_inplace = sqr.inplace_version()
sqr_inplace.set_impl(lambda x: x.__imul__(x))




class sqrt(elemwise):
    impl = numpy.sqrt
    def grad(x, gz):
        return scale(div(gz, sqrt(x)), 0.5)
    def c_foreach((x_i, ), (z_i, )):
        return "z_i = pow(x_i, 0.5);"

sqrt_inplace = sqrt.inplace_version()
sqrt_inplace.set_impl(lambda x: x.__ipow__(0.5))


class exp(elemwise):
    def impl(x): return numpy.exp(x)
    def grad(x, gz): return gz * exp(x)
    def c_foreach((x_i, ), (z_i, )): return "z_i = exp(x_i);"
    
class log(elemwise):
    def impl(x): return numpy.log(x)
    def grad(x, gz): return gz / x
    def c_foreach((x_i, ), (z_i, )): return "z_i = log(x_i);"

class log2(elemwise):
    def impl(x): return numpy.log2(x)
    def grad(x, gz): return gz / (x * numpy.log(2))
    def c_foreach((x_i, ), (z_i, )): return "z_i = log2(x_i);"

class pow_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        raise NotImplemented # no gs
        return gz * s * (pow_elemwise(x, s-1.0))
    def c_foreach((x_i, s_i), (z_i, )):
        return "z_i = pow(x_i, s_i)"

pow_elemwise_inplace = pow_elemwise.inplace_version()
pow_elemwise_inplace.set_impl(assert_same_shapes(numpy.ndarray.__ipow__))

class pow_scalar_l(tensor_scalar_op):
    impl = tensor_scalar_impl(lambda x, y: numpy.ndarray.__pow__(y, x))
    def grad(x, s, gz):
        raise NotImplemented # no gs
        return gz * x * (pow_scalar_l(s,x-1.0))
    c_expr = "pow(a, x_i)"

class pow_scalar_r(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        gx = gz * s * (pow_scalar_r(x,s-1.0))
        gs = sum(gz * pow_scalar_r(x,s) * log(x))
        return gx, gs
    c_expr = "pow(x_i, a)"

pow_scalar_r_inplace = pow_scalar_r.inplace_version()
pow_scalar_r_inplace.set_impl(tensor_scalar_impl(numpy.ndarray.__ipow__))


## Others ##

class minmax(elemwise):
    nout = 2
    def impl(x):
        return x.min, x.max
    def specs(x):
        return [(numpy.ndarray, x[1], ())] * 2
#     def alloc((x, ), (_min, _max)):
#         _min.data = numpy.ndarray((), x.dtype)
#         _max.data = numpy.ndarray((), x.dtype)
    def c_init((x, ), (_min, _max)):
        raise NotImplementedError
        return """
        _x_dtype min = _x[0];
        _x_dtype max = _x[0];
        """
    def c_foreach((x, ), (_min, _max)):
        return """
        if (x < min) min = x;
        if (x > max) max = x;
        """
    def c_finalize((x, ), (_min, _max)):
        return """
        _min[0] = min;
        _max[0] = max;
        """


class fill(elemwise):
    impl = lambda model, value: (model * 0) + value
    def c_init((model, value), (z, )):
        return "value_dtype value0 = ((value_dtype*)PyArray_DATA(value))[0];"
    def c_foreach((model_i, value), (z_i, )):
        return "z_i = value0;"

fill_inplace = fill.inplace_version()

class sum(elemwise):
    impl = numpy.sum
    def grad(x, gz):
        return fill(x, gz)
    def refresh_shape(self):
        return ()
    def c_init((x, ), (sum, )):
        return "sum_dtype* sump = ((sum_dtype*)PyArray_DATA(sum)); sump[0] = 0;"
    def c_foreach((x_i, ), (sum, )):
        return "sump[0] += x_i;"

class ones_like(elemwise):
    impl = numpy.ones_like
    def grad(x, gz): return Undefined

class zeros_like(elemwise):
    impl = numpy.zeros_like
    def grad(x, gz): return Undefined

## Array slicing ##

class get_slice(omega_op):
    def view_map(self): return {self.out: [self.inputs[0]]}
    def impl(x, item): 
        rval = x.__getitem__(item)
        #print 'get_slice running', rval
        return rval
    def grad(x, gz): raise NotImplemented
    def refresh_shape(self): 
        x,item = self.inputs
        rval = x.data.__getitem__(item.data).shape 
        #print 'refresh_shape', rval
        return rval
    def refresh_dtype(self):
        return self.inputs[0].data.dtype


add = scalar_switch(add_elemwise, add_scalar, add_scalar)
add_inplace = scalar_switch(add_elemwise_inplace, add_scalar_inplace)

sub = scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
sub_inplace = scalar_switch(sub_elemwise_inplace, sub_scalar_r_inplace)

mul = scalar_switch(mul_elemwise, scale, scale)
mul_inplace = scalar_switch(mul_elemwise_inplace, scale_inplace)

div = scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
div_inplace = scalar_switch(div_elemwise_inplace, div_scalar_r_inplace)

pow = scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
pow_inplace = scalar_switch(pow_elemwise_inplace, pow_scalar_r_inplace)













