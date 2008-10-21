## This file contain ops that are not currently integrated in the core of threano. 
## Not all of those ops have been thoroughly tested.

import theano
from theano import tensor, scalar
import numpy

############
#
# SCALAR OPS
#

class ScalarSigmoid(scalar.UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return 1.0 
        return 1.0 / (1.0 + numpy.exp(-x))
    def impl(self, x):
        return ScalarSigmoid.st_impl(x)
    def grad(self, (x,), (gz,)):
        y = scalar_sigmoid(x)
        return [gz * y * (1.0 - y)]
    def c_code(self, node, name, (x,), (z,), sub):
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                %(x)s < -30.0 
                ? 0.0 
                : %(x)s > 30.0 
                   ? 1.0
                   : 1.0 /(1.0+exp(-%(x)s));""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
scalar_sigmoid = ScalarSigmoid(scalar.upgrade_to_float, name='scalar_sigmoid')
sigmoid = tensor.Elemwise(scalar_sigmoid, name='sigmoid')

class ScalarSoftplus(scalar.UnaryScalarOp):
    @staticmethod
    def static_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return x
        return numpy.log1p(numpy.exp(x))
    def impl(self, x):
        return ScalarSoftplus.static_impl(x)
    def grad(self, (x,), (gz,)):
        return [gz * scalar_sigmoid(x)]
    def c_code(self, node, name, (x,), (z,), sub):
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                %(x)s < -30.0 
                ? 0.0 
                : %(x)s > 30.0 
                   ? %(x)s
                   : log1p(exp(%(x)s));""" % locals()
        raise NotImplementedError('only floating point x is implemented')
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float, name='scalar_softplus')
softplus = tensor.Elemwise(scalar_softplus, name='softplus')


############
#
# TENSOR OPS
#


class SoftmaxWithBias(theano.Op):
    """
    An L{Op} for the output of neural-net multiclass classifiers.

    @type x: is a matrix of floats (32 or 64)
    @type b: is a [row] vector of floats (32 or 64), length is number of cols in x

    This L{Op}'s output is softmax(x+b).
    softmax(x[i]) is the i'th distribution over len(x[i]) options.
    """

    nin = 2
    nout = 1
    def __init__(self, **kwargs):
        theano.Op.__init__(self, **kwargs)

    def make_node(self, x, b):
        x = tensor.as_tensor(x)
        b = tensor.as_tensor(b)
        if x.type.ndim != 2 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('x must be 2-d tensor of floats')
        if b.type.ndim != 1 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('b must be 1-d tensor of floats')

        sm = x.type.make_result()
        return theano.Apply(self, [x, b], [sm])

    def perform(self, node, input_storage, output_storage):
        x, b = input_storage
        if b.shape[0] != x.shape[1]:
            raise ValueError('b must have same number of columns as x')

        sm = numpy.zeros_like(x)
        for i in xrange(sm.shape[0]):
            row = x[i] + b
            sm[i] = numpy.exp(row - numpy.max(row))
            sm[i] *= 1.0 / numpy.sum(sm[i])
        output_storage[0][0] = sm

    def grad(self, (x, b), (g_sm,)):
        sm = softmax_with_bias(x, b)
        dx = SoftmaxWithBiasDx()(g_sm, sm)
        db = tensor.sum(dx, axis = 0)
        return dx, db

    def c_headers(self):
        return ['<iostream>']

    @staticmethod
    def c_code_template():
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        #TODO: put this into a templated function, in the support code
        #TODO: declare the max of each row as an Op output

        #TODO: set error messages for failures in this code

        #TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        init_decl = """
        npy_intp* Nx = %(x)s->dimensions;

        if (%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "a not 2d tensor");
            %(fail)s;
        }
        if (%(b)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
        if (%(x)s->descr->type_num != PyArray_DOUBLE)
        {
            PyErr_SetString(PyExc_TypeError, "a not float64");
            %(fail)s;
        }
        if (%(b)s->descr->type_num != PyArray_DOUBLE)
        {
            PyErr_SetString(PyExc_TypeError, "b not float64");
            %(fail)s;
        }
        if ((%(x)s->dimensions[1] != %(b)s->dimensions[0]))
        {
            PyErr_SetString(PyExc_ValueError, "dimension mismatch in arguments");
            %(fail)s;
        }

        if ((NULL == %(sm)s)
            || (%(sm)s->dimensions[0] != %(x)s->dimensions[0])
            || (%(sm)s->dimensions[1] != %(x)s->dimensions[1]))
        {
            if (NULL != %(sm)s) Py_XDECREF(%(sm)s);
            %(sm)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(x)s), type_num_%(x)s);
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc sm output");
                %(fail)s
            }
        }
        """

        begin_row_loop = """
        for (size_t i = 0; i < Nx[0]; ++i)
        {
            size_t j;
            double sum = 0.0;
            bool  discount_max = false;

            const double* __restrict__ x_i = (double*)(%(x)s->data + %(x)s->strides[0] * i);
            const double* __restrict__ b_i = (double*)(%(b)s->data);
            double* __restrict__ sm_i = (double*)(%(sm)s->data + %(sm)s->strides[0] * i);
        """

        inside_row_loop = """
            npy_intp Sx = %(x)s->strides[1]/sizeof(double);
            npy_intp Sb = %(b)s->strides[0]/sizeof(double);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(double);

            size_t row_max_j=0;
            double row_max = x_i[0] + b_i[0];
            // Get the maximum value of the row
            for (j = 0; j < Nx[1]; ++j)
            {
                double row_ij = x_i[j * Sx] +  b_i[j * Sb];
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }

            for (j = 0; j < Nx[1]; ++j)
            {
                double row_ij = x_i[j * Sx] +  b_i[j * Sb];
                double sm_ij = exp(row_ij - row_max);
                sum += sm_ij;
                sm_i[j * Ssm] = sm_ij;
            }
            if ( (0.0 == sum) || (isinf(sum)))
            {
                //that was our best...
                %(fail)s;
            }

            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm] *= sum_inv;
            }

        """

        end_row_loop = """
        }
        """

        return (init_decl, begin_row_loop, inside_row_loop, end_row_loop)


    def c_code(self, node, name, (x, b), (sm,), sub):
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)

softmax_with_bias = SoftmaxWithBias()


class SoftmaxWithBiasDx(theano.Op):
    nin = 2
    nout = 1
    """Gradient wrt x of the SoftmaxWithBias Op"""

    def __init__(self, **kwargs):
        theano.Op.__init__(self, **kwargs)

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor(dy)
        sm = tensor.as_tensor(sm)
        return theano.Apply(self, [dy, sm], [sm.type.make_result()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = numpy.zeros_like(sm)
        #dx[i,j] = - (\sum_k dy[i,k] sm[i,k]) sm[i,j] + dy[i,j] sm[i,j]
        for i in xrange(sm.shape[0]):
            dy_times_sm_i = dy[i] * sm[i]
            dx[i] = dy_times_sm_i - sum(dy_times_sm_i) * sm[i]
        output_storage[0][0] = dx

    def grad(self, *args):
        raise NotImplementedError()

    def c_code(self, node, name, (dy, sm), (dx,), sub):
        return '''
        if ((%(dy)s->descr->type_num != PyArray_DOUBLE)
            || (%(sm)s->descr->type_num != PyArray_DOUBLE))
        {
            PyErr_SetString(PyExc_TypeError, "types should be float64, float64");
            %(fail)s;
        }
        if ((%(dy)s->nd != 2)
            || (%(sm)s->nd != 2))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (%(dy)s->dimensions[0] != %(sm)s->dimensions[0])
        {
            PyErr_SetString(PyExc_ValueError, "dy.shape[0] != sm.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (%(dx)s->dimensions[0] != %(sm)s->dimensions[0])
            || (%(dx)s->dimensions[1] != %(sm)s->dimensions[1]))
        {
            Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_SimpleNew(2, PyArray_DIMS(%(sm)s),
                                                        type_num_%(sm)s);
            if (!%(dx)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
                %(fail)s;
            }
        }

        for (size_t i = 0; i < %(dx)s->dimensions[0]; ++i)
        {
            const double* __restrict__ dy_i = (double*) (%(dy)s->data + %(dy)s->strides[0] * i);
            npy_intp Sdy = %(dy)s->strides[1]/sizeof(double);
            const double* __restrict__ sm_i = (double*) (%(sm)s->data + %(sm)s->strides[0] * i);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(double);
            double* __restrict__ dx_i = (double*) (%(dx)s->data + %(dx)s->strides[0] * i);
            npy_intp Sdx = %(dx)s->strides[1]/sizeof(double);

            double sum_dy_times_sm = 0.;
            for (size_t j = 0; j < %(dx)s->dimensions[1]; ++j)
            {
                dx_i[j * Sdx] = dy_i[j * Sdy] * sm_i[j * Ssm];
                sum_dy_times_sm += dx_i[j * Sdx];
            }
            for (size_t j = 0; j < %(dx)s->dimensions[1]; ++j)
            {
                dx_i[j * Sdx] -= sum_dy_times_sm * sm_i[j * Ssm];
            }
        }
        ''' % dict(locals(), **sub)

def softmax(x, **kwargs):
    b = tensor.zeros_like(x[0,:])
    return softmax_with_bias(x, b, **kwargs)


class CrossentropySoftmaxArgmax1HotWithBias(theano.Op):
    """A special compound L{Op} for the output of neural-net classifiers.

    @type x: is a matrix of floats (32 or 64)
    @type b: is a [row] vector of floats (32 or 64), length is number of cols in x
    @type y_idx: a [column] vector of int (32 or 64), length is number of rows in x

    @precondition: every entry in y_idx is a valid (non-negative) column index into x

    This L{Op} has three outputs:
     - KL(softmax(x+b), y)
     - softmax(x+b)
     - argmax(x+b)

    softmax(x[i]) is the i'th distribution over len(x[i]) options
    argmax(x) is the index of x's greatest element
    y_idx[i] is an integer index, encoding a 1-hot distribution. 

    In practice, when we're trying to do classification, we have one row in x
    and y_idx per example, and y[i] is the index of the (correct) class of the
    i'th example.

    """
    nin=3
    nout=3
    def __init__(self, **kwargs):
        theano.Op.__init__(self, **kwargs)

    def make_node(self, x, b, y_idx):
        x = tensor.as_tensor(x)
        b = tensor.as_tensor(b)
        y_idx = tensor.as_tensor(y_idx)
        if x.type.ndim != 2 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('x must be 2-d tensor of floats')
        if b.type.ndim != 1 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('b must be 1-d tensor of floats')
        if y_idx.type.ndim != 1 \
                or y_idx.type.dtype not in ['int8', 'int16', 'int32', 'int64']:
            raise ValueError('y_idx must be 1-d tensor of ints')

#       TODO: Is this correct? It used to be y, not y_idx
        nll = tensor.Tensor(x.type.dtype,
                y_idx.type.broadcastable).make_result()
#        nll = Tensor(x.dtype, y.broadcastable)
        sm = x.type.make_result()
        am = y_idx.type.make_result()
        return theano.Apply(self, [x, b, y_idx], [nll, sm, am])
    def perform(self, node, input_storage, output_storage):
        """
        The math, where x is an input vector, and t is a target index:

            softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
            nll(x,t) = -log(softmax(x)[t])

        We compute this by subtracting off the max of x. This avoids numerical instability.

            m = max_j x[j]
            softmax(x)[i] = exp(x[i] -m) / sum_j(exp(x[j] - m))

            nll = -log(exp(x[t] -m) / sum_j(exp(x[j] - m)))
                = -x[t] + m + log( sum_j(exp(x[j] - m)))

        """
        x, b, y_idx = input_storage
        if b.shape[0] != x.shape[1]:
            raise ValueError('b must have same number of columns as x')
        if y_idx.shape[0] != x.shape[0]:
            raise ValueError('y_idx must have same number of rows as x')

        sm = numpy.zeros_like(x) # softmax
        nll = numpy.zeros(x.shape[0]) #nll(y | softmax(x))
        am = numpy.zeros_like(y_idx)
        for i in xrange(sm.shape[0]):
            #add the bias vector to the i'th row of x
            row = x[i] + b 

            #get the maximum value of i'th row for numerically safe softmax / nll
            am[i] = numpy.argmax(row)
            m = row[am[i]]

            #compute the unnormalized softmax, and normalization constant
            sm[i] = numpy.exp(row - m) 
            sum_j = numpy.sum(sm[i]) # sum_j(exp(x[j] - m))

            #normalized our softmax
            sm[i] *= 1.0 / sum_j

            # store the nll
            nll[i] = -row[y_idx[i]] + m + numpy.log(sum_j)
            
        output_storage[0][0] = nll
        output_storage[1][0] = sm
        output_storage[2][0] = am
    def grad(self, (x, b, y_idx), (g_nll, g_sm, g_am)):
        if g_sm is not None or g_am is not None:
            raise NotImplementedError()
        nll, sm = crossentropy_softmax_1hot_with_bias(x, b, y_idx)
        #dx = CrossentropySoftmax1HotWithBiasDx()(g_nll, sm, y_idx)
        dx = crossentropy_softmax_1hot_with_bias_dx(g_nll, sm, y_idx)
        db = tensor.sum(dx, axis = [0])
        return dx, db, None

    def c_headers(self): return ['<iostream>']

    @staticmethod
    def c_code_template():
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        #TODO: put this into a templated function, in the support code
        #TODO: declare the max of each row as an Op output

        #TODO: set error messages for failures in this code

        #TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        (init_decl, begin_row_loop, inside_row_loop, end_row_loop) = \
                SoftmaxWithBias.c_code_template()
        return (init_decl,
                """
        if (%(y_idx)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "y_idx not 1d tensor");
            %(fail)s;
        }
        if ((%(y_idx)s->descr->type_num != PyArray_INT64)
            && (%(y_idx)s->descr->type_num != PyArray_INT32)
            && (%(y_idx)s->descr->type_num != PyArray_INT16)
            && (%(y_idx)s->descr->type_num != PyArray_INT8))
        {
            PyErr_SetString(PyExc_TypeError, "y_idx not int8, int16, int32, or int64");
            %(fail)s;
        }
        if (%(x)s->dimensions[0] != %(y_idx)s->dimensions[0])
        {
            PyErr_SetString(PyExc_ValueError, "dimension mismatch in arguments");
            %(fail)s;
        }

        if ((NULL == %(nll)s) //initial condition
            || (%(nll)s->dimensions[0] != %(y_idx)s->dimensions[0]))
        {
            if (NULL != %(nll)s) Py_XDECREF(%(nll)s);
            %(nll)s = (PyArrayObject*)PyArray_SimpleNew(1, PyArray_DIMS(%(y_idx)s), type_num_%(x)s);
            if(!%(nll)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc nll output");
                %(fail)s;
            }
        }
        if ((NULL == %(am)s)
            || (%(am)s->dimensions[0] != %(y_idx)s->dimensions[0]))
        {
            Py_XDECREF(%(am)s);
            %(am)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(y_idx)s), type_num_%(y_idx)s);
            if(!%(am)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc am output");
                %(fail)s;
            }
        }
                """,
                begin_row_loop,
                """
            const %(y_idx_type)s y_i = ((%(y_idx_type)s*)(%(y_idx)s->data + %(y_idx)s->strides[0] * i))[0];
            double* __restrict__ nll_i = (double*)(%(nll)s->data + %(nll)s->strides[0] * i);
            %(am_type)s* __restrict__ am_i = (%(am_type)s*) (%(am)s->data + %(am)s->strides[0] * i);
                """,
                inside_row_loop,
                """
            nll_i[0] = - x_i[y_i*Sx]
                       - b_i[y_i*Sb]
                       + row_max
                       + log(sum);
            am_i[0] = row_max_j;
                """,
                end_row_loop)


    def c_code(self, node, name, (x, b, y_idx), (nll, sm, am), sub):
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        am_type = y_idx_type
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)

class CrossentropySoftmax1HotWithBiasDx (theano.Op):
    nin=3
    nout=1
    """Gradient wrt x of the CrossentropySoftmax1Hot Op"""
    def __init__(self, **kwargs):
        theano.Op.__init__(self,**kwargs)
    def make_node(self, dy, sm, y_idx,**kwargs):
        dy = tensor.as_tensor(dy)
        sm = tensor.as_tensor(sm)
        y_idx = tensor.as_tensor(y_idx)
        return theano.Apply(self, [dy, sm, y_idx],[sm.type.make_result()])
    def perform(self, node, input_storage, output_storage):
        dy,sm,y_idx = input_storage
        dx = numpy.zeros_like(sm)
        for i in xrange(sm.shape[0]):
            dx[i] = dy[i] * sm[i] #vector scale
            dx[i, y_idx[i]] -= dy[i] #scalar decrement
        output_storage[0][0] = dx
    def grad(self, *args):
        raise NotImplementedError()
    def c_code(self, node, name, (dnll, sm, y_idx), (dx,), sub):
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        return """

        if ((%(dnll)s->descr->type_num != PyArray_DOUBLE)
            || (%(sm)s->descr->type_num != PyArray_DOUBLE)
            )
        {
            PyErr_SetString(PyExc_TypeError, "types should be float64, float64, int64");
            %(fail)s;
        }
        if ((%(y_idx)s->descr->type_num != PyArray_INT64)
            && (%(y_idx)s->descr->type_num != PyArray_INT32)
            && (%(y_idx)s->descr->type_num != PyArray_INT16)
            && (%(y_idx)s->descr->type_num != PyArray_INT8))
        {
            PyErr_SetString(PyExc_TypeError, "y_idx not int8, int16, int32, or int64");
            %(fail)s;
        }
        if ((%(dnll)s->nd != 1)
            || (%(sm)s->nd != 2)
            || (%(y_idx)s->nd != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (%(dnll)s->dimensions[0] != %(sm)s->dimensions[0])
        {
            PyErr_SetString(PyExc_ValueError, "dnll.shape[0] != sm.shape[0]");
            %(fail)s;
        }
        if (%(dnll)s->dimensions[0] != %(y_idx)s->dimensions[0])
        {
            PyErr_SetString(PyExc_ValueError, "dnll.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (%(dx)s->dimensions[0] != %(sm)s->dimensions[0])
            || (%(dx)s->dimensions[1] != %(sm)s->dimensions[1]))
        {
            if (NULL != %(dx)s) Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(sm)s), type_num_%(sm)s);
            if(!%(dx)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
                %(fail)s
            }
        }

        for (size_t i = 0; i < %(dx)s->dimensions[0]; ++i)
        {
            const double dnll_i = ((double*)(%(dnll)s->data + %(dnll)s->strides[0] * i))[0];

            const %(y_idx_type)s y_i = ((%(y_idx_type)s*)(%(y_idx)s->data + %(y_idx)s->strides[0] * i))[0];

            const double* __restrict__ sm_i = (double*)(%(sm)s->data + %(sm)s->strides[0] * i);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(double);

            double* __restrict__ dx_i = (double*)(%(dx)s->data + %(dx)s->strides[0] * i);
            npy_intp Sdx = %(dx)s->strides[1]/sizeof(double);

            for (size_t j = 0; j < %(dx)s->dimensions[1]; ++j)
            {
                dx_i[j * Sdx] = dnll_i * sm_i[j * Ssm];
            }
            if (y_i >= %(dx)s->dimensions[1])
            {
                %(fail)s;
            }
            dx_i[y_i * Sdx] -= dnll_i;
        }
        """ % dict(locals(), **sub)

crossentropy_softmax_argmax_1hot_with_bias = \
    CrossentropySoftmaxArgmax1HotWithBias()

crossentropy_softmax_1hot_with_bias_dx = \
    CrossentropySoftmax1HotWithBiasDx()

def crossentropy_softmax_1hot_with_bias(x, b, y_idx, **kwargs):
    return crossentropy_softmax_argmax_1hot_with_bias(x, b, y_idx, **kwargs)[0:2]

def crossentropy_softmax_1hot(x, y_idx, **kwargs):
    b = tensor.zeros_like(x[0,:])
    return crossentropy_softmax_1hot_with_bias(x, b, y_idx, **kwargs)


class MultinomialCrossentropy1Hot(theano.Op):
    pass


def binary_crossentropy(output, target):
    """
    Compute the crossentropy of binary output wrt binary target.
    @note: We do not sum, crossentropy is computed by component.
    @todo: Rewrite as a scalar, and then broadcast to tensor.
    @todo: This is essentially duplicated as cost.cross_entropy
    @warning: OUTPUT and TARGET are reversed in cost.cross_entropy
    """
    return -(target * tensor.log(output) + (1 - target) * tensor.log(1 - output))



class Prepend_scalar_constant_to_each_row(theano.Op):
    def __init__(self, val = 0):
        if isinstance(val, float):
            val = scalar.constant(val)
        self.val = val

    def make_node(self, mat):
        #check type of input
        if not isinstance(mat,theano.Result) or not mat.type==tensor.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = tensor.as_tensor(mat)
        y = tensor.as_tensor(self.val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")

        node = theano.Apply(op=self, inputs=[mat], outputs=[tensor.matrix()])
        return node

    def perform(self, node, (mat, ), (output, )):
        new_shape=(mat.shape[0],mat.shape[1]+1)
        if output[0] == None:
            output[0]=numpy.empty(new_shape,dtype=mat.dtype)
            out=output[0]
        else:
            if output[0].shape!=new_shape:
                try:
                    output[0].resize(new_shape)
                except:
                    output[0]=numpy.empty(new_shape, dtype=mat.dtype)
            out=output[0]

        out[:,0].fill(self.val.data)
        out[:,1:]=mat

    def grad(self, (mat,), (goutput,)):
        return goutput[:,1:]

class Prepend_scalar_to_each_row(theano.Op):
    def make_node(self, val, mat):
        #check type of input
        if isinstance(val, float):
            val = scalar.constant(val)
        if not isinstance(mat,theano.Result) or not mat.type==tensor.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = tensor.as_tensor(mat)
        y = tensor.as_tensor(val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")

        node = theano.Apply(op=self, inputs=[val,mat], outputs=[tensor.matrix()])
        return node

    def perform(self, node, (val,mat), (output, )):
        new_shape=(mat.shape[0],mat.shape[1]+1)
        if output[0] == None:
            output[0]=numpy.empty(new_shape,dtype=mat.dtype)
            out=output[0]
        else:
            if output[0].shape!=new_shape:
                try:
                    output[0].resize(new_shape)
                except:
                    output[0]=numpy.empty(new_shape, dtype=mat.dtype)
            out=output[0]
        out[:,0].fill(val)
        out[:,1:]=mat

    def grad(self, (val, mat), (goutput,)):
        return goutput[:,0], goutput[:,1:]

prepend_scalar_to_each_row = Prepend_scalar_to_each_row()
prepend_0_to_each_row = Prepend_scalar_constant_to_each_row(0.)
prepend_1_to_each_row = Prepend_scalar_constant_to_each_row(1.)

class solve(theano.Op):
    """
    Find the solution to the linear equation Ax=b,
    where A is a 2d matrix and b is a 1d or 2d matrix.
    It use numpy.solve to find the solution.
    """

    def make_node(self, A, b):
        if not isinstance(A, theano.Result) or not A.type==tensor.matrix().type:
            raise TypeError("We expected that A had a matrix type")
        if not isinstance(B, theano.Result) or not B.type==tensor.matrix().type:
            raise TypeError("We expected that B had a matrix type")

        node = theano.Apply(op=self, inputs=[A, B], outputs=[tensor.matrix()])
        return node

    def perform(self, node, (A, B), (output, )):
        ret=numpy.solve(A,B)
        output[0]=ret

    def grad(self, (theta, A, B), (gtheta,)):
        raise NotImplementedError()


