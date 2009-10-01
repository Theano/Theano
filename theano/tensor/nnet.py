"""Provides neural-network specific Ops.

:note: TODO: factor this out into a neural-network toolbox.
"""

from theano import gof
from theano import scalar
from theano import printing
from theano.printing import pprint
import basic as tensor
import elemwise
import numpy
import opt
from theano.compile import optdb

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
    def c_code_cache_version(self):
        return (1,)
scalar_sigmoid = ScalarSigmoid(scalar.upgrade_to_float, name='scalar_sigmoid')
sigmoid = elemwise.Elemwise(scalar_sigmoid, name='sigmoid')

pprint.assign(sigmoid, printing.FunctionPrinter('sigmoid'))


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
    def c_code_cache_version(self):
        return (1,)
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float, name='scalar_softplus')
softplus = elemwise.Elemwise(scalar_softplus, name='softplus')

pprint.assign(softplus, printing.FunctionPrinter('softplus'))


############
#
# TENSOR OPS
#

class SoftmaxWithBias(gof.Op):
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
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)

    def make_node(self, x, b):
        x = tensor.as_tensor_variable(x)
        b = tensor.as_tensor_variable(b)
        if x.type.ndim != 2 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('x must be 2-d tensor of floats')
        if b.type.ndim != 1 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('b must be 1-d tensor of floats')

        sm = x.type.make_variable()
        return gof.Apply(self, [x, b], [sm])

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
        dx = softmax_grad(g_sm, sm)
        db = tensor.sum(dx, axis = 0)
        return dx, db

    def c_headers(self):
        return ['<iostream>','<cmath>']

    def c_code_cache_version(self):
        return (3,)
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
        if ((%(x)s->descr->type_num != PyArray_DOUBLE)&&(%(x)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "a not float");
            %(fail)s;
        }
        if ((%(b)s->descr->type_num != PyArray_DOUBLE) && (%(b)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "b not float");
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

            const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(%(x)s->data + %(x)s->strides[0] * i);
            const dtype_%(b)s* __restrict__ b_i = (dtype_%(b)s*)(%(b)s->data);
            dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*)(%(sm)s->data + %(sm)s->strides[0] * i);
        """

        inside_row_loop = """
            npy_intp Sx = %(x)s->strides[1]/sizeof(dtype_%(x)s);
            npy_intp Sb = %(b)s->strides[0]/sizeof(dtype_%(b)s);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(dtype_%(sm)s);

            size_t row_max_j=0;
            dtype_%(sm)s row_max = x_i[0] + b_i[0];
            // Get the maximum value of the row
            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] +  b_i[j * Sb];
//                std::cout << "1" << row_ij << "\\n";
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }

            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] +  b_i[j * Sb];
//                std::cout << "2" << row_ij << "\\n";
                dtype_%(sm)s sm_ij = exp(row_ij - row_max);
//                std::cout << "3" << sm_ij << "\\n";
                sum += sm_ij;
                sm_i[j * Ssm] = sm_ij;
            }
            if (std::isinf(sum))
            {
                //that was our best...
                PyErr_SetString(PyExc_ValueError, "softmax is impossible (inf)!");
                %(fail)s;
            }

            if (0.0 == sum)
            {
                //that was our best...
                PyErr_SetString(PyExc_ValueError, "softmax is impossible (zero)!");
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



class SoftmaxGrad(gof.Op):
    """Gradient wrt x of the Softmax Op"""
    nin = 2
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        return gof.Apply(self, [dy, sm], [sm.type.make_variable()])

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

    def c_code_cache_version(self):
        return (3,)
    def c_code(self, node, name, (dy, sm), (dx,), sub):
        return '''
        if ((%(dy)s->descr->type_num != PyArray_DOUBLE) && (%(dy)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "types should be float or float64");
            %(fail)s;
        }
        if ((%(sm)s->descr->type_num != PyArray_DOUBLE) && (%(sm)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "types should be float or float64");
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
            const dtype_%(dy)s* __restrict__ dy_i = (dtype_%(dy)s*) (%(dy)s->data + %(dy)s->strides[0] * i);
            npy_intp Sdy = %(dy)s->strides[1]/sizeof(dtype_%(dy)s);
            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*) (%(sm)s->data + %(sm)s->strides[0] * i);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(dtype_%(sm)s);
            dtype_%(dx)s* __restrict__ dx_i = (dtype_%(dx)s*) (%(dx)s->data + %(dx)s->strides[0] * i);
            npy_intp Sdx = %(dx)s->strides[1]/sizeof(dtype_%(dx)s);

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
softmax_grad = SoftmaxGrad()

class Softmax(gof.Op):
    """
    WRITEME
    """

    nin = 1
    nout = 1
    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 2 \
                or x.type.dtype not in ['float32', 'float64']:
            raise ValueError('x must be 2-d tensor of floats')

        sm = x.type.make_variable()
        return gof.Apply(self, [x], [sm])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        sm = numpy.zeros_like(x)
        for i in xrange(sm.shape[0]):
            row = x[i]
            sm[i] = numpy.exp(row - numpy.max(row))
            sm[i] /= numpy.sum(sm[i])
        output_storage[0][0] = sm

    def grad(self, (x,), (g_sm,)):
        sm = softmax(x)
        return [softmax_grad(g_sm, sm)]
softmax = Softmax()

@opt.register_specialize
@gof.local_optimizer([softmax])
def local_softmax_with_bias(node):
    if node.op == softmax:
        x, = node.inputs
        if x.owner and x.owner.op == tensor.add:
            vectors = []
            non_vectors = []
            for x_in in x.owner.inputs:
                if list(x_in.type.broadcastable) == [True, False]:
                    print isinstance(x_in.owner.op, tensor.DimShuffle)
                    #since specialization comes relatively late in optimization, 
                    # we don't want to put in extra DimShuffles un-necessarily.
                    if x_in.owner and isinstance(x_in.owner.op, tensor.DimShuffle)\
                            and list(x_in.owner.inputs[0].type.broadcastable)==[False]:
                        # cut out the DimShuffle that was broadcasting a vector
                        vectors.append(x_in.owner.inputs[0])
                    else:
                        # insert an extra DimShuffle to correct the old one
                        vectors.append(tensor.DimShuffle((True, False), (1,))(x_in))
                else:
                    non_vectors.append(x_in)

            assert non_vectors #not empty
            if vectors:
                #we're in business...
                if len(vectors)>1:
                  vector_sum = tensor.add(*vectors)
                else:
                  vector_sum = vectors[0]
                #backport
                #vector_sum = tensor.add(*vectors) if len(vectors)>1 else vectors[0]

                if len(non_vectors)>1:
                  non_vector_sum = tensor.add(*non_vectors)
                else:
                  non_vector_sum = non_vectors[0]

                #non_vector_sum = tensor.add(*non_vectors) if len(non_vectors)>1 else non_vectors[0]
                try:
                    sm_bias = softmax_with_bias(non_vector_sum, vector_sum)
                except:
                    #if our arguments have the wrong types, then forget about it
                    return
                return [sm_bias]


class CrossentropySoftmaxArgmax1HotWithBias(gof.Op):
    """A special compound L{Op} for the output of neural-net classifiers.

    :type x: is a matrix of floats (32 or 64)
    :type b: is a [row] vector of floats (32 or 64), length is number of cols in x
    :type y_idx: a [column] vector of int (32 or 64), length is number of rows in x

    :returns:  row-wise NLL, softmax(x+b), row-wise argmax of (x+b)

    @precondition: every entry in y_idx is a valid (non-negative) column index into x

    This L{Op} has three outputs:
     - KL(softmax(x+b), y)
     - softmax(x+b)
     - argmax(x+b)

    softmax(x[i]) is the i'th distribution over len(x[i]) options
    argmax(x) is the index of x's greatest element
    y_idx[i] is an integer index, encoding a 1-hot distribution. 

    In practice, when we are trying to do classification, we have one row in x
    and y_idx per example, and y[i] is the index of the (correct) class of the
    i'th example.

    """
    nin=3
    nout=3
    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)

    def make_node(self, x, b, y_idx):
        x = tensor.as_tensor_variable(x)
        b = tensor.as_tensor_variable(b)
        y_idx = tensor.as_tensor_variable(y_idx)
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
        nll = tensor.TensorType(x.type.dtype,
                y_idx.type.broadcastable).make_variable()
#        nll = TensorType(x.dtype, y.broadcastable)
        sm = x.type.make_variable()
        am = y_idx.type.make_variable()
        return gof.Apply(self, [x, b, y_idx], [nll, sm, am])
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
        nll = numpy.zeros(x.shape[0], dtype=node.outputs[0].type.dtype) #nll(y | softmax(x))
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

    def c_headers(self):
        return ['<iostream>', '<cmath>']

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
            dtype_%(nll)s* __restrict__ nll_i = (dtype_%(nll)s*)(%(nll)s->data + %(nll)s->strides[0] * i);
            %(am_type)s* __restrict__ am_i = (%(am_type)s*) (%(am)s->data + %(am)s->strides[0] * i);
                """,
                inside_row_loop,
                """
            if ((y_i >= %(x)s->dimensions[1]) || (y_i < 0))
            {
                PyErr_SetString(PyExc_ValueError, "y_i value out of bounds");
                %(fail)s;
            }
            nll_i[0] = - x_i[y_i*Sx]
                       - b_i[y_i*Sb]
                       + row_max
                       + log(sum);
            am_i[0] = row_max_j;
                """,
                end_row_loop)


    def c_code_cache_version(self):
        return (2,)
    def c_code(self, node, name, (x, b, y_idx), (nll, sm, am), sub):
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        am_type = y_idx_type
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)

class CrossentropySoftmax1HotWithBiasDx (gof.Op):
    nin=3
    nout=1
    """Gradient wrt x of the CrossentropySoftmax1Hot Op"""
    def __init__(self, **kwargs):
        gof.Op.__init__(self,**kwargs)
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)
    def make_node(self, dy, sm, y_idx,**kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        y_idx = tensor.as_tensor_variable(y_idx)
        return gof.Apply(self, [dy, sm, y_idx],[sm.type.make_variable()])
    def perform(self, node, input_storage, output_storage):
        dy,sm,y_idx = input_storage
        dx = numpy.zeros_like(sm)
        for i in xrange(sm.shape[0]):
            dx[i] = dy[i] * sm[i] #vector scale
            dx[i, y_idx[i]] -= dy[i] #scalar decrement
        output_storage[0][0] = dx
    def grad(self, *args):
        raise NotImplementedError()
    def c_code_cache_version(self):
        return (2,)
    def c_code(self, node, name, (dnll, sm, y_idx), (dx,), sub):
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        return """

        if ((%(dnll)s->descr->type_num != PyArray_DOUBLE) && (%(dnll)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "dnll type should be float32 or float64");
            %(fail)s;
        }
        if ((%(sm)s->descr->type_num != PyArray_DOUBLE) && (%(sm)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "sm type should be float32 or float64");
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
            const dtype_%(dnll)s dnll_i = ((dtype_%(dnll)s*)(%(dnll)s->data + %(dnll)s->strides[0] * i))[0];

            const %(y_idx_type)s y_i = ((%(y_idx_type)s*)(%(y_idx)s->data + %(y_idx)s->strides[0] * i))[0];

            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*)(%(sm)s->data + %(sm)s->strides[0] * i);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(dtype_%(sm)s);

            dtype_%(dx)s* __restrict__ dx_i = (dtype_%(dx)s*)(%(dx)s->data + %(dx)s->strides[0] * i);
            npy_intp Sdx = %(dx)s->strides[1]/sizeof(dtype_%(dx)s);

            for (size_t j = 0; j < %(dx)s->dimensions[1]; ++j)
            {
                dx_i[j * Sdx] = dnll_i * sm_i[j * Ssm];
            }
            if (y_i >= %(dx)s->dimensions[1])
            {
                PyErr_SetString(PyExc_ValueError, "y_i >= dx dimensions[1]");
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

def crossentropy_softmax_max_and_argmax_1hot_with_bias(x, b, y_idx, **kwargs):
    """
    @return: The cross-entropy, the softmax output, the max probability, and the argmax index
    @todo: Since we are recomputing the argmax, we might as well assert that it is correct.
    @todo: Make this entire function is
    unnecessary? e.g. CrossentropySoftmaxArgmax1HotWithBias should return
    the appropriate information (i.e. the max probability)?
    """
    (xent, softmax) = crossentropy_softmax_1hot_with_bias(x, b, y_idx, **kwargs)
    (max_pr, argmax) = tensor.max_and_argmax(softmax)
    return (xent, softmax, max_pr, argmax)
def crossentropy_softmax_max_and_argmax_1hot(x, y_idx, **kwargs):
    b = tensor.zeros_like(x[0,:])
    return crossentropy_softmax_max_and_argmax_1hot_with_bias(x, b, y_idx, **kwargs)

class CrossentropyCategorical1HotGrad(gof.Op):

    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)
    def make_node(self, g_y, coding_dist, true_one_of_n):
        return gof.Apply(self, [g_y, coding_dist, true_one_of_n], [coding_dist.type()])
    def perform(self, node, (g_y, coding_dist, true_one_of_n), (g_coding_strg,)):
        g_coding = numpy.zeros_like(coding_dist)
        for i in xrange(len(g_y)):
            g_coding[i, true_one_of_n[i]] = -g_y[i]/coding_dist[i, true_one_of_n[i]]

        g_coding_strg[0] = g_coding
crossentropy_categorical_1hot_grad = CrossentropyCategorical1HotGrad()

class CrossentropyCategorical1Hot(gof.Op):

    """Compute the cross entropy between a coding distribution and 
    a true distribution of the form [0, 0, ... 0, 1, 0, ..., 0]

    .. math::

        y[i] = - \log(coding_dist[i, one_of_n[i])


    :note:
    In the case that the coding distribution is the output of a softmax, an application of this
    Op will probably be optimized away in favour of one with a C implementation.

    """

    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)
    def make_node(self, coding_dist, true_one_of_n):
        """
        :type coding_dist: dense matrix

        :type true_one_of_n: lvector

        :rtype: dvector
        """
        _coding_dist = tensor.as_tensor_variable(coding_dist)
        _true_one_of_n = tensor.as_tensor_variable(true_one_of_n)
        if _coding_dist.type.ndim != 2:
            raise TypeError('matrix required for argument: coding_dist')
        if _true_one_of_n.type not in (tensor.lvector, tensor.ivector):
            raise TypeError('integer vector required for argument: true_one_of_n'
                    '(got type: %s instead of: %s)' % (_true_one_of_n.type,
                        tensor.lvector))

        return gof.Apply(self, [_coding_dist, _true_one_of_n],
                [tensor.Tensor(dtype=_coding_dist.dtype, broadcastable=[False])()])

    def perform(self, node, (coding, one_of_n), (y_out,)):
        y = numpy.zeros_like(coding[:,0])
        for i in xrange(len(y)):
            y[i] = -numpy.log(coding[i, one_of_n[i]])
        y_out[0] = y
    
    def grad(self, (coding, one_of_n), (g_y,)):
        return [crossentropy_categorical_1hot_grad(g_y, coding, one_of_n), None]
crossentropy_categorical_1hot = CrossentropyCategorical1Hot()

@gof.optimizer
def crossentropy_to_crossentropy_with_softmax(env):
    #not a local optimization because we are replacing outputs from several nodes at once

    def search_make_one_sub():
        for node in env.toposort():
            if node.op == crossentropy_categorical_1hot:
                nll, = node.outputs
                sm, one_of_n = node.inputs
                if sm.owner and sm.owner.op == softmax:
                    x, = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x,
                            tensor.zeros_like(x[0]), one_of_n)
                    env.replace_all_validate([(nll, new_nll),(sm, new_sm)], reason="Merge")
                    return True
                if sm.owner and sm.owner.op == softmax_with_bias:
                    x, b = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x, b,
                            one_of_n)
                    env.replace_all_validate([(nll, new_nll),(sm, new_sm)], reason="Merge")
                    return True

        return False

    while search_make_one_sub():
        pass
    return
optdb.register('XentThing', crossentropy_to_crossentropy_with_softmax, 60.00,
        'fast_run', 'inplace', 'xent')

@gof.local_optimizer([softmax_grad])
def local_crossentropy_to_crossentropy_with_softmax_grad(node):
    if node.op == softmax_grad:
        g_coding_dist, coding_dist = node.inputs
        if g_coding_dist.owner and g_coding_dist.owner.op == crossentropy_categorical_1hot_grad:
            g_nll, coding_dist, true_one_of_n = g_coding_dist.owner.inputs
            dx = crossentropy_softmax_1hot_with_bias_dx(g_nll, coding_dist, true_one_of_n)
            return [dx]
opt.register_specialize(local_crossentropy_to_crossentropy_with_softmax_grad)

@opt.register_specialize
@gof.local_optimizer([tensor._max_and_argmax])
def local_argmax_pushdown(node):
    if node.op == tensor._max_and_argmax:
        x_max, x_argmax = node.outputs
        x, axis = node.inputs
        #TODO: Make a list/set of monotonic ops...
        if x.owner and x.owner.op in (softmax, softplus, tensor.exp, tensor.log, tensor.tanh,
                sigmoid):
            pre_x, = x.owner.inputs
            return tensor._max_and_argmax(pre_x, axis)
        if x.owner and x.owner.op == softmax_with_bias:
            pre_x, pre_bias = x.owner.inputs
            return tensor._max_and_argmax(pre_x+tensor.DimShuffle(pre_bias.broadcastable,
                ('x',0))(pre_bias), axis)



def binary_crossentropy(output, target):
    """
    Compute the crossentropy of binary output wrt binary target.
    @note: We do not sum, crossentropy is computed by component.
    @todo: Rewrite as a scalar, and then broadcast to tensor.
    @todo: This is essentially duplicated as cost.cross_entropy
    @warning: OUTPUT and TARGET are reversed in cost.cross_entropy
    """
    return -(target * tensor.log(output) + (1.0 - target) * tensor.log(1.0 - output))

def categorical_crossentropy(coding_dist, true_dist):
    """
    WARNING: THIS FUNCTION IS UNNECESSARILY POLYMORPHIC.
    We ultimately don't want the polymorphism, and will move this function to pylearn.algorithms.cost.
    The 1hot version will be removed.
    The length of the documentation here is a form of code smell.
    
    Return the cross-entropy between an approximating distribution and a true distribution

    The cross entropy between two probability distributions measures the average number of bits
    needed to identify an event from a set of possibilities, if a coding scheme is used based
    on a given probability distribution q, rather than the "true" distribution p.

    Mathematically it is defined as follows:

    .. math::

        H(p,q) = - \sum_x p(x) \log(q(x))

    :type coding_dist: a dense matrix.
    :param coding_dist: Each slice along axis represents one distribution.

    :type true_dist: a dense matrix or sparse matrix or integer vector.
    :param coding_dist: In the case of a matrix argument, each slice along axis represents one
    distribution.  In the case of an integer vector argument, each element represents the
    position of the '1' in a 1-of-N encoding.

    :type axis: int
    :param axis: the dimension over which each distribution runs. (1 for row distributions, 0
    for column distributions)

    :rtype: tensor of rank one-less-than `coding_dist`
    :returns: the cross entropy between each coding and true distribution.

    """
    if true_dist.ndim == coding_dist.ndim:
        return -theano.sum(true_dist * log(coding_dist), axis=coding_dist.ndim-1)
    elif true_dist.ndim == coding_dist.ndim - 1:
        return crossentropy_categorical_1hot(coding_dist, true_dist)
    else:
        raise TypeError('rank mismatch between coding and true distributions')



class Prepend_scalar_constant_to_each_row(gof.Op):
    def __init__(self, val = 0):
        if isinstance(val, float):
            val = scalar.constant(val)
        self.val = val

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.val == other.val)
    def __hash__(self):
        return tensor.hashtype(self) ^ hash(self.val.value)

    def make_node(self, mat):
        #check type of input
        if not isinstance(mat,gof.Variable) or not mat.type==tensor.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = tensor.as_tensor_variable(mat)
        y = tensor.as_tensor_variable(self.val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")

        node = gof.Apply(op=self, inputs=[mat], outputs=[tensor.matrix()])
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

class Prepend_scalar_to_each_row(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return tensor.hashtype(self)

    def make_node(self, val, mat):
        #check type of input
        if isinstance(val, float):
            val = scalar.constant(val)
        if not isinstance(mat,gof.Variable) or not mat.type==tensor.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = tensor.as_tensor_variable(mat)
        y = tensor.as_tensor_variable(val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")

        node = gof.Apply(op=self, inputs=[val,mat], outputs=[tensor.matrix()])
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

class solve(gof.Op):
    """
    Find the solution to the linear equation Ax=b,
    where A is a 2d matrix and b is a 1d or 2d matrix.
    It use numpy.solve to find the solution.
    """

    def make_node(self, A, b):
        if not isinstance(A, gof.Variable) or not A.type==tensor.matrix().type:
            raise TypeError("We expected that A had a matrix type")
        if not isinstance(B, gof.Variable) or not B.type==tensor.matrix().type:
            raise TypeError("We expected that B had a matrix type")

        node = gof.Apply(op=self, inputs=[A, B], outputs=[tensor.matrix()])
        return node

    def perform(self, node, (A, B), (output, )):
        ret=numpy.solve(A,B)
        output[0]=ret

    def grad(self, (theta, A, B), (gtheta,)):
        raise NotImplementedError()




logsigm_to_softplus = gof.PatternSub(
    (tensor.log, (sigmoid, 'x')),
    (tensor.neg, (softplus, (tensor.neg, 'x'))),
    allow_multiple_clients = True)
log1msigm_to_softplus = gof.PatternSub(
    (tensor.log, (tensor.sub, tensor.constant([[1.0]]), (sigmoid, 'x'))),
    (tensor.neg, (softplus, 'x')),
    allow_multiple_clients = True)

opt.register_specialize(logsigm_to_softplus, name = 'logsigm_to_softplus')
opt.register_specialize(log1msigm_to_softplus, name = 'log1msigm_to_softplus')
