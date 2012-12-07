"""Provides neural-network specific Ops.

:note: TODO: factor this out into a neural-network toolbox.
"""
import logging
import numpy

import theano
from theano import gof
from theano.tensor import basic as tensor
from theano.tensor import elemwise, dmatrix, fmatrix, dvector, fvector
from theano.tensor import opt
from theano.compile import optdb
from theano.gof import Apply

from theano.tensor.nnet.sigm import sigmoid, softplus
from theano.gradient import DisconnectedType
from theano.gradient import grad_not_implemented


############
#
# TENSOR OPS
#

class SoftmaxWithBias(gof.Op):
    """
    An L{Op} for the output of neural-net multiclass classifiers.

    @type x: is a matrix of floats (32 or 64)
    @type b: is a [row] vector of floats (32 or 64),
             length is number of cols in x

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

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, b):
        x = tensor.as_tensor_variable(x)
        b = tensor.as_tensor_variable(b)
        if x.type.ndim != 2 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 2-d tensor of floats')
        if b.type.ndim != 1 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('b must be 1-d tensor of floats')

        sm = x.type.make_variable()
        return Apply(self, [x, b], [sm])

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

    def grad(self, inp, grads):
        x, b = inp
        g_sm, = grads

        if isinstance(g_sm.type, DisconnectedType):
            return [DisconnectedType()(), DisconnectedType()()]

        sm = softmax_with_bias(x, b)
        dx = softmax_grad(g_sm, sm)
        db = tensor.sum(dx, axis=0)
        return dx, db

    def infer_shape(self, node, shape):
        return [shape[0]]

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
        init_decl = """
        npy_intp* Nx = PyArray_DIMS(%(x)s);

        if (PyArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "a not 2d tensor");
            %(fail)s;
        }
        if (PyArray_NDIM(%(b)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
        if ((PyArray_DESCR(%(x)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(x)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "a not float");
            %(fail)s;
        }
        if ((PyArray_DESCR(%(b)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(b)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "b not float");
            %(fail)s;
        }
        if ((PyArray_DIMS(%(x)s)[1] != PyArray_DIMS(%(b)s)[0]))
        {
            PyErr_Format(PyExc_ValueError,
                         "number of columns in x (%%ld) does not match length of b (%%ld)",
                (long int)PyArray_DIMS(%(x)s)[1], (long int)PyArray_DIMS(%(b)s)[0]);
            %(fail)s;
        }

        if ((NULL == %(sm)s)
            || (PyArray_DIMS(%(sm)s)[0] != PyArray_DIMS(%(x)s)[0])
            || (PyArray_DIMS(%(sm)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
            if (NULL != %(sm)s) Py_XDECREF(%(sm)s);
            %(sm)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(x)s),
                                                       type_num_%(x)s);
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc sm output");
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

            const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * i);
            const dtype_%(b)s* __restrict__ b_i = (dtype_%(b)s*)(PyArray_BYTES(%(b)s));
            dtype_%(sm) s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
        """

        inside_row_loop = """
            npy_intp Sx = PyArray_STRIDES(%(x)s)[1]/sizeof(dtype_%(x)s);
            npy_intp Sb = PyArray_STRIDES(%(b)s)[0]/sizeof(dtype_%(b)s);
            npy_intp Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);

            size_t row_max_j=0;
            dtype_%(sm)s row_max = x_i[0] + b_i[0];
            //std::cout << "0 " << row_max << "\\n";
            // Get the maximum value of the row
            for (j = 1; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] +  b_i[j * Sb];
                //std::cout << "1 " << row_ij << "\\n";
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }

            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] +  b_i[j * Sb];
                //std::cout << "2 " << j << " " << row_ij << " " << row_max << "\\n";
                dtype_%(sm)s sm_ij = exp(row_ij - row_max);
                //std::cout << "3 " << j << " " << sm_ij << "\\n";
                sum += sm_ij;
                sm_i[j * Ssm] = sm_ij;
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

    def c_code(self, node, name, inp, out, sub):
        x, b = inp
        sm, = out
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (6,)

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

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        return Apply(self, [dy, sm], [sm.type.make_variable()])

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

    def infer_shape(self, node, shape):
        return [shape[1]]

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        dy, sm = inp
        dx, = out
        return '''
        if ((PyArray_DESCR(%(dy)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(dy)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "types should be float or float64");
            %(fail)s;
        }
        if ((PyArray_DESCR(%(sm)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(sm)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "types should be float or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(dy)s) != 2)
            || (PyArray_NDIM(%(sm)s) != 2))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(dy)s)[0] != PyArray_DIMS(%(sm)s)[0])
        {
            PyErr_SetString(PyExc_ValueError, "dy.shape[0] != sm.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(sm)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(sm)s)[1]))
        {
            Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_SimpleNew(2,
                                                        PyArray_DIMS(%(sm)s),
                                                        type_num_%(sm)s);
            if (!%(dx)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s;
            }
        }

        for (size_t i = 0; i < PyArray_DIMS(%(dx)s)[0]; ++i)
        {
            const dtype_%(dy)s* __restrict__ dy_i = (dtype_%(dy)s*) (PyArray_BYTES(%(dy)s) + PyArray_STRIDES(%(dy)s)[0] * i);
            npy_intp Sdy = PyArray_STRIDES(%(dy)s)[1]/sizeof(dtype_%(dy)s);
            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*) (PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
            npy_intp Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);
            dtype_%(dx) s* __restrict__ dx_i = (dtype_%(dx)s*) (PyArray_BYTES(%(dx)s) + PyArray_STRIDES(%(dx)s)[0] * i);
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);

            double sum_dy_times_sm = 0.;
            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
            {
                dx_i[j * Sdx] = dy_i[j * Sdy] * sm_i[j * Ssm];
                sum_dy_times_sm += dx_i[j * Sdx];
            }
            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
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

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats')
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)
        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        sm = numpy.zeros_like(x)
        for i in xrange(sm.shape[0]):
            row = x[i]
            sm[i] = numpy.exp(row - numpy.max(row))
            sm[i] /= numpy.sum(sm[i])
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = softmax(x)
        return [softmax_grad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, shape):
        return shape

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
        init_decl = """
        npy_intp* Nx = %(x)s->dimensions;

        if (%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "a not 2d tensor");
            %(fail)s;
        }
        if ((%(x)s->descr->type_num != PyArray_DOUBLE) &&
            (%(x)s->descr->type_num != PyArray_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "a not float");
            %(fail)s;
        }

        if ((NULL == %(sm)s)
            || (%(sm)s->dimensions[0] != %(x)s->dimensions[0])
            || (%(sm)s->dimensions[1] != %(x)s->dimensions[1]))
        {
            if (NULL != %(sm)s) Py_XDECREF(%(sm)s);
            %(sm)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(x)s),
                                                       type_num_%(x)s);
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc sm output");
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
            dtype_%(sm) s* __restrict__ sm_i = (dtype_%(sm)s*)(%(sm)s->data + %(sm)s->strides[0] * i);
        """

        inside_row_loop = """
            npy_intp Sx = %(x)s->strides[1]/sizeof(dtype_%(x)s);
            npy_intp Ssm = %(sm)s->strides[1]/sizeof(dtype_%(sm)s);

            size_t row_max_j=0;
            dtype_%(sm)s row_max = x_i[0];
            //std::cout << "0 " << row_max << "\\n";
            // Get the maximum value of the row
            for (j = 1; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] ;
                //std::cout << "1 " << row_ij << "\\n";
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }

            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx] ;
                //std::cout << "2 " << j << " " << row_ij << " " << row_max << "\\n";
                dtype_%(sm)s sm_ij = exp(row_ij - row_max);
                //std::cout << "3 " << j << " " << sm_ij << "\\n";
                sum += sm_ij;
                sm_i[j * Ssm] = sm_ij;
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

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        sm, = out
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (1,)

softmax = Softmax()


@opt.register_specialize
@gof.local_optimizer([softmax])
def local_softmax_with_bias(node):
    """Try to turn softmax(sum_of_stuff) -> softmax_w_bias(matrix, bias)
    """
    if node.op == softmax:
        x, = node.inputs
        if x.owner and x.owner.op == tensor.add:
            vectors = []
            non_vectors = []
            for x_in in x.owner.inputs:
                if list(x_in.type.broadcastable) == [True, False]:
                    # print isinstance(x_in.owner.op,
                    #tensor.DimShuffle) since specialization comes
                    #relatively late in optimization, we don't want to
                    #put in extra DimShuffles un-necessarily.
                    if (x_in.owner and isinstance(x_in.owner.op,
                                                 tensor.DimShuffle)
                 and list(x_in.owner.inputs[0].type.broadcastable) == [False]):
                        # cut out the DimShuffle that was broadcasting a vector
                        vectors.append(x_in.owner.inputs[0])
                    else:
                        # insert an extra DimShuffle to correct the old one
                        vectors.append(tensor.
                            DimShuffle((True, False), (1,))(x_in))
                else:
                    non_vectors.append(x_in)

            # If all the inputs were vectors or broadcasted vectors,
            # we broadcast one of them to be used as a matrix
            if len(non_vectors) == 0:
                assert len(vectors) > 0  # we should have at least 1 input...
                promoted_vector = vectors.pop()
                non_vectors.append(tensor.shape_padleft(promoted_vector))
            assert non_vectors  # not empty

            if vectors:
                #we're in business...
                if len(vectors) > 1:
                    vector_sum = tensor.add(*vectors)
                else:
                    vector_sum = vectors[0]

                if len(non_vectors) > 1:
                    non_vector_sum = tensor.add(*non_vectors)
                else:
                    non_vector_sum = non_vectors[0]

                try:
                    sm_bias = softmax_with_bias(non_vector_sum, vector_sum)
                except Exception:
                    #if our arguments have the wrong types, then
                    #forget about it
                    return

                if sm_bias.type == node.outputs[0].type:
                    #This condition is not always true. See the test
                    #nnet/tests/test_nnet.py:T_SoftmaxWithBias.test_broadcast
                    return [sm_bias]


def softmax_simplifier(numerators, denominators):
    for numerator in list(numerators):
        #TODO: a single softmax'd vector??
        if not numerator.type.dtype.startswith('float'):
            continue

        if not numerator.type.broadcastable == (False, False):
            continue
        if numerator.owner and numerator.owner.op == tensor.exp:
            x = numerator.owner.inputs[0]
        else:
            continue

        matching_denom = None

        for denominator in denominators:
            if denominator.owner and isinstance(denominator.owner.op,
                                                tensor.DimShuffle):
                if denominator.owner.op.new_order == (0, 'x'):
                    z = denominator.owner.inputs[0]
                          # thing getting dimshuffled
                    if z.owner and isinstance(z.owner.op, tensor.Sum):
                        #print 'ASDF', denominator.owner.op.new_order
                        #print z.owner.op.axis
                        if z.owner.op.axis == (1,):
                            #print "almost there.. softmax", x, z.owner.inputs[0]
                            if z.owner.inputs[0] is numerator:
                                matching_denom = denominator
                                break
        if matching_denom:
            numerators.remove(numerator)
            denominators.remove(matching_denom)
            numerators.append(softmax(x))
    return numerators, denominators
opt.local_mul_canonizer.add_simplifier(softmax_simplifier,
     'softmax_simplifier')

if 0:
    @opt.register_specialize
    @gof.local_optimizer([])
    def local_softmax_grad(node):
        '''dy*sm - DimShuffle{0,'x'}(sum{1}(dy*sm))*sm -> softmax_grad(dy,sm)'''
        #TODO what if the signs are changed?
        #TODO and if a scalar is distributed before each of the terms?
        #TODO 'dy' could also be a product
        if node.op == tensor.add and node.out.ndim == 2:
            add_inputs = node.inputs
            # Trying to locate two nodes in the sum:
            #   dy * sm, prod_term
            #   - DimShuffle{0,'x'}(sum{1}(dy*sm))*sm
            prod_term = None
            other_terms = []
            # First, prod_term
            for add_in in add_inputs:
                if (add_in.owner and
                    add_in.owner.op == tensor.mul and
                    prod_term is None):
                    mul_inputs = add_in.owner.inputs
                    if (len(mul_inputs) == 2 and
                        all([mul_in.ndim == 2 for mul_in in mul_inputs])):
                        prod_term = add_in
                    else:
                        other_terms.append(add_in)
                else:
                    other_terms.append(add_in)
            if prod_term is None:
                #print 'no prod_term'
                return
            assert len(other_terms) == len(add_inputs) - 1

            ds_term = None
            rest = []
            for add_in in other_terms:
                if add_in.owner and add_in.owner.op == tensor.neg:
                    neg_input = add_in.owner.inputs[0]
                    if neg_input.owner and neg_input.owner.op == tensor.mul:
                        mul2_inputs = neg_input.owner.inputs
                        if len(mul2_inputs) != 2:
                            rest.append(add_in)
                            #print 'len(mul2_inputs) =', len(mul2_inputs)
                            continue
                        # Try and find DimShuffle(Sum)
                        maybe_ds = None
                        for i, mul2_in in enumerate(mul2_inputs):
                            if mul2_in.owner and isinstance(mul2_in.owner.op,
                                                        elemwise.DimShuffle):
                                maybe_ds = mul2_in
                                maybe_sm = mul2_inputs[1 - i]  # The other one
                        if (maybe_ds is None or
                            maybe_ds.ndim != 2 or
                            maybe_sm.ndim != 2):
                            rest.append(add_in)
                            #print 'maybe_ds =', maybe_ds
                            #if maybe_ds:
                            #    print 'maybe_ds.ndim =', maybe_ds.ndim, ', maybe_sm.ndim =', maybe_sm.ndim
                            continue

                        if maybe_sm is mul_inputs[0]:
                            maybe_dy = mul_inputs[1]
                        elif maybe_sm is mul_inputs[1]:
                            maybe_dy = mul_inputs[0]
                        else:
                            rest.append(add_in)
                            #print 'maybe_sm, maybe_dy =', maybe_sm, maybe_dy
                            #print 'mul_inputs =', mul_inputs
                            continue

                        ds_order = maybe_ds.owner.op.new_order
                        ds_input = maybe_ds.owner.inputs[0]
                        axis = None
                        if ds_input.owner and isinstance(ds_input.owner.op,
                                                         elemwise.Sum):
                            axis = ds_input.owner.op.axis
                            sum_input = ds_input.owner.inputs[0]

                        if ((ds_order != (0, 'x')) or
                            (axis != (1,)) or
                            (sum_input is not prod_term)):
                            rest.append(add_in)
                            #print 'ds_order =', ds_order
                            #print 'axis =', axis
                            #if axis is not None:
                            #    print 'sum_input =', sum_input, ', prod_term =', prod_term
                            #else:
                            #    print 'ds_input.owner =', ds_input.owner
                            #print 'add_in =', add_in
                            continue

                        ds_term = add_in

                    else:
                        #print 'neg_input.owner =', neg_input.owner
                        rest.append(add_in)
                else:
                    #print 'add_in.owner =', add_in.owner
                    rest.append(add_in)

            if ds_term is None:
                #print 'no ds_term'
                return
            if len(rest) == 0:
                return [softmax_grad(maybe_dy, maybe_sm)]
            else:
                return [tensor.add(softmax_grad(maybe_dy, maybe_sm), *rest)]


class CrossentropySoftmaxArgmax1HotWithBias(gof.Op):
    """A special compound L{Op} for the output of neural-net classifiers.

    :type x: is a matrix of floats (32 or 64)
    :type b: is a [row] vector of floats (32 or 64),
             length is number of cols in x
    :type y_idx: a [column] vector of int (32 or 64),
                 length is number of rows in x

    :returns:  row-wise NLL, softmax(x+b), row-wise argmax of (x+b)

    @precondition: every entry in y_idx is a valid (non-negative)
                   column index into x

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
    nin = 3
    nout = 3

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, b, y_idx):
        x = tensor.as_tensor_variable(x)
        b = tensor.as_tensor_variable(b)
        y_idx = tensor.as_tensor_variable(y_idx)
        if x.type.ndim != 2 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 2-d tensor of floats', x.type)
        if b.type.ndim != 1 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('b must be 1-d tensor of floats', b.type)
        if y_idx.type.ndim != 1 \
                or y_idx.type.dtype not in tensor.discrete_dtypes:
            raise ValueError('y_idx must be 1-d tensor of [u]ints', y_idx.type)

#       TODO: Is this correct? It used to be y, not y_idx
        nll = tensor.TensorType(x.type.dtype,
                y_idx.type.broadcastable).make_variable()
#        nll = TensorType(x.dtype, y.broadcastable)
        sm = x.type.make_variable()
        am = y_idx.type.make_variable()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def perform(self, node, input_storage, output_storage):
        """The math, where x is an input vector, and t is a target index:

            softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
            nll(x,t) = -log(softmax(x)[t])

        We compute this by subtracting off the max of x. This avoids
        numerical instability.

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

        sm = numpy.zeros_like(x)  # softmax
        nll = numpy.zeros(x.shape[0], dtype=node.outputs[0].type.
            dtype)  # nll(y | softmax(x))
        am = numpy.zeros_like(y_idx)
        for i in xrange(sm.shape[0]):
            #add the bias vector to the i'th row of x
            row = x[i] + b

            #get the maximum value of i'th row for numerically safe
            #softmax / nll
            am[i] = numpy.argmax(row)
            m = row[am[i]]

            #compute the unnormalized softmax, and normalization constant
            sm[i] = numpy.exp(row - m)
            sum_j = numpy.sum(sm[i])  # sum_j(exp(x[j] - m))

            #normalized our softmax
            sm[i] *= 1.0 / sum_j

            # store the nll
            nll[i] = -row[y_idx[i]] + m + numpy.log(sum_j)

        output_storage[0][0] = nll
        output_storage[1][0] = sm
        output_storage[2][0] = am

    def infer_shape(self, node, shapes):
        x_shp, b_shp, idx_shp = shapes
        nll_shp = (x_shp[0],)
        sm_shp = x_shp
        am_shp = idx_shp
        return [nll_shp, sm_shp, am_shp]

    def connection_pattern(self, node):

        return [[True, True, True],  # x
                [True, True, True],  # b
                [False, False, True]]  # y_idx

    def grad(self, inp, grads):
        x, b, y_idx = inp
        g_nll, g_sm, g_am = grads

        dx_terms = []
        db_terms = []
        d_idx_terms = []

        if not isinstance(g_nll.type, DisconnectedType):
            nll, sm = crossentropy_softmax_1hot_with_bias(x, b, y_idx)
            dx = crossentropy_softmax_1hot_with_bias_dx(g_nll, sm, y_idx)
            db = tensor.sum(dx, axis=[0])
            dx_terms.append(dx)
            db_terms.append(db)

        if not isinstance(g_sm.type, DisconnectedType):
            dx, db = softmax_with_bias.grad((x, b), (g_sm, ))
            dx_terms.append(dx)
            db_terms.append(db)

        if not isinstance(g_am.type, DisconnectedType):
            dx_terms.append(x.zeros_like())
            db_terms.append(b.zeros_like())
            d_idx_terms.append(y_idx.zeros_like())

        def fancy_sum(terms):
            if len(terms) == 0:
                return DisconnectedType()()
            rval = terms[0]
            for term in terms[1:]:
                rval = rval + term
            return rval

        return [fancy_sum(terms) for terms in
                [dx_terms, db_terms, d_idx_terms]]

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
        if (PyArray_NDIM(%(y_idx)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "y_idx not 1d tensor");
            %(fail)s;
        }
        if (PyArray_DIMS(%(x)s)[0] != PyArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                "number of rows in x (%%ld) does not match length of y (%%ld)",
                (long int)PyArray_DIMS(%(x)s)[0],
                (long int)PyArray_DIMS(%(y_idx)s)[0]);
            %(fail)s;
        }

        if ((NULL == %(nll)s) //initial condition
            || (PyArray_DIMS(%(nll)s)[0] != PyArray_DIMS(%(y_idx)s)[0]))
        {
            if (NULL != %(nll)s) Py_XDECREF(%(nll)s);
            %(nll)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(y_idx)s), type_num_%(x)s);
            if(!%(nll)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc nll output");
                %(fail)s;
            }
        }
        if ((NULL == %(am)s)
            || (PyArray_DIMS(%(am)s)[0] != PyArray_DIMS(%(y_idx)s)[0]))
        {
            Py_XDECREF(%(am)s);
            %(am)s = (PyArrayObject*) PyArray_SimpleNew(1,
                PyArray_DIMS(%(y_idx)s), type_num_%(y_idx)s);
            if(!%(am)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc am output");
                %(fail)s;
            }
        }
                """,
                begin_row_loop,
                """
            const %(y_idx_type) s y_i = ((%(y_idx_type)s*)(PyArray_BYTES(%(y_idx)s) + PyArray_STRIDES(%(y_idx)s)[0] * i))[0];
            dtype_%(nll) s* __restrict__ nll_i = (dtype_%(nll)s*)(PyArray_BYTES(%(nll)s) + PyArray_STRIDES(%(nll)s)[0] * i);
            %(am_type)s* __restrict__ am_i = (%(am_type)s*) (PyArray_BYTES(%(am)s) + PyArray_STRIDES(%(am)s)[0] * i);
                """,
                inside_row_loop,
                """
            if ((y_i >= PyArray_DIMS(%(x)s)[1]) || (y_i < 0))
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
        return (5,) + SoftmaxWithBias.c_code_cache_version()

    def c_code(self, node, name, inp, out, sub):
        x, b, y_idx = inp
        nll, sm, am = out
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        am_type = y_idx_type
        code_template = ''.join(self.c_code_template())
        return code_template % dict(locals(), **sub)


class CrossentropySoftmax1HotWithBiasDx (gof.Op):
    nin = 3
    nout = 1
    """Gradient wrt x of the CrossentropySoftmaxArgmax1HotWithBias Op"""
    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, y_idx, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        y_idx = tensor.as_tensor_variable(y_idx)
        if (dy.type.ndim != 1 or
            dy.type.dtype not in tensor.float_dtypes):
            raise ValueError('dy must be 1-d tensor of floats', dy.type)
        if (sm.type.ndim != 2 or
            sm.type.dtype not in tensor.float_dtypes):
            raise ValueError('sm must be 2-d tensor of floats', sm.type)
        if (y_idx.type.ndim != 1 or
            y_idx.type.dtype not in tensor.discrete_dtypes):
            raise ValueError('y_idx must be 1-d tensor of [u]ints', y_idx.type)
        return Apply(self, [dy, sm, y_idx], [sm.type.make_variable()])

    def perform(self, node, input_storage, output_storage):
        dy, sm, y_idx = input_storage
        dx = numpy.zeros_like(sm)
        for i in xrange(sm.shape[0]):
            dx[i] = dy[i] * sm[i]  # vector scale
            dx[i, y_idx[i]] -= dy[i]  # scalar decrement
        output_storage[0][0] = dx

    def infer_shape(self, node, shapes):
        return [shapes[1]]

    def grad(self, inp, grads):
        dy, sm, y_idx = inp
        g_dx, = grads
        # TODO: currently we do not compute the gradient w.r.t. dy, because
        # advanced indexing is not working yet. When it works, do it to avoid
        # potentially misleading behavior in gradient computations! (although
        # typically we should not need the gradient w.r.t. dy).
        y_idx_range = tensor.arange(y_idx.shape[0])
        g_dy = tensor.sum(
                g_dx * tensor.AdvancedIncSubtensor()(
                    sm, tensor.fill(dy, -1), y_idx_range, y_idx),
                axis=1)
        g_sm = dy.dimshuffle(0, 'x') * g_dx
        g_y_idx = grad_not_implemented(self, 2, y_idx)
        return [g_dy, g_sm, g_y_idx]

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        dnll, sm, y_idx = inp
        dx, = out
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        return """

        if ((PyArray_DESCR(%(dnll)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(dnll)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "dnll type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_DESCR(%(sm)s)->type_num != NPY_DOUBLE) &&
            (PyArray_DESCR(%(sm)s)->type_num != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "sm type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_NDIM(%(dnll)s) != 1)
            || (PyArray_NDIM(%(sm)s) != 2)
            || (PyArray_NDIM(%(y_idx)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyArray_DIMS(%(dnll)s)[0] != PyArray_DIMS(%(sm)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] (%%ld) != sm.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(dnll)s)[0],
                         (long int)PyArray_DIMS(%(sm)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(dnll)s)[0] != PyArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] (%%ld) != y_idx.shape[0] (%%ld)",
                         (long int)PyArray_DIMS(%(dnll)s)[0],
                         (long int)PyArray_DIMS(%(y_idx)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(sm)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(sm)s)[1]))
        {
            if (NULL != %(dx)s) Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_SimpleNew(2,
                                                        PyArray_DIMS(%(sm)s),
                                                        type_num_%(sm)s);
            if(!%(dx)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s
            }
        }

        for (size_t i = 0; i < PyArray_DIMS(%(dx)s)[0]; ++i)
        {
            const dtype_%(dnll)s dnll_i = ((dtype_%(dnll)s*)(PyArray_BYTES(%(dnll)s) + PyArray_STRIDES(%(dnll)s)[0] * i))[0];

            const %(y_idx_type) s y_i = ((%(y_idx_type)s*)(PyArray_BYTES(%(y_idx)s) + PyArray_STRIDES(%(y_idx)s)[0] * i))[0];

            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
            npy_intp Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);

            dtype_%(dx) s* __restrict__ dx_i = (dtype_%(dx)s*)(PyArray_BYTES(%(dx)s) + PyArray_STRIDES(%(dx)s)[0] * i);
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);

            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
            {
                dx_i[j * Sdx] = dnll_i * sm_i[j * Ssm];
            }
            if (y_i >= PyArray_DIMS(%(dx)s)[1])
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
    return crossentropy_softmax_argmax_1hot_with_bias(x, b, y_idx,
                                                      **kwargs)[0:2]


def crossentropy_softmax_1hot(x, y_idx, **kwargs):
    b = tensor.zeros_like(x[0, :])
    return crossentropy_softmax_1hot_with_bias(x, b, y_idx, **kwargs)


def crossentropy_softmax_max_and_argmax_1hot_with_bias(x, b, y_idx, **kwargs):
    """
    @return: The cross-entropy, the softmax output, the max probability,
             and the argmax index

    @todo: Since we are recomputing the argmax,
           we might as well assert that it is correct.

    @todo: Make this entire function is
    unnecessary? e.g. CrossentropySoftmaxArgmax1HotWithBias should return
    the appropriate information (i.e. the max probability)?
    """
    (xent, softmax) = crossentropy_softmax_1hot_with_bias(x, b, y_idx,
         **kwargs)
    (max_pr, argmax) = tensor.max_and_argmax(softmax, axis=-1)
    return (xent, softmax, max_pr, argmax)


def crossentropy_softmax_max_and_argmax_1hot(x, y_idx, **kwargs):
    b = tensor.zeros_like(x[0, :])
    return crossentropy_softmax_max_and_argmax_1hot_with_bias(x, b, y_idx,
                                                              **kwargs)


class CrossentropyCategorical1HotGrad(gof.Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, g_y, coding_dist, true_one_of_n):
        return Apply(self, [g_y, coding_dist, true_one_of_n],
                     [coding_dist.type()])

    def perform(self, node, inp, out):
        g_y, coding_dist, true_one_of_n = inp
        g_coding_strg, = out
        g_coding = numpy.zeros_like(coding_dist)
        for i in xrange(len(g_y)):
            g_coding[i, true_one_of_n[i]] = -g_y[i] / coding_dist[i,
                                                        true_one_of_n[i]]
        g_coding_strg[0] = g_coding

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

crossentropy_categorical_1hot_grad = CrossentropyCategorical1HotGrad()


class CrossentropyCategorical1Hot(gof.Op):

    """Compute the cross entropy between a coding distribution and
    a true distribution of the form [0, 0, ... 0, 1, 0, ..., 0]

    .. math::

        y[i] = - \log(coding_dist[i, one_of_n[i])


    :note: In the case that the coding distribution is the output of a
           softmax, an application of this Op will probably be optimized
           away in favour of one with a C implementation.

    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

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
            raise TypeError(
                'integer vector required for argument: true_one_of_n'
                '(got type: %s instead of: %s)' % (_true_one_of_n.type,
                                                   tensor.lvector))

        return Apply(self, [_coding_dist, _true_one_of_n],
                [tensor.Tensor(dtype=_coding_dist.dtype,
                               broadcastable=[False])()])

    def perform(self, node, inp, out):
        coding, one_of_n = inp
        y_out, = out
        y = numpy.zeros_like(coding[:, 0])
        for i in xrange(len(y)):
            y[i] = -numpy.log(coding[i, one_of_n[i]])
        y_out[0] = y

#Enabling this infer_shape method make 2 tests fail:
#theano/tensor/nnet/tests/test_nnet.py:T_CrossentropyCategorical1Hot.
#     {test_softmax_grad_optimizations,test_softmax_grad_optimizations_vector}
# This is caused by the local_fill_to_alloc that call broadcast_like
# that look into the shape feature and return a Rebroadcast instead of an alloc.
# I disable this infer_shape until we fix the optimizations or determine that
# this is not needed anymore and we update the tests.
        # see issue gh-788
#    def infer_shape(self, node, in_shapes):
#        return [(in_shapes[0][0],)]

    def grad(self, inp, grads):
        coding, one_of_n = inp
        g_y, = grads
        return [crossentropy_categorical_1hot_grad(g_y, coding, one_of_n),
                grad_not_implemented(self, 1, one_of_n)]

crossentropy_categorical_1hot = CrossentropyCategorical1Hot()


@opt.register_stabilize
@opt.register_specialize
@gof.optimizer
def crossentropy_to_crossentropy_with_softmax_with_bias(fgraph):
    """This is a stabilization optimization

    :note: not a local optimization because we are replacing outputs
    from several nodes at once

    """

    def search_make_one_sub():
        for node in fgraph.toposort():
            if node.op == crossentropy_categorical_1hot:
                nll, = node.outputs
                sm, one_of_n = node.inputs
                if sm.owner and sm.owner.op == softmax_with_bias:
                    x, b = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x, b,
                            one_of_n)
                    fgraph.replace_all_validate([(nll, new_nll), (sm, new_sm)],
                reason="crossentropy_to_crossentropy_with_softmax_with_bias")
                    return True

        return False

    while search_make_one_sub():
        pass
    return


@gof.optimizer
def crossentropy_to_crossentropy_with_softmax(fgraph):
    """This is a stabilization optimization that is more general then
    crossentropy_to_crossentropy_with_softmax_with_bias

    It must be executed after local_softmax_with_bias optimization in
    specialize

    :todo: This is a stabilization optimization! How to make this more cleanly?

    :note: not a local optimization because we are replacing outputs
           from several nodes at once

    """

    def search_make_one_sub():
        for node in fgraph.toposort():
            if node.op == crossentropy_categorical_1hot:
                nll, = node.outputs
                sm, one_of_n = node.inputs
                if sm.owner and sm.owner.op == softmax:
                    x, = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x,
                            tensor.zeros_like(x[0]), one_of_n)
                    fgraph.replace_all_validate([(nll, new_nll), (sm, new_sm)],
                            reason="crossentropy_to_crossentropy_with_softmax")
                    return True
                if sm.owner and sm.owner.op == softmax_with_bias:
                    x, b = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x, b,
                            one_of_n)
                    fgraph.replace_all_validate([(nll, new_nll), (sm, new_sm)],
                            reason="crossentropy_to_crossentropy_with_softmax")
                    return True

        return False

    while search_make_one_sub():
        pass
    return

optdb.register('crossentropy_to_crossentropy_with_softmax',
               crossentropy_to_crossentropy_with_softmax, 2.01,
               'fast_run', 'xent')


@gof.local_optimizer([softmax_grad])
def local_crossentropy_to_crossentropy_with_softmax_grad(node):
    if node.op == softmax_grad:
        g_coding_dist, coding_dist = node.inputs
        if (g_coding_dist.owner and
            g_coding_dist.owner.op == crossentropy_categorical_1hot_grad):
            g_nll, coding_dist, true_one_of_n = g_coding_dist.owner.inputs
            dx = crossentropy_softmax_1hot_with_bias_dx(g_nll,
                 coding_dist, true_one_of_n)
            return [dx]
opt.register_specialize(local_crossentropy_to_crossentropy_with_softmax_grad)


@opt.register_specialize
@gof.local_optimizer([tensor._max_and_argmax])
def local_argmax_pushdown(node):
    if node.op == tensor._max_and_argmax and node.inputs[0].owner and \
            len(node.outputs[0].clients) > 0 and node.inputs[0].owner.op in \
            (softmax, softplus, tensor.exp, tensor.log, tensor.tanh, sigmoid,
             softmax_with_bias):
        if theano.config.warn.argmax_pushdown_bug:
            logging.getLogger('theano.tensor.nnet.nnet').warn("WARNING: there "
                    "was a bug in Theano fixed on May 27th, 2010 in this case."
                    " I.E. when we take the max of a softplus, softmax, exp, "
                    "log, tanh, sigmoid, softmax_with_bias op, we were doing "
                    "the max of the parent of the input. To remove this "
                    "warning set the Theano flags 'warn.argmax_pushdown_bug' "
                    "to False")

    if (node.op == tensor._max_and_argmax and
        node.inputs[0].owner and len(node.outputs[0].clients) == 0):
        x_max, x_argmax = node.outputs
        x, axis = node.inputs
        #TODO: Make a list/set of monotonic ops...
        if x.owner and x.owner.op in (softmax, softplus, tensor.exp,
                                      tensor.log, tensor.tanh, sigmoid):
            pre_x, = x.owner.inputs
            return tensor._max_and_argmax(pre_x, axis)
        if x.owner and x.owner.op == softmax_with_bias:
            pre_x, pre_bias = x.owner.inputs
            return tensor._max_and_argmax(pre_x +
                                          tensor.DimShuffle(
                                              pre_bias.broadcastable,
                                              ('x', 0))(pre_bias), axis)

# Utility function used by the two next optimizations


def _check_rows_is_arange_len_labels(rows, labels):
    '''Check that 'rows' is the same node as T.arange(labels.shape[0])

    Also considers the case where labels.shape[0] is constant and equal
    to 1, and T.arange(labels.shape[0]) has been constant-folded into 0.
    '''

    if labels.owner and hasattr(labels.owner.fgraph, 'shape_feature'):
        shape_of = labels.owner.fgraph.shape_feature.shape_of
        # TODO: consider cases where shape_of[labels] is constant, and
        # has a value different from 1.
        # This case is harder, as _is_const only accepts a scalar value
        # as second argument, so checking for
        # _is_const(rows, numpy.arange(...)) does not work for the moment.
        if len(shape_of[labels]) == 1 and _is_const(shape_of[labels][0], 1):
            return _is_const(rows, 0)

    if rows.owner and isinstance(rows.owner.op, tensor.ARange):
        start, stop, step = rows.owner.inputs
        if getattr(start, 'data', None) != 0:  # constants will have data
            return False
        if getattr(step, 'data', None) != 1:  # constant step will have data
            return False
        if not stop.owner:
            return False

        # Not sure if that case happens any more after the introduction of
        # ShapeOptimizer, but we keep it if ShapeOptimizer is not present
        if isinstance(stop.owner.op, tensor.Subtensor):
            shape_subtensor = stop.owner
            if list(shape_subtensor.op.idx_list) == [0]:
                shape_var, = shape_subtensor.inputs
                if shape_var.owner and shape_var.owner.op == tensor._shape:
                    return shape_var.owner.inputs[0] is labels
        else:
            shape_of = stop.owner.fgraph.shape_feature.shape_of
            return shape_of[labels][0] is stop


def _is_const(z, val, approx=False):
    try:
        maybe = opt.get_constant_value(z)
    except TypeError:
        return False
    if approx:
        return numpy.allclose(maybe, val)
    else:
        return numpy.all(maybe == val)


@opt.register_specialize
@gof.local_optimizer([])
def local_advanced_indexing_crossentropy_onehot(node):
    log = None
    sm = None
    # First case: log(softmax(x))[rows, labels]
    if isinstance(node.op, tensor.AdvancedSubtensor):
        try:
            log, rows, labels = node.inputs
        except Exception:
            pass
        if log and log.owner and log.owner.op == tensor.log:
            sm = log.owner.inputs[0]

    # Second case: log(softmax(x)[rows, labels])
    if node.op == tensor.log:
        pre_log = node.inputs[0].owner
        if pre_log and isinstance(pre_log.op, tensor.AdvancedSubtensor):
            try:
                sm, rows, labels = pre_log.inputs
            except Exception:
                pass

    if sm is not None and sm.owner and sm.owner.op in (softmax,
                                                       softmax_with_bias):
        sm_w_bias = local_softmax_with_bias.transform(sm.owner)
        if sm_w_bias:
            assert sm_w_bias[0].owner.op == softmax_with_bias
            x_var, b_var = sm_w_bias[0].owner.inputs
        else:
            x_var = sm.owner.inputs[0]
            b_var = tensor.zeros_like(x_var[0])

        # Check that rows == arange(labels.shape[0])
        if _check_rows_is_arange_len_labels(rows, labels):
            if labels.ndim == 1 and x_var.ndim == 2:
                return [-crossentropy_softmax_argmax_1hot_with_bias(x_var,
                                                                    b_var,
                                                                    labels)[0]]


@opt.register_specialize
@gof.local_optimizer([softmax_grad])
def local_advanced_indexing_crossentropy_onehot_grad(node):
    if not (node.op == softmax_grad):
        return

    sm = None
    try:
        d_sm, sm = node.inputs
    except Exception:
        return

    if (sm is not None) and sm.owner and (sm.owner.op in (softmax,
                                                          softmax_with_bias)):
        sm_w_bias = local_softmax_with_bias.transform(sm.owner)
        if sm_w_bias:
            assert sm_w_bias[0].owner.op == softmax_with_bias
            x_var, b_var = sm_w_bias[0].owner.inputs
        else:
            x_var = sm.owner.inputs[0]
    else:
        return

    # If the arg to softmax is a broadcasted vector, d_sm has the form:
    #   DimShuffle{x,0}(Sum{0}(...))
    # we consider what's inside of the sum instead
    vector_softmax = False
    if d_sm.owner and isinstance(d_sm.owner.op, tensor.DimShuffle):
        ds_op = d_sm.owner.op
        if ds_op.input_broadcastable == (False,) and ds_op.new_order == ('x', 0):
            maybe_sum = d_sm.owner.inputs[0]
            if maybe_sum.owner and isinstance(maybe_sum.owner.op, tensor.Sum):
                if sm.broadcastable == (True, False)\
                        and maybe_sum.owner.op.axis == (0,)\
                        and len(maybe_sum.owner.inputs) == 1:
                    vector_softmax = True
                    d_sm = maybe_sum.owner.inputs[0]

    # Two cases are supported:
    # 1. AdvancedIncSubtensor(
    #           zeros_like(softmax(x)),
    #           -out_grad / AdvancedSubtensor(softmax(x), arange(y.shape[0]), y),
    #           arange(y.shape[0]),
    #           y)
    #   which arises from the gradient of log(softmax(x)[arange(y.shape[0]), y])
    #
    # 2. AdvancedIncSubtensor(
    #           zeros_like(log(softmax(x))),
    #           -out_grad,
    #           arange(y.shape[0]),
    #           y)
    #           / softmax(x)
    #   which arises from the gradient of log(softmax(x))[arange(y.shape[0]), y]
    #
    # out_grad represents the gradient of the (final) cost wrt the output.

    #
    # N.B. Regarding clients -- This substitution is important for numerical stability, so we
    # perform the substitution even when intermediate values have multiple clients.
    #

    # First case.
    # After the check for AdvancedIncSubtensor, if anything does not fit with
    # the formula above, there's no way to fit it with the the second case,
    # so we return immediately.
    if d_sm.owner and isinstance(d_sm.owner.op, tensor.AdvancedIncSubtensor):
        try:
            z, incr, rows, labels = d_sm.owner.inputs
        except Exception:
            return
        # Check that z == zeros_like(softmax(x))
        # We know z has the right size because z has the same size as d_sm,
        # and d_sm and sm are both inputs of softmax_grad (so they have
        # the same size).
        if not _is_const(z, 0):
            return

        # In the base case (output gradient = 1), incr is -1./sm[arange(len(y)), y]
        # Here, we are looking for the AdvancedSubtensor term (sm[arange(len(y)), y]),
        # and constructing out_grad by incorporating the other terms.
        # out_grad will be constructed in 3 steps as follow:
        # out_grad = +/- 1. (according to sign)
        # out_grad *= -numerator
        # out_grad /= denominator
        # Then, if out_grad is a scalar, it will be allocated as a vector
        adv_subtensor = None
        out_grad = 1.

        # If there's a 'minus' sign before the whole expression, put it in
        # out_grad and iterate
        if incr.owner and incr.owner.op == tensor.neg:
            out_grad = - out_grad
            incr = incr.owner.inputs[0]

        if incr.owner and incr.owner.op == tensor.true_div:
            num, denom = incr.owner.inputs

            # set out_grad according to the numerator, it may be divided later
            # num should be a vector or a scalar
            if num.ndim == 1 or numpy.all(num.broadcastable):
                out_grad *= -num
            else:
                return

            if not denom.owner:
                return

            if isinstance(denom.owner.op, tensor.AdvancedSubtensor):
                # Base case
                adv_subtensor = denom
                #out_grad /= 1.
            elif denom.owner.op == tensor.mul:
                # Try to find the AdvancedSubtensor node mentionned above,
                # and the output gradient
                for i, input in enumerate(denom.owner.inputs):
                    if input.owner and isinstance(input.owner.op,
                                                  tensor.AdvancedSubtensor):
                        other_inputs = [in_ for (j,
                             in_) in enumerate(denom.owner.inputs) if j != i]
                        if len(other_inputs) == 1:
                            rest = other_inputs[0]
                        else:
                            rest = tensor.mul(*[other_inputs])

                        # Check that rest is a vector or a scalar
                        if rest.ndim == 1 or numpy.all(rest.broadcastable):
                            adv_subtensor = input
                            out_grad /= rest
                            break
            else:
                return

            # The output gradient needs to be a vector
            out_grad = tensor.fill(x_var[:, 0], out_grad)

            if adv_subtensor is not None:
                try:
                    maybe_sm, maybe_rows, \
                        maybe_labels = adv_subtensor.owner.inputs
                except Exception:
                    return

                if (not (maybe_sm is sm and
                         maybe_rows is rows and
                         maybe_labels is labels)):
                    return
                #else: OK
            else:
                return
        else:
            return

        # Check that rows is arange(labels.shape[0])
        if not _check_rows_is_arange_len_labels(rows, labels):
            return
        # else, arguments of AdvancedIncSubtensor are OK,
        # it was really case 1.

    # Second case
    elif d_sm.owner and d_sm.owner.op == tensor.true_div:
        # we're looking for
        # AdvIncSubtensor(zeros, grad_nll, arange(len(y)), y) / softmax
        try:
            num, denom = d_sm.owner.inputs
        except Exception:
            return

        if denom != sm:
            return

        # Check the numerator (AdvancedIncSubtensor)
        if num.owner and isinstance(num.owner.op, tensor.AdvancedIncSubtensor):
            try:
                z, incr, rows, labels = num.owner.inputs
            except Exception:
                return

            # Check z is zeros_like(log(sm))
            if not _is_const(z, 0):
                return
            if z.type not in (dmatrix, fmatrix):
                if not (vector_softmax and z.broadcastable == (True, False)):
                    return
            # here we know that we are incrementing a matrix of zeros
            # (or a broadcasted vector).
            # Since d_sm and sm are the inputs of softmax_grad,
            # if the graph is valid, they have the same shape, so we
            # also know that z has the right shape.

            if incr.type not in (dvector, fvector):
                return

            # here we know that we are incrementing some part of matrix z by a vector

            # unless the user has taken care to mark that the data and labels have the
            # same number of rows, we cannot be sure here that
            # len(y) == len(z)
            # However, in the common case that these are predictions and labels it is true.
            # We leave it to the Op to crash (and the user to complain) if this assumption is
            # ever not true.

            out_grad = -incr

            # Check that rows is arange(labels.shape[0])
            if not _check_rows_is_arange_len_labels(rows, labels):
                return
            # else, arguments of AdvancedIncSubtensor are OK
        else:
            return

        # numerator and denominator are OK,
        # it was really case 2.

    else:
        return

    # Dimension check before substitution
    if labels.ndim == 1 and x_var.ndim == 2:
        return [crossentropy_softmax_1hot_with_bias_dx(out_grad, sm, labels)]
    else:
        return


@opt.register_specialize
@gof.local_optimizer([softmax_with_bias])
def graph_merge_softmax_with_crossentropy_softmax(node):
    if node.op == softmax_with_bias:
        x, b = node.inputs
        for x_client in x.clients:
            if x_client[0].op == crossentropy_softmax_argmax_1hot_with_bias:
                big_client = x_client[0]
                if big_client in [b_client[0] for b_client in b.clients]:
                    xx, bb, ll = big_client.inputs
                    mergeable_client = big_client.op(x, b, ll)
                    return [mergeable_client[1]]


def binary_crossentropy(output, target):
    """
    Compute the crossentropy of binary random variables
    output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.
    @note: we could use the x log y op to support output=0
    @ and output=1. The gradient would still be undefined though.
    @note: We do not sum, crossentropy is computed by component.
    @todo: Rewrite as a scalar, and then broadcast to tensor.
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
        return -tensor.sum(true_dist * tensor.log(coding_dist), axis=coding_dist.ndim-1)
    elif true_dist.ndim == coding_dist.ndim - 1:
        return crossentropy_categorical_1hot(coding_dist, true_dist)
    else:
        raise TypeError('rank mismatch between coding and true distributions')


from theano import scalar


class Prepend_scalar_constant_to_each_row(gof.Op):
    def __init__(self, val=0):
        if isinstance(val, float):
            val = scalar.constant(val)
        self.val = val

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.val == other.val)

    def __hash__(self):
        return tensor.hashtype(self) ^ hash(self.val.data)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.val)

    def make_node(self, mat):
        #check type of input
        x = tensor.as_tensor_variable(mat)
        if not mat.type.broadcastable == (False, False):
            raise TypeError("Expected a matrix as input")
        y = tensor.as_tensor_variable(self.val)
        assert y.ndim == 0
        if x.type.dtype != y.type.dtype:
            TypeError(
                "the value to prepend don't have the same type as the matrix")

        node = Apply(op=self, inputs=[mat], outputs=[mat.type()])
        return node

    def perform(self, node, inp, out):
        mat, = inp
        output, = out
        new_shape = (mat.shape[0], mat.shape[1] + 1)
        if output[0] is None:
            output[0] = numpy.empty(new_shape, dtype=mat.dtype)
            out = output[0]
        else:
            if output[0].shape != new_shape:
                try:
                    output[0].resize(new_shape)
                except Exception:
                    output[0] = numpy.empty(new_shape, dtype=mat.dtype)
            out = output[0]

        out[:, 0].fill(self.val.data)
        out[:, 1:] = mat

    def infer_shape(self, node, in_shapes):
        shp = (in_shapes[0][0], in_shapes[0][1] + 1)
        return [shp]

    def grad(self, inp, grads):
        mat, = inp
        goutput, = grads
        return goutput[:, 1:]


class Prepend_scalar_to_each_row(gof.Op):
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, val, mat):
        #check type of input
        x = tensor.as_tensor_variable(mat)
        if isinstance(val, float):
            val = scalar.constant(val)
        if not mat.type.broadcastable == (False, False):
            raise TypeError("Expected a matrix as input")
        y = tensor.as_tensor_variable(val)
        assert y.ndim == 0
        if x.type.dtype != y.type.dtype:
            TypeError(
                "the value to prepend don't have the same type as the matrix")

        node = Apply(op=self, inputs=[val, mat], outputs=[mat.type()])
        return node

    def perform(self, node, inp, out):
        val, mat = inp
        output, = out
        new_shape = (mat.shape[0], mat.shape[1] + 1)
        if output[0] is None:
            output[0] = numpy.empty(new_shape, dtype=mat.dtype)
            out = output[0]
        else:
            if output[0].shape != new_shape:
                try:
                    output[0].resize(new_shape)
                except Exception:
                    output[0] = numpy.empty(new_shape, dtype=mat.dtype)
            out = output[0]
        out[:, 0].fill(val)
        out[:, 1:] = mat

    def infer_shape(self, node, in_shapes):
        shp = (in_shapes[1][0], in_shapes[1][1] + 1)
        return [shp]

    def grad(self, inp, grads):
        val, mat = inp
        goutput, = grads
        return goutput[:, 0], goutput[:, 1:]

prepend_scalar_to_each_row = Prepend_scalar_to_each_row()
prepend_0_to_each_row = Prepend_scalar_constant_to_each_row(0.)
prepend_1_to_each_row = Prepend_scalar_constant_to_each_row(1.)


#numerically stabilize log softmax (X)
# as  X-X.max(axis=1).dimshuffle(0,'x') - log(exp(X-X.max(axis=1).dimshuffle(0,'x')).sum(axis=1)).dimshuffle(0,'x)
def make_out_pattern(X):
    stabilized_X = X - X.max(axis=1).dimshuffle(0, 'x')
    out_var = stabilized_X - tensor.log(tensor.exp(stabilized_X).sum(
        axis=1)).dimshuffle(0, 'x')
    #tell DEBUG_MODE that it's OK if the original graph produced NaN and the optimized graph does not
    out_var.values_eq_approx = out_var.type.values_eq_approx_remove_nan
    return out_var


local_log_softmax = gof.PatternSub(in_pattern=(tensor.log, (softmax, 'x')),
                                    out_pattern=(make_out_pattern, 'x'),
                                   allow_multiple_clients=True)

#don't do register_stabilize, this is to make local_log_softmax run
#only after another more specific optimization that stabilizes cross entropy
#opt.register_stabilize(local_log_softmax, name = 'local_log_softmax')
opt.register_specialize(local_log_softmax, name='local_log_softmax')
