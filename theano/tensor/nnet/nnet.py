"""
Provides neural-network specific Ops.

Notes
-----
TODO: factor this out into a neural-network toolbox.

We register all optimization with the gpu tag as we don't
implement all the intermediate case on the GPU (in particular
AdvancedSubtensor). So to make sure it run well on the gpu with
fast_compile, we register them as needed for the GPU. This can be
revisited later when all the intermediate part are on the GPU.

"""
from __future__ import absolute_import, print_function, division
import logging
import numpy
from six.moves import xrange

import theano
from theano import gof
from theano import scalar
from theano.tensor import extra_ops
from theano.gof.opt import copy_stack_trace
from theano.tensor import basic as tensor, subtensor, opt, elemwise
from theano.tensor.type import (values_eq_approx_remove_inf,
                                values_eq_approx_remove_nan)
from theano.compile import optdb
from theano.gof import Apply

from theano.tensor.nnet.sigm import sigmoid, softplus
from theano.gradient import DisconnectedType
from theano.gradient import grad_not_implemented
from theano.tensor.nnet.blocksparse import sparse_block_dot

############
#
# TENSOR OPS
#


class SoftmaxWithBias(gof.Op):
    """
    An L{Op} for the output of neural-net multiclass classifiers.

    Attributes
    ----------
    x : a matrix of floats (32 or 64)
    b : a [row] vector of floats (32 or 64), length is number of cols in x

    This L{Op}'s output is softmax(x+b).
    softmax(x[i]) is the i'th distribution over len(x[i]) options.

    """

    nin = 2
    nout = 1
    __props__ = ()

    def make_node(self, x, b):
        x = tensor.as_tensor_variable(x)
        b = tensor.as_tensor_variable(b)
        if x.type.ndim != 2 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 2-d tensor of floats')
        if b.type.ndim != 1 \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('b must be 1-d tensor of floats')

        sm = x.type()
        return Apply(self, [x, b], [sm])

    def perform(self, node, input_storage, output_storage):
        x, b = input_storage
        if b.shape[0] != x.shape[1]:
            raise ValueError('b must have same number of columns as x')

        # sm = numpy.zeros_like(x)
        # for i in xrange(sm.shape[0]):
            # row = x[i] + b
            # sm[i] = numpy.exp(row - numpy.max(row))
            # sm[i] *= 1.0 / numpy.sum(sm[i])
        # output_storage[0][0] = sm

        if x.size == 0:
            # Numpy doesn't like the max of a zero-sized object.
            output_storage[0][0] = numpy.zeros(x.shape, dtype=x.dtype)
            return

        x_dtype = x.dtype
        # Perform computations in float32 otherwise the result is too imprecise
        if x.dtype == 'float16':
            x = x.astype('float32')

        x_plus_b = x + b[None, :]
        e_x = numpy.exp(x_plus_b - x_plus_b.max(axis=1)[:, None])
        e_x *= 1.0 / e_x.sum(axis=1)[:, None]
        # default for copy is True and we don't need a copy if the
        # data type matches.
        output_storage[0][0] = e_x.astype(x_dtype, copy=False)

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
    def c_code_template(dtype):
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        # TODO: put this into a templated function, in the support code
        # TODO: declare the max of each row as an Op output

        # TODO: set error messages for failures in this code

        # TODO: use this to accept float32 and int32:
        # node.inputs[0].type.dtype_specs()[1]
        init_decl = """
        npy_intp* Nx = PyArray_DIMS(%(x)s);
        npy_intp Sx = 0;
        npy_intp Sb = 0;
        npy_intp Ssm = 0;


        if (PyArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "not a 2d tensor");
            %(fail)s;
        }
        if (PyArray_NDIM(%(b)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(x)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(x)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "not a float");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(b)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(b)s) != NPY_FLOAT))
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
                                                       PyArray_TYPE(%(x)s));
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc sm output");
                %(fail)s
            }
        }
        Sx = PyArray_STRIDES(%(x)s)[1]/sizeof(dtype_%(x)s);
        Sb = PyArray_STRIDES(%(b)s)[0]/sizeof(dtype_%(b)s);
        Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);

        """

        begin_row_loop = """
        for (size_t i = 0; i < Nx[0]; ++i)
        {
            size_t j;
            double sum = 0.0;

            const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * i);
            const dtype_%(b)s* __restrict__ b_i = (dtype_%(b)s*)(PyArray_BYTES(%(b)s));
            dtype_%(sm) s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);

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

        """

        inside_row_loop = """
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

        # Get the vectorized version of exp if it exist
        try:
            vec_exp = theano.scalar.exp.c_code_contiguous_raw(dtype,
                                                              "Nx[1]", "sm_i", "sm_i")
            inside_row_loop_contig = """
            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%%(sm)s row_ij = x_i[j * Sx] +  b_i[j * Sb];
                //std::cout << "2 " << j << " " << row_ij << " " << row_max << "\\n";
                dtype_%%(sm)s sm_ij = row_ij - row_max;
                //std::cout << "3 " << j << " " << sm_ij << "\\n";
                sm_i[j * Ssm] = sm_ij;
            }
            %(vec_exp)s;
            for (j = 0; j < Nx[1]; ++j)
            {
                sum += sm_i[j * Ssm];
            }

            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm] *= sum_inv;
            }

        """ % locals()
            inside_row_loop = """
            if(Ssm == 1){
                %(inside_row_loop_contig)s
            }else{
                %(inside_row_loop)s
            }
            """ % locals()
        except theano.gof.utils.MethodNotDefined:
            pass
        end_row_loop = """
        }
        """

        return (init_decl, begin_row_loop, inside_row_loop, end_row_loop)

    def c_code(self, node, name, inp, out, sub):
        x, b = inp
        sm, = out
        code_template = ''.join(self.c_code_template(
            node.inputs[0].type.dtype_specs()[1]))
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (8,)

softmax_with_bias = SoftmaxWithBias()


class SoftmaxGrad(gof.Op):
    """
    Gradient wrt x of the Softmax Op.

    """

    nin = 2
    nout = 1
    __props__ = ()

    def make_node(self, dy, sm):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        if dy.type.ndim not in (1, 2) \
                or dy.type.dtype not in tensor.float_dtypes:
            raise ValueError('dy must be 1-d or 2-d tensor of floats. Got ',
                             dy.type)
        if dy.ndim == 1:
            dy = tensor.shape_padleft(dy, n_ones=1)
        if sm.ndim == 1:
            sm = tensor.shape_padleft(sm, n_ones=1)
        return Apply(self, [dy, sm], [sm.type()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = numpy.zeros_like(sm)
        # dx[i,j] = - (\sum_k dy[i,k] sm[i,k]) sm[i,j] + dy[i,j] sm[i,j]
        for i in xrange(sm.shape[0]):
            dy_times_sm_i = dy[i] * sm[i]
            dx[i] = dy_times_sm_i - sum(dy_times_sm_i) * sm[i]
        output_storage[0][0] = dx

    def grad(self, inp, grads):
        dy, sm = inp
        g, = grads

        tmp = g + tensor.neg(tensor.sum(g * sm, axis=1).dimshuffle((0, 'x')))
        g_dy = tmp * sm

        tmp2 = tensor.sum(dy * sm, axis=1).dimshuffle((0, 'x'))
        g_sm = tmp * dy - g * tmp2

        return g_dy, g_sm

    def infer_shape(self, node, shape):
        return [shape[1]]

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        dy, sm = inp
        dx, = out
        return '''
        if ((PyArray_TYPE(%(dy)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(dy)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "types should be float or float64");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(sm)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(sm)s) != NPY_FLOAT))
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
                                                        PyArray_TYPE(%(sm)s));
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
    Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    """

    nin = 1
    nout = 1
    __props__ = ()

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        e_x = numpy.exp(x - x.max(axis=1)[:, None])
        sm = e_x / e_x.sum(axis=1)[:, None]
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = softmax_op(x)
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
    def c_code_template(dtype):
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        # TODO: put this into a templated function, in the support code
        # TODO: declare the max of each row as an Op output

        # TODO: set error messages for failures in this code

        # TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        init_decl = """
        npy_intp* Nx = PyArray_DIMS(%(x)s);
        npy_intp Sx1 = 0;
        npy_intp Ssm1 = 0;

        if (PyArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "not a 2d tensor");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(x)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(x)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError, "not a float");
            %(fail)s;
        }

        if ((NULL == %(sm)s)
            || (PyArray_DIMS(%(sm)s)[0] != PyArray_DIMS(%(x)s)[0])
            || (PyArray_DIMS(%(sm)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(sm)s);
            %(sm)s = (PyArrayObject*)PyArray_SimpleNew(2, PyArray_DIMS(%(x)s),
                                                       PyArray_TYPE(%(x)s));
            if(!%(sm)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc sm output");
                %(fail)s
            }
        }
        Sx1 = PyArray_STRIDES(%(x)s)[1]/sizeof(dtype_%(x)s);
        Ssm1 = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);
        """

        begin_row_loop = """
        for (size_t i = 0; i < Nx[0]; ++i)
        {
            size_t j;
            double sum = 0.0;

            const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * i);
            dtype_%(sm) s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);

            dtype_%(sm)s row_max = x_i[0];
            //std::cout << "0 " << row_max << "\\n";
            // Get the maximum value of the row
            for (j = 1; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx1] ;
                //std::cout << "1 " << row_ij << "\\n";
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }

        """

        inside_row_loop = """
            for (j = 0; j < Nx[1]; ++j)
            {
                dtype_%(sm)s row_ij = x_i[j * Sx1] ;
                //std::cout << "2 " << j << " " << row_ij << " " << row_max << "\\n";
                dtype_%(sm)s sm_ij = exp(row_ij - row_max);
                //std::cout << "3 " << j << " " << sm_ij << "\\n";
                sum += sm_ij;
                sm_i[j * Ssm1] = sm_ij;
            }

            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] *= sum_inv;
            }

        """
        # Get the vectorized version of exp if it exist
        try:
            vec_exp = theano.scalar.exp.c_code_contiguous_raw(dtype,
                                                              "Nx[1]", "sm_i", "sm_i")
            inside_row_loop_contig = """
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] = x_i[j * Sx1] - row_max;
            }
            %(vec_exp)s;
            for (j = 0; j < Nx[1]; ++j)
            {
                sum += sm_i[j * Ssm1];
            }

            //cblas_dscal(x.N, 1.0 / sum, &mat_at(s,i,0), s.n);
            double sum_inv = 1.0 / sum;
            for (j = 0; j < Nx[1]; ++j)
            {
                sm_i[j * Ssm1] *= sum_inv;
            }

            """ % locals()

            inside_row_loop = """
            if(Ssm1 == 1){
                %(inside_row_loop_contig)s
            }else{
                %(inside_row_loop)s
            }
            """ % locals()
        except theano.gof.utils.MethodNotDefined:
            pass

        end_row_loop = """
        }
        """
        return (init_decl, begin_row_loop, inside_row_loop, end_row_loop)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        sm, = out
        code_template = ''.join(self.c_code_template(
            node.inputs[0].type.dtype_specs()[1]))
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (3,)

softmax_op = Softmax()


class LogSoftmax(gof.Op):
    """
    LogSoftmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\e^{(\mathbf{x}_j - log{\sum_{k=1}^K e^{\mathbf{x}_k})}}
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    """
    __props__ = ()

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        xdev = x - x.max(axis=1)[:, None]
        lsm = xdev - numpy.log(numpy.sum(numpy.exp(xdev), axis=1,
                               keepdims=True))
        output_storage[0][0] = lsm

    def grad(self, inp, grads):
        x, = inp
        sm = softmax_op(x)
        return [grads[0] - tensor.sum(grads[0], axis=1, keepdims=True) * sm]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, shape):
        return shape

    def c_headers(self):
        return ['<cmath>']

    @staticmethod
    def c_code_template(dtype):
        init_decl = """
          npy_intp* Nx = PyArray_DIMS(%(x)s);
          npy_intp Sx1 = 0;
          npy_intp Ssm1 = 0;

          if (PyArray_NDIM(%(x)s) != 2)
          {
              PyErr_SetString(PyExc_ValueError, "not a 2d tensor");
              %(fail)s;
          }
          if ((PyArray_TYPE(%(x)s) != NPY_DOUBLE) &&
              (PyArray_TYPE(%(x)s) != NPY_FLOAT))
          {
              PyErr_SetString(PyExc_TypeError, "not a float");
              %(fail)s;
          }

          if ((NULL == %(sm)s)
              || (PyArray_DIMS(%(sm)s)[0] != PyArray_DIMS(%(x)s)[0])
              || (PyArray_DIMS(%(sm)s)[1] != PyArray_DIMS(%(x)s)[1]))
          {
              Py_XDECREF(%(sm)s);
              %(sm)s = (PyArrayObject*)PyArray_SimpleNew(
                  2, PyArray_DIMS(%(x)s),
                  PyArray_TYPE(%(x)s));
              if(!%(sm)s) {
                  PyErr_SetString(PyExc_MemoryError,
                       "failed to alloc sm output");
                  %(fail)s
              }
          }
          Sx1 = PyArray_STRIDES(%(x)s)[1]/sizeof(dtype_%(x)s);
          Ssm1 = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);
          """

        begin_row_loop = """
          // minibatch loop
          for (size_t i = 0; i < Nx[0]; ++i)
          {
              size_t j;
              double sum = 0.0;

              const dtype_%(x)s* __restrict__ x_i = (dtype_%(x)s*)(
                  PyArray_BYTES(%(x)s) + PyArray_STRIDES(%(x)s)[0] * i);
              dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*)(
                  PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);

              dtype_%(sm)s row_max = x_i[0];
              // Get the maximum value of the row
              for (j = 1; j < Nx[1]; ++j)
              {
                  dtype_%(sm)s x_ij = x_i[j * Sx1] ;
                  row_max = (x_ij > row_max) ? x_ij : row_max;
              }
              """

        inside_row_loop = """
              // Compute xdev and sum(exp(xdev), axis=1)
              double xdev_exp_row_sum = 0.0;
              for (j = 0; j < Nx[1]; j++)
              {
                  // use sm_i to temporary store xdev
                  sm_i[j * Ssm1] = (dtype_%(sm)s) (x_i[j * Sx1] - row_max);
                  xdev_exp_row_sum += exp(sm_i[j * Ssm1]);
              }

              // Write sm = xdev - log(sum(exp(xdev), axis=1))
              xdev_exp_row_sum = log(xdev_exp_row_sum);
              for (j = 0; j < Nx[1]; ++j)
              {
                  sm_i[j * Ssm1] -= (dtype_%(sm)s) xdev_exp_row_sum;
              }
              """
        end_row_loop = """
          }
          """
        return (init_decl, begin_row_loop, inside_row_loop, end_row_loop)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        sm, = out
        code_template = ''.join(self.c_code_template(
            node.inputs[0].type.dtype_specs()[1]))
        return code_template % dict(locals(), **sub)

    @staticmethod
    def c_code_cache_version():
        return (0,)

logsoftmax_op = LogSoftmax()


# This is not registered in stabilize, as it cause some crossentropy
# optimization to not be inserted.
@opt.register_specialize('stabilize', 'fast_compile')
@gof.local_optimizer([tensor.Elemwise])
def local_logsoftmax(node):
    """
    Detect Log(Softmax(x)) and replace it with LogSoftmax(x)

    Note: only forward pass is affected
    """
    if (isinstance(node.op, tensor.Elemwise) and
            isinstance(node.op.scalar_op, scalar.basic.Log) and
            len(node.inputs) == 1 and
            node.inputs[0].owner is not None and
            isinstance(node.inputs[0].owner.op, Softmax)):
        inVars = node.inputs[0].owner.inputs[0]
        new_op = LogSoftmax()
        ret = new_op(inVars)
        ret .tag.values_eq_approx = values_eq_approx_remove_inf
        copy_stack_trace([node.inputs[0], node.outputs[0]], ret)
        return [ret]


# This is not registered in stabilize, as it cause some crossentropy
# optimization to not be inserted.
@opt.register_specialize('stabilize', 'fast_compile')
@gof.local_optimizer([SoftmaxGrad])
def local_logsoftmax_grad(node):
    """
    Detect Log(Softmax(x))'s grad and replace it with LogSoftmax(x)'s grad

    Note: only grad is affected
    """
    if (isinstance(node.op, SoftmaxGrad) and
        len(node.inputs) == 2 and
        node.inputs[0].owner is not None and
        node.inputs[0].owner.op == tensor.true_div and
        len(node.inputs[0].owner.inputs) >= 2 and
        node.inputs[0].owner.inputs[1].owner is not None and
        node.inputs[0].owner.inputs[1].owner.op == softmax_op and
        node.inputs[1] == node.inputs[0].owner.inputs[1] and
        not (
            # skip if it will be optimized by
            # local_advanced_indexing_crossentropy_onehot_grad
            node.inputs[0].owner.op == tensor.true_div and
            node.inputs[0].owner.inputs[0].owner is not None and
            isinstance(node.inputs[0].owner.inputs[0].owner.op,
                       subtensor.AdvancedIncSubtensor))):
        # get parameters from unoptimized op
        sm = node.inputs[0].owner.inputs[1]
        # sm_input = node.inputs[1].owner.inputs[0]
        grads = node.inputs[0].owner.inputs[0]
        if grads.broadcastable[1] and not sm.broadcastable[1]:
            grads = tensor.alloc(grads, grads.shape[0], sm.shape[1])
        ret = grads - tensor.sum(grads, axis=1, keepdims=True) * sm
        ret.tag.values_eq_approx = values_eq_approx_remove_nan
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


def softmax_graph(c):
    return tensor.exp(c) / tensor.exp(c).sum(axis=-1, keepdims=True)


def softmax(c):
    return softmax_op(c)


def logsoftmax(c):
    return logsoftmax_op(c)


@opt.register_specialize('fast_compile_gpu')
@gof.local_optimizer([softmax_op])
def local_softmax_with_bias(node):
    """
    Try to turn softmax(sum_of_stuff) -> softmax_w_bias(matrix, bias).

    """
    if node.op == softmax_op:
        x, = node.inputs
        if x.owner and x.owner.op == tensor.add:
            vectors = []
            non_vectors = []
            for x_in in x.owner.inputs:
                if list(x_in.type.broadcastable) == [True, False]:
                    # print isinstance(x_in.owner.op,
                    # tensor.DimShuffle) since specialization comes
                    # relatively late in optimization, we don't want to
                    # put in extra DimShuffles un-necessarily.
                    if (x_in.owner and
                            isinstance(x_in.owner.op, tensor.DimShuffle) and
                            list(x_in.owner.inputs[0].type.broadcastable) == [False]):
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
                # we're in business...
                if len(vectors) > 1:
                    vector_sum = tensor.add(*vectors)
                    copy_stack_trace(x_in, vector_sum)
                else:
                    vector_sum = vectors[0]

                if len(non_vectors) > 1:
                    non_vector_sum = tensor.add(*non_vectors)
                    copy_stack_trace(x_in, non_vector_sum)
                else:
                    non_vector_sum = non_vectors[0]

                try:
                    sm_bias = softmax_with_bias(non_vector_sum, vector_sum)
                    copy_stack_trace(node.outputs[0], sm_bias)
                except Exception:
                    # if our arguments have the wrong types, then
                    # forget about it
                    return

                if sm_bias.type == node.outputs[0].type:
                    # This condition is not always true. See the test
                    # nnet/tests/test_nnet.py:T_SoftmaxWithBias.test_broadcast
                    return [sm_bias]


def softmax_simplifier(numerators, denominators):
    for numerator in list(numerators):
        # TODO: a single softmax'd vector??
        if not numerator.type.dtype.startswith('float'):
            continue

        if numerator.ndim != 2:
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
                        # print 'ASDF', denominator.owner.op.new_order
                        # print z.owner.op.axis
                        if z.owner.op.axis == (1,):
                            # print "almost there.. softmax", x, z.owner.inputs[0]
                            if z.owner.inputs[0] is numerator:
                                matching_denom = denominator
                                break
        if matching_denom:
            softmax = softmax_op(x)
            copy_stack_trace(numerator, softmax)
            numerators.remove(numerator)
            denominators.remove(matching_denom)
            numerators.append(softmax)

    return numerators, denominators
opt.local_mul_canonizer.add_simplifier(softmax_simplifier, 'softmax_simplifier')


class CrossentropySoftmaxArgmax1HotWithBias(gof.Op):
    """
    A special compound L{Op} for the output of neural-net classifiers.

    Parameters
    ----------
    x : a matrix of floats (32 or 64)
    b : a [row] vector of floats (32 or 64), length is number of cols in x
    y_idx : a [column] vector of int (32 or 64), length is number of rows in x

    Returns
    -------
    object
        row-wise NLL, softmax(x+b), row-wise argmax of (x+b).

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
    __props__ = ()

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

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
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def perform(self, node, input_storage, output_storage):
        """
        The math, where x is an input vector, and t is a target index:

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
        if any(y_idx < 0):
            raise ValueError("y_i value out of bounds")
        sm = numpy.zeros_like(x)  # softmax
        nll = numpy.zeros(x.shape[0], dtype=node.outputs[0].type.dtype)  # nll(y | softmax(x))
        am = numpy.zeros_like(y_idx)
        for i in xrange(sm.shape[0]):
            # add the bias vector to the i'th row of x
            row = x[i] + b

            # get the maximum value of i'th row for numerically safe
            # softmax / nll
            am[i] = numpy.argmax(row)
            m = row[am[i]]

            # compute the unnormalized softmax, and normalization constant
            sm[i] = numpy.exp(row - m)
            sum_j = numpy.sum(sm[i])  # sum_j(exp(x[j] - m))

            # normalized our softmax
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
    def c_code_template(dtype):
        # this implementation was lifted from
        # /u/bergstrj/cvs/bergstrj/src/feb07/nn.cxx

        # TODO: put this into a templated function, in the support code
        # TODO: declare the max of each row as an Op output

        # TODO: set error messages for failures in this code

        # TODO: use this to accept float32 and int32: node.inputs[0].type.dtype_specs()[1]
        (init_decl, begin_row_loop, inside_row_loop, end_row_loop) = \
            SoftmaxWithBias.c_code_template(dtype)
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
                PyArray_DIMS(%(y_idx)s), PyArray_TYPE(%(x)s));
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
                PyArray_DIMS(%(y_idx)s), PyArray_TYPE(%(y_idx)s));
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
        dtype = node.inputs[0].type.dtype_specs()[1]
        code_template = ''.join(self.c_code_template(dtype))
        return code_template % dict(locals(), **sub)


class CrossentropySoftmax1HotWithBiasDx(gof.Op):
    """
    Gradient wrt x of the CrossentropySoftmaxArgmax1HotWithBias Op.

    """

    nin = 3
    nout = 1
    __props__ = ()

    def make_node(self, dy, sm, y_idx, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        y_idx = tensor.as_tensor_variable(y_idx)
        if (dy.type.ndim > 1 or
                dy.type.dtype not in tensor.float_dtypes):
            raise ValueError('dy must be {0,1}-d tensor of floats', dy.type)
        if (sm.type.ndim != 2 or
                sm.type.dtype not in tensor.float_dtypes):
            raise ValueError('sm must be 2-d tensor of floats', sm.type)
        if (y_idx.type.ndim != 1 or
                y_idx.type.dtype not in tensor.discrete_dtypes):
            raise ValueError('y_idx must be 1-d tensor of [u]ints', y_idx.type)
        return Apply(self, [dy, sm, y_idx], [sm.type()])

    def perform(self, node, input_storage, output_storage):
        dy, sm, y_idx = input_storage
        if any(y_idx < 0):
            raise ValueError("y_i value out of bounds")
        dx = numpy.zeros_like(sm)
        if dy.ndim == 0:
            dy = dy[None]
        incr = int(dy.shape[0] > 1)
        for i in xrange(sm.shape[0]):
            dy_i = dy[i * incr]
            dx[i] = dy_i * sm[i]  # vector scale
            dx[i, y_idx[i]] -= dy_i  # scalar decrement
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
            g_dx * subtensor.AdvancedIncSubtensor()(
                sm, tensor.fill(dy, -1), y_idx_range, y_idx), axis=1)
        g_sm = dy.dimshuffle(0, 'x') * g_dx
        g_y_idx = grad_not_implemented(self, 2, y_idx)
        return [g_dy, g_sm, g_y_idx]

    def c_code_cache_version(self):
        return (6,)

    def c_code(self, node, name, inp, out, sub):
        dnll, sm, y_idx = inp
        dx, = out
        y_idx_type = node.inputs[2].type.dtype_specs()[1]
        return """
        if ((PyArray_TYPE(%(dnll)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(dnll)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "dnll type should be float32 or float64");
            %(fail)s;
        }
        if ((PyArray_TYPE(%(sm)s) != NPY_DOUBLE) &&
            (PyArray_TYPE(%(sm)s) != NPY_FLOAT))
        {
            PyErr_SetString(PyExc_TypeError,
                 "sm type should be float32 or float64");
            %(fail)s;
        }

        // new scope because of variable declaration
        // TODO: proper indentation, but the diff will get messy
        {
        // Get `dnll.shape[0]` or set it to zero if `dnll` is a scalar.
        const npy_intp %(dnll)s_dims0 = (PyArray_NDIM(%(dnll)s) > 0 ?
                                         PyArray_DIMS(%(dnll)s)[0] :
                                         (npy_intp) 0);

        // Get `dnll.strides[0]` and set it to zero if `dnll` is a scalar
        // or a vector with just one element.
        const npy_intp %(dnll)s_strides0 = (%(dnll)s_dims0 > 1 ?
                                            PyArray_STRIDES(%(dnll)s)[0] :
                                            (npy_intp) 0);

        if ((PyArray_NDIM(%(dnll)s) > 1)
            || (PyArray_NDIM(%(sm)s) != 2)
            || (PyArray_NDIM(%(y_idx)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (%(dnll)s_dims0 != PyArray_DIMS(%(sm)s)[0] && %(dnll)s_dims0 > 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] (%%ld) != sm.shape[0] (%%ld)",
                         (long int)%(dnll)s_dims0,
                         (long int)PyArray_DIMS(%(sm)s)[0]);
            %(fail)s;
        }
        if (%(dnll)s_dims0 != PyArray_DIMS(%(y_idx)s)[0] && %(dnll)s_dims0 > 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] (%%ld) != y_idx.shape[0] (%%ld)",
                         (long int)%(dnll)s_dims0,
                         (long int)PyArray_DIMS(%(y_idx)s)[0]);
            %(fail)s;
        }
        if (PyArray_DIMS(%(sm)s)[0] !=
            PyArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "sm.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyArray_DIMS(%(dx)s)[0] != PyArray_DIMS(%(sm)s)[0])
            || (PyArray_DIMS(%(dx)s)[1] != PyArray_DIMS(%(sm)s)[1]))
        {
            if (NULL != %(dx)s) Py_XDECREF(%(dx)s);
            %(dx)s = (PyArrayObject*) PyArray_SimpleNew(2,
                                                        PyArray_DIMS(%(sm)s),
                                                        PyArray_TYPE(%(sm)s));
            if(!%(dx)s) {
                PyErr_SetString(PyExc_MemoryError,
                     "failed to alloc dx output");
                %(fail)s
            }
        }

        for (size_t i = 0; i < PyArray_DIMS(%(dx)s)[0]; ++i)
        {
            const dtype_%(dnll)s dnll_i = ((dtype_%(dnll)s*)(PyArray_BYTES(%(dnll)s) + %(dnll)s_strides0 * i))[0];

            const %(y_idx_type) s y_i = ((%(y_idx_type)s*)(PyArray_BYTES(%(y_idx)s) + PyArray_STRIDES(%(y_idx)s)[0] * i))[0];

            const dtype_%(sm)s* __restrict__ sm_i = (dtype_%(sm)s*)(PyArray_BYTES(%(sm)s) + PyArray_STRIDES(%(sm)s)[0] * i);
            npy_intp Ssm = PyArray_STRIDES(%(sm)s)[1]/sizeof(dtype_%(sm)s);

            dtype_%(dx) s* __restrict__ dx_i = (dtype_%(dx)s*)(PyArray_BYTES(%(dx)s) + PyArray_STRIDES(%(dx)s)[0] * i);
            npy_intp Sdx = PyArray_STRIDES(%(dx)s)[1]/sizeof(dtype_%(dx)s);

            for (size_t j = 0; j < PyArray_DIMS(%(dx)s)[1]; ++j)
            {
                dx_i[j * Sdx] = dnll_i * sm_i[j * Ssm];
            }
            if (y_i >= PyArray_DIMS(%(dx)s)[1] || (y_i < 0))
            {
                PyErr_SetString(PyExc_ValueError, "y_i >= dx dimensions[1] or y_i < 0.");
                %(fail)s;
            }
            dx_i[y_i * Sdx] -= dnll_i;
        }
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
    Returns
    -------
    object
        The cross-entropy, the softmax output, the max probability,
        and the argmax index.

    TODO: Since we are recomputing the argmax,
           we might as well assert that it is correct.

    TODO: Make this entire function is
    unnecessary? e.g. CrossentropySoftmaxArgmax1HotWithBias should return
    the appropriate information (i.e. the max probability)?

    """
    (xent, softmax) = crossentropy_softmax_1hot_with_bias(x, b, y_idx, **kwargs)
    (max_pr, argmax) = tensor.max_and_argmax(softmax, axis=-1)
    return (xent, softmax, max_pr, argmax)


def crossentropy_softmax_max_and_argmax_1hot(x, y_idx, **kwargs):
    b = tensor.zeros_like(x[0, :])
    return crossentropy_softmax_max_and_argmax_1hot_with_bias(x, b, y_idx,
                                                              **kwargs)


class CrossentropyCategorical1HotGrad(gof.Op):

    __props__ = ()

    def make_node(self, g_y, coding_dist, true_one_of_n):
        return Apply(self, [g_y, coding_dist, true_one_of_n],
                     [coding_dist.type()])

    def perform(self, node, inp, out):
        g_y, coding_dist, true_one_of_n = inp
        g_coding_strg, = out
        g_coding = numpy.zeros_like(coding_dist)
        for i in xrange(len(g_y)):
            g_coding[i, true_one_of_n[i]] = (-g_y[i] /
                                             coding_dist[i, true_one_of_n[i]])
        g_coding_strg[0] = g_coding

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

crossentropy_categorical_1hot_grad = CrossentropyCategorical1HotGrad()


class CrossentropyCategorical1Hot(gof.Op):
    """
    Compute the cross entropy between a coding distribution and
    a true distribution of the form [0, 0, ... 0, 1, 0, ..., 0].

    .. math::

        y[i] = - \log(coding_dist[i, one_of_n[i])

    Notes
    -----
    In the case that the coding distribution is the output of a
    softmax, an application of this Op will probably be optimized
    away in favour of one with a C implementation.

    """
    __props__ = ()

    def make_node(self, coding_dist, true_one_of_n):
        """
        Parameters
        ----------
        coding_dist : dense matrix
        true_one_of_n : lvector

        Returns
        -------
        dvector

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

    def infer_shape(self, node, in_shapes):
        return [(in_shapes[0][0],)]

    def grad(self, inp, grads):
        coding, one_of_n = inp
        g_y, = grads
        return [crossentropy_categorical_1hot_grad(g_y, coding, one_of_n),
                grad_not_implemented(self, 1, one_of_n)]

crossentropy_categorical_1hot = CrossentropyCategorical1Hot()


@opt.register_stabilize('fast_compile_gpu')
@opt.register_specialize('fast_compile_gpu')
@gof.optimizer
def crossentropy_to_crossentropy_with_softmax_with_bias(fgraph):
    """
    This is a stabilization optimization.

    Notes
    -----
    Not a local optimization because we are replacing outputs
    from several nodes at once.

    """

    def search_make_one_sub():
        for node in fgraph.toposort():
            if node.op == crossentropy_categorical_1hot:
                nll, = node.outputs
                sm, one_of_n = node.inputs
                if sm.owner and sm.owner.op == softmax_with_bias:
                    x, b = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(
                        x, b, one_of_n)
                    fgraph.replace_all_validate(
                        [(nll, new_nll), (sm, new_sm)],
                        reason="crossentropy_to_crossentropy_with_softmax_with_bias")
                    return True

        return False

    while search_make_one_sub():
        pass
    return


@gof.optimizer
def crossentropy_to_crossentropy_with_softmax(fgraph):
    """
    This is a stabilization optimization that is more general than
    crossentropy_to_crossentropy_with_softmax_with_bias.

    It must be executed after local_softmax_with_bias optimization in
    specialize.

    TODO : This is a stabilization optimization! How to make this more cleanly?

    Notes
    -----
    Not a local optimization because we are replacing outputs from several
    nodes at once.

    """

    def search_make_one_sub():
        for node in fgraph.toposort():
            if node.op == crossentropy_categorical_1hot:
                nll, = node.outputs
                sm, one_of_n = node.inputs
                if sm.owner and sm.owner.op == softmax_op:
                    x, = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(
                        x, tensor.zeros_like(x[0]), one_of_n)
                    fgraph.replace_all_validate(
                        [(nll, new_nll), (sm, new_sm)],
                        reason="crossentropy_to_crossentropy_with_softmax")
                    return True
                if sm.owner and sm.owner.op == softmax_with_bias:
                    x, b = sm.owner.inputs
                    new_nll, new_sm, new_am = crossentropy_softmax_argmax_1hot_with_bias(x, b,
                                                                                         one_of_n)
                    fgraph.replace_all_validate(
                        [(nll, new_nll), (sm, new_sm)],
                        reason="crossentropy_to_crossentropy_with_softmax")
                    return True

        return False

    while search_make_one_sub():
        pass
    return

optdb.register('crossentropy_to_crossentropy_with_softmax',
               crossentropy_to_crossentropy_with_softmax, 2.01,
               'fast_run', 'xent', 'fast_compile_gpu')


@opt.register_specialize(
    'fast_compile_gpu',
    'local_crossentropy_to_crossentropy_with_softmax_grad')  # old name
@gof.local_optimizer([softmax_grad])
def local_softmax_grad_to_crossentropy_with_softmax_grad(node):
    if node.op == softmax_grad:
        g_coding_dist, coding_dist = node.inputs
        if (g_coding_dist.owner and
                g_coding_dist.owner.op == crossentropy_categorical_1hot_grad):
            g_nll, coding_dist, true_one_of_n = g_coding_dist.owner.inputs
            dx = crossentropy_softmax_1hot_with_bias_dx(g_nll, coding_dist,
                                                        true_one_of_n)
            copy_stack_trace(node.outputs[0], dx)
            return [dx]


@opt.register_specialize('fast_compile_gpu')
@gof.local_optimizer([tensor.MaxAndArgmax])
def local_argmax_pushdown(node):
    if isinstance(node.op, tensor.MaxAndArgmax) and node.inputs[0].owner and \
            len(node.outputs[0].clients) > 0 and node.inputs[0].owner.op in \
            (softmax_op, softplus, tensor.exp, tensor.log, tensor.tanh, sigmoid,
             softmax_with_bias):
        if theano.config.warn.argmax_pushdown_bug:
            logging.getLogger('theano.tensor.nnet.nnet').warn(
                "WARNING: there "
                "was a bug in Theano fixed on May 27th, 2010 in this case."
                " I.E. when we take the max of a softplus, softmax, exp, "
                "log, tanh, sigmoid, softmax_with_bias op, we were doing "
                "the max of the parent of the input. To remove this "
                "warning set the Theano flags 'warn.argmax_pushdown_bug' "
                "to False")

    if (isinstance(node.op, tensor.MaxAndArgmax) and
            node.inputs[0].owner and len(node.outputs[0].clients) == 0):
        x_max, x_argmax = node.outputs
        x = node.inputs[0]
        axis = node.op.get_params(node)
        # TODO: Make a list/set of monotonic ops...
        if x.owner and x.owner.op in (softmax_op, softplus, tensor.exp,
                                      tensor.log, tensor.tanh, sigmoid):
            pre_x, = x.owner.inputs
            ret = tensor.max_and_argmax(pre_x, axis)
            copy_stack_trace(x_max, ret)
            return ret
        if x.owner and x.owner.op == softmax_with_bias:
            pre_x, pre_bias = x.owner.inputs
            ret = tensor.max_and_argmax(pre_x +
                                        tensor.DimShuffle(
                                            pre_bias.broadcastable,
                                            ('x', 0))(pre_bias), axis)
            # copy both stack traces
            copy_stack_trace(x_max, ret)
            return ret

# Utility function used by the two next optimizations


def _check_rows_is_arange_len_labels(rows, labels):
    """
    Check that 'rows' is the same node as T.arange(labels.shape[0]).

    Also considers the case where labels.shape[0] is constant and equal
    to 1, and T.arange(labels.shape[0]) has been constant-folded into 0.

    """

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
        if isinstance(stop.owner.op, subtensor.Subtensor):
            shape_subtensor = stop.owner
            if shape_subtensor.op.get_constant_idx(shape_subtensor.inputs,
                                                   allow_partial=True) == [0]:
                shape_var = shape_subtensor.inputs[0]
                if shape_var.owner and shape_var.owner.op == tensor.shape:
                    return shape_var.owner.inputs[0] is labels
        else:
            shape_of = stop.owner.fgraph.shape_feature.shape_of
            return shape_of[labels][0] is stop


def _is_const(z, val, approx=False):
    try:
        maybe = opt.get_scalar_constant_value(z)
    except tensor.NotScalarConstantError:
        return False
    if approx:
        return numpy.allclose(maybe, val)
    else:
        return numpy.all(maybe == val)


@opt.register_specialize('fast_compile_gpu')
@gof.local_optimizer([subtensor.AdvancedSubtensor, tensor.log])
def local_advanced_indexing_crossentropy_onehot(node):
    log = None
    sm = None
    # First case: log(softmax(x))[rows, labels]
    if isinstance(node.op, subtensor.AdvancedSubtensor):
        try:
            log, rows, labels = node.inputs
        except Exception:
            pass
        if log and log.owner and log.owner.op == tensor.log:
            sm = log.owner.inputs[0]

    # Second case: log(softmax(x)[rows, labels])
    elif node.op == tensor.log:
        pre_log = node.inputs[0].owner
        if pre_log and isinstance(pre_log.op, subtensor.AdvancedSubtensor):
            try:
                sm, rows, labels = pre_log.inputs
            except Exception:
                pass

    if sm is not None and sm.owner and sm.owner.op in (softmax_op,
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
                minus_ret = crossentropy_softmax_argmax_1hot_with_bias(x_var,
                                                                       b_var,
                                                                       labels)[0]
                ret = -minus_ret
                copy_stack_trace(node.outputs[0], [minus_ret, ret])
                return [ret]


@opt.register_specialize('fast_compile_gpu')
@gof.local_optimizer([softmax_grad])
def local_advanced_indexing_crossentropy_onehot_grad(node):
    if not (node.op == softmax_grad):
        return

    sm = None
    try:
        d_sm, sm = node.inputs
    except Exception:
        return

    if (sm is not None) and sm.owner and (sm.owner.op in (softmax_op,
                                                          softmax_with_bias)):
        sm_w_bias = local_softmax_with_bias.transform(sm.owner)
        if sm_w_bias:
            assert sm_w_bias[0].owner.op == softmax_with_bias
            x_var, b_var = sm_w_bias[0].owner.inputs
        else:
            x_var = sm.owner.inputs[0]
    else:
        return

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
    if d_sm.owner and isinstance(d_sm.owner.op, subtensor.AdvancedIncSubtensor):
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

            if isinstance(denom.owner.op, subtensor.AdvancedSubtensor):
                # Base case
                adv_subtensor = denom
                # out_grad /= 1.
            elif denom.owner.op == tensor.mul:
                # Try to find the AdvancedSubtensor node mentionned above,
                # and the output gradient
                for i, input in enumerate(denom.owner.inputs):
                    if input.owner and isinstance(input.owner.op,
                                                  subtensor.AdvancedSubtensor):
                        other_inputs = [in_ for (j, in_) in
                                        enumerate(denom.owner.inputs) if j != i]
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
                # else: OK
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
        if num.owner and isinstance(num.owner.op, subtensor.AdvancedIncSubtensor):
            try:
                z, incr, rows, labels = num.owner.inputs
            except Exception:
                return

            # Check z is zeros_like(log(sm))
            if not _is_const(z, 0):
                return
            if z.broadcastable not in [(False, False), (True, False)]:
                    return
            # here we know that we are incrementing a matrix of zeros
            # (or a broadcasted vector).
            # Since d_sm and sm are the inputs of softmax_grad,
            # if the graph is valid, they have the same shape, so we
            # also know that z has the right shape.

            if incr.ndim != 1 or incr.dtype not in tensor.float_dtypes:
                return

            # here we know that we are incrementing some part of
            # matrix z by a vector

            # unless the user has taken care to mark that the data and
            # labels have the same number of rows, we cannot be sure
            # here that len(y) == len(z) However, in the common case
            # that these are predictions and labels it is true.  We
            # leave it to the Op to crash (and the user to complain)
            # if this assumption is ever not true.

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
        ret = crossentropy_softmax_1hot_with_bias_dx(out_grad, sm, labels)
        # The stack trace is not added to output_grad, sm and labels at
        # the moment but may need to be added at a future point
        copy_stack_trace(node.outputs[0], ret)
        return [ret]
    else:
        return


@opt.register_specialize('fast_compile_gpu')
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
                    copy_stack_trace(node.outputs[0], mergeable_client[1])
                    return [mergeable_client[1]]


@opt.register_specialize
@opt.register_stabilize
@opt.register_canonicalize
@gof.local_optimizer([CrossentropySoftmax1HotWithBiasDx])
def local_useless_crossentropy_softmax_1hot_with_bias_dx_alloc(node):
    """
    Replace a CrossentropySoftmax1HotWithBiasDx op, whose incoming gradient is
    an `alloc` of a scalar variable or one that has either broadcastable or
    matching dimensions with the output variable, by one that skips the
    intermediate `alloc`.

    """
    if isinstance(node.op, CrossentropySoftmax1HotWithBiasDx):
        dy, sm, y_idx = node.inputs

        # Those cases are directly handled by the internal broadcasting of the
        # `CrossentropySoftmax1HotWithBiasDx` op.
        if dy.ndim == 0:
            return False
        if dy.ndim == 1 and dy.broadcastable[0]:
            return False

        assert dy.ndim == 1

        if dy.owner is not None and isinstance(dy.owner.op, tensor.Alloc):
            # dz is the input of the Alloc op, i.e. T.alloc(dz, <shape>)
            dz = dy.owner.inputs[0]

            try:
                shape_feature = node.fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            shape_of = shape_feature.shape_of
            same_shape = shape_feature.same_shape

            # Build `dz_broad` explicitly to include extra implicit dimensions.
            dz_broad = (True,) * (dy.ndim - dz.ndim) + dz.broadcastable

            # If we can infer statically that the shape of `sm` and
            # `dy` are the same in dimension `k` or the shape of `dy` is equal
            # to 1 (which triggers the internal broadcasting in
            # `CrossentropySoftmax1HotWithBiasDx`) we do not need to
            # check it at runtime.
            if (dz_broad[0] and
                    not same_shape(sm, dy, dim_x=0, dim_y=0) and
                    shape_of[dy][0] != 1):
                # If `dz` is broadcastable, we need to check whether the shapes
                # of `dy` and `sm` are the same or whether the shape of `dy` is
                # equal to 1.
                cond = tensor.or_(tensor.eq(dy.shape[0], 1),
                                  tensor.eq(dy.shape[0], sm.shape[0]))
                msg = '`sm` and `dy` do not have the same shape.'
                dz = opt.Assert(msg)(dz, cond)

            ret = node.op(dz, sm, y_idx)
            copy_stack_trace(node.outputs[0], ret)
            return [ret]


def binary_crossentropy(output, target):
    """
    Compute the crossentropy of binary random variables.

    Output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.

    Notes
    -----
    We could use the x log y op to support output=0 and output=1.
    The gradient would still be undefined though.

    We do not sum, crossentropy is computed by component.
    TODO : Rewrite as a scalar, and then broadcast to tensor.

    """
    return -(target * tensor.log(output) + (1.0 - target) * tensor.log(1.0 - output))


def categorical_crossentropy(coding_dist, true_dist):
    """
    Return the cross-entropy between an approximating distribution and a true
    distribution.

    .. warning:: THIS FUNCTION IS UNNECESSARILY POLYMORPHIC.
    We ultimately don't want the polymorphism, and will move this function
    to pylearn.algorithms.cost. The 1hot version will be removed.
    The length of the documentation here is a form of code smell.

    The cross entropy between two probability distributions measures the average
    number of bits needed to identify an event from a set of possibilities, if a
    coding scheme is used based on a given probability distribution q, rather
    than the "true" distribution p.

    Mathematically it is defined as follows:

    .. math::

        H(p,q) = - \sum_x p(x) \log(q(x))

    Parameters
    ----------
    coding_dist : a dense matrix
        Each slice along axis represents one distribution.
    true_dist : a dense matrix or sparse matrix or integer vector
        In the case of a matrix argument, each slice along axis represents one
        distribution. In the case of an integer vector argument, each element
        represents the position of the '1' in a 1-of-N encoding.

    Returns
    -------
    tensor of rank one-less-than `coding_dist`
        The cross entropy between each coding and true distribution.

    Notes
    -----
    axis : int
        The dimension over which each distribution runs
        (1 for row distributions, 0 for column distributions).

    """
    if true_dist.ndim == coding_dist.ndim:
        return -tensor.sum(true_dist * tensor.log(coding_dist),
                           axis=coding_dist.ndim - 1)
    elif true_dist.ndim == coding_dist.ndim - 1:
        return crossentropy_categorical_1hot(coding_dist, true_dist)
    else:
        raise TypeError('rank mismatch between coding and true distributions')


class Prepend_scalar_constant_to_each_row(gof.Op):

    __props__ = ()

    def __init__(self, val=0):
        if isinstance(val, float):
            val = scalar.constant(val)
        self.val = val

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.val)

    def make_node(self, mat):
        # check type of input
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

    __props__ = ()

    def make_node(self, val, mat):
        # check type of input
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


def relu(x, alpha=0):
    """
    Compute the element-wise rectified linear activation function.

    .. versionadded:: 0.7.1

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.

    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    # This is probably the fastest implementation for GPUs. Both the forward
    # pass and the gradient get compiled into a single GpuElemwise call.
    # TODO: Check if it's optimal for CPU as well; add an "if" clause if not.
    # TODO: Check if there's a faster way for the gradient; create an Op if so.
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        # We can't use 0.5 and 1 for one and half.  as if alpha is a
        # numpy dtype, they will be considered as float64, so would
        # cause upcast to float64.
        alpha = tensor.as_tensor_variable(alpha)
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)


def h_softmax(x, batch_size, n_outputs, n_classes, n_outputs_per_class,
              W1, b1, W2, b2, target=None):
    """ Two-level hierarchical softmax.

    This function implements a two-layer hierarchical softmax. It is commonly
    used as an alternative of the softmax when the number of outputs is
    important (it is common to use it for millions of outputs). See
    reference [1]_ for more information about the computational gains.

    The `n_outputs` outputs are organized in `n_classes` classes, each class
    containing the same number `n_outputs_per_class` of outputs.
    For an input `x` (last hidden activation), the first softmax layer predicts
    its class and the second softmax layer predicts its output among its class.

    If `target` is specified, it will only compute the outputs of the
    corresponding targets. Otherwise, if `target` is `None`, it will compute
    all the outputs.

    The outputs are grouped in classes in the same order as they are initially
    defined: if `n_outputs=10` and `n_classes=2`, then the first class is
    composed of the outputs labeled `{0,1,2,3,4}` while the second class is
    composed of `{5,6,7,8,9}`. If you need to change the classes, you have to
    re-label your outputs.

    .. versionadded:: 0.7.1

    Parameters
    ----------
    x: tensor of shape (batch_size, number of features)
        the minibatch input of the two-layer hierarchical softmax.
    batch_size: int
        the size of the minibatch input x.
    n_outputs: int
        the number of outputs.
    n_classes: int
        the number of classes of the two-layer hierarchical softmax. It
        corresponds to the number of outputs of the first softmax. See note at
        the end.
    n_outputs_per_class: int
        the number of outputs per class. See note at the end.
    W1: tensor of shape (number of features of the input x, n_classes)
        the weight matrix of the first softmax, which maps the input x to the
        probabilities of the classes.
    b1: tensor of shape (n_classes,)
        the bias vector of the first softmax layer.
    W2: tensor of shape (n_classes, number of features of the input x,
            n_outputs_per_class)
        the weight matrix of the second softmax, which maps the input x to
        the probabilities of the outputs.
    b2: tensor of shape (n_classes, n_outputs_per_class)
        the bias vector of the second softmax layer.
    target: tensor of shape either (batch_size,) or (batch_size, 1)
        (optional, default None)
        contains the indices of the targets for the minibatch
        input x. For each input, the function computes the output for its
        corresponding target. If target is None, then all the outputs are
        computed for each input.

    Returns
    -------
    tensor of shape (`batch_size`, `n_outputs`) or (`batch_size`, 1)
        Output tensor of the two-layer hierarchical softmax for input `x`.
        Depending on argument `target`, it can have two different shapes.
        If `target` is not specified (`None`), then all the outputs are
        computed and the returned tensor has shape (`batch_size`, `n_outputs`).
        Otherwise, when `target` is specified, only the corresponding outputs
        are computed and the returned tensor has thus shape (`batch_size`, 1).

    Notes
    -----
    The product of `n_outputs_per_class` and `n_classes` has to be greater or
    equal to `n_outputs`. If it is strictly greater, then the irrelevant
    outputs will be ignored.
    `n_outputs_per_class` and `n_classes` have to be the same as the
    corresponding dimensions of the tensors of `W1`, `b1`, `W2` and `b2`.
    The most computational efficient configuration is when
    `n_outputs_per_class` and `n_classes` are equal to the square root of
    `n_outputs`.

    Examples
    --------
    The following example builds a simple hierarchical softmax layer.

    >>> import numpy as np
    >>> import theano
    >>> from theano import tensor
    >>> from theano.tensor.nnet import h_softmax
    >>>
    >>> # Parameters
    >>> batch_size = 32
    >>> n_outputs = 100
    >>> dim_x = 10  # dimension of the input
    >>> n_classes = int(np.ceil(np.sqrt(n_outputs)))
    >>> n_outputs_per_class = n_classes
    >>> output_size = n_outputs_per_class * n_outputs_per_class
    >>>
    >>> # First level of h_softmax
    >>> floatX = theano.config.floatX
    >>> W1 = theano.shared(
    ...     np.random.normal(0, 0.001, (dim_x, n_classes)).astype(floatX))
    >>> b1 = theano.shared(np.zeros((n_classes,), floatX))
    >>>
    >>> # Second level of h_softmax
    >>> W2 = np.random.normal(0, 0.001,
    ...     size=(n_classes, dim_x, n_outputs_per_class)).astype(floatX)
    >>> W2 = theano.shared(W2)
    >>> b2 = theano.shared(np.zeros((n_classes, n_outputs_per_class), floatX))
    >>>
    >>> # We can now build the graph to compute a loss function, typically the
    >>> # negative log-likelihood:
    >>>
    >>> x = tensor.imatrix('x')
    >>> target = tensor.imatrix('target')
    >>>
    >>> # This only computes the output corresponding to the target.
    >>> # The complexity is O(n_classes + n_outputs_per_class).
    >>> y_hat_tg = h_softmax(x, batch_size, output_size, n_classes,
    ...                      n_outputs_per_class, W1, b1, W2, b2, target)
    >>>
    >>> negll = -tensor.mean(tensor.log(y_hat_tg))
    >>>
    >>> # We may need to compute all the outputs (at test time usually):
    >>>
    >>> # This computes all the outputs.
    >>> # The complexity is O(n_classes * n_outputs_per_class).
    >>> output = h_softmax(x, batch_size, output_size, n_classes,
    ...                    n_outputs_per_class, W1, b1, W2, b2)


    References
    ----------
    .. [1] J. Goodman, "Classes for Fast Maximum Entropy Training,"
        ICASSP, 2001, <http://arxiv.org/abs/cs/0108006>`.
    """

    # First softmax that computes the probabilities of belonging to each class
    class_probs = theano.tensor.nnet.softmax(tensor.dot(x, W1) + b1)

    if target is None:  # Computes the probabilites of all the outputs

        # Second softmax that computes the output probabilities
        activations = tensor.tensordot(x, W2, (1, 1)) + b2
        output_probs = theano.tensor.nnet.softmax(
            activations.reshape((-1, n_outputs_per_class)))
        output_probs = output_probs.reshape((batch_size, n_classes, -1))
        output_probs = class_probs.dimshuffle(0, 1, 'x') * output_probs
        output_probs = output_probs.reshape((batch_size, -1))
        # output_probs.shape[1] is n_classes * n_outputs_per_class, which might
        # be greater than n_outputs, so we ignore the potential irrelevant
        # outputs with the next line:
        output_probs = output_probs[:, :n_outputs]

    else:  # Computes the probabilities of the outputs specified by the targets

        target = target.flatten()

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_outputs_per_class

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot(
            W2.dimshuffle('x', 0, 1, 2), x.dimshuffle(0, 'x', 1),
            tensor.zeros((batch_size, 1), dtype='int32'), b2,
            target_classes.dimshuffle(0, 'x'))

        output_probs = theano.tensor.nnet.softmax(activations.dimshuffle(0, 2))
        target_class_probs = class_probs[tensor.arange(batch_size),
                                         target_classes]
        output_probs = output_probs[tensor.arange(batch_size),
                                    target_outputs_in_class]
        output_probs = target_class_probs * output_probs

    return output_probs


def elu(x, alpha=1):
    """
    Compute the element-wise exponential linear activation function.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar


    Returns
    -------
    symbolic tensor
        Element-wise exponential linear activation function applied to `x`.

    References
    -----
    .. [1] Djork-Arne Clevert,  Thomas Unterthiner, Sepp Hochreiter
        "Fast and Accurate Deep Network Learning by
        Exponential Linear Units (ELUs)" <http://arxiv.org/abs/1511.07289>`.
    """
    return tensor.switch(x > 0, x, alpha * (tensor.exp(x) - 1))


class ScalarSoftsign(theano.scalar.UnaryScalarOp):
    """
    Softsign activation function
    :math:`\\varphi(\\mathbf{x}) = \\frac{1}{1+|x|}`

    """
    @staticmethod
    def static_impl(x):
        return x / (1.0 + abs(x))

    def impl(self, x):
        return ScalarSoftsign.static_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if 'float' in x.type.dtype:
            d = (1.0 + abs(x))
            return [gz / (d * d)]
        else:
            return NotImplemented

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [theano.scalar.float32,
                                   theano.scalar.float64]:
            return "%(z)s = %(x)s / (1.0+fabs(%(x)s));" % locals()
        raise NotImplementedError('only floating point x is implemented')

scalar_softsign = ScalarSoftsign(theano.scalar.upgrade_to_float,
                                 name='scalar_softsign')
softsign = elemwise.Elemwise(scalar_softsign, name='softsign')


def confusion_matrix(actual, pred):
    """
    Computes the confusion matrix of given vectors containing
    actual observations and predicted observations.

    Parameters
    ----------
    actual : 1-d tensor variable
    pred : 1-d tensor variable

    Returns
    -------
    conf_mat : Confusion matrix of actual and predictions observations as shown below.

               | Predicted
    ___________|___________
       Actual  |
               |

    order : 1-d array of order of entries in rows and columns

    Examples
    --------
    >>> import theano
    >>> from theano.tensor.nnet import confusion_matrix

    >>> x = theano.tensor.vector()
    >>> y = theano.tensor.vector()
    >>> f = theano.function([x, y], confusion_matrix(x, y))
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> print(f(y_true, y_pred))
    [array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]]), array([ 0.,  1.,  2.])]
    """
    if actual.ndim != 1:
        raise ValueError('actual must be 1-d tensor variable')
    if pred.ndim != 1:
        raise ValueError('pred must be 1-d tensor variable')

    order = extra_ops.Unique(False, False, False)(tensor.concatenate([actual, pred]))

    colA = actual.dimshuffle(0, 'x')
    colP = pred.dimshuffle(0, 'x')

    oneHotA = tensor.eq(colA, order).astype('int64')
    oneHotP = tensor.eq(colP, order).astype('int64')

    conf_mat = tensor.dot(oneHotA.T, oneHotP)
    return [conf_mat, order]
