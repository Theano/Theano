import numpy
import scipy.sparse

from theano import gof, tensor
from theano.tensor.opt import register_specialize
from theano.sparse.basic import (
    as_sparse_variable, SparseType, add_s_s, neg,
    mul_s_s, mul_s_d, dot,
    CSMProperties, CSM,
    _is_sparse_variable, _is_dense_variable, CSC, CSR,
    csm_properties, csm_data, csm_indices, csm_indptr, csm_shape,
    _is_sparse,
    # To maintain compatibility
    Remove0, remove0,
    Cast, bcast, wcast, icast, lcast, fcast, dcast, ccast, zcast,
    HStack, hstack, VStack, vstack,
    AddSSData, add_s_s_data,
    MulSV, mul_s_v,
    structured_monoid,
    structured_sigmoid, structured_exp, structured_log, structured_pow,
    structured_minimum, structured_maximum, structured_add,
    StructuredAddSV, structured_add_s_v,
    SamplingDot, sampling_dot)

# Probability Ops are currently back in sandbox, because they do not respect
# Theano's Op contract, as their behaviour is not reproducible: calling
# the perform() method twice with the same argument will yield different
# results.
#from theano.sparse.basic import (
#    Multinomial, multinomial, Poisson, poisson,
#    Binomial, csr_fbinomial, csc_fbinomial, csr_dbinomial, csc_dbinomial)

# Also for compatibility
from theano.sparse.opt import (
    MulSDCSC, mul_s_d_csc, MulSDCSR, mul_s_d_csr,
    MulSVCSR, mul_s_v_csr,
    StructuredAddSVCSR, structured_add_s_v_csr,
    SamplingDotCSR, sampling_dot_csr,
    local_mul_s_d, local_mul_s_v,
    local_structured_add_s_v, local_sampling_dot_csr)


# Alias to maintain compatibility
EliminateZeros = Remove0
eliminate_zeros = remove0


# Probability
class Poisson(gof.op.Op):
    """Return a sparse having random values from a Poisson density
    with mean from the input.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Theano's contract for Ops

    :param x: Sparse matrix.

    :return: A sparse matrix of random integers of a Poisson density
             with mean of `x` element wise.
    """

    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x = as_sparse_variable(x)
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, (x, ), (out, )):
        assert _is_sparse(x)
        assert x.format in ["csr", "csc"]
        out[0] = x.copy()
        out[0].data = numpy.asarray(numpy.random.poisson(out[0].data),
                                    dtype=x.dtype)
        out[0].eliminate_zeros()

    def grad(self, inputs, outputs_gradients):
        return [None]

    def infer_shape(self, node, ins_shapes):
        return ins_shapes

    def __str__(self):
        return self.__class__.__name__
poisson = Poisson()


class Binomial(gof.op.Op):
    """Return a sparse matrix having random values from a binomial
    density having number of experiment `n` and probability of succes
    `p`.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Theano's contract for Ops

    :param n: Tensor scalar representing the number of experiment.
    :param p: Tensor scalar representing the probability of success.
    :param shape: Tensor vector for the output shape.

    :return: A sparse matrix of integers representing the number
             of success.
    """

    def __init__(self, format, dtype):
        self.format = format
        self.dtype = dtype

    def __eq__(self, other):
        return ((type(self) == type(other)) and
                self.format == other.format and
                self.dtype == other.dtype)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.format) ^ hash(self.dtype)

    def make_node(self, n, p, shape):
        n = tensor.as_tensor_variable(n)
        p = tensor.as_tensor_variable(p)
        shape = tensor.as_tensor_variable(shape)
        return gof.Apply(self, [n, p, shape],
                         [SparseType(dtype=self.dtype,
                                     format=self.format).make_variable()])

    def perform(self, node, (n, p, shape, ), (out, )):
        binomial = numpy.random.binomial(n, p, size=shape)
        csx_matrix = getattr(scipy.sparse, self.format + '_matrix')
        out[0] = csx_matrix(binomial, dtype=self.dtype)

    def grad(self, (n, p, shape, ), (gz,)):
        return None, None, None

    def infer_shape(self, node, ins_shapes):
        return [(node.inputs[2][0], node.inputs[2][1])]

    def __str__(self):
        return self.__class__.__name__

csr_fbinomial = Binomial('csr', 'float32')
csc_fbinomial = Binomial('csc', 'float32')
csr_dbinomial = Binomial('csr', 'float64')
csc_dbinomial = Binomial('csc', 'float64')


class Multinomial(gof.op.Op):
    """Return a sparse matrix having random values from a multinomial
    density having number of experiment `n` and probability of succes
    `p`.

    WARNING: This Op is NOT deterministic, as calling it twice with the
    same inputs will NOT give the same result. This is a violation of
    Theano's contract for Ops

    :param n: Tensor type vector or scalar representing the number of
              experiment for each row. If `n` is a scalar, it will be
              used for each row.
    :param p: Sparse matrix of probability where each row is a probability
              vector representing the probability of succes. N.B. Each row
              must sum to one.

    :return: A sparse matrix of random integers from a multinomial density
             for each row.

    :note: It will works only if `p` have csr format.
    """

    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, n, p):
        n = tensor.as_tensor_variable(n)
        p = as_sparse_variable(p)
        assert p.format in ["csr", "csc"]

        return gof.Apply(self, [n, p], [p.type()])

    def perform(self, node, (n, p), (out, )):
        assert _is_sparse(p)

        if p.format != 'csr':
            raise NotImplemented()

        out[0] = p.copy()

        if n.ndim == 0:
            for i in xrange(p.shape[0]):
                k, l = p.indptr[i], p.indptr[i + 1]
                out[0].data[k:l] = numpy.random.multinomial(n, p.data[k:l])
        elif n.ndim == 1:
            if n.shape[0] != p.shape[0]:
                raise ValueError('The number of element of n must be '
                                 'the same as the number of row of p.')
            for i in xrange(p.shape[0]):
                k, l = p.indptr[i], p.indptr[i + 1]
                out[0].data[k:l] = numpy.random.multinomial(n[i], p.data[k:l])

    def grad(self, inputs, outputs_gradients):
        return [None, None]

    def infer_shape(self, node, ins_shapes):
        return [ins_shapes[1]]

    def __str__(self):
        return self.__class__.__name__
multinomial = Multinomial()
