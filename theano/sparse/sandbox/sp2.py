import theano
import numpy
import scipy.sparse

from theano import gof, tensor, scalar, sparse
from theano.tensor import blas

from theano.sparse.basic import (
    as_sparse_variable, SparseType, add_s_s, neg,
    mul_s_s, mul_s_d, dot,
    CSMProperties, CSM, register_specialize,
    _is_sparse_variable, _is_dense_variable, CSC, CSR,
    csm_properties, csm_data, csm_indices, csm_indptr, csm_shape,
    _is_sparse,
    Remove0, remove0,
    # To maintain compatibility
    Cast, bcast, wcast, icast, lcast, fcast, dcast, ccast, zcast,
    HStack, hstack, VStack, vstack,
    AddSSData, add_s_s_data,
    MulSDCSC, mul_s_d_csc, MulSDCSR, mul_s_d_csr,
    Multinomial, multinomial, Poisson, poisson,
    structured_monoid,
    structured_sigmoid, structured_exp, structured_log, structured_pow,
    structured_minimum, structured_maximum, structured_add,
    MulSV, mul_s_v, MulSVCSR, mul_s_v_csr,
    StructuredAddSV, structured_add_s_v,
    StructuredAddSVCSR, structured_add_s_v_csr,
    SamplingDot, sampling_dot, SamplingDotCSR, sampling_dot_csr)

# Alias to maintain compatibility
EliminateZeros = Remove0
eliminate_zeros = remove0


# register a specialization to replace MulSD -> MulSDCSX
@gof.local_optimizer([mul_s_d])
def local_mul_s_d(node):
    if node.op == mul_s_d:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        # y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 2:
            return False
        if svar.type.format == 'csc':
            CSx = CSC
            mul_s_d_csx = mul_s_d_csc
        elif svar.type.format == 'csr':
            CSx = CSR
            mul_s_d_csx = mul_s_d_csr
        else:
            raise NotImplemented()

        c_data = mul_s_d_csx(csm_data(svar), csm_indices(svar),
                             csm_indptr(svar), dvar)

        return [CSx(c_data, csm_indices(svar), csm_indptr(svar),
                    csm_shape(svar))]

    return False
register_specialize(local_mul_s_d)


class Binomial(gof.op.Op):
    # TODO This op is not an equivalent of numpy.random.binomial. In
    # facts, this does not follow a binomial distribution at all.
    # To see it, just try with p = 1.

    # """Return a sparse matrix having random values from a binomial
    # density having number of experiment `n` and probability of succes
    # `p`.

    # :param n: Tensor scalar representing the number of experiment.
    # :param p: Tensor scalar representing the probability of success.
    # :param shape: Tensor vector for the output shape.

    # :return: A sparse matrix of integers representing the number
    #          of success.
    # """

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
        return gof.Apply(self, [n, p, shape], [SparseType(dtype=self.dtype,
                                 format=self.format).make_variable()])

    def perform(self, node, (n, p, shape, ), (out, )):
        N = n * p * shape[0] * shape[1]
        data = numpy.ones(N, dtype=self.dtype)
        row = numpy.random.randint(0, shape[0], N)
        col = numpy.random.randint(0, shape[1], N)

        res = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)

        out[0] = getattr(res, 'to' + self.format)()
        out[0].data = numpy.ones_like(out[0].data)

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


@gof.local_optimizer([mul_s_v])
def local_mul_s_v(node):
    if node.op == mul_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == 'csr':
            CSx = CSR
            mul_s_v_csx = mul_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = csm_properties(svar)

        c_data = mul_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_mul_s_v)


@gof.local_optimizer([structured_add_s_v])
def local_structured_add_s_v(node):
    if node.op == structured_add_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        #y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == 'csr':
            CSx = CSR
            structured_add_s_v_csx = structured_add_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = csm_properties(svar)

        c_data = structured_add_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
register_specialize(local_structured_add_s_v)


# register a specialization to replace SamplingDot -> SamplingDotCsr
@gof.local_optimizer([sampling_dot])
def local_sampling_dot_csr(node):
    if node.op == sampling_dot:
        x, y, p = node.inputs
        if p.type.format == 'csr':
            p_data, p_ind, p_ptr, p_shape = csm_properties(p)

            z_data, z_ind, z_ptr = sampling_dot_csr(x, y, p_data,
                p_ind, p_ptr, p_shape[1])

            return [CSR(z_data, z_ind, z_ptr, p_shape)]
    return False
register_specialize(local_sampling_dot_csr, name='local_sampling_dot_csr')
