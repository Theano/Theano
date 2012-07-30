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
    # To maintain compatibility
    Remove0, remove0,
    Cast, bcast, wcast, icast, lcast, fcast, dcast, ccast, zcast,
    HStack, hstack, VStack, vstack,
    AddSSData, add_s_s_data,
    MulSDCSC, mul_s_d_csc, MulSDCSR, mul_s_d_csr,
    Multinomial, multinomial, Poisson, poisson,
    Binomial, csr_fbinomial, csc_fbinomial, csr_dbinomial, csc_dbinomial,
    structured_monoid,
    structured_sigmoid, structured_exp, structured_log, structured_pow,
    structured_minimum, structured_maximum, structured_add,
    MulSV, mul_s_v, MulSVCSR, mul_s_v_csr,
    StructuredAddSV, structured_add_s_v,
    StructuredAddSVCSR, structured_add_s_v_csr,
    SamplingDot, sampling_dot, SamplingDotCSR, sampling_dot_csr)

# Also for compatibility
from theano.sparse.opt import (
    local_mul_s_d, local_mul_s_v,
    local_structured_add_s_v, local_sampling_dot_csr)

# Alias to maintain compatibility
EliminateZeros = Remove0
eliminate_zeros = remove0
