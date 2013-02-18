from itertools import izip

import theano
import numpy
from theano import gof, scalar
from theano.sparse import (CSC, CSR, csm_properties, Remove0,
                           register_specialize,
                           csm_grad, csm_grad_c,
                           usmm_csc_dense, usmm)
from theano.sparse import basic as sparse

from basic import _is_sparse_variable


# This is tested in tests/test_basic.py:UsmmTests
local_usmm = gof.opt.PatternSub(
    (theano.tensor.sub, 'z',
     (theano.tensor.mul,
      {'pattern': 'alpha',
       'constraint': lambda expr: numpy.all(expr.type.broadcastable)},
    (sparse._dot, 'x', 'y'))),
    (usmm, (theano.tensor.neg, 'alpha'), 'x', 'y', 'z'))
register_specialize(local_usmm, name="local_usmm")


# This is tested in tests/test_opt.py:test_local_csm_grad_c
@gof.local_optimizer([csm_grad(None)])
def local_csm_grad_c(node):
    """ csm_grad(None) -> csm_grad_c """
    if node.op == csm_grad(None):
        return [csm_grad_c(*node.inputs)]
    return False
register_specialize(local_csm_grad_c)


# This is tested in tests/test_opt.py:test_local_csm_properties_csm
@gof.local_optimizer([csm_properties])
def local_csm_properties_csm(node):
    """if we find csm_properties(CSM(*args)), then we can replace that with the
    *args directly"""
    if node.op == csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            # csm.owner.inputs could be broadcastable. In that case, we have
            # to adjust the broadcasting flag here.
            ret_var = [theano.tensor.patternbroadcast(i, o.broadcastable)
                       for i, o in izip(csm.owner.inputs, node.outputs)]
            return ret_var

    return False
sparse.register_specialize(local_csm_properties_csm)


# This is tested in tests/test_basic.py:test_remove0
@gof.local_optimizer([None])
def local_inplace_remove0(node):
    """
    Optimization to insert inplace versions of Remove0.
    """
    if isinstance(node.op, sparse.Remove0) and not node.op.inplace:
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
theano.compile.optdb.register('local_inplace_remove0',
                              gof.TopoOptimizer(local_inplace_remove0,
    failure_callback=gof.TopoOptimizer.warn_inplace),
                              60, 'fast_run', 'inplace')


# register a specialization to replace StructuredDot -> StructuredDotCSx
# This is tested in tests/test_basic.py:792
@gof.local_optimizer([sparse._structured_dot])
def local_structured_dot(node):
    if node.op == sparse._structured_dot:
        a, b = node.inputs
        if a.type.format == 'csc':
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            a_nsparse = a_shape[0]
            return [sd_csc(a_val, a_ind, a_ptr, a_nsparse, b)]
        if a.type.format == 'csr':
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            return [sd_csr(a_val, a_ind, a_ptr, b)]
    return False

# Commented out because
# a) it is only slightly faster than scipy these days, and sometimes a little
# slower, and
# b) the resulting graphs make it very difficult for an op to do size checking
# on the matrices involved.  dimension mismatches are hard to detect sensibly.
#register_specialize(local_structured_dot)


# This is tested in tests/test_basic.py:UsmmTests
@gof.local_optimizer([usmm_csc_dense])
def local_usmm_csc_dense_inplace(node):
    if node.op == usmm_csc_dense:
        return [sparse.usmm_csc_dense_inplace(*node.inputs)]
register_specialize(local_usmm_csc_dense_inplace, 'inplace')


@gof.local_optimizer([sparse.csm_properties])
def local_csm_properties_csm(node):
    """if we find csm_properties(CSM(*args)), then we can replace that with the
    *args directly"""
    if node.op == sparse.csm_properties:
        csm, = node.inputs
        if csm.owner and (csm.owner.op == sparse.CSC or
                          csm.owner.op == sparse.CSR):
            # csm.owner.inputs could be broadcastable. In that case, we have
            # to adjust the broadcasting flag here.
            ret_var = [theano.tensor.patternbroadcast(i, o.broadcastable)
                    for i, o in izip(csm.owner.inputs, node.outputs)]
            return ret_var


# This is tested in tests/test_basic.py:UsmmTests
@gof.local_optimizer([usmm])
def local_usmm_csx(node):
    """ usmm -> usmm_csc_dense """
    if node.op == usmm:
        alpha, x, y, z = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable and not y_is_sparse_variable:
            if x.type.format == 'csc':
                x_val, x_ind, x_ptr, x_shape = csm_properties(x)
                x_nsparse = x_shape[0]
                dtype_out = scalar.upcast(alpha.type.dtype, x.type.dtype,
                                          y.type.dtype, z.type.dtype)
                if dtype_out not in ('float32', 'float64'):
                    return False
                # Sparse cast is not implemented.
                if y.type.dtype != dtype_out:
                    return False

                return [usmm_csc_dense(alpha, x_val, x_ind, x_ptr,
                                       x_nsparse, y, z)]
    return False
sparse.register_specialize(local_usmm_csx)


# register a specialization to replace MulSD -> MulSDCSX
@gof.local_optimizer([sparse.mul_s_d])
def local_mul_s_d(node):
    if node.op == sparse.mul_s_d:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 2:
            return False
        if svar.type.format == 'csc':
            CSx = sparse.CSC
            mul_s_d_csx = sparse.mul_s_d_csc
        elif svar.type.format == 'csr':
            CSx = sparse.CSR
            mul_s_d_csx = sparse.mul_s_d_csr
        else:
            raise NotImplemented()

        c_data = mul_s_d_csx(sparse.csm_data(svar),
                             sparse.csm_indices(svar),
                             sparse.csm_indptr(svar), dvar)

        return [CSx(c_data,
                    sparse.csm_indices(svar),
                    sparse.csm_indptr(svar),
                    sparse.csm_shape(svar))]

    return False
sparse.register_specialize(local_mul_s_d)


@gof.local_optimizer([sparse.mul_s_v])
def local_mul_s_v(node):
    if node.op == sparse.mul_s_v:
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
            CSx = sparse.CSR
            mul_s_v_csx = sparse.mul_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = mul_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
sparse.register_specialize(local_mul_s_v)


@gof.local_optimizer([sparse.structured_add_s_v])
def local_structured_add_s_v(node):
    if node.op == sparse.structured_add_s_v:
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
            CSx = sparse.CSR
            structured_add_s_v_csx = sparse.structured_add_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = structured_add_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False
sparse.register_specialize(local_structured_add_s_v)


# register a specialization to replace SamplingDot -> SamplingDotCsr
@gof.local_optimizer([sparse.sampling_dot])
def local_sampling_dot_csr(node):
    if node.op == sparse.sampling_dot:
        x, y, p = node.inputs
        if p.type.format == 'csr':
            p_data, p_ind, p_ptr, p_shape = sparse.csm_properties(p)

            z_data, z_ind, z_ptr = sparse.sampling_dot_csr(x, y, p_data,
                p_ind, p_ptr, p_shape[1])

            return [sparse.CSR(z_data, z_ind, z_ptr, p_shape)]
    return False
sparse.register_specialize(local_sampling_dot_csr,
                           name='local_sampling_dot_csr')
