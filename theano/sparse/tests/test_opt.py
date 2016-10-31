from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import numpy as np
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano
from theano import sparse, config, tensor
from theano.sparse import enable_sparse
from theano.tests import unittest_tools as utt
if not enable_sparse:
    raise SkipTest('Optional package sparse disabled')

from theano.sparse.tests.test_basic import random_lil


def test_local_csm_properties_csm():
    data = tensor.vector()
    indices, indptr, shape = (tensor.ivector(), tensor.ivector(),
                              tensor.ivector())
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_csm_properties_csm")
    for CS, cast in [(sparse.CSC, sp.csc_matrix),
                     (sparse.CSR, sp.csr_matrix)]:
        f = theano.function([data, indices, indptr, shape],
                            sparse.csm_properties(
                                CS(data, indices, indptr, shape)),
                            mode=mode)
        assert not any(
            isinstance(node.op, (sparse.CSM, sparse.CSMProperties))
            for node in f.maker.fgraph.toposort())
        v = cast(random_lil((10, 40),
                            config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


def test_local_csm_grad_c():
    raise SkipTest("Opt disabled as it don't support unsorted indices")
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    data = tensor.vector()
    indices, indptr, shape = (tensor.ivector(), tensor.ivector(),
                              tensor.ivector())
    mode = theano.compile.mode.get_default_mode()

    if theano.config.mode == 'FAST_COMPILE':
        mode = theano.compile.Mode(linker='c|py', optimizer='fast_compile')

    mode = mode.including("specialize", "local_csm_grad_c")
    for CS, cast in [(sparse.CSC, sp.csc_matrix), (sparse.CSR, sp.csr_matrix)]:
        cost = tensor.sum(sparse.DenseFromSparse()(CS(data, indices, indptr, shape)))
        f = theano.function(
            [data, indices, indptr, shape],
            tensor.grad(cost, data),
            mode=mode)
        assert not any(isinstance(node.op, sparse.CSMGrad) for node
                       in f.maker.fgraph.toposort())
        v = cast(random_lil((10, 40),
                            config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)


def test_local_mul_s_d():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_d")

    for sp_format in sparse.sparse_formats:
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.matrix()]

        f = theano.function(inputs,
                            sparse.mul_s_d(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, sparse.MulSD) for node
                       in f.maker.fgraph.toposort())


def test_local_mul_s_v():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_v")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.vector()]

        f = theano.function(inputs,
                            sparse.mul_s_v(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, sparse.MulSV) for node
                       in f.maker.fgraph.toposort())


def test_local_structured_add_s_v():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_structured_add_s_v")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.vector()]

        f = theano.function(inputs,
                            sparse.structured_add_s_v(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, sparse.StructuredAddSV) for node
                       in f.maker.fgraph.toposort())


def test_local_sampling_dot_csr():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_sampling_dot_csr")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [tensor.matrix(),
                  tensor.matrix(),
                  getattr(theano.sparse, sp_format + '_matrix')()]

        f = theano.function(inputs,
                            sparse.sampling_dot(*inputs),
                            mode=mode)

        if theano.config.blas.ldflags:
            assert not any(isinstance(node.op, sparse.SamplingDot) for node
                       in f.maker.fgraph.toposort())
        else:
            # SamplingDotCSR's C implementation needs blas, so it should not
            # be inserted
            assert not any(isinstance(node.op, sparse.opt.SamplingDotCSR) for node
                       in f.maker.fgraph.toposort())


def test_local_dense_from_sparse_sparse_from_dense():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("local_dense_from_sparse_sparse_from_dense")

    m = theano.tensor.matrix()
    for op in [theano.sparse.csr_from_dense, theano.sparse.csc_from_dense]:
        s = op(m)
        o = theano.sparse.dense_from_sparse(s)
        f = theano.function([m], o, mode=mode)
        # We should just have a deep copy.
        assert len(f.maker.fgraph.apply_nodes) == 1
        f([[1, 2], [3, 4]])

def test_sd_csc():

    A = sp.rand(4, 5, density=0.60, format='csc', dtype=np.float32)
    b = np.random.rand(5,2).astype(np.float32)
    target = A*b
    
    a_val = theano.tensor.as_tensor_variable(A.data)
    a_ind = theano.tensor.as_tensor_variable(A.indices)
    a_ptr = theano.tensor.as_tensor_variable(A.indptr)
    nrows = theano.tensor.as_tensor_variable(np.int32(A.shape[0]))
    b = theano.tensor.as_tensor_variable(b)
    
    res = theano.sparse.opt.sd_csc(a_val, a_ind, a_ptr, nrows, b).eval()
    
    utt.assert_allclose(res, target)

