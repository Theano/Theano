from nose.plugins.skip import SkipTest
import numpy
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano
from theano import sparse, config, tensor
from theano.sparse import enable_sparse
from theano.gof.python25 import any
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
