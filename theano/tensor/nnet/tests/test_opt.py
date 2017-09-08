from __future__ import absolute_import, print_function, division
import theano
from theano import tensor
from theano.tests.unittest_tools import assertFailure_fast
from theano.gof.opt import check_stack_trace
from theano.tensor.nnet.blocksparse import (
    sparse_block_dot, sparse_block_gemv_inplace, sparse_block_outer_inplace,
    sparse_block_gemv, sparse_block_outer)


def test_blocksparse_inplace_gemv_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)

    if theano.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=[sparse_block_gemv])
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=[sparse_block_gemv_inplace])

if theano.config.mode != 'FAST_COMPILE':
    test_blocksparse_inplace_gemv_opt = assertFailure_fast(test_blocksparse_inplace_gemv_opt)


def test_blocksparse_inplace_outer_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx],
                        [o, tensor.grad(o.sum(), wrt=W)])

    if theano.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=sparse_block_outer)
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace
        assert check_stack_trace(f, ops_to_check=sparse_block_outer_inplace)
