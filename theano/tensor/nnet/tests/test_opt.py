from __future__ import absolute_import, print_function, division
import theano
from theano import tensor
from theano.tensor.nnet.blocksparse import sparse_block_dot


def test_blocksparse_inplace_gemv_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)
    assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

    if theano.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace


def test_blocksparse_inplace_outer_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    theano.printing.debugprint(tensor.grad(o.sum(), wrt=W))

    f = theano.function([W, h, iIdx, b, oIdx],
                        [o, tensor.grad(o.sum(), wrt=W)])
    assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

    if theano.config.mode == "FAST_COMPILE":
        assert not f.maker.fgraph.toposort()[-1].op.inplace
    else:
        assert f.maker.fgraph.toposort()[-1].op.inplace
