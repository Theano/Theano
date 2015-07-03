import theano
from theano import tensor
from theano.sandbox.blocksparse import CpuSparseBlockGemv, CpuSparseBlockOuter, sparse_block_dot


def test_blocksparse_cpu_gemv_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    f = theano.function([W, h, iIdx, b, oIdx], o)

    assert isinstance(f.maker.fgraph.toposort()[-1].op, CpuSparseBlockGemv)


def test_blocksparse_cpu_outer_opt():
    b = tensor.fmatrix()
    W = tensor.ftensor4()
    h = tensor.ftensor3()
    iIdx = tensor.lmatrix()
    oIdx = tensor.lmatrix()

    o = sparse_block_dot(W, h, iIdx, b, oIdx)

    theano.printing.debugprint(tensor.grad(o.sum(),wrt=W))

    f = theano.function([W, h, iIdx, b, oIdx], [o, tensor.grad(o.sum(),wrt=W)])
    
    assert isinstance(f.maker.fgraph.toposort()[-1].op, CpuSparseBlockOuter)
