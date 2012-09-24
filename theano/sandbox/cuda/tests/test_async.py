# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import numpy as np
from theano.sandbox.cuda.async import (local_async_gpu, async_optimizer,
        GpuFromHostWait, HostFromGpuWait,GpuFromHostSend, HostFromGpuSend,
        gpu_cmps, send_wait, gpu_ops_first, send_in_order, tiebreaker)
from theano.sandbox.cuda.basic_ops import (gpu_from_host, host_from_gpu,
        GpuFromHost)
import theano
from theano.gof.sched import sort_schedule_fn, key_to_cmp
import theano.sandbox.linalg as linalg

gpu_scheduler = sort_schedule_fn(*gpu_cmps)
gpu_linker = theano.OpWiseCLinker(schedule=gpu_scheduler)
gpu_mode = theano.Mode(linker = gpu_linker, optimizer = async_optimizer)
full_opt = theano.compile.mode.get_default_mode().optimizer

def test_async_to_gpu():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)
    assert len(gx2) == 1
    gx2 = gx2[0]
    assert (gx.dtype, type(gx)) == (gx2.dtype, type(gx2))
    assert isinstance(gx2.owner.op, GpuFromHostWait)


def test_async_to_host():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    y = theano.sandbox.cuda.host_from_gpu(gx)
    y2 = local_async_gpu.transform(y.owner)
    assert len(y2) == 1
    y2 = y2[0]

    assert (y.dtype, type(y)) == (y2.dtype, type(y2))
    assert isinstance(y2.owner.op, HostFromGpuWait)


def test_compile():
    """ Can we compile such a function without failing?"""
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    theano.function([x], gx2, mode=theano.Mode(optimizer=None, linker='c|py'))


def test_execute():
    """Can we run such a function without failing?"""
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)

    f = theano.function([x], gx2, mode=theano.Mode(optimizer=None,
                                                   linker='c|py'))
    xx = np.ones((5, 5), dtype=x.dtype)
    f(xx)


def test_optimizer():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    fgraph = theano.FunctionGraph([x], [gx])
    async_optimizer.optimize(fgraph)
    ops = set([node.op for node in fgraph.apply_nodes])
    assert len(filter(lambda op: isinstance(op, GpuFromHostWait), ops)) == 1
    assert len(filter(lambda op: isinstance(op, GpuFromHostSend), ops)) == 1
    assert len(filter(lambda op: isinstance(op, GpuFromHost), ops))     == 0


def test_optimizer2():
    """ Test that the optimization is correctly registered"""
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    mode = theano.Mode(linker='c|py').including("local_async_gpu", 'gpu')
    f = theano.function([x], gx, mode=mode)
    xx = np.ones((5, 5), dtype=x.dtype)
    f(xx)

def test_send_wait():
    send_wait_cmp = key_to_cmp(send_wait)
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    gx2 = local_async_gpu.transform(gx.owner)[0]
    y = x + 1

    waitnode = x.owner
    sendnode = gx2.owner.inputs[0].owner
    waitnode = gx2.owner
    addnode = y.owner
    assert send_wait_cmp(sendnode, waitnode) < 0 # send happens first
    assert send_wait_cmp(sendnode, addnode) < 0 # send happens first
    assert send_wait_cmp(waitnode, addnode) > 0 # wait happens last

# This shouldn't necessarily happen.
def _test_gpu_ops_first():
    a = theano.tensor.fmatrix('a')
    b = theano.tensor.fmatrix('b')
    c = theano.tensor.dot(a, a)
    d = linalg.solve(a, b)

    mode = theano.Mode(linker=gpu_linker, optimizer=full_opt)
    f = theano.function((a,b), (c, d), mode=mode)
    nodes = f.maker.linker.make_all()[-1]

    index_of_solve = nodes.index(
           filter(lambda n: isinstance(n.op, linalg.ops.Solve),
                   nodes)[0])

    index_of_gemm  = nodes.index(
           filter(lambda n: isinstance(n.op, theano.sandbox.cuda.blas.GpuDot22),
                   nodes)[0])

    assert index_of_solve > index_of_gemm

def test_send_order():
    a = theano.tensor.fmatrix('a')
    b = theano.tensor.fmatrix('b')
    c = theano.tensor.fmatrix('c')
    d = theano.tensor.dot(a, b)
    e = theano.tensor.dot(d, c)

    mode = theano.Mode(linker=gpu_linker, optimizer=full_opt)
    f = theano.function((a,b,c), e, mode=mode)
    nodes = f.maker.linker.make_all()[-1]

    assert set(node.inputs[0].name for node in nodes[:2]) == {a.name, b.name}
    # C is the last needed. Ensure that it happens last
    assert nodes[2].inputs[0].name == c.name

def test_tiebreaker():
    a = theano.tensor.fmatrix('a')
    b = theano.tensor.fmatrix('b')
    c = theano.tensor.fmatrix('c')
    d = a+b
    e = b+c
    assert tiebreaker(d.owner) < tiebreaker(e.owner)

def test_gpu_schedule():
    x = theano.tensor.fmatrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(x)
    y = x + 1
    f = theano.function((x,), (gx, y), mode = gpu_mode)

    nodes = f.maker.linker.make_all()[-1]
    assert isinstance(nodes[0].op, GpuFromHostSend)
    assert isinstance(nodes[1].op, theano.tensor.DimShuffle)
    assert isinstance(nodes[2].op, theano.tensor.Elemwise)
    assert isinstance(nodes[3].op, GpuFromHostWait)
