import sys, time
import theano, theano.sandbox.conv
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import theano_cuda_ndarray as tcn

import logging
logging.getLogger('theano.gradient').setLevel(logging.INFO)


def run_nnet(use_gpu):
    n_batch = 16
    n_in = 1024
    n_hid = 2048
    n_out = 10

    if use_gpu:
        w = tcn.shared_constructor(0.01*(numpy.random.rand(n_in,n_hid)-0.5), 'w')
        b = tcn.shared_constructor(numpy.zeros(n_hid), 'b')
        v = tcn.shared_constructor(numpy.zeros((n_hid, n_out)), 'c')
        c = tcn.shared_constructor(numpy.zeros(n_out), 'c')
    else:
        w = shared(numpy.asarray(0.01*(numpy.random.rand(n_in,n_hid)-0.5), dtype='float32'), 'w')
        b = shared(numpy.asarray(numpy.zeros(n_hid), dtype='float32'), 'b')
        v = shared(numpy.asarray(numpy.zeros((n_hid, n_out)), dtype='float32'), 'c')
        c = shared(numpy.asarray(numpy.zeros(n_out), dtype='float32'), 'c')

    x = tensor.fmatrix('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    hid = tensor.tanh(tensor.dot(x, w)+b)
    out = tensor.tanh(tensor.dot(hid, v)+c)
    loss = tensor.sum(0.5 * (out-y)**2 * lr)
    print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = theano.compile.ProfileMode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

    for i, n in enumerate(train.maker.env.toposort()):
        print i, n

    xval = numpy.asarray(numpy.random.rand(n_batch, n_in), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch, n_out), dtype='float32')
    lr = numpy.asarray(0.01, dtype='float32')

    for i in xrange(100):
        rval = train(xval, yval, lr)
    mode.print_summary()
    return rval
    
def test_run_nnet():
    numpy.random.seed(23456)
    rval_cpu = run_nnet(False)
    numpy.random.seed(23456)
    rval_gpu = run_nnet(True)
    assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-6)


def run_conv_nnet1(shared_fn):
    n_batch = 16
    n_kern = 20
    shape_img = (n_batch, 1, 32, 32)
    shape_kern = (n_kern, 1, 5, 5)

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d((32,32),(5,5), 'valid')
    n_hid = n_kern * logical_hid_shape[0] * logical_hid_shape[1]
    n_out = 10

    w = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern)-0.5), dtype='float32'), 'w')
    b = shared_fn(numpy.asarray(numpy.zeros((n_kern,1,1)), dtype='float32'), 'b')
    v = shared_fn(numpy.asarray(numpy.zeros((n_hid, n_out)), dtype='float32'), 'c')
    c = shared_fn(numpy.asarray(numpy.zeros(n_out), dtype='float32'), 'c')

    x = tensor.Tensor(dtype='float32', broadcastable=(0,0,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = theano.sandbox.conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)

    hid = tensor.tanh(conv_op(x, w)+b)
    hid_flat = hid.reshape((n_batch, n_hid))
    out = tensor.tanh(tensor.dot(hid_flat, v)+c)
    loss = tensor.sum(0.5 * (out-y)**2 * lr)
    print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = theano.compile.ProfileMode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

    for i, n in enumerate(train.maker.env.toposort()):
        print i, n

    xval = numpy.asarray(numpy.random.rand(*shape_img), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch, n_out), dtype='float32')
    lr = numpy.asarray(0.01, dtype='float32')

    for i in xrange(10):
        rval = train(xval, yval, lr)
    mode.print_summary()
    return rval

def test_conv_nnet1():
    numpy.random.seed(23456)
    rval_cpu = run_conv_nnet1(shared)
    numpy.random.seed(23456)
    rval_gpu = run_conv_nnet1(tcn.shared_constructor)
    assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-6)

