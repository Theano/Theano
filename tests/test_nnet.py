import sys, time
import theano
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
        w = shared(0.01*(numpy.random.rand(n_in,n_hid)-0.5), 'w')
        b = shared(numpy.zeros(n_hid), 'b')
        v = shared(numpy.zeros((n_hid, n_out)), 'c')
        c = shared(numpy.zeros(n_out), 'c')

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
        train(xval, yval, lr)
    mode.print_summary()
    
def test_nnet_cpu():
    run_nnet(False)
def test_nnet_gpu():
    run_nnet(True)
