import time

import numpy
from scipy import ndimage
import theano
from theano.sandbox import cuda

from conv3d2d import *


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_get_diagonal_subtensor_view():

    x = numpy.arange(20).reshape(5, 4)
    xv01 = get_diagonal_subtensor_view(x, 0, 1)

    # test that it works in 2d
    assert numpy.all(xv01 == [[12, 9, 6, 3], [16, 13, 10, 7]])

    x = numpy.arange(24).reshape(4, 3, 2)
    xv01 = get_diagonal_subtensor_view(x, 0, 1)
    xv02 = get_diagonal_subtensor_view(x, 0, 2)
    xv12 = get_diagonal_subtensor_view(x, 1, 2)

    #print 'x', x
    #print 'xv01', xv01
    #print 'xv02', xv02
    assert numpy.all(xv01 == [
        [[12, 13], [8, 9], [4, 5]],
        [[18, 19], [14, 15], [10, 11]]])

    assert numpy.all(xv02 == [
        [[6, 1], [8, 3], [10, 5]],
        [[12, 7], [14, 9], [16, 11]],
        [[18, 13], [20, 15], [22, 17]],
        ])

    # diagonal views of each leading matrix is the same
    # as the slices out of the diagonal view of the entire 3d tensor
    for xi, xvi in zip(x, xv12):
        assert numpy.all(xvi == get_diagonal_subtensor_view(xi, 0, 1))


def test_get_diagonal_subtensor_view_gpu():
    x = numpy.arange(20, dtype='float32').reshape(5, 4)
    x = cuda.CudaNdarray(x)
    xv01 = get_diagonal_subtensor_view(x, 0, 1)

    # test that it works in 2d
    assert numpy.all(numpy.asarray(xv01) ==
                     [[12, 9, 6, 3], [16, 13, 10, 7]])

    x = numpy.arange(24).reshape(4, 3, 2)
    xv01 = get_diagonal_subtensor_view(x, 0, 1)
    xv02 = get_diagonal_subtensor_view(x, 0, 2)
    xv12 = get_diagonal_subtensor_view(x, 1, 2)

    #print 'x', x
    #print 'xv01', xv01
    #print 'xv02', xv02
    assert numpy.all(numpy.asarray(xv01) == [
        [[12, 13], [8, 9], [4, 5]],
        [[18, 19], [14, 15], [10, 11]]])

    assert numpy.all(numpy.asarray(xv02) == [
        [[6, 1], [8, 3], [10, 5]],
        [[12, 7], [14, 9], [16, 11]],
        [[18, 13], [20, 15], [22, 17]],
        ])

    # diagonal views of each leading matrix is the same
    # as the slices out of the diagonal view of the entire 3d tensor
    for xi, xvi in zip(x, numpy.asarray(xv12)):
        assert numpy.all(numpy.asarray(xvi) ==
                         numpy.asarray(get_diagonal_subtensor_view(xi, 0, 1)))


def pyconv3d(signals, filters):
    Ns, Ts, C, Hs, Ws = signals.shape
    Nf, Tf, C, Hf, Wf = filters.shape

    Tf2 = Tf//2
    Hf2 = Hf//2
    Wf2 = Wf//2

    rval = numpy.zeros((Ns, Ts-Tf+1, Nf, Hs-Hf+1, Ws-Wf+1))
    for ns in xrange(Ns):
        for nf in xrange(Nf):
            for c in xrange(C):
                s_i = signals[ns,:,c,:,:]
                f_i = filters[nf,:,c,:,:]
                r_i = rval[ns, :, nf, :, :]
                o_i = ndimage.convolve(s_i, f_i, mode='constant', cval=1)
                #print s_i.shape, f_i.shape, r_i.shape, o_i.shape
                r_i += o_i[Tf2:-Tf2, Hf2:-Hf2, Wf2:-Wf2]


def test_conv3d():

    Ns, Ts, C, Hs, Ws = 3, 10, 3, 32, 32
    Nf, Tf, C, Hf, Wf = 32, 5 , 3, 5 , 5

    signals = numpy.arange(Ns*Ts*C*Hs*Ws).reshape(Ns, Ts, C, Hs, Ws).astype('float32')
    filters = numpy.arange(Nf*Tf*C*Hf*Wf).reshape(Nf, Tf, C, Hf, Wf).astype('float32')

    t0 = time.time()
    pyconv3d(signals, filters)
    print time.time() - t0

    modes = [(mode_without_gpu, theano.tensor._shared)]
    if cuda.cuda_available:
        modes.append((mode_with_gpu, cuda.shared_constructor))

    for mode, shared in modes:
        s_signals = shared(signals)
        s_filters = shared(filters)
        s_output = shared(signals*0)

        out = conv3d(s_signals, s_filters,
                     signals_shape=signals.shape,
                     filters_shape=filters.shape)

        newconv3d = theano.function([], [],
                                    updates={s_output: out},
                                    mode=mode)

        t0 = time.time()
        newconv3d()
        print time.time() - t0
        gsignals, gfilters = theano.grad(out.sum(), [s_signals, s_filters])
        gnewconv3d = theano.function([], [],
                                     updates=[(s_filters, gfilters),
                                              (s_signals, gsignals)],
                                     mode=mode,
                                     name='grad')

        t0 = time.time()
        gnewconv3d()
        print 'grad', time.time() - t0
