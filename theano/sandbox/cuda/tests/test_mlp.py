import copy
import logging
import time
from itertools import izip

from nose.plugins.skip import SkipTest
import numpy

import theano
from theano.compile.sharedvalue import shared
from theano.compile.pfunc import pfunc
from theano import tensor
from theano import config
import theano.tensor.nnet.conv as conv
import theano.tensor.signal.downsample as downsample
import theano.sandbox.cuda as tcn
import theano.tests.unittest_tools as utt


if theano.config.mode not in ['FAST_RUN', 'Mode', 'ProfileMode']:
    raise SkipTest('Skip test_mlp when not in normal optimization mode as '
                   'otherwise it is too slow!')

# Skip test if cuda_ndarray is not available.
if tcn.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


logging.getLogger('theano.sandbox.cuda.tests.test_nnet').setLevel(logging.INFO)


def my_rand(*shape):
    return theano._asarray(numpy.random.rand(*shape), dtype='float32')


def my_randn(*shape):
    return theano._asarray(numpy.random.randn(*shape), dtype='float32')


def my_zeros(*shape):
    return theano._asarray(numpy.zeros(*shape), dtype='float32')


def get_mode(use_gpu, check_isfinite=True):
    if theano.config.mode != 'FAST_COMPILE':
        ret = theano.compile.get_default_mode()
    else:
        ret = theano.compile.mode.get_mode('FAST_RUN')
    if isinstance(ret, theano.compile.ProfileMode):
        ret = copy.copy(ret)
    if isinstance(ret, theano.compile.DebugMode):
        ret = copy.copy(ret)
        ret.check_isfinite = check_isfinite
    if use_gpu:
        ret = ret.including('gpu')
    else:
        ret = ret.excluding('gpu')
    return ret


def print_mode(mode):
    if mode != None and isinstance(mode, (theano.compile.ProfileMode,)):
        mode.print_summary()


def print_diff_mode(a, b):
    if (a is not None and
        isinstance(a, (theano.compile.ProfileMode,)) and
        isinstance(b, (theano.compile.ProfileMode,))):

        a.print_diff_summary(b)


def run_nnet(use_gpu, n_batch=60, n_in=1024, n_hid=2048, n_out=10,
             n_train=100):

    if config.mode == 'DEBUG_MODE':
        n_train = 1

    if use_gpu:
        w = tcn.shared_constructor(0.01 * (my_rand(n_in, n_hid) - 0.5), 'w')
        b = tcn.shared_constructor(my_zeros(n_hid), 'b')
        v = tcn.shared_constructor(my_zeros((n_hid, n_out)), 'c')
        c = tcn.shared_constructor(my_zeros(n_out), 'c')
    else:
        w = shared(0.01 * (my_rand(n_in, n_hid) - 0.5), 'w')
        b = shared(my_zeros(n_hid), 'b')
        v = shared(my_zeros((n_hid, n_out)), 'c')
        c = shared(my_zeros(n_out), 'c')

    x = tensor.fmatrix('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    hid = tensor.tanh(tensor.dot(x, w) + b)
    out = tensor.tanh(tensor.dot(hid, v) + c)
    loss = tensor.sum(0.5 * (out - y) ** 2 * lr)
    if 0:
        print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = get_mode(use_gpu)

    print 'building pfunc ...'
    train = pfunc([x, y, lr], [loss], mode=mode,
                  updates=[(p, p - g) for p, g in izip(params, gparams)])

    if 0:
        for i, n in enumerate(train.maker.env.toposort()):
            print i, n

    xval = my_rand(n_batch, n_in)
    yval = my_rand(n_batch, n_out)
    lr = theano._asarray(0.01, dtype='float32')

    t0 = time.time()
    rval = []
    for i in xrange(n_train):
        rval.append(train(xval, yval, lr))
    dt = time.time() - t0

    print_mode(mode)
    return numpy.asarray(rval), dt


def test_run_nnet():
    for n_in in 1024, 2048, 4096:
        for n_hid in 1024, 2048, 4096:
            utt.seed_rng()  # Seeds numpy rng with utt.fetch_seed()
            rval_cpu, tc = run_nnet(False, n_in=n_in, n_hid=n_hid)
            utt.seed_rng()
            rval_gpu, tg = run_nnet(True, n_in=n_in, n_hid=n_hid)
            #print "cpu:", rval_cpu
            #print "gpu:", rval_gpu
            abs_diff, rel_diff = \
                    theano.gradient.numeric_grad.abs_rel_err(rval_gpu,
                                                             rval_cpu)
            max_abs_diff = abs_diff.max()
            print "max abs diff=%e max rel diff=%e n_in=%d n_hid=%d" % (
                max_abs_diff, rel_diff.max(), n_in, n_hid)
            print "time cpu: %f, time gpu: %f, speed up %f" % (tc, tg, tc / tg)
            rtol = 1e-4
            if n_in * n_hid >= 2048 * 4096:
                rtol = 7e-4
            assert numpy.allclose(rval_cpu, rval_gpu, rtol=rtol, atol=1e-6), \
                   ("max_abs_diff, max_rel_diff, n_in, n_hid", max_abs_diff,
                    rel_diff.max(), n_in, n_hid)


def test_run_nnet_med():
    utt.seed_rng()
    rval_cpu = run_nnet(False, 10, 128, 50, 4, n_train=10000)


def test_run_nnet_small():
    utt.seed_rng()
    rval_cpu = run_nnet(False, 10, 10, 4, 4, n_train=100000)


def run_conv_nnet1(use_gpu):
    if use_gpu:
        shared_fn = tcn.shared_constructor
    else:
        shared_fn = shared
    n_batch = 16
    n_kern = 20
    shape_img = (n_batch, 1, 32, 32)
    shape_kern = (n_kern, 1, 5, 5)
    n_train = 10
    if config.mode == 'DEBUG_MODE':
        n_train = 1

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d(shape_img[2:],shape_kern[2:], 'valid')
    n_hid = n_kern * logical_hid_shape[0] * logical_hid_shape[1]
    n_out = 10

    w = shared_fn(0.01*(my_rand(*shape_kern)-0.5), 'w')
    b = shared_fn(my_zeros((n_kern,)), 'b')
    v = shared_fn(my_zeros((n_hid, n_out)), 'c')
    c = shared_fn(my_zeros(n_out), 'c')

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)
    conv_op.set_flops()

    hid = tensor.tanh(conv_op(x, w)+b.dimshuffle((0,'x','x')))
    hid_flat = hid.reshape((n_batch, n_hid))
    out = tensor.tanh(tensor.dot(hid_flat, v)+c)
    loss = tensor.sum(0.5 * (out-y)**2 * lr)
    print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = get_mode(use_gpu)

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

#    for i, n in enumerate(train.maker.env.toposort()):
#        print i, n

    xval = my_rand(*shape_img)
    yval = my_rand(n_batch, n_out)
    lr = theano._asarray(0.01, dtype='float32')

    for i in xrange(n_train):
        rval = train(xval, yval, lr)
    print 'training done'
    print_mode(mode)
    return rval


def test_conv_nnet1():
    utt.seed_rng()
    rval_cpu = run_conv_nnet1(False)
    utt.seed_rng()
    rval_gpu = run_conv_nnet1(True)
    assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-6)


def run_conv_nnet2(use_gpu): # pretend we are training LeNet for MNIST
    if use_gpu:
        shared_fn = tcn.shared_constructor
    else:
        shared_fn = shared

    #cumulativ rounding error affect this comparaison of result. So we lower the tolerance.
    #TODO: why the last two example see the error lower? We are converging?
    #n_train=10, n_batch=3, n_kern=1, n_kern1=1, error see of 1e-9
    #n_train=10, n_batch=3, n_kern=10, n_kern1=1, error see of -1.27777e-06
    #n_train=10, n_batch=3, n_kern=10, n_kern1=10, error see of -6.91377e-05
    #n_train=10, n_batch=30, n_kern=10, n_kern1=10, error see of -0.00185963
    #n_train=10, n_batch=60, n_kern=10, n_kern1=10, error see of -5.26905e-05
    #n_train=30, n_batch=60, n_kern=10, n_kern1=10, error see of -3.8147e-06


    #n_train=30, n_batch=60, n_kern=20, n_kern1=10, error see of 6.82771e-05
    #n_train=30, n_batch=60, n_kern=20, n_kern1=30, error see of 0.000231534

    n_batch = 60
    shape_img = (n_batch, 1, 32, 32)

    n_kern = 20
    shape_kern = (n_kern, 1, 5, 5)

    n_kern1 = 10
    shape_kern1 = (n_kern1, n_kern, 5, 5)

    n_train=30
    if config.mode=='DEBUG_MODE': n_train=1

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d(tuple(shape_img[2:]),tuple(shape_kern[2:]), 'valid')
    logical_hid_shape1 = tcn.blas.GpuConv.logical_output_shape_2d((logical_hid_shape[0]/2, logical_hid_shape[1]/2), tuple(shape_kern1[2:]), 'valid')
    n_hid = n_kern1 * logical_hid_shape1[0] * logical_hid_shape1[1]
    n_out = 10

    w0 = shared_fn(0.01*(my_rand(*shape_kern)-0.5), 'w0')
    b0 = shared_fn(my_zeros((n_kern,)), 'b0')
    w1 = shared_fn(0.01*(my_rand(*shape_kern1)-0.5), 'w1')
    b1 = shared_fn(my_zeros((n_kern1,)), 'b1')
    v = shared_fn(my_zeros((n_hid, n_out)), 'c')
    c = shared_fn(my_zeros(n_out), 'c')

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)
    conv_op1 = conv.ConvOp((n_kern,logical_hid_shape[0]/2, logical_hid_shape[1]/2), shape_kern1[2:], n_kern1, n_batch, 1, 1)
    conv_op.set_flops()
    conv_op1.set_flops()

    hid = tensor.tanh(conv_op(x, w0)+b0.dimshuffle((0,'x','x')))
    hid1 = tensor.tanh(conv_op1(hid[:,:,::2,::2], w1) + b1.dimshuffle((0,'x','x')))
    hid_flat = hid1.reshape((n_batch, n_hid))
    out = tensor.tanh(tensor.dot(hid_flat, v)+c)
    loss = tensor.sum(0.5 * (out-y)**2 * lr)
    print 'loss type', loss.type

    params = [w0, b0, w1, b1, v, c]
    gparams = tensor.grad(loss, params)

    mode = get_mode(use_gpu)

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

#    for i, n in enumerate(train.maker.env.toposort()):
#        print i, n

    xval = my_rand(*shape_img)
    yval = my_rand(n_batch,n_out)#int32 make all 0...
    lr = theano._asarray(0.01, dtype='float32')
    for i in xrange(n_train):
        rval = train(xval, yval, lr)

    print_mode(mode)
    return rval


def test_conv_nnet2():
    utt.seed_rng()
    rval_gpu = run_conv_nnet2(True)
    if True:
        utt.seed_rng()
        rval_cpu = run_conv_nnet2(False)
        print rval_cpu[0], rval_gpu[0],rval_cpu[0]-rval_gpu[0]
        assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-4)


def build_conv_nnet2_classif(use_gpu, isize, ksize, n_batch,
                             downsample_ops=True, verbose=0, version=-1,
                             check_isfinite=True):
    if use_gpu:
        shared_fn = tcn.shared_constructor
    else:
        shared_fn = shared

    isize1=isize
    isize2=isize
    if isinstance(isize,(tuple,)):
        isize1=isize[0]
        isize2=isize[1]

    shape_img = (n_batch, 1, isize1, isize2)

    n_kern = 20  # 6 were used in LeNet5
    shape_kern = (n_kern, 1, ksize, ksize)

    n_kern1 = 30 # 16 were used in LeNet5
    shape_kern1 = (n_kern1, n_kern, ksize, ksize)

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d((isize1, isize2), (ksize, ksize), 'valid')
    logical_hid_shape1 = tcn.blas.GpuConv.logical_output_shape_2d((logical_hid_shape[0]/2,
        logical_hid_shape[1]/2), (ksize, ksize), 'valid')
    n_hid = n_kern1 * logical_hid_shape1[0] * logical_hid_shape1[1]
    n_out = 10


    w0 = shared_fn(0.01*(my_rand(*shape_kern)-0.5), 'w0')
    b0 = shared_fn(my_zeros((n_kern,)), 'b0')
    w1 = shared_fn(0.01*(my_rand(*shape_kern1)-0.5), 'w1')
    b1 = shared_fn(my_zeros((n_kern1,)), 'b1')
    v = shared_fn(0.01*my_randn(n_hid, n_out), 'v')
    c = shared_fn(my_zeros(n_out), 'c')

    print 'ALLOCATING ARCH: w0 shape', w0.get_value(borrow=True).shape
    print 'ALLOCATING ARCH: w1 shape', w1.get_value(borrow=True).shape
    print 'ALLOCATING ARCH: v shape', v.get_value(borrow=True).shape

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern,
                          n_batch, 1, 1, verbose=verbose, version=version)
    conv_op1 = conv.ConvOp(
        (n_kern,logical_hid_shape[0]/2, logical_hid_shape[1]/2),
        shape_kern1[2:], n_kern1, n_batch, 1, 1,verbose=verbose, version=version)
    conv_op.set_flops()
    conv_op1.set_flops()

    ds_op = downsample.DownsampleFactorMax((2,2), ignore_border=False)
    if downsample_ops:
        hid = tensor.tanh(ds_op(conv_op(x, w0)+b0.dimshuffle((0,'x','x'))))
    else:
        hid = tensor.tanh((conv_op(x, w0)+b0.dimshuffle((0,'x','x')))[:,:,::2,::2])
    hid1 = tensor.tanh(conv_op1(hid, w1) + b1.dimshuffle((0,'x','x')))
    hid_flat = hid1.reshape((n_batch, n_hid))
    out = tensor.nnet.softmax(tensor.dot(hid_flat, v)+c)
    loss = tensor.sum(tensor.nnet.crossentropy_categorical_1hot(out, tensor.argmax(y, axis=1)) * lr)
    print 'loss type', loss.type

    params = [w0, b0, w1, b1, v, c]
    gparams = tensor.grad(loss, params, warn_type=True)

    mode = get_mode(use_gpu, check_isfinite)

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

    if verbose:
        theano.printing.debugprint(train)
    if use_gpu:
        # Check that GpuConv is used
        topo = train.maker.env.toposort()
        assert len([n for n in topo if isinstance(n.op, tcn.blas.GpuConv)]) > 0

    shape_target = (n_batch,n_out)
    return train, params, shape_img, shape_target, mode


def run_conv_nnet2_classif(use_gpu, seed, isize, ksize, bsize,
                           n_train=10,
                           check_isfinite=True,
                           pickle=False,
                           verbose=0,
                           version=-1):
    """Run the train function returned by build_conv_nnet2_classif on one device.
    """

    utt.seed_rng(seed) # Seeds numpy.random with seed
    train, params, x_shape, y_shape, mode = build_conv_nnet2_classif(
            use_gpu=use_gpu,
            isize=isize,
            ksize=ksize,
            n_batch=bsize,
            verbose=verbose,
            version=version,
            check_isfinite=check_isfinite)

    if use_gpu:
        device = 'GPU'
    else:
        device = 'CPU'

    xval = my_rand(*x_shape)
    yval = my_rand(*y_shape)
    lr = theano._asarray(0.01, dtype='float32')

    rvals = my_zeros(n_train)
    t0 = time.time()
    for i in xrange(n_train):
        rvals[i] = train(xval, yval, lr)[0]
    t1 = time.time()
    print_mode(mode)

    if pickle and isinstance(mode, theano.compile.ProfileMode):
        import pickle
        print "BEGIN %s profile mode dump" % device
        print pickle.dumps(mode)
        print "END %s profile mode dump" % device

    print "%s time: %.3f" % (device, t1-t0)
    print "estimated time for one pass through MNIST with %s: %f" % (
            device, (t1-t0) * (60000.0 / (n_train*bsize)))


def cmp_run_conv_nnet2_classif(seed, isize, ksize, bsize,
                               ignore_error=False,
                               n_train=10,
                               gpu_only=False,
                               cpu_only=False,
                               float_atol=1e-06,
                               check_isfinite=True,
                               pickle=False,
                               verbose=0,
                               version=-1):
    """Run the nnet2 function on 1 or 2 devices, and compares the results.

       float_atol: None mean use the default value.
       check_isfinite: the debug mode option. We forward this value to debug mode.
                       For some parameter CrossentropyCategorical1Hot op generate inf when not optimized.
    """
    if config.mode == 'DEBUG_MODE':
        n_train = 1

    # Change global tolerance, used in DebugMode for instance
    orig_float32_atol = theano.tensor.basic.float32_atol
    try:
        if float_atol:
            print "float_atol", float_atol
            theano.tensor.basic.float32_atol = float_atol

        if gpu_only and cpu_only:
            raise ValueError("Please use only one of cpu_only and gpu_only")
        elif cpu_only:
            use_gpu = False
            compare = False
        elif gpu_only:
            use_gpu = True
            compare = False
        else:
            compare = True

        if not compare:
            return run_conv_nnet2_classif(use_gpu=use_gpu,
                    seed=seed, isize=isize, ksize=ksize, bsize=bsize,
                    n_train=n_train,
                    check_isfinite=check_isfinite,
                    pickle=pickle,
                    verbose=verbose,
                    version=version)

        utt.seed_rng(seed) # Seeds numpy.random with seed
        train_cpu, params_cpu, x_shape, y_shape, mode_cpu = \
                build_conv_nnet2_classif(
                        use_gpu=False,
                        isize=isize,
                        ksize=ksize,
                        n_batch=bsize,
                        verbose=verbose,
                        version=version,
                        check_isfinite=check_isfinite)

        utt.seed_rng(seed) # Seeds numpy.random with seed
        train_gpu, params_gpu, x_shape_gpu, y_shape_gpu, mode_gpu = \
                build_conv_nnet2_classif(
                        use_gpu=True,
                        isize=isize,
                        ksize=ksize,
                        n_batch=bsize,
                        verbose=verbose,
                        version=version,
                        check_isfinite=check_isfinite)

        assert x_shape == x_shape_gpu
        assert y_shape == y_shape_gpu

        xval = my_rand(*x_shape)
        yval = my_rand(*y_shape)
        lr = theano._asarray(0.01, dtype='float32')

        time_cpu = 0
        time_gpu = 0

        for i in range(n_train):
            # Train one batch on CPU
            t0 = time.time()
            rval_cpu = train_cpu(xval, yval, lr)[0]
            t1 = time.time()
            time_cpu += (t1-t0)

            # Train one batch on GPU
            t0 = time.time()
            rval_gpu = train_gpu(xval, yval, lr)[0]
            t1 = time.time()
            time_gpu += (t1-t0)

            # Compare results
            if (verbose or not
                    numpy.allclose(rval_cpu, rval_gpu, rtol=1e-5, atol=float_atol)):
                print "At batch:", i+1
                print "CPU:", rval_cpu
                print "GPU:", rval_gpu
                print "abs diff:", numpy.absolute(rval_gpu-rval_cpu)
                print "rel diff:", numpy.absolute((rval_gpu-rval_cpu)/rval_gpu)

            if not ignore_error:
                assert numpy.allclose(rval_cpu, rval_gpu, rtol=1e-5, atol=float_atol)

            # Synchronize parameters to start from the same point next time
            if i < n_train-1:
                for cpu_p, gpu_p in zip(params_cpu, params_gpu):
                    cpu_p.set_value(gpu_p.get_value(borrow=False), borrow=True)

    finally:
        theano.tensor.basic.float32_atol = orig_float32_atol

    if pickle:
        if isinstance(cpu_mode, theano.compile.ProfileMode):
            import pickle
            print "BEGIN CPU profile mode dump"
            print pickle.dumps(cpu_mode)
            print "END CPU profile mode dump"
        if isinstance(gpu_mode, theano.compile.ProfileMode):
            import pickle
            print "BEGIN GPU profile mode dump"
            print pickle.dumps(gpu_mode)
            print "END GPU profile mode dump"

    print "CPU time: %.3f, GPU time: %.3f, speed up %f" % (
            (time_cpu, time_gpu, time_cpu/time_gpu))
    print "Estimated time for one pass through MNIST with CPU: %f" % (
            (time_cpu * (60000.0 / (n_train*bsize))))
    print "Estimated time for one pass through MNIST with GPU: %f" % (
            (time_gpu * (60000.0 / (n_train*bsize))))


# Default parameters for all subsequent tests
gpu_only=False
cpu_only=False
ignore_error=False
verbose=0
version=-1
seed = utt.fetch_seed()


def test_lenet_28(): #MNIST
    cmp_run_conv_nnet2_classif(seed, 28, 5, 60, n_train=10,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose, version=version)


def test_lenet_32(): #CIFAR10 / Shapeset
    cmp_run_conv_nnet2_classif(seed, 32, 5, 60, n_train=8,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               verbose=verbose, version=version)


def test_lenet_32_long(): #CIFAR10 / Shapeset
    # this tests the gradient of downsample on the GPU,
    # which does not recieve specific testing
    cmp_run_conv_nnet2_classif(seed, 32, 5, 30, n_train=50,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose, version=version)


def test_lenet_64(): # ???
    #float_atol need to pass in debug mode
    #needed as cpu use extended precision and gpu don't
    cmp_run_conv_nnet2_classif(seed, 64, 7, 10, n_train=10,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose,
                               check_isfinite=True, version=version)


def test_lenet_108(): # NORB
    cmp_run_conv_nnet2_classif(seed, 108, 7, 5, n_train=4,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose,
                               check_isfinite=True, version=version)


def test_lenet_256(): # ImageNet
    cmp_run_conv_nnet2_classif(seed, 256, 9, 2, n_train=5,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose,
                               check_isfinite=True, version=version, float_atol=5e-5)


#I did a wanted error in the name as we don't want it to execute automatically for now as it don't work
def tes_lenet_hd(): #HD 720p: 1280(wid)x720(len)
    cmp_run_conv_nnet2_classif(seed, (720,1280), 9, 2, n_train=3,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose,
                               check_isfinite=True, version=version)


#I did a wanted error in the name as we don't want it to execute automatically for now as it don't work
def tes_lenet_full_hd(): #HD 1080p: 1920(wid)x1080(len)
    cmp_run_conv_nnet2_classif(seed, (1080,1920), 9, 2, n_train=3,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               cpu_only=cpu_only, verbose=verbose,
                               check_isfinite=True, version=version)
