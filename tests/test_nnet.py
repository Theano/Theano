import sys, time
import theano

from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor
import theano.tensor.nnet

import theano.sandbox.conv
import theano.sandbox.downsample

import numpy

import theano_cuda_ndarray as tcn

import logging
logging.getLogger('test_cuda_ndarray.tests.test_nnet').setLevel(logging.INFO)


def get_mode():
    if theano.compile.default_mode == 'CLINKER_MODE':
        return theano.compile.mode.Mode(optimizer='fast_run', linker='c')
    return None if theano.compile.default_mode != "PROFILE_MODE" else theano.compile.ProfileMode()

def print_mode(mode):
    if mode != None and isinstance(mode,(theano.compile.ProfileMode,)):
        mode.print_summary()

def print_diff_mode(a,b):
    if a != None and isinstance(a,(theano.compile.ProfileMode,)) and isinstance(b,(theano.compile.ProfileMode,)):
        a.print_diff_summary(b)

def run_nnet(use_gpu, n_batch=60, n_in=1024, n_hid=2048, n_out=10, n_iter=100):

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
    if 0: print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = get_mode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

    if 0:
        for i, n in enumerate(train.maker.env.toposort()):
            print i, n

    xval = numpy.asarray(numpy.random.rand(n_batch, n_in), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch, n_out), dtype='float32')
    lr = numpy.asarray(0.01, dtype='float32')

    t0 = time.time()
    rval = []
    for i in xrange(n_iter):
        rval.append(train(xval, yval, lr))
    dt = time.time() - t0
        
    print_mode(mode)
    return numpy.asarray(rval), dt
    
def test_run_nnet():
    for n_in in 1024, 2048, 4096:
        for n_hid in 1024, 2048, 4096:
            numpy.random.seed(23456)
            rval_cpu, tc = run_nnet(False, n_in=n_in, n_hid=n_hid)
            numpy.random.seed(23456)
            rval_gpu, tg = run_nnet(True, n_in=n_in, n_hid=n_hid)
            #print "cpu:", rval_cpu
            #print "gpu:", rval_gpu
            print "max abs diff:", numpy.max(numpy.absolute(rval_gpu-rval_cpu))
            print "time cpu: %f, time gpu: %f, speed up %f"%(tc, tg, tc/tg)
            assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-6)

def test_run_nnet_med():
    numpy.random.seed(23456)
    rval_cpu = run_nnet(False, 10, 128, 50, 4, n_iter=10000)

def test_run_nnet_small():
    numpy.random.seed(23456)
    rval_cpu = run_nnet(False, 10, 10, 4, 4, n_iter=100000)

def run_conv_nnet1(shared_fn):
    n_batch = 16
    n_kern = 20
    shape_img = (n_batch, 1, 32, 32)
    shape_kern = (n_kern, 1, 5, 5)

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d(shape_img[2:],shape_kern[2:], 'valid')
    n_hid = n_kern * logical_hid_shape[0] * logical_hid_shape[1]
    n_out = 10

    w = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern)-0.5), dtype='float32'), 'w')
    b = shared_fn(numpy.asarray(numpy.zeros((n_kern,)), dtype='float32'), 'b')
    v = shared_fn(numpy.asarray(numpy.zeros((n_hid, n_out)), dtype='float32'), 'c')
    c = shared_fn(numpy.asarray(numpy.zeros(n_out), dtype='float32'), 'c')

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = theano.sandbox.conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)
    conv_op.set_flops()

    hid = tensor.tanh(conv_op(x, w)+b.dimshuffle((0,'x','x')))
    hid_flat = hid.reshape((n_batch, n_hid))
    out = tensor.tanh(tensor.dot(hid_flat, v)+c)
    loss = tensor.sum(0.5 * (out-y)**2 * lr)
    print 'loss type', loss.type

    params = [w, b, v, c]
    gparams = tensor.grad(loss, params)

    mode = get_mode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

#    for i, n in enumerate(train.maker.env.toposort()):
#        print i, n

    xval = numpy.asarray(numpy.random.rand(*shape_img), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch, n_out), dtype='float32')
    lr = numpy.asarray(0.01, dtype='float32')

    for i in xrange(10):
        rval = train(xval, yval, lr)
    print 'training done'
    print_mode(mode)
    return rval

def test_conv_nnet1():
    numpy.random.seed(23456)
    rval_cpu = run_conv_nnet1(shared)
    numpy.random.seed(23456)
    rval_gpu = run_conv_nnet1(tcn.shared_constructor)
    assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-6)

def run_conv_nnet2(shared_fn): # pretend we are training LeNet for MNIST

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

    logical_hid_shape = tcn.blas.GpuConv.logical_output_shape_2d(tuple(shape_img[2:]),tuple(shape_kern[2:]), 'valid')
    logical_hid_shape1 = tcn.blas.GpuConv.logical_output_shape_2d((logical_hid_shape[0]/2, logical_hid_shape[1]/2), tuple(shape_kern1[2:]), 'valid')
    n_hid = n_kern1 * logical_hid_shape1[0] * logical_hid_shape1[1]
    n_out = 10

    w0 = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern)-0.5), dtype='float32'), 'w0')
    b0 = shared_fn(numpy.asarray(numpy.zeros((n_kern,)), dtype='float32'), 'b0')
    w1 = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern1)-0.5), dtype='float32'), 'w1')
    b1 = shared_fn(numpy.asarray(numpy.zeros((n_kern1,)), dtype='float32'), 'b1')
    v = shared_fn(numpy.asarray(numpy.zeros((n_hid, n_out)), dtype='float32'), 'c')
    c = shared_fn(numpy.asarray(numpy.zeros(n_out), dtype='float32'), 'c')

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = theano.sandbox.conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)
    conv_op1 = theano.sandbox.conv.ConvOp((n_kern,logical_hid_shape[0]/2, logical_hid_shape[1]/2), shape_kern1[2:], n_kern1, n_batch, 1, 1)
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

    mode = get_mode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

#    for i, n in enumerate(train.maker.env.toposort()):
#        print i, n

    xval = numpy.asarray(numpy.random.rand(*shape_img), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch,n_out), dtype='float32')#int32 make all 0...
    lr = numpy.asarray(0.01, dtype='float32')
    for i in xrange(n_train):
        rval = train(xval, yval, lr)

    print_mode(mode)
    return rval

def test_conv_nnet2():
    numpy.random.seed(23456)
    rval_gpu = run_conv_nnet2(tcn.shared_constructor)
    if True:
        numpy.random.seed(23456)
        rval_cpu = run_conv_nnet2(shared)
        print rval_cpu[0], rval_gpu[0],rval_cpu[0]-rval_gpu[0]
        assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-4,atol=1e-4)

def run_conv_nnet2_classif(shared_fn, isize, ksize, n_batch, n_iter,
                           downsample_ops=True):
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


    w0 = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern)-0.5), dtype='float32'), 'w0')
    b0 = shared_fn(numpy.asarray(numpy.zeros((n_kern,)), dtype='float32'), 'b0')
    w1 = shared_fn(numpy.asarray(0.01*(numpy.random.rand(*shape_kern1)-0.5), dtype='float32'), 'w1')
    b1 = shared_fn(numpy.asarray(numpy.zeros((n_kern1,)), dtype='float32'), 'b1')
    v = shared_fn(numpy.asarray(0.01*numpy.random.randn(n_hid, n_out), dtype='float32'), 'v')
    c = shared_fn(numpy.asarray(numpy.zeros(n_out), dtype='float32'), 'c')

    print 'ALLOCATING ARCH: w0 shape', w0.value.shape
    print 'ALLOCATING ARCH: w1 shape', w1.value.shape
    print 'ALLOCATING ARCH: v shape', v.value.shape

    x = tensor.Tensor(dtype='float32', broadcastable=(0,1,0,0))('x')
    y = tensor.fmatrix('y')
    lr = tensor.fscalar('lr')

    conv_op = theano.sandbox.conv.ConvOp(shape_img[2:], shape_kern[2:], n_kern, n_batch, 1, 1)
    conv_op1 = theano.sandbox.conv.ConvOp((n_kern,logical_hid_shape[0]/2, logical_hid_shape[1]/2), shape_kern1[2:], n_kern1, n_batch, 1, 1)
    conv_op.set_flops()
    conv_op1.set_flops()

    ds_op = theano.sandbox.downsample.DownsampleFactorMax((2,2), ignore_border=False)
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

    mode = get_mode()

    print 'building pfunc ...'
    train = pfunc([x,y,lr], [loss], mode=mode, updates=[(p, p-g) for p,g in zip(params, gparams)])

    if False:
        for i, n in enumerate(train.maker.env.toposort()):
            print i, n

    xval = numpy.asarray(numpy.random.rand(*shape_img), dtype='float32')
    yval = numpy.asarray(numpy.random.rand(n_batch,n_out), dtype='float32')
    lr = numpy.asarray(0.01, dtype='float32')

    rvals=numpy.zeros(n_iter)
    t0 = time.time()
    for i in xrange(n_iter):
        rvals[i] = train(xval, yval, lr)[0]
    t1 = time.time()
    print_mode(mode)
    return rvals, t1-t0, mode

def cmp_run_conv_nnet2_classif(seed, isize, ksize, bsize, 
                               ignore_error=False, 
                               n_iter=10,
                               gpu_only=False,
                               float_atol=1e-08,
                               check_isfinite=True):
    """
       float_atol: None mean use the default value.
       check_isfinite: the debug mode option. We forward this value to debug mode.
                       For some parameter CrossentropyCategorical1Hot op generate inf when not optimized.
    """
    numpy.random.seed(seed)

    import theano.tensor.basic
    import theano.compile.debugmode
    from theano.compile.mode import predefined_modes
    orig_float32_atol = theano.tensor.basic.float32_atol
    orig_check_isfinite = predefined_modes["DEBUG_MODE"].check_isfinite
    
    try:
        predefined_modes["DEBUG_MODE"].check_isfinite = check_isfinite
        if float_atol:
            print "float_atol",float_atol
            theano.tensor.basic.float32_atol=float_atol
        rval_gpu, tg, gpu_mode = run_conv_nnet2_classif(tcn.shared_constructor, isize, ksize, bsize, n_iter)
    finally:
        predefined_modes["DEBUG_MODE"].check_isfinite = orig_check_isfinite
        theano.tensor.basic.float32_atol=orig_float32_atol

    if gpu_only:
        return
    
    try:
        predefined_modes["DEBUG_MODE"].check_isfinite = check_isfinite
        numpy.random.seed(seed)
        rval_cpu, tc, cpu_mode = run_conv_nnet2_classif(shared, isize, ksize, bsize, n_iter)
        if isinstance(cpu_mode,(theano.compile.ProfileMode,)):
            import pickle
            print "BEGIN GPU profile mode dump"
            #print pickle.dumps(gpu_mode)
            print "END GPU profile mode dump"
            print "BEGIN CPU profile mode dump"
            print pickle.dumps(cpu_mode)
            print "END CPU profile mode dump"

    finally:
        predefined_modes["DEBUG_MODE"].check_isfinite = orig_check_isfinite
        theano.tensor.basic.float32_atol=orig_float32_atol

    print "cpu:", rval_cpu
    print "gpu:", rval_gpu
    print "abs diff:", numpy.absolute(rval_gpu-rval_cpu)
    print "time cpu: %f, time gpu: %f, speed up %f"%(tc, tg, tc/tg)
    print "estimated time for one pass through MNIST with cpu: %f" % (tc * (60000.0 / (n_iter*bsize)))
    print "estimated time for one pass through MNIST with gpu: %f" % (tg * (60000.0 / (n_iter*bsize)))

    if not ignore_error:
        assert numpy.allclose(rval_cpu, rval_gpu,rtol=1e-3,atol=float_atol)

gpu_only=False
ignore_error=False

def test_lenet_28(): #MNIST
    cmp_run_conv_nnet2_classif(23485, 28, 5, 60, n_iter=10,
                                ignore_error=ignore_error, gpu_only=gpu_only)

def test_lenet_32(): #CIFAR10 / Shapeset
    cmp_run_conv_nnet2_classif(23485, 32, 5, 60, n_iter=10,
                               ignore_error=ignore_error, gpu_only=gpu_only)

def test_lenet_32_long(): #CIFAR10 / Shapeset
    # this tests the gradient of downsample on the GPU, 
    # which does not recieve specific testing
    cmp_run_conv_nnet2_classif(23485, 32, 5, 30, n_iter=50,
                               ignore_error=ignore_error, gpu_only=gpu_only)

def test_lenet_64(): # ???
    #float_atol needd to pass in debug mode
    #needed as cpu use extended precision and gpu don't
    cmp_run_conv_nnet2_classif(23485, 64, 7, 10, n_iter=10,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               float_atol=5e-4, check_isfinite=True)

def test_lenet_108(): # NORB
    cmp_run_conv_nnet2_classif(23485, 108, 7, 10, n_iter=5,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               check_isfinite=True)

def test_lenet_256(): # ImageNet
    cmp_run_conv_nnet2_classif(23485, 256, 9, 2, n_iter=3,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               check_isfinite=True)

#I did a wanted error in the name as we don't want it to execute automatically for now as it don't work
def tes_lenet_hd(): #HD 720p: 1280(wid)x720(len)
    cmp_run_conv_nnet2_classif(23485, (720,1280), 9, 2, n_iter=3,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               check_isfinite=True)

#I did a wanted error in the name as we don't want it to execute automatically for now as it don't work
def tes_lenet_full_hd(): #HD 1080p: 1920(wid)x1080(len)
    cmp_run_conv_nnet2_classif(23485, (1080,1920), 9, 2, n_iter=3,
                               ignore_error=ignore_error, gpu_only=gpu_only,
                               check_isfinite=True)
