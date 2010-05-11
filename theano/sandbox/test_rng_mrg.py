import sys, time
import numpy
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
#TODO: test gpu
#TODO: test MRG_RandomStreams
#TODO: test optimizer mrg_random_make_inplace
#def test_rng_mrg_cpu():
#TODO: make tests work when no flags gived. Now need: THEANO_FLAGS=device=gpu0,floatX=float32
#TODO: bug fix test_normal0, in normal() fct, n_samples currently need to be numpy.prod(size) not self.n_streams(size)

mode = theano.config.mode

def test_rng0():

    def basictest(f, steps, prefix=""):
        dt = 0.0
        for i in xrange(steps):
            t0 = time.time()
            ival = f()
            dt += time.time() - t0
            ival = numpy.asarray(ival)
            if i == 0:
                mean = numpy.array(ival, copy=True)
                min_ = ival.min()
                max_ = ival.max()
            else:
                alpha = 1.0 / (1+i)
                mean = alpha * ival + (1-alpha)*mean
                min_ = min(min_,ival.min())
                max_ = max(max_,ival.max())
            assert ival.min()>0 and ival.max()<1

        print prefix, 'mean', numpy.mean(mean)
        assert abs(numpy.mean(mean) - 0.5) < .01, 'bad mean?'
        print prefix, 'time', dt
        print prefix, 'elements', steps*sample_size[0]*sample_size[1]
        print prefix, 'samples/sec', steps*sample_size[0]*sample_size[1] / dt
        print prefix, 'min',min_,'max',max_

    if mode in ['DEBUG_MODE','FAST_COMPILE']:
        sample_size = (10,100)
        steps = int(1e2)
    else:
        sample_size = (1000,100)
        steps = int(1e3)
    
    print ''
    print 'ON CPU:'

    R = MRG_RandomStreams(234, use_cuda=False)
    u = R.uniform(size=sample_size)
    f = theano.function([], u, mode=mode)
    theano.printing.debugprint(f)
    print 'random?[:10]\n', f()[0,0:10]
    basictest(f, steps, prefix='mrg  cpu')

    if mode!='FAST_COMPILE':
        print ''
        print 'ON GPU:'
        R = MRG_RandomStreams(234, use_cuda=True)
        u = R.uniform(size=sample_size, dtype='float32')
        assert u.dtype == 'float32' #well, it's really that this test w GPU doesn't make sense otw
        f = theano.function([], theano.Out(
                theano.sandbox.cuda.basic_ops.gpu_from_host(u),
                borrow=True), mode=mode)
        theano.printing.debugprint(f)
        print 'random?[:10]\n', numpy.asarray(f())[0,0:10]
        basictest(f, steps, prefix='mrg  gpu')

    print ''
    print 'ON CPU w NUMPY:'
    RR = theano.tensor.shared_randomstreams.RandomStreams(234)

    uu = RR.uniform(size=sample_size)
    ff = theano.function([], uu, mode=mode)

    basictest(ff, steps, prefix='numpy')




def test_normal0():

    def basictest(f, steps, target_avg, target_std, prefix=""):
        dt = 0.0
        avg_std = 0.0
        for i in xrange(steps):
            t0 = time.time()
            ival = f()
            dt += time.time() - t0
            ival = numpy.asarray(ival)
            if i == 0:
                mean = numpy.array(ival, copy=True)
                avg_std = numpy.std(ival)
            else:
                alpha = 1.0 / (1+i)
                mean = alpha * ival + (1-alpha)*mean
                avg_std = alpha * numpy.std(ival) + (1-alpha)*avg_std

        print prefix, 'mean', numpy.mean(mean)
        assert abs(numpy.mean(mean) - target_avg) < .01, 'bad mean?'
        print prefix, 'std', avg_std
        assert abs(avg_std - target_std) < .01, 'bad std?'
        print prefix, 'time', dt
        print prefix, 'elements', steps*sample_size[0]*sample_size[1]
        print prefix, 'samples/sec', steps*sample_size[0]*sample_size[1] / dt

    sample_size = (999,100)
    print ''
    print 'ON CPU:'

    R = MRG_RandomStreams(234, use_cuda=False)
    n = R.normal(size=sample_size, avg=-5.0, std=2.0)
    f = theano.function([], n, mode=mode)
    theano.printing.debugprint(f)
    print 'random?[:10]\n', f()[0,0:10]
    basictest(f, 50, -5.0, 2.0, prefix='mrg ')

    sys.stdout.flush()

    # now with odd number of samples
    sample_size = (999,99)


    print ''
    print 'ON GPU:'
    R = MRG_RandomStreams(234, use_cuda=True)
    n = R.normal(size=sample_size, avg=-5.0, std=2.0, dtype='float32')
    assert n.dtype == 'float32' #well, it's really that this test w GPU doesn't make sense otw
    f = theano.function([], theano.Out(
        theano.sandbox.cuda.basic_ops.gpu_from_host(n),
        borrow=True), mode=mode)
    
    theano.printing.debugprint(f)
    sys.stdout.flush()
    print 'random?[:10]\n', numpy.asarray(f())[0,0:10]
    print '----'
    sys.stdout.flush()
    basictest(f, 50, -5.0, 2.0, prefix='gpu mrg ')


    print ''
    print 'ON CPU w NUMPY:'
    RR = theano.tensor.shared_randomstreams.RandomStreams(234)

    nn = RR.normal(size=sample_size, avg=-5.0, std=2.0)
    ff = theano.function([], nn, mode=mode)

    basictest(ff, 50, -5.0, 2.0, prefix='numpy ')


   
