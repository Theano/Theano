import numpy
import theano
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

def test_uniform_basic():
    rng = CURAND_RandomStreams(234)

    u0 = rng.uniform((10,10))
    u1 = rng.uniform((10,10))

    f0 = theano.function([], u0)
    f1 = theano.function([], u1)

    v0list = [f0() for i in range(3)]
    v1list = [f1() for i in range(3)]

    #print v0list
    #print v1list
    # assert that elements are different in a few ways
    assert numpy.all(v0list[0] != v0list[1])
    assert numpy.all(v1list[0] != v1list[1])
    assert numpy.all(v0list[0] != v1list[0])

    for v in v0list:
        assert v.shape == (10,10)
        assert v.min() >= 0
        assert v.max() <= 1
        assert v.min() < v.max()
        assert .25 <= v.mean() <= .75


def test_normal_basic():
    rng = CURAND_RandomStreams(234)

    u0 = rng.normal((10,10))
    u1 = rng.normal((10,10))

    f0 = theano.function([], u0)
    f1 = theano.function([], u1)

    v0list = [f0() for i in range(3)]
    v1list = [f1() for i in range(3)]

    #print v0list
    #print v1list
    # assert that elements are different in a few ways
    assert numpy.all(v0list[0] != v0list[1])
    assert numpy.all(v1list[0] != v1list[1])
    assert numpy.all(v0list[0] != v1list[0])

    for v in v0list:
        assert v.shape == (10,10)
        assert v.min() < v.max()
        assert -.5 <= v.mean() <= .5


def compare_speed():
    # To run this speed comparison
    # cd <directory of this file>
    # THEANO_FLAGS=device=gpu \
    #   python -c 'import test_rng_curand; test_rng_curand.compare_speed()'

    mrg = MRG_RandomStreams()
    crn = CURAND_RandomStreams(234)

    N=1000*100

    dest = theano.shared(numpy.zeros(N,dtype=theano.config.floatX))

    mrg_u = theano.function([], [], updates={dest:mrg.uniform((N,))},
            profile='mrg uniform')
    crn_u = theano.function([], [], updates={dest:crn.uniform((N,))},
            profile='crn uniform')
    mrg_n = theano.function([], [], updates={dest:mrg.normal((N,))},
            profile='mrg normal')
    crn_n = theano.function([], [], updates={dest:crn.normal((N,))},
            profile='crn normal')

    for f in mrg_u, crn_u, mrg_n, crn_n:
        # don't time the first call, it has some startup cost
        print 'DEBUGPRINT'
        print '----------'
        theano.printing.debugprint(f)

    for i in range(100):
        for f in mrg_u, crn_u, mrg_n, crn_n:
            # don't time the first call, it has some startup cost
            f.fn.time_thunks = (i>0)
            f()

