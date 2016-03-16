from __future__ import absolute_import, print_function, division
import numpy
import theano
from theano.tensor import constant
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available is False:
    raise SkipTest('Optional package cuda disabled')

# The PyCObject that represents the cuda random stream object
# can't be deep copied. This is needed for DebugMode
if theano.config.mode in ['FAST_COMPILE', 'DebugMode', 'DEBUG_MODE']:
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def check_uniform_basic(shape_as_symbolic, dim_as_symbolic=False):
    """
    check_uniform_basic(shape_as_symbolic, dim_as_symbolic=False)

    Runs a basic sanity check on the `uniform` method of a
    `CURAND_RandomStreams` object.

    Checks that variates

     * are in the range [0, 1]
     * have a mean in the right neighbourhood (near 0.5)
     * are of the specified shape
     * successive calls produce different arrays of variates

    Parameters
    ----------
    shape_as_symbolic : boolean
        If `True`, est the case that the shape tuple is a symbolic
        variable rather than known at compile-time.

    dim_as_symbolic : boolean
        If `True`, test the case that an element of the shape
        tuple is a Theano symbolic. Irrelevant if `shape_as_symbolic`
        is `True`.
    """
    rng = CURAND_RandomStreams(234)
    if shape_as_symbolic:
        # instantiate a TensorConstant with the value (10, 10)
        shape = constant((10, 10))
    else:
        # Only one dimension is symbolic, with the others known
        if dim_as_symbolic:
            shape = (10, constant(10))
        else:
            shape = (10, 10)
    u0 = rng.uniform(shape)
    u1 = rng.uniform(shape)

    f0 = theano.function([], u0, mode=mode_with_gpu)
    f1 = theano.function([], u1, mode=mode_with_gpu)

    v0list = [f0() for i in range(3)]
    v1list = [f1() for i in range(3)]

    # print v0list
    # print v1list
    # assert that elements are different in a few ways
    assert numpy.all(v0list[0] != v0list[1])
    assert numpy.all(v1list[0] != v1list[1])
    assert numpy.all(v0list[0] != v1list[0])

    for v in v0list:
        assert v.shape == (10, 10)
        assert v.min() >= 0
        assert v.max() <= 1
        assert v.min() < v.max()
        assert .25 <= v.mean() <= .75


def test_uniform_basic():
    """
    Run the tests for `uniform` with different settings for the
    shape tuple passed in.
    """
    yield check_uniform_basic, False
    yield check_uniform_basic, False, True
    yield check_uniform_basic, True


def check_normal_basic(shape_as_symbolic, dim_as_symbolic=False):
    """
    check_normal_basic(shape_as_symbolic, dim_as_symbolic=False)

    Runs a basic sanity check on the `normal` method of a
    `CURAND_RandomStreams` object.

    Checks that variates

     * have a mean in the right neighbourhood (near 0)
     * are of the specified shape
     * successive calls produce different arrays of variates

    Parameters
    ----------
    shape_as_symbolic : boolean
        If `True`, est the case that the shape tuple is a symbolic
        variable rather than known at compile-time.

    dim_as_symbolic : boolean
        If `True`, test the case that an element of the shape
        tuple is a Theano symbolic. Irrelevant if `shape_as_symbolic`
        is `True`.
    """
    rng = CURAND_RandomStreams(234)
    if shape_as_symbolic:
        # instantiate a TensorConstant with the value (10, 10)
        shape = constant((10, 10))
    else:
        if dim_as_symbolic:
            # Only one dimension is symbolic, with the others known
            shape = (10, constant(10))
        else:
            shape = (10, 10)
    u0 = rng.normal(shape)
    u1 = rng.normal(shape)

    f0 = theano.function([], u0, mode=mode_with_gpu)
    f1 = theano.function([], u1, mode=mode_with_gpu)

    v0list = [f0() for i in range(3)]
    v1list = [f1() for i in range(3)]

    # print v0list
    # print v1list
    # assert that elements are different in a few ways
    assert numpy.all(v0list[0] != v0list[1])
    assert numpy.all(v1list[0] != v1list[1])
    assert numpy.all(v0list[0] != v1list[0])

    for v in v0list:
        assert v.shape == (10, 10)
        assert v.min() < v.max()
        assert -.5 <= v.mean() <= .5


def test_normal_basic():
    """
    Run the tests for `normal` with different settings for the
    shape tuple passed in.
    """
    yield check_normal_basic, False
    yield check_normal_basic, False, True
    yield check_normal_basic, True


def compare_speed():
    # To run this speed comparison
    # cd <directory of this file>
    # THEANO_FLAGS=device=gpu \
    #   python -c 'import test_rng_curand; test_rng_curand.compare_speed()'

    mrg = MRG_RandomStreams()
    crn = CURAND_RandomStreams(234)

    N = 1000 * 100

    dest = theano.shared(numpy.zeros(N, dtype=theano.config.floatX))

    mrg_u = theano.function([], [], updates={dest: mrg.uniform((N,))},
                            profile='mrg uniform')
    crn_u = theano.function([], [], updates={dest: crn.uniform((N,))},
                            profile='crn uniform')
    mrg_n = theano.function([], [], updates={dest: mrg.normal((N,))},
                            profile='mrg normal')
    crn_n = theano.function([], [], updates={dest: crn.normal((N,))},
                            profile='crn normal')

    for f in mrg_u, crn_u, mrg_n, crn_n:
        # don't time the first call, it has some startup cost
        print('DEBUGPRINT')
        print('----------')
        theano.printing.debugprint(f)

    for i in range(100):
        for f in mrg_u, crn_u, mrg_n, crn_n:
            # don't time the first call, it has some startup cost
            f.fn.time_thunks = (i > 0)
            f()
